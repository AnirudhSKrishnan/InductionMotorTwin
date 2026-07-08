"""Train an LSTM digital twin of a three-phase induction motor.

Learns the discrete-time dynamics mapping electrical inputs (phase currents,
phase voltages, DC bus voltage) plus the recent mechanical state to the change
in mechanical state (electromagnetic torque, rotor speed), NARX-style, from
time-series data exported from a Simulink simulation.

Training has two phases:
  1. One-step teacher forcing with noise injected on the fed-back state
     channels, so the model learns to correct small state errors.
  2. Multi-step fine-tuning: the model is unrolled for K steps feeding its own
     predictions back, and the loss is taken against the true trajectory. This
     directly optimizes closed-loop simulation stability.

The trained model is validated two ways:
  1. One-step-ahead prediction on a held-out chronological test segment.
  2. Closed-loop rollout over the full run: after a short seed window the
     model's own torque/speed predictions are fed back as inputs, so it
     simulates the motor forward in time like the physics model.

Usage:
    python digital_twin_lstm.py --csv data/im_no_load_run1.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn

FEATURE_COLS = [
    "sig1_1",       # phase current A
    "sig1_2",       # phase current B
    "sig1_3",       # phase current C
    "sig2_1",       # phase voltage A
    "sig2_2",       # phase voltage B
    "sig2_3",       # phase voltage C
    "sig1_vdscTT",  # DC bus voltage
]
TARGET_COLS = [
    "x_Te_",        # electromagnetic torque (N·m)
    "sig1_wTT",     # rotor speed (rad/s)
]
TARGET_LABELS = {
    "x_Te_": "Electromagnetic torque (N·m)",
    "sig1_wTT": "Rotor speed (rad/s)",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _parse_time_column(raw_series: pd.Series) -> pd.Series:
    """Convert time strings like '0.001 sec' into float seconds."""
    cleaned = raw_series.astype(str).str.replace(" sec", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def load_motor_dataframe(csv_path: Path, dt: float) -> pd.DataFrame:
    """Load the Simulink export and resample onto a uniform time grid.

    The variable-step solver emits irregular timestamps (near-duplicate points
    around discontinuities), so every column is linearly interpolated onto a
    fixed-step grid before windowing.
    """
    df = pd.read_csv(csv_path)
    df["Time"] = _parse_time_column(df["Time"])
    df = df.dropna(subset=["Time"]).sort_values("Time", kind="mergesort")
    df = df.drop_duplicates(subset="Time").reset_index(drop=True)

    t_raw = df["Time"].to_numpy()
    t_uniform = np.arange(t_raw[0], t_raw[-1] + dt / 2, dt)
    resampled = {"Time": t_uniform}
    for column in df.columns:
        if column == "Time":
            continue
        resampled[column] = np.interp(t_uniform, t_raw, df[column].to_numpy())
    return pd.DataFrame(resampled)


class LSTMForecaster(nn.Module):
    """Maps a window of [exogenous inputs, state] to the next state delta."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_out, _ = self.rnn(x)
        return self.head(seq_out[:, -1, :])


class MotorSegment:
    """A contiguous, standardized segment of the run kept as device tensors."""

    def __init__(
        self,
        exog_scaled: np.ndarray,
        state_scaled: np.ndarray,
        delta_scale: np.ndarray,
        device: torch.device,
    ) -> None:
        self.exog = torch.from_numpy(exog_scaled.astype(np.float32)).to(device)
        self.state = torch.from_numpy(state_scaled.astype(np.float32)).to(device)
        self.delta_scale = torch.from_numpy(delta_scale.astype(np.float32)).to(device)

    def __len__(self) -> int:
        return len(self.state)


def build_windows(segment: MotorSegment, anchors: torch.Tensor, history: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (exog_window, state_window) of shape [B, history, ·] ending just before each anchor."""
    offsets = torch.arange(-history, 0, device=anchors.device)
    idx = anchors[:, None] + offsets[None, :]
    return segment.exog[idx], segment.state[idx]


def rollout_steps(
    model: nn.Module,
    segment: MotorSegment,
    anchors: torch.Tensor,
    history: int,
    steps: int,
    ar_noise: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unroll the model `steps` times feeding its own state predictions back.

    Returns (predicted states, true states) with shape [B, steps, D] in scaled
    state units. Exogenous inputs always come from the recorded data; only the
    mechanical state is fed back.
    """
    exog_w, state_w = build_windows(segment, anchors, history)
    if ar_noise > 0:
        state_w = state_w + torch.randn_like(state_w) * ar_noise

    preds: List[torch.Tensor] = []
    truths: List[torch.Tensor] = []
    for k in range(steps):
        x = torch.cat([exog_w, state_w], dim=-1)
        delta = model(x) * segment.delta_scale
        new_state = state_w[:, -1, :] + delta
        preds.append(new_state)
        truths.append(segment.state[anchors + k])
        if k + 1 < steps:
            next_exog = segment.exog[anchors + k].unsqueeze(1)
            exog_w = torch.cat([exog_w[:, 1:], next_exog], dim=1)
            state_w = torch.cat([state_w[:, 1:], new_state.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=1), torch.stack(truths, dim=1)


def run_training_phase(
    model: nn.Module,
    train_seg: MotorSegment,
    val_seg: MotorSegment,
    history: int,
    steps: int,
    epochs: int,
    batch_size: int,
    lr: float,
    ar_noise: float,
    label: str,
    loss_history: dict,
) -> None:
    """Train with a `steps`-long unrolled loss; steps=1 is teacher forcing."""
    device = train_seg.state.device
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    train_anchors = torch.arange(history, len(train_seg) - steps + 1, device=device)
    val_anchors = torch.arange(history, len(val_seg) - steps + 1, device=device)

    best_val = float("inf")
    best_state = None
    patience = max(4, epochs // 4)
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = train_anchors[torch.randperm(len(train_anchors), device=device)]
        total_loss, total_n = 0.0, 0
        for start in range(0, len(perm), batch_size):
            anchors = perm[start : start + batch_size]
            optimizer.zero_grad()
            preds, truths = rollout_steps(model, train_seg, anchors, history, steps, ar_noise)
            loss = criterion(preds, truths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(anchors)
            total_n += len(anchors)
        train_loss = total_loss / max(total_n, 1)

        model.eval()
        with torch.no_grad():
            val_loss, val_n = 0.0, 0
            for start in range(0, len(val_anchors), batch_size):
                anchors = val_anchors[start : start + batch_size]
                preds, truths = rollout_steps(model, val_seg, anchors, history, steps)
                val_loss += criterion(preds, truths).item() * len(anchors)
                val_n += len(anchors)
            val_loss /= max(val_n, 1)
        scheduler.step(val_loss)
        loss_history.setdefault(label, {"train_loss": [], "val_loss": []})
        loss_history[label]["train_loss"].append(train_loss)
        loss_history[label]["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{label}] Epoch {epoch:03d} | train_loss={train_loss:.6e} | val_loss={val_loss:.6e} | lr={lr_now:.2e}")
        if wait >= patience:
            print(f"[{label}] Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)


@torch.no_grad()
def one_step_predictions(
    model: nn.Module,
    segment: MotorSegment,
    history: int,
    batch_size: int,
    target_scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """Teacher-forced next-step predictions over a segment, in physical units."""
    model.eval()
    device = segment.state.device
    anchors_all = torch.arange(history, len(segment), device=device)
    preds_list: List[np.ndarray] = []
    truth_list: List[np.ndarray] = []
    for start in range(0, len(anchors_all), batch_size):
        anchors = anchors_all[start : start + batch_size]
        preds, truths = rollout_steps(model, segment, anchors, history, steps=1)
        preds_list.append(preds[:, 0, :].cpu().numpy())
        truth_list.append(truths[:, 0, :].cpu().numpy())
    preds = target_scaler.inverse_transform(np.concatenate(preds_list, axis=0))
    truth = target_scaler.inverse_transform(np.concatenate(truth_list, axis=0))
    return preds, truth


@torch.no_grad()
def closed_loop_simulation(
    model: nn.Module,
    segment: MotorSegment,
    history: int,
    target_scaler: StandardScaler,
) -> np.ndarray:
    """Simulate the whole segment feeding predictions back, in physical units.

    Only the first `history` true state samples seed the simulation; afterwards
    the model sees exclusively the recorded electrical signals and its own
    previous outputs.
    """
    model.eval()
    device = segment.state.device
    anchors = torch.tensor([history], device=device)
    preds, _ = rollout_steps(model, segment, anchors, history, steps=len(segment) - history)
    sim_scaled = segment.state.cpu().numpy().copy()
    sim_scaled[history:] = preds[0].cpu().numpy()
    return target_scaler.inverse_transform(sim_scaled)


def compute_metrics(preds: np.ndarray, targets: np.ndarray, target_columns: Sequence[str]) -> dict:
    metrics = {}
    for col_idx, column in enumerate(target_columns):
        metrics[column] = {
            "rmse": float(root_mean_squared_error(targets[:, col_idx], preds[:, col_idx])),
            "r2": float(r2_score(targets[:, col_idx], preds[:, col_idx])),
        }
    return metrics


def chronological_split(
    arr: np.ndarray, train_ratio: float, val_ratio: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = len(arr)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    return arr[:train_end], arr[train_end:val_end], arr[val_end:]


def plot_loss_curves(loss_history: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(loss_history), figsize=(7 * len(loss_history), 5), squeeze=False)
    for ax, (label, hist) in zip(axes[0], loss_history.items()):
        ax.plot(hist["train_loss"], label="train")
        ax.plot(hist["val_loss"], label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss (scaled state)")
        ax.set_yscale("log")
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_series_comparison(
    time_axis: np.ndarray,
    preds: np.ndarray,
    truth: np.ndarray,
    target_columns: Sequence[str],
    metrics: dict,
    title_prefix: str,
    out_path: Path,
    split_times: Sequence[float] | None = None,
) -> None:
    fig, axes = plt.subplots(len(target_columns), 1, figsize=(11, 4 * len(target_columns)), sharex=True)
    axes = np.atleast_1d(axes)
    for col_idx, (ax, column) in enumerate(zip(axes, target_columns)):
        ax.plot(time_axis, truth[:, col_idx], label="Simulink (ground truth)", linewidth=1.2)
        ax.plot(time_axis, preds[:, col_idx], "--", label="LSTM digital twin", linewidth=1.0, alpha=0.85)
        stat = metrics[column]
        ax.set_ylabel(TARGET_LABELS.get(column, column))
        ax.set_title(f"{title_prefix}: {TARGET_LABELS.get(column, column)} — "
                     f"RMSE {stat['rmse']:.4f}, R² {stat['r2']:.4f}")
        if split_times:
            for st in split_times:
                ax.axvline(st, color="gray", linestyle=":", linewidth=1)
        ax.legend()
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv", type=Path, default=Path("data/im_no_load_run1.csv"))
    parser.add_argument("--epochs", type=int, default=30, help="Teacher-forcing phase epochs.")
    parser.add_argument("--finetune-epochs", type=int, default=15, help="Multi-step fine-tuning epochs.")
    parser.add_argument("--rollout-steps", type=int, default=25, help="Unroll length K for fine-tuning.")
    parser.add_argument("--ar-noise", type=float, default=0.05,
                        help="Std of noise injected on fed-back state channels during training.")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--history", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-cols", nargs="*", default=FEATURE_COLS)
    parser.add_argument("--target-cols", nargs="*", default=TARGET_COLS)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--sample-dt", type=float, default=1e-3)
    parser.add_argument("--skip-simulation", action="store_true", help="Skip the full closed-loop simulation pass.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    df = load_motor_dataframe(args.csv, dt=args.sample_dt)
    missing_columns = set(args.feature_cols + args.target_cols) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Columns not found in CSV: {sorted(missing_columns)}")

    exog = df[args.feature_cols].to_numpy(dtype=np.float64)
    targets = df[args.target_cols].to_numpy(dtype=np.float64)

    # Split first, then fit scalers on the training segment only so that
    # validation/test statistics never leak into preprocessing.
    e_train, e_val, e_test = chronological_split(exog, args.train_ratio, args.val_ratio)
    t_train, t_val, t_test = chronological_split(targets, args.train_ratio, args.val_ratio)

    exog_scaler = StandardScaler().fit(e_train)
    target_scaler = StandardScaler().fit(t_train)

    # The model predicts state deltas; scale them so the network output is O(1).
    delta_scale = np.diff(target_scaler.transform(t_train), axis=0).std(axis=0)
    print(f"Delta scale (scaled state units): {delta_scale}")

    def make_segment(e: np.ndarray, t: np.ndarray) -> MotorSegment:
        return MotorSegment(exog_scaler.transform(e), target_scaler.transform(t), delta_scale, device)

    train_seg = make_segment(e_train, t_train)
    val_seg = make_segment(e_val, t_val)
    test_seg = make_segment(e_test, t_test)
    full_seg = make_segment(exog, targets)
    print(f"Samples: train={len(train_seg)}, val={len(val_seg)}, test={len(test_seg)}")

    model = LSTMForecaster(
        input_size=len(args.feature_cols) + len(args.target_cols),
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        output_dim=len(args.target_cols),
        dropout=args.dropout,
    ).to(device)

    loss_history: dict = {}
    run_training_phase(
        model, train_seg, val_seg, args.history, steps=1, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.learning_rate, ar_noise=args.ar_noise,
        label="phase1_one_step", loss_history=loss_history,
    )
    if args.finetune_epochs > 0:
        run_training_phase(
            model, train_seg, val_seg, args.history, steps=args.rollout_steps,
            epochs=args.finetune_epochs, batch_size=args.batch_size,
            lr=args.learning_rate / 4, ar_noise=args.ar_noise,
            label="phase2_multi_step", loss_history=loss_history,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_loss_curves(loss_history, plots_dir / "loss_curves.png")

    # --- One-step-ahead evaluation on the held-out test segment ---
    preds_test, truth_test = one_step_predictions(model, test_seg, args.history, args.batch_size, target_scaler)
    one_step_metrics = compute_metrics(preds_test, truth_test, args.target_cols)
    print("\nOne-step-ahead test metrics:")
    for column, stat in one_step_metrics.items():
        print(f"  {column}: rmse={stat['rmse']:.6f}, r2={stat['r2']:.4f}")

    test_time = np.arange(len(preds_test)) * args.sample_dt
    plot_series_comparison(
        test_time, preds_test, truth_test, args.target_cols, one_step_metrics,
        "One-step-ahead (test segment)", plots_dir / "test_one_step.png",
    )

    # --- Closed-loop simulation over the full run ---
    sim_metrics = {}
    if not args.skip_simulation:
        print("\nRunning closed-loop simulation over the full run...")
        sim_real = closed_loop_simulation(model, full_seg, args.history, target_scaler)
        eval_slice = slice(args.history, None)  # exclude the seeded region
        sim_metrics = compute_metrics(sim_real[eval_slice], targets[eval_slice], args.target_cols)
        print("Closed-loop simulation metrics (full run, after seed window):")
        for column, stat in sim_metrics.items():
            print(f"  {column}: rmse={stat['rmse']:.6f}, r2={stat['r2']:.4f}")

        full_time = df["Time"].to_numpy()
        train_end_t = full_time[int(len(full_time) * args.train_ratio)]
        val_end_t = full_time[int(len(full_time) * (args.train_ratio + args.val_ratio))]
        plot_series_comparison(
            full_time, sim_real, targets, args.target_cols, sim_metrics,
            "Closed-loop simulation", plots_dir / "closed_loop_simulation.png",
            split_times=[train_end_t, val_end_t],
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "exog_scaler_mean": exog_scaler.mean_,
            "exog_scaler_scale": exog_scaler.scale_,
            "target_scaler_mean": target_scaler.mean_,
            "target_scaler_scale": target_scaler.scale_,
            "delta_scale": delta_scale,
            "feature_columns": args.feature_cols,
            "target_columns": args.target_cols,
            "config": {
                "history": args.history,
                "hidden_size": args.hidden_size,
                "layers": args.layers,
                "dropout": args.dropout,
                "sample_dt": args.sample_dt,
            },
        },
        args.output_dir / "lstm_digital_twin.pt",
    )

    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump({"one_step_test": one_step_metrics, "closed_loop_simulation": sim_metrics}, fp, indent=2)

    metadata = {
        "train_samples": len(train_seg),
        "val_samples": len(val_seg),
        "test_samples": len(test_seg),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "history": args.history,
        "batch_size": args.batch_size,
        "sample_dt": args.sample_dt,
        "rollout_steps": args.rollout_steps,
        "ar_noise": args.ar_noise,
        "loss_history": loss_history,
    }
    with open(args.output_dir / "training_run.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"\nArtifacts saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
