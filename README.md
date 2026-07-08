# LSTM Digital Twin of a Three-Phase Induction Motor

A data-driven digital twin of a three-phase squirrel-cage induction motor. A physics-based
Simulink model generates high-resolution operating data, and an LSTM network is trained on it
to learn the motor's dynamics — predicting **electromagnetic torque** and **rotor speed** from
the stator's electrical signals. Once trained, the network replaces the physics model: given
only voltages and currents, it simulates the motor's mechanical behavior forward in time.

## How it works

```
Simulink model (inducmotor.slx)
        │  variable-step simulation, 10 s run
        ▼
ds2tt.m → timetable → im_no_load_run1.csv        (~260k rows)
        │
        ▼
digital_twin_lstm.py
        ├── resample onto a uniform 1 ms grid
        ├── chronological 70/15/15 train/val/test split
        ├── standardize using training-set statistics only
        ├── NARX-style windowing: [V_abc, I_abc, V_dc, past torque & speed] → next state delta
        ├── phase 1: one-step training with noise injected on the fed-back state
        ├── phase 2: multi-step fine-tuning (model unrolled 25 steps on its own predictions)
        └── evaluate: one-step-ahead test metrics + full closed-loop simulation
```

### Model formulation

Rotor speed is an integrated mechanical state — it cannot be recovered from a short window of
instantaneous voltages and currents alone. The model is therefore formulated as a NARX
(Nonlinear AutoRegressive with eXogenous inputs) state-transition model: at each step a
2-layer LSTM (hidden 128) sees a 50 ms window of the electrical signals *and* the recent
mechanical state, and predicts the **change** in torque and speed over the next step.

Two training details make the closed-loop simulation stable rather than divergent:

- **State-noise injection** — during training, Gaussian noise is added to the fed-back
  torque/speed channels, teaching the model to pull small state errors back toward the true
  trajectory instead of compounding them.
- **Multi-step fine-tuning** — after teacher-forced pre-training, the model is unrolled for
  25 steps feeding its own predictions back, with the loss taken against the true trajectory,
  directly optimizing simulation stability.

Evaluation covers both regimes:

- **One-step-ahead** — teacher-forced prediction on the held-out chronological test segment.
- **Closed-loop simulation** — after a 50 ms seed, the model's own predictions are fed back
  for the remaining ~10 s, so it must reproduce the startup transient (inrush torque
  oscillations, acceleration ramp) and steady state from electrical inputs alone, exactly
  like the physics model.

## Results

<!-- METRICS -->

Plots are written to `artifacts/plots/`:

- `loss_curves.png` — training/validation loss
- `test_one_step.png` — one-step predictions vs. Simulink ground truth on the test segment
- `closed_loop_simulation.png` — full 10 s closed-loop simulation vs. ground truth

## Repository layout

```
├── digital_twin_lstm.py    # full training + evaluation pipeline (CLI)
├── notebooks/train.ipynb   # thin Jupyter/Colab runner for the same pipeline
├── matlab/
│   ├── inducmotor.slx      # Simulink induction motor model (data source)
│   └── ds2tt.m             # Simulink Dataset → timetable export helper
├── data/
│   └── im_no_load_run1.csv # 10 s no-load run exported from Simulink
└── artifacts/              # trained model, metrics, plots (generated)
```

## Setup & usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python digital_twin_lstm.py --csv data/im_no_load_run1.csv
```

Useful flags (see `--help` for all):

| Flag | Default | Description |
|---|---|---|
| `--history` | 50 | Input window length (samples) |
| `--sample-dt` | 1e-3 | Resampling period in seconds |
| `--hidden-size` / `--layers` | 128 / 2 | LSTM capacity |
| `--epochs` | 30 | Teacher-forcing phase epochs (early stopping enabled) |
| `--finetune-epochs` / `--rollout-steps` | 15 / 25 | Multi-step fine-tuning schedule |
| `--ar-noise` | 0.05 | Noise std on fed-back state during training |
| `--skip-simulation` | off | Skip the closed-loop simulation pass |

Training runs on CUDA, Apple Silicon (MPS), or CPU automatically.

## Dataset

`data/im_no_load_run1.csv` is a 10-second no-load startup run: the motor accelerates from
standstill to synchronous speed and settles into steady state. Columns:

| Column | Signal |
|---|---|
| `sig1_1..3` | Stator phase currents I_abc (A) |
| `sig2_1..3` | Stator phase voltages V_abc (V) |
| `sig1_vdscTT` | DC bus voltage (V) |
| `x_Te_` | Electromagnetic torque (N·m) — target |
| `sig1_wTT` | Rotor speed (rad/s) — target |

## Possible extensions

- Train across multiple load conditions and fault scenarios (broken rotor bar, phase imbalance)
- Use the twin for anomaly detection by monitoring residuals against live measurements
- Export to ONNX / TorchScript for real-time deployment alongside the drive controller
