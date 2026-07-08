% Time step and axis
dt = 0.001;
t  = -1:dt:4;    % wide enough to see everything

% Unit step function
u = @(x) double(x >= 0);

% Define signals:
% x(t) = u(t) - u(t-1)  (rectangular pulse on [0,1])
x = u(t) - u(t - 1);

% h(t) = t * u(t) * u(2 - t)  (ramp from 0 to 2)
h = t .* u(t) .* u(2 - t);

% Convolution (numerical)
y = conv(x, h) * dt;

% Time axis for convolution result
t_conv = (t(1) + t(1)) : dt : (t(end) + t(end));

% Plot
figure;

subplot(3,1,1);
plot(t, x, 'LineWidth', 1.5); grid on;
title('x(t) = u(t) - u(t-1)');
xlabel('t'); ylabel('x(t)');

subplot(3,1,2);
plot(t, h, 'LineWidth', 1.5); grid on;
title('h(t) = t u(t) u(2-t)');
xlabel('t'); ylabel('h(t)');

subplot(3,1,3);
plot(t_conv, y, 'LineWidth', 1.5); grid on;
title('y(t) = x(t) * h(t)');
xlabel('t'); ylabel('y(t)');
