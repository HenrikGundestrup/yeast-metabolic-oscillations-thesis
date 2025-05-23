import numpy as np
import deepxde as dde
import torch
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ----------------------------
# 1. Define the true system
# ----------------------------

def oscillator(y, t, gamma, omega):
    """Damped harmonic oscillator ODE system."""
    x, v = y
    dxdt = v
    dvdt = -2 * gamma * v - omega**2 * x
    return [dxdt, dvdt]

# ----------------------------
# 2. Generate synthetic data
# ----------------------------

gamma_true = 0.2
omega_true = 2 * np.pi
y0 = [1.0, 1.0]  # initial position and velocity
times = np.arange(0, 10, 0.01)
y_true = odeint(oscillator, y0, times, args=(gamma_true, omega_true))

# ----------------------------
# 3. Define learnable variables
# ----------------------------

gamma = dde.Variable(0.5)  # initial guess
omega = dde.Variable(1.0)

# ----------------------------
# 4. Define residuals for PINN
# ----------------------------

def ode_residuals(t, y):
    x = y[:, 0:1]
    v = y[:, 1:2]

    dx_dt = dde.grad.jacobian(y, t, i=0)
    dv_dt = dde.grad.jacobian(y, t, i=1)

    dx = v
    dv = -2 * gamma * v - omega**2 * x

    return [dx_dt - dx, dv_dt - dv]

# ----------------------------
# 5. Geometry and Initial Conditions
# ----------------------------

geom = dde.geometry.TimeDomain(0, 10)

def initial_condition(_, on_initial):
    return on_initial

# Initial conditions: x(0) = 1.0, v(0) = 0.0
ic1 = dde.DirichletBC(geom, lambda _: np.array([[1.0]]), initial_condition, component=0)
ic2 = dde.DirichletBC(geom, lambda _: np.array([[0.0]]), initial_condition, component=1)

# Add synthetic data as supervised points
data_points = dde.PointSetBC(times.reshape(-1, 1), y_true)

# ----------------------------
# 6. Create dataset
# ----------------------------

data = dde.data.PDE(
    geom,
    ode_residuals,
    [ic1, ic2, data_points],
    num_domain=100,
    num_boundary=2,
)

# ----------------------------
# 7. Define the neural network
# ----------------------------

net = dde.nn.FNN([1] + [100] * 5 + [2], "tanh", "Glorot normal")

# ----------------------------
# 8. Compile and train the model
# ----------------------------

model = dde.Model(data, net)

# Adam optimizer phase
model.compile(
    "adam",
    lr=0.001,
    external_trainable_variables=[gamma, omega],
    loss_weights=[10, 0.1, 0.1, 0.1, 100]
)
losshistory, train_state = model.train(iterations=5000)

# Switch to L-BFGS optimizer for fine-tuning
model.compile("L-BFGS")
losshistory, train_state = model.train()

# ----------------------------
# 9. Results and evaluation
# ----------------------------

print("True parameters:")
print(f"gamma = {gamma_true}, omega = {omega_true}")
print("Learned parameters:")
print(f"gamma = {gamma.detach().numpy()}, omega = {omega.detach().numpy()}")

error = ((gamma_true - gamma) / gamma_true + (omega_true - omega) / gamma_true) / 2
print("Relative Error (avg):", error.item())

# Predictions
t_test = times
y_pred = model.predict(t_test.reshape(-1, 1))

# R² score
r2 = (r2_score(y_true[:, 0], y_pred[:, 0]) + r2_score(y_true[:, 1], y_pred[:, 1])) / 2
print("R² score =", r2)

# ----------------------------
# 10. Plot results
# ----------------------------

plt.plot(times, y_true[:, 0], "r", label='True Position')
plt.plot(times, y_true[:, 1], "b", label='True Velocity')
plt.plot(t_test, y_pred[:, 0], "k-", label='PINN Position')
plt.plot(t_test, y_pred[:, 1], "k--", label='PINN Velocity')
plt.xlabel("Time (s)")
plt.ylabel("Position / Velocity")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Save training plots
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
