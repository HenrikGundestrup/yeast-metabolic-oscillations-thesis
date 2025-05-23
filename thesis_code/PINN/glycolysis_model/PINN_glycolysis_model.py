import numpy as np
import deepxde as dde
import torch
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ----------------------------
# 1. Define the ODE system
# ----------------------------

def Cell(y, t, p):
    """2-state glycolysis ODE model."""
    n, c = y
    v, k1, K, n_hill, k2 = p
    dn = v - k1 * (1 + (c / K) ** n_hill) * n
    dc = k1 * (1 + (c / K) ** n_hill) * n - k2 * c
    return [dn, dc]

# ----------------------------
# 2. Generate synthetic data
# ----------------------------

times = np.arange(0, 15, 0.01)
v_true, k1_true, K_true, n_true, k2_true = 14, 1, 2, 3, 4
params_true = (v_true, k1_true, K_true, n_true, k2_true)
y_true = odeint(Cell, y0=[0, 0], t=times, args=(params_true,))

# ----------------------------
# 3. Trainable variables
# ----------------------------

v  = dde.Variable(10.0)
k1 = dde.Variable(1.0)
K  = dde.Variable(1.0)
n  = dde.Variable(1.0)
k2 = dde.Variable(1.0)

# ----------------------------
# 4. Define ODE residuals for PINN
# ----------------------------

def ode_residuals(t, y):
    y_n, y_c = y[:, 0:1], y[:, 1:2]

    dn_dt = dde.grad.jacobian(y, t, i=0)
    dc_dt = dde.grad.jacobian(y, t, i=1)

    eps = 1e-6  # avoid division by zero
    hill_term = (torch.clamp(y_c / (K + eps), min=eps)) ** n
    reaction = k1 * (1 + hill_term)

    dn = v - reaction * y_n
    dc = reaction * y_n - k2 * y_c

    return [dn_dt - dn, dc_dt - dc]

# ----------------------------
# 5. Domain and initial conditions
# ----------------------------

geom = dde.geometry.TimeDomain(0, 15)

def initial_condition(_, on_initial):
    return on_initial

ic1 = dde.DirichletBC(geom, lambda _: np.array([[0.0]]), initial_condition, component=0)  # n(0) = 0
ic2 = dde.DirichletBC(geom, lambda _: np.array([[0.0]]), initial_condition, component=1)  # c(0) = 0

# Include full trajectory as supervised data
data_points = dde.PointSetBC(times.reshape(-1, 1), y_true)

# ----------------------------
# 6. Set up PINN dataset
# ----------------------------

data = dde.data.PDE(
    geom,
    ode_residuals,
    [ic1, ic2, data_points],
    num_domain=100,
    num_boundary=1,
)

# ----------------------------
# 7. Define the neural network
# ----------------------------

net = dde.nn.FNN([1] + [100] * 10 + [2], "tanh", "Glorot normal")

model = dde.Model(data, net)

# ----------------------------
# 8. Compile and train
# ----------------------------

model.compile(
    "adam",
    lr=0.001,
    external_trainable_variables=[v, k1, K, n, k2],
    loss_weights=[10, 0.1, 0.1, 0.1, 100]  # residual1, residual2, ic1, ic2, data
)
losshistory, train_state = model.train(iterations=5000)

# Fine-tune with L-BFGS
model.compile("L-BFGS")
losshistory, train_state = model.train()

# ----------------------------
# 9. Print learned parameters
# ----------------------------

print("Learned parameters:")
print("v  =", v.detach().numpy())
print("k1 =", k1.detach().numpy())
print("K  =", K.detach().numpy())
print("n  =", n.detach().numpy())
print("k2 =", k2.detach().numpy())

# Compute relative errors
print("\nRelative Errors:")
print("v error  =", (v - v_true) / v_true)
print("k1 error =", (k1 - k1_true) / k1_true)
print("K error  =", (K - K_true) / k1_true)
print("n error  =", (n - n_true) / n_true)
print("k2 error =", (k2 - k2_true) / k1_true)

# Total error metric
Variable_diff = (
    abs(v - v_true) / v_true
    + (k1 - k1_true) / k1_true
    + (K - K_true) / k1_true
    + (n - n_true) / n_true
    + (k2 - k2_true) / k1_true
)
Variable_diff = 100 * Variable_diff / 5
print("\nAverage % Parameter Error:", Variable_diff.item())

# ----------------------------
# 10. Predict and plot
# ----------------------------

t_test = times.reshape(-1, 1)
y_pred = model.predict(t_test)

# R² Score
r2 = (r2_score(y_true[:, 0], y_pred[:, 0]) + r2_score(y_true[:, 1], y_pred[:, 1])) / 2
print("R² Score =", r2)

# Plot results
plt.plot(times, y_true[:, 0], "r-", label='Observed $F6P(t)$')
plt.plot(times, y_true[:, 1], "b-", label='Observed $F16BP(t)$')
plt.plot(t_test, y_pred[:, 0], "k-", label='PINN $F6P(t)$')
plt.plot(t_test, y_pred[:, 1], "k--", label='PINN $F16BP(t)$')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Save loss plots
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
