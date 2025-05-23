import numpy as np
import deepxde as dde
import torch
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# ---------------------------
# Define the ODE system
# ---------------------------
def Cell(y, t, p):
    dn = p[0] - p[1] * (1 + (y[1] / p[2])**p[3]) * y[0]
    dc = p[1] * (1 + (y[1] / p[2])**p[3]) * y[0] - p[4] * y[1]
    return [dn, dc]

# Generate ground truth solution using SciPy ODE solver
times = np.arange(0, 15, 0.01)
y_true = odeint(Cell, y0=[0, 0], t=times, args=((14, 1, 2, 3, 4),))

# Define range of training data fractions
train_fracs = np.linspace(0.1, 0.9, 9)
r2_train, r2_future = [], []

# Loop over training set sizes
for frac in train_fracs:
    N = int(len(times) * frac)
    t_train = times[:N].reshape(-1, 1)
    y_train = y_true[:N]

    # Define trainable parameters
    v, k1, K, n, k2 = [dde.Variable(val) for val in [10.0, 1.0, 1.0, 1.0, 1.0]]

    # Define ODE residuals for the PINN
    def ode_res(t, y):
        y_n, y_c = y[:, 0:1], y[:, 1:2]
        dn_dt = dde.grad.jacobian(y, t, i=0)
        dc_dt = dde.grad.jacobian(y, t, i=1)
        eps = 1e-6  # To prevent division by zero
        f_n = v - k1 * (1 + torch.clamp(y_c / (K + eps), min=eps)**n) * y_n
        f_c = k1 * (1 + torch.clamp(y_c / (K + eps), min=eps)**n) * y_n - k2 * y_c
        return [dn_dt - f_n, dc_dt - f_c]

    # Define geometry and boundary conditions
    geom = dde.geometry.TimeDomain(0, 15)
    ic1 = dde.DirichletBC(geom, lambda _: np.array([[0.0]]), lambda _, on_initial: on_initial, component=0)
    ic2 = dde.DirichletBC(geom, lambda _: np.array([[0.0]]), lambda _, on_initial: on_initial, component=1)
    data_pts = dde.PointSetBC(t_train, y_train)

    # Combine data and PDE residual
    data = dde.data.PDE(
        geom, ode_res, [ic1, ic2, data_pts],
        num_domain=2000, num_boundary=2
    )

    # Define neural network architecture
    net = dde.nn.FNN([1] + [100] * 5 + [2], "tanh", "Glorot normal")

    # Define model and compile
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, external_trainable_variables=[v, k1, K, n, k2])
    model.train(iterations=5000, display_every=1000)

    # Predict across full time range
    y_pred = model.predict(times.reshape(-1, 1))

    # Compute R² scores
    r2_train.append(r2_score(y_true[:N, 0], y_pred[:N, 0]))
    r2_future.append(r2_score(y_true[N:, 0], y_pred[N:, 0]) if N < len(times) else 0)

    # Plot prediction vs. ground truth
    plt.plot(times, y_true[:, 0], "r", label="True $x_0$")
    plt.plot(times, y_true[:, 1], "b", label="True $x_1$", alpha=0.4)
    plt.plot(times[N:], y_pred[N:, 0], "k--", label="Pred $x_0$")
    plt.plot(times[N:], y_pred[N:, 1], "k--", label="Pred $x_1$")
    plt.plot(times[:N], y_pred[:N, 0], "k-")
    plt.plot(times[:N], y_pred[:N, 1], "k-")
    plt.axvline(x=0.01 * N, color='g', linestyle='--', label="Training Split")
    plt.title(f"PINN Prediction (Train Size = {frac:.1f})")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend()
    plt.show()

# ---------------------------
# Plot R² evolution summary
# ---------------------------
plt.figure()
plt.plot(train_fracs, r2_train, "bo-", label="R² (Training)")
plt.plot(train_fracs, r2_future, "rx-", label="R² (Future)")
plt.xlabel("Training Size Fraction")
plt.ylabel("R² Score")
plt.legend()
plt.grid(True)
plt.title("PINN R² Score Evolution with Training Size")
plt.show()
