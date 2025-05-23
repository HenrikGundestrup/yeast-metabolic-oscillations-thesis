import numpy as np
import deepxde as dde
import torch
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Define the ODE system
def Cell(y, t, p):
    dn = p[0] - p[1] * (1 + (y[1] / p[2])**p[3]) * y[0]
    dc = p[1] * (1 + (y[1] / p[2])**p[3]) * y[0] - p[4] * y[1]
    return [dn, dc]

# True parameters
v_true, k1_true, K_true, n_true, k2_true = 15, 1, 2, 3, 4
times = np.arange(0, 15, 0.01)
y_clean = odeint(Cell, t=times, y0=[0, 0], args=((v_true, k1_true, K_true, n_true, k2_true),))

noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
r2_scores = []

for noise_std in noise_levels:
    print(f"\n--- Training with noise std: {noise_std} ---")

    y_noisy = y_clean + np.random.normal(0, noise_std, y_clean.shape)

    # Define trainable variables for each run
    v  = dde.Variable(10.0)
    k1 = dde.Variable(1.0)
    K  = dde.Variable(1.0)
    n  = dde.Variable(1.0)
    k2 = dde.Variable(1.0)

    def ode_residuals(t, y):
        y_n, y_c = y[:, 0:1], y[:, 1:2]
        dn_dt = dde.grad.jacobian(y, t, i=0)
        dc_dt = dde.grad.jacobian(y, t, i=1)
        eps = 1e-6
        dn = v - k1 * (1 + torch.clamp((y_c / (K + eps)), min=eps) ** n) * y_n
        dc = k1 * (1 + torch.clamp((y_c / (K + eps)), min=eps) ** n) * y_n - k2 * y_c
        return [dn_dt - dn, dc_dt - dc]

    geom = dde.geometry.TimeDomain(0, 15)

    def boundary_initial(_, on_initial):
        return on_initial

    ic1 = dde.DirichletBC(geom, lambda _: np.array([[0.0]]), boundary_initial, component=0)
    ic2 = dde.DirichletBC(geom, lambda _: np.array([[0.0]]), boundary_initial, component=1)

    data_points = dde.PointSetBC(times.reshape(-1, 1), y_noisy)

    data = dde.data.PDE(
        geom,
        ode_residuals,
        [ic1, ic2, data_points],
        num_domain=2000,
        num_boundary=2,
    )

    net = dde.nn.FNN([1] + [100] * 5 + [2], "tanh", "Glorot normal")

    model = dde.Model(data, net)

    model.compile(
        "adam",
        lr=0.001,
        external_trainable_variables=[v, k1, K, n, k2],
        loss_weights=[1, 0, 0, 0.1, 10]
    )

    model.train(iterations=5000)

    # Predict and evaluate
    t_test = times.reshape(-1, 1)
    y_pred = model.predict(t_test)
    r2 = (r2_score(y_clean[:, 0], y_pred[:, 0]) + r2_score(y_clean[:, 1], y_pred[:, 1])) / 2
    r2_scores.append(r2)

    # Plot for each noise level
    plt.figure(figsize=(10, 4))
    plt.plot(times, y_clean[:, 0], "r-", label="n (True)")
    plt.plot(times, y_clean[:, 1], "b-", label="c (True)")
    plt.plot(t_test, y_pred[:, 0], "k--", label="n (Pred)")
    plt.plot(t_test, y_pred[:, 1], "k--", label="c (Pred)")
    plt.xlabel("Time")
    plt.ylabel("Concentrations")
    plt.title(f"PINN Fit with Noise Std = {noise_std:.3f} | R² = {r2:.4f}")
    plt.legend()
    plt.show()

# Final summary plot
plt.plot(noise_levels, r2_scores, marker='o', linestyle='-', color='b', label="Average R²")
plt.xlabel("Noise Standard Deviation")
plt.ylabel("R²")
#plt.title("PINN Robustness to Noise")
plt.legend()
plt.grid(True)
plt.show()

# Print final R² for each noise level
for noise, r2 in zip(noise_levels, r2_scores):
    print(f"Noise std: {noise:.3f}, R²: {r2:.4f}")



