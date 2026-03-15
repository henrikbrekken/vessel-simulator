import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from vessel import FarSpica
from functions.rk4 import rk4


vessel = FarSpica()


time = 100      # In secons
dt = 0.1       # Time step
n_steps = int(time / dt)

Vc = 0
betaVc = 0

x = np.zeros(12)  # Initial state vector

# Store data for plotting
eta_data = np.zeros((n_steps, 6)) # Rows and columns
nu_data = np.zeros((n_steps, 6))

for i in range(n_steps):
    # Compute forces and moments

    tau = np.array([1000e3, 0, 0, 0, 0, 0])  # Surge force

    vessel.x = rk4(vessel.dynamics, dt, vessel.x, tau, Vc, betaVc)
    vessel.nu = vessel.x[0:6]
    vessel.eta = vessel.x[6:12]

    nu_data[i, :] = vessel.nu
    eta_data[i, :] = vessel.eta

# Plotting
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(np.arange(n_steps) * dt, eta_data[:, 0], label='Surge (x)')
plt.ylabel('Surge (m)')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(np.arange(n_steps) * dt, eta_data[:, 1], label='Sway (y)')
plt.ylabel('Sway (m)')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(np.arange(n_steps) * dt, eta_data[:, 5], label='Yaw (psi)')
plt.xlabel('Time (s)')
plt.ylabel('Yaw (rad)')
plt.legend()
plt.tight_layout()
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(np.arange(n_steps) * dt, nu_data[:, 0], label='Surge velocity (u)')
plt.ylabel('Surge velocity (m/s)')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(np.arange(n_steps) * dt, nu_data[:, 1], label='Sway velocity (v)')
plt.ylabel('Sway velocity (m/s)')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(np.arange(n_steps) * dt, nu_data[:, 5], label='Yaw rate (r)')
plt.xlabel('Time (s)')
plt.ylabel('Yaw rate (rad/s)')
plt.legend()
plt.tight_layout()
plt.show()