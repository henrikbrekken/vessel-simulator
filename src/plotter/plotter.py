import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    def __init__(self, n_steps: int, dt: float):
        self._n_steps = n_steps
        self._dt = dt

    def plot_eta(self, eta_data, eta_d_data, polar=False):
        if polar:
            x_label = r"$\rho$"
            y_label = r"$\gamma$"
            y_unit = " (m)"
        else:
            x_label = "x"
            y_label = "y"
            y_unit = " (m)"

        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(self._n_steps) * self._dt, eta_data[:, 0], label=x_label)
        plt.plot(
            np.arange(self._n_steps) * self._dt,
            eta_d_data[:, 0],
            label=x_label + "_d",
        )
        plt.ylabel(x_label + " (m)")
        plt.legend()
        plt.grid()
        plt.subplot(3, 1, 2)
        plt.plot(
            np.arange(self._n_steps) * self._dt,
            np.rad2deg(eta_data[:, 1]),
            label=y_label,
        )
        plt.plot(
            np.arange(self._n_steps) * self._dt,
            np.rad2deg(eta_d_data[:, 1]),
            label=y_label + "_d",
        )
        plt.ylabel(y_label + y_unit)
        plt.legend()
        plt.grid()
        plt.subplot(3, 1, 3)
        plt.plot(
            np.arange(self._n_steps) * self._dt,
            np.rad2deg(eta_data[:, 2]),
            label=r"$\psi$",
        )
        plt.plot(
            np.arange(self._n_steps) * self._dt,
            np.rad2deg(eta_d_data[:, 2]),
            label=r"$\psi_d$",
        )
        plt.xlabel("Time (s)")
        plt.ylabel(r"$\psi€$ (deg)")
        plt.legend()
        plt.grid()
        plt.tight_layout()

    def plot_nu(self, nu_data):
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(self._n_steps) * self._dt, nu_data[:, 0])
        plt.ylabel("Surge velocity (m/s)")
        plt.grid()
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(self._n_steps) * self._dt, nu_data[:, 1])
        plt.ylabel("Sway velocity (m/s)")
        plt.grid()
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(self._n_steps) * self._dt, nu_data[:, 2])
        plt.xlabel("Time (s)")
        plt.ylabel("Yaw rate (rad/s)")
        plt.grid()
        plt.tight_layout()

    def plot_tau(self, tau_data):
        if len(tau_data[0, :]) == 6:
            tau_data = tau_data[:, [0, 1, 5]]
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(self._n_steps) * self._dt, tau_data[:, 0] / 1000)
        plt.ylabel("X (kN)")
        plt.grid()
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(self._n_steps) * self._dt, tau_data[:, 1] / 1000)
        plt.ylabel("Y (kN)")
        plt.grid()
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(self._n_steps) * self._dt, tau_data[:, 2] / 1000)
        plt.ylabel("Z (kN)")
        plt.grid()
        plt.tight_layout()
