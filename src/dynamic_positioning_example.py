import matplotlib
import numpy as np

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from control.dp_controller import DPController
from plotter.plotter import Plotter
from reference_model.second_order_reference_model import \
    SecondOrderVelocityAndAccelerationSaturationReferenceModel
from vessel.far_spicka import FarSpica

time = 300  # In secons
dt = 0.1  # Time step
n_steps = int(time / dt)

vessel = FarSpica(np.array([0, 0, 0, 0, 0, 0]))
controller = DPController(vessel, np.array([1, 1, 1]), np.array([1, 1, 1]), dt)

x_ref_model = SecondOrderVelocityAndAccelerationSaturationReferenceModel(
    x_init=0, h=dt, wn_f=0.2, zeta_f=2, delta_f=1, v_max=2, a_max=0.05
)
y_ref_model = SecondOrderVelocityAndAccelerationSaturationReferenceModel(
    x_init=0, h=dt, wn_f=0.2, zeta_f=2, delta_f=1, v_max=2, a_max=0.05
)
psi_ref_model = SecondOrderVelocityAndAccelerationSaturationReferenceModel.angular(
    x_init=0, h=dt, wn_f=0.04, zeta_f=1, delta_f=1, v_max=2, a_max=0.01
)

# Weather state
Vc = 0
betaVc = 0

# Store data for plotting
eta_data = np.zeros((n_steps, 3))  # Rows and columns
eta_d_data = np.zeros((n_steps, 3))  # Rows and columns
nu_data = np.zeros((n_steps, 3))
tau_data = np.zeros((n_steps, 3))

for i in range(n_steps):
    # Compute forces and moments
    x_d = x_ref_model.x_d(20)
    y_d = y_ref_model.x_d(10)
    psi_d = psi_ref_model.x_d(np.deg2rad(30))
    eta_d = np.array([x_d, y_d, psi_d])

    tau = controller.compute_control(eta_d)

    vessel.integrate_dynamics(dt, tau, Vc, betaVc)

    eta_data[i, :] = vessel.eta_3DOF
    eta_d_data[i, :] = eta_d
    nu_data[i, :] = vessel.nu_3DOF
    tau_data[i, :] = tau[[0, 1, 5]]

# Plotting
plotter = Plotter(n_steps, dt)
plotter.plot_eta(eta_data, eta_d_data)
plotter.plot_tau(tau_data)
plt.show()
