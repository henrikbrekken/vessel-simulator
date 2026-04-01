import matplotlib
import numpy as np

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

from control.polar_controller import PolarController
from plotter.plotter import Plotter
from reference_model.second_order_reference_model import \
    SecondOrderVelocityAndAccelerationSaturationReferenceModel
from vessel.far_spicka import FarSpica
from vessel.polar_transformation import polarTransformation

time = 300  # In secons
dt = 0.1  # Time step
n_steps = int(time / dt)

vessel = FarSpica(np.array([100, 0, 0, 0, 0, 0]))
vessel_polar_transformed = polarTransformation(vessel, 0, 0)
polar_controller = PolarController(
    vessel_polar_transformed, 0.1 * np.array([5, 3, 3]), np.array([1, 1, 1]), dt
)

# Reference models
rho_reference_model = SecondOrderVelocityAndAccelerationSaturationReferenceModel(
    x_init=vessel_polar_transformed.eta[0],
    h=dt,
    wn_f=0.1,
    zeta_f=2,
    delta_f=1,
    v_max=2,
    a_max=0.05,
)
gamma_reference_model = (
    SecondOrderVelocityAndAccelerationSaturationReferenceModel.angular(
        x_init=vessel_polar_transformed.eta[1],
        h=dt,
        wn_f=0.04,
        zeta_f=1,
        delta_f=1,
        v_max=2 / vessel_polar_transformed.eta[0],
        a_max=0.01 / vessel_polar_transformed.eta[0],
    )
)
psi_reference_model = (
    SecondOrderVelocityAndAccelerationSaturationReferenceModel.angular(
        x_init=vessel_polar_transformed.eta[2],
        h=dt,
        wn_f=0.04,
        zeta_f=1,
        delta_f=1,
        v_max=2,
        a_max=0.01,
    )
)
# Weather state
Vc = 0
betaVc = 0

# Store data for plotting eta_data = np.zeros((n_steps, 3))  # Rows and columns
polar_eta_data = np.zeros((n_steps, 3))
nu_data = np.zeros((n_steps, 3))
polar_eta_d_data = np.zeros((n_steps, 3))
tau_data = np.zeros((n_steps, 6))

for i in range(n_steps):
    # Compute forces and moments
    rho_d = rho_reference_model.x_d(130)
    gamma_d = gamma_reference_model.x_d(np.deg2rad(30))
    psi_d = psi_reference_model.x_d(np.deg2rad(40))
    eta_d = np.array([rho_d, gamma_d, psi_d])

    tau = polar_controller.compute_control(eta_d)

    vessel_polar_transformed.integrate_dynamics(dt, tau, Vc, betaVc)

    polar_eta_data[i, :] = vessel_polar_transformed.eta
    polar_eta_d_data[i, :] = eta_d
    nu_data[i, :] = vessel_polar_transformed.nu
    tau_data[i, :] = tau

# Plotting
polar_plotter = Plotter(n_steps, dt)
polar_plotter.plot_eta(polar_eta_data, polar_eta_d_data, polar=True)
polar_plotter.plot_nu(nu_data)
polar_plotter.plot_tau(tau_data)

plt.show()
