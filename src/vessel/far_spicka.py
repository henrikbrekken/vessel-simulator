from pathlib import Path

import numpy as np
from scipy.io import loadmat

from functions.crossFlowDrag import crossFlowDrag
from functions.Dmtrx import Dmtrx
from functions.forceSurgeDamping import forceSurgeDamping
from functions.Gmtrx import Gmtrx
from functions.m2c import m2c
from functions.rbody import rbody
from functions.rk4 import rk4
from functions.Rzyx import Rzyx
from functions.Tzyx import Tzyx
from vessel.vessel import Vessel


class FarSpica(Vessel):
    def __init__(self, eta0):
        self.nu = np.zeros(6)
        self.eta = eta0

        # Ship model parameters
        self.L = 81
        self.B = 18
        self.T = 4.9
        self.rho = 1025
        self.Cb = 0.75

        self.nabla = self.Cb * self.L * self.B * self.T
        self.m = self.rho * self.nabla

        self.r_bg = np.array([-5.3, 0, -1.0])

        self.thrust_max = 1000e3
        self.U_max = 7.7

        self.Cw = 0.8
        self.Awp = self.Cw * self.B * self.L

        self.KB = (1 / 3) * (5 * self.T / 2 - self.nabla / self.Awp)

        self.k_munro_smith = (6 * self.Cw**3) / ((1 + self.Cw) * (1 + 2 * self.Cw))

        self.r_bb = np.array([-5.3, 0, self.T - self.KB])
        self.BG = self.r_bb[2] - self.r_bg[2]

        self.I_T = self.k_munro_smith * (self.B**3 * self.L) / 12
        self.I_L = 0.7 * (self.L**3 * self.B) / 12

        self.BM_T = self.I_T / self.nabla
        self.BM_L = self.I_L / self.nabla

        self.GM_T = self.BM_T - self.BG
        self.GM_L = self.BM_L - self.BG

        # Radii of gyration
        self.R44 = 0.35 * self.B
        self.R55 = 0.25 * self.L
        self.R66 = 0.25 * self.L

        # Rigid body mass matrix
        self.MRB, _ = rbody(
            self.m, self.R44, self.R55, self.R66, np.zeros(3), self.r_bg
        )

        # Added mass
        supply_file = Path(__file__).with_name("supply.mat")
        supply = loadmat(supply_file)
        self.MA = supply["vessel"]["A"][0][0][:, :, 0]

        # Calibration
        self.MA[2, 2] = 0.8e7
        self.MA[3, 3] = 10.2e7
        self.MA[4, 4] = 4.2e9

        # Total mass matrix
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)
        self.Mdiag_3dof = np.diag([self.M[0, 0], self.M[1, 1], self.M[5, 5]])
        self.M_3DOF = self.M[np.ix_([0, 1, 5], [0, 1, 5])]

        # Hydrostatics
        self.LCF = -0.5

        self.G = Gmtrx(
            self.nabla, self.Awp, self.GM_T, self.GM_L, self.LCF, np.zeros(3)
        )

        # Damping parameters
        self.T1 = 50
        self.T2 = 10
        self.T6 = 1

        self.zeta4 = 0.30
        self.zeta5 = 0.35

        self.D = Dmtrx(
            [self.T1, self.T2, self.T6],
            [self.zeta4, self.zeta5],
            self.MRB,
            self.MA,
            self.G,
        )
        self.D_3DOF = self.D[np.ix_([0, 1, 5], [0, 1, 5])]

        # Wind parameters
        scale = 18 * 81 / (self.L * self.B)

        self.ALw = scale * 336.80
        self.AFw = scale * 137.48
        self.sH = scale * 4.08
        self.sL = scale * 7.95

    def dynamics(self, x, tau, Vc, betaVc):

        nu = x[0:6]
        eta = x[6:12]

        # Ocean current
        u_c = Vc * np.cos(betaVc - eta[5])
        v_c = Vc * np.sin(betaVc - eta[5])

        nu_c = np.array([u_c, v_c, 0, 0, 0, 0])
        nu_dot_c = np.array([nu[5] * v_c, -nu[5] * u_c, 0, 0, 0, 0])

        nu_r = nu - nu_c

        # Coriolis
        _, self.CRB = rbody(self.m, self.R44, self.R55, self.R66, nu[3:6], self.r_bg)

        self.CA = m2c(self.MA, nu_r)

        # Surge damping
        self.S = self.L * self.B + 2 * self.T * self.B

        X_drag, _, _ = forceSurgeDamping(
            0,
            nu_r[0],
            self.m,
            self.S,
            self.L,
            self.T1,
            self.rho,
            self.U_max,
            self.thrust_max,
        )

        tau_surge = np.zeros(6)
        tau_surge[0] = X_drag

        self.D[0, 0] = 0

        tau_cross = crossFlowDrag(self.L, self.B, self.T, nu_r)

        # Kinematics
        self.J = np.block(
            [
                [Rzyx(eta[3], eta[4], eta[5]), np.zeros((3, 3))],
                [np.zeros((3, 3)), Tzyx(eta[3], eta[4])],
            ]
        )

        # Dynamics
        self.eta_dot = self.J @ nu

        self.nu_dot = nu_dot_c + self.Minv @ (
            tau
            + tau_surge
            + tau_cross
            - (self.CRB + self.CA + self.D) @ nu_r
            - self.G @ eta
        )

        xdot = np.concatenate((self.nu_dot, self.eta_dot))

        return xdot

    def integrate_dynamics(self, dt, tau, Vc, betaVc):

        x = np.concatenate([self.nu, self.eta])
        x = rk4(self.dynamics, dt, x, tau, Vc, betaVc)
        self.nu = x[0:6]
        self.eta = x[6:12]
