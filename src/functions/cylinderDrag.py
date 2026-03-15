import numpy as np


def cylinderDrag(L, B, nu_r):
    """
    Reynolds number and aspect-ratio dependent 2D drag coefficient for cylinders.

    Parameters
    ----------
    L : float
        Length
    B : float
        Beam
    nu_r : array-like (6,)
        Relative velocity vector [u-u_c, v-v_c, w-w_c, p, q, r]

    Returns
    -------
    Cd_2D : float
        2D drag coefficient
    """

    nu_r = np.asarray(nu_r)

    CD_DATA = np.array([
        [10211.0405297256, 1.20769],
        [15543.5423211490, 1.20369],
        [24434.8681540938, 1.20823],
        [35719.8807222993, 1.21060],
        [60880.7508493947, 1.21093],
        [86174.7436190610, 1.20901],
        [118086.431856986, 1.21134],
        [149262.255516420, 1.21148],
        [214708.876786124, 1.21171],
        [230940.445891927, 1.20109],
        [271581.385758833, 1.16918],
        [297098.185911873, 1.11163],
        [325252.018159382, 1.00926],
        [361942.848517458, 0.89411],
        [409798.156149477, 0.70855],
        [475302.392082697, 0.53155],
        [524734.960427919, 0.40785],
        [578921.928291835, 0.32683],
        [638292.879987135, 0.28422],
        [744046.366191920, 0.29711],
        [853237.628592033, 0.32280],
        [1068685.70450627, 0.38909],
        [1393381.37138896, 0.47033],
        [1831850.38455424, 0.53878],
        [2258395.50114555, 0.58372],
        [2899011.91287642, 0.62868],
        [3663279.03082383, 0.64803],
        [4937305.49552761, 0.67808],
    ])

    KAPPA_SUBCRITICAL_DATA = np.array([
        [2, 0.58],
        [5, 0.62],
        [10, 0.68],
        [20, 0.74],
        [40, 0.82],
        [50, 0.87],
        [100, 0.98]
    ])

    KAPPA_SUPERCRITICAL_DATA = np.array([
        [2, 0.80],
        [5, 0.80],
        [10, 0.82],
        [20, 0.90],
        [40, 0.98],
        [50, 0.99],
        [100, 1.00]
    ])

    # Cross-flow velocity
    U_crossflow = np.sqrt(nu_r[1]**2 + nu_r[2]**2)

    # Reynolds number
    Re = U_crossflow * L * 1e6

    # Cd interpolation
    Cd = np.interp(
        Re,
        CD_DATA[:, 0],
        CD_DATA[:, 1],
        left=CD_DATA[0, 1],
        right=CD_DATA[-1, 1]
    )

    AR = L / B

    # Aspect ratio correction
    if Re < 2e5:
        kappa = np.interp(
            AR,
            KAPPA_SUBCRITICAL_DATA[:, 0],
            KAPPA_SUBCRITICAL_DATA[:, 1],
            left=KAPPA_SUBCRITICAL_DATA[0, 1],
            right=KAPPA_SUBCRITICAL_DATA[-1, 1]
        )
    else:
        kappa = np.interp(
            AR,
            KAPPA_SUPERCRITICAL_DATA[:, 0],
            KAPPA_SUPERCRITICAL_DATA[:, 1],
            left=KAPPA_SUPERCRITICAL_DATA[0, 1],
            right=KAPPA_SUPERCRITICAL_DATA[-1, 1]
        )

    Cd_2D = Cd * kappa

    return Cd_2D