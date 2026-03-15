import numpy as np

def Smtrx(v):
    """
    Skew-symmetric matrix such that Smtrx(v) @ w = v × w
    """
    v1, v2, v3 = v
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])