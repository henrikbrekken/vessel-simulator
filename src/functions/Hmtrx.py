import numpy as np
from .Smtrx import Smtrx

def Hmtrx(r):
    """
    6x6 screw transformation matrix.
    Used to shift inertia matrices from CG to CO.
    """
    S = Smtrx(r)
    I3 = np.eye(3)
    O3 = np.zeros((3, 3))

    H = np.block([[I3, S.T],
                  [O3, I3]])
    
    return H