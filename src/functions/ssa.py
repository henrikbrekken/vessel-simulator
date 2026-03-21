import numpy as np

def ssa(angle, unit="rad"):
    if unit == "rad":
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi
    elif unit == "deg":
        return np.mod(angle + 180, 360) - 180
    else:
        raise ValueError(
            f"Invalid unit argument: '{unit}'. Unit must be either 'deg' or 'rad'"
        )
