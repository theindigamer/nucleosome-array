import numpy as np

def mean_std(x, **kwargs):
    """Compute both mean and standard deviation of a DataArray.

    Provided purely for convenience, useful for making plots.
    """
    return (x.mean(**kwargs), x.std(**kwargs))


def norm(dataarray, dim='axis'):
    """Computes the Euclidean norm of a DataArray along a dimension.

    Like numpy.linalg.norm but you can use a dimension instead of an axis.
    """
    return ((dataarray**2).sum(dim=dim))**0.5


def partition(n_parts, total):
    arr = np.array([total // n_parts] * n_parts if total >= n_parts else [])
    rem = total % n_parts
    if rem != 0:
        return np.append(arr, rem)
    return arr


def twist_steps(default_step_size, twists):
    try:
        f = np.float(twists)
        tmp = default_step_size if abs(default_step_size) <= abs(f) else f
        return smart_arange(0., f, tmp)
    except TypeError:
        if type(twists) is tuple:
            x = len(twists)
            if x == 2:
                return smart_arange(0., twists[0], twists[1])
            elif x == 3:
                return smart_arange(*twists)
            else:
                raise ValueError("The tuple must be of length 2 or 3.")
        else:
            return np.array(twists, dtype=float)


def smart_arange(start, stop, step, incl=True):
    s = step if (stop - start) * step > 0 else -step
    return np.arange(start, stop + (s if incl else 0.0), s)
