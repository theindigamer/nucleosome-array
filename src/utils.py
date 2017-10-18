import numpy as np
from numba import jit
import xarray as xr
from copy import copy

#-------------------#
# Data manipulation #
#-------------------#

def concat_datasets(datasets, concat_data_vars, new_dims, new_coords,
                    concat_attrs=[]):
    """Concatentate multiple datasets by adding new dimensions.

    The primary use for this function will be combining datasets across multiple
    runs. That way one can simply write a function ``foo`` that does 1 run of a
    simulation, and another helper function which simply calls ``foo`` n times,
    collects all the datasets in a list, and combines them using this function,
    instead of separately implementing multiple run functionality for every
    simulation.

    This function is somewhat like an n-dimensional generalization of
    ``xarray``'s ``concat`` function.

    Args:
        datasets (List[xarray.Dataset]): Datasets to be concatenated.
        concat_data_vars (List[str]): Data variables to be concatentated.
        new_dims (List[str]): The new dimensions for concatenation.
        new_coords (List[List[T]]): New coordinates for each new dimension.
        concat_attrs (List[str]): Attributes to be concatentated.

    Returns:
        A new dataset representing the desired concatenation.

    Note:
        * If a new dimension was saved in the datasets as either a data variable
          or as an attribute, it will be removed. Example: say you have a
          3 datasets with [DV: X, DIM: Y, ATTR:Z] and you want to concatenate
          along Z, then the final dataset will be like [DV: X, DIM: Y Z, ATTR:].
        * You can also use numpy arrays instead of lists.
        * Lists of length 1 should be used if needed instead of "unwrapping";
          the latter may give unexpected results.
    """
    if len(new_dims) != len(new_coords):
        raise ValueError("The number of dimensions should be equal to the"
                         " number of coordinate lists. This mismatch might have"
                         " happened if you forgot to wrap new_dims or"
                         " new_coords in a list.")
    lens = tuple(map(len, new_coords))

    # Some data variables and attributes are shared across datasets, so these
    # will not get a nested structure.
    common_data_vars = []
    for k in datasets[0].data_vars.keys():
        if not (k in concat_data_vars or k in new_dims):
            common_data_vars = [k] + common_data_vars
    common_attrs = []
    for k in datasets[0].attrs.keys():
        if not (k in concat_attrs or k in new_dims):
            common_attrs = [k] + common_attrs

    ds = np.empty(len(datasets), dtype="object")
    for (i, d) in enumerate(datasets):
        ds[i] = d
    ds = np.reshape(ds, lens)

    def f(datasets, new_dims, new_coords):
        if len(new_dims) == 1:
            nonlocal common_data_vars
            data_vars = {k: datasets[0].data_vars[k] for k in common_data_vars}
            coords = copy(datasets[0].coords)
            coords.update({new_dims[0]: new_coords[0]})
            nonlocal common_attrs
            attrs = {k: datasets[0].attrs[k] for k in common_attrs}
            nonlocal concat_data_vars
            for k in concat_data_vars:
                data_array = xr.concat([ds.data_vars[k] for ds in datasets], new_dims[0])
                data_vars.update({k: data_array})
            nonlocal concat_attrs
            for k in concat_attrs:
                attr = np.concatenate([ds.attrs[k] for ds in datasets])
                attrs.update({k: attr})
            return xr.Dataset(data_vars, coords=coords, attrs=attrs)
        else:
            tmp = []
            for i in range(len(new_coords[0])):
                tmp.append(f(datasets[i], new_dims[1:], new_coords[1:]))
            return f(tmp, new_dims[0:1], new_coords[0:1])

    return f(ds, new_dims, new_coords)


#------------------------------#
# Simulation utility functions #
#------------------------------#

@jit(cache=True, nopython=True)
def metropolis(reject, deltaE, even=True):
    """Updates reject in-place using the Metropolis algorithm."""
    for i in range(0 if even else 1, reject.size, 2):
        if deltaE[i] < 0:
            reject[i] = 0.
        elif deltaE[i] < 16 and np.exp(-deltaE[i]) > np.random.rand():
            reject[i] = 0.


@jit(cache=True, nopython=True)
def twist_bend_angles(Deltas, squared):
    n = len(Deltas)
    if squared:
        beta_sq = np.empty(n)
        Gamma_sq = np.empty(n)
        for i in range(n):
            beta_sq[i] = 2.0 * (1.0 - Delta[i, 2, 2])
            Gamma_sq[i] = 1.0 - Delta[i, 0, 0] - Delta[i, 1, 1] + Delta[i, 2, 2]
        # TODO: fix the second dummy value
        return (beta_sq, beta_sq, Gamma_sq)
    else:
        beta_1 = np.empty(n)
        beta_2 = np.empty(n)
        Gamma = np.empty(n)
        for i in range(n):
            beta_1[i] = (Delta[i, 1, 2] - Delta[i, 2, 1]) / 2.0
            beta_2[i] = (Delta[i, 2, 0] - Delta[i, 0, 2]) / 2.0
            Gamma[i]  = (Delta[i, 0, 1] - Delta[i, 1, 0]) / 2.0
        return (beta_1, beta_2, Gamma)

@jit(cache=True, nopython=True)
def rotation_matrices(euler):
    """Computes rotation matrices element-wise.

    Assumes: indices 0, 1 and 2 ↔ phi, theta, psi.
    """
    n = len(euler)
    R = np.empty((n, 3, 3))
    for i in range(n):
        phi = euler[i][0]
        theta = euler[i][1]
        psi = euler[i][2]
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)
        R[i, 0, 0] = cos_phi * cos_psi - cos_theta * sin_phi * sin_psi
        R[i, 0, 1] = cos_phi * sin_psi + cos_theta * cos_psi * sin_phi
        R[i, 0, 2] = sin_theta * sin_phi
        R[i, 1, 0] = -cos_psi * sin_phi - cos_theta * cos_phi * sin_psi
        R[i, 1, 1] = -sin_phi * sin_psi + cos_theta * cos_phi * cos_psi
        R[i, 1, 2] = cos_phi * sin_theta
        R[i, 2, 0] = sin_theta * sin_psi
        R[i, 2, 1] = -cos_psi * sin_theta
        R[i, 2, 2] = cos_theta
    return R

def partition(n_parts, total):
    arr = np.array([total // n_parts] * n_parts if total >= n_parts else [])
    rem = total % n_parts
    if rem != 0:
        return np.append(arr, rem)
    return arr

def smart_arange(start, stop, step, incl=True):
    s = step if (stop - start) * step > 0 else -step
    return np.arange(start, stop + (s if incl else 0.0), s)

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

_r_0 = 4.18 # in nm, for central line of DNA wrapped around nucleosome
_z_0 = 2.39 # pitch of superhelix in nm
_n_wrap = 1.65 # number of times DNA winds around nucleosome
_zeta_max = 2.*np.pi*_n_wrap
_helix_entry_tilt = np.arctan2(-2.*np.pi*_r_0, _z_0)
# called lambda in the notes

def normalize(v):
    norm_v = np.linalg.norm(v)
    assert norm_v > 0.
    return v/norm_v

# Final normal, tangent and binormal vectors
_n_f = np.array([-np.cos(_zeta_max), np.sin(_zeta_max), 0.])
_t_f = normalize(np.array([
    -_r_0 * np.sin(_zeta_max),
    -_r_0 * np.cos(_zeta_max),
     _z_0 / (2.*np.pi)
]))
_b_f = np.cross(_t_f, _n_f)
_nf_tf_matrix = np.array([_n_f, _b_f, _t_f])

@jit(cache=True, nopython=True)
def axialRotMatrix(theta, axis=2):
    """Returns an Euler matrix for a passive rotation about a particular axis.

    `axis` should be 0 (x), 1 (y) or 2 (z).
    **Warning**: This function does not do input validation.
    """
    # Using tuples because nested lists don't work with array in Numba 0.35.0
    if axis == 2:
        rot = np.array((
            ( np.cos(theta), np.sin(theta), 0.),
            (-np.sin(theta), np.cos(theta), 0.),
            (            0.,            0., 1.)
        ))
    elif axis == 0:
        rot = np.array((
            (1.,             0.,            0.),
            (0.,  np.cos(theta), np.sin(theta)),
            (0., -np.sin(theta), np.cos(theta))
        ))
    else:
        rot = np.array((
            (np.cos(theta), 0., -np.sin(theta)),
            (           0., 1.,             0.),
            (np.sin(theta), 0.,  np.cos(theta))
        ))
    return rot

@jit(cache=True, nopython=True)
def anglesOfEulerMatrix(m):
    """Returns an array of angles in the order phi, theta, psi.

    This function is the inverse of eulerMatrixOfAngles.
    """
    # NOTE: order in arctan2 is opposite to that of Mathematica
    if m[2][2] == 1.0:
        l = np.array([0., 0., np.arctan2(m[0][1], m[0][0])])
    elif m[2][2] == -1.0:
        l = np.array([0., np.pi, np.arctan2(m[0][1], m[0][0])])
    else:
        l = np.array([
            np.arctan2(m[0][2], m[1][2]),
            np.arccos(m[2][2]),
            np.arctan2(m[2][0], -m[2][1])
        ])
    return l

@jit(cache=True, nopython=True)
def _multiplyMatrices3(m1, m2, m3):
    """Multiplies the given matrices from left to right."""
    # using np.dot as Numba 0.35.0 doesn't support np.matmul (@ operator).
    # See https://github.com/numba/numba/issues/2101
    return np.dot(np.dot(m1, m2), m3)

@jit(cache=True, nopython=True)
def eulerMatrixOfAngles(angles):
    """Returns the Euler matrix for passive rotations for the given angles.

    `angles` is a 1D array of length 3 with angles phi, theta and psi.
    This function is the inverse of anglesOfEulerMatrix
    """
    return _multiplyMatrices3(
        axialRotMatrix(angles[0], axis=2),
        axialRotMatrix(angles[1], axis=0),
        axialRotMatrix(angles[2], axis=2)
    )

@jit(cache=True, nopython=True)
def _AmatrixFromMatrix(entryMatrix):
    # Numba 0.35.0 supports .transpose() but not np.transpose().
    return _multiplyMatrices3(
        axialRotMatrix(_helix_entry_tilt, 0),
        axialRotMatrix(np.pi, 2),
        entryMatrix.transpose()
    )
    # TODO:
    # I don't understand why the last transpose is needed.
    # According to the Mathematica code, it shouldn't be needed.

@jit(cache=True, nopython=True)
def _AmatrixFromAngles(entryangles):
    return _AmatrixFromMatrix(eulerMatrixOfAngles(entryangles))

# angles given in order phi, theta, psi
@jit(cache=True, nopython=True)
def exitMatrix(entryangles):
    return (np.dot(_nf_tf_matrix, _AmatrixFromAngles(entryangles))).transpose()

@jit(cache=True, nopython=True)
def exitAngles(entryangles):
    return anglesOfEulerMatrix(exitMatrix(entryangles))

@jit(cache=True, nopython=True)
def calc_deltas(deltas, nucs, Rs):
    for i in nucs:
        deltas[i] = _multiplyMatrices3(_nf_tf_matrix, _AmatrixFromMatrix(Rs[i-1]), Rs[i])
