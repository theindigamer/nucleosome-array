import numpy as np
from numba import jit
import xarray as xr
from copy import copy


#-------------------#
# Data manipulation #
#-------------------#

def _scale(arr, last=0.):
    """Linearly rescale data to give a fixed value for the last element."""
    a = arr.copy()
    L = np.shape(arr)[-1]
    for x in range(1, L):
        a[..., x] = a[..., x] - x * a[..., -1] / (L - 1) + last / (L - 1)
    return a

def generate_rw_2d(d, B, count, L, last=0.):
    u"""Generates random walks in 2D with the right boundary conditions.

    Args:
        d (float): length of 1 rod in nm
        B (float): usual bending constant in nm·kT
        count (int): number of random walks to generate
        L (int): number of rods

    Returns:
        θ values (Array[(count, L)]) corresponding to the random walk.
    """
    sigma = np.sqrt(d / B)
    beta = np.empty((count, L))
    beta[:, 0] = 0.
    beta[:, 1:] = sigma * np.random.randn(count, L - 1)
    theta = np.cumsum(beta, axis=1)
    # Last angle might be non-zero now, so we fix that with linear scaling.
    return _scale(theta, last=last)

def generate_rw_3d(d, B, count, L, C=None, final_psi=None):
    """Generates random walks in 3D with the right boundary conditions.

    Args:
        d (float): length of 1 rod in nm
        B (float): usual bending constant in nm kT
        count (int): number of random walks to generate
        L (int): number of rods
        C (float): usual twisting constant in nm kT
        final_psi (float): optionally specify value of psi at the end

    Returns:
        [φ, θ, ψ] values corresponding to the random walk as an array of shape
        (count, L, 3). If C is not supplied, psi values are zeroed out.
        If final_psi and C are both supplied, the last value of psi is fixed
        to the final_psi value. This can be useful if you want to emulate
        twisting.

    FIXME:
        Implementation is wholly incorrect.
    """
    # We know H(φ, θ, ψ) / kT -> we should use this for sampling instead of
    # misusing B and C directly. Possible approaches:
    #   * Metropolis
    #   * Slice sampling - see Neal Radford, Colin has code.
    #   * Event Chain Monte Carlo (Krauth et al.) - Colin says it's very fast.
    #     https://arxiv.org/pdf/0903.2954.pdf
    theta = generate_rw_2d(d, B, count, L)
    if C is None:
        psi = np.zeros((count, L))
    else:
        psi = generate_rw_2d(
            d, C, count, L , last=(0. if final_psi is None else final_psi))
    # the value of phi doesn't affect the energy
    phi = 2. * np.pi * np.random.rand(count, L)
    return np.moveaxis(np.array([phi, theta, psi]), 0, 2)

def autocorr_fft(arr):
    L = arr.shape[-2]
    lengths = arr.shape[:-2]
    tangents = np.empty_like(arr)
    for inds in np.ndindex(lengths):
        tangents[inds] = unit_tangent_vectors(arr[inds])
    tmpk = np.fft.rfft(tangents, axis=-2)
    corr = np.fft.irfft(tmpk * tmpk.conj(), axis=-2)
    # needs additional normalization after summing
    return np.sum(corr, axis=-1)/L


@jit(cache=True, nopython=True)
def autocorr_brute_force(arr):
    L = arr.shape[-2]
    lengths = arr.shape[:-2]
    tangents = np.empty_like(arr)
    arr3 = 0. * arr
    counters = np.zeros(L)
    for m in range(L):
        for n in range(m, L):
            counters[n - m] += 1
    for inds in np.ndindex(lengths):
        tangents[inds] = unit_tangent_vectors(arr[inds])
        for m in range(L):
            for n in range(m, L):
                arr3[inds][n - m] += tangents[inds][m] * tangents[inds][n]
        for i in range(3):
            arr3[inds][:, i] /= counters
    return np.sum(arr3, axis=-1)


def bend_angles(arr, axis=-1, n_axis=-2):
    u"""
    Computes successive bends using the formula β_i = θ_{i+1} - θ_i if the
    other angles are set to zero, otherwise computes dot products of
    consecutive tangent vectors and takes an inverse cosine.

    In the first case, the result is signed, whereas in the second case it
    is always positive.
    """
    if (arr[..., 0] == 0.).all() and (arr[..., 2] == 0.).all():
        shape = np.shape(arr)
        if axis == len(shape) - 1 and n_axis == axis - 1:
            beta = np.empty(shape[:-1])
            beta[..., -1] = 0.
            beta[..., :-1] = arr[..., 1:, 1] - arr[..., :-1, 1]
            return beta
        else:
            raise ValueError("Code assumes that last axis is 'angle_str' and the"
                             " penultimate axis is 'n'.")
    else:
        lengths = arr.shape[:-2]
        tangents = np.empty_like(arr)
        for inds in np.ndindex(lengths):
            tangents[inds] = unit_tangent_vectors(arr[inds])
        beta = np.empty(arr.shape[:-1])
        beta[..., -1] = 0.
        beta[..., :-1] = np.arccos(
            np.sum(tangents[..., 1:, :] * tangents[..., :-1, :], axis=-1))
        return beta


def bend_autocorr(arr, axis=None, n_axis=None, method="fft"):
    u"""
    Args:
        arr (ndarray): Array containing raw angles. Last axis ↔ φ, θ, ψ.
        axis (int): Should be the last axis of arr. This kwargs is required
                    by xarray's reduce operation (the dim kwarg for reduce
                    gets mapped to axis). It corresponds to the three angles.
        n_axis (int): Should be the second last axis of arr. This corresponds
                      to rod number.
        method (str): One of "fft" or "brute force".

    Returns:
        An array for bend autocorrelation. The shape is the same as the input
        array with the last dimension removed.
    """
    shape = np.shape(arr)
    if axis == len(shape) - 1 and n_axis == axis - 1:
        if method == "fft":
            return autocorr_fft(arr)
        elif method == "brute force":
            return autocorr_brute_force(arr)
        else:
            raise ValueError("Unrecognized method argument.")
    else:
        raise ValueError("Code assumes that last axis is 'angle_str' and the"
                         " penultimate axis is rod number.")


def compute_bend_autocorr(dataset, method="fft"):
    return (dataset["angles"].copy()
            .reduce(bend_autocorr, dim="angle_str",
                    n_axis=dataset["angles"].get_axis_num('n'), method=method)
            .rename("bend_autocorr"))


def add_bend_autocorr(dataset, method="fft"):
    da = compute_bend_autocorr(dataset, method=method)
    da2 = (dataset["angles"].copy()
           .reduce(bend_angles, dim="angle_str",
                   n_axis=dataset["angles"].get_axis_num('n'))
           .rename("bend_angle"))
    ds = xr.merge([dataset, da, da2])
    # For some reason, the merge function destroys attributes.
    # See https://github.com/pydata/xarray/issues/1614 and links therein.
    # So we copy the attributes separately.
    ds.attrs = dataset.attrs
    return ds


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
          3 datasets with [DV: X, DIM: Y, ATTR: Z] and you want to concatenate
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
    lengths = tuple(map(len, new_coords))

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
    ds = np.reshape(ds, lengths)

    def f(datasets, new_dims, new_coords):
        if len(new_dims) == 1:
            data_vars = {k: datasets[0].data_vars[k] for k in common_data_vars}
            coords = copy(datasets[0].coords)
            coords.update({new_dims[0]: new_coords[0]})
            attrs = {k: datasets[0].attrs[k] for k in common_attrs}
            for k in concat_data_vars:
                data_array = xr.concat([ds.data_vars[k] for ds in datasets], new_dims[0])
                data_vars.update({k: data_array})
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
def unit_tangent_vector1(euler):
    t = np.empty(3)
    phi = euler[0]
    theta = euler[1]
    sin_theta = np.sin(theta)
    t[0] = sin_theta * np.sin(phi)
    t[1] = sin_theta * np.cos(phi)
    t[2] = np.cos(theta)
    return t


@jit(cache=True, nopython=True)
def unit_tangent_vectors(euler):
    n = len(euler)
    t = np.empty((n, 3))
    for i in range(n):
        phi = euler[i, 0]
        theta = euler[i, 1]
        sin_theta = np.sin(theta)
        t[i, 0] = sin_theta * np.sin(phi)
        t[i, 1] = sin_theta * np.cos(phi)
        t[i, 2] = np.cos(theta)
    return t


@jit(cache=True, nopython=True)
def metropolis(reject, deltaE, even=True):
    """Updates reject in-place using the Metropolis algorithm.

    Args:
        reject (Array[(x,); bool]):
            Array to be modified in-place. If even is True, it is assumed that
            reject has True at even indices and similarly when even is False.
        deltaE (Array[(x,)]):
            Local energy changes used to check for rejection. Energy should be
            in units of kT.
        even (bool): Indicates if even/odd indices of deltaE should be checked.

    Returns:
        None

    Note:
        The x in the sizes indicates that the two array sizes have to be equal.
        For example, you will have x = L - 1 when the two rods at the end have
        fixed orientation.
    """
    for i in range(0 if even else 1, reject.size, 2):
        if deltaE[i] < 0:
            reject[i] = False
        elif deltaE[i] < 16 and np.exp(-deltaE[i]) > np.random.rand():
            reject[i] = False


@jit(cache=True, nopython=True)
def twist_bend_angles(Deltas, squared):
    u"""Computes twist and bend values for an array of Delta matrices.

    Args:
        Deltas (Array[(X, 3, 3)]):
            Matrices describing relative twists and bends at hinges.
        squared (bool): Returns

    Returns:
        (β², β², Γ²) if squared is true.
        (β₁, β₂, Γ) if squared is false.
        Individual terms are arrays of shape (X,).

    Note:
        See [DS, Appendix D] for equations.

        The dummy value of β² is present if squared is true because Numba
        requires the type signatures of possible return values to be the
        same.
    """
    n = len(Deltas)
    if squared:
        beta_sq = np.empty(n)
        Gamma_sq = np.empty(n)
        for i in range(n):
            beta_sq[i] = 2.0 * (1.0 - Deltas[i, 2, 2])
            Gamma_sq[i] = (1.0 - Deltas[i, 0, 0] - Deltas[i, 1, 1]
                           + Deltas[i, 2, 2])
        return (beta_sq, beta_sq, Gamma_sq)
    else:
        beta_1 = np.empty(n)
        beta_2 = np.empty(n)
        Gamma = np.empty(n)
        for i in range(n):
            beta_1[i] = (Deltas[i, 1, 2] - Deltas[i, 2, 1]) / 2.0
            beta_2[i] = (Deltas[i, 2, 0] - Deltas[i, 0, 2]) / 2.0
            Gamma[i]  = (Deltas[i, 0, 1] - Deltas[i, 1, 0]) / 2.0
        return (beta_1, beta_2, Gamma)


@jit(cache=True, nopython=True)
def rotation_matrix(angles):
    u"""Computes a single rotation matrix.

    Args:
        angles (Array[(3,)]): Euler angles [φ, θ, ψ].

    Returns:
        A 3x3 rotation matrix

    Note:
        Represents [DS, Eqn. (B1, B2)].
    """
    tmp = np.empty((3,3))
    set_rotation_matrix(angles, tmp)
    return tmp


@jit(cache=True, nopython=True)
def set_rotation_matrix(angles, res):
    u"""Computes a single rotation matrix.

    Args:
        angles (Array[(3,)]): Euler angles [φ, θ, ψ].
        res (Array[(3, 3)]): result array to save rotation matrix.

    Returns:
        None. res is mutated in-place.

    Note:
        Represents [DS, Eqn. (B1, B2)].
    """
    phi = angles[0]
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    theta = angles[1]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    psi = angles[2]
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    res[0, 0] = cos_phi * cos_psi - cos_theta * sin_phi * sin_psi
    res[0, 1] = cos_phi * sin_psi + cos_theta * cos_psi * sin_phi
    res[0, 2] = sin_theta * sin_phi

    res[1, 0] = -cos_psi * sin_phi - cos_theta * cos_phi * sin_psi
    res[1, 1] = -sin_phi * sin_psi + cos_theta * cos_phi * cos_psi
    res[1, 2] = cos_phi * sin_theta

    res[2, 0] = sin_theta * sin_psi
    res[2, 1] = -cos_psi * sin_theta
    res[2, 2] = cos_theta

@jit(cache=True, nopython=True)
def rotation_matrices(start, euler, end):
    u"""Computes rotation matrices element-wise.

    Args:
        start (Array[(3,)]): Euler angles for (-1)-th point, ordered [φ, θ, ψ].
        euler (Array[(L, 3)]): Euler angles for rods, ordered [φ, θ, ψ].
        end (Array[(3,)]): Euler angles for last point, ordered [φ, θ, ψ].

    Returns:
        Passive rotation matrices in an array of shape (L+2, 3, 3).

    Note:
        Represents [DS, Eqn. (B1, B2)].
    """
    n = len(euler)
    R = np.empty((n + 2, 3, 3))
    set_rotation_matrix(start, R[0])
    for i in range(n):
        set_rotation_matrix(euler[i], R[i + 1])
    set_rotation_matrix(end, R[n + 1])
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

_r_0 = 4.18    # in nm, for central line of DNA wrapped around nucleosome
_z_0 = 2.39    # pitch of superhelix in nm
_n_wrap = 1.65 # number of times DNA winds around nucleosome
_zeta_max = 2. * np.pi * _n_wrap
_helix_entry_tilt = np.arctan2(-2. * np.pi * _r_0, _z_0)
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
    # Using tuples because nested lists don't work with np.array in Numba 0.35.0
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

@jit(cache=True, nopython=True)
def md_jacobian(tangent):
    n = len(tangent)
    J = np.empty((n, 4, 4))
    for i in range(n):
        t = tangent[i]
        D_sq = t[0]**2 + t[1]**2 + t[2]**2
        D = np.sqrt(D_sq)
        p_sq = t[0]**2 + t[1]**2 + 1.E-16
        p = np.sqrt(p_sq)

        J[i, 0, 0] = -t[0] / D
        J[i, 0, 1] = -t[1] / D
        J[i, 0, 2] = -t[2] / D
        J[i, 0, 3] = 0.

        J[i, 1, 0] = -t[1] / p_sq
        J[i, 1, 1] = +t[0] / p_sq
        J[i, 1, 2] = 0.
        J[i, 1, 3] = 0.

        J[i, 2, 0] = -t[0] * t[2] / (p * D_sq)
        J[i, 2, 1] = -t[1] * t[2] / (p * D_sq)
        J[i, 2, 2] = p / D_sq
        J[i, 2, 3] = 0.

        # J[i, 3, 0] = 0.
        # J[i, 3, 1] = 0.
        # non-zero terms as per [DS, Eq (C6)] but with a - sign
        J[i, 3, 0] = -t[1] * t[2] / (p_sq * D)
        J[i, 3, 1] = +t[0] * t[2] / (p_sq * D)
        J[i, 3, 2] = 0.
        J[i, 3, 3] = 0.
    return J

@jit(cache=True, nopython=True)
def md_derivative_rotation_matrices(euler):
    n = len(euler)
    DR = np.empty((n, 3, 3, 3))
    for i in range(n):
        phi = euler[i, 0]
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        theta = euler[i, 1]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        psi = euler[i, 2]
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        DR[i, 0, 0, 0] = -sin_phi * cos_psi - cos_phi * cos_theta * sin_psi
        DR[i, 0, 0, 1] = sin_phi * sin_theta * sin_psi
        DR[i, 0, 0, 2] = -sin_phi * cos_theta * cos_psi - cos_phi * sin_psi

        DR[i, 0, 1, 0] = cos_phi * cos_theta * cos_psi - sin_phi * sin_psi
        DR[i, 0, 1, 1] = -sin_phi * sin_theta * cos_psi
        DR[i, 0, 1, 2] = cos_phi * cos_psi - sin_phi * cos_theta * sin_psi

        DR[i, 0, 2, 0] = cos_phi * sin_theta
        DR[i, 0, 2, 1] = sin_phi * cos_theta
        DR[i, 0, 2, 2] = 0.

        DR[i, 1, 0, 0] = -cos_phi * cos_psi + sin_phi * cos_theta * sin_psi
        DR[i, 1, 0, 1] = cos_phi * sin_theta * sin_psi
        DR[i, 1, 0, 2] = -cos_phi * cos_theta * cos_psi + sin_phi * sin_psi

        DR[i, 1, 1, 0] = -sin_phi * cos_theta * cos_psi - cos_phi * sin_psi
        DR[i, 1, 1, 1] = -cos_phi * sin_theta * cos_psi
        DR[i, 1, 1, 2] = -sin_phi * cos_psi - cos_phi * cos_theta * sin_psi

        DR[i, 1, 2, 0] = -sin_phi * sin_theta
        DR[i, 1, 2, 1] = cos_phi * cos_theta
        DR[i, 1, 2, 2] = 0.

        DR[i, 2, 0, 0] = 0.
        DR[i, 2, 0, 1] = cos_theta * sin_psi
        DR[i, 2, 0, 2] = sin_theta * cos_psi

        DR[i, 2, 1, 0] = 0.
        DR[i, 2, 1, 1] = -cos_theta * cos_psi
        DR[i, 2, 1, 2] = sin_theta * sin_psi

        DR[i, 2, 2, 0] = 0.
        DR[i, 2, 2, 1] = -sin_theta
        DR[i, 2, 2, 2] = 0.
    return DR

@jit(cache=True, nopython=True)
def md_effective_torques(RStart, Rs, REnd, DRs, L, C, B, d):
    tau = np.zeros((L, 4))
    c1 = - (C + 2. * B) / (2. * d)
    c2 = - C / (2. * d)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = c1 if k == 2 else c2
                tau[0, i + 1] += (c * (RStart[j, k] + Rs[1, j, k])
                                  * DRs[0, j, k, i])
                tau[-1, i + 1] += (c * (Rs[-2, j, k] + REnd[j, k])
                                   * DRs[-1, j, k, i])
                tau[1:-1, i + 1] += (c * (Rs[:-2, j, k] + Rs[2:, j, k])
                                     * DRs[1:-1, j, k, i])
    return tau
