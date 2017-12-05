import numpy as np
import fast_calc


class Environment:
    """Describes the environment of the DNA/nucleosome array.

    Future properties one might include: salt/ion concentration etc.
    """
    ROOM_TEMP = 296.65 # in Kelvin. This value is used in [F, page 2] when
                       # measuring B and C.
    MIN_TEMP = 1E-10   # in Kelvin
    def __init__(self, T=ROOM_TEMP):
        self.T = T


class AngularDescription:
    """A barebones description of the angular properties of a strand.

    This class provides several basic functions to work with angular properties.
    Methods should be overriden for specific purposes, such as measuring
    intermediate results. For example, the Monte Carlo simulation overrides
    some functions to record timings for individual computations.
    """

    def __init__(self, L, B, C, T, strand_len, euler=None, end=None):
        u"""Initialize the angular description of a strand.

        Args:
            L (int): number of rods
            B (float): bending modulus
            C (float): twisting modulus
            T (float): temperature in Kelvin
            strand_len (float): total length of strand
            euler (Array[(L, 3)]): Euler angles for all the rods.
            end (Array[(3,)]): Euler angles at the end of the final rod.

        Note:
            B and C should be specified in units of z·kT where T is the
            simulation temperature, and z is the unit used for the strand length
            (usually nm).
        """
        self.L = L
        self.B = B
        self.C = C
        self.env = Environment(T=T)
        self.strand_len = strand_len
        self.d = strand_len / L
        if euler is None:
            self.euler = np.zeros((self.L, 3))
        else:
            if np.shape(euler) != (L, 3):
                raise ValueError("Unexpected shape.")
            self.euler = euler
        if end is None:
            self.end = np.zeros(3)
        else:
            if np.shape(end) != (3,):
                raise ValueError("Unexpected shape.")
            self.end = end

    def rotation_matrices(self):
        """Rotation matrices along the DNA string.

        Returns:
            Rotation matrices all rod ends, in an array of shape (L+1, 3, 3).
        """
        return fast_calc.rotation_matrices(self.euler, self.end)

    def delta_matrices(self, Rs=None):
        u"""Returns Δ matrices describing bends and twists for each hinge.

        Args:
            Rs (Array[(L+1, 3, 3)]): Rotation matrices to use.

        Returns:
            Array of Δ_i = R_i^T · R_{i+1} in an array of shape (L, 3, 3).
        """
        if Rs is None:
            Rs = self.rotation_matrices()
        R_i_transpose = np.swapaxes(Rs[:-1], 1, 2)
        R_i_plus_1 = Rs[1:]
        return R_i_transpose @ R_i_plus_1

    def twist_bend_angles(self, Deltas=None):
        u"""Computes the twist and bend angles.

        Args:
            Deltas (Array[(L, 3, 3)]): Delta matrices to use.

        Returns:
            (β², β², Γ²), each an array of shape (L,).
        """
        if Deltas is None:
            Deltas = self.delta_matrices()
        return fast_calc.twist_bend_angles(Deltas, True)

    def _total(self, f, *args, energy_density=None, **kwargs):
        if energy_density is None:
            energy_density = f(*args, **kwargs)
        if type(energy_density) is tuple:
            # second argument contains angles, discard those
            return np.sum(energy_density[0])
        return np.sum(energy_density)

    def bend_energy_density(self, twist_bends=None):
        """Computes bending energy at each hinge.

        Args:
            twist_bends (Tuple[(3,); Array[(L,)]]):
                (β², _, _) (output of ``twist_bend_angles()``).

        Returns:
            A two element tuple with bending energy for each hinge
            (Array[(L,)]) and twist_bends. The energy_density is in units of
            B / d.
        """
        if twist_bends is None:
            twist_bends = self.twist_bend_angles()
        energy_density = self.B / (2.0 * self.d) * twist_bends[0]
        return energy_density, twist_bends

    def bend_energy(self, *args, **kwargs):
        return self._total(self.bend_energy_density, *args, **kwargs)

    def twist_energy_density(self, twist_bends=None):
        """Computes twisting energy at each hinge.

        Args:
            twist_bends (Tuple[(3,); Array[(L,)]]):
                (_, _, Γ²) (output of ``twist_bend_angles()``).

        Returns:
            A two element tuple with twisting energy for each hinge
            (Array[(L,)]) and twist_bends. The energy density is in units of
            C / d.
        """
        if twist_bends is None:
            twist_bends = self.twist_bend_angles()
        energy_density = self.C / (2.0 * self.d) * twist_bends[2]
        return energy_density, twist_bends

    def twist_energy(self, *args, **kwargs):
        return self._total(self.twist_energy_density, *args, **kwargs)

    def unit_tangent_vectors(self):
        """Unit tangent vectors for each rod (Array[(L, 3)])."""
        return fast_calc.unit_tangent_vectors(self.euler)

    def tangent_vectors(self):
        """Tangent vectors for each rod (Array[(L, 3)]) scaled with rod length.

        Also see ``unit_tangent_vectors``.
        """
        return self.d * fast_calc.unit_tangent_vectors(self.euler)

    def position_vectors(self, tangents=None):
        """Position vectors for each rod (Array[(L, 3)]) scaled with rod length.

        Note:
            The "start" position is not included; it is implicitly set to 0.
        """
        if tangents is None:
            tangents = self.tangent_vectors()
        return np.cumsum(tangents, axis=0)

    def stretch_energy_density(self, force, tangents=None):
        u"""Computes stretching energy for all but the last rod.

        Computes -F·t_i/(k·T) for i in [0, ..., L-1], using
        k = 1.38E-2 pN·nm/K.

        Args:
            force (Array[(3,)]): Applied force in pN.
            tangents (Array[(L, 3)]): tangent vectors for each rod in nm.

        Returns:
            energy_density (Array[(L,)]) in units of kT.
            otherwise.
        """
        T = max(self.env.T, Environment.MIN_TEMP)
        prefactor = 1.0 / (1.38E-2 * T)
        if tangents is None:
            tangents = self.tangent_vectors()
        return -prefactor * np.einsum('i,ji->j', force, tangents)

    def stretch_energy(self, *args, **kwargs):
        return self._total(self.stretch_energy_density, *args, **kwargs)

    def total_energy_density(self, force, tangents=None):
        """Computes total energy for each rod.

        The energy has three pieces:

            E_tot = E_bend + E_twist + E_stretch

        Args:
            force (Array[(3,)]): applied force in pN.
            T (float): temperature in K.
            tangents (Array[(L, 3)]): tangent vectors for each rod in nm.

        Returns:
            energy_density (Array[(L,)]) in units of kT.

        Note:
            Caller should ensure that B/d and C/d are in units of kT.
        """
        energy_density, twist_bends = self.bend_energy_density()
        # discard the second element as twist_bends was computed already
        energy_density += self.twist_energy_density(twist_bends=twist_bends)[0]
        energy_density += self.stretch_energy_density(
            force, tangents=tangents)
        return energy_density

    def total_energy(self, *args, **kwargs):
        return self._total(self.total_energy_density, *args, **kwargs)
