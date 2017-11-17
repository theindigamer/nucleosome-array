import numpy as np
import fast_calc


class AngularDescription:
    """A barebones description of the angular properties of a strand.

    This class provides several basic functions to work with angular properties.
    Methods should be overriden for specific purposes, such as measuring
    intermediate results. For example, the Monte Carlo simulation overrides
    some functions to record timings for individual computations.
    """

    def __init__(self, L, B, C, strand_len, euler=None):
        """
        Args:
            L (int): number of rods
            B (float): bending modulus
            C (float): twisting modulus
            strand_len (float): total length of strand
            euler (Array[(L, 3)]): Euler angles for all the rods.
        """
        self.L = L
        self.B = B
        self.C = C
        self.strand_len = strand_len
        self.d = strand_len / L
        if euler is None:
            self.euler = np.zeros((self.L, 3))
        else:
            if np.shape(euler) != (L, 3):
                raise ValueError("Unexpected shape.")
            self.euler = euler

    def rotation_matrices(self):
        """Rotation matrices along the DNA string.

        Returns:
            Rotation matrices for each rod, in an array of shape (L, 3, 3).
        """
        return fast_calc.rotation_matrices(self.euler)

    def delta_matrices(self, Rs=None):
        u"""Returns Δ matrices describing bends/twists between consecutive rods.

        Args:
            Rs (Array[(L, 3, 3)]): Rotation matrices to use.

        Returns:
            Array of Δ_i = R_i^T . R_{i+1} in an array of shape (L - 1, 3, 3).
        """
        if Rs is None:
            Rs = self.rotation_matrices()
        R_i_transpose = np.swapaxes(Rs[:-1], 1, 2)
        R_i_plus_1 = Rs[1:]
        return R_i_transpose @ R_i_plus_1

    def twist_bend_angles(self, Deltas=None):
        u"""Computes the twist and bend angles.

        Args:
            Deltas (Array[(L-1, 3, 3)]): Delta matrices to use.

        Returns:
            (β², β², Γ²), each an array of shape (L-1,).
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
            twist_bends (Tuple[(3,); Array[(L-1,)]]):
                (β², _, _) (output of ``twist_bend_angles()``).

        Returns:
            A two element tuple with energy_density (Array[(L-1,)]) and
            twist_bends. The energy_density is in units of B / d.
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
            twist_bends (Tuple[(3,); Array[(L-1,)]]):
                (_, _, Γ²) (output of ``twist_bend_angles()``).

        Returns:
            A two element tuple with energy_density (Array[(L-1,)]) and
            twist_bends. The energy density is in units of C / d.
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
            The "start" coordinate is not included.
        """
        if tangents is None:
            tangents = self.tangent_vectors()
        return np.cumsum(tangents, axis=0)

    def stretch_energy_density(self, force, T, tangents=None):
        u"""Computes stretching energy for all but the last rod.

        Computes -F·t_i/(k·T) for i in [0, ..., L-1), using
        k = 1.38E-2 pN·nm/K.

        Args:
            force (float):
                z component of applied force in pN. Other components are
                assumed to be zero.
            T (float): temperature in K.
            tangents (Array[(L, 3)]):
                tangent vectors for each rod in nm.

        Returns:
            energy_density (Array[(L-1,)]) in units of kT.

        Note:
            The last rod's energy is not being accounted for as there will
            be a shape mismatch inside the total_energy_density calculation
            otherwise.
        """
        prefactor = 1.0 / (1.38E-2 * T)
        if tangents is None:
            tangents = self.tangent_vectors()
        return -force * prefactor * tangents[:-1, 2]

    def stretch_energy(self, *args, **kwargs):
        return self._total(self.stretch_energy_density, *args, **kwargs)

    def total_energy_density(self, force, T, tangents=None):
        """Computes total energy for each rod.

        The energy has three pieces:

            E_tot = E_bend + E_twist + E_stretch

        Args:
            force (float):
                z component of applied force in pN. Other components are
                assumed to be zero.
            T (float): temperature in K.
            tangents (Array[(L, 3)]): tangent vectors for each rod in nm.

        Returns:
            energy_density (Array[(L-1,)]) in units of kT.

        Note:
            Caller should ensure that B / d and C / d are in units of kT.
        """
        energy_density, twist_bends = self.bend_energy_density()
        # discard the second element as twist_bends was computed already
        energy_density += self.twist_energy_density(twist_bends=twist_bends)[0]
        energy_density += self.stretch_energy_density(
            force, T, tangents=tangents)
        return energy_density

    def total_energy(self, *args, **kwargs):
        return self._total(self.total_energy_density, *args, **kwargs)
