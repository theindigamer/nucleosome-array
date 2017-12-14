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

    def __init__(self, L, B, C, T, strand_len, euler=None, start=None, end=None):
        u"""Initialize the angular description of a strand.

        Args:
            L (int): number of rods
            B (float): bending modulus
            C (float): twisting modulus
            T (float): temperature in Kelvin
            strand_len (float): total length of strand
            euler (Array[(L, 3)]): Euler angles for all the rods.
            start (Array[(3,)]): Euler angles at the start of the (-1)-th rod.
            end (Array[(3,)]): Euler angles at the end of the final rod.

        Note:
            If B and C are specified in units of z·kT where T is the
            simulation temperature, and z is the unit used for the strand length
            (usually nm), the force, should be specified in pN.

            If B and C are specified in some other units, for example, if
            B/strand_len is in Joules, then you should write a custom
            stretch_energy_density function which appropriately adjusts the
            prefactor for force computation.

            The key point is to make sure that all the functions in the
            total_energy_density functions have the same units.
        """
        self.L = L
        self.B = B
        self.C = C
        self.env = Environment(T=T)
        self.strand_len = strand_len
        self.d = strand_len / L

        def check_shape_and_save(attr_name, x, default):
            nonlocal self
            if x is None:
                setattr(self, attr_name, default)
            else:
                x_shape = np.shape(x)
                def_shape = np.shape(default)
                if x_shape != def_shape:
                    raise ValueError("Unexpected shape. Supplied shape is {0},"
                                     " whereas shape {1} was expected.",
                                     x_shape, def_shape)
                else:
                    setattr(self, attr_name, x)

        check_shape_and_save("euler", euler, np.zeros((self.L, 3)))
        check_shape_and_save("start", start, np.zeros(3))
        check_shape_and_save("end", end, np.zeros(3))

    def rotation_matrices(self):
        """Rotation matrices along the DNA string.

        Returns:
            Rotation matrices for start + euler + end (L+2, 3, 3).
        """
        return fast_calc.rotation_matrices(self.start, self.euler, self.end)

    def delta_matrices(self, Rs=None):
        u"""Returns Δ matrices describing bends and twists for each hinge.

        Args:
            Rs (Array[(L+2, 3, 3)]): Rotation matrices to use.

        Returns:
            Array of Δ_i = R_i^T · R_{i+1} in an array of shape (L+1, 3, 3).
        """
        if Rs is None:
            Rs = self.rotation_matrices()
        R_i_transpose = np.swapaxes(Rs[:-1], 1, 2)
        R_i_plus_1 = Rs[1:]
        return R_i_transpose @ R_i_plus_1

    def twist_bend_angles(self, Deltas=None):
        u"""Computes the twist and bend angles.

        Args:
            Deltas (Array[(L+1, 3, 3)]): Delta matrices to use.

        Returns:
            (β², β², Γ²), each an array of shape (L+1,).
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
            twist_bends (Tuple[(3,); Array[(L+1,)]]):
                (β², _, _) (output of ``twist_bend_angles()``).

        Returns:
            A two element tuple with bending energy for each hinge
            (Array[(L+1,)]) and twist_bends. The energy_density is in units of
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
            twist_bends (Tuple[(3,); Array[(L+1,)]]):
                (_, _, Γ²) (output of ``twist_bend_angles()``).

        Returns:
            A two element tuple with twisting energy for each hinge
            (Array[(L+1,)]) and twist_bends. The energy density is in units of
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
        u"""Computes stretching energy for all L rods.

        Computes -F·t_i/(k·T) for i in [0, ..., L-1], using
        k = 1.38E-2 pN·nm/K.

        Args:
            force (Array[(3,)]): Applied force in pN.
            tangents (Array[(L, 3)]): tangent vectors for each rod in nm.

        Returns:
            energy_density (Array[(L,)]) in units of kT.
        """
        T = max(self.env.T, Environment.MIN_TEMP)
        prefactor = 1.0 / (1.38E-2 * T)
        if tangents is None:
            tangents = self.tangent_vectors()
        return prefactor * np.einsum('i,ji->j', -force, tangents)

    def stretch_energy(self, *args, **kwargs):
        return self._total(self.stretch_energy_density, *args, **kwargs)

    def all_energy_densities(self, force=None, tangents=None,
                             include=(True, True, True)):
        """Computes all the energy densities and returns them separately.

        Args:
            force (Array[(3,)]): applied force (vector) in pN.
            tangents (Array[(L, 3)]): tangent vectors for each rod in nm.
            include (List[(3,)]): If bend/twist/stretch energy density should
            be computed.

        Returns:
            Energy densities in a tuple (bend, twist, stretch).
            If include[i] is set to False, the i-th tuple element is None.
            See the corresponding energy density functions for shapes.

        Note:
            You must supply a force value if the last element of include is
            set to True; it cannot be automatically inferred.
        """
        bend_include, twist_include, stretch_include = include
        twist_bends = None
        if bend_include:
            bend_ed, twist_bends = self.bend_energy_density()
        else:
            bend_ed = None
        if twist_include:
            twist_ed, _ = self.twist_energy_density(twist_bends=twist_bends)
        else:
            twist_ed = None
        if stretch_include:
            stretch_ed = self.stretch_energy_density(force, tangents=tangents)
        else:
            stretch_ed = None
        return (bend_ed, twist_ed, stretch_ed)

    def total_energy_density(self, force, tangents=None):
        """Computes total energy for each rod.

        The energy has three pieces:

            E_tot = E_bend + E_twist + E_stretch

        This function is only provided as sugar for total_energy.
        DO NOT CALL IT DIRECTLY for dynamics,
        as there is a mix of indexing - rods and hinges -
        due to unequal sizes of the stretch and bend/twist energies densities.
        Use the individual energy density functions, or ``all_energy_densities``
        instead.

        Args:
            force (Array[(3,)]): applied force in pN.
            tangents (Array[(L, 3)]): tangent vectors for each rod in nm.

        Returns:
            energy_density (Array[(L+1,)]) in units of kT.

        Note:
            Caller should ensure that B/d and C/d are in units of kT.
        """
        energy_density, twist_bends = self.bend_energy_density()
        # discard the second element as twist_bends was computed already
        energy_density += self.twist_energy_density(twist_bends=twist_bends)[0]
        # skip last element due to extra length
        energy_density[:-1] += self.stretch_energy_density(
            force, tangents=tangents)
        return energy_density

    def total_energy(self, *args, **kwargs):
        return self._total(self.total_energy_density, *args, **kwargs)
