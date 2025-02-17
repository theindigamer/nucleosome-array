import numpy as np
import time
import fast_calc
import xarray as xr

from sim_utils import AngularDescription, Environment

class Simulation:
    """Simulation parameters that can be varied."""
    DEFAULT_KICK_SIZE = 0.1
    ntimers = 10
    timer_descr = {
        0 : "Half of inner loop in metropolis_update",
        1 : "Half of half of inner loop in metropolis_update",
        2 : "Calls to bend_energy_density",
        3 : "Calls to twist_energy_density",
        4 : "Calculating Rs in delta_matrices via rotation_matrices",
        5 : "Calculating a in delta_matrices",
        6 : "Calculating b in delta_matrices",
        7 : "Calculating deltas in delta_matrices",
        8 : "Total time",
    }

    # TODO: fix defaultKickSize
    # Afterwards, kickSize should not be specified as an argument.
    # It should automatically be determined from the temperature.
    def __init__(self, kickSize=DEFAULT_KICK_SIZE):
        self.kickSize = kickSize
        self.timers = np.zeros(Simulation.ntimers)
        # Describes whether computations use approximate forms for
        # β^2 and Γ^2 or β and Γ. The former should be more accurate.
        self.squared = True


class Evolution:
    """Records simulation parameters and the evolution of a DNA strand."""

    def __init__(self, dna, nsteps, force, twists=None, initial=False):
        """Initialize an evolution object.

        ``twists`` is either a nonempty ``numpy`` array (twisting) or ``None``
        (relaxation).

        ``nsteps`` represents successive number of MC steps.

        If ``twists`` is ``None``, ``nsteps`` is interpreted to represent the
        entire simulation. So [10, 10, 5] means that the simulation has 25 steps
        total.
        If ``twists`` is a numpy array, ``nsteps`` is interpreted to be the
        number of steps between consecutive twists.
        So ``twists = [0, pi/2]`` and ``nsteps=[10, 10, 5]`` implies there are
        50 MC steps in total, split up as 10, 10, 5, 10, 10, 5.
        """
        self.initial = initial
        self.tstep_i = 0
        if twists is None:
            tsteps = np.cumsum(nsteps)
            end = None
        else:
            tmp = np.cumsum(nsteps)
            tsteps = np.concatenate([tmp + tmp[-1] * i
                                     for i in range(len(twists))])
            end = np.array([[tw for _ in nsteps] for tw in twists]).flatten()
        if initial:
            tsteps = np.insert(tsteps, 0, 0)
        self.data = dna.constants()
        self.data.update({
            # TODO: Check if None can be saved in a netcdf file.
            "twists": twists,
            "tsteps": tsteps,
            "angles": np.empty((tsteps.size, dna.L, 3)),
            "end"   : end,
            "extension": np.empty((tsteps.size, 3)),
            "energy": np.empty(tsteps.size),
            "acceptance": np.empty((tsteps.size, 3)),
        })
        if initial:
            (self.save_energy(dna.total_energy(force))
             .save_extension(dna.total_extension())
             .save_acceptance(np.array([0.5, 0.5, 0.5]))
             .save_angles(dna))

    def update(self, dictionary):
        self.data.update(dictionary)
        return self

    def save_angles(self, dna):
        self.data["angles"][self.tstep_i] = dna.euler
        self.tstep_i += 1
        return self

    def save_extension(self, extension):
        self.data["extension"][self.tstep_i] = extension
        return self

    def save_energy(self, energy):
        self.data["energy"][self.tstep_i] = energy
        return self

    def save_timing(self, timing):
        self.data.update({"timing": timing})
        return self

    def save_acceptance(self, acceptance):
        self.data["acceptance"][self.tstep_i] = acceptance
        return self

    def to_dict(self):
        return self.data

    def to_dataset(self):
        data = self.data
        timing_keys, timing = list(zip(*data["timing"].items()))
        timing_keys = list(timing_keys)
        timing = list(timing)
        # variables that may be involved in plots
        data_vars = {
            "twists": data["twists"],
            "angles": (["tsteps", "n", "angle_str"], data["angles"]),
            "end": (["tsteps", "angle_str"], data["end"]),
            "extension": (["tsteps", "axis"], data["extension"]),
            "energy": (["tsteps"], data["energy"]),
            "acceptance": (["tsteps", "angle_str"], data["acceptance"]),
            "timing": (["timing_keys"], timing),
            "force": data["force"],
            "kickSize": data["kickSize"],
            "Pinv": data["Pinv"],
        }
        coords = {
            "tsteps": data["tsteps"],
            "angle_str": ["phi", "theta", "psi"],
            "axis": ['x', 'y', 'z'],
            "timing_keys": timing_keys,
        }
        attrs = {
            "initial": int(self.initial),
        }
        if "nucleosome" in data.keys():
            attrs.update({"nucleosome": data["nucleosome"]})
        saved_keys = (list(data_vars.keys()) + list(coords.keys())
                      + list(attrs.keys()))
        for (k, v) in data.items():
            if not k in saved_keys:
                attrs.update({k: v})
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

class NakedDNA(AngularDescription):
    """A strand of DNA without any nucleosomes.

    Euler angles are ordered as phi, theta and psi.
    Values of B and C are from [DS, Table I].
    [DS, Eqn. (30)] describes the microscopic parameter Bm computed from B.
    """

    DEFAULT_TWIST_STEP = np.pi/4

    def __init__(self, L=740, B=43.0, C=89.0, strand_len=740.0,
                 T=Environment.ROOM_TEMP,
                 kickSize=Simulation.DEFAULT_KICK_SIZE):
        """Initialize a DNA strand.

        Note:
            B and C are specified in units of nm·kT, where T is the temperature
            specified (not necessarily room temperature). If you're using a
            small temperature for testing stiffness, make sure that you increase
            the B and C values appropriately.
        """
        super().__init__(
            L, B, C, T, strand_len, euler=np.zeros((L, 3)), end=np.zeros(3))
        if T < 273 and B == 43.0 and C == 89.0:
            print("WARNING: You are using a low temperature but haven't changed"
                  " either B or C.")
        self.Pinv = 0 # no intrinsic disorder
        self.Bm = B   # dummy value, kept for consistency with disordered DNA.
        self.sim = Simulation(kickSize=kickSize)
        self.oddMask = np.array([i % 2 == 1 for i in range(self.L - 1)])
        self.oddMask.setflags(write=False)
        self.evenMask = np.logical_not(self.oddMask)
        self.evenMask.setflags(write=False)

    # FIXME: Units of different variables should also be saved.
    def constants(self):
        return {
            "temperature": self.env.T,
            "B": self.B,
            "Pinv": self.Pinv,
            "Bm": self.Bm,
            "C": self.C,
            "rodCount": self.L,
            "strandLength": self.strand_len,
            "rodLength": self.d,
            "kickSize": self.sim.kickSize,
        }

    def delta_matrices(self, Rs=None):
        """Returns Δ matrices describing bends/twists between consecutive rods."""
        timers = self.sim.timers
        start = time.clock()
        if Rs is None:
            Rs = self.rotation_matrices()
        timers[4] += time.clock() - start

        start = time.clock()
        R_i_transpose = np.swapaxes(Rs[:-1], 1, 2)
        timers[5] += time.clock() - start

        start = time.clock()
        R_i_plus_1 = Rs[1:]
        timers[6] += time.clock() - start

        start = time.clock()
        deltas = R_i_transpose @ R_i_plus_1
        timers[7] += time.clock() - start
        return deltas

    def twist_bend_angles(self, Deltas=None):
        """Computes the twist and bend angles.

        See ``fast_calc.twist_bend_angles`` for details. The squared argument
        is set from an attribute.
        """
        if Deltas is None:
            Deltas = self.delta_matrices()
        return fast_calc.twist_bend_angles(Deltas, self.sim.squared)

    def bend_energy_density(self, twist_bends=None):
        """Computes bending energy at each hinge."""
        if twist_bends is None:
            twist_bends = self.twist_bend_angles()
        if self.sim.squared:
            energy_density = self.B / (2.0 * self.d) * twist_bends[0]
        else:
            energy_density = (self.B / (2.0 * self.d)
                              * (twist_bends[0]**2 + twist_bends[1]**2))
        return energy_density, twist_bends

    def twist_energy_density(self, twist_bends=None):
        """Computes twisting energy for each rod."""
        if twist_bends is None:
            twist_bends = self.twist_bend_angles()
        if self.sim.squared:
            energy_density = self.C / (2.0 * self.d) * twist_bends[2]
        else:
            energy_density = self.C / (2.0 * self.d) * twist_bends[2]**2
        return energy_density, twist_bends

    def total_energy_density(self, force):
        """Computes total energy for each rod.

        WARNING: The force is a scalar here, unlike the base class.

        Room for optimization - avoid computing x and y components of tangent
        vectors and corresponding energy terms by specializing the
        implementation of stretch_energy_density.
        """
        timers = self.sim.timers

        start = time.clock()
        energy_density, twist_bends = self.bend_energy_density()
        timers[2] += time.clock() - start

        start = time.clock()
        energy_density += self.twist_energy_density(twist_bends=twist_bends)[0]
        timers[3] += time.clock() - start

        energy_density += self.stretch_energy_density(np.array([0., 0., force]))
        return energy_density

    def metropolis_update_seq(self, force, E0, acceptance=False):
        u"""Updates Euler angles sequentially using the Metropolis algorithm.

        The iteration order is φ_even, φ_odd, θ_even, θ_odd, ψ_even, ψ_odd.

        Returns:
            If acceptance is true, a tuple with the energy density and the acceptance
            ratios. Otherwise, it return the energy density.

            The energy density is of shape (L,).

        Note:
            It is not obvious that this scheme satisfies detailed balance, so we
            should use metropolis_update instead.
        """
        sigma = self.sim.kickSize
        timers = self.sim.timers

        moves = np.random.normal(loc=0.0, scale=sigma, size=(self.L - 1, 3))
        moves[np.abs(moves) >= 5.0*sigma] = 0.
        if acceptance:
            accepted_frac = np.zeros(3)

        for i in range(3):
            start = time.clock()
            # Move even rods first.
            # oddMask is False, True, False, ... and we start from euler[1]
            # so only euler[2], euler[4], ... are changed
            self.euler[1:, i] += moves[:, i] * self.oddMask
            Ef = self.total_energy_density(force)
            # Next step is explained in metropolis_update
            deltaE = (Ef - E0)[:-1] + (Ef - E0)[1:]
            timers[1] += time.clock() - start

            # true ⇔ move is rejected, first reject all moves of even rods
            reject = self.oddMask.copy()
            # Now reject == [False, True, False, True, ...]
            # Some True values should be made False according to the Metropolis
            # algorithm. We only need to examine the _odd_ indices of reject.
            if self.env.T <= Environment.MIN_TEMP:
                reject[deltaE <= 0.] = False
            else:
                fast_calc.metropolis(reject, deltaE, even=False)
            self.euler[1:, i] -= moves[:, i] * reject
            E0 = self.total_energy_density(force)
            if acceptance:
                accepted_frac[i] += 0.5 - np.count_nonzero(reject)/reject.size
            timers[0] += time.clock() - start

            # Move odd rods now.
            self.euler[1:, i] += moves[:, i] * self.evenMask
            Ef = self.total_energy_density(force)
            deltaE = (Ef - E0)[:-1] + (Ef - E0)[1:]

            reject = self.evenMask.copy()
            if self.env.T <= Environment.MIN_TEMP:
                reject[deltaE <= 0.] = False
            else:
                fast_calc.metropolis(reject, deltaE, even=True)
            self.euler[1:, i] -= moves[:, i] * reject
            E0 = self.total_energy_density(force)
            if acceptance:
                accepted_frac[i] += 0.5 - np.count_nonzero(reject)/reject.size

        if acceptance:
            return E0, accepted_frac
        return E0

    def metropolis_update(self, force, E0, acceptance=False):
        u"""Updates Euler angles using the Metropolis algorithm.

        In contrast to metropolis_update_seq, the iteration order is random,
        i.e. we randomly pick one of even/odd and one of φ/θ/ψ and give kicks to
        the corresponding angles.

        Args:
            force (float): z component of force. Other components assumed 0.
            E0 (Array[(L,)]): Initial energy density. This argument is mutated.
            acceptance (bool): Whether acceptance ratios should be recorded.

        Returns:
            If acceptance is true, a tuple with the energy density and the acceptance
            ratios. Otherwise, it return the energy density.

            The energy density is of shape (L,).

            WARNING: Currently the acceptance values returned are incorrect.
        """
        sigma = self.sim.kickSize
        timers = self.sim.timers

        moves = np.random.normal(loc=0.0, scale=sigma, size=(self.L - 1, 3))
        moves[np.abs(moves) >= 5.0 * sigma] = 0.
        if acceptance:
            accepted_frac = np.zeros(3)

        # TODO: fix acceptance computation
        def update_rods(i, even=True):
            mask = self.oddMask if even else self.evenMask
            # opposite as "even=True" means even rods ought to be moved
            # but the moves are applied starting from rod #1.
            self.euler[1:, i] += moves[:, i] * mask
            Ef = self.total_energy_density(force)
            nonlocal E0
            deltaE = (Ef - E0)[:-1] + (Ef - E0)[1:]
            # If only rod i is moved (i > 0), 5 terms will change in the energy
            # computation: 4 due to hinges i-1 and i, and 1 due to stretching.
            #
            #     |         | i-1       | i       |
            #     |---------+-----------+---------|
            #     | twist   | changed   | changed |
            #     | bend    | changed   | changed |
            #     | stretch | unchanged | changed |
            #
            # deltaE[i-1] = Ef[i] - E0[i] + Ef[i-1] - E0[i-1]
            # will solely affect reject[i-1] and finally euler[i].

            reject = mask.copy()
            if self.env.T <= Environment.MIN_TEMP:
                reject[deltaE <= 0.] = False
            else:
                fast_calc.metropolis(reject, deltaE, even=(not even))
            self.euler[1:, i] -= moves[:, i] * reject
            E0 = self.total_energy_density(force)
            if acceptance:
                accepted_frac[i] += 0.5 - np.count_nonzero(reject)/reject.size

        # We have duplication to mimic the time scale in metropolis_update_seq
        # instead of having an overall factor of two.
        for i in np.random.permutation(3):
            start = time.clock()
            parity = np.random.rand() > 0.5
            update_rods(i, even=parity)
            timers[0] += time.clock() - start
            update_rods(i, even=(not parity))

        if acceptance:
            return E0, accepted_frac
        return E0

    def total_extension(self):
        u"""Returns [Δx, Δy, Δz] given as r_bead - r_bottom."""
        return self.d * np.sum(self.unit_tangent_vectors(), axis=0)

    def mc_relaxation(self, force, E0, mcSteps, record_final_only=True):
        """Monte Carlo relaxation using Metropolis algorithm."""
        if record_final_only:
            for _ in range(mcSteps - 1):
                E0 = self.metropolis_update(force, E0, acceptance=False)
            E0, acc = self.metropolis_update(force, E0, acceptance=True)
            return E0, self.total_extension(), acc
        else:
            energies = np.empty((mcSteps, self.L))
            extensions = np.empty((mcSteps, 3))
            acceptance_ratios = np.empty((mcSteps, 3))
            for i in range(mcSteps):
                E0, acceptance_ratios[i] = (
                    self.metropolis_update(force, E0, acceptance=True)
                )
                energies[i] = E0
                extensions[i] = self.total_extension()
            return energies, extensions, acceptance_ratios

    def torsion_protocol(self, force=1.96, E0=None, twists=2*np.pi,
                         mcSteps=100, nsamples=1, includeStart=False):
        """Simulate a torsion protocol defined by twists.

        The twists argument can be described in several ways:
        1. ``twists=stop`` will twist from 0 to stop (inclusive) with an
        automatically chosen step size.
        2. ``twists=(stop, step)`` will twist from 0 to stop (inclusive) with
        step size ``step``.
        3. ``twists=(start, stop, step)`` is self-explanatory.
        4. If ``twists`` is a list or numpy array, it will be used directly.

        **Warning**: Twisting is *absolute* not relative.
        """
        start = time.clock()
        timers = self.sim.timers
        nsteps = fast_calc.partition(nsamples, mcSteps)
        tmp_twists = fast_calc.twist_steps(self.DEFAULT_TWIST_STEP, twists)
        evol = Evolution(self, nsteps, force, twists=tmp_twists, initial=includeStart)
        evol.update({"force": force, "mcSteps": mcSteps})
        E0 = self.total_energy_density(force)
        for x in tmp_twists:
            self.end[2] = x
            for nstep in nsteps:
                E0, extension, acceptance = self.mc_relaxation(
                    force, E0, nstep)
                (evol.save_energy(np.sum(E0))
                 .save_extension(extension)
                 .save_acceptance(acceptance)
                 .save_angles(self))
        timers[8] += time.clock() - start
        timing = {s : timers[i] for (i, s) in Simulation.timer_descr.items()}
        evol.save_timing(timing)
        return evol.to_dataset()

    def relaxation_protocol(self, force=1.96, E0=None,
                            mcSteps=1000, nsamples=4, includeStart=False):
        """Simulate a relaxation for an initial twist profile.

        The DNA should be specified with the required profile already applied.
        """
        start = time.clock()
        timers = self.sim.timers
        nsteps = fast_calc.partition(nsamples, mcSteps)
        evol = Evolution(self, nsteps, force, initial=includeStart)
        evol.update({"force": force, "mcSteps": mcSteps})
        E0 = self.total_energy_density(force)
        for nstep in nsteps:
            E0, extension, acceptance = self.mc_relaxation(
                force, E0, nstep)
            (evol.save_energy(np.sum(E0))
             .save_extension(extension)
             .save_acceptance(acceptance)
             .save_angles(self))
        timers[8] += time.clock() - start
        timing = {s : timers[i] for (i, s) in Simulation.timer_descr.items()}
        evol.save_timing(timing)
        return evol.to_dataset()


class DisorderedNakedDNA(NakedDNA):

    # [DS, Appendix 1] describes equations related to disorder.
    def __init__(self, Pinv=1./300, **kwargs):
        # The 1/300 value has no particular significance.
        # [DS] considers values from ~1/1000 to ~1/100
        NakedDNA.__init__(self, **kwargs)
        self.Pinv = Pinv # inverse of P value in 1/nm
        self.Bm = self.B / (1 - self.B * Pinv)
        sigma_b = (Pinv * self.d) ** 0.5
        # xi represents [dot(xi_m, n_1), dot(xi_m, n_2)], see [DS, Eqn. (E1)]
        xi = np.random.randn(2, self.L - 1)
        self.bend_zeros = sigma_b * xi
        self.sim.squared = False

    def bend_energy_density(self, angles=None):
        if angles is None:
            angles = self.twist_bend_angles()
        energy_density = self.Bm / (2.0 * self.d) * (
            (angles[0] - self.bend_zeros[0]) ** 2
            + (angles[1] - self.bend_zeros[1]) ** 2
        )
        return energy_density, angles


class NucleosomeArray(NakedDNA):
    def __init__(self, nucPos=np.array([]), strandLength=740, **dna_kwargs):
        """Initialize the nucleosome array in the default 'vertical' configuration.

        By vertical configuration, we mean that all rods are vertical except
        the ones immediately 'coming out' of a nucleosome.

        A nucleosome at position ``n`` is between rods of indices ``n-1`` and
        ``n`` (zero-based).
        ``strandLength`` should be specified in nm. It should only include the
        length of the DNA between the nucleosome core(s) plus the spacer at the
        ends. It must **not** include the length of the DNA wrapped around the
        nucleosome core(s).
        ``dna_kwargs`` should be specified according to ``NakedDNA.__init__``'s
        kwargs.

        NOTE: You may want to use ``create`` instead of calling the constructor
        directly.
        """
        NakedDNA.__init__(self, **dna_kwargs)
        self.strandLength = strandLength
        self.d = strandLength/dna_kwargs["L"]
        self.nuc = nucPos
        self.euler[self.nuc] = fast_calc.exitAngles([0., 0., 0.])

    def constants(self):
        d = NakedDNA.constants(self)
        d.update({"nucleosome": self.nuc})
        return d

    @staticmethod
    def create(nucArrayType, nucleosomeCount=36, basePairsPerRod=10,
               linker=60, spacer=600, kickSize=Simulation.DEFAULT_KICK_SIZE):
        """Initializes a nucleosome array in one of the predefined styles.

        ``nucArrayType`` takes the values:
        * 'standard' -> nucleosomes arranged roughly vertically
        * 'relaxed'  -> initial twists and bends between rods are zero
        ``linker`` and ``spacer`` are specified in base pairs.
        """
        if linker % basePairsPerRod != 0:
            raise ValueError(
                "Number of rods in linker DNA should be an integer.\n"
                "linker value should be divisible by basePairsPerRod."
            )
        if spacer % basePairsPerRod != 0:
            raise ValueError(
                "Number of rods in spacers should be an integer.\n"
                "spacer value should be divisible by basePairsPerRod."
            )
        # 60 bp linker between cores
        #           |-|           600 bp spacers on either side
        # ~~~~~~~~~~O~O~O...O~O~O~~~~~~~~~~
        basePairArray = [spacer] + ([linker] * (nucleosomeCount - 1)) + [spacer]
        basePairLength = 0.34 # in nm
        strandLength = float(np.sum(basePairArray) * basePairLength)
        numRods = np.array(basePairArray) // basePairsPerRod
        L = int(np.sum(numRods))
        nucPos = np.cumsum(numRods)[:-1]
        dna = NucleosomeArray(L=L, nucPos=nucPos,
                              strandLength=strandLength, kickSize=kickSize)

        if nucArrayType == "standard":
            pass
        elif nucArrayType == "relaxed":
            prev = np.array([0., 0., 0.])
            for i in range(L):
                if nucPos.size != 0 and i == nucPos[0]:
                    tmp = fast_calc.exitAngles(prev)
                    dna.euler[i] = np.copy(tmp)
                    prev = tmp
                    nucPos = nucPos[1:]
                else:
                    dna.euler[i] = np.copy(prev)
        else:
            raise ValueError("nucArrayType should be either 'standard' or 'relaxed'.")

        return dna

    def delta_matrices(self, Rs=None):
        """Returns Δ matrices describing bends/twists between consecutive rods."""
        start = time.clock()
        timers = self.sim.timers
        if Rs is None:
            Rs = self.rotation_matrices()
        timers[4] += time.clock() - start

        start = time.clock()
        a = np.swapaxes( Rs[:-1], 1, 2 )
        timers[5] += time.clock() - start

        start = time.clock()
        b = Rs[1:]
        timers[6] += time.clock() - start

        start = time.clock()
        deltas = a @ b
        fast_calc.calc_deltas(deltas, self.nuc, Rs)
        # TODO: check if avoiding recalculation for nucleosome ends is faster
        timers[7] += time.clock() - start

        return deltas

    def partialRotationMatrices(self, inds):
        """Returns rotation matrices only for specific rods"""
        phi = self.euler[inds, 0]
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        theta = self.euler[inds, 1]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        psi = self.euler[inds, 2]
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        R = np.zeros((len(inds), 3, 3))
        R[:, 0, 0] = cos_phi * cos_psi - cos_theta * sin_phi * sin_psi
        R[:, 1, 0] = -cos_psi * sin_phi - cos_theta * cos_phi * sin_psi
        R[:, 2, 0] = sin_theta * sin_psi
        R[:, 0, 1] = cos_phi * sin_psi + cos_theta * cos_psi * sin_phi
        R[:, 1, 1] = -sin_phi * sin_psi + cos_theta * cos_phi * cos_psi
        R[:, 2, 1] = -cos_psi * sin_theta
        R[:, 0, 2] = sin_theta * sin_phi
        R[:, 1, 2] = cos_phi * sin_theta
        R[:, 2, 2] = cos_theta
        return R

    def anglesForDummyRods(self):
        return np.array([fast_calc.exitAngles(self.euler[n-1]) for n in self.nuc])

    def relaxation_protocol(self, force=1.96, E0=None,
                            mcSteps=1000, nsamples=4, includeStart=False,
                            includeDummyRods=False):
        """Simulate a relaxation for an initial twist profile.

        The DNA should be specified with the required profile already applied.
        """
        start = time.clock()
        timers = self.sim.timers
        nsteps = fast_calc.partition(nsamples, mcSteps)
        dummyRodAngles = []
        evol = Evolution(self, nsteps, force, initial=includeStart)
        if includeStart and includeDummyRods:
            dummyRodAngles.append(self.anglesForDummyRods())
        E0 = self.total_energy_density(force)
        for nstep in nsteps:
            E0, extension, acceptance = self.mc_relaxation(force, E0, nstep)
            (evol.save_energy(np.sum(E0))
             .save_extension(extension)
             .save_acceptance(acceptance)
             .save_angles(self))
            if includeDummyRods:
                dummyRodAngles.append(self.anglesForDummyRods())
        timers[8] += time.clock() - start
        timing = {s : timers[i] for (i, s) in Simulation.timer_descr.items()}
        evol.save_timing(timing)
        if dummyRodAngles:
            evol.update({"dummyRodAngles": np.array(dummyRodAngles)})
        return evol.to_dataset()
