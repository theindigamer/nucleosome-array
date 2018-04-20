import numpy as np
import time
import fast_calc
import xarray as xr
import subject
import strands

from strands import AngularDescription, Environment, QuaternionDescription

class MonteCarloSimulation(Subject):
    __slots__ = (
        "strand",
        "sim", # simulation parameters which may affect the accuracy of results
        "env", # physical environmental parameters such as force etc.
        "obsCounter",
        "oddMask",
        "evenMask",
    )
    def __init__(self, strand, env):
        super().__init__()
        self.strand = strand
        self.env = env
        self.obsCounter = 0
        self.oddMask = np.array([i % 2 == 1 for i in range(strand.mask_length())])
        self.oddMask.setflags(write=False)
        self.evenMask = np.logical_not(self.oddMask)
        self.evenMask.setflags(write=False)

    def metropolis_step(self):
        """"""
        moves = self.sim.kickSize * self.strand.random_unit_moves()

        def per_rod_energy(E):
            bend_E, twist_E, stretch_E = E
            return bend_E[:-1] + bend_E[1:] + twist_E[:-1] + twist_E[1:] + stretch_E

        def update_rods(even=True):
            mask = self.evenMask if even else self.oddMask
            nonlocal E0
            Ei = per_rod_energy(E0)
            self.quats += mask[:, np.newaxis] * moves
            self.quats /= np.linalg.norm(self.quats, axis=1)[:, np.newaxis]
            Ef = per_rod_energy(self.all_energy_densities(force=force))
            deltaE = Ef - Ei
            reject = mask.copy()
            if self.env.T <= Environment.MIN_TEMP:
                reject[deltaE <= 0.] = False
            else:
                fast_calc.metropolis(reject, deltaE, even=even)
            self.quats -= reject[:, np.newaxis] * moves
            self.quats /= np.linalg.norm(self.quats, axis=1)[:, np.newaxis]
            E0 = self.all_energy_densities(force=force)
            if acceptance:
                accepted_frac[0] += 0.5 - np.count_nonzero(reject)/reject.size

        parity = np.random.rand() > 0.5
        update_rods(even=parity)
        update_rods(even=(not parity))
        if acceptance:
            return E0, accepted_frac
        return E0


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

    RUN_DEPENDENT_PARAMS = [
        "angles", "start", "end", "extension", "timing", "acceptance",
        "energy", "bendEnergy", "twistEnergy", "stretchEnergy"
    ]

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
            "twists": twists,
            "tsteps": tsteps,
            "quats" : np.empty((tsteps.size, dna.L, 4)),
            "startQ": np.empty((tsteps.size, 4)),
            "endQ"  : np.empty((tsteps.size, 4)),
            "angles": np.empty((tsteps.size, dna.L, 3)),
            "start" : np.empty((tsteps.size, 3)),
            "end"   : np.empty((tsteps.size, 3)),
            "extension": np.empty((tsteps.size, 3)),
            "acceptance": np.empty((tsteps.size, 3)),
            "energy": np.empty(tsteps.size),
            "bendEnergy": np.empty(tsteps.size),
            "twistEnergy": np.empty(tsteps.size),
            "stretchEnergy": np.empty(tsteps.size),
            "bendEnergyDensity": np.empty((tsteps.size, dna.L + 1)),
            "twistEnergyDensity": np.empty((tsteps.size, dna.L + 1)),
            "stretchEnergyDensity": np.empty((tsteps.size, dna.L)),
        })
        if initial:
            # energies = [np.sum(x) for x in dna.all_energy_densities(force)]
            (self.save_energy(dna.all_energy_densities(force))
             .save_extension(dna.total_extension())
             .save_acceptance(np.array([0.5, 0.5, 0.5]))
             .save_angles(dna))

    def update(self, dictionary):
        self.data.update(dictionary)
        return self

    def save_angles(self, dna):
        self.data["quats"][self.tstep_i] = dna.quats
        self.data["startQ"][self.tstep_i] = dna.start_quat
        self.data["endQ"][self.tstep_i] = dna.end_quat
        self.data["angles"][self.tstep_i] = dna.euler
        self.data["start"][self.tstep_i] = dna.start
        self.data["end"][self.tstep_i] = dna.end
        self.tstep_i += 1
        return self

    def save_extension(self, extension):
        self.data["extension"][self.tstep_i] = extension
        return self

    def save_energy(self, energy):
        self.data["bendEnergyDensity"][self.tstep_i] = energy[0]
        self.data["twistEnergyDensity"][self.tstep_i] = energy[1]
        self.data["stretchEnergyDensity"][self.tstep_i] = energy[2]
        self.data["bendEnergy"][self.tstep_i] = np.sum(energy[0])
        self.data["twistEnergy"][self.tstep_i] = np.sum(energy[1])
        self.data["stretchEnergy"][self.tstep_i] = np.sum(energy[2])
        self.data["energy"][self.tstep_i] = (
            self.data["bendEnergy"][self.tstep_i] +
            self.data["twistEnergy"][self.tstep_i] +
            self.data["stretchEnergy"][self.tstep_i]
        )
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
        # FIXME: None and boolean values cannot be saved in netcdf format.
        data = self.data
        timing_keys, timing = list(zip(*data["timing"].items()))
        timing_keys = list(timing_keys)
        timing = list(timing)
        data_vars = {"timing": (["timing_keys"], timing)}
        # variables that may be involved in plots
        indexed_by = {
            "twists": None,
            "quats":  ["tsteps", "n", "q"],
            "startQ": ["tsteps", "q"],
            "endQ":   ["tsteps", "q"],
            # "angles": ["tsteps", "n", "angle_str"],
            "start":  ["tsteps", "angle_str"],
            "end":    ["tsteps", "angle_str"],
            "extension":  ["tsteps", "axis"],
            "acceptance": ["tsteps", "angle_str"],
            "energy":        ["tsteps"],
            "bendEnergy":    ["tsteps"],
            "twistEnergy":   ["tsteps"],
            "stretchEnergy": ["tsteps"],
            "bendEnergyDensity":    ["tsteps", "m"],
            "twistEnergyDensity":   ["tsteps", "m"],
            "stretchEnergyDensity": ["tsteps", "n"],
            "force":    None,
            "kickSize": None,
            "Pinv":     None,
        }
        exclude_keys = {
            "angles",
            # "bendEnergyDensity",
            # "twistEnergyDensity",
            # "stretchEnergyDensity",
        }
        for (phys_param, dims) in indexed_by.items():
            if dims is None:
                data_vars[phys_param] = data[phys_param]
            else:
                data_vars[phys_param] = (dims, data[phys_param])
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
            if (k not in saved_keys) and (k not in exclude_keys):
                attrs.update({k: v})
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

class NakedDNA(QuaternionDescription):
    """A strand of DNA without any nucleosomes.

    Euler angles are ordered as phi, theta and psi.
    Values of B and C are from [DS, Table I].
    [DS, Eqn. (30)] describes the microscopic parameter Bm computed from B.
    """

    DEFAULT_TWIST_STEP = np.pi/4
    B_ROOM_TEMP = 43.0
    C_ROOM_TEMP = 89.0

    def __init__(self, L=740, B=B_ROOM_TEMP, C=C_ROOM_TEMP, strand_len=740.0,
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
            L=L, B=B, C=C, T=T, strand_len=strand_len,
            euler=np.zeros((L, 3)), end=np.zeros(3)
        )
        if T < 273 and (B == B_ROOM_TEMP or C == C_ROOM_TEMP):
            print("WARNING: You are using a low temperature but haven't changed"
                  " both B and C.")
        self.Pinv = 0 # no intrinsic disorder
        self.Bm = B   # dummy value, kept for consistency with disordered DNA.
        self.sim = Simulation(kickSize=kickSize)

        self.oddMask = np.array([i % 2 == 1 for i in range(self.L - 1)])
        self.oddMask.setflags(write=False)
        self.evenMask = np.logical_not(self.oddMask)
        self.evenMask.setflags(write=False)

        self.oddMaskL = np.array([i % 2 == 1 for i in range(self.L)])
        self.oddMaskL.setflags(write=False)
        self.evenMaskL = np.logical_not(self.oddMaskL)
        self.evenMaskL.setflags(write=False)

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
dd
dd
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

    def metropolis_update_quat(self, force, E0, acceptance=False):
        """

        WARNING: The boundary conditions are different here --
        all the rods are free to move.
        """
        if type(self.sim.kickSize) is float:
            scale_factor = self.sim.kickSize
        else:
            scale_factor = max(self.sim.kickSize)
        timers = self.sim.timers

        moves = scale_factor * fast_calc.random_unit_quaternions(self.L)
        if acceptance:
            accepted_frac = np.zeros(3)

        def per_rod_energy(E):
            bend_E, twist_E, stretch_E = E
            return bend_E[:-1] + bend_E[1:] + twist_E[:-1] + twist_E[1:] + stretch_E

        def update_rods(even=True):
            mask = self.evenMaskL if even else self.oddMaskL
            nonlocal E0
            Ei = per_rod_energy(E0)
            self.quats += mask[:, np.newaxis] * moves
            self.quats /= np.linalg.norm(self.quats, axis=1)[:, np.newaxis]
            Ef = per_rod_energy(self.all_energy_densities(force=force))
            deltaE = Ef - Ei
            reject = mask.copy()
            if self.env.T <= Environment.MIN_TEMP:
                reject[deltaE <= 0.] = False
            else:
                fast_calc.metropolis(reject, deltaE, even=even)
            self.quats -= reject[:, np.newaxis] * moves
            self.quats /= np.linalg.norm(self.quats, axis=1)[:, np.newaxis]
            E0 = self.all_energy_densities(force=force)
            if acceptance:
                accepted_frac[0] += 0.5 - np.count_nonzero(reject)/reject.size

        start = time.clock()
        parity = np.random.rand() > 0.5
        update_rods(even=parity)
        timers[0] += time.clock() - start
        update_rods(even=(not parity))
        if acceptance:
            return E0, accepted_frac
        return E0

    def metropolis_update(self, force, E0, acceptance=False):
        u"""Updates Euler angles using the Metropolis algorithm.

        The iteration order is random,
        i.e. we randomly pick one of even/odd and one of φ/θ/ψ and give kicks to
        the corresponding angles.

        Args:
            force (Array[(3,)]): force as a vector
            E0 (Tuple[(3,); Array[...]]):
                Initial energy densities in the order bend, twist, stretch.
            acceptance (bool): Whether acceptance ratios should be recorded.

        Returns:
            If acceptance is true, a tuple with the energy density and the acceptance
            ratios. Otherwise, it return the energy density.

            The energy density is of shape (L,).

        WARNING:
            Currently the acceptance values returned are incorrect.
        """
        sigma = self.sim.kickSize
        timers = self.sim.timers

        moves = np.random.normal(loc=0.0, scale=sigma, size=(self.L - 1, 3))
        moves[np.abs(moves) >= 5.0 * sigma] = 0.
        if acceptance:
            accepted_frac = np.zeros(3)

        # TODO: fix acceptance computation
        def update_rods(i, even=True):
            # ψ is updated from 1 to L whereas φ and θ are updated from 1 to L-1
            if i == 2:
                mask = self.oddMask if even else self.evenMask
                rod_slice = slice(1, None)
                reject_even = not even
            else:
                mask = self.evenMask if even else self.oddMask
                rod_slice = slice(0, -1)
                reject_even = even

            nonlocal E0
            bend_Ei, twist_Ei, stretch_Ei = E0
            Ei = bend_Ei[rod_slice] + twist_Ei[rod_slice] + stretch_Ei

            _, _, old_gammasq = self.twist_bend_angles()
            old_euler = self.euler.copy()

            self.euler[rod_slice, i] += moves[:, i] * mask
            bend_Ef, twist_Ef, stretch_Ef = self.all_energy_densities(force=force)
            Ef = bend_Ef[rod_slice] + twist_Ef[rod_slice] + stretch_Ef
            deltaE = (Ef - Ei)[:-1] + (Ef - Ei)[1:]
            # If only rod j is moved, 5 terms will change in the energy
            # computation: 4 due to hinges j and j+1, and 1 due to stretching.
            #
            #     |         | j+1       | j       |
            #     |---------+-----------+---------|
            #     | twist   | changed   | changed |
            #     | bend    | changed   | changed |
            #     | stretch | unchanged | changed |
            #
            # deltaE[j] = Ef[j] - Ei[j] + Ef[j+1] -Ei[j+1] will solely affect
            # reject[j] and finally euler[j] (for φ/θ) or euler[j+1] (for ψ).

            reject = mask.copy()
            if self.env.T <= Environment.MIN_TEMP:
                reject[deltaE <= 0.] = False
            else:
                fast_calc.metropolis(reject, deltaE, even=reject_even)

            # diffstr = "{0:<5}: {1:7.4} -> {2:7.4} | {3:6} = {4:7.4}"
            # bluebold = '\033[33m\033[1m'
            # endcolor = '\033[0m'

            # def degree(ang):
            #     return ang * 180 / np.pi

            # color = False

            # def diff(param_str, old_param, new_param, scale_f=lambda x: x):
            #     param_diff_str = "Δ" + param_str
            #     old = scale_f(old_param)
            #     new = scale_f(new_param)
            #     param_diff = new - old
            #     nonlocal color
            #     if color:
            #         s = bluebold + diffstr + endcolor
            #     else:
            #         s = diffstr
            #     print(s.format(param_str, old, new, param_diff_str, param_diff))

            # def probe():
            #     if i == 2:
            #         _, _, gammasq = self.twist_bend_angles()
            #         # gammasq == 2 (1 - cos(Δψ)) when θ ~ 0
            #         for j in range(1, self.L+1):
            #             if gammasq[j] > 2 * (1 - np.cos(np.pi/2)):
            #                 if j == self.L:
            #                     if reject[j-2]:
            #                         return
            #                     first = old_euler[-1][2]
            #                     first_mv = self.euler[-1][2]
            #                     snd = self.end[2]
            #                     snd_mv = self.end[2]
            #                 else:
            #                     if reject[j-2] or reject[j-1]:
            #                         return
            #                     first = old_euler[j-1][2]
            #                     first_mv = self.euler[j-1][2]
            #                     snd = old_euler[j][2]
            #                     snd_mv = self.euler[j][2]
            #                 if gammasq[j] > 2 * (1 - np.cos(3 * np.pi / 4)):
            #                     print("BIG ANGLES!")
            #                     # nonlocal color
            #                     # color = True
            #                 diff1 = np.abs(snd - first)
            #                 diff2 = np.abs(snd_mv - first_mv)
            #                 diff("ψ" + str(j-1), first, first_mv, degree)
            #                 diff("ψ" + str(j), snd, snd_mv, degree)
            #                 diff("Δψ", diff1, diff2, degree)
            #                 diff("Γ", old_gammasq[j], gammasq[j], lambda x: degree(np.sqrt(x)))
            #                 diff("twE", twist_Ei[j], twist_Ef[j])
            #                 print("---")
            #                 if gammasq[j] > 2 * (1 - np.cos(3 * np.pi / 4)):
            #                     pass
            #                     # color = False
            # # probe()
            # def energy_probe():
            #     for (rej, dE) in zip(reject, deltaE):
            #         if dE > 4.0 and not rej:
            #             print(dE)

            # energy_probe()

            self.euler[rod_slice, i] -= moves[:, i] * reject
            E0 = self.all_energy_densities(force=force)
            if acceptance:
                accepted_frac[i] += 0.5 - np.count_nonzero(reject)/reject.size

        for i in np.random.permutation(3):
            start = time.clock()
            parity = np.random.rand() > 0.5
            update_rods(i, even=parity)
            timers[0] += time.clock() - start
            update_rods(i, even=(not parity))

        if acceptance:
            return E0, accepted_frac
        return E0

    #-----------# TEMPORARY SUBSTITUTION #-----------#
    metropolis_update = metropolis_update_quat
    #------------------------------------------------#

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
        force_vector = np.array([0., 0., force])
        evol = Evolution(self, nsteps, force_vector, twists=tmp_twists, initial=includeStart)
        evol.update({"force": force, "mcSteps": mcSteps})
        E0 = self.all_energy_densities(force_vector)
        for x in tmp_twists:
            self.end[2] = x
            self.end_quat = fast_calc.quaternion_of_euler1(self.end)
            for nstep in nsteps:
                E0, extension, acceptance = self.mc_relaxation(
                    force_vector, E0, nstep)
                # can't use sum directly due to mismatched shapes
                # E_total = ([np.sum(x) for x in E0])
                (evol.save_energy(E0)
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
        force_vector = np.array([0., 0., force])
        evol = Evolution(self, nsteps, force_vector, initial=includeStart)
        evol.update({"force": force, "mcSteps": mcSteps})
        E0 = self.all_energy_densities(force_vector)
        for nstep in nsteps:
            E0, extension, acceptance = self.mc_relaxation(
                force_vector, E0, nstep)
            # can't use sum directly due to mismatched shapes
            # E_total = ([np.sum(x) for x in E0])
            (evol.save_energy(E0)
             .save_extension(extension)
             .save_acceptance(acceptance)
             .save_angles(self))
        timers[8] += time.clock() - start
        timing = {s : timers[i] for (i, s) in Simulation.timer_descr.items()}
        evol.save_timing(timing)
        return evol.to_dataset()


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
        * 'vertical' -> nucleosomes arranged roughly vertically
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

        if nucArrayType == "vertical":
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
            raise ValueError("nucArrayType should be either 'vertical' or 'relaxed'.")

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
        force_vector = np.array([0., 0., force])
        evol = Evolution(self, nsteps, force_vector, initial=includeStart)
        if includeStart and includeDummyRods:
            dummyRodAngles.append(self.anglesForDummyRods())
        E0 = self.all_energy_densities(force_vector)
        for nstep in nsteps:
            E0, extension, acceptance = self.mc_relaxation(
                force_vector, E0, nstep)
            # can't use sum directly due to mismatched shapes
            # E_total = ([np.sum(x) for x in E0])
            (evol.save_energy(E0)
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
