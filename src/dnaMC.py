import numpy as np
import time
import utils
import xarray as xr


class Environment:
    """Describes the environment of the DNA/nucleosome array.

    Future properties one might include: salt/ion concentration etc.
    """
    ROOM_TEMP = 293.15 # in Kelvin
    MIN_TEMP = 1E-10   # in Kelvin
    def __init__(self, T=ROOM_TEMP):
        self.T = T


class Simulation:
    """Simulation parameters that can be varied."""
    DEFAULT_KICK_SIZE = 0.1
    ntimers = 10
    timer_descr = {
        0 : "Half of inner loop in metropolisUpdate",
        1 : "Half of half of inner loop in metropolisUpdate",
        2 : "Calls to bendingEnergyDensity",
        3 : "Calls to twistEnergyDensity",
        4 : "Calculating Rs in deltaMatrices via rotationMatrices",
        5 : "Calculating a in deltaMatrices",
        6 : "Calculating b in deltaMatrices",
        7 : "Calculating deltas in deltaMatrices",
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

    def __init__(self, dna, nsteps, twists=None, initial=False):
        """Initialize an evolution object.

        ``twists`` is either a nonempty ``numpy`` array (twisting) or ``None``
        (relaxation).
        ``nsteps`` represents successive number of MC steps.
        If ``twists`` is ``None``, it is interpreted to represent the entire
        simulation. So [10, 10, 5] means that the simulation has 25 steps total.
        If ``twists`` is a numpy array, it is interpreted to be the number of
        steps between consecutive twists.
        So ``twists = [0, pi/2]`` and ``nsteps=[10, 10, 5]`` implies there are
        50 MC steps in total, split up as 10, 10, 5, 10, 10, 5.
        """
        self.initial = initial
        self.tstep_i = 0
        if twists is None:
            tsteps = np.cumsum(nsteps)
        else:
            tmp = np.cumsum(nsteps)
            tsteps = np.concatenate([tmp + tmp[-1] * i
                                     for i in range(len(twists))])
        if initial:
            tsteps = np.insert(tsteps, 0, 0)
        self.data = dna.constants()
        self.data.update({
            "twists": twists,
            "tsteps": tsteps,
            "angles": np.empty((tsteps.size, dna.L, 3)),
            "extension": np.empty((tsteps.size, 3)),
            "energy": np.empty(tsteps.size),
            "acceptance": np.empty((tsteps.size, 3)),
        })
        if initial:
            (self.save_energy(dna.totalEnergy())
             .save_extension(dna.totalExtension())
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
            "initial": self.initial,
        }
        if "nucleosome" in data.keys():
            attrs.update({"nucleosome": data["nucleosome"]})
        saved_keys = (list(data_vars.keys()) + list(coords.keys())
                      + list(attrs.keys()))
        for (k, v) in data.items():
            if not k in saved_keys:
                attrs.update({k: v})
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

class NakedDNA:
    """A strand of DNA without any nucleosomes.

    Euler angles are ordered as phi, theta and psi.
    Values of B and C are from [1, Table I].
    [1, (30)] describes the microscopic parameter Bm computed from B.
    """

    DEFAULT_TWIST_STEP = np.pi/4

    def __init__(self, L=740, B=43.0, C=89.0,
                 T=Environment.ROOM_TEMP,
                 kickSize=Simulation.DEFAULT_KICK_SIZE):
        self.L = L
        # B, Bm and C are in nm kT
        self.B = B
        self.Pinv = 0 # no intrinsic disorder
        self.Bm = B
        self.C = C
        self.env = Environment(T=T)
        self.sim = Simulation(kickSize=kickSize)
        self.strandLength = 740.0 # in nm
        self.d = self.strandLength/L # total length is 740 nm
        self.euler = np.zeros((self.L, 3))
        self.oddMask = np.array([i % 2 == 1 for i in range(self.L - 2)])
        self.evenMask = np.roll(self.oddMask, 1)

    def constants(self):
        return {
            "temperature": self.env.T,
            "B": self.B,
            "Pinv": self.Pinv,
            "Bm": self.Bm,
            "C": self.C,
            "rodCount": self.L,
            "strandLength": self.strandLength,
            "rodLength": self.d,
            "kickSize": self.sim.kickSize,
        }

    def rotationMatrices(self):
        """Returns rotation matrices along the DNA string"""
        return utils.rotation_matrices(self.euler)

    def deltaMatrices( self, Rs=None ):
        """Returns Δ matrices describing bends/twists between consecutive rods."""
        timers = self.sim.timers
        start = time.clock()
        if Rs is None:
            Rs = self.rotationMatrices()
        timers[4] += time.clock() - start

        start = time.clock()
        a = np.swapaxes( Rs[:-1], 1, 2 )
        timers[5] += time.clock() - start

        start = time.clock()
        b = Rs[1:]
        timers[6] += time.clock() - start

        start = time.clock()
        deltas = a @ b
        timers[7] += time.clock() - start
        return deltas

    def twistBendAngles(self, Ds=None):
        """ Returns the twist and bending angles."""
        if Ds is None:
            Ds = self.deltaMatrices()
        if self.sim.squared:
            betaSq = 2.0 * (1 - Ds[:, 2, 2])
            GammaSq = 1.0 - Ds[:, 0, 0] - Ds[:, 1, 1] + Ds[:, 2, 2]
            return betaSq, GammaSq
        else:
            beta1 = (Ds[:, 1, 2] - Ds[:, 2, 1]) / 2.0
            beta2 = (Ds[:, 2, 0] - Ds[:, 0, 2]) / 2.0
            Gamma = (Ds[:, 0, 1] - Ds[:, 1, 0]) / 2.0
            return beta1, beta2, Gamma

    def bendingEnergyDensity(self, angles=None):
        """ Returns the bending energy density.
            Enter angles in a tuple( arrays ) format."""
        if angles is None:
            angles = self.twistBendAngles()
        if self.sim.squared:
            energy_density = self.Bm / (2.0*self.d) * angles[0]
        else:
            energy_density = (self.Bm / (2.0*self.d)
                              * (angles[0]**2 + angles[1]**2))
        return energy_density, angles

    def bendingEnergy(self, bendEnergyDensity=None):
        """ Returns the total bending energy."""
        if bendEnergyDensity is None:
            bendEnergyDensity, _ = self.bendingEnergyDensity()
        return np.sum(bendEnergyDensity)

    def twistEnergyDensity(self, angles=None):
        """ Returns the twist energy density."""
        if angles is None:
            angles = self.twistBendAngles()
        if self.sim.squared:
            return self.C * angles[-1] / (2.0*self.d)
        else:
            return self.C * angles[-1]**2 / (2.0*self.d)

    def twistEnergy(self, twistEnergyDensity=None ):
        """ Returns the total twist energy. """
        if twistEnergyDensity is None:
            twistEnergyDensity = self.twistEnergyDensity()
        return np.sum(twistEnergyDensity)

    def stretchEnergyDensity(self, force, tangent=None):
        """ Returns the stretching energy density.
            Enter the force in pN
            Our energy is in unit of kT.
            Es = force * tangent * prefactor
            prefactor = 1E-12 1E-9/ (1.38E-23 T).
            Change prefactor to change the temperature."""
        T = max(self.env.T, Environment.MIN_TEMP)
        prefactor = 1.0 / (1.38E-2 * T)
        if tangent is None:
            tangent = self.tVector()
        return -force * prefactor * tangent[:-1,2]

    def stretchEnergy(self, force, stretchEnergyDensity=None):
        """ Returns the total stretching energy. """
        if stretchEnergyDensity is None:
            stretchEnergyDensity = self.stretchEnergyDensity(force)
        return np.sum(stretchEnergyDensity)

    def totalEnergyDensity(self, force):
        """ Returns the total energy density."""
        timers = self.sim.timers

        start = time.clock()
        E, angles = self.bendingEnergyDensity()
        timers[2] += time.clock() - start

        start = time.clock()
        E += self.twistEnergyDensity(angles=angles)
        timers[3] += time.clock() - start

        E += self.stretchEnergyDensity(force)
        return E

    def totalEnergy(self, force, totalEnergyDensity=None):
        """ Returns the total energy. """
        if totalEnergyDensity is None:
            totalEnergyDensity = self.totalEnergyDensity(force)
        return np.sum(totalEnergyDensity)

    def tVector( self ):
        """ Returns the tangent vectors. """
        ( phi, theta ) = ( self.euler[:,0], self.euler[:,1] )
        t = np.zeros(( self.L, 3 ))
        t[:, 0] = np.sin(theta) * np.sin(phi)
        t[:, 1] = np.cos(phi) * np.sin(theta)
        t[:, 2] = np.cos(theta)
        return t

    def rVector( self, t=None ):
        """ Returns the end points of the t-vectors."""
        if t is None:
            t = self.tVector()
        return np.cumsum( t, axis=0 )

    def metropolisUpdate(self, force, E0, acceptance=False):
        """ Updates dnaClass Euler angles using Metropolis algorithm.
        Returns the total energy density.
        Temperature T is in Kelvin.
        """
        sigma = self.sim.kickSize
        timers = self.sim.timers

        moves = np.random.normal(loc=0.0, scale=sigma, size=(self.L - 2, 3))
        moves[np.abs(moves) >= 5.0*sigma] = 0
        if acceptance:
            accepted_frac = np.zeros(3)

        for i in range(3):
            start = time.clock()
            # Move even rods first.
            # oddMask is False, True, False, ... and we start from euler[1]
            # so only euler[2], euler[4], ... are changed
            self.euler[1:-1, i] += moves[:, i] * self.oddMask
            Ef = self.totalEnergyDensity(force)
            deltaE = ( Ef - E0 )[:-1] + ( Ef - E0 )[1:]
            timers[1] += time.clock() - start

            # 1.0 ⇔ true ⇔ move is rejected, first reject all moves of even rods
            reject = 1.0 * self.oddMask
            # Now reject == [0., 1., 0., 1., ...]
            # Wait! Some of these should be accepted according to the Metropolis
            # algorithm. We only need to examine _odd_ indices of reject.
            if self.env.T == Environment.MIN_TEMP:
                reject[deltaE <= 0.] = 0.
            else:
                utils.metropolis(reject, deltaE, even=False)
            self.euler[1:-1,i] -= moves[:, i] * reject
            E0 = self.totalEnergyDensity(force)
            if acceptance:
                accepted_frac[i] += 0.5 - np.count_nonzero(reject)/reject.size
            timers[0] += time.clock() - start

            # Move odd rods now.
            self.euler[1:-1, i] += moves[:, i] * self.evenMask
            Ef = self.totalEnergyDensity(force)
            deltaE = (Ef - E0)[:-1] + (Ef - E0)[1:]

            reject = 1.0 * self.evenMask
            utils.metropolis(reject, deltaE, even=True)
            self.euler[1:-1,i] -= moves[:, i] * reject
            E0 = self.totalEnergyDensity(force)
            if acceptance:
                accepted_frac[i] += 0.5 - np.count_nonzero(reject)/reject.size

        if acceptance:
            return E0, accepted_frac
        return E0

    def totalExtension(self):
        """Returns [Δx, Δy, Δz] given as r_bead - r_bottom."""
        return self.d * np.sum(self.tVector(), axis=0)

    def mcRelaxation(self, force, E0, mcSteps, record_final_only=True):
        """Monte Carlo relaxation using Metropolis algorithm."""
        if record_final_only:
            for _ in range(mcSteps - 1):
                E0 = self.metropolisUpdate(force, E0, acceptance=False)
            E0, acc = self.metropolisUpdate(force, E0, acceptance=True)
            return E0, self.totalExtension(), acc
        else:
            energies = np.empty((mcSteps, self.L))
            extensions = np.empty((mcSteps, 3))
            acceptance_ratios = np.empty((mcSteps, 3))
            for i in range(mcSteps):
                E0, acceptance_ratios[i] = (
                    self.metropolisUpdate(force, E0, acceptance=True)
                )
                energies[i] = E0
                extensions[i] = self.totalExtension()
            return energies, extensions, acceptance_ratios

    def torsionProtocol(self, force=1.96, E0=None, mcSteps=100,
                        twists=2*np.pi, nsamples=1, includeStart=False):
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
        nsteps = utils.partition(nsamples, mcSteps)
        tmp_twists = utils.twist_steps(self.DEFAULT_TWIST_STEP, twists)
        evol = Evolution(self, nsteps, twists=tmp_twists, initial=includeStart)
        evol.update({"force": force, "mcSteps": mcSteps})
        E0 = self.totalEnergyDensity(force)
        for x in tmp_twists:
            for nstep in nsteps:
                self.euler[-1, 2] = x
                E0, extension, acceptance = self.mcRelaxation(force, E0,
                                                                  nstep)
                (evol.save_energy(np.sum(E0))
                 .save_extension(extension)
                 .save_acceptance(acceptance)
                 .save_angles(self))
        timers[8] += time.clock() - start
        timing = {s : timers[i] for (i, s) in Simulation.timer_descr.items()}
        evol.save_timing(timing)
        return evol.to_dataset()

    def relaxationProtocol(self, force=1.96, E0=None,
                           mcSteps=1000, nsamples=4, includeStart=False):
        """
        Simulate a relaxation for an initial twist profile.

        The DNA should be specified with the required profile already applied.
        """
        start = time.clock()
        timers = self.sim.timers
        nsteps = utils.partition(nsamples, mcSteps)
        evol = Evolution(self, nsteps, initial=includeStart)
        evol.update({"force": force, "mcSteps": mcSteps})
        E0 = self.totalEnergyDensity(force)
        for nstep in nsteps:
            E0, extension, acceptance = self.mcRelaxation(force, E0, nstep)
            (evol.save_energy(np.sum(E0))
             .save_extension(extension)
             .save_acceptance(acceptance)
             .save_angles(self))
        timers[8] += time.clock() - start
        timing = {s : timers[i] for (i, s) in Simulation.timer_descr.items()}
        evol.save_timing(timing)
        return evol.to_dataset()


class DisorderedNakedDNA(NakedDNA):

    # [1, Appendix 1] describes equations related to disorder.
    def __init__(self, Pinv=1./300, **kwargs):
        # The 1/300 value has no particular significance.
        # [1] considers values from ~1/1000 to ~1/100
        NakedDNA.__init__(self, **kwargs)
        self.Pinv = Pinv # inverse of P value in 1/nm
        self.Bm = self.B / (1 - self.B * Pinv)
        sigma_b = (Pinv * self.d) ** 0.5
        # xi represents [dot(xi_m, n_1), dot(xi_m, n_2)], see [1, (E1)]
        xi = np.random.randn(2, self.L - 1)
        self.bend_zeros = sigma_b * xi
        self.sim.squared = False

    def bendingEnergyDensity(self, angles=None):
        if angles is None:
            angles = self.twistBendAngles()
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
        self.euler[self.nuc] = utils.exitAngles([0., 0., 0.])

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
                    tmp = utils.exitAngles(prev)
                    dna.euler[i] = np.copy(tmp)
                    prev = tmp
                    nucPos = nucPos[1:]
                else:
                    dna.euler[i] = np.copy(prev)
        else:
            raise ValueError("nucArrayType should be either 'standard' or 'relaxed'.")

        return dna

    def deltaMatrices(self, Rs=None):
        """Returns Δ matrices describing bends/twists between consecutive rods."""
        start = time.clock()
        timers = self.sim.timers
        if Rs is None:
            Rs = self.rotationMatrices()
        timers[4] += time.clock() - start

        start = time.clock()
        a = np.swapaxes( Rs[:-1], 1, 2 )
        timers[5] += time.clock() - start

        start = time.clock()
        b = Rs[1:]
        timers[6] += time.clock() - start

        start = time.clock()
        deltas = a @ b
        utils.calc_deltas(deltas, self.nuc, Rs)
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
        return np.array([utils.exitAngles(self.euler[n-1]) for n in self.nuc])

    def relaxationProtocol(self, force=1.96, E0=None,
                           mcSteps=1000, nsamples=4, includeStart=False,
                           includeDummyRods=False):
        """Simulate a relaxation for an initial twist profile.

        The DNA should be specified with the required profile already applied.
        """
        start = time.clock()
        timers = self.sim.timers
        nsteps = utils.partition(nsamples, mcSteps)
        dummyRodAngles = []
        evol = Evolution(self, nsteps, initial=includeStart)
        if includeStart and includeDummyRods:
            dummyRodAngles.append(self.anglesForDummyRods())
        E0 = self.totalEnergyDensity(force)
        for nstep in nsteps:
            E0, extension, acceptance = self.mcRelaxation(force, E0, nstep)
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
