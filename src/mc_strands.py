from abc import ABC, abstractmethod
from strands import AngularDescription, QuaternionDescription

OVERRIDE_ERR_MSG = "Forgot to override this method?"

class MonteCarloSim(ABC):

    @abstractmethod
    def _mask_length(self):
        raise NotImplementedError(OVERRIDE_ERR_MSG)

    @abstractmethod
    def metropolis_step(self, force, temperature, acceptance=False):
        raise NotImplementedError(OVERRIDE_ERR_MSG)

class AngularMCSim(AngularDescription, MonteCarloSim):
    __slots__ = ("oddMask", "evenMask")

    def _mask_length(self):
        return self.L - 1

    def __init__(self, *args, **kwargs):
        AngularDescription.__init__(*args, **kwargs)
        self.oddMask, self.evenMask = oddEvenMasks(self._mask_length())

class QuaternionMCSim(QuaternionDescription, MonteCarloSim):
    __slots__ = ("oddMask", "evenMask", "E")

    def _mask_length(self):
        return self.L

    def __init__(self, *args, **kwargs):
        AngularDescription.__init__(*args, **kwargs)
        self.oddMask, self.evenMask = oddEvenMasks(self._mask_length())

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



class QuaternionMCSim(QuaternionDescription, MonteCarloSim):

def oddEvenMasks(n):
    oddMask = np.array([i % 2 == 1 for i in range(n)])
    oddMask.setflags(write=False)
    evenMask = np.logical_not(oddMask)
    evenMask.setflags(write=False)
    return (oddMask, evenMask)
