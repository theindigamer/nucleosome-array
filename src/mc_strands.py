import numpy as np
from abc import ABC, abstractmethod

import fast_calc
from strands import EulerAngleDescription, QuaternionDescription

OVERRIDE_ERR_MSG = "Forgot to override this method?"


class MetropolisABC(ABC):
    """Abstract base class for a strand that supports MC simulation."""

    __slots__ = ()

    @abstractmethod
    def _mask_length(self):
        raise NotImplementedError(OVERRIDE_ERR_MSG)

    @abstractmethod
    def metropolis_step(self, force, temperature, kick_size, acceptance=None):
        raise NotImplementedError(OVERRIDE_ERR_MSG)

    @abstractmethod
    def set_end(self, euler1):
        """Set the angles at the end of the strand using [φ, θ, ψ]."""
        raise NotImplementedError(OVERRIDE_ERR_MSG)


class EulerAngleMCSim(EulerAngleDescription, MetropolisABC):
    __slots__ = ("oddMask", "evenMask", "E")

    def _mask_length(self):
        return self.L - 1

    def __init__(self, *args, **kwargs):
        EulerAngleDescription.__init__(self, *args, **kwargs)
        self.oddMask, self.evenMask = oddEvenMasks(self._mask_length())

    def set_end(self, euler1):
        self.end[:] = euler1


class QuaternionMCSim(QuaternionDescription, MetropolisABC):

    # E is the energy calculated in the last step.
    __slots__ = ("oddMask", "evenMask", "E")

    def _mask_length(self):
        return self.L

    def __init__(self, *args, force=None, temperature=None, **kwargs):
        QuaternionDescription.__init__(self, *args, **kwargs)
        if force is None or temperature is None:
            raise ValueError(
                "Force and temperature are needed to compute the initial"
                " energy when initializing the strand.")
        if np.shape(force) != (3,):
            raise ValueError("Force should have shape (3,).")
        self.E = self.all_energy_densities(
            force=force, temperature=temperature)
        self.oddMask, self.evenMask = oddEvenMasks(self._mask_length())

    def metropolis_step(self, force, temperature, kick_size, acceptance=None):
        """

        Args:
            force (Array[(3,), float]):
            kick_size (float):
            acceptance: Either None (meaning it shouldn't be calculated or saved)
              or some arbitrary value (ignored).

        Returns:
            Updated value of acceptance if it was a floating point value.
        """
        moves = kick_size * fast_calc.random_unit_quaternions(self.L)

        if acceptance is not None:
            acceptance = 0.

        def per_rod_energy(E):
            bend_E, twist_E, stretch_E = E
            return (bend_E[:-1] + bend_E[1:] + twist_E[:-1] + twist_E[1:] +
                    stretch_E)

        def update_rods(even=True):
            mask = self.evenMask if even else self.oddMask
            Ei = per_rod_energy(self.E)
            self.quats += mask[:, np.newaxis] * moves
            self.quats /= np.linalg.norm(self.quats, axis=1)[:, np.newaxis]
            Ef = per_rod_energy(
                self.all_energy_densities(
                    force=force, temperature=temperature))
            deltaE = Ef - Ei
            reject = mask.copy()
            # if self.env.T <= Environment.MIN_TEMP:
            #     reject[deltaE <= 0.] = False
            # else:
            fast_calc.metropolis(reject, deltaE, even=even)
            self.quats -= reject[:, np.newaxis] * moves
            self.quats /= np.linalg.norm(self.quats, axis=1)[:, np.newaxis]
            self.E = self.all_energy_densities(force=force, temperature=temperature)
            nonlocal acceptance
            if acceptance is not None:
                acceptance += 0.5 - np.count_nonzero(reject) / reject.size

        parity = np.random.rand() > 0.5
        update_rods(even=parity)
        update_rods(even=(not parity))
        return acceptance

    def set_end(self, euler1):
        self.end_quat = fast_calc.quaternion_of_euler1(euler1)


def oddEvenMasks(n):
    oddMask = np.array([i % 2 == 1 for i in range(n)])
    oddMask.setflags(write=False)
    evenMask = np.logical_not(oddMask)
    evenMask.setflags(write=False)
    return (oddMask, evenMask)
