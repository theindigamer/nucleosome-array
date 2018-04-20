import numpy as np
from abc import ABC, abstractmethod

OVERRIDE_ERR_MSG = "Forgot to override this method?"

class Observer(ABC):
    @abstractmethod
    def notify(self, subject):
        """Notify the observer that a change was made."""
        raise NotImplementedError(OVERRIDE_ERR_MSG)

    @abstractmethod
    def collect(self):
        """Return the collected observations, if any."""
        raise NotImplementedError(OVERRIDE_ERR_MSG)

class EnergyObserver(Observer):
    __slots__ = ("bendE", "twistE", "stretchE", "totalE")

    def __init__(self, num_recordings):
        self.bendE = np.empty(num_recordings)
        self.twistE = np.empty(num_recordings)
        self.stretchE = np.empty(num_recordings)
        self.totalE = np.empty(num_recordings)

    def notify(self, dna):
        idx = dna.tstepCounter
        self.bendE[idx], self.twistE[idx], self.stretchE[idx] = (
            dna.all_energy_densities(dna.force))
        self.totalE[idx] = (self.bendE[idx] + self.twistE[idx] + self.stretchE[idx])

    def collect(self):
        return self.energy

class ExtensionObserver(Observer):
    __slots__ = ("extension", "idx")

    def __init__(num_recordings):
        self.extension = np.empty(num_recordings)

    def notify(self, dna):
        self.extension[dna.tstepCounter] = dna.total_extension()
