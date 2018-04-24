import numpy as np
from abc import ABC, abstractmethod

OVERRIDE_ERR_MSG = "Forgot to override this method?"


class Semigroup(ABC):
    @abstractmethod
    def join(self, next_obs):
        """Combines the observations from two observers to give a new one."""
        raise NotImplementedError(OVERRIDE_ERR_MSG)


class Observer(Semigroup):
    @abstractmethod
    def update(self, subject, **kwargs):
        """Update the observer that a change was made."""
        raise NotImplementedError(OVERRIDE_ERR_MSG)

    @abstractmethod
    def collect(self):
        """Return the collected observations, if any."""
        raise NotImplementedError(OVERRIDE_ERR_MSG)


class EnergyObserver(Observer, Semigroup):
    __slots__ = ("bendE", "twistE", "stretchE", "totalE")

    def __init__(self, num_recordings):
        self.bendE = np.empty(num_recordings)
        self.twistE = np.empty(num_recordings)
        self.stretchE = np.empty(num_recordings)
        self.totalE = np.empty(num_recordings)

    def update(self, counter, strand, force=None, **kwargs):
        idx = counter
        self.bendE[idx], self.twistE[idx], self.stretchE[idx] = (
            strand.all_energy_densities(force))
        self.totalE[idx] = (
            self.bendE[idx] + self.twistE[idx] + self.stretchE[idx])

    def collect(self):
        return (self.bendE, self.twistE, self.stretchE, self.totalE)

    def join(self, next_obs):
        totalE = np.append(self.totalE, next_obs.totalE)
        tmp = EnergyObserver(totalE.size)
        tmp.totalE = totalE
        tmp.bendE = np.append(self.bendE, next_obs.bendE)
        tmp.twistE = np.append(self.twistE, next_obs.twistE)
        tmp.stretchE = np.append(self.stretchE, next_obs.stretchE)
        return tmp


class ExtensionObserver(Observer, Semigroup):
    __slots__ = ("extension")

    def __init__(self, num_recordings):
        self.extension = np.empty(num_recordings)

    def update(self, counter, strand):
        self.extension[counter] = strand.total_extension()

    def collect(self):
        return self.extension

    def join(self, next_obs):
        ext = np.append(self.extension, next_obs.extension)
        tmp = ExtensionObserver(ext.size)
        tmp.extension = ext
        return tmp


class EulerAngleObserver(Observer):
    __slots__ = ("start", "euler", "end")

    def __init__(self, num_recordings):
        self.start = np.empty((num_recordings, 3))
        self.start = np.empty((num_recordings, 3))
        self.end = np.empty((num_recordings, 3))

    # def update(self, counter, strand):
