from abc import ABC

OVERRIDE_ERR_MSG = "Forgot to override this method?"

class Subject:
    __slots__ = ("_observers")

    def __init__(self):
        self._observers = {}

    def register(self, name, observer):
        """Registers an observer with Subject."""
        self._observers.update({name: observer})

    def notify(self, name):
        """Notify one particular observer."""
        self._observers[name].update(self)
        return self

    def notify_all(self):
        """Notifies all observers that Subject's data has changed."""
        for obs in self._observers.values():
            obs.update(self)
