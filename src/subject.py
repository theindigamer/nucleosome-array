from abc import ABC

OVERRIDE_ERR_MSG = "Forgot to override this method?"

class Subject:
    __slots__ = ("observers")

    def __init__(self):
        self.observers = {}

    def register(self, name, observer):
        """Registers an observer with Subject."""
        self.observers.update({name: observer})

    def notify(self, name):
        """Notify one particular observer."""
        self.observers[name].notify()

    def notify_all(self):
        """Notifies all observers that Subject's data has changed."""
        for obs in self.observers.values():
            obs.notify(self)

