class Environment:
    """Describes the environment of the DNA/nucleosome array.

    Future properties one might include: salt/ion concentration etc.
    """
    ROOM_TEMP = 296.65 # in Kelvin. This value is used in [F, page 2] when
                       # measuring B and C.
    MIN_TEMP = 1E-10   # in Kelvin

    __slots__ = ("T")

    def __init__(self, T=ROOM_TEMP):
        self.T = T
