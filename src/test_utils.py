import numpy as np
import utils
from hypothesis import given
from hypothesis.strategies import floats, lists, composite

@composite
def _float_array(draw, elements=floats(allow_nan=False, allow_infinity=False)):
    return (lambda n: np.array(draw(lists(elements, min_size=n, max_size=n))))

@given(_float_array())
def test_eulerMatrix(f):
    """Checks that eulerMatrixOfAngles and anglesOfEulerMatrix are inverses."""
    # We compare the rotation matrices and not the angles because several
    # combinations of angles are degenerate and the inverse function
    # ``anglesOfEulerMatrix`` picks a specific one.
    x1 = f(3)
    m1 = utils.eulerMatrixOfAngles(x1)
    x2 = utils.anglesOfEulerMatrix(m1)
    m2 = utils.eulerMatrixOfAngles(x2)
    assert np.allclose(m1, m2, atol=1.E-8)
