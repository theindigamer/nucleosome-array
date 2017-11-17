import chromatinMD as cmd
import numpy as np
import fast_calc
import hypothesis.extra.numpy as hnp
from hypothesis import given
from hypothesis.strategies import floats, composite

FINITE_FLOATS = floats(allow_nan=False, allow_infinity=False)
ANGULAR_FLOATS = floats(
    min_value=0., max_value=2 * np.pi, allow_nan=False, allow_infinity=False)


@composite
def _array(draw, elements=FINITE_FLOATS):
    return (lambda n: draw(hnp.arrays(float, n, elements=elements)))


@composite
def _strand_r(draw):
    f = draw(_array(ANGULAR_FLOATS))

    def g(n):
        phi, theta, psi = f((3, n))
        z = np.cos(theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        return np.append(np.cumsum([x, y, z], axis=1), [psi], axis=0).T

    return g


@given(_array())
def test_eulerMatrix(f):
    """Checks that eulerMatrixOfAngles and anglesOfEulerMatrix are inverses."""
    # We compare the rotation matrices and not the angles because several
    # combinations of angles are degenerate and the inverse function
    # ``anglesOfEulerMatrix`` picks a specific one.
    x1 = f(3)
    m1 = fast_calc.eulerMatrixOfAngles(x1)
    x2 = fast_calc.anglesOfEulerMatrix(m1)
    m2 = fast_calc.eulerMatrixOfAngles(x2)
    assert np.allclose(m1, m2, atol=1.E-8)


@given(_strand_r())
def test_derivative_rotation_matrices(f):
    s = cmd.strand()
    s.r[:] = f(s.L)
    t = s.tangent_vectors()
    ang = cmd.angular(s, tangent=t)
    m1 = ang.derivativeRotationMatrices()
    m2 = ang.oldDerivativeRotationMatrices()
    assert np.allclose(m1, m2, atol=1.E-16)


@given(_strand_r())
def test_jacobian(f):
    s = cmd.strand()
    s.r[:] = f(s.L)
    t = s.tangent_vectors()
    m1 = s.jacobian(tangent=t)
    m2 = s.oldJacobian(tangent=t)
    assert np.allclose(m1, m2, atol=1.E-12)
