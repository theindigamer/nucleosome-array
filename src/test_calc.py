import chromatinMD as cmd
import numpy as np
import fast_calc
import hypothesis.extra.numpy as hnp
from hypothesis import given, settings
from hypothesis.strategies import floats, composite

FINITE_FLOATS = floats(allow_nan=False, allow_infinity=False)
ANGULAR_FLOATS = floats(min_value=0., max_value=(2 * np.pi))


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


@composite
def _angular(draw):
    strand = cmd.strand()
    strand.r[:] = draw(_strand_r())(strand.L)
    t = strand.tangent_vectors()
    ang = cmd.angular(strand, tangent=t)
    return ang


@given(_array())
@settings(deadline=2000)
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


@given(_array(elements=floats(min_value=-96*np.pi, max_value=96*np.pi)))
def test_quaternions(f):
    euler1 = f(3)
    quat1 = fast_calc.quaternion_of_euler1(euler1)
    print(str(euler1))
    euler2 = fast_calc.euler_of_quaternion1(quat1)
    print(str(euler2))
    m1 = fast_calc.eulerMatrixOfAngles(euler1)
    m2 = fast_calc.eulerMatrixOfAngles(euler2)
    assert np.allclose(m1, m2, atol=1.E-5)


@given(_angular())
@settings(deadline=2000)
def test_derivative_rotation_matrices(ang):
    m1 = ang.derivativeRotationMatrices()
    m2 = ang.oldDerivativeRotationMatrices()
    assert np.allclose(m1, m2, atol=1.E-16)


# FIXME: Update this test
# @given(_strand_r())
# def test_jacobian(f):
#     s = cmd.strand()
#     s.r[:] = f(s.L)
#     t = s.tangent_vectors()
#     m1 = s.jacobian(tangent=t)
#     m2 = s.oldJacobian(tangent=t)
#     assert np.allclose(m1, m2, atol=1.E-12)

# FIXME: Update this test
# @given(_angular())
# def test_effective_torques(ang):
#     tau1 = ang.oldEffectiveTorques()
#     tau2 = ang.effectiveTorques()
#     assert np.allclose(tau1, tau2, atol=1.E-16)


@given(floats(min_value=-20., max_value=20.))
def test_metropolis(E):
    size = 10000
    acceptance_expect = np.exp(-E) if E > 0 else 1.
    sigma = 1. / np.sqrt(size) if E > 0 else 1E-16
    deltaE = np.repeat(E, size)
    rej_even = np.repeat(True, size)
    rej_odd = rej_even.copy()
    fast_calc.metropolis(rej_even, deltaE, even=True)
    fast_calc.metropolis(rej_odd, deltaE, even=False)
    acceptance_even = 1. - np.count_nonzero(rej_even[0::2]) / (size / 2)
    acceptance_odd = 1. - np.count_nonzero(rej_odd[1::2]) / (size / 2)
    assert np.abs(acceptance_even - acceptance_expect) < 3 * sigma
    assert np.abs(acceptance_odd - acceptance_expect) < 3 * sigma
