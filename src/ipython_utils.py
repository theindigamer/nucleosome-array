import dnaMC
import numpy as np
import matplotlib.pyplot as plt

def test(n=256, L=32, n_steps=20, step_size=np.pi/32):
    dna = dnaMC.nakedDNA(L=L)
    a, b, c = dnaMC.torsionProtocol(dna,
                                    twists = step_size * np.arange(1, n+1, 1),
                                    mcSteps=n_steps)
    return (dna, (a, b, c))

def tot_time(t):
    return t[2]["Total time"]

def plot_angles(dna, t, show=True):
    euler = dna.euler/(2*np.pi)
    plt.plot(euler[:,0], label="φ")
    plt.plot(euler[:,1], label="ϑ")
    plt.plot(euler[:,2], label="ψ")
    plt.plot(euler[:,0] + euler[:,2], label="φ+ψ")
    plt.legend(loc="upper left")
    plt.title("Running time {0:.1f} s".format(tot_time(t)))
    plt.ylabel("Angle/2π radians")
    if show:
        plt.show()
