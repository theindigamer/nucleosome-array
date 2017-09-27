import dnaMC

import matplotlib.pyplot as plt
import numpy as np
from numpy import log10, sqrt
import scipy
from scipy.optimize import curve_fit

import json
import pickle

import copy
import pprint

def toListlike(results):
    tmp = copy.deepcopy(results)
    tmp["angles"] = tmp["angles"].tolist()
    tmp["tsteps"] = tmp["tsteps"].tolist()
    return tmp

def savedata(results, fname):
    tmp = toListlike(results)
    if fname.endswith(".pckl"):
        with open(fname, "wb") as f:
            pickle.dump(tmp, f)
    elif fname.endswith(".json"):
        with open(fname, "w") as f:
            json.dump(tmp, f)
    else:
        print("Unrecognized file extension. Use .json or .pckl.")

def test(n=256, L=32, mcSteps=20, step_size=np.pi/32, nsamples=1):
    """Twisting a DNA from one end.

    Returns a DNA object in the twisted form and a dictionary containing
    relevant parameters.
    """
    dna = dnaMC.nakedDNA(L=L)
    result = dna.torsionProtocol(twists = step_size * np.arange(1, n+1, 1),
                                 mcSteps=mcSteps, nsamples=nsamples)
    return (dna, result)

def testDeltafn(L=32, height=np.pi/4, mcSteps=100, nsamples=4):
    """Testing the relaxation of a delta function set at L/2.

    height should be given a small value (w.r.t 2pi) as energy calculations
    are invariant under pi additional twist between rods.
    """
    dna = dnaMC.nakedDNA(L=L)
    dna.euler[L//2,2] = height
    results = dna.relaxationProtocol(mcSteps=mcSteps, nsamples=nsamples)
    return (dna, results)

def testStepfn(L=32, height=np.pi/4, mcSteps=100, nsamples=4):
    """Testing the relaxation of a step function rising at L/2.

    height should be given a small value (w.r.t 2pi) as energy calculations
    are invariant under 2pi additional twist between rods.
    """
    dna = dnaMC.nakedDNA(L=L)
    dna.euler[L//2:,2] = height
    results = dna.relaxationProtocol(mcSteps=mcSteps, nsamples=nsamples)
    return (dna, results)

def totalTime(result):
    return result["timing"]["Total time"]

def plotAngles(dna, result, totalOnly=True, show=True):
    """Make a plot of angles as a function of x (rod number).

    Use case: after using twistProtocol on the dna object.
    """
    euler = dna.euler/(2*np.pi)
    if not totalOnly:
        plt.plot(euler[:,0], label="φ")
        plt.plot(euler[:,1], label="ϑ")
        plt.plot(euler[:,2], label="ψ")
    plt.plot(euler[:,0] + euler[:,2], label="φ+ψ")
    plt.legend(loc="upper left")
    plt.title("Running time {0:.1f} s".format(totalTime(result)))
    plt.ylabel("Angle/2π radians")
    if show:
        plt.show()

def gaussian(x, mu, A, sig):
    return A * np.exp(-(x-mu)**2/(2*sig**2)) / (sqrt(2*np.pi)*sig)

def erf(x, mu, A, sig):
    return A * scipy.special.erf((x-mu)/(sqrt(2)*sig)) + A

def __fitTwist(L, angles, fitfn):
    return curve_fit(fitfn, np.arange(0,L), angles[:,0] + angles[:,2], p0=(L/2, 1, 5))

pointsPerRod = 5 # for smoother plot of fit

def fitEvolution(L, results, fitfn, values=False):
    """Fits to the profiles specified in results at different points of time.

    Returns the fitting parameters and optionally the fitted profile as well.
    """

    params = []
    for angles in results["angles"]:
        params.append(__fitTwist(L, angles, fitfn))
    if values:
        fits = []
        for (p, _) in params:
            fits.append(fitfn(np.arange(0,L,1/pointsPerRod), *p))
        return (params, fits)
    else:
        return params

def derivative(a):
    return np.append(np.insert(a[2:] - a[:-2], 0, [0]), 0)

def areas(results):
    pp = pprint.PrettyPrinter()
    table = [["tsteps", "twist-area", "dtwist/dx-area"]]
    for (tstep, res) in zip(results["tsteps"], results["angles"]):
        twist = res[:,0] + res[:,2]
        table.append([tstep, np.sum(twist), np.sum(derivative(twist))])
    pp.pprint(table)
    return table[1:]

def plotEvolution(results, show=True, fits=None):
    """Make a plot of angles as a function of x (rod number) at different times.

    Use case: after using relaxationProtocol.
    """
    if fits != None:
        L = len(fits[0])/pointsPerRod
        for fit in fits:
            plt.plot(np.arange(0,L,1/pointsPerRod), fit)
    for (tstep, res) in zip(results["tsteps"], results["angles"]):
        plt.plot(res[:,0]+res[:,2], label=str(tstep))
    plt.legend(loc="upper left")
    plt.title("Running time {0:.1f} s".format(totalTime(results)))
    plt.ylabel("Angle/2π radians")
    if show:
        plt.show()

def diffusionSigma(logt, D, power):
    return power*(log10(2*D) + logt)

def fitSigma(params, tsteps, show=True):
    """Checks if the scale factor depends as a power law on time.
    """
    sig = np.array([p[0][2] for p in params])
    plt.scatter(log10(tsteps), log10(sig))
    popt, pcov = curve_fit(diffusionSigma, log10(tsteps), log10(sig))
    D, power = popt[0], popt[1]
    deltaD, deltaPower = sqrt(pcov[0][0]), sqrt(pcov[1][1])
    print("D = {0:.2f} +- {1:.2f}, pow = {2:.2f} +- {3:.2f}".format(
        D, deltaD, power, deltaPower))
    if show:
        plt.plot(log10(tsteps), diffusionSigma(log10(tsteps), *popt),
                 label=("σ = (2Dt)^{2:.3f}, D = {0:.2f}+-{1:.2f}".format(D, deltaD, power)))
        plt.legend(loc="upper left")
        plt.xlabel("log(t) (t in simulation steps)")
        plt.ylabel("log(σ) (σ in number of rods)")
        plt.show()
    return ((D, deltaD), (power, deltaPower))
