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

#-------------------#
# Utility functions #
#-------------------#

def toListlike(results):
    tmp = copy.deepcopy(results)
    for (key, val) in tmp.items():
        if type(val) is np.ndarray:
            tmp[key] = val.tolist()
    return tmp


def totalTime(result):
    return result["timing"]["Total time"]


def savedata(results, fname):
    tmp = toListlike(results)
    if fname.endswith(".pckl"):
        with open(fname, "wb") as f:
            pickle.dump(tmp, f)
    elif fname.endswith(".json"):
        with open(fname, "w") as f:
            json.dump(tmp, f)
    else:
        raise ValueError("Unrecognized file extension. Use .json or .pckl.")

#--------------------------#
# Key simulation functions #
#--------------------------#

def test(n=256, L=32, mcSteps=20, step_size=np.pi/32, nsamples=1):
    """Twisting a DNA from one end.

    Returns a DNA object in the twisted form and a dictionary containing
    relevant parameters.
    """
    dna = dnaMC.NakedDNA(L=L)
    result = dna.torsionProtocol(twists = step_size * np.arange(1, n+1, 1),
                                 mcSteps=mcSteps, nsamples=nsamples)
    return (dna, result)

def testFineSampling(L=32, mcSteps=100):
    """Skips sampling for some steps initially and then does fine sampling.

    Use-case: collecting better data (say for visualization) and skip the boring
    initial part.
    """
    step_size = np.pi/32
    sampling_start_twist_per_rod = np.pi * 75 / 180
    pre_sampling_steps = int(sampling_start_twist_per_rod * L / step_size)

    dna = dnaMC.NakedDNA(L=L)
    res = dna.torsionProtocol(
        twists = step_size * np.arange(1, pre_sampling_steps + 1, 1),
        mcSteps=mcSteps//2
    )

    max_twist_per_rod = 90 * np.pi / 180
    # written this way so it is easier to change the 90 value to something else
    total_steps = int(max_twist_per_rod * L / step_size)

    result = dna.torsionProtocol(
        twists = step_size * np.arange(pre_sampling_steps, total_steps + 1, 1),
        mcSteps=mcSteps,
        nsamples=mcSteps
    )

    for k, v in res["timing"].items():
        result["timing"][k] += v

    return (dna, result)


def testNucleosome(n=256, L=32, mcSteps=20, step_size=np.pi/32, nsamples=1, nucpos=[16]):
    dna = dnaMC.NucleosomeArray(L=L, nucleosomePos=np.array(nucpos))
    results = dna.torsionProtocol(twists = step_size * np.arange(1, n+1, 1),
                                  mcSteps=mcSteps, nsamples=nsamples)
    return (dna, results)


def testNucleosomeArray(protocol, T=293.15, nucArrayType="standard",
                        nucleosomeCount=36, basePairsPerRod=10,
                        linker=60, spacer=600, **protocol_kwargs):
    """Simulate a nucleosome array.

    ``protocol`` should be one of 'twist', 'relax' or 'config'.
    If protocol is 'config', then ``protocol_kwargs`` should be empty.
    Otherwise, see the kwargs for torsionProtocol/relaxationProtocol.
    Other arguments are explained under ``dnaMC.NucleosomeArray.create``.
    """
    dna = dnaMC.NucleosomeArray.create(
        nucArrayType=nucArrayType,
        nucleosomeCount=nucleosomeCount,
        basePairsPerRod=basePairsPerRod,
        linker=linker,
        spacer=spacer
    )
    dna.env.T = T
    if protocol == "twist":
        results = dna.torsionProtocol(**protocol_kwargs)
    elif protocol == "relax":
        results = dna.relaxationProtocol(**protocol_kwargs)
    elif protocol == "config":
        if protocol_kwargs != {}:
            raise ValueError("Unexpected kwargs. Did you intend to use the "
                             "'twist' or 'relax' protocols?")
        results = {
            "angles": np.array([dna.euler]),
            "nucleosome": np.array(dna.nuc),
            "totalLength": dna.strandLength,
            "rodLength": dna.d,
        }
    else:
        raise ValueError("The first argument protocol must be one of 'twist',"
                         " 'relax' or 'config'.")
    return (dna, results)


def testDiffusion(initialFn, L=32, T=dnaMC.Environment.roomTemp, height=np.pi/4,
                  mcSteps=100, nsamples=4):
    """Testing for diffusion in DNA using a delta or a step profile initially.

    height should be small compared to 2*pi.
    """
    dna = dnaMC.NakedDNA(L=L, T=T)
    if initialFn == "delta":
        dna.euler[L//2, 2] = height
    elif initialFn == "step":
        dna.euler[L//2:, 2] = height
    else:
        raise ValueError("The first argument initialFn should be either 'delta'"
                         " or 'step'.")
    results = dna.relaxationProtocol(mcSteps=mcSteps, nsamples=nsamples)
    return (dna, results)

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

def _fitTwist(L, angles, fitfn):
    return curve_fit(fitfn, np.arange(0,L), angles[:,0] + angles[:,2], p0=(L/2, 1, 5))

pointsPerRod = 5 # for smoother plot of fit

def fitEvolution(L, results, fitfn, values=False):
    """Fits to the profiles specified in results at different points of time.

    Returns the fitting parameters and optionally the fitted profile as well.
    """

    params = []
    for angles in results["angles"]:
        params.append(_fitTwist(L, angles, fitfn))
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
