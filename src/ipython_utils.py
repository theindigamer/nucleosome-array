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
    for (key, val) in tmp.items():
        if type(val) is np.ndarray:
            tmp[key] = val.tolist()
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
        raise ValueError("Unrecognized file extension. Use .json or .pckl.")

def test(n=256, L=32, mcSteps=20, step_size=np.pi/32, nsamples=1):
    """Twisting a DNA from one end.

    Returns a DNA object in the twisted form and a dictionary containing
    relevant parameters.
    """
    dna = dnaMC.nakedDNA(L=L)
    result = dna.torsionProtocol(twists = step_size * np.arange(1, n+1, 1),
                                 mcSteps=mcSteps, nsamples=nsamples)
    return (dna, result)

def testFineSampling(L=32, mcSteps=100):
    step_size = np.pi/32
    sampling_start_twist_per_rod = np.pi * 75 / 180
    pre_sampling_steps = int(sampling_start_twist_per_rod * L / step_size)

    dna = dnaMC.nakedDNA(L=L)
    res = dna.torsionProtocol(twists = step_size * np.arange(1, pre_sampling_steps + 1, 1), mcSteps=mcSteps//2)

    max_twist_per_rod = np.pi * 90 / 180 # np.pi/2
    total_steps = int(max_twist_per_rod * L / step_size)

    result = dna.torsionProtocol(twists = step_size * np.arange(pre_sampling_steps, total_steps + 1, 1),
                                 mcSteps=mcSteps,
                                 nsamples=mcSteps
    )

    for k,v in res["timing"].items():
        result["timing"][k] += v

    return (dna, result)


def testNucleosome(n=256, L=32, mcSteps=20, step_size=np.pi/32, nsamples=1, nucpos=[16]):
    dna = dnaMC.nucleosomeArray(L=L, nucleosomePos=np.array(nucpos))
    results = dna.torsionProtocol(twists = step_size * np.arange(1, n+1, 1),
                                  mcSteps=mcSteps, nsamples=nsamples)
    return (dna, results)

def standardNucleosomeArray(nucleosomeCount=36, basePairsPerRod=10):
    #     60 bp between cores
    #           |-|           600 bp spacers on either side
    # ~~~~~~~~~~O~O~O...O~O~O~~~~~~~~~~
    basePairArray = [180] + ([60] * (nucleosomeCount - 1)) + [180]
    # basePairArray = [600] + ([60] * (nucleosomeCount - 1)) + [600]
    basePairLength = 0.34 # in nm.
    totalLength = float(np.sum(basePairArray) * basePairLength)
    numRods = np.array(basePairArray) // basePairsPerRod
    L = int(np.sum(numRods))
    nucleosomePos = np.cumsum(numRods)[:-1]
    dna = dnaMC.nucleosomeArray(L=L, nucleosomePos=nucleosomePos,
                                strandLength=totalLength)
    return dna

def relaxedNucleosomeArray(nucleosomeCount=36, basePairsPerRod=10):
    # all local bends and twists are set to zero here
    basePairArray = [180] + ([60] * (nucleosomeCount - 1)) + [180]
    # basePairArray = [600] + ([60] * (nucleosomeCount - 1)) + [600]
    basePairLength = 0.34 # in nm.
    totalLength = float(np.sum(basePairArray) * basePairLength)
    numRods = np.array(basePairArray) // basePairsPerRod
    L = int(np.sum(numRods))
    nucleosomePos = np.cumsum(numRods)[:-1]
    dna = dnaMC.nucleosomeArray(L=L, nucleosomePos=nucleosomePos,
                                strandLength=totalLength)
    # nakedDNA.__init__(self, L=L, B=B, C=C)
    prev = np.array([0., 0., 0.])
    for i in range(L):
        if nucleosomePos.size != 0 and i == nucleosomePos[0]:
            next = dnaMC.exitAngles(prev)
            dna.euler[i] = copy.copy(next)
            prev = next
            nucleosomePos = nucleosomePos[1:]
        else:
            dna.euler[i] = copy.copy(prev)
        # print(prev)
    return dna

def testNucleosomeArray(n=256, mcSteps=100, step_size=np.pi/32, nsamples=1,
                        nucleosomeCount=36, basePairsPerRod=10):
    dna = standardNucleosomeArray(nucleosomeCount=nucleosomeCount,
                                  basePairsPerRod=basePairsPerRod)
    results = dna.torsionProtocol(twists = step_size * np.arange(1, n+1, 1),
                                  mcSteps=mcSteps, nsamples=nsamples)
    results["rodLength"] = dna.d # in nm
    results["totalLength"] = dna.strandLength # in nm
    return (dna, results)

def testNucleosomeArrayRelax(mcSteps=1000, nsamples=1,
                             nucleosomeCount=36, basePairsPerRod=10, force=1.96):
    dna = standardNucleosomeArray(nucleosomeCount=nucleosomeCount,
                                  basePairsPerRod=basePairsPerRod)
    results = dna.relaxationProtocol(mcSteps=mcSteps, nsamples=nsamples,
                                     force=force)
    results["rodLength"] = dna.d # in nm
    results["totalLength"] = dna.strandLength # in nm
    return (dna, results)

def nucleosomeInitialConfig(L=32, nucpos=[16]):
    dna = dnaMC.nucleosomeArray(L=L, nucleosomePos=np.array(nucpos))
    _ = dna.deltaMatrices()
    results = {}
    results["angles"] = np.array([dna.euler])
    results["nucleosome"] = np.array(nucpos)
    results["rodLength"] = dna.d
    return (dna, results)

def nucleosomeArrayInitialConfig(nucleosomeCount=36, basePairsPerRod=10, relaxed=False):
    if relaxed:
        dna = relaxedNucleosomeArray(nucleosomeCount=nucleosomeCount,
                                     basePairsPerRod=basePairsPerRod)
    else:
        dna = standardNucleosomeArray(nucleosomeCount=nucleosomeCount,
                                      basePairsPerRod=basePairsPerRod)
    results = {}
    results["angles"] = np.array([dna.euler])
    results["nucleosome"] = np.array(dna.nuc)
    results["totalLength"] = dna.strandLength # in nm
    results["rodLength"] = dna.d # in nm
    return (dna, results)

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
