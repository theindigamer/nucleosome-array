import dnaMC
import utils

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import log10, sqrt
import pandas as pd
import scipy
from scipy.optimize import curve_fit
import xarray as xr

import joblib
import json
import pickle

import copy
import itertools
import pprint

#------------------#
# Common constants #
#------------------#

ANGLES_STR = ["φ", "θ", "ψ"]

def mean_std(x, **kwargs):
    return (x.mean(**kwargs), x.std(**kwargs))

def _norm(da, dim='axis'):
    """Computes the Euclidean norm of a DataArray along a dimension."""
    return ((da**2).sum(dim=dim))**0.5

#-------------------#
# Utility functions #
#-------------------#

def _toListlike(results):
    tmp = copy.deepcopy(results)
    for (key, val) in tmp.items():
        if type(val) is np.ndarray:
            tmp[key] = val.tolist()
    return tmp


def total_time(result):
    return result["timing"][..., result["timing_keys"] == "Total time"]

# TODO: fix this function to work with datasets
def save_data(results, fname):
    tmp = _toListlike(results)
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

def run_sim(parallel, runs, f, *args, **kwargs):
    if parallel:
        tmp = joblib.Parallel(n_jobs=runs)(joblib.delayed(f)(*args, **kwargs)
                                           for _ in range(runs))
    else:
        tmp = [f(*args, **kwargs) for _ in range(runs)]
    if isinstance(tmp[0], tuple):
        flag = True
        dnas, results = list(zip(*tmp))
    else:
        flag = False
        results = tmp
    results = utils.concat_datasets(
        results, ["angles", "extension", "energy", "acceptance", "timing"],
        ["run"], [np.arange(runs)])
    if flag:
        return dnas, results
    else:
        return results


def relax_rods1(L=3, rod_len=5, mcSteps=10000, nsamples=10000,
                T=dnaMC.Environment.ROOM_TEMP, force=0.,
                kickSizes=[[0., 0.1, 0.], [0., 0.3, 0.], [0., 0.5, 0.]],
                B=43.0):
    results = []
    for ks in kickSizes:
        dna = dnaMC.NakedDNA(L=L, T=T, kickSize=np.array(ks),
                             strand_len=rod_len * L, B=B)
        result = dna.relaxationProtocol(
            force=force, mcSteps=mcSteps, nsamples=nsamples)
        results.append(result)
    results = utils.concat_datasets(
        results, ["angles", "extension", "energy", "acceptance", "timing"],
        ["kickSize"], [np.array(kickSizes)[:, 1]])
    return results


def relax_rods(runs=10, **kwargs):
    return run_sim(True, runs, relax_rods1, **kwargs)


def simulate_dna1(n=128, L=32, mcSteps=20, step_size=np.pi/16, nsamples=1,
                  T=0., kickSize=dnaMC.Simulation.DEFAULT_KICK_SIZE,
                  dnaClass=dnaMC.NakedDNA):
    """Twisting a DNA with one end."""
    dna = dnaClass(L=L, T=T, kickSize=kickSize)
    result = dna.torsionProtocol(twists = step_size * np.arange(1, n+1, 1),
                                 mcSteps=mcSteps, nsamples=nsamples)
    return (dna, result)

def simulate_dna(runs=5, **kwargs):
    """Twisting a DNA from one end (multiple runs).

    Args:
        runs (int): Number of runs for the simulation.
        kwargs: See simulate_dna1.

    Returns:
        A list of DNA strands in the final state, and the combined results of
        the multiple simulations in one dataset.
    """
    return run_sim(runs, simulate_dna1, **kwargs)


def simulate_dna_fine_sampling(L=32, mcSteps=100, dnaClass=dnaMC.NakedDNA):
    """Skips sampling for some steps initially and then does fine sampling.

    Use-case: collecting better data (say for visualization) and skip the boring
    initial part.
    """
    step_size = np.pi/32
    sampling_start_twist_per_rod = np.pi * 75 / 180
    pre_sampling_steps = int(sampling_start_twist_per_rod * L / step_size)

    dna = dnaClass(L=L)
    res = dna.torsionProtocol(
        twists = step_size * np.arange(1, pre_sampling_steps + 1, 1),
        mcSteps = mcSteps//2)

    max_twist_per_rod = 90 * np.pi / 180
    # written this way so it is easier to change the 90 value to something else
    total_steps = int(max_twist_per_rod * L / step_size)

    result = dna.torsionProtocol(
        twists = step_size * np.arange(pre_sampling_steps, total_steps + 1, 1),
        mcSteps=mcSteps, nsamples=mcSteps)

    for k, v in res["timing"].items():
        result["timing"][k] += v

    return (dna, result)


# def simulate_nucleosome(n=256, L=32, mcSteps=20, step_size=np.pi/32, nsamples=1, nucpos=[16]):
#     dna = dnaMC.NucleosomeArray(L=L, nucPos=np.array(nucpos))
#     results = dna.torsionProtocol(twists = step_size * np.arange(1, n+1, 1),
#                                   mcSteps=mcSteps, nsamples=nsamples)
#     return (dna, results)

def simulate_nuc_array(protocol, T=293.15, nucArrayType="standard",
                        nucleosomeCount=36, basePairsPerRod=10,
                        linker=60, spacer=600, **protocol_kwargs):
    """Simulate a nucleosome array.

    `protocol` should be one of 'twist', 'relax' or 'config'.
    If protocol is 'config', then `protocol_kwargs` should be empty.
    Otherwise, see the kwargs for torsionProtocol/relaxationProtocol.
    Other arguments are explained under `dnaMC.NucleosomeArray.create`.
    """
    dna = dnaMC.NucleosomeArray.create(
        nucArrayType=nucArrayType, nucleosomeCount=nucleosomeCount,
        basePairsPerRod=basePairsPerRod, linker=linker, spacer=spacer)
    dna.env.T = T
    if protocol == "twist":
        results = dna.torsionProtocol(**protocol_kwargs)
    elif protocol == "relax":
        results = dna.relaxationProtocol(**protocol_kwargs)
    elif protocol == "config":
        if protocol_kwargs:
            raise ValueError("Unexpected kwargs. Did you intend to use the "
                             "'twist' or 'relax' protocol?")
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


def simulate_diffusion1(initialFn, L=32, T=dnaMC.Environment.ROOM_TEMP,
                       height=np.pi/4, mcSteps=100, nsamples=4,
                       dnaClass=dnaMC.NakedDNA,
                       kickSize=dnaMC.Simulation.DEFAULT_KICK_SIZE):
    """Testing for diffusion in DNA using a delta or a step profile initially.

    initialFn should be one of "delta" or "step".

    height should be small compared to 2*pi.
    """
    dna = dnaClass(L=L, T=T, kickSize=kickSize)
    if initialFn == "delta":
        dna.euler[L//2, 2] = height
    elif initialFn == "step":
        dna.euler[L//2:, 2] = height
    else:
        raise ValueError("The first argument initialFn should be either 'delta'"
                         " or 'step'.")
    results = dna.relaxationProtocol(mcSteps=mcSteps, nsamples=nsamples)
    return (dna, results)


def simulate_diffusion(*args, runs=5, **kwargs):
    return run_sim(runs, simulate_diffusion1, *args, **kwargs)


def dna_check_acceptance(Ts, kickSizes, *args, mode="product", **kwargs):
    """Check kick acceptance rates for different kick sizes and temperatures.

    Ts and kickSizes are nonempty lists of temperatures and kick sizes to try.

    If mode is "product", all possible combinations of the two are used.
    If mode is "zip", the two lists are zipped and used.

    args and kwargs are for the simulate_diffusion function.
    """
    if mode == "product":
        T_ks = itertools.product(Ts, kickSizes)
    elif mode == "zip":
        if len(Ts) != len(kickSizes):
            raise ValueError(
                "The temperature and kick size lists have mismatched sizes."
                " Did you intend to use mode='product' instead?")
        T_ks = zip(Ts, kickSizes)
    else:
        raise ValueError("Unrecognized value of mode."
                         " Recognized values are 'product' and 'zip'.")
    results = []
    for (T, ks) in T_ks:
        _, res = simulate_diffusion(*args, T=T, kickSize=ks, **kwargs)
        results.append(res.copy())
    results = utils.concat_datasets(
        results, ["angles", "extension", "energy", "acceptance", "timing"],
        ["temperature", "kickSize"], [Ts, kickSizes])
    return results


def marko_siggia_curve(B, strandLength):
    # B is expressed in nm kT so Lp = B/kT -> Lp = B
    kT = 1
    Lp = B
    L0 = strandLength
    def f(x):
        return (kT/Lp) * (1 / (4 * (1 - x/L0)**2) - 1/4 + x/L0)
    xs = np.arange(0, strandLength, 5)
    return (xs, f(xs))


def compute_extension1(forces=np.arange(0, 10, 1), kickSizes=[0.1, 0.3, 0.5],
                       disordered=True, demo=False, Pinv=1/150):
    """Compute force vs extension and optionally acceptance vs force.

    ``forces`` is some nonempty iterable with the desired force values to use.
    Similarly for ``kickSizes``. If ``kickSizes`` contains more than one
    element, multiple graphs are draw side-by-side.
    ``disordered`` creates "disordered" DNA, i.e., the zeros of bending energy
    are randomly shifted from zero physical bend.
    ``demo`` is provided for quickly debugging the drawing code without
    worrying about the actual physical values.
    """
    L = 128
    B = 35.0
    kickSizes_arr = np.array(kickSizes)
    forces_arr = np.array(forces)

    if demo:
        pre_steps = 10
        extra_steps = 10
        nsamples = 2
    else:
        pre_steps = 1000
        extra_steps = 1000
        nsamples = 100
    if disordered:
        dnaClass = dnaMC.DisorderedNakedDNA
        opt_kwargs = {'Pinv': Pinv}
    else:
        Pinv = 0
        dnaClass = dnaMC.NakedDNA
        opt_kwargs = {}

    dnas = []
    datasets = []
    i = 0
    tot = kickSizes_arr.size * forces_arr.size
    print(' {0} out of {1}'.format(i, tot), end='', flush=True)
    for (kickSize, force) in itertools.product(kickSizes_arr, forces_arr):
        dna = dnaClass(L=L, kickSize=kickSize, B=B,
                       T=dnaMC.Environment.ROOM_TEMP, **opt_kwargs)
        _ = dna.relaxationProtocol(force=force, mcSteps=pre_steps, nsamples=1)
        datasets.append(dna.relaxationProtocol(force=force, mcSteps=extra_steps,
                                               nsamples=nsamples))
        datasets[-1]["tsteps"] += pre_steps
        dnas.append(dna)
        i += 1
        print('\x1b[0G {0} out of {1}'.format(i, tot), end='', flush=True)
    print("")

    results = utils.concat_datasets(
        datasets, ["angles", "extension", "energy", "acceptance", "timing"],
        ["kickSize", "force"], [kickSizes_arr, forces_arr])
    results.attrs.update({
        "pre_steps": pre_steps,
        "extra_steps": extra_steps,
        "nsamples": nsamples,
    })

    return (dnas, results)


def compute_extension(runs=5, **kwargs):
    return run_sim(runs, compute_extension1, **kwargs)


def draw_force_extension(dataset, acceptance=True):
    kickSizes = dataset.coords["kickSize"].values
    forces = dataset.coords["force"].values
    runs = dataset.coords["run"].values
    B = dataset.attrs["B"]
    L = dataset.attrs["rodCount"]
    T = dataset.attrs["temperature"]
    Pinv = dataset.data_vars["Pinv"].values
    pre_steps = dataset.attrs["pre_steps"]
    extra_steps = dataset.attrs["extra_steps"]
    nsamples = dataset.attrs["nsamples"]

    fig, axes = plt.subplots(
        nrows=(2 if acceptance else 1), ncols=kickSizes.size,
        sharex="row", sharey="row", squeeze=False)
    sns.set_style("darkgrid")
    ms_curve_x, ms_curve_y = marko_siggia_curve(B, 740)

    for (j, ks) in enumerate(kickSizes):
        # 'axis' dimension is the last dimension
        # there doesn't seem to be a simple way to broadcast np.linalg.norm
        tmp = _norm(dataset["extension"].sel(kickSize=ks), dim='axis')
        mean, stdev = mean_std(tmp.groupby("force"))
        print(mean.values)
        print(forces)
        axes[0, j].errorbar(mean.values, forces, xerr=stdev.values,
                            capsize=4.0, linestyle='')
        axes[0, j].plot(ms_curve_x, ms_curve_y)
        axes[0, j].set_xlim(right=740)
        axes[0, j].set_ylim(bottom=-0.5, top=10+0.5)
        axes[0, j].set_title("kick size = {0}".format(ks))
        axes[0, j].set_xlabel("")
        axes[0, j].set_ylabel("")
    axes[0, 0].set_ylabel("Force (pN)")
    axes[0, kickSizes.size//2].set_xlabel("Extension (nm)")

    # Molecular dynamics simulation will not have acceptance values.
    acceptance = acceptance and ("acceptance" in dataset)
    if acceptance:
        for (j, ks) in enumerate(kickSizes):
            tmp = dataset["acceptance"].sel(kickSize=ks)
            for (j_a, angle) in enumerate(ANGLES_STR):
                mean, stdev = mean_std(tmp.isel(angle_str=j_a).groupby("force"))
                axes[1, j].errorbar(forces, mean.values, yerr=stdev.values,
                                    label=angle)
                axes[1, j].set_ylim(bottom=0, top=1)
            axes[1, 0].set_ylabel("Acceptance ratio")
            axes[1, kickSizes.size//2].set_xlabel("Force (pN)")

    fig.suptitle(
        ("Contour length (straight) L_0 = 740 nm, #rods = {6}, T = {7:.0f} K\n"
         "Effective persistence length Lp = {5:.1f} nm, "
         "Intrinsic disorder persistence length P = {0:.1f} nm\n"
         "Thermalization steps = {1}, extra steps = {2}, "
         "#samples in extra steps = {3}, runs = {4}").format(
             1/Pinv if Pinv != 0 else np.inf, pre_steps,
             extra_steps, nsamples, runs, B, L, T),
        fontdict = {"fontsize": 10})
    plt.legend(loc="upper right")
    plt.show(block=False)


#---------------------#
# Naked DNA evolution #
#---------------------#

def gaussian(x, mu, A, sig):
    return A * np.exp(-(x-mu)**2/(2*sig**2)) / (sqrt(2*np.pi)*sig)

def erf(x, mu, A, sig):
    return A * scipy.special.erf((x-mu)/(sqrt(2)*sig)) + A

def _fitTwist(L, angles, fitfn):
    return curve_fit(fitfn, np.arange(0,L), angles[:,0] + angles[:,2], p0=(L/2, 1, 5))

POINTS_PER_ROD = 5 # for smoother plot of fit

# TODO: fix this function to work with datasets
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
            fits.append(fitfn(np.arange(0,L,1/POINTS_PER_ROD), *p))
        return (params, fits)
    else:
        return params

def derivative(a):
    return np.append(np.insert(a[2:] - a[:-2], 0, [0]), 0)

# TODO: fix this function to work with datasets
def areas(results):
    pp = pprint.PrettyPrinter()
    table = [["tsteps", "twist-area", "dtwist/dx-area"]]
    for (tstep, res) in zip(results["tsteps"], results["angles"]):
        twist = res[:,0] + res[:,2]
        table.append([tstep, np.sum(twist), np.sum(derivative(twist))])
    pp.pprint(table)
    return table[1:]

#--------------------#
# Plotting functions #
#--------------------#

from scipy.stats import norm

def draw_angle_probability(dataset, angle_str="theta", run=0):
    if angle_str == "theta":
        expected_std = np.sqrt(dataset.attrs["rodLength"]/(2 * dataset.attrs["B"]))
    elif angle_str == "psi":
        expected_std = np.sqrt(dataset.attrs["rodLength"]/(2 * dataset.attrs["C"]))
    else:
        raise ValueError("Unrecognized string.")
    # if run == "avg":
    #     tmp = dataset["angles"].sel(angle_str=angle_str, n=1).mean(dim='run')
    # else:
    tmp = dataset["angles"].sel(angle_str=angle_str, n=1, run=run)
    tmp_std = tmp.std(dim='tsteps')
    for ks in dataset["kickSize"]:
        sns.distplot(
            tmp.sel(kickSize=ks),
            fit=norm,
            kde=False,
            label="kick={0:.1f} rad, std={1:.3f}".format(
                float(ks), float(tmp_std.sel(kickSize=ks)))
        )
    plt.title("Expected std = {0:.3f}".format(expected_std))
    plt.legend(loc="upper right")
    plt.show(block=False)


def draw_binned_bend_autocorr(dataset):
    # NOTE: There is a factor of 2 when only θ is changing and other two angles
    # are fixed.
    expected_p = 2 * dataset.attrs["B"]
    L = dataset.attrs["rodCount"]
    x = dataset.attrs["rodLength"] * np.arange(L)
    # TODO: Replace this with correct correlation function, which will not
    # be just a pair of exponentials, especially for a small value of L.
    def f(x, p, strand_len, L):
        return np.concatenate((
            np.exp(-x[:L//2 + 1]/p),
            np.exp(-(strand_len-x[L//2 + 1:])/p)
        ))
    y = f(x, expected_p, dataset.attrs["strandLength"], L)
    t = dataset['tsteps']
    start = 0
    step = t[-1] // 5
    stop = t[-1] + step
    tmp = (dataset["bend_autocorr"]
           .isel(kickSize=0, run=slice(min(5, dataset["run"][-1])))
           .groupby_bins('tsteps', np.arange(start, stop, step))
           .mean(dim='tsteps')
           .to_dataframe())
    tmp.reset_index(level=tmp.index.names, inplace=True)
    tmp['n'] = tmp['n'] * dataset.attrs["rodLength"]
    g = sns.FacetGrid(tmp, col='tsteps_bins', row='run')
    g = g.map(plt.plot, 'n', 'bend_autocorr')
    for ax in np.reshape(g.axes, (-1,)):
        ax.plot(x, y, color='green', label="Naive")
    plt.show(g)
    return tmp


def draw_bend_autocorr(dataset, energy=False):
    fig, axes = plt.subplots(
        nrows=2 if energy else 1, ncols = len(dataset["run"]),
        sharey='row', squeeze=False)
    # NOTE: There is a factor of 2 when only θ is changing and other two angles
    # are fixed.
    expected_p = 2 * dataset.attrs["B"]
    L = dataset.attrs["rodCount"]
    x = dataset.attrs["rodLength"] * np.arange(L)
    # TODO: Replace this with correct correlation function, which will not
    # be just a pair of exponentials, especially for a small value of L.
    def f(x, p, strand_len, L):
        return np.concatenate((
            np.exp(-x[:L//2 + 1]/p),
            np.exp(-(strand_len-x[L//2 + 1:])/p)
        ))
    y = f(x, expected_p, dataset.attrs["strandLength"], L)
    for (i, ax) in enumerate(axes[0]):
        for ks in dataset["kickSize"]:
            tmp = (dataset["bend_autocorr"]
                   .sel(run=i, kickSize=ks, tsteps=slice(500, None)))
            tmp_mean, tmp_std = mean_std(tmp, dim='tsteps')
            ax.errorbar(x, tmp_mean.values, # yerr=tmp_std.values,
                        capsize=2.0, label=str(ks.values))
            ax.set_yscale('log')
        ax.plot(x, y, label="Naive")
        ax.set_title("Run# = {0}".format(i))
        ax.legend(loc="lower right")
    axes[0][0].set_ylabel("log(Tangent vector autocorrelation)")
    axes[0][len(axes[0])//2].set_xlabel("Length (nm)")
    if energy:
        for (i, ax) in enumerate(axes[1]):
            for ks in dataset["kickSize"]:
                draw_energy_autocorr(
                    dataset.sel(kickSize=ks).isel(run=slice(i, i+1)), axis=ax)
        axes[1][0].set_ylabel("Energy autocorrelation")
        axes[1][len(axes[1])//2].set_xlabel("Time")
    fig.suptitle(
        "#rods={1}, Correlation function averaged over t=500 to t={0}".format(
            int(dataset["tsteps"][-1]), dataset.attrs["rodCount"]))
    plt.show(block=False)
    return (fig, axes)


# TODO: fix this function to work with datasets
def plot_angles(dna, result, totalOnly=True, show=True):
    """Make a plot of angles as a function of x (rod number).

    Use case: after using twistProtocol on the dna object.
    """
    euler = dna.euler/(2*np.pi)
    if not totalOnly:
        plt.plot(euler[:,0], label=ANGLES_STR[0])
        plt.plot(euler[:,1], label=ANGLES_STR[1])
        plt.plot(euler[:,2], label=ANGLES_STR[2])
    plt.plot(euler[:,0] + euler[:,2], label=(ANGLES_STR[0] + "+" + ANGLES_STR[2]))
    plt.legend(loc="upper left")
    plt.title("Running time {0:.1f} s".format(total_time(result)))
    plt.ylabel("Angle/2π radians")
    if show:
        plt.show()

def draw_energy(dataset, axis=None, show=None):
    flag = axis is None
    if flag:
        fig, axis = plt.subplots()
    # else:
    #     new_axis = axis
    tsteps = dataset["tsteps"]
    mean, stdev = mean_std(dataset["energy"].sel(tsteps=tsteps), dim=['run'])
    if len(np.shape(mean)) != 1:
        print("WARNING: draw_energy encountered unexpected dimensions"
              " after averaging over runs.")
    axis.errorbar(tsteps.values, mean.values, yerr=stdev.values, capsize=2.)
    if flag:
        return (fig, axis)
    return axis


def draw_angle_profile(dataset, axis=None, total_only=False, show=None):

    def tsteps_slice(ndraw):
        nonlocal dataset
        n = len(dataset["tsteps"])
        if ndraw > n:
            ndraw = n
            print("WARNING: attempting to draw at more points than sampled.")
        return dataset["tsteps"][::n//ndraw]

    if axis is None:
        fig, new_axis = plt.subplots()
    else:
        new_axis = axis

    if show is None:
        ndraw = 5
        tsteps = tsteps_slice(ndraw)
    elif isinstance(show, int):
        ndraw = show
        tsteps = tsteps_slice(ndraw)
    elif show == "all":
        tsteps = dataset["tsteps"]
    else:
        raise ValueError("show should be one of None, int (> 0) or the string 'all'.")

    mean, stdev = mean_std(dataset["angles"].sel(tsteps=tsteps)/(2*np.pi), dim=['run'])
    if len(np.shape(mean)) != 3: # one for time, one for rods, one for 3 angles
        print("WARNING: draw_angle_profile encountered unexpected dimensions"
              " after averaging over runs.")
    x = np.arange(dataset.attrs["rodCount"])
    # if not total_only:
    #     for (i, a) in enumerate(ANGLES_STR):
    #         new_axis.errorbar(x, mean.isel(angle_str=i),
    #                           yerr=stdev.isel(angle_str=i), label=a)
    # TODO: replace phi + psi (as a proxy) with actual twist value
    # ? How to compute total twist from phi, theta, psi values
    for tstep in tsteps:
        tmp_mean, tmp_std = mean.sel(tsteps=tstep), stdev.sel(tsteps=tstep)
        new_axis.plot(
            x, tmp_mean.isel(angle_str=0) + tmp_mean.isel(angle_str=2),
            # yerr=(tmp_std.isel(angle_str=0)**2 + tmp_std.isel(angle_str=2)**2)**0.5,
            label=("t = " + str(tstep.values)))
    new_axis.legend(loc="upper left")
    new_axis.set_ylabel("Angle/2π radians")
    new_axis.set_xlabel("Rod number")
    if axis is None:
        return (fig, new_axis)
    else:
        return axis


def draw_energy_autocorr(dataset, axis=None):
    if axis is None:
        fig, new_axis = plt.subplots()
    else:
        new_axis = axis
    tsteps = dataset["tsteps"]
    en = dataset["energy"].values
    en_k = np.fft.fft(en, axis=-1)
    en_corr = np.fft.ifft((en_k * en_k.conj()), axis=-1).real
    corr_ds = xr.DataArray(en_corr, coords=dataset["energy"].coords,
                           attrs=dataset["energy"].attrs)
    # TODO: StackOverflow - ? easier way to cast fft over dataset.
    mean, stdev = mean_std(corr_ds, dim=['run'])
    # TODO: add fitting+plotting code to determine+display correlation time
    if len(np.shape(mean)) != 1:
        print("WARNING: draw_energy encountered unexpected dimensions"
              " after averaging over runs.")
    # new_axis.plot(np.log10(tsteps.values), np.log10(mean.values))
    new_axis.errorbar(tsteps.values, mean.values,# yerr=stdev.values,
                      capsize=2.)
    if axis is None:
        return (fig, new_axis)
    else:
        return axis


def draw_acceptance(dataset, axis=None):
    if axis is None:
        fig, new_axis = plt.subplots()
    else:
        new_axis = axis
    for (i, a) in enumerate(ANGLES_STR):
        mean, stdev = mean_std(dataset["acceptance"].isel(angle_str=i),
                               dim=["run"])
        new_axis.errorbar(
            dataset["tsteps"].values, mean.values, yerr=stdev.values, label=a)
    new_axis.legend(loc="upper right")
    new_axis.set_xlabel("Time (Monte Carlo steps)")
    new_axis.set_ylabel("Acceptance probability")
    if axis is None:
        return (fig, new_axis)
    else:
        return axis


def draw_diffusion(dataset):
    # nrows = int(np.sqrt(len(plots)))
    # ncols = nrows if plots % nrows == 0 else nrows + 1
    fig, ax = plt.subplots(nrows=2, ncols=3, squeeze=False)
    draw_angle_profile(dataset, ax[0][1])
    # draw_bend_twist(dataset, ax[0][2])
    # draw_sigma_fit(dataset, ax[0][0])
    draw_acceptance(dataset, ax[1][0])
    draw_energy(dataset, ax[1][1])
    draw_energy_autocorr(dataset, ax[1][2])

# TODO: fix this function to work with datasets
def plot_evolution(results, show=True, fits=None):
    """Make a plot of angles as a function of x (rod number) at different times.

    Use case: after using relaxationProtocol.
    """
    if fits != None:
        L = len(fits[0])/POINTS_PER_ROD
        for fit in fits:
            plt.plot(np.arange(0,L,1/POINTS_PER_ROD), fit)
    for (tstep, res) in zip(results["tsteps"], results["angles"]):
        plt.plot(res[:,0]+res[:,2], label=str(tstep))
    plt.legend(loc="upper left")
    plt.title("Running time {0:.1f} s".format(total_time(results)))
    plt.ylabel("Angle/2π radians")
    if show:
        plt.show()

def diffusionSigma(logt, D, power):
    return power*(log10(2*D) + logt)


# TODO: Show a plot of log log plot of sigma vs t and compute D.
def draw_sigma_fit(dataset, axis=None):
    raise NotImplementedError()


# TODO: Show a plot of β and Γ values.
def draw_bend_twist(dataset, axis=None):
    raise NotImplementedError()


def fitSigma(params, full_tsteps, show=True, start=0):
    """Checks if the scale factor depends as a power law on time."""
    tsteps = full_tsteps[start:]
    sig = np.array([p[0][2] for p in params])[start:]
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
