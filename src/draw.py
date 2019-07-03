import utils as gu

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr


def hat_curve(dataset):
    """Draws a curve for z_bead vs turns."""
    twists = dataset["twists"]
    tsteps = dataset["tsteps"]
    tslice = slice(0, tsteps.size, tsteps.size // twists.size)
    # FIXME: Add special case when 'run' is not present; there's only 1 run
    try:
        z_mean, z_stdev = gu.mean_std(
            dataset["extension"].isel(tsteps=tslice).sel(axis='z'), dim='run')
    except ValueError:
        z_mean = dataset["extension"].isel(tsteps=tslice).sel(axis='z')
        z_stdev = z_mean.copy()
        # print(z_stdev)
        z_stdev.values = np.zeros_like(z_stdev)

    def f(x):
        tmp = x.rename({"tsteps": "twists"})
        tmp["twists"] = twists
        return tmp

    z_mean = f(z_mean).to_dataset().rename({"extension": "mean"})
    z_stdev = f(z_stdev).to_dataset().rename({"extension": "stdev"})

    z = xr.merge([z_mean, z_stdev])
    fig, axis = plt.subplots()
    axis.errorbar(
        z["twists"].values / (2 * np.pi),
        z["mean"].values,
        yerr=z["stdev"].values,
        label='z')
    plt.legend(loc='upper left')
    plt.show(block=False)
