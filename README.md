## Dependencies

The source code for the Monte Carlo uses the `@` matrix multiplication operator,
so it needs Python 3.5 or later.

Apart from the usual Python scientific suite, additional packages required
for the Monte Carlo simulation
include
[`numba`][numba],
[`seaborn`][seaborn],
[`xarray`][xarray]
and optionally
[`hypothesis`][hypothesis]
and
[`pytest`][pytest] for running tests.
These can be easily installed using [`conda`][conda]
(`numba` in particular recommends using `conda`):

```
conda install numba seaborn xarray hypothesis pytest
```

Tests can be run by `cd`-ing into `src` and running `pytest`.

[numba]: http://numba.pydata.org/
[seaborn]: http://seaborn.pydata.org/
[xarray]: http://xray.readthedocs.io/en/stable/index.html
[hypothesis]: http://hypothesis.works/
[pytest]: https://docs.pytest.org/en/latest/index.html
[conda]: https://github.com/conda/conda

## Bibliography

The source code comments sometimes refer to papers for values etc.
The bibliography for those references is given here for easy cross-verification.

[1] Daniels and Sethna - Nucleation at the DNA Supercoiling transition

[2] Bancaud et al. - Structural plasticity of single chromatin fibers revealed
by torsional manipulation
