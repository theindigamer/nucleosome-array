There are three branches you might be interested in -

* `master` contains some extra Mathematica code + data files for
  visualization -- these are not present in the other branches.

* `fix-mc-boundary` - It has not been
  merged into `master`, as we were mid-refactoring before leaving the project.

* `danilo` - contains latest version of code for molecular dynamics

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

There is an additional package [`nbstripout`][nbstripout] which is useful for
working with Jupyter notebooks. Before using `git commit`, you can run the
`backup_nb.py` file (it is executable) so that it will generate new copies
of notebooks without output. These can be put in version control without
large file sizes. For some reason, the `conda` version of `nbstripout`
doesn't seem to be working properly, so you should install it using `pip`.

```
pip install nbstripout
# if that doesn't work, try sudo -H pip install nbstripout
```

Tests can be run by `cd`-ing into `src` and running `pytest`.

[numba]: http://numba.pydata.org/
[seaborn]: http://seaborn.pydata.org/
[xarray]: http://xray.readthedocs.io/en/stable/index.html
[hypothesis]: http://hypothesis.works/
[pytest]: https://docs.pytest.org/en/latest/index.html
[conda]: https://github.com/conda/conda
[nbstripout]: https://github.com/kynan/nbstripout

## Bibliography

The source code comments sometimes refer to papers for values etc.
The bibliography for those references is given here for easy cross-verification.

[B] Bancaud et al. - Structural plasticity of single chromatin fibers revealed
by torsional manipulation

[DS] Daniels and Sethna - Nucleation at the DNA Supercoiling transition

[F] Forth et al. - Abrupt Buckling Transition Observed during the Plectoneme
Formation of Individual DNA Molecules
