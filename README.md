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

### A warning w.r.t. Numba

Unlike `numpy`, which has bounds checking on by default,
`numba` doesn't offer bounds checking as of version 0.36,
although negative indices do work as expected.
So whenever you are tweaking `numba`-related code,
it might be best to first comment out the `@jit` decorator(s),
make sure the changes work as expected, and then de-comment it.

## Bibliography

The source code comments sometimes refer to papers for values etc.
The bibliography for those references is given here for easy cross-verification.

[B] Bancaud et al. - Structural plasticity of single chromatin fibers revealed
by torsional manipulation

[DS] Daniels and Sethna - Nucleation at the DNA Supercoiling transition

[F] Forth et al. - Abrupt Buckling Transition Observed during the Plectoneme
Formation of Individual DNA Molecules

## Boundary conditions

The internal representation of the angles works with hinges, not rods.
However, it is easier to think of numbering in terms of rods.
We have `L` rods between the base and the bead,
so the angles are indexed `[0, .., L-1]`.
We have two additional sets of angles `start` (`s`) and `end` (`e`).
These are held fixed when the DNA relaxes.
Typically we would want `φ_s = φ_e`, `θ_s = θ_e`, and `ψ_s = 0`.

* The `0`-th element represents the point where the DNA attaches to the base.
* The `L-1`-th element represents the "bottom" of the last rod attached to the bead.
* The `start` element represents the "bottom" of a rod embedded in the base,
  so that we can account for the bending energy between `start` and `0`.
* The `end` element represent the point where the DNA attaches to the bead.
  So if we want to twist the DNA, we can change `ψ_e`.

The `0`-th rod is only free to bend, but not to twist.
So `ψ_0 = ψ_s` whereas `φ_0` and `θ_0` evolve with time.

The `L-1`-th rod is only free to twist, but not to bend.
So `φ_{L-1} = φ_e` and `θ_{L-1} = θ_e` whereas `ψ_{L-1}` evolves with time.

If we look at the energies, there are `L+1` bending/twisting energy terms,
one for each hinge (numbered `0 ↔ start~0`, `1 ↔ 0~1`, ..., `L ↔ L-1~end`).
However, there are `L` stretching energy terms, one for each rod.

For a given angle, only `L-1` terms can be changed in `euler`, which will affect
`L` out of the `L+1` possible terms. This is tabulated below:

```
| Angle changed | E_bend   | E_twist  | E_stretch |
|---------------|----------|----------|-----------|
| φ_0           | 0, 1     | 0, 1     | -         |
| θ_0           | 0, 1     | 0, 1     | 0         |
| ψ_1           | -        | 1, 2     | -         |
| φ_1           | 1, 2     | 1, 2     | -         |
| θ_1           | 1, 2     | 1, 2     | 1         |
| ψ_2           | -        | 1, 2     | -         |
| ...           | ...      | ...      | ...       |
| φ_{L-2}       | L-2, L-1 | L-2, L-1 | -         |
| θ_{L-2}       | L-2, L-1 | L-2, L-1 | L-2       |
| ψ_{L-1}       | -        | L-1, L   | -         |
```
