## Dependencies

The source code for the Monte Carlo uses the `@` matrix multiplication operator,
so it needs Python 3.5 or later.

Apart from the usual Python scientific suite, additional packages required
include [`numba`][numba] and optionally [`hypothesis`][hypothesis] and
[`pytest`][pytest] for running tests.
These can be easily installed using [`conda`][conda]:

```
conda install numba hypothesis pytest
```

Tests can be run by `cd`-ing into `src` and running `pytest`.

[numba]: http://numba.pydata.org/
[hypothesis]: http://hypothesis.works/
[pytest]: https://docs.pytest.org/en/latest/index.html
[conda]: https://github.com/conda/conda

## Bibliography

The source code comments sometimes refer to papers for values etc.
The bibliography for those references is given here for easy cross-verification.

[1] Daniels and Sethna - Nucleation at the DNA Supercoiling transition
[2] Bancaud et al. - Structural plasticity of single chromatin fibers revealed
by torsional manipulation
