def mean_std(x, **kwargs):
    """Compute both mean and standard deviation of a DataArray.

    Provided purely for convenience, useful for making plots.
    """
    return (x.mean(**kwargs), x.std(**kwargs))

def norm(dataarray, dim='axis'):
    """Computes the Euclidean norm of a DataArray along a dimension.

    Like numpy.linalg.norm but you can use a dimension instead of an axis.
    """
    return ((dataarray**2).sum(dim=dim))**0.5
