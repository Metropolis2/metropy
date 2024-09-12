import os
import time

import numpy as np
import polars as pl
from scipy.special import binom

import metropy.utils.io as metro_io

np.seterr("raise")


def rlaptrans(n, func, rng, tol=1e-7, x0=1, xinc=2, m=11, L=1, A=19, nburn=38, **kwargs):
    maxiter = 500
    # Derived quantities that need only be calculated once, including the binomial coefficients.
    nterms = nburn + m * L
    seqbtL = np.arange(nburn - 1, nterms, L)
    y = np.pi * 1j * np.arange(1, nterms + 1) / L
    expy = np.exp(y)
    A2L = 0.5 * A / L
    expxt = np.exp(A2L) / L
    coef = binom(m, np.arange(m + 1)) / 2**m
    u = rng.uniform(size=n)
    # Generate sorted uniform random numbers.
    u.sort()
    # `xrand` will store the corresponding x values
    xrand = np.empty(n, dtype=np.float64)
    # Begin by finding an x-value that can act as an upper bound throughout. This will be stored in
    # `upplim`. Its value is based on the maximum value in u. We also use the first value calculated
    # (along with its pdf and cdf) as a starting value for finding the solution to F(x) = u_min.
    # (This is used only once, so doesn't need to be a good starting value.)
    t = x0 / xinc
    cdf = 0
    kount0 = 0
    set1st = False
    while kount0 < maxiter and cdf < u[n - 1]:
        t *= xinc
        kount0 += 1
        pdf, cdf = get_pdf_cdf(A2L, t, y, func, expy, expxt, coef, seqbtL, **kwargs)
        if not set1st and cdf > u[0]:
            cdf1 = cdf
            pdf1 = pdf
            t1 = t
            set1st = True
    if kount0 >= maxiter:
        raise Exception("Cannot locate upper quantile")
    upplim = t
    # Now use modified Newton-Raphson.
    lower = 0
    t = t1
    cdf = cdf1
    pdf = pdf1
    kount = np.zeros(n)
    maxiter = 1000
    for j in range(n):
        # Initial bracketing of solution.
        upper = upplim
        kount[j] = 0
        while kount[j] < maxiter and abs(u[j] - cdf) > tol:
            kount[j] += 1
            # Update t. Try Newton-Raphson approach. If this goes outside the bounds, use midpoint
            # instead.
            t -= (cdf - u[j]) / pdf
            if t < lower or t > upper:
                t = 0.5 * (lower + upper)
            # Calculate the cdf and pdf at the updated value of t.
            pdf, cdf = get_pdf_cdf(A2L, t, y, func, expy, expxt, coef, seqbtL, **kwargs)
            # Update the bounds.
            if cdf <= u[j]:
                lower = t
            else:
                upper = t
        if kount[j] >= maxiter:
            print("Warning: Desired accuracy not achieved for F(x)=u")
        xrand[j] = t
        lower = t
    rng.shuffle(xrand)
    return xrand


def get_pdf_cdf(A2L, t, y, func, expy, expxt, coef, seqbtL, **kwargs):
    x = A2L / t
    z = x + y / t
    ltx = func(x, **kwargs)
    ltzexpy = func(z, **kwargs) * expy
    par_sum = 0.5 * np.real(ltx) + np.cumsum(np.real(ltzexpy))
    par_sum2 = 0.5 * np.real(ltx / x) + np.cumsum(np.real(ltzexpy / z))
    pdf = expxt * np.sum(coef * par_sum[seqbtL]) / t
    cdf = expxt * np.sum(coef * par_sum2[seqbtL]) / t
    return pdf, cdf


def read_agent_ids(directory: str):
    lf = metro_io.scan_dataframe(os.path.join(directory, "trips.parquet"))
    cols = lf.collect_schema().names()
    if "tour_id" in cols:
        agent_ids = lf.select(pl.col("tour_id").alias("agent_id")).unique().collect().to_series()
    else:
        agent_ids = lf.select("agent_id").unique().collect().to_series()
    return agent_ids


def get_nest_map(nests):
    nest_map = dict()
    for nest_idx, nest in enumerate(nests):
        for mode in nest:
            nest_map[mode] = nest_idx
    return nest_map


def positive_stable_distribution_laplace(s, l=1.0):
    return np.exp(-(s**l))


def draw_epsilons(
    agent_ids: pl.Series, mu: float, nests: list[list[str]], lambdas: list[float], random_seed=None
):
    rng = np.random.default_rng(random_seed)
    assert len(nests) == len(
        lambdas
    ), "Invalid nests: the number of lambdas is not equal to the number of nests"
    nest_map = get_nest_map(nests)
    nb_agents = len(agent_ids)
    nb_modes = len(nest_map)
    # Draw mode-specific gumbel epsilons.
    e = rng.gumbel(scale=mu, size=(nb_agents, nb_modes))
    # Draw nest-specific epsilons.
    j = 0
    for i, nest in enumerate(nests):
        print(nest)
        nest_log_z = np.log(
            rlaptrans(nb_agents, positive_stable_distribution_laplace, rng, l=lambdas[i])
        )
        for _ in range(len(nest)):
            e[:, j] += nest_log_z
            e[:, j] *= lambdas[i]
            j += 1
    epsilons = pl.DataFrame(e, schema=sum(nests, start=[]))
    epsilons = epsilons.with_columns(agent_ids.alias("agent_id"))
    return epsilons


if __name__ == "__main__":
    from metropy.config import read_config, check_keys

    config = read_config()
    mandatory_keys = [
        "population_directory",
        "demand.epsilons.nests",
        "demand.epsilons.lambdas",
    ]
    check_keys(config, mandatory_keys)

    t0 = time.time()
    random_seed = config.get("random_seed")

    agent_ids = read_agent_ids(config["population_directory"])

    epsilons = draw_epsilons(
        agent_ids,
        config["demand"]["epsilons"]["mu"],
        config["demand"]["epsilons"]["nests"],
        config["demand"]["epsilons"]["lambdas"],
        random_seed,
    )

    metro_io.save_dataframe(
        epsilons, os.path.join(config["population_directory"], "mode_epsilons.parquet")
    )
    t = time.time() - t0
    print("Total running time: {:.2f} seconds".format(t))
