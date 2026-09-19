"""Short end-to-end PyMC -> NumPyro -> CUDA NUTS validation."""
import jax
import numpy as np
import pymc as pm

print("devices", jax.devices())
with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    pm.Normal("y", mu=mu, sigma=1, observed=np.array([-0.2, 0.1, 0.3]))
    idata = pm.sample(
        draws=100,
        tune=100,
        chains=2,
        random_seed=20260816,
        progressbar=False,
        nuts_sampler="numpyro",
        nuts_sampler_kwargs={"chain_method": "vectorized"},
    )

print("posterior_mean", float(idata.posterior["mu"].mean()))
print("divergences", int(idata.sample_stats["diverging"].sum()))
print("compute_device", jax.devices()[0])
