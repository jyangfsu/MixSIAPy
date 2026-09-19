"""Minimal WSL2/JAX GPU validation without model sampling."""
import jax
import jax.numpy as jnp
import mixsiapy
import numpy
import numpyro
import pymc
import blackjax
import nutpie
import arviz
import xarray
from mixsiapy import backend_status

print("jax", jax.__version__)
print("numpy", numpy.__version__)
print("pymc", pymc.__version__)
print("numpyro", numpyro.__version__)
print("blackjax", blackjax.__version__)
print("nutpie", nutpie.__version__)
print("arviz", arviz.__version__)
print("xarray", xarray.__version__)
print("backends", backend_status())
print("devices", jax.devices())
x = jnp.ones((1024, 1024))
y = (x @ x).block_until_ready()
print("compute_device", y.device)
print("mixsiapy", mixsiapy.__file__)
