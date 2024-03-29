import jax_dataloader as jdl
import jax.numpy as jnp
import pytest


def test_jax():
    ds = jdl.ArrayDataset(jnp.ones((10, 3)), jnp.ones((10, 3)))
    assert len(ds) == 10
    dl = jdl.DataLoader(ds, 'jax', batch_size=2)
    batch = next(iter(dl))
    for x, y in dl:
        z = x + y
  

def test_torch():
    with pytest.raises(ModuleNotFoundError):
        ds = jdl.ArrayDataset(jnp.ones((10, 3)), jnp.ones((10, 3)))
        dl = jdl.DataLoader(ds, 'pytorch', batch_size=2)
        batch = next(iter(dl))
        for x, y in dl: z = x + y


def test_tf():
    with pytest.raises(ModuleNotFoundError):
        ds = jdl.ArrayDataset(jnp.ones((10, 3)), jnp.ones((10, 3)))
        dl = jdl.DataLoader(ds, 'tensorflow', batch_size=2)
        batch = next(iter(dl))
        for x, y in dl: z = x + y

