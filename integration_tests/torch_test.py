import jax_dataloader as jdl
import torch
import numpy as np
import jax.numpy as jnp
from torch.utils.data import TensorDataset


def test_jax_ds():
    ds = jdl.ArrayDataset(jnp.ones((10, 3)), jnp.ones((10, 3)))
    assert len(ds) == 10
    dl = jdl.DataLoader(ds, 'pytorch', batch_size=2)
    for x, y in dl:
        z = x + y
  

def test_torch():
    ds = TensorDataset(torch.ones((10, 3)), torch.ones((10, 3)))    
    dl = jdl.DataLoader(ds, 'pytorch', batch_size=2)
    for x, y in dl: 
        z = x + y
        assert isinstance(z, np.ndarray)

