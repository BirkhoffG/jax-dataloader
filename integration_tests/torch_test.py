import jax_dataloader as jdl
import torch, jax
from torch.utils.data import TensorDataset


def test_jax():
    ds = TensorDataset(torch.ones((10, 3)), torch.ones((10, 3)))
    assert len(ds) == 10
    dl = jdl.DataLoader(ds, 'jax', batch_size=2)
    for x, y in dl:
        z = x + y
  

def test_torch():
    ds = TensorDataset(torch.ones((10, 3)), torch.ones((10, 3)))    
    dl = jdl.DataLoader(ds, 'pytorch', batch_size=2)
    for x, y in dl: 
        z = x + y
        assert isinstance(z, jax.Array)

