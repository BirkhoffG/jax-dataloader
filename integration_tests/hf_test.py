import jax_dataloader as jdl
import numpy as np
import datasets as hfds


def test_hf():
    ds = hfds.Dataset.from_dict({"feats": np.ones((10, 3)), "labels": np.ones((10, 3))})
    dl = jdl.DataLoader(ds, 'jax', batch_size=2)
    batch = next(iter(dl))
    for batch in dl:
        x, y = batch['feats'], batch['labels']
        z = x + y
        assert isinstance(z, np.ndarray)
