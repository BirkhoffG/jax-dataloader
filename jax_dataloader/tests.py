# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/tests.ipynb.

# %% ../nbs/tests.ipynb 2
from __future__ import print_function, division, annotations
from .imports import *
from .datasets import ArrayDataset
import jax_dataloader as jdl

# %% auto 0
__all__ = ['test_shuffle_reproducible', 'test_dataloader']

# %% ../nbs/tests.ipynb 3
def get_batch(batch):
    if isinstance(batch, dict):
        return batch['feats'], batch['labels']
    else:
        return batch

# %% ../nbs/tests.ipynb 4
def test_no_shuffle(cls, ds, batch_size: int, feats, labels):
    dl = cls(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    assert len(dl) == len(feats) // batch_size + bool(len(feats) % batch_size)
    for _ in range(2):
        X_list, Y_list = [], []
        for batch in dl:
            x, y = get_batch(batch)
            X_list.append(x)
            Y_list.append(y)
        _X, _Y = map(jnp.concatenate, (X_list, Y_list))
        assert jnp.array_equal(_X, feats)
        assert jnp.array_equal(_Y, labels)

# %% ../nbs/tests.ipynb 5
def test_no_shuffle_drop_last(cls, ds, batch_size: int, feats, labels):
    dl = cls(ds, batch_size=batch_size, shuffle=False, drop_last=True)
    assert len(dl) == len(feats) // batch_size
    for _ in range(2):
        X_list, Y_list = [], []
        for batch in dl:
            x, y = get_batch(batch)
            X_list.append(x)
            Y_list.append(y)
        _X, _Y = map(jnp.concatenate, (X_list, Y_list))
        last_idx = len(X_list) * batch_size
        assert jnp.array_equal(_X, feats[: last_idx])
        assert jnp.array_equal(_Y, labels[: last_idx])

# %% ../nbs/tests.ipynb 6
def test_shuffle(cls, ds, batch_size: int, feats, labels):
    dl = cls(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    last_X, last_Y = jnp.array([]), jnp.array([])
    assert len(dl) == len(feats) // batch_size + bool(len(feats) % batch_size)
    for _ in range(2):
        X_list, Y_list = [], []
        for batch in dl:
            x, y = get_batch(batch)
            assert jnp.array_equal(x[:, :1], y)
            X_list.append(x)
            Y_list.append(y)
        _X, _Y = map(jnp.concatenate, (X_list, Y_list))
        assert not jnp.array_equal(_X, feats)
        assert not jnp.array_equal(_Y, labels)
        assert jnp.sum(_X) == jnp.sum(feats), \
            f"jnp.sum(_X)={jnp.sum(_X)}, jnp.sum(feats)={jnp.sum(feats)}"
        assert not jnp.array_equal(_X, last_X)
        assert not jnp.array_equal(_Y, last_Y)
        last_X, last_Y = _X, _Y

# %% ../nbs/tests.ipynb 7
def test_shuffle_drop_last(cls, ds, batch_size: int, feats, labels):
    dl = cls(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    assert len(dl) == len(feats) // batch_size
    for _ in range(2):
        X_list, Y_list = [], []
        for batch in dl:
            x, y = get_batch(batch)
            assert jnp.array_equal(x[:, :1], y)
            X_list.append(x)
            Y_list.append(y)
        _X, _Y = map(jnp.concatenate, (X_list, Y_list))
        assert not jnp.array_equal(_X, feats)
        assert not jnp.array_equal(_Y, labels)
        assert len(_X) == len(X_list) * batch_size

# %% ../nbs/tests.ipynb 8
def test_shuffle_reproducible(cls, ds, batch_size: int, feats, labels):
    """Test that the shuffle is reproducible"""
    def _iter_dataloader(dataloader):
        X_list, Y_list = [], []
        for batch in dataloader:
            x, y = get_batch(batch)
            X_list.append(x)
            Y_list.append(y)
        return X_list, Y_list

    # Test that the shuffle is reproducible
    jdl.manual_seed(0)
    dl_1 = cls(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    X_list_1, Y_list_1 = _iter_dataloader(dl_1)
    dl_2 = cls(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    X_list_2, Y_list_2 = _iter_dataloader(dl_2)
    assert jnp.array_equal(jnp.concatenate(X_list_1), jnp.concatenate(X_list_2))

    # Test that the shuffle is different if the seed is different
    jdl.manual_seed(1234)
    dl_3 = cls(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    X_list_3, Y_list_3 = _iter_dataloader(dl_3)
    assert not jnp.array_equal(jnp.concatenate(X_list_1), jnp.concatenate(X_list_3))

# %% ../nbs/tests.ipynb 9
def test_dataloader(cls, ds_type='jax', samples=1000, batch_size=12):
    feats = np.arange(samples).repeat(10).reshape(samples, 10)
    labels = np.arange(samples).reshape(samples, 1)

    if ds_type == 'jax':
        ds = ArrayDataset(feats, labels)
    elif ds_type == 'torch':
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(feats), torch.from_numpy(labels))
    elif ds_type == 'tf':
        ds = tf.data.Dataset.from_tensor_slices((feats, labels))
    elif ds_type == "hf":
        ds = hf_datasets.Dataset.from_dict({"feats": feats, "labels": labels})
    else:
        raise ValueError(f"Unknown ds_type: {ds_type}")
    
    test_no_shuffle(cls, ds, batch_size, feats, labels)
    test_no_shuffle_drop_last(cls, ds, batch_size, feats, labels)
    test_shuffle(cls, ds, batch_size, feats, labels)
    test_shuffle_drop_last(cls, ds, batch_size, feats, labels)
    test_shuffle_reproducible(cls, ds, batch_size, feats, labels)
