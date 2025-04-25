import jax_dataloader as jdl
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import jax.random as jrand


def test_jax():
    ds = jdl.ArrayDataset(np.ones((10, 3)), np.ones((10, 3)))
    dl = jdl.DataLoader(ds, 'tensorflow', batch_size=2)
    batch = next(iter(dl))
    for x, y in dl:
        z = x + y
        assert isinstance(z, np.ndarray)
  

def test_tf():
    ds = tf.data.Dataset.from_tensor_slices((tf.ones((10, 3)), tf.ones((10, 3))))
    dl = jdl.DataLoader(ds, 'tensorflow', batch_size=2)
    batch = next(iter(dl))
    for x, y in dl: 
        z = x + y
        assert isinstance(z, np.ndarray)

def test_generator():
    ds = jdl.ArrayDataset(np.ones((10, 3)), np.ones((10, 3)))

    g1 = jdl.Generator().manual_seed(123)
    g2 = jrand.PRNGKey(jdl.get_config().global_seed)

    # Create two different dataloaders with different generators
    dl = jdl.DataLoader(ds, 'tensorflow', batch_size=2, generator=g1, shuffle=True)
    batch = next(iter(dl))

    dl = jdl.DataLoader(ds, 'tensorflow', batch_size=2, generator=g2, shuffle=True)
    new_batch = next(iter(dl))
    # Check that batches are equal using tree_map
    def are_equal(a, b):
        return np.all(a == b)
    # Map the equality function over the entire pytree structure
    equal_elements = jdl.tree_map(are_equal, batch, new_batch)
    # Check all elements are True
    all_equal = all(jdl.tree_leaves(equal_elements))
    assert all_equal
