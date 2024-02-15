import jax_dataloader as jdl
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf


def test_jax():
    ds = jdl.ArrayDataset(np.ones((10, 3)), np.ones((10, 3)))
    dl = jdl.DataLoader(ds, 'tensorflow', batch_size=2)
    for x, y in dl:
        z = x + y
        assert isinstance(z, np.ndarray)
  

def test_tf():
    ds = tf.data.Dataset.from_tensor_slices((tf.ones((10, 3)), tf.ones((10, 3))))
    dl = jdl.DataLoader(ds, 'tensorflow', batch_size=2)
    for x, y in dl: 
        z = x + y
        assert isinstance(z, np.ndarray)

