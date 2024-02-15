import jax_dataloader as jdl
import jax
import tensorflow_datasets as tfds
import tensorflow as tf


def test_jax():
    ds = tf.data.Dataset.from_tensor_slices((tf.ones((10, 3)), tf.ones((10, 3))))
    dl = jdl.DataLoader(ds, 'jax', batch_size=2)
    for x, y in dl:
        z = x + y
        assert isinstance(z, jax.Array)
  

def test_tf():
    ds = tf.data.Dataset.from_tensor_slices((tf.ones((10, 3)), tf.ones((10, 3))))
    dl = jdl.DataLoader(ds, 'tensorflow', batch_size=2)
    for x, y in dl: 
        z = x + y
        assert isinstance(z, jax.Array)

