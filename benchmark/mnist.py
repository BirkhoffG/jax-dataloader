import jax_dataloader as jdl
import haiku as hk
import numpy as np
import jax
import jax.numpy as jnp
import optax
from functools import partial
import time

from absl import app, flags, logging

try:
    from torchvision.datasets import FashionMNIST
except ImportError:
    FashionMNIST = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    import tensorflow_datasets as tfds
except ImportError:
    tfds = None


flags.DEFINE_string("backend", "jax", "Backend to use.")
flags.DEFINE_integer("batch_size", 32, "Batch size.")

FLAGS = flags.FLAGS

class FlattenAndCast(object):
    """Converts a batch of images to numpy and flattens them."""

    def __call__(self, pic):
        return np.array(pic, dtype=float)


def net_fn(imgs: jnp.ndarray):
    B, H, W = imgs.shape
    imgs = imgs.reshape(B, H, W, 1)
    x = imgs.astype(jnp.float32) / 255
    cov = hk.Sequential(
        [
            hk.Conv2D(32, 3, 2),
            jax.nn.relu,
            hk.Conv2D(64, 3, 2),
            jax.nn.relu,
            hk.Conv2D(128, 3, 2),
            jax.nn.relu,
            hk.Flatten(),
            hk.Linear(256),
            jax.nn.relu,
            hk.Linear(10),
        ]
    )
    return cov(x)


def loss(
    params: hk.Params,
    classifier: hk.Transformed,
    imgs: jnp.ndarray,
    labels: jnp.ndarray,
):
    logits = classifier.apply(params, imgs)
    return jax.vmap(optax.softmax_cross_entropy_with_integer_labels)(
        logits, labels=labels
    ).mean()


def init():
    classifier = hk.without_apply_rng(hk.transform(net_fn))
    opt = optax.adam(1e-3)
    params = classifier.init(jax.random.PRNGKey(42), jnp.ones((FLAGS.batch_size, 28, 28)))
    opt_state = opt.init(params)
    return classifier, opt, params, opt_state


@partial(jax.jit, static_argnums=(2,3))
def update(
    params: hk.Params,
    opt_state: optax.OptState,
    classifier: hk.Transformed,
    opt: optax.GradientTransformation,
    imgs: jnp.ndarray,
    labels: jnp.ndarray
):
    grads = jax.grad(loss)(params, classifier, imgs, labels)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state


def get_img_labels(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        # print(batch[0])
        if isinstance(batch[0], dict):
            imgs, labels = batch[0]['image'], batch[0]['label']
        else:
            imgs, labels = batch
    elif isinstance(batch, dict):
        imgs, labels = batch['image'], batch['label']
    else:
        raise ValueError(f'Unknown batch type: {type(batch)}', )
    return imgs, labels


def train(
    train_ds,
    backend: str,
    batch_size: int,
    shuffle: bool = True,
    n_epochs: int = 1
):
    train_dl = jdl.DataLoader(
        train_ds, backend=backend, batch_size=batch_size, shuffle=shuffle)
    # imgs_list= []
    classifier, opt, params, opt_state = init()
    
    train_start_time = time.time()
    for i in range(n_epochs):
        epoch_start_time = time.time()
        for batch in train_dl:
            imgs, labels = get_img_labels(batch)
            params, opt_state = update(
                params, opt_state, classifier, opt, imgs, labels
            )
            # imgs_list.append(imgs)
        
        epoch_time = time.time() - epoch_start_time
        logging.info(f'Epoch {i} took {epoch_time: .3f} seconds')
        logging.info(f'Per batch: {epoch_time / len(train_dl) * 1000: .3f} ms.')
    
    train_time = time.time() - train_start_time
    logging.info(f'Training took {train_time: .3f} seconds')

    # imgs_list = jnp.concatenate(imgs_list)
    # assert imgs_list.shape == (len(train_ds), 28, 28)
    return train_time


def load_datasets():
    train_ds_torch = FashionMNIST(
    '/tmp/mnist/', download=True, transform=FlattenAndCast(), train=True)
    test_ds_torch = FashionMNIST(
        '/tmp/mnist/', download=True, transform=FlattenAndCast(), train=False)

    train_ds_jax = jdl.ArrayDataset(
        train_ds_torch.data.numpy(), train_ds_torch.targets.numpy())
    test_ds_jax = jdl.ArrayDataset(
        test_ds_torch.data.numpy(), test_ds_torch.targets.numpy())
    
    train_ds_hf = load_dataset('fashion_mnist', split='train')
    test_ds_hf = load_dataset('fashion_mnist', split='test')

    train_ds_tf = tfds.load('fashion_mnist', split='train')
    test_ds_tf = tfds.load('fashion_mnist', split='test')

    assert len(train_ds_hf) == len(train_ds_tf)
    assert len(train_ds_jax) == len(test_ds_tf)

    return train_ds_jax, test_ds_jax, train_ds_hf, test_ds_hf, train_ds_tf, test_ds_tf


def main(_):
    backend = FLAGS.backend
    batch_size = FLAGS.batch_size
    
    logging.info(f'Using backend: {backend}')
    logging.info(f'Batch size: {batch_size}')
    logging.info(f"Device: {jax.devices()}")
    
    # load datasets
    train_ds_jax, test_ds_jax, train_ds_hf, test_ds_hf, train_ds_tf, test_ds_tf = load_datasets()
    logging.info(f'Number of training examples: {len(train_ds_jax)}')

    train_time = train(train_ds_jax, backend, batch_size)
    logging.info(f'Training time: {train_time: .3f} seconds')


if __name__ == "__main__":
    app.run(main)
