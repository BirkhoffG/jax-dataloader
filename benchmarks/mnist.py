# Adapted from https://github.com/google/flax/blob/main/examples/mnist/train.py

from torchvision.datasets import FashionMNIST
from jax_dataloader.core import (
    get_backend_compatibilities,
    SUPPORTED_DATASETS,
    JAXDataset,
)
from jax_dataloader.imports import *
import jax_dataloader as jdl
import optax
import ml_collections
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import time
import rich
import einops


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = einops.rearrange(x, "b h w -> b h w 1")
        x = x / 255.0
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return x


@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, images)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def get_img_labels(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        # print(batch[0])
        if isinstance(batch[0], dict):
            imgs, labels = batch[0]["image"], batch[0]["label"]
        else:
            imgs, labels = batch
    elif isinstance(batch, dict):
        imgs, labels = batch["image"], batch["label"]
    else:
        raise ValueError(
            f"Unknown batch type: {type(batch)}",
        )
    return imgs, labels


def train_epoch(state, dataloader):
    """Train for a single epoch."""

    epoch_loss = []
    epoch_accuracy = []

    for batch in dataloader:
        images, labels = get_img_labels(batch)
        # print(images.shape, labels.shape)
        grads, loss, accuracy = apply_model(state, images, labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = np.mean(epoch_loss)
    train_accuracy = np.mean(epoch_accuracy)
    return state, train_loss, train_accuracy


def create_train_state(rng, config):
    """Creates initial `TrainState`."""
    cnn = CNN()
    params = cnn.init(rng, jnp.ones([32, 28, 28]))["params"]
    tx = optax.sgd(config.learning_rate, config.momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> train_state.TrainState:
    """Execute model training and evaluation loop.

    Args:
      config: Hyperparameter configuration for training and evaluation.
      workdir: Directory where the tensorboard summaries are written to.

    Returns:
      The train state (which includes the `.params`).
    """
    train_ds, test_ds = get_datasets(config.dataset_type)
    train_dl, test_dl = map(
        lambda ds: jdl.DataLoader(
            ds, backend=config.backend, batch_size=config.batch_size, shuffle=True
        ),
        (train_ds, test_ds),
    )
    rng = jax.random.key(0)

    summary_writer = tensorboard.SummaryWriter(workdir)
    summary_writer.hparams(dict(config))

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, config)

    runtime_per_epoch = []
    for epoch in range(1, config.num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        start = time.time()
        state, train_loss, train_accuracy = train_epoch(state, train_dl)
        runtime_per_epoch.append(time.time() - start)

        summary_writer.scalar("train_loss", train_loss, epoch)
        summary_writer.scalar("train_accuracy", train_accuracy, epoch)
        summary_writer.scalar("runtime_per_epoch", runtime_per_epoch[-1], epoch)
        # summary_writer.scalar('test_loss', test_loss, epoch)
        # summary_writer.scalar('test_accuracy', test_accuracy, epoch)

    summary_writer.flush()
    return state, runtime_per_epoch


def get_datasets(ds_type: Literal["jax", "torch", "tf", "hf"]):
    """Returns train and test datasets."""

    train_ds_torch = FashionMNIST(
        "/tmp/mnist/",
        download=True,
        transform=lambda x: np.array(x, dtype=float),
        train=True,
    )
    test_ds_torch = FashionMNIST(
        "/tmp/mnist/",
        download=True,
        transform=lambda x: np.array(x, dtype=float),
        train=False,
    )

    if ds_type == "jax":
        train_ds = jdl.ArrayDataset(
            train_ds_torch.data.numpy(), train_ds_torch.targets.numpy()
        )
        test_ds = jdl.ArrayDataset(
            test_ds_torch.data.numpy(), test_ds_torch.targets.numpy()
        )
    elif ds_type == "torch":
        train_ds, test_ds = train_ds_torch, test_ds_torch
    elif ds_type == "hf":
        train_ds = hf_datasets.load_dataset("fashion_mnist", split="train")
        test_ds = hf_datasets.load_dataset("fashion_mnist", split="test")
    elif ds_type == "tf":
        train_ds = tfds.load("fashion_mnist", split="train")
        test_ds = tfds.load("fashion_mnist", split="test")
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")
    return train_ds, test_ds


def get_config():
    config = ml_collections.ConfigDict()

    config.dataset_type = "jax"
    config.backend = "jax"
    config.batch_size = 128
    config.num_epochs = 10
    config.learning_rate = 0.1
    config.momentum = 0.9
    return config


def main():
    """Benchmark the training time for compatible backends and datasets."""

    compat = get_backend_compatibilities()
    type2ds_name = {
        JAXDataset: "jax",
        TorchDataset: "torch",
        TFDataset: "tf",
        HFDataset: "hf",
    }
    runtime = {}
    config = get_config()
    for backend, ds in compat.items():
        if len(ds) > 0:
            _supported = [s in ds for s in SUPPORTED_DATASETS]
            runtime["backend=" + backend] = {}
            config.backend = backend

            for i, ds_type in enumerate(SUPPORTED_DATASETS):
                if _supported[i]:
                    ds_name = type2ds_name[ds_type]
                    config.dataset_type = ds_name
                    _, runtime_per_epoch = train_and_evaluate(config, "/tmp/mnist")
                    runtime[backend]["dataset=" + ds_name] = runtime_per_epoch

                    rich.print(
                        f"[backend={backend}, dataset={ds_name}] Runtime per epoch: {np.mean(runtime_per_epoch)} (std={np.std(runtime_per_epoch)})."
                    )
                else:
                    runtime[backend]["dataset=" + ds_name] = []

    return runtime


if __name__ == "__main__":
    # main()
    rich.print_json(main())
