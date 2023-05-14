# Jax-Dataloader

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

![Python](https://img.shields.io/pypi/pyversions/jax-dataloader.svg)
![CI
status](https://github.com/BirkhoffG/jax-dataloader/actions/workflows/nbdev.yaml/badge.svg)
![Docs](https://github.com/BirkhoffG/jax-dataloader/actions/workflows/deploy.yaml/badge.svg)
![pypi](https://img.shields.io/pypi/v/jax-dataloader.svg) ![GitHub
License](https://img.shields.io/github/license/BirkhoffG/jax-dataloader.svg)

## Overview

`jax_dataloader` provides a high-level *pytorch-like* dataloader API for
`jax`. It supports

- **downloading and pre-processing datasets** via [huggingface
  datasets](https://github.com/huggingface/datasets), [pytorch
  Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset),
  and tensorflow dataset (forthcoming)

- **iteratively loading batches** via (vanillla) [jax
  dataloader](https://birkhoffg.github.io/jax-dataloader/core.html#jax-dataloader),
  [pytorch
  dataloader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader),
  tensorflow (forthcoming), and merlin (forthcoming).

A minimum `jax-dataloader` example:

``` python
import jax_dataloader as jdl

dataloader = jdl.DataLoader(
    dataset, # Can be a jdl.Dataset or pytorch or huggingface dataset
    backend='jax', # Use 'jax' for loading data (also supports `pytorch`)
)

batch = next(iter(dataloader)) # iterate next batch
```

## Installation

The latest `jax-dataloader` release can directly be installed from PyPI:

``` sh
pip install jax-dataloader
```

or install directly from the repository:

``` sh
pip install git+https://github.com/BirkhoffG/jax-dataloader.git
```

<div>

> **Note**
>
> We will only install `jax`-related dependencies. If you wish to use
> integration of `pytorch` or huggingface `datasets`, you should try to
> manually install them, or run `pip install jax-dataloader[all]` for
> installing all the dependencies.

</div>

## Usage

[`jax_dataloader.core.DataLoader`](https://birkhoffg.github.io/jax-dataloader/core.html#dataloader)
follows similar API as the pytorch dataloader.

- The `dataset` argument takes `jax_dataloader.core.Dataset` or
  `torch.utils.data.Dataset` or (the huggingface) `datasets.Dataset` as
  an input from which to load the data.
- The `backend` argument takes `"jax"` or`"pytorch"` as an input, which
  specifies which backend dataloader to use batches.

``` python
import jax_dataloader as jdl
import jax.numpy as jnp
```

### Using [`ArrayDataset`](https://birkhoffg.github.io/jax-dataloader/dataset.html#arraydataset)

The `jax_dataloader.core.ArrayDataset` is an easy way to wrap multiple
`jax.numpy.array` into one Dataset. For example, we can create an
[`ArrayDataset`](https://birkhoffg.github.io/jax-dataloader/dataset.html#arraydataset)
as follows:

``` python
# Create features `X` and labels `y`
X = jnp.arange(100).reshape(10, 10)
y = jnp.arange(10)
# Create an `ArrayDataset`
arr_ds = jdl.ArrayDataset(X, y)
```

This `arr_ds` can be loaded by both `"jax"` and `"pytorch"` dataloaders.

``` python
# Create a `DataLoader` from the `ArrayDataset` via jax backend
dataloader = jdl.DataLoader(arr_ds, 'jax', batch_size=5, shuffle=True)
# Or we can use the pytorch backend
dataloader = jdl.DataLoader(arr_ds, 'pytorch', batch_size=5, shuffle=True)
```

### Using Pytorch Datasets

The [pytorch Dataset](https://pytorch.org/docs/stable/data.html) and its
ecosystems (e.g.,
[torchvision](https://pytorch.org/vision/stable/index.html),
[torchtext](https://pytorch.org/text/stable/index.html),
[torchaudio](https://pytorch.org/audio/stable/index.html)) supports many
built-in datasets. `jax_dataloader` supports directly passing the
pytorch Dataset.

<div>

> **Note**
>
> Unfortuantely, the [pytorch
> Dataset](https://pytorch.org/docs/stable/data.html) can only work with
> `backend=pytorch`. See the belowing example.

</div>

``` python
from torchvision.datasets import MNIST
import numpy as np
```

We load the MNIST dataset from `torchvision`. The `ToNumpy` object
transforms images to `numpy.array`.

``` python
class ToNumpy(object):
  def __call__(self, pic):
    return np.array(pic, dtype=float)
```

``` python
pt_ds = MNIST('/tmp/mnist/', download=True, transform=ToNumpy(), train=False)
```

This `pt_ds` can **only** be loaded via `"pytorch"` dataloaders.

``` python
dataloader = jdl.DataLoader(pt_ds, 'pytorch', batch_size=5, shuffle=True)
```

### Using Huggingface Datasets

The huggingface [datasets](https://github.com/huggingface/datasets) is a
morden library for downloading, pre-processing, and sharing datasets.
`jax_dataloader` supports directly passing the huggingface datasets.

``` python
from datasets import load_dataset
```

For example, We load the `"squad"` dataset from `datasets`:

``` python
hf_ds = load_dataset("squad")
```

This `hf_ds` can be loaded via `"jax"` and `"pytorch"` dataloaders.

``` python
# Create a `DataLoader` from the `datasets.Dataset` via jax backend
dataloader = jdl.DataLoader(hf_ds['train'], 'jax', batch_size=5, shuffle=True)
# Or we can use the pytorch backend
dataloader = jdl.DataLoader(hf_ds['train'], 'pytorch', batch_size=5, shuffle=True)
```
