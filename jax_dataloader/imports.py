from __future__ import annotations
import numpy as np
from typing import (
    List,
    Tuple,
    Dict,
    Any,
    Optional,
    Iterable,
    Sequence,
    Iterator,
    Literal,
    Union,
    Annotated,
)
import jax
from jax import vmap, grad, jit, numpy as jnp, random as jrand
from abc import ABC
from dataclasses import dataclass
from plum import dispatch
from beartype.vale import Is
from beartype.door import is_bearable
from beartype import beartype as typecheck

try:
    import torch.utils.data as torch_data
    import torch

    TorchDataset = torch_data.Dataset
except ModuleNotFoundError:
    torch_data = None
    torch = None
    TorchDataset = Annotated[None, Is[lambda _: torch_data is not None]]

try:
    import datasets as hf_datasets

    HFDataset = Annotated[
        Union[
            hf_datasets.Dataset,
            hf_datasets.DatasetDict,
            hf_datasets.IterableDatasetDict,
            hf_datasets.IterableDataset,
        ],
        Is[lambda _: hf_datasets is not None],
    ]
except ModuleNotFoundError:
    hf_datasets = None
    HFDataset = Annotated[None, Is[lambda _: hf_datasets is not None]]

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds

    # TFDataset = Annotated[
    #     tf.data.Dataset,
    #     Is[lambda _: tf is not None],
    # ]
    TFDataset = tf.data.Dataset
except ModuleNotFoundError:
    tf = None
    tfds = None
    TFDataset = Annotated[None, Is[lambda _: tf is not None]]

try:
    import haiku as hk
except ModuleNotFoundError:
    hk = None
