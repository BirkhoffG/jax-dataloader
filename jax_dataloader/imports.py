from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Iterable, Sequence, Iterator
import jax
from jax import vmap, grad, jit, numpy as jnp, random as jrand
from abc import ABC
from dataclasses import dataclass

try: 
    import torch.utils.data as torch_data
    import torch
except ModuleNotFoundError: 
    torch_data = None
    torch = None

try: import haiku as hk 
except ModuleNotFoundError: hk = None

try: import datasets as hf_datasets
except ModuleNotFoundError: hf_datasets = None

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ModuleNotFoundError:
    tf = None
    tfds = None

