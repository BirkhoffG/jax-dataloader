# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/loader.tf.ipynb.

# %% ../../nbs/loader.tf.ipynb 2
from __future__ import print_function, division, annotations
from ..imports import *
from . import BaseDataLoader
from ..datasets import Dataset, ArrayDataset, JAXDataset
from ..utils import check_tf_installed, get_config, Generator
from ..types import GeneratorType
from ..tests import *
from jax.tree_util import tree_map
import warnings

# %% auto 0
__all__ = ['to_tf_dataset', 'get_seed', 'DataLoaderTensorflow']

# %% ../../nbs/loader.tf.ipynb 4
@dispatch
def to_tf_dataset(dataset: JAXDataset) -> tf.data.Dataset:
    return tf.data.Dataset.from_tensor_slices(dataset[:])

@dispatch
def to_tf_dataset(dataset: TFDataset) -> tf.data.Dataset:
    return dataset

@dispatch
def to_tf_dataset(dataset: HFDataset) -> tf.data.Dataset:
    return dataset.to_tf_dataset()

# %% ../../nbs/loader.tf.ipynb 5
def get_seed(generator: Optional[Generator | jax.Array | torch.Generator] = None) -> int:
    if generator is None:
        generator = Generator()
    
    if not isinstance(generator, Generator):
        generator = Generator(generator=generator)
    
    seed = generator.seed()
    if seed is None:
        warnings.warn("No random seed provided. Using default seed which may not guarantee reproducible results.")
    return seed

class DataLoaderTensorflow(BaseDataLoader):
    """Tensorflow Dataloader"""
    
    @typecheck
    def __init__(
        self, 
        dataset: Union[JAXDataset, TFDataset, HFDataset],
        batch_size: int = 1,  # Batch size
        shuffle: bool = False,  # If true, dataloader shuffles before sampling each batch
        drop_last: bool = False, # Drop last batch or not
        generator: Optional[GeneratorType] = None, # Random seed generator
        **kwargs
    ):
        super().__init__(dataset, batch_size, shuffle, drop_last)
        check_tf_installed()
        # get random seed from generator
        seed = get_seed(generator)

        # Convert to tf dataset
        ds = to_tf_dataset(dataset)
        ds = ds.shuffle(buffer_size=len(dataset), seed=seed) if shuffle else ds
        ds = ds.batch(batch_size, drop_remainder=drop_last)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        self.dataloader = ds

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        return next(self.dataloader)

    def __iter__(self):
        return self.dataloader.as_numpy_iterator()
