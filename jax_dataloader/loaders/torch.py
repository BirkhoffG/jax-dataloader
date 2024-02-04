# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/loader.torch.ipynb.

# %% ../../nbs/loader.torch.ipynb 2
from __future__ import print_function, division, annotations
from ..imports import *
from . import BaseDataLoader
from ..datasets import Dataset, ArrayDataset, JAXDataset
from ..utils import check_pytorch_installed
from ..tests import *
from jax.tree_util import tree_map
import warnings


# %% auto 0
__all__ = ['to_torch_dataset', 'DataLoaderPytorch']

# %% ../../nbs/loader.torch.ipynb 5
# adapted from https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
def _numpy_collate(batch):
  return tree_map(np.asarray, torch_data.default_collate(batch))

# %% ../../nbs/loader.torch.ipynb 6
@dispatch
def to_torch_dataset(dataset: JAXDataset) -> torch_data.Dataset:
    class DatasetPytorch(torch_data.Dataset):
        def __init__(self, dataset: Dataset): self.dataset = dataset
        def __len__(self): return len(self.dataset)
        def __getitem__(self, idx): return self.dataset[idx]
    
    return DatasetPytorch(dataset)

@dispatch
def to_torch_dataset(dataset: TorchDataset):
    return dataset

@dispatch
def to_torch_dataset(dataset: HFDataset):
    return dataset.with_format("numpy")

# %% ../../nbs/loader.torch.ipynb 7
class DataLoaderPytorch(BaseDataLoader):
    """Pytorch Dataloader"""
    
    @typecheck
    def __init__(
        self, 
        dataset: Union[JAXDataset, TorchDataset, HFDataset],
        batch_size: int = 1,  # Batch size
        shuffle: bool = False,  # If true, dataloader shuffles before sampling each batch
        drop_last: bool = False, # Drop last batch or not
        **kwargs
    ):
        super().__init__(dataset, batch_size, shuffle, drop_last)
        check_pytorch_installed()
        from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

        if 'sampler' in kwargs:
            warnings.warn("`sampler` is currently not supported. We will ignore it and use `shuffle` instead.")
            del kwargs['sampler']

        dataset = to_torch_dataset(dataset)
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)

        self.dataloader = torch_data.DataLoader(
            dataset, 
            batch_sampler=batch_sampler,
            # batch_size=batch_size, 
            # shuffle=shuffle, 
            # drop_last=drop_last,
            collate_fn=_numpy_collate,
            **kwargs
        )

    def __len__(self):
        return len(self.dataloader)

    def __next__(self):
        return next(self.dataloader)

    def __iter__(self):
        return self.dataloader.__iter__()
