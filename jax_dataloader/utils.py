# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/utils.ipynb.

# %% ../nbs/utils.ipynb 3
from __future__ import print_function, division, annotations
from .imports import *
import jax_dataloader as jdl

# %% auto 0
__all__ = ['Config', 'get_config', 'PRNGSequence', 'check_pytorch_installed', 'has_pytorch_tensor', 'check_hf_installed',
           'check_tf_installed', 'is_hf_dataset', 'is_torch_dataset', 'is_jdl_dataset', 'is_tf_dataset']

# %% ../nbs/utils.ipynb 4
@dataclass
class Config:
    """Global configuration for the library"""
    rng_reserve_size: int
    global_seed: int

    @classmethod
    def default(cls) -> Config:
        return cls(rng_reserve_size=1, global_seed=42)

# %% ../nbs/utils.ipynb 5
main_config = Config.default()

# %% ../nbs/utils.ipynb 6
def get_config() -> Config:
    return main_config

# %% ../nbs/utils.ipynb 7
class PRNGSequence(Iterator[PRNGKey]):
    """An Interator of Jax PRNGKey (minimal version of `haiku.PRNGSequence`)."""

    def __init__(self, seed: int):
        self._key = jax.random.PRNGKey(seed)
        self._subkeys = collections.deque()

    def reserve(self, num):
        """Splits additional ``num`` keys for later use."""
        if num > 0:
            new_keys = tuple(jax.random.split(self._key, num + 1))
            self._key = new_keys[0]
            self._subkeys.extend(new_keys[1:])
            
    def __next__(self):
        if not self._subkeys:
            self.reserve(get_config().rng_reserve_size)
        return self._subkeys.popleft()

# %% ../nbs/utils.ipynb 9
def check_pytorch_installed():
    if torch_data is None:
        raise ModuleNotFoundError("`pytorch` library needs to be installed. "
            "Try `pip install torch`. Please refer to pytorch documentation for details: "
            "https://pytorch.org/get-started/.")


# %% ../nbs/utils.ipynb 10
def has_pytorch_tensor(batch) -> bool:
    if isinstance(batch[0], torch.Tensor):
        return True
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return any([has_pytorch_tensor(samples) for samples in transposed])
    else:
        return False

# %% ../nbs/utils.ipynb 11
def check_hf_installed():
    if hf_datasets is None:
        raise ModuleNotFoundError("`datasets` library needs to be installed. "
            "Try `pip install datasets`. Please refer to huggingface documentation for details: "
            "https://huggingface.co/docs/datasets/installation.html.")

# %% ../nbs/utils.ipynb 12
def check_tf_installed():
    if tf is None:
        raise ModuleNotFoundError("`tensorflow` library needs to be installed. "
            "Try `pip install tensorflow`. Please refer to tensorflow documentation for details: "
            "https://www.tensorflow.org/install/pip.")

# %% ../nbs/utils.ipynb 13
def is_hf_dataset(dataset):
    return hf_datasets and (
        isinstance(dataset, hf_datasets.Dataset) 
        or isinstance(dataset, hf_datasets.DatasetDict)
    )


# %% ../nbs/utils.ipynb 14
def is_torch_dataset(dataset):
    return torch_data and isinstance(dataset, torch_data.Dataset)

# %% ../nbs/utils.ipynb 15
def is_jdl_dataset(dataset):
    return isinstance(dataset, jdl.Dataset)

# %% ../nbs/utils.ipynb 16
def is_tf_dataset(dataset):
    return tf and isinstance(dataset, tf.data.Dataset)