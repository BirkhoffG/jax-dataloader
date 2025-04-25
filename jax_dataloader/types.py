from .imports import *
from .utils import Generator

GeneratorType = Union[Generator, jax.Array, 'torch.Generator']