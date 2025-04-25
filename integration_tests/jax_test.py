import jax_dataloader as jdl
import jax.numpy as jnp
import pytest
import jax.random as jrand
import jax.tree_util as jtu


def test_jax():
    ds = jdl.ArrayDataset(jnp.ones((10, 3)), jnp.ones((10, 3)))
    assert len(ds) == 10
    dl = jdl.DataLoader(ds, 'jax', batch_size=2)
    batch = next(iter(dl))
    for x, y in dl:
        z = x + y
  

def test_torch():
    with pytest.raises(ModuleNotFoundError):
        ds = jdl.ArrayDataset(jnp.ones((10, 3)), jnp.ones((10, 3)))
        dl = jdl.DataLoader(ds, 'pytorch', batch_size=2)
        batch = next(iter(dl))
        for x, y in dl: z = x + y


def test_tf():
    with pytest.raises(ModuleNotFoundError):
        ds = jdl.ArrayDataset(jnp.ones((10, 3)), jnp.ones((10, 3)))
        dl = jdl.DataLoader(ds, 'tensorflow', batch_size=2)
        batch = next(iter(dl))
        for x, y in dl: z = x + y

def test_generator():
    ds = jdl.ArrayDataset(jnp.ones((10, 3)), jnp.ones((10, 3)))

    g1 = jdl.Generator()
    g2 = jrand.PRNGKey(jdl.get_config().global_seed)

    # Create two different dataloaders with different generators
    dl = jdl.DataLoader(ds, 'jax', batch_size=2, generator=g1, shuffle=True)
    batch = next(iter(dl))

    dl = jdl.DataLoader(ds, 'jax', batch_size=2, generator=g2, shuffle=True)
    new_batch = next(iter(dl))
    
    # Check that batches are equal using tree_map
    def are_equal(a, b):
        return jnp.all(a == b)
    
    # Map the equality function over the entire pytree structure
    equal_elements = jtu.tree_map(are_equal, batch, new_batch)
    
    # Check all elements are True
    all_equal = all(jtu.tree_leaves(equal_elements))
    assert all_equal
    
    # Also verify the tree structures match
    assert jtu.tree_structure(batch) == jtu.tree_structure(new_batch)
