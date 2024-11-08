### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/generator.py ####

### NEED TO UPDATE FOR OUR CODE ###
# from our modified files
from orchard_types import SimpleOrchardApple, SimpleOrchardState

from typing import Tuple

import chex
import jax
from jax import numpy as jnp

# This replaces the RandomGenerator class
class SimpleOrchardGenerator:
    """
    Randomly generates simple orchard states in a reproduceable manner using JAX natives.
    """
    # Updated for bots, trees, apples
    # fov, max_agent_level, and force_coop were removed from the logic.
    # Why remove FOV?
    # assertion steps for checking values are removed. Why?
    def __init__(
        self,
        width: int,
        height: int,
        num_bots: int = 2,
        num_trees: int = 5,
        num_apples: int = 10
    ) -> None:
        """
        Initalizes the parameters required for the simple orchard environment
        """

        self._width = width
        self._height = height
        self._num_bots = num_bots
        self._num_trees = num_trees
        self._num_apples = num_apples

    # Updated to include getter functions for each attribute. Why?
    def get_width(self) -> int:
        return self._width

    def get_height(self) -> int:
        return self._height

    def get_num_bots(self) -> int:
        return self._num_bots

    def get_num_trees(self) -> int:
        return self._num_trees

    def get_num_apples(self) -> int:
        return self._num_apples

    # new method added for the trees attribute.
    # very similar code to the method `sample_food` with the adjustment for unique height
    # and width inputs.
    def sample_trees(self, key: chex.PRNGKey) -> chex.Array:
        """
        Randomly place trees ensuring no trees are placed on the edge and no two trees are adjacent

        :param key: The key for the deterministic random number generator

        Returns: An array containing the flat indices of the trees on the grid.
        """

        flat_size = self._width * self._height
        pos_keys = jax.random.split(key, self._num_trees)

        # Create a mask to exclude edges
        mask = jnp.ones(flat_size, dtype=bool)
        mask = mask.at[jnp.arange(self._width)].set(False)  # top
        mask = mask.at[jnp.arange(flat_size - self._width, flat_size)].set(False)  # bottom
        mask = mask.at[jnp.arange(0, flat_size, self._width)].set(False)  # left
        mask = mask.at[jnp.arange(self._width - 1, flat_size, self._width)].set(False)  # right

        def take_positions(
            mask: chex.Array, key: chex.PRNGKey
        ) -> Tuple[chex.Array, chex.Array]:
            tree_flat_pos = jax.random.choice(key=key, a=flat_size, shape=(), p=mask)

            # Mask out adjacent positions to avoid placing food items next to each other
            adj_positions = jnp.array(
                [
                    tree_flat_pos,
                    tree_flat_pos + 1,  # right
                    tree_flat_pos - 1,  # left
                    tree_flat_pos + self._width,  # up
                    tree_flat_pos - self._width,  # down
                ]
            )

            return mask.at[adj_positions].set(False), tree_flat_pos

        _, tree_flat_positions = jax.lax.scan(take_positions, mask, pos_keys)

        # Unravel indices to get the 2D coordinates (x, y)
        tree_positions_x, tree_positions_y = jnp.unravel_index(
            tree_flat_positions, (self._width, self._height)
        )
        tree_positions = jnp.stack([tree_positions_x, tree_positions_y], axis=1)

        return tree_positions

    # This replaces `sample_food` method.
    def sample_apples(self, key: chex.PRNGKey, mask: chex.Array) -> chex.Array:
        """Randomly samples apple positions on the grid, avoiding positions occupied by trees.

        Args:
            key (chex.PRNGKey): The random key.
            mask (chex.Array): The mask of the grid where 1s correspond to empty cells
            and 0s to full cells.

        Returns:
            chex.Array: An array containing the positions of apples on the grid.
                        Each row corresponds to the (x, y) coordinates of an apple.
        """
        apple_flat_positions = jax.random.choice(
            key=key,
            a=self._height * self._width,
            shape=(self._num_apples,),
            replace=False,  # Avoid agent positions overlaping
            p=mask,
        )
        # Unravel indices to get x and y coordinates
        apple_positions_x, apple_positions_y = jnp.unravel_index(
            apple_flat_positions, (self._width, self._height)
        )

        # Stack x and y coordinates to form a 2D array
        return jnp.stack([apple_positions_x, apple_positions_y], axis=1)

    # identical to `sample_agents` of lbf
    def sample_agents(self, key: chex.PRNGKey, mask: chex.Array) -> chex.Array:
        """Randomly samples agent positions on the grid, avoiding positions occupied by trees.

        Args:
            key (chex.PRNGKey): The random key.
            mask (chex.Array): The mask of the grid where 1s correspond to empty cells
            and 0s to full cells.

        Returns:
            chex.Array: An array containing the positions of agents on the grid.
                        Each row corresponds to the (x, y) coordinates of an agent.
        """
        agent_flat_positions = jax.random.choice(
            key=key,
            a=self._height * self._width,
            shape=(self._num_bots,),
            replace=False,  # Avoid agent positions overlaping
            p=mask,
        )
        # Unravel indices to get x and y coordinates
        agent_positions_x, agent_positions_y = jnp.unravel_index(
            agent_flat_positions, (self._width, self._height)
        )

        # Stack x and y coordinates to form a 2D array
        return jnp.stack([agent_positions_x, agent_positions_y], axis=1)


    # `sample_levels` has been removed from the class


    # replaces the `__call__` method, but also with secondary `__call__` function below
    # levels removed from the function.
    # without a specific Agents class it does not contain a "loading" attribute
    def sample_orchard(self, key: chex.PRNGKey) -> SimpleOrchardState:
        """
        Randomly creates an initial orchard state
        """

        (
            tree_pos_key,
            apple_pos_key,
            agent_pos_key,
            key,
        ) = jax.random.split(key, 4)

        # Generate Tree locations
        tree_positions = self.sample_trees(tree_pos_key)

        # Create a mask. The mask contains 0's where something already exists,
        # 1's where new things can be placed.
        mask = jnp.ones((self._width, self._height), dtype=bool)
        mask = mask.at[tree_positions].set(False)

        # Make the mask flat to pass into the choice function, then generate the apple locations
        apple_positions = self.sample_apples(key=apple_pos_key, mask=mask.ravel())

        mask = mask.at[apple_positions].set(False)

        agent_positions = self.sample_agents(key=agent_pos_key, mask=mask.ravel())

        # Create a pytree of generic entitys for the trees
        # The ID is just its position in the generated array
        trees = jax.vmap(SimpleOrchardEntity)(
            id=jnp.arange(self._num_trees),
            position=tree_positions
        )

        # Default the collected status to false
        apples = jax.vmap(SimpleOrchardApple)(
            id=jnp.arange(self._num_apples),
            position=apple_positions,
            collected=jnp.zeros((self._num_apples,), dtype=bool)
        )

        agents = jax.vmap(SimpleOrchardEntity)(
            id=jnp.arange(self._num_bots),
            position=agent_positions
        )

        time = jnp.array(0, jnp.int32)

        return SimpleOrchardState(
            key=key,
            time=time,
            bots=agents,
            trees=trees,
            apples=apples
        )

    def __call__(self, key: chex.PRNGKey) -> SimpleOrchardState:
        return self.sample_orchard(key)