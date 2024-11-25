### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/generator.py ####

### NEED TO UPDATE FOR OUR CODE ###
# from our modified files
from jumanji_env.environments.complex_orchard.orchard_types import ComplexOrchardApple, ComplexOrchardState, ComplexOrchardBasket, ComplexOrchardTree, ComplexOrchardBot
from jumanji_env.environments.complex_orchard.constants import (
    TREE_DISTANCE_ROW,
    TREE_DISTANCE_COL,
    TREE_VARIATION,
    TREE_DIAMETER,
    ORCHARD_FERTILITY,
    ROBOT_DIAMETER,
    BASKET_DIAMETER,
    APPLE_DIAMETER,
    APPLE_DENSITY,
    JaxArray
)

from typing import Tuple

import chex
import jax
from jax import numpy as jnp
import numpy as np

# This replaces the RandomGenerator class
class ComplexOrchardGenerator:
  """
  Randomly generates complex orchard states in a reproduceable manner using JAX natives.
  """
  def __init__(
    self,
    width: int,
    height: int,
    num_picker_bots: int = 2,
    num_pusher_bots: int = 0,
    num_baskets: int = 1
  ) -> None:
    """
    Initalizes the parameters required for the complex orchard environment
    """

    self.width = width
    self.height = height
    self.num_picker_bots = num_picker_bots
    self.num_pusher_bots = num_pusher_bots
    self.num_baskets = num_baskets

  @staticmethod
  def random_normal(key: chex.PRNGKey, shape: Tuple[int], bounds: Tuple[float, float]) -> JaxArray['...shape']:
    """
    Generates a random normal distribution with the given shape and a bound defining 99% of the values.

    :param key: The key for the deterministic random number generator
    :param shape: The shape of the array to generate
    :param bounds: The bounds of the distribution. 99% of the values will be within these bounds.
    100% of the values will be within 2 * bounds.

    Returns: A random normal distribution with the given shape and bounds.
    """
    print(f"bounds: {bounds}, type: {type(bounds)}")
    print(f"shape: {shape}, type: {type(shape)}")
    print(f"key: {key}, type: {type(key)}")
#     # holding on to old code incase
#     dist = (bounds[1] - bounds[0]) / 2

#     choices = jax.random.normal(key, shape) / 6 * (bounds[1] - bounds[0]) + bounds[0]
#     return jnp.clip(choices, bounds[0] - dist, bounds[1] + dist)

    lower, upper = bounds
    dist = (upper - lower) / 2

    # Generate normal random values
    choices = jax.random.normal(key, shape) / 6 * (upper - lower) + lower

    # Concretize values (if needed outside tracing context)
    choices = jax.device_get(choices)
    
    print(f"choices: {choices}, traced: {isinstance(choices, jax.core.Tracer)}")

    # Clip values
    return jnp.clip(choices, lower - dist, upper + dist)

  def sample_trees(self, key: chex.PRNGKey, tree_row_distance: float, tree_col_distance: float) -> Tuple[JaxArray['num_trees', 2], JaxArray['num_trees'], JaxArray['num_trees']]:
    """
    Randomly place trees ensuring no trees are placed on the edge and no two trees are adjacent

    :param key: The key for the deterministic random number generator
    :param tree_row_distance: The distance between trees in the row direction
    :param tree_col_distance: The distance between trees in the column direction

    Returns: An array containing the flat indices of the trees on the grid.
    The array is length 2 * num_trees, with each tree represented (x, y) one after the other.

    The next return is a flat array of the fertility of each tree.

    The final return is a flat array of the diameter of each tree.
    """

    tree_x = jnp.arange(tree_row_distance, self.width - tree_row_distance, tree_row_distance / 2)
    tree_y = jnp.arange(tree_col_distance * 2, self.height - tree_col_distance / 2, tree_col_distance / 2)
    position_offset = self.random_normal(key, (len(tree_x), len(tree_y), 2), TREE_VARIATION)

    positions = jnp.array([
      [tree_x[i_x] + position_offset[i_x][i_y][0], tree_y[i_y] + position_offset[i_x][i_y][1]]
      for i_x in range(len(tree_x))
      for i_y in range(len(tree_y))
    ])

    fertility = jax.random.uniform(key, (len(tree_x) * len(tree_y)))
    diameter = self.random_normal(key, (len(tree_x) * len(tree_y)), TREE_DIAMETER)

    return positions, fertility, diameter

  # This replaces `sample_food` method.
  def sample_apples(
    self,
    key: chex.PRNGKey,
    tree_positions: JaxArray['num_trees', 2],
    tree_fertility: JaxArray['num_trees'],
    tree_diameter: JaxArray['num_trees'],
    obs_positions: JaxArray['num_obs', 2],
    obs_diameter: JaxArray['num_obs']
  ) -> Tuple[JaxArray['num_apples', 2], JaxArray['num_apples'], JaxArray['num_apples'], JaxArray['num_apples']]:
    """
    Randomly place apples on the grid.

    :param key: The key for the deterministic random number generator
    :param tree_positions: The x and y positions of the trees
    :param tree_fertility: The fertility of the trees
    :param tree_diameter: The diameter of the trees
    :param obs_positions: The x and y positions of the obstacles that apples can't spawn in
    :param obs_diameter: The diameter of the obstacles

    Returns: A tuple of the following arrays:
    - An array containing the flat indices of the apples on the grid.
    - An array containing the diameters of the apples.
    - An array containing if the apple has been collected.
    - An array containing if the apple is being held.
    """
    
    num_trees = len(tree_diameter)

    theta_offsets = jax.random.uniform(key, (num_trees), maxval=2 * jnp.pi)
    num_apples = jnp.floor(jax.nn.relu(self.random_normal(key, (num_trees), ORCHARD_FERTILITY) * tree_fertility)).astype(jnp.int32)
    max_num_apples = jnp.max(num_apples)

    max_apple_size = APPLE_DIAMETER[1] * 2

    def create_apples(theta: float, num_apples: int, tree_pos: JaxArray[2], tree_diameter: float) -> JaxArray['max_num_apples', 2]:
      """
      This is a function that generates the apple positions for a single tree.
      It will get vmaped over to generate the apple positions for all trees.

      :param theta: The angle offset for the apples
      :param num_apples: The number of apples to generate
      :param tree_pos: The x and y position of the tree
      :param tree_diameter: The diameter of the tree

      Returns: an array of the apple positions
      """

      # We always generate the maximum number of apples and then set the positions of the unused apples to be out of bounds
      # because we can't have a variable number of apples in the array (jax requires fixed size arrays)
      thetas = jnp.linspace(0, 2 * jnp.pi, max_num_apples, endpoint=False) + theta
      thetas += jax.random.uniform(key, (max_num_apples,), maxval=0.1)

      radii = jax.random.uniform(key, (max_num_apples,))
      radii = 1 / jnp.pow(radii, APPLE_DENSITY) + tree_diameter + max_apple_size

      apple_x = radii * jnp.cos(thetas) + tree_pos[0]
      apple_y = radii * jnp.sin(thetas) + tree_pos[1]

      apple_x = jnp.where(jnp.arange(max_num_apples) < num_apples, apple_x, -1)
      apple_y = jnp.where(jnp.arange(max_num_apples) < num_apples, apple_y, -1)

      return jnp.ravel(jnp.stack([apple_x, apple_y], axis=1))
    
    apple_positions = jax.vmap(create_apples)(theta_offsets, num_apples, tree_positions, tree_diameter)
    apple_positions = jnp.ravel(apple_positions)

    # Filter out the out of bounds apples
    apple_x = apple_positions[::2]
    apple_y = apple_positions[1::2]

    def check_in_obstacle(apple_x: float, apple_y: float, obs_x: JaxArray['num_obs'], obs_y: JaxArray['num_obs'], obs_diameter: JaxArray['num_obs']) -> bool:
      """
      This function checks if an apple is in any of the obstacles.

      :param apple_x: The x position of the apple
      :param apple_y: The y position of the apple
      :param obs_x: The x positions of the obstacles
      :param obs_y: The y positions of the obstacles
      :param obs_diameter: The diameter of the obstacles

      Returns: True if the apple is in a valid location, False otherwise
      """

      return jnp.any(jnp.hypot(apple_x - obs_x, apple_y - obs_y) > obs_diameter + max_apple_size * 2)

    not_in_obstacle = jax.vmap(check_in_obstacle, in_axes=(0, 0, None, None, None))(apple_x, apple_y, obs_positions[:, 0], obs_positions[:, 1], obs_diameter)
    in_bounds = (apple_x > max_apple_size) & (apple_y > max_apple_size) & (apple_x < self.width - max_apple_size) & (apple_y < self.height - max_apple_size) & not_in_obstacle
    apple_x = apple_x[in_bounds]
    apple_y = apple_y[in_bounds]

    apple_positions = jnp.stack([apple_x, apple_y], axis=1)

    apple_diameters = self.random_normal(key, (len(apple_x),), APPLE_DIAMETER)
    apple_collected = jnp.zeros((len(apple_x),), dtype=bool)
    apple_held = jnp.zeros((len(apple_x),), dtype=bool)

    return apple_positions, apple_diameters, apple_collected, apple_held
  
  def sample_bots(self, tree_col_distance: float) -> Tuple[JaxArray['num_bots', 2], JaxArray['num_bots'], JaxArray['num_bots'], JaxArray['num_bots'], JaxArray['num_bots']]:
    """
    Always place the bots at the top of the grid, evenly spaced.

    :param tree_col_distance: The distance between trees in the column direction

    Returns: A tuple of the following arrays:
    - An array containing the flat indices of the bots on the grid.
    - An array containing the diameters of the bots.
    - An array containing the index of the apple that the bot is holding. -1 if the bot is not holding an apple.
    - An array containing the job of the bots. 0 for picker, 1 for pusher.
    - An array containing the orientation of the bots.
    """
    
    num_total_bots = self.num_picker_bots + self.num_pusher_bots

    bot_x = jnp.linspace(0, self.width, num_total_bots + 1, endpoint=False)[1:]
    bot_y = jnp.full((num_total_bots,), tree_col_distance)

    bot_positions = jnp.stack([bot_x, bot_y], axis=1)
    bot_diameters = jnp.full((num_total_bots,), ROBOT_DIAMETER)
    bot_holding = jnp.full((num_total_bots,), -1)
    bot_jobs = jnp.concatenate([jnp.zeros((self.num_picker_bots,)), jnp.ones((self.num_pusher_bots,))])
    bot_orientations = jnp.zeros((num_total_bots,))

    return bot_positions, bot_diameters, bot_holding, bot_jobs, bot_orientations
  
  def sample_baskets(self, tree_col_distance: float) -> Tuple[JaxArray['num_baskets', 2], JaxArray['num_baskets'], JaxArray['num_baskets']]:
    """
    Always place the baskets at the top of the grid, evenly spaced.

    :param tree_col_distance: The distance between trees in the column direction

    Returns: A tuple of the following arrays:
    - An array containing the flat indices of the baskets on the grid.
    - An array containing the diameters of the baskets.
    - An array containing if the basket is held by a bot.
    """

    basket_x = jnp.linspace(0, self.width, self.num_baskets + 1, endpoint=False)[1:]
    basket_y = jnp.full((self.num_baskets,), tree_col_distance)

    basket_positions = jnp.stack([basket_x, basket_y], axis=1)
    basket_diameters = jnp.full((self.num_baskets,), BASKET_DIAMETER)
    basket_orientations = jnp.zeros((self.num_baskets,), dtype=bool)

    return basket_positions, basket_diameters, basket_orientations

  def sample_orchard(self, key: chex.PRNGKey) -> ComplexOrchardState:
    """
    Randomly creates an initial orchard state
    """


    # unsuccessful attempts at using numpy to address tracer value error which created downstream issues
#     tree_row_distance = np.random.normal(
#                                         loc=(TREE_DISTANCE_ROW[0] + TREE_DISTANCE_ROW[1]) / 2,
#                                         scale=(TREE_DISTANCE_ROW[1] - TREE_DISTANCE_ROW[0]) / 6,
#                                     )
    
#     tree_col_distance = np.random.normal(
#                                         loc=(TREE_DISTANCE_COL[0] + TREE_DISTANCE_COL[1]) / 2,
#                                         scale=(TREE_DISTANCE_COL[1] - TREE_DISTANCE_COL[0]) / 6,
#                                     )
    # original call for tree dims, creating tracer values
    tree_row_distance = self.random_normal(key, (1,), TREE_DISTANCE_ROW)[0]
    tree_col_distance = self.random_normal(key, (1,), TREE_DISTANCE_COL)[0]

    (
      tree_pos_key,
      apple_pos_key,
      key
    ) = jax.random.split(key, 3)

    # Generate Tree locations
    tree_positions, tree_fertilities, tree_diameters = self.sample_trees(tree_pos_key, tree_row_distance, tree_col_distance)

    bot_positions, bot_diameters, bot_holding, bot_jobs, bot_orientations = self.sample_bots(tree_col_distance)

    basket_positions, basket_diameters, basket_orientations = self.sample_baskets(tree_col_distance)

    print(f"tree_positions shape: {tree_positions.shape}")
    print(f"bot_positions shape: {bot_positions.shape}")
    print(f"basket_positions shape: {basket_positions.shape}")
    obs_positions = jnp.concatenate([tree_positions, bot_positions, basket_positions])
    obs_diameter = jnp.concatenate([tree_diameters, bot_diameters, basket_diameters])
    apple_positions, apple_diameters, apple_held, apple_collected = self.sample_apples(apple_pos_key, tree_positions, tree_fertilities, tree_diameters, obs_positions, obs_diameter)

    # Create a pytree of generic entitys for the trees
    # The ID is just its position in the generated array
    trees = jax.vmap(ComplexOrchardTree)(
      id=jnp.arange(len(tree_diameters)),
      position=tree_positions,
      diameter=tree_diameters,
      fertility=tree_fertilities
    )

    # Default the collected status to false
    apples = jax.vmap(ComplexOrchardApple)(
      id=jnp.arange(len(apple_diameters)),
      position=apple_positions,
      diameter=apple_diameters,
      held=apple_held,
      collected=apple_collected
    )

    bots = jax.vmap(ComplexOrchardBot)(
      id=jnp.arange(len(bot_diameters)),
      position=bot_positions,
      diameter=bot_diameters,
      holding=bot_holding,
      job=bot_jobs,
      orientation=bot_orientations
    )

    baskets = jax.vmap(ComplexOrchardBasket)(
      id=jnp.arange(len(basket_diameters)),
      position=basket_positions,
      diameter=basket_diameters,
      held=basket_orientations
    )

    step_count = jnp.array(0, jnp.int32)

    return ComplexOrchardState(
      key=key,
      step_count=step_count,
      width=jnp.array(self.width, jnp.int32),
      height=jnp.array(self.height, jnp.int32),
      bots=bots,
      trees=trees,
      apples=apples,
      baskets=baskets
    )

  def __call__(self, key: chex.PRNGKey) -> ComplexOrchardState:
    return self.sample_orchard(key)
