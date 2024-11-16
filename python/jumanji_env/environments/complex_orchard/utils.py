import jax.numpy as jnp
import jax

from jumanji_env.environments.complex_orchard.constants import ROBOT_MOVE_SPEED, JaxArray
from jumanji_env.environments.complex_orchard.orchard_types import (
    ComplexOrchardState,
    ComplexOrchardEntity
)

def normalize_angles(angles: JaxArray['num_angles']) -> JaxArray['num_angles']:
  """
  Normalize the angle to be between -pi and pi.

  :param angles: The angle to normalize. Shape: (#,)

  Returns: The normalized angle. Shape: (#,)
  """
  return jnp.mod(angles + jnp.pi, 2 * jnp.pi) - jnp.pi

def distances_between_entities(
  target_entities: ComplexOrchardEntity,
  other_entities: ComplexOrchardEntity
) -> JaxArray['num_target_entities', 'num_other_entities']:
  """
  Calculate the distance between the target entities and each of the other entities.

  :param target_entities: The target entities to calculate the distance from
  :param other_entities: The other entities to calculate the distance to

  :return: The distance between the target entities and the other entities. Shape: (num_target_entities, num_other_entities)
  """
  dist_calc = lambda target, other: jnp.linalg.norm(target - other, axis=1)

  return jax.vmap(dist_calc, in_axes=(0, None))(target_entities, other_entities)

def are_intersecting(target_entities: ComplexOrchardEntity, other_entities: ComplexOrchardEntity) -> JaxArray['num_target_entities', 'num_other_entities']:
  """
  Check if the target entities are intersecting with the other entities.

  :param target_entities: The target entities to check if they are intersecting
  :param other_entities: The other entities to check if they are intersecting with the target entities

  :return: A boolean array indicating if the target entities are intersecting with the other entities. Shape: (num_target_entities, num_other_entities)
  """
  # Calculate the distance between the target entities and the other entities
  distances = distances_between_entities(target_entities, other_entities)

  # Calcuate the sum of the diameters of the target and other entities
  radius_calc = lambda target, other: (target + other) / 2
  radius_sum = jax.vmap(radius_calc, in_axes=(0, None))(target_entities.diameter, other_entities.diameter)

  # Check if the entities are intersecting
  return (distances - radius_sum) < 0

def are_any_intersecting(target_entities: ComplexOrchardEntity, other_entities: ComplexOrchardEntity) -> JaxArray['num_target_entities']:
  """
  Check if any of the target entities are intersecting with the other entities.

  :param target_entities: The target entities to check if they are intersecting
  :param other_entities: The other entities to check if they are intersecting with the target entities

  :return: A boolean array indicating if the target entities are intersecting with the other entities. Shape: (num_target_entities,)
  """
  return jnp.any(are_intersecting(target_entities, other_entities), axis=1)

def bots_possible_moves(state: ComplexOrchardState) -> JaxArray['num_bots', 2, 3]:
  """
  Simulates moving forward and backward for every bot.

  :param state: The current state of the environment.

  :return: The possible moves for every bot. Shape: (num_bots, 2, 3) where they are [bot][forward, backward][x, y, is_possible]
  """

  num_bots = state.bots.position.shape[0]
  direction: JaxArray['num_bots', 2] = jnp.stack([jnp.cos(state.bots.orientation), jnp.sin(state.bots.orientation)], axis=1) * ROBOT_MOVE_SPEED

  # Calculate the new position of all of the bots
  new_forward_positions: JaxArray['num_bots', 2] = state.bots.position + direction
  new_backward_positions: JaxArray['num_bots', 2] = state.bots.position - direction

  # Make space for the is_possible value
  new_forward_positions: JaxArray['num_bots', 3] = jnp.pad(new_forward_positions, (0, 1), constant_values=1)[:-1]
  new_backward_positions: JaxArray['num_bots', 3] = jnp.pad(new_backward_positions, (0, 1), constant_values=1)[:-1]

  new_positions: JaxArray['num_bots', 2, 3] = jnp.stack([new_forward_positions, new_backward_positions], axis=1)
  new_x = new_positions[:, :, 0]
  new_y = new_positions[:, :, 1]

  # Check if the bot is within the bounds
  is_possible = (new_x >= 0) & (new_y <= state.width) & (new_x >= 0) & (new_y <= state.height)

  new_positions = new_positions.at[:, :, 2].set(is_possible)

  return new_positions
