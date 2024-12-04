### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/observer.py ####

import abc
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp

### NEED TO UPDATE FOR OUR CODE ###
# from our modified files
from jumanji_env.environments.complex_orchard.constants import NUM_ACTIONS, JaxArray
from jumanji_env.environments.complex_orchard.utils import normalize_angles, bots_possible_moves, distances_between_entities
from jumanji_env.environments.complex_orchard.orchard_types import (
    ComplexOrchardBasket,
    ComplexOrchardBot,
    ComplexOrchardTree,
    ComplexOrchardApple,
    ComplexOrchardObservation,
    ComplexOrchardState,
)

from jumanji import specs

class ComplexOrchardObserver(abc.ABC):
  """
  Base class for the complex orchard observers.
  """

  def __init__(self, fov: int, width: int, height: int, num_agents: int) -> None:
    """
    Initalizes the observer object
    """

    self.fov = fov
    self.width = width
    self.height = height
    self.num_agents = num_agents

  @abc.abstractmethod
  def state_to_observation(self, state: ComplexOrchardState) -> ComplexOrchardObservation:
    """
    Convert the current state of the environment into observations for all agents.

    Args:
        state (State): The current state containing agent and food information.

    Returns:
        Observation: An Observation object containing the agents' views, action masks,
        and step count for all agents.
    """
    pass

  @abc.abstractmethod
  def observation_spec(self, time_limit: int) -> specs.Spec[ComplexOrchardObservation]:
    """
    Returns the observation spec for the environment
    """
    pass

  def _action_mask_spec(self) -> specs.BoundedArray:
    """
    Returns the action spec for the environment.

    The action mask is a boolean array of shape (num_agents, 7). '7' is the number of actions.
    """
    return specs.BoundedArray(
      shape=(self.num_agents, NUM_ACTIONS),
      dtype=bool,
      minimum=False,
      maximum=True,
      name="action_mask"
    )
  
  def _time_spec(self, time_limit: int) -> specs.BoundedArray:
    """Returns the step count spec for the environment."""
    return specs.BoundedArray(
      shape=(),
      dtype=jnp.int32,
      minimum=0,
      maximum=time_limit,
      name="step_count",
    )

class BasicObserver(ComplexOrchardObserver):
  """
  The most basic observer for the complex orchard environment.

  The agent can see only 3 pieces of information:
  - the horizontal distance to the nearest apple
  - the vertical distance to the nearest apple
  - the angle difference to the nearest apple

  It is completely blind to everything else
  """
  
  def __init__(self, fov: int, width: int, height: int, num_agents: int) -> None:
    """
    Initalizes the basic observer object
    """

    super().__init__(fov, width, height, num_agents)

  def _observe(
    self,
    bot_position: JaxArray,
    bot_orientation: float,
    bot_job: int,
    bot_holding: int,
    apples_position: JaxArray,
    apples_held: JaxArray,
    apples_collected: JaxArray,
    baskets_position: JaxArray,
  ) -> JaxArray:
    """
    Makes the observation for a single agent.

    :param bot_position: The position of the agent. Shape: (2,)
    :param bot_orientation: The orientation of the agent. Shape: ()
    :param bot_job: The job of the agent. Shape: ()
    :param bot_holding: The apple held by the agent. Shape: ()
    :param apples_position: The positions of the apples. Shape: (num_apples, 2)
    :param apples_held: The apples held by the agents. Shape: (num_apples,)
    :param apples_collected: The apples collected by the agents. Shape: (num_apples,)
    :param baskets_position: The positions of the baskets. Shape: (num_baskets, 2)

    Returns: The observation for the agent. Shape: (3,)
    """

    is_apple_valid: JaxArray['num_apples'] = ~apples_held & ~apples_collected

    def find_nearest_apple() -> JaxArray['num_bots']:
        """
        Finds the nearest apple to the bot's nose. Assumes that there is at least one valid apple.

        :return: The id of the nearest apple. Shape: (num_bots,)
        """
        distances: JaxArray['num_apples'] = jnp.linalg.norm(apples_position - bot_position, axis=1)

        def invalidate_distances(is_apple_valid: bool, distance: float) -> float:
            """
            Invalidates the distances to the apples that are not valid. Makes their distance infinity so they will never be the minimum

            :param is_apple_valid: A boolean indicating if the apple is valid
            :param distance: The distance to the apples

            :return: The distance to the apples
            """
            return jax.lax.cond(
                is_apple_valid,
                lambda: distance,
                lambda: jnp.inf
            )
        
        distances: JaxArray['num_apples'] = jax.vmap(invalidate_distances, in_axes=(0, 0))(is_apple_valid, distances)

        return jnp.argmin(distances, axis=0)

    def get_nearest_basket() -> JaxArray[2]:
      return jax.lax.cond(
        baskets_position.shape[0] == 0,
        lambda: jnp.array([0, 0], dtype=jnp.float32),
        lambda: baskets_position[jnp.argmin(jnp.linalg.norm(baskets_position - bot_position, axis=1))]
      )

    # Determine if the targets are apples or baskets
    nearest_target_position: JaxArray[2] = jax.lax.cond(
      (bot_job == 0) & (bot_holding == -1),
      lambda: apples_position[find_nearest_apple()],
      get_nearest_basket
    )

    return jnp.array([
      nearest_target_position[0] - bot_position[0],
      nearest_target_position[1] - bot_position[1],
      normalize_angles(jnp.arctan2(nearest_target_position[1] - bot_position[1], nearest_target_position[0] - bot_position[0]) - bot_orientation)
    ])
  
  def _create_action_mask(self, state: ComplexOrchardState) -> JaxArray['num_agents', 'NUM_ACTIONS']:
    """
    Creates the action mask for all agents.

    :param state: The current state of the environment

    Returns: The boolean action mask for all agents. Shape: (num_agents, 7)
    """

    possible_forward_backward, _, _ = bots_possible_moves(state)
    possible_forward_backward = possible_forward_backward[:, :, 2]
    possible_actions: JaxArray['num_agents', 'NUM_ACTIONS'] = jnp.ones((len(state.bots.diameter), NUM_ACTIONS), dtype=bool)

    # The 1:2 is a references to the FORWARD = 1 and BACKWARD = 2 actions defined in the constants.py file
    possible_actions: JaxArray['num_agents', 'NUM_ACTIONS'] = possible_actions.at[:, 1:3].set(possible_forward_backward > 0.5)

    return possible_actions

  def state_to_observation(self, state: ComplexOrchardState) -> ComplexOrchardObservation:
    """
    Convert the current state of the environment into observations for all agents.

    Args:
        state (State): The current state containing agent and food information.

    Returns:
        Observation: An Observation object containing the agents' views, action masks,
        and step count for all agents.
    """

    num_agents = len(state.bots.diameter)

    # Placeholder for the agents' views
    agents_view: JaxArray['num_agents', 3] = jax.vmap(self._observe, in_axes=(0, 0, 0, 0, None, None, None, None))(
      state.bots.position,
      state.bots.orientation,
      state.bots.job,
      state.bots.holding,
      state.apples.position,
      state.apples.held,
      state.apples.collected,
      state.baskets.position
    )

    # Placeholder for the action mask
    action_mask = self._create_action_mask(state)

    return ComplexOrchardObservation(agents_view=agents_view, action_mask=action_mask, step_count=state.step_count)

  def observation_spec(self, time_limit: int) -> specs.Spec[ComplexOrchardObservation]:
    """
    Returns the observation spec for the environment
    """

    return specs.Spec(
      ComplexOrchardObservation,
      "ComplexOrchardObservationSpec",
      agents_view=specs.BoundedArray(
        shape=(self.num_agents, 3),
        dtype=jnp.float32,
        minimum=-jnp.inf,
        maximum=jnp.inf,
        name="agents_view"
      ),
      action_mask=self._action_mask_spec(),
      step_count=self._time_spec(time_limit)
    )

class IntermediateObserver(BasicObserver):
    """
    This observer is making minor advancements to Observations be including the location of other objects in the environment. Itself, basket, and trees. 
    
    The agent can see only 3 pieces of information:
    - the horizontal distance to the nearest apple
    - the vertical distance to the nearest apple
    - the angle difference to the nearest apple
    
    It is completely blind to everything else
    """
    
    def __init__(self, fov: int, width: int, height: int, num_agents: int) -> None:
        """
        Initalizes the basic observer object
        """

        super().__init__(fov, width, height, num_agents)

    def _observe(
        self,
        bot_position: JaxArray,
        bot_orientation: float,
        bot_job: int,
        bot_holding: int,
        apples_position: JaxArray,
        apples_held: JaxArray,
        apples_collected: JaxArray,
        baskets_position: JaxArray,
      ) -> JaxArray:
        """
        Makes the observation for a single agent.
    
        :param bot_position: The position of the agent. Shape: (2,)
        :param bot_orientation: The orientation of the agent. Shape: ()
        :param bot_job: The job of the agent. Shape: ()
        :param bot_holding: The apple held by the agent. Shape: ()
        :param apples_position: The positions of the apples. Shape: (num_apples, 2)
        :param apples_held: The apples held by the agents. Shape: (num_apples,)
        :param apples_collected: The apples collected by the agents. Shape: (num_apples,)
        :param baskets_position: The positions of the baskets. Shape: (num_baskets, 2)
    
        Returns: The observation for the agent. Shape: (3,)
        """
    
        is_apple_valid: JaxArray['num_apples'] = ~apples_held & ~apples_collected
    
        def find_nearest_objects() -> Tuple[JaxArray['10, 2'], JaxArray['2']]:
            """
            Finds the 10 closest apples to the bot's position and the closest basket.
            
            :return: A tuple containing:
                     - The positions of the 10 nearest apples, padded if fewer than 10 valid apples. Shape: (10, 2)
                     - The position of the nearest basket. Shape: (2,)
            """
            # Calculate distances to all apples
            distances_to_apples: JaxArray['num_apples'] = jnp.linalg.norm(apples_position - bot_position, axis=1)
        
            # Invalidate distances for invalid apples
            def invalidate_distances(is_apple_valid: bool, distance: float) -> float:
                """
                Invalidates the distances to the apples that are not valid. Makes their distance infinity so they will never be selected.
                
                :param is_apple_valid: A boolean indicating if the apple is valid
                :param distance: The distance to the apples
                
                :return: The distance to the apples
                """
                return jax.lax.cond(
                    is_apple_valid,
                    lambda: distance,
                    lambda: jnp.inf
                )
            count = 1
            distances_to_apples: JaxArray['num_apples'] = jax.vmap(invalidate_distances, in_axes=(0, 0))(is_apple_valid, distances_to_apples)
        
            # Get the indices of the nearest apples sorted by distance
            sorted_apple_indices: JaxArray['num_apples'] = jnp.argsort(distances_to_apples)
        
            # Select the 10 nearest apples
            nearest_apple_indices: JaxArray['count'] = sorted_apple_indices[:count]
        
            # Pad with [-1, -1] if there are fewer than 10 valid apples
            num_valid_apples = jnp.sum(is_apple_valid)
            nearest_apples_padded = jnp.where(
                jnp.arange(count)[:, None] < num_valid_apples,  # Check if each position is within the number of valid apples
                apples_position[nearest_apple_indices],     # Use the actual positions if within bounds
                jnp.array([-1, -1], dtype=jnp.float32)      # Pad with [-1, -1] for invalid positions
            )
        
            # Calculate distances to baskets
            distances_to_baskets: JaxArray['num_baskets'] = jnp.linalg.norm(baskets_position - bot_position, axis=1)
        
            # Find the nearest basket
            nearest_basket: JaxArray[2] = jax.lax.cond(
                baskets_position.shape[0] == 0,  # Check if there are no baskets
                lambda: jnp.array([0, 0], dtype=jnp.float32),  # Default value if no baskets
                lambda: baskets_position[jnp.argmin(distances_to_baskets)]  # Closest basket
            )

            return nearest_apples_padded, nearest_basket

        # def find_nearest_bot() -> JaxArray[3]:
        #     # Exclude the current bot from the distance calculation
        #     mask_self = jnp.all(bot_position == bot_position, axis=1)
        #     valid_bots_positions = jnp.where(mask_self[:, None], jnp.inf, all_bots_positions)
    
        #     # Calculate distances to all other bots
        #     distances_to_bots = jnp.linalg.norm(valid_bots_positions - bot_position, axis=1)
    
        #     # Find the nearest bot
        #     nearest_bot_idx = jnp.argmin(distances_to_bots)
        #     nearest_bot = valid_bots_positions[nearest_bot_idx]
    
        #     # Return relative distance and orientation
        #     return nearest_bot

        # nearest_bot = find_nearest_bot()
        
        #         bot_distance = jnp.array([
        #     nearest_bot[0] - bot_position[0],
        #     nearest_bot[1] - bot_position[1],
        #     normalize_angles(jnp.arctan2(
        #         nearest_bot[1] - bot_position[1],
        #         nearest_bot[0] - bot_position[0]
        #     ) - bot_orientation)
        # ])

        nearest_apples, nearest_basket = find_nearest_objects()

        apple_distances = jax.vmap(
            lambda apple: jnp.array([
                apple[0] - bot_position[0],  # Horizontal distance to the apple
                apple[1] - bot_position[1],  # Vertical distance to the apple
                normalize_angles(jnp.arctan2(
                    apple[1] - bot_position[1],
                    apple[0] - bot_position[0]
                ) - bot_orientation)  # Orientation difference to the apple
            ])
        )(nearest_apples)  # nearest_apples is a (10, 2) array

        basket_distance = jnp.array([
                nearest_basket[0] - bot_position[0],  # Horizontal distance to the basket
                nearest_basket[1] - bot_position[1],  # Vertical distance to the basket
                normalize_angles(jnp.arctan2(
                    nearest_basket[1] - bot_position[1],
                    nearest_basket[0] - bot_position[0]
                ) - bot_orientation)  # Orientation difference to the basket
            ])

        return jnp.concatenate([apple_distances.flatten(), basket_distance])
                       
    def observation_spec(self, time_limit: int) -> specs.Spec[ComplexOrchardObservation]:
        """
        Returns the observation spec for the environment
        """
    
        return specs.Spec(
          ComplexOrchardObservation,
          "ComplexOrchardObservationSpec",
          agents_view=specs.BoundedArray(
            shape=(self.num_agents, 6),
            dtype=jnp.float32,
            minimum=-jnp.inf,
            maximum=jnp.inf,
            name="agents_view"
          ),
          action_mask=self._action_mask_spec(),
          step_count=self._time_spec(time_limit)
)