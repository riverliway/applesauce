### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/observer.py ####

import abc
from typing import Any, Tuple, Union

import jax
import jax.numpy as jnp

### NEED TO UPDATE FOR OUR CODE ###
# from our modified files
from jumanji_env.environments.complex_orchard.constants import NUM_ACTIONS, JaxArray
from jumanji_env.environments.complex_orchard.utils import normalize_angles, bots_possible_moves
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

    valid_apples_position: JaxArray['num_valid_apples', 2] = apples_position[(~apples_held) & (~apples_collected)]

    def get_nearest_apple() -> JaxArray[2]:
      # Filter out the apples that are held or collected
      return jax.lax.cond(
        valid_apples_position.shape[0] == 0,
        lambda: jnp.array([0, 0], dtype=jnp.float32),
        lambda: valid_apples_position[jnp.argmin(jnp.linalg.norm(valid_apples_position - bot_position, axis=1))]
      )

    def get_nearest_basket() -> JaxArray[2]:
      return jax.lax.cond(
        baskets_position.shape[0] == 0,
        lambda: jnp.array([0, 0], dtype=jnp.float32),
        lambda: baskets_position[jnp.argmin(jnp.linalg.norm(baskets_position - bot_position, axis=1))]
      )

    # Determine if the targets are apples or baskets
    nearest_target_position: JaxArray[2] = jax.lax.cond(
      (bot_job == 0) & (bot_holding == -1),
      get_nearest_apple,
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

    possible_forward_backward: JaxArray['num_agents', 2] = bots_possible_moves(state)[:, :, 2]
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
