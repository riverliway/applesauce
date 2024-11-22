### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/env.py` ####
### This also includes functions that were brought over from `jumanji/jumanji/environments/routing/lbf/utils.py`

from typing import Dict, Optional, Tuple, Any

import chex
import jax
import jax.numpy as jnp

### NEED TO UPDATE FOR OUR CODE ###
# from our modified files
from jumanji_env.environments.complex_orchard.constants import (
  NUM_ACTIONS,
  JaxArray,
  ROBOT_TURN_SPEED,
  ROBOT_INTERACTION_DISTANCE,
  FORWARD,
  BACKWARD,
  LEFT,
  RIGHT,
  PICK,
  DROP,
  APPLE_DIAMETER,
  REWARD_OUT_OF_BOUNDS,
  REWARD_BAD_PICK,
  REWARD_BAD_DROP,
  REWARD_COLLECT_APPLE
)
from jumanji_env.environments.complex_orchard.generator import ComplexOrchardGenerator
from jumanji_env.environments.complex_orchard.observer import BasicObserver
from jumanji_env.environments.complex_orchard.utils import bots_possible_moves, are_intersecting, distances_between_entities, are_any_intersecting
from jumanji_env.environments.complex_orchard.orchard_types import ComplexOrchardApple, ComplexOrchardObservation, ComplexOrchardState, ComplexOrchardEntity, ComplexOrchardBot

# directly from jumanji
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, restart, termination, transition, truncation

# we are still calling Observation in functions so importing. Assuming this needs to be addressed.
from mava.types import Observation

# replaces `LevelBasedForaging` Class
class ComplexOrchard(Environment[ComplexOrchardState]):
    def __init__(
        self,
        generator: Optional[ComplexOrchardGenerator] = None,
        time_limit: int = 100,
        normalize_reward: bool = True,
        penalty: float = 0.0,
    ) -> None:
        super().__init__()

        self._generator = generator or ComplexOrchardGenerator(
            width=2000,
            height=1600
        )
        self.time_limit = time_limit
        self.width: int = self._generator.width
        self.height: int = self._generator.height
        self.num_picker_bots = self._generator.num_picker_bots
        self.num_pusher_bots = self._generator.num_pusher_bots
        self.num_baskets = self._generator.num_baskets
        self.fov = 5
        # adding the following two because mava and jumanji wrappers expect these
        self.action_dim: int = NUM_ACTIONS
        self.num_agents: int = self.num_picker_bots + self.num_pusher_bots

        self.normalize_reward = normalize_reward
        self.penalty = penalty

        self._observer = BasicObserver(
            fov=self.fov,
            width=self.width,
            height=self.height
        )

    def __repr__(self) -> str:
        return (
            "ComplexOrchard(\n"
            + f"\t grid_width={self.width},\n"
            + f"\t grid_height={self.height},\n"
            + f"\t num_picker_bots={self.num_picker_bots}, \n"
            + f"\t num_pusher_bots={self.num_pusher_bots}, \n"
            + f"\t num_baskets={self.num_baskets}, \n"
            ")"
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[ComplexOrchardState, TimeStep]:
        """Resets the environment.

        Args:
            key (chex.PRNGKey): Used to randomly generate the new `State`.

        Returns:
            Tuple[State, TimeStep]: `State` object corresponding to the new initial state
            of the environment and `TimeStep` object corresponding to the initial timestep.
        """
        state = self._generator(key)
        observation = self._observer.state_to_observation(state)
        timestep = restart(observation, shape=self.num_bots)
        timestep.extras = self._get_extra_info(state, timestep)

        return state, timestep

    def step(self, state: ComplexOrchardState, actions: JaxArray['num_bots']) -> Tuple[ComplexOrchardState, TimeStep]:
        """Simulate one step of the environment.

        Args:
            state (State): State  containing the dynamics of the environment.
            actions (chex.Array): Array containing the actions to take for each agent.

        Returns:
            Tuple[State, TimeStep]: `State` object corresponding to the next state and
            `TimeStep` object corresponding the timestep returned by the environment.
        """
        # Perform the actions for the bots
        new_bot_positions, did_collide = self._perform_movement(state, actions == FORWARD, actions == BACKWARD)
        new_bot_orientations = self._perform_turn(state, actions == LEFT, actions == RIGHT)
        new_holding, new_held, did_try_bad_pick = self._perform_pick(state, actions == PICK)
        new_holding, new_held, new_collected, new_apple_position, did_try_bad_drop, did_collect_apple = self._perform_drop(state, new_holding, new_held, actions == DROP)
        
        # Calculate the reward for each bot
        reward = self.get_reward(did_collide, did_try_bad_pick, did_try_bad_drop, did_collect_apple)

        # Update the state
        new_bots = jax.vmap(ComplexOrchardBot)(
            id=state.bots.id,
            position=new_bot_positions,
            diameter=state.bots.diameter,
            holding=new_holding,
            job=state.bots.job,
            orientation=new_bot_orientations
        )

        new_apples = jax.vmap(ComplexOrchardApple)(
            id=state.apples.id,
            position=new_apple_position,
            diameter=state.apples.diameter,
            held=new_held,
            collected=new_collected
        )

        new_state = ComplexOrchardState(
            key=state.key,
            step_count=state.step_count + 1,
            width=state.width,
            height=state.height,
            bots=new_bots,
            trees=state.trees,
            apples=new_apples,
            baskets=state.baskets
        )

        # Determine if the episode is over
        terminate = jnp.all(state.apples.collected)
        truncate = state.step_count >= self.time_limit
        observation = self._observer.state_to_observation(state)

        timestep = jax.lax.switch(
            terminate + 2 * truncate,
            [
                # !terminate !trunc
                lambda rew, obs: transition(
                    reward=rew, observation=obs, shape=self.num_bots
                ),
                # terminate !truncate
                lambda rew, obs: termination(
                    reward=rew, observation=obs, shape=self.num_bots
                ),
                # !terminate truncate
                lambda rew, obs: truncation(
                    reward=rew, observation=obs, shape=self.num_bots
                ),
                # terminate truncate
                lambda rew, obs: termination(
                    reward=rew, observation=obs, shape=self.num_bots
                ),
            ],
            reward,
            observation,
        )

        timestep.extras = self._get_extra_info(new_state, timestep)

        return new_state, timestep
    
    def _perform_movement(
        self,
        state: ComplexOrchardState,
        move_forwards_mask: JaxArray['num_bots'],
        move_backwards_mask: JaxArray['num_bots'],
    ) -> Tuple[JaxArray['num_bots', 2], JaxArray['num_bots']]:
        """
        Perform the movement for the bots.

        :param state: The current state of the environment
        :param move_forwards_mask: A boolean for each bot indicating if they are moving forward
        :param move_backwards_mask: A boolean for each bot indicating if they are moving backward

        :return: The new positions of all of the bots. Shape: (num_bots, 2)
        and a boolean array indicating if the bot tried to collide with something. Shape: (num_bots,)
        """

        possible_moves: JaxArray['num_bots', 2, 3] = bots_possible_moves(state)

        # If the bot is even capable of moving to that location
        can_forwards = possible_moves[:, 0, 2] > 0.5
        can_backwards = possible_moves[:, 1, 2] > 0.5

        # Update the positions of the bots
        new_positions: JaxArray['num_bots', 2] = state.bots.position.at[move_forwards_mask & can_forwards].set(possible_moves[:, 0, 0:1])
        new_positions: JaxArray['num_bots', 2] = state.bots.position.at[move_backwards_mask & can_backwards].set(possible_moves[:, 1, 0:1])

        # Check if any bots are colliding with each other because of the move
        
        # Create new entities with the new positions to check if they are intersecting with the other entities
        # These entities are not added to the state because they're just used for calculations
        new_bots = jax.vmap(ComplexOrchardEntity)(
            id=state.bots.id,
            position=new_positions,
            diameter=state.bots.diameter,
        )

        # Check if each bot is intersecting with any other bot (including itself)
        is_intersecting_other_bots: JaxArray['num_bots', 'num_bots'] = are_intersecting(new_bots, new_bots)

        # Now mask out the bots that are intersecting with themselves
        mask: JaxArray['num_bots', 'num_bots'] = jnp.eye(self.num_agents, dtype=jnp.bool)
        is_intersecting_other_bots: JaxArray['num_bots'] = jnp.any(is_intersecting_other_bots & (~mask), axis=1)

        # If any bots are intersecting with each other, then revert their position to before moving
        new_positions: JaxArray['num_bots', 2] = state.bots.position.at[is_intersecting_other_bots].set(state.bots.position)

        # Calculate if the bots collided or if they tried to run into something
        did_collide = is_intersecting_other_bots | (move_forwards_mask & ~can_forwards) | (move_backwards_mask & ~can_backwards)

        return new_positions, did_collide
    
    def _perform_turn(
        self,
        state: ComplexOrchardState,
        turn_left_mask: JaxArray['num_bots'],
        turn_right_mask: JaxArray['num_bots'],
    ) -> JaxArray['num_bots']:
        """
        Perform the turning for the bots.

        :param state: The current state of the environment
        :param turn_left_mask: A boolean for each bot indicating if they are turning left
        :param turn_right_mask: A boolean for each bot indicating if they are turning right

        :return: The new orientations of all of the bots. Shape: (num_bots,)
        """
        return state.bots.orientation + (turn_right_mask - turn_left_mask) * ROBOT_TURN_SPEED
    
    def _perform_pick(
        self,
        state: ComplexOrchardState,
        pick_mask: JaxArray['num_bots']
    ) -> Tuple[JaxArray['num_bots'], JaxArray['num_apples'], JaxArray['num_bots']]:
        """
        Performs the pick action for the bots.

        :param state: The current state of the environment
        :param pick_mask: A boolean for each bot indicating if they are picking up an apple

        :return: The new state of the bots.holding, apples.held, and the did_try_bad_pick boolean array
        """

        # Perform the pickup action
        nearest_apple_id: JaxArray['num_bots'] = self._nearest_apple(state)

        def is_close() -> JaxArray['num_bots']:
            """
            Determines if each bot is close enough to their nearest apple to pick it up.

            :return: a boolean array indicating if the bot is close enough to pick up the apple. Shape: (num_bots,)
            """
            nose = self._calculate_bot_nose(state.bots.position)
            apple_position = state.apples.position[state.apples.id == nearest_apple_id]

            return jnp.linalg.norm(apple_position - nose, axis=1) <= ROBOT_INTERACTION_DISTANCE

        can_pick: JaxArray['num_bots'] = pick_mask & (state.bots.holding == -1) & jax.lax.cond(
            nearest_apple_id == -1,
            lambda: jnp.repeat(False, self.num_agents),
            is_close
        )
        new_holding: JaxArray['num_bots'] = state.bots.holding.at[can_pick].set(nearest_apple_id[can_pick])

        # Create a mask for the apples that are being picked up to update their state
        held_mask = jax.vmap(lambda id, targets: jnp.any(id == targets), in_axes=(0, None))(state.apples.id, nearest_apple_id[can_pick])
        new_held: JaxArray['num_apples'] = state.apples.held.at[held_mask].set(True)

        return new_holding, new_held, pick_mask & (~can_pick)

    def _perform_drop(
        self,
        state: ComplexOrchardState,
        new_holding: JaxArray['num_bots'],
        new_held: JaxArray['num_apples'],
        drop_mask: JaxArray['num_bots']
    ) -> Tuple[JaxArray['num_bots'], JaxArray['num_apples'], JaxArray['num_apples'], JaxArray['num_apples', 2], JaxArray['num_bots'], JaxArray['num_bots']]:
        """
        Performs the drop action for the bots.

        :param state: The current state of the environment
        :param new_holding: The new state of the bots.holding created by the _perform_pick function
        :param new_held: The new state of the apples.held created by the _perform_pick function
        :param drop_mask: A boolean for each bot indicating if they are dropping an apple

        :return: The new state of the bots.holding, apples.held, apples.collected, apples.position, did_try_bad_drop, did_collect_apple
        """

        can_drop: JaxArray['num_bots'] = (new_holding != -1) & drop_mask
        dropped_apple_ids = new_holding[drop_mask]

        def update_apple_position(
            id: int,
            current_position: JaxArray[2],
            new_positions: JaxArray['num_dropped_apples', 2],
            dropped_apple_ids: JaxArray['num_dropped_apples']
        ) -> JaxArray[2]:
            """
            Gets either the new position or the old position of the apple based on if it was dropped.

            :param id: The id of the apple we're deciding the position for
            :param current_position: The current position of the apple we're deciding the position for
            :param new_positions: The new positions of all of the dropped apples
            :param dropped_apple_ids: The ids of the apples that were dropped

            :return: The new position of the apple. Shape: (2,)
            """

            return jax.lax.cond(
                jnp.any(dropped_apple_ids == id),
                lambda: new_positions[jnp.argmax(dropped_apple_ids == id)],
                lambda: current_position
            )

        # Update the new apple positions after being dropped
        bot_nose_position: JaxArray['num_bots', 2] = self._calculate_bot_nose(state)
        new_apple_position: JaxArray['num_apples', 2] = jax.vmap(update_apple_position, in_axes=(0, 0, None, None))(state.apples.id, state.apples.position, bot_nose_position[can_drop], dropped_apple_ids)

        # These entities are not added to the state because they're just used for checking if the apple was deposited in a basket
        interaction_check_entities = jax.vmap(ComplexOrchardEntity)(
            id=jnp.arange(bot_nose_position.shape[0]),
            position=bot_nose_position,
            diameter=jnp.repeat(ROBOT_INTERACTION_DISTANCE + APPLE_DIAMETER[1] / 2, bot_nose_position.shape[0]),
        )

        is_near_basket: JaxArray['num_bots'] = are_any_intersecting(interaction_check_entities, state.baskets)
        dropped_are_collected: JaxArray['num_apples'] = jax.vmap(lambda apple_id, collected_ids: jnp.any(apple_id == collected_ids), in_axes=(0, None))(state.apples.id, dropped_apple_ids[is_near_basket[can_drop]])
        new_collected: JaxArray['num_apples'] = state.apples.collected | dropped_are_collected

        new_holding: JaxArray['num_bots'] = new_holding.at[can_drop].set(-1)
        new_held: JaxArray['num_apples'] = new_held.at[can_drop].set(False)

        return new_holding, new_held, new_collected, new_apple_position, drop_mask & (~can_drop), is_near_basket & can_drop

    def _nearest_apple(self, state: ComplexOrchardState) -> JaxArray['num_bots']:
        """
        Get the nearest apple id to each bot.

        :param state: The current state of the environment

        :return: The id of the nearest apple. Shape: (num_bots,)
        If there are no apples, then the id is -1.
        """
        # TODO: Right now we calculate this for every bot, but we could optimize this by only calculating it for the bots that are trying to pick up an apple

        is_apple_valid = (~state.apples.held) & (~state.apples.collected)
        valid_apples_position: JaxArray['num_valid_apples', 2] = state.apples.position[is_apple_valid]
        valid_apples_id: JaxArray['num_valid_apples'] = state.apples.id[is_apple_valid]

        # Calculate the bot's nose position because we want to find the closest apple to the nose
        nose_positions = self._calculate_bot_nose(state)

        return jax.lax.cond(
            valid_apples_position.shape[0] == 0,
            lambda: jnp.repeat(-1, self.num_agents),
            lambda: valid_apples_id[jnp.argmin(distances_between_entities(nose_positions, valid_apples_position), axis=1)],
        )
    
    def _calculate_bot_nose(self, state: ComplexOrchardState) -> JaxArray['num_bots', 2]:
        """
        Calculate the position of the nose of the bot.

        :param state: The current state of the environment

        :return: The position of the nose of the bot. Shape: (num_bots, 2)
        """
        direction = jnp.stack([jnp.cos(state.bots.orientation) * state.bots.diameter, jnp.sin(state.bots.orientation) * state.bots.diameter], axis=1)
        return state.bots.position + direction / 2

    def _get_extra_info(self, state: ComplexOrchardState, timestep: TimeStep) -> Dict:
        """Computes extras metrics to be returned within the timestep."""
        n_eaten = state.apples.collected.sum() + timestep.extras.get(
            "eaten_food", jnp.float32(0)
        )

        percent_eaten = (n_eaten / len(state.apples.id)) * 100
        return {"percent_eaten": percent_eaten}

    def get_reward(
        self,
        did_collide: JaxArray['num_bots'],
        did_try_bad_pick: JaxArray['num_bots'],
        did_try_bad_drop: JaxArray['num_bots'],
        did_collect_apple: JaxArray['num_bots']
    ) -> JaxArray['num_bots']:
        """
        Calculates the reward for each bot

        :param did_collide: A boolean for each bot indicating if they collided with something
        :param did_try_bad_pick: A boolean for each bot indicating if they tried to pick up an apple but failed
        :param did_try_bad_drop: A boolean for each bot indicating if they dropped an apple but not in a basket
        :param did_collect_apple: A boolean for each bot indicating if they successfully collected an apple
        """

        reward = did_collide * REWARD_OUT_OF_BOUNDS
        reward += did_try_bad_pick * REWARD_BAD_PICK
        reward += did_try_bad_drop * REWARD_BAD_DROP
        reward += did_collect_apple * REWARD_COLLECT_APPLE

        return reward
 
    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the environment.

        Returns:
            specs.Spec[Observation]: Spec for the `Observation` with fields grid,
            action_mask, and step_count.
        """
        return self._observer.observation_spec(
            self.time_limit,
        )
    
    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec for the Level Based Foraging environment.

        Returns:
            specs.MultiDiscreteArray: Action spec for the environment with shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([self.action_dim] * self.num_agents),
            dtype=jnp.int32,
            name="action",
        )
    
    def reward_spec(self) -> specs.Array:
        """Returns the reward specification for the `LevelBasedForaging` environment.

        Since this is a multi-agent environment each agent gets its own reward.

        Returns:
            specs.Array: Reward specification, of shape (num_agents,) for the  environment.
        """
        return specs.Array(shape=(self.num_agents,), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        """Describes the discount returned by the environment.

        Returns:
            discount_spec: a `specs.BoundedArray` spec.
        """
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=float,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )
