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
  NOOP,
  APPLE_DIAMETER,
  REWARD_OUT_OF_BOUNDS,
  REWARD_BAD_PICK,
  REWARD_BAD_DROP,
  REWARD_COLLECT_APPLE,
  REWARD_COST_OF_STEP,
  REWARD_PICK_APPLE,
  REWARD_DROPPED_APPLE,
  REWARD_COLLIDING,
  REWARD_NOOPING
)
from jumanji_env.environments.complex_orchard.generator import ComplexOrchardGenerator
from jumanji_env.environments.complex_orchard.observer import BasicObserver, IntermediateObserver
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
        time_limit: int = 500,
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

        self._observer = IntermediateObserver(
            fov=self.fov,
            num_agents = self.num_agents,
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
        timestep = restart(observation, shape=self.num_agents)
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
        new_bot_positions, out_of_bounds, any_collisions = self._perform_movement(state, actions == FORWARD, actions == BACKWARD)
        new_bot_orientations = self._perform_turn(state, actions == LEFT, actions == RIGHT)
        new_holding, new_held, did_try_bad_pick, did_pick_apple = self._perform_pick(state, actions == PICK)
        new_holding, new_held, new_collected, new_apple_position, did_try_bad_drop, did_collect_apple, did_bad_apple_drop = self._perform_drop(state, new_holding, new_held, actions == DROP)
        
        # Calculate the reward for each bot
        reward = self.get_reward(out_of_bounds, any_collisions, did_try_bad_pick, did_try_bad_drop, did_pick_apple, did_collect_apple, did_bad_apple_drop, actions)

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
        observation = self._observer.state_to_observation(new_state)

        timestep = jax.lax.switch(
            terminate + 2 * truncate,
            [
                # !terminate !trunc
                lambda rew, obs: transition(
                    reward=rew, observation=obs, shape=self.num_agents
                ),
                # terminate !truncate
                lambda rew, obs: termination(
                    reward=rew, observation=obs, shape=self.num_agents
                ),
                # !terminate truncate
                lambda rew, obs: truncation(
                    reward=rew, observation=obs, shape=self.num_agents
                ),
                # terminate truncate
                lambda rew, obs: termination(
                    reward=rew, observation=obs, shape=self.num_agents
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

        possible_moves, out_of_bounds, any_collisions = bots_possible_moves(state)

        # If the bot is even capable of moving to that location
        can_forwards: JaxArray['num_bots'] = possible_moves[:, 0, 2] > 0.5
        can_backwards: JaxArray['num_bots'] = possible_moves[:, 1, 2] > 0.5

        def update_position(should_update: bool, new_pos: JaxArray[2], current_pos: JaxArray[2]) -> JaxArray[2]:
            """
            Determines if the bot should update its position based on if it can move to the new position.

            :param should_update: A boolean indicating if the bot should update its position
            :param new_pos: The new position of the bot
            :param current_pos: The current position of the bot
            """

            return jax.lax.cond(
                should_update,
                lambda: new_pos,
                lambda: current_pos
            )

        # Update the positions of the bots
        updater = jax.vmap(update_position, in_axes=(0, 0, 0))
        new_positions: JaxArray['num_bots', 2] = updater(move_forwards_mask & can_forwards, possible_moves[:, 0, 0:2], state.bots.position)
        new_positions: JaxArray['num_bots', 2] = updater(move_backwards_mask & can_backwards, possible_moves[:, 1, 0:2], new_positions)

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
        new_positions: JaxArray['num_bots', 2] = updater(is_intersecting_other_bots, state.bots.position, new_positions)

        # Calculate if the bots collided or if they tried to run into something
        
        did_collide: JaxArray['num_bots'] = (move_forwards_mask & ~can_forwards) | (move_backwards_mask & ~can_backwards)

        return new_positions, out_of_bounds, any_collisions
    
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
        return state.bots.orientation + (turn_right_mask * 1 - turn_left_mask * 1) * ROBOT_TURN_SPEED
    
    def _perform_pick(
        self,
        state: ComplexOrchardState,
        pick_mask: JaxArray['num_bots']
    ) -> Tuple[JaxArray['num_bots'], JaxArray['num_apples'], JaxArray['num_bots']]:
        """
        Performs the pick action for the bots.

        :param state: The current state of the environment
        :param pick_mask: A boolean for each bot indicating if they are picking up an apple

        :return: The new state of the bots.holding, apples.held, the did_try_bad_pick boolean array, and a did_pick_apple boolean array
        """

        # Perform the pickup action
        nearest_apple_id: JaxArray['num_bots'] = self._nearest_apple(state)
        nose: JaxArray['num_bots', 2] = self._calculate_bot_nose(state)
        
        def is_close_check(nearest_apple_id: int, bot_nose: JaxArray[2]) -> bool:
            """
            Determines if a bot is close enough to their nearest apple to pick it up.

            :param nearest_apple_id: The id of the nearest apple to the bot
            :param bot_nose: The position of the bot's nose

            :return: a boolean array indicating if the bot is close enough to pick up the apple. Shape: (num_bots,)
            """
            apple_position: JaxArray[2] = state.apples.position[nearest_apple_id]

            return jax.lax.cond(
                nearest_apple_id != -1,
                lambda: jnp.linalg.norm(apple_position - bot_nose) <= ROBOT_INTERACTION_DISTANCE,
                lambda: False,
            )

        is_close: JaxArray['num_bots'] = jax.vmap(is_close_check)(nearest_apple_id, nose)
                                            
        can_pick: JaxArray['num_bots'] = is_close & (state.bots.holding == -1) & pick_mask

        def update_holding(can_pick: bool, nearest_apple_id: int, existing_holding: int) -> int:
            """
            Updates the holding of the bot based on if they can pick up the apple.

            :param can_pick: A boolean indicating if the bot can pick up the apple
            :param nearest_apple_id: The id of the nearest apple to the bot
            :param existing_holding: The id of the apple that the bot is currently holding

            :return: The id of the apple that the bot is holding. If the bot isn't holding an apple, then the id is -1.
            """

            return jax.lax.cond(
                can_pick,
                lambda: nearest_apple_id,
                lambda: existing_holding
            )

        new_holding: JaxArray['num_bots'] = jax.vmap(update_holding)(can_pick, nearest_apple_id, state.bots.holding)

        def update_held(apple_id: int, held: bool, nearest_apple_id: JaxArray['num_bots'], can_pick: JaxArray['num_bots']) -> bool:
            """
            Updates the held state of the apples based on if they were picked up.

            :param apple_id: The id of the apple we're deciding the state for
            :param nearest_apple_id: The id of the nearest apple to each bot
            :param held: The current state of the apples.held
            :param can_pick: A boolean indicating if the bot can pick up the apple
            """

            return jax.lax.cond(
                jnp.any((apple_id == nearest_apple_id) & can_pick),
                lambda: True,
                lambda: held
            )

        new_held: JaxArray['num_apples'] = jax.vmap(update_held, in_axes=(0, 0, None, None))(state.apples.id, state.apples.held, nearest_apple_id, can_pick)

        return new_holding, new_held, pick_mask & (~can_pick), pick_mask & can_pick

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

        def update_apple_position(
            id: int,
            current_position: JaxArray[2],
            new_positions: JaxArray['num_bots', 2],
            new_holding: JaxArray['num_bots']
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
                jnp.any(new_holding == id),
                lambda: new_positions[jnp.argmax(new_holding == id)],
                lambda: current_position
            )

        # Update the new apple positions after being dropped
        bot_nose_position: JaxArray['num_bots', 2] = self._calculate_bot_nose(state)
        new_apple_position: JaxArray['num_apples', 2] = jax.vmap(update_apple_position, in_axes=(0, 0, None, None))(state.apples.id, state.apples.position, bot_nose_position, new_holding)

        # These entities are not added to the state because they're just used for checking if the apple was deposited in a basket
        interaction_check_entities = jax.vmap(ComplexOrchardEntity)(
            id=jnp.arange(self.num_agents),
            position=bot_nose_position,
            diameter=jnp.repeat(ROBOT_INTERACTION_DISTANCE + APPLE_DIAMETER[1] / 2, self.num_agents),
        )

        is_near_basket: JaxArray['num_bots'] = are_any_intersecting(interaction_check_entities, state.baskets)
        
        # Determine if a bad drop occurred
        bad_apple_drop: JaxArray['num_bots'] = can_drop & ~is_near_basket
    
        def update_collected(id: int, collected: bool, apple_ids_dropped_near_baskets: JaxArray['num_bots']) -> bool:
            """
            Update the collected state of the apples based on if they were dropped.

            :param id: The id of the apple we're deciding the state for
            :param collected: The current state of the apples.collected
            :param dropped_apple_ids: The ids of the apples that were dropped near the baskets (is -1 if not dropped near a basket)
            """

            return jax.lax.cond(
                jnp.any(id == apple_ids_dropped_near_baskets),
                lambda: True,
                lambda: collected
            )

        apple_ids_dropped_near_baskets: JaxArray['num_bots'] = jax.vmap(lambda near, id: jax.lax.cond(near, lambda: id, lambda: -1))(is_near_basket & can_drop, new_holding)
        new_collected: JaxArray['num_apples'] = jax.vmap(update_collected, in_axes=(0, 0, None))(state.apples.id, state.apples.collected, apple_ids_dropped_near_baskets)

        new_holding: JaxArray['num_bots'] = jax.vmap(lambda can_drop, holding: jax.lax.cond(can_drop, lambda: -1, lambda: holding))(can_drop, new_holding)

        def update_held(id: int, currently_held: bool, new_holding: JaxArray['num_bots']) -> bool:
            """
            Updates the held state of the apples based on if they were dropped.

            :param id: The id of the apple we're deciding the state for
            :param new_holding: The new state of the bots.holding
            :param currently_held: The current state of the apples.held
            """

            return jax.lax.cond(
                jnp.any(id == new_holding),
                lambda: False,
                lambda: currently_held
            )

        new_held: JaxArray['num_apples'] = jax.vmap(update_held, in_axes=(0, 0, None))(state.apples.id, new_held, new_holding)

        return new_holding, new_held, new_collected, new_apple_position, drop_mask & (~can_drop), is_near_basket & can_drop, bad_apple_drop

    def _nearest_apple(self, state: ComplexOrchardState) -> JaxArray['num_bots']:
        """
        Get the nearest apple id to each bot.

        :param state: The current state of the environment

        :return: The id of the nearest apple. Shape: (num_bots,)
        If there are no apples, then the id is -1.
        """

        is_apple_valid: JaxArray['num_apples'] = (~state.apples.held) & (~state.apples.collected)

        # Calculate the bot's nose position because we want to find the closest apple to the nose
        nose_positions: JaxArray['num_bots', 2] = self._calculate_bot_nose(state)

        def find_nearest_apple() -> JaxArray['num_bots']:
            """
            Finds the nearest apple to the bot's nose. Assumes that there is at least one valid apple.

            :return: The id of the nearest apple. Shape: (num_bots,)
            """
            distances: JaxArray['num_bots', 'num_apples'] = distances_between_entities(nose_positions, state.apples.position)

            def invalidate_distances(is_apple_valid: bool, distances: JaxArray['num_bots']) -> JaxArray['num_bots']:
                """
                Invalidates the distances to the apples that are not valid. Makes their distance infinity so they will never be the minimum

                :param is_apple_valid: A boolean indicating if the apple is valid
                :param distances: The distances to the apples

                :return: The distances to the apples. Shape: (num_bots,)
                """
                return jax.lax.cond(
                    is_apple_valid,
                    lambda: distances,
                    lambda: jnp.repeat(jnp.inf, self.num_agents)
                )
            
            distances: JaxArray['num_apples', 'num_bots'] = jax.vmap(invalidate_distances, in_axes=(0, 1))(is_apple_valid, distances)

            return jnp.argmin(distances, axis=0)

        return jax.lax.cond(
            jnp.any(is_apple_valid),
            lambda: find_nearest_apple(),
            lambda: jnp.repeat(-1, self.num_agents),
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
    
        n_picked = state.apples.held.sum()         
        n_collected = state.apples.collected.sum()
        
        percent_collected = (n_collected / len(state.apples.id)) * 100
        
        return {"percent_collected": percent_collected,
               "number_picked": n_picked}

    def get_reward(
        self,
        out_of_bounds: JaxArray['num_bots'],
        any_collisions: JaxArray['num_bots'],
        did_try_bad_pick: JaxArray['num_bots'],
        did_try_bad_drop: JaxArray['num_bots'],
        did_pick_apple: JaxArray['num_bots'],
        did_collect_apple: JaxArray['num_bots'],
        did_bad_apple_drop: JaxArray['num_bots'],
        actions: JaxArray['num_bots']
    ) -> JaxArray['num_bots']:
        """
        Calculates the reward for each bot

        :param out_of_bounds: A boolean for each bot indicating if they went out of bounds
        :param did_try_bad_pick: A boolean for each bot indicating if they tried to pick up an apple but failed
        :param did_try_bad_drop: A boolean for each bot indicating if they dropped an apple but not in a basket
        :param did_collect_apple: A boolean for each bot indicating if they successfully collected an apple
        :param did_bad_apple_drop: A boolean for each bot indicating if had an apple and then dropped it somewhere other than the basket
        :param actions: The actions taken by each bot
        """
        
        reward = REWARD_COST_OF_STEP
        reward += out_of_bounds * REWARD_OUT_OF_BOUNDS
        reward += any_collisions * REWARD_COLLIDING
        reward += did_try_bad_pick * REWARD_BAD_PICK
        reward += did_try_bad_drop * REWARD_BAD_DROP
        reward += did_pick_apple * REWARD_PICK_APPLE
        reward += did_bad_apple_drop * REWARD_DROPPED_APPLE
        reward += did_collect_apple * REWARD_COLLECT_APPLE
        reward += (actions == NOOP) * REWARD_NOOPING
        

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
