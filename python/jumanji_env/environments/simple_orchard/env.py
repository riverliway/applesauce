### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/env.py` ####
### This also includes functions that were brought over from `jumanji/jumanji/environments/routing/lbf/utils.py`

from functools import cached_property
from typing import Dict, Optional, Sequence, Tuple, Union, Any

import chex
import jax
import jax.numpy as jnp
import matplotlib
from numpy.typing import NDArray

### NEED TO UPDATE FOR OUR CODE ###
# from our modified files
from jumanji_env.environments.simple_orchard.constants import MOVES, LOAD
from jumanji_env.environments.simple_orchard.generator import SimpleOrchardGenerator
from jumanji_env.environments.simple_orchard.observer import SimpleOrchardObserver
from jumanji_env.environments.simple_orchard.orchard_types import SimpleOrchardApple, SimpleOrchardObservation, SimpleOrchardState, SimpleOrchardAgent, SimpleOrchardEntity

# directly from jumanji
import jumanji.environments.routing.lbf.utils as utils
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.lbf.viewer import LevelBasedForagingViewer
from jumanji.types import TimeStep, restart, termination, transition, truncation
from jumanji.viewer import Viewer

# we are still calling Observation in functions so importing. Assuming this needs to be addressed.
from mava.types import Observation

##################### The classes and functions have been re-organized to match the 
##################### order that they appear in the equivalent jumanji file.

# replaces `LevelBasedForaging` Class
# why removal of specs.MultiDiscreteArray, Observation from class arguments?
class SimpleOrchard(Environment[SimpleOrchardState]):
    """
    An implementation of the Level-Based Foraging environment where agents need to
    cooperate to collect food and split the reward.

    Original implementation: https://github.com/semitable/lb-foraging/tree/master

    - `observation`: `Observation`
        - `agent_views`: Depending on the `observer` passed to `__init__`, it can be a
          `GridObserver` or a `VectorObserver`.
            - `GridObserver`: Returns an agent's view with a shape of
              (num_agents, 3, 2 * fov + 1, 2 * fov +1).
            - `VectorObserver`: Returns an agent's view with a shape of
              (num_agents, 3 * (num_food + num_agents).
        - `action_mask`: JAX array (bool) of shape (num_agents, 6)
          indicating for each agent which size actions
          (no-op, up, down, left, right, load) are allowed.
        - `step_count`: int32, the number of steps since the beginning of the episode.

    - `action`: JAX array (int32) of shape (num_agents,). The valid actions for each
        agent are (0: noop, 1: up, 2: down, 3: left, 4: right, 5: load).

    - `reward`: JAX array (float) of shape (num_agents,)
        When one or more agents load food, the food level is rewarded to the agents, weighted
        by the level of each agent. The reward is then normalized so that, at the end,
        the sum of the rewards (if all food items have been picked up) is one.

    - Episode Termination:
        - All food items have been eaten.
        - The number of steps is greater than the limit.

    - `state`: `State`
        - `agents`: Stacked Pytree of `Agent` objects of length `num_agents`.
            - `Agent`:
                - `id`: JAX array (int32) of shape ().
                - `position`: JAX array (int32) of shape (2,).
                - `level`: JAX array (int32) of shape ().
                - `loading`: JAX array (bool) of shape ().
        - `food_items`: Stacked Pytree of `Food` objects of length `num_food`.
            - `Food`:
                - `id`: JAX array (int32) of shape ().
                - `position`: JAX array (int32) of shape (2,).
                - `level`: JAX array (int32) of shape ().
                - `eaten`: JAX array (bool) of shape ().
        - `step_count`: JAX array (int32) of shape (), the number of steps since the beginning
          of the episode.
        - `key`: JAX array (uint) of shape (2,)
            JAX random generation key. Ignored since the environment is deterministic.

    Example:
    ```python
    from jumanji.environments import LevelBasedForaging
    env = LevelBasedForaging()
    key = jax.random.key(0)
    state, timestep = jax.jit(env.reset)(key)
    env.render(state)
    action = env.action_spec().generate_value()
    state, timestep = jax.jit(env.step)(state, action)
    env.render(state)
    ```

    Initialization Args:
    - `generator`: A `Generator` object that generates the initial state of the environment.
        Defaults to a `RandomGenerator` with the following parameters:
            - `grid_size`: 8
            - `fov`: 8 (full observation of the grid)
            - `num_agents`: 2
            - `num_food`: 2
            - `max_agent_level`: 2
            - `force_coop`: True
    - `time_limit`: The maximum number of steps in an episode. Defaults to 200.
    - `grid_observation`: If `True`, the observer generates a grid observation (default is `False`).
    - `normalize_reward`: If `True`, normalizes the reward (default is `True`).
    - `penalty`: The penalty value (default is 0.0).
    - `viewer`: Viewer to render the environment. Defaults to `LevelBasedForagingViewer`.
    """

    # Only setting up a single observer to match VectorObserver
    # removal of viewer
    # level functionality removed
    # fov hard-coded at 5
    def __init__(
        self,
        generator: Optional[SimpleOrchardGenerator] = None,
        time_limit: int = 200,
        normalize_reward: bool = True,
        penalty: float = 0.01,
    ) -> None:
        super().__init__()

        self._generator = generator or SimpleOrchardGenerator(
            width=10,
            height=10,
            num_bots=2,
            num_apples=10,
        )
        self.time_limit = time_limit
        self.width: int = self._generator.get_width()
        self.height: int = self._generator.get_height()
        self.num_bots: int = self._generator.get_num_bots()
        self.num_apples: int = self._generator.get_num_apples()
        self.num_trees: int = self._generator.get_num_trees()
        self.fov = 5
        # adding the following two because mava and jumanji wrappers expect these
        self.action_dim: int = 6
        self.num_agents: int = self._generator.get_num_bots()


        self.normalize_reward = normalize_reward
        self.penalty = penalty
        self.num_obs_features = jnp.array(2 * (self.num_bots + self.num_apples + self.num_trees), jnp.int32)

        self._observer = SimpleOrchardObserver(
            fov=self.fov,
            width=self.width,
            height=self.height,
            num_bots=self.num_bots,
            num_apples=self.num_apples,
            num_trees=self.num_trees
        )

    def __repr__(self) -> str:
        return (
            "LevelBasedForaging(\n"
            + f"\t grid_width={self.width},\n"
            + f"\t grid_height={self.height},\n"
            + f"\t num_agents={self.num_bots}, \n"
            + f"\t num_food={self.num_apples}, \n"
            + f"\t num_trees={self.num_trees}\n"
            ")"
        )

    # same other than naming convention
    def reset(self, key: chex.PRNGKey) -> Tuple[SimpleOrchardState, TimeStep]:
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
    
    
    def step(self, state: SimpleOrchardState, actions: chex.Array) -> Tuple[SimpleOrchardState, TimeStep]:
        """Simulate one step of the environment.

        Args:
            state (State): State  containing the dynamics of the environment.
            actions (chex.Array): Array containing the actions to take for each agent.

        Returns:
            Tuple[State, TimeStep]: `State` object corresponding to the next state and
            `TimeStep` object corresponding the timestep returned by the environment.
        """
        # Move agents, fix collisions that may happen and set loading status.
        moved_agents = self._update_agent_positions(state.bots, actions, state.apples)

        # Collect the food
        apple_items, collected_this_step, all_food_collected = jax.vmap(
            self._collect_food, (None, 0)
        )(moved_agents, state.apples)

        reward = self.get_reward(apple_items, all_food_collected, collected_this_step)

        state = SimpleOrchardState(
            bots=moved_agents,
            apples=apple_items,
            trees=state.trees,
            step_count=state.step_count + 1,
            key=state.key,
        )
        observation = self._observer.state_to_observation(state)

        # First condition is truncation, second is termination.
        terminate = jnp.all(state.apples.collected)
        truncate = state.step_count >= self.time_limit

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

        timestep.extras = self._get_extra_info(state, timestep)

        return state, timestep

    def _get_extra_info(self, state: SimpleOrchardState, timestep: TimeStep) -> Dict:
        """Computes extras metrics to be returned within the timestep."""
        n_collected = state.apples.collected.sum() + timestep.extras.get(
            "collected_food", jnp.float32(0)
        )

        percent_collected = (n_collected / self.num_apples) * 100
        return {"percent_collected": percent_collected}

    # MODIFIED CODE FROM ORIGINAL. 
    # naming conventions
    # treated code the same but with the original 'level' logic set to be 1 for all entities. 
    def get_reward(
        self,
        apples: SimpleOrchardApple,
        all_food_collected: chex.Array,
        collected_this_step: chex.Array,
    ) -> chex.Array:
        """Returns a reward for all agents given all food items.

        Args:
            apples (SimpleOrchardApple): All the apples in the environment.
            all_food_collected (chex.Array): Count of all agents adjacent to all foods. 
            collected_this_step (chex.Array): Whether the apple was collected or not (this step). Boolean array of len num_apples
        """

        def get_reward_per_food(
            apple: SimpleOrchardApple,
            all_food_collected: chex.Array,
            collected_this_step: chex.Array,
        ) -> chex.Array:
            """Returns the reward for all agents given a single apple."""

            # If the food has already been eaten or is not loaded, the sum will be equal to 0
            sum_agents = jnp.sum(all_food_collected)
            
            # penalize agents for not being near an apple
            no_apple_penalty = jnp.where(
                (all_food_collected == 0),
                self.penalty,
                0,
            )
            
            # # penalize agents for trying to pick up apples when there isn't one
            # invalid_load_penalty = jnp.where(
            #     agent.loading & (collected_this_step == 0), 2 * self.penalty, 0
            # )

            # Zero out all agents if food was not collected and add penalty
            reward = (all_food_collected * collected_this_step) - no_apple_penalty # - invalid_load_penalty
            
            # jnp.nan_to_num: Used in the case where no agents are adjacent to the food
            normalizer = sum_agents * self.num_apples
            reward = jnp.where(
                self.normalize_reward, jnp.nan_to_num(reward / normalizer), reward
            )
            return reward

        # Get reward per food for all food items,
        # then sum it on the agent dimension to get reward per agent.
        reward_per_food = jax.vmap(get_reward_per_food, in_axes=(0, 0, 0))(
            apples, all_food_collected, collected_this_step
        )
        return jnp.sum(reward_per_food, axis=0) 
 
    ######## render and animate functions are ommitted ######################
    
    # removed levels
    def observation_spec(self) -> specs.Spec[Observation]:
        """Specifications of the observation of the environment.

        Returns:
            specs.Spec[Observation]: Spec for the `Observation` with fields grid,
            action_mask, and step_count.
        """
        return self._observer.observation_spec(
            self.time_limit,
        )
    
    # copied and renamed
    def action_spec(self) -> specs.MultiDiscreteArray:
        """Returns the action spec for the Level Based Foraging environment.

        Returns:
            specs.MultiDiscreteArray: Action spec for the environment with shape (num_agents,).
        """
        return specs.MultiDiscreteArray(
            num_values=jnp.array([len(MOVES)] * self.num_bots),
            dtype=jnp.int32,
            name="action",
        )
    
    # copied and renamed
    def reward_spec(self) -> specs.Array:
        """Returns the reward specification for the `LevelBasedForaging` environment.

        Since this is a multi-agent environment each agent gets its own reward.

        Returns:
            specs.Array: Reward specification, of shape (num_agents,) for the  environment.
        """
        return specs.Array(shape=(self.num_bots,), dtype=float, name="reward")
    
    # copied and renamed
    def discount_spec(self) -> specs.BoundedArray:
        """Describes the discount returned by the environment.

        Returns:
            discount_spec: a `specs.BoundedArray` spec.
        """
        return specs.BoundedArray(
            shape=(self.num_bots,),
            dtype=float,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )

################### THE FOLLOWING FUNCTIONS WERE BROUGHT OVER FROM `lbf.utils.py` ###########################################

    # identical besides state naming conventions
    def are_entities_adjacent(self, entity_a: SimpleOrchardEntity, entity_b: SimpleOrchardEntity) -> chex.Array:
        """
        Check if two entities are adjacent in the grid.

        Args:
            entity_a (SimpleOrchardEntity): The first entity.
            entity_b (SimpleOrchardEntity): The second entity.

        Returns:
            chex.Array: True if entities are adjacent, False otherwise.
        """
        distance = jnp.abs(entity_a.position - entity_b.position)
        return jnp.sum(distance) == 1
    
    #identical
    def _flag_duplicates(self, a: chex.Array) -> chex.Array:
        """Return a boolean array indicating which elements of `a` are duplicates.

        Example:
            a = jnp.array([1, 2, 3, 2, 1, 5])
            flag_duplicates(a)  # jnp.array([True, False, True, False, True, True])
        """
        # https://stackoverflow.com/a/11528078/5768407
        _, indices, counts = jnp.unique(
            a, return_inverse=True, return_counts=True, size=len(a), axis=0
        )
        return ~(counts[indices] == 1)
    
    # naming conventions changed
    # out of bounds code modified for width & height
    def _simulate_agent_movement(
        self, agent: SimpleOrchardAgent, action: chex.Array, apples: SimpleOrchardApple, agents: SimpleOrchardAgent
    ) -> SimpleOrchardAgent:
        """
        Move the agent based on the specified action.

        Args:
            agent (SimpleOrchardAgent): The agent to move.
            action (chex.Array): The action to take.
            apples (SimpleOrchardApple): All apples in the grid.
            agents (SimpleOrchardAgent): All agents in the grid.

        Returns:
            Agent: The agent with its updated position.
        """

        # Calculate the new position based on the chosen action
        new_position = agent.position + MOVES[action]
        
        # Check if the new position is out of bounds
        out_of_bounds = jnp.any((new_position < 0) | (new_position[0] >= self.width) | (new_position[1] >= self.height))

        # Check if the new position is occupied by food or another agent
        agent_at_position = jnp.any(
            jnp.all(new_position == agents.position, axis=1) & (agent.id != agents.id)
        )
        apple_at_position = jnp.any(
            jnp.all(new_position == apples.position, axis=1) & ~apples.collected
        )
        entity_at_position = jnp.any(agent_at_position | apple_at_position)

        # Move the agent to the new position if it's a valid position,
        # otherwise keep the current position
        new_agent_position = jnp.where(
            out_of_bounds | entity_at_position, agent.position, new_position
        )

        # Return the agent with the updated position
        return agent.replace(position=new_agent_position)  # type: ignore        )

    # adapted naming conventions
    # grid size removed, height and width being used implictly 
    def _update_agent_positions(
        self, agents: SimpleOrchardAgent, actions: chex.Array, apples: SimpleOrchardApple
    ) -> Any:
        """
        Update agent positions based on actions and resolve collisions.

        Args:
            agents (SimpleOrchardAgent): The current state of agents.
            actions (chex.Array): Actions taken by agents.
            apples (SimpleOrchardApple): All apples in the grid.

        Returns:
            Agent: Agents with updated positions.
        """
        # Move the agent to a valid position
        moved_agents = jax.vmap(self._simulate_agent_movement, (0, 0, None, None))(
            agents,
            actions,
            apples,
            agents
        )

        # Fix collisions
        moved_agents = self._fix_collisions(moved_agents, agents)
        
        moved_agents = jax.vmap(lambda agent, action: agent.replace(loading=(action == LOAD)))(
            moved_agents, actions
        )

        return moved_agents
    

    # adapted naming conventions
    # omission of agent level from return
    def _fix_collisions(self, moved_agents: SimpleOrchardAgent, original_agents: SimpleOrchardAgent) -> SimpleOrchardAgent:
        """
        Fix collisions in the moved agents by resolving conflicts with the original agents.
        If a number 'N' of agents end up in the same position after the move, the initial
        position of the agents is retained.

        Args:
            moved_agents (Agent): Agents with potentially updated positions.
            original_agents (Agent): Original agents with their initial positions.

        Returns:
            Agent: Agents with collisions resolved.
        """
        # Detect duplicate positions
        duplicates = self._flag_duplicates(moved_agents.position)
        duplicates = duplicates.reshape((duplicates.shape[0], -1))

        # If there are duplicates, use the original agent position.
        new_positions = jnp.where(
            duplicates,
            original_agents.position,
            moved_agents.position,
        )

        # Recreate agents with new positions
        agents: SimpleOrchardAgent = jax.vmap(SimpleOrchardAgent)(
            id=original_agents.id,
            position=new_positions,
            loading=original_agents.loading,
        )
        return agents    


    # UTIL COPY BUT WITH HEAVY MODIFICATIONS, 
    # LIKELY SOURCE FOR TRAINING ISSUES. . .
    # see notes within function for more details
    def _collect_food(self, agents: SimpleOrchardAgent, apple: SimpleOrchardApple) -> Tuple[SimpleOrchardApple, chex.Array]:
        """Try to eat the provided food if possible.

        Args:
            agents(Agent): All agents in the grid.
            apple(SimpleOrchardApple): The food to attempt to eat.

        Returns:
            new_food (Food): Updated state of the food, indicating whether it was eaten.
            is_food_eaten_this_step (chex.Array): Whether or not the food was eaten at this step.
        """
        
        # instead of getting adjacent agent levels, we are returning a static 1
        # this is acting as our check for agents that can and try to collect
        def is_collected(agent: SimpleOrchardAgent, apple: SimpleOrchardApple) -> chex.Array:
            """Return 1 if the agent is adjacent to the food, else 0."""
            return jax.lax.select(
                self.are_entities_adjacent(agent, apple) & agent.loading & ~apple.collected,
                1,
                0,
            )

        # getting all adjacent agents that are trying to load food
        all_food_collected = jax.vmap(is_collected, (0, None))(agents, apple)

        # If the food has already been collected or is not loaded, the sum will be equal to 0
        food_collected_this_step = jnp.sum(all_food_collected) > 0

        # Set food to collected if it was collected
        new_food = apple.replace(collected=food_collected_this_step | apple.collected)  # type: ignore
        
        # original function has a third return of agent levels. leaving it here, because it factors into partial rewards
        return new_food, food_collected_this_step, all_food_collected 

    # function originally missing from util copy over
    # updated for proper naming conventions
    # modified to address height and width
    def compute_action_mask(agent: SimpleOrchardAgent, state: SimpleOrchardState) -> chex.Array:
        """
        Calculate the action mask for a given agent based on the current state.

        Args:
            agent (Agent): The agent for which to calculate the action mask.
            state (State): The current state of the environment.

        Returns:
            chex.Array: A boolean array representing the action mask for the given agent,
                where `True` indicates a valid action, and `False` indicates an invalid action.
        """
        next_positions = agent.position + MOVES

        def check_pos_fn(next_pos: Any, entities: SimpleOrchardEntity, condition: bool) -> Any:
            return jnp.any(jnp.all(next_pos == entities.position, axis=-1) & condition)

        # Check if any agent is in a next position.
        agent_occupied = jax.vmap(check_pos_fn, (0, None, None))(
            next_positions, state.bots, (state.bots.id != agent.id)
        )

        # Check if any food is in a next position (Food must be uneaten)
        food_occupied = jax.vmap(check_pos_fn, (0, None, None))(
            next_positions, state.apples, ~state.apples.collected
        )
        # Check if the next position is out of bounds
        out_of_bounds = jnp.any((next_positions < 0) | (new_position[0] >= self.width) | (new_position[1] >= self.height))

        action_mask = ~(food_occupied | agent_occupied | out_of_bounds)

        # Check if the agent can load food (if placed in the neighborhood)
        num_adj_food = (
            jax.vmap(are_entities_adjacent, (0, None))(state.apples, agent)
            & ~state.apples.collected
        )
        is_food_adj = jnp.sum(num_adj_food) > 0

        action_mask = jnp.where(is_food_adj, action_mask, action_mask.at[-1].set(False))

        return action_mask