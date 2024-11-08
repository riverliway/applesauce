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
from constants import MOVES
from generator import SimpleOrchardGenerator
from observer import SimpleOrchardObserver
from orchard_types import SimpleOrchardApple, SimpleOrchardObservation, SimpleOrchardState, SimpleOrchardEntity

# directly from jumanji
import jumanji.environments.routing.lbf.utils as utils
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.lbf.viewer import LevelBasedForagingViewer
from jumanji.types import TimeStep, restart, termination, transition, truncation
from jumanji.viewer import Viewer

# we are still calling Observation in functions so importing. Assuming this needs to be addressed.
from mava.types import Observation



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

    # why removal of grid observation
    # noticed fov is hard set which explains why it is not in generator
    # removal of viewer
    # added super.init -- SASHA REPO

    def __init__(
        self,
        generator: Optional[SimpleOrchardGenerator] = None,
        time_limit: int = 100,
        normalize_reward: bool = True,
        penalty: float = 0.0,
    ) -> None:
        super().__init__()

        self._generator = generator or SimpleOrchardGenerator(
            width=10,
            height=11
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

    ## FROM UTIL FILE AND BROUGHT IN ##
    ## adapted for width and height, naming conventions
    def _simulate_agent_movement(
        self, agent: SimpleOrchardEntity, action: chex.Array, apples: SimpleOrchardApple, agents: SimpleOrchardEntity
    ) -> SimpleOrchardEntity:
        """
        Move the agent based on the specified action.

        Args:
            agent (SimpleOrchardEntity): The agent to move.
            action (chex.Array): The action to take.
            apples (SimpleOrchardApple): All apples in the grid.
            agents (SimpleOrchardEntity): All agents in the grid.

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
        return SimpleOrchardEntity(
            id=agent.id,
            position=new_agent_position
        )

   # UTILS
   # Copy
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

# UTIL
# COPY
    def _fix_collisions(self, moved_agents: SimpleOrchardEntity, original_agents: SimpleOrchardEntity) -> SimpleOrchardEntity:
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
        agents: SimpleOrchardEntity = jax.vmap(SimpleOrchardEntity)(
            id=original_agents.id,
            position=new_positions
        )
        return agents

# UTIL
# naming conventions
    def _update_agent_positions(
        self, agents: SimpleOrchardEntity, actions: chex.Array, apples: SimpleOrchardApple
    ) -> Any:
        """
        Update agent positions based on actions and resolve collisions.

        Args:
            agents (SimpleOrchardEntity): The current state of agents.
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

        return moved_agents

# UTIL COPY
    def _are_entities_adjacent(self, entity_a: SimpleOrchardEntity, entity_b: SimpleOrchardEntity) -> chex.Array:
        """
        Check if two entities are adjacent in the grid.

        Args:
            entity_a (SimpleOrchardEntity): The first entity.
            entity_b (SimpleOrchardEntity): The second entity.

        Returns:
            chex.Array: True if entities are adjacent, False otherwise.
        """
        distance = jnp.abs(entity_a.position - entity_b.position)
        return jnp.where(jnp.sum(distance) == 1, True, False)

# UTIL COPY BUT WITH HEAVY MODIFICATIONS
    def _eat_food(self, agents: SimpleOrchardEntity, apple: SimpleOrchardApple) -> Tuple[SimpleOrchardApple, chex.Array]:
        """Try to eat the provided food if possible.

        Args:
            agents(Agent): All agents in the grid.
            apple(SimpleOrchardApple): The food to attempt to eat.

        Returns:
            new_food (Food): Updated state of the food, indicating whether it was eaten.
            is_food_eaten_this_step (chex.Array): Whether or not the food was eaten at this step.
        """

        def is_eaten(agent: SimpleOrchardEntity, food: SimpleOrchardApple) -> chex.Array:
            """Return 1 if the agent is adjacent to the food, else 0."""
            return jax.lax.select(
                self._are_entities_adjacent(agent, food) & ~food.collected, #ERIK ADD #################################################
                1,
                0,
            )

        # Get the level of all adjacent agents that are trying to load the food
        adj_loading_agents_levels = jax.vmap(is_eaten, (0, None))(agents, apple) # ERIK ADD ############################################

        # If the food has already been eaten or is not loaded, the sum will be equal to 0
        is_food_eaten_this_step = jnp.sum(adj_loading_agents_levels) >= 0

        # Set food to eaten if it was eaten.
        new_food = apple.replace(collected=is_food_eaten_this_step | apple.collected)  # type: ignore

        return new_food, is_food_eaten_this_step

# COPY
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

        # Eat the food
        new_apples, eaten_this_step = jax.vmap(
            self._eat_food, (None, 0)
        )(moved_agents, state.apples)
        print("eaten_this_step:", eaten_this_step)
        print("new_apples:", new_apples)
        reward = self.get_reward(new_apples, eaten_this_step)

        state = SimpleOrchardState(
            bots=moved_agents,
            apples=new_apples,
            trees=state.trees, ############ERIK ADD######################################
            time=state.time + 1,
            key=state.key,
        )
        observation = self._observer.state_to_observation(state)

        # First condition is truncation, second is termination.
        terminate = jnp.all(state.apples.collected)
        truncate = state.time >= self.time_limit

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
        n_eaten = state.apples.collected.sum() + timestep.extras.get(
            "eaten_food", jnp.float32(0)
        )

        percent_eaten = (n_eaten / self.num_apples) * 100
        return {"percent_eaten": percent_eaten}

# MODIFIED CODE FROM ORIGINAL. LIKELY CAUSE OF SHAPE ISSUE.
    def get_reward(
        self,
        apples: SimpleOrchardApple,
        eaten_this_step: chex.Array,
    ) -> chex.Array:
        """Returns a reward for all agents given all food items.

        Args:
            apples (SimpleOrchardApple): All the apples in the environment.
            eaten_this_step (chex.Array): Whether the apple was eaten or not (this step). Boolean array of len num_apples
        """

        def get_reward_per_food(
            apple: SimpleOrchardApple,
            eaten_this_step: chex.Array,
        ) -> chex.Array:
            """Returns the reward for all agents given a single apple."""

            # Zero out all agents if food was not eaten and add penalty
            reward = (eaten_this_step - self.penalty) * jnp.ones(self.num_bots)

            # jnp.nan_to_num: Used in the case where no agents are adjacent to the food
            normalizer = self.num_apples
            reward = jnp.where(
                self.normalize_reward, jnp.nan_to_num(reward / normalizer), reward
            )

            return reward

        # Get reward per food for all food items,
        # then sum it on the agent dimension to get reward per agent.
        reward_per_food = jax.vmap(get_reward_per_food, in_axes=(0, 0))(
            apples, eaten_this_step
        )
        print("reward_per_food shape before sum:", reward_per_food.shape)
        print("after sum:", jnp.sum(reward_per_food, axis=0).shape)
        return jnp.sum(reward_per_food, axis=0) #, keepdims=True).reshape(4, 3)
 # copied and renamed
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


###### Erik added to the class for rendering purposes ##############################