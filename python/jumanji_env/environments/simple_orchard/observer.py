### The following has been adapted from `jumanji/jumanji/environments/routing/lbf/observer.py ####

import abc
from typing import Any, Tuple, Union

import chex
import jax
import jax.numpy as jnp

### NEED TO UPDATE FOR OUR CODE ###
# from our modified files
from jumanji_env.environments.simple_orchard.constants import MOVES
from jumanji_env.environments.simple_orchard.orchard_types import (
    SimpleOrchardEntity,
    SimpleOrchardApple,
    SimpleOrchardObservation,
    SimpleOrchardState,
    SimpleOrchardAgent,
)
# directly from jumanji 
from jumanji import specs
from jumanji.environments.routing.lbf import utils


# This replaces VectorObserver(LbfObserver) Class
# Instead of the class & sub-class structure they had to allow for grid and vector observation, we opted for a single vector observation
class SimpleOrchardObserver:
    """
    This class vectorizes the SimpleOrchardEnvironment to pass into the neural network
    """

    def __init__(self, fov: int, width: int, height: int, num_bots: int, num_apples: int, num_trees: int) -> None:
        """
        Initalizes the observer object
        """

        self.fov = fov
        self.width = width
        self.height = height
        self.num_bots = num_bots
        self.num_apples = num_apples
        self.num_trees = num_trees

    # We have removed self.grid_size as an input to the compute_action_mask, so that was removed here.
    # As noted for `compute_action_mask`, it does not exist in main repo for jumanji, need to better understand how
    # this function was structured without it. 
    def state_to_observation(self, state: SimpleOrchardState) -> SimpleOrchardObservation:
        """
        Convert the current state of the environment into observations for all agents.

        Args:
            state (State): The current state containing agent and food information.

        Returns:
            Observation: An Observation object containing the agents' views, action masks,
            and step count for all agents.
        """
        # Create the observation
        agents_view = jax.vmap(self.make_observation, (0, None))(state.bots, state)

        # Compute the action mask
        action_mask = jax.vmap(self.compute_action_mask, (0, None))(state.bots, state)

        return SimpleOrchardObservation(
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=state.step_count,
        )
    # same
    def observation_spec(self, time_limit: int) -> specs.Spec[SimpleOrchardObservation]:
        """
        Returns the observation spec for the environment
        """
        max_observation = jnp.max(jnp.array([self.width, self.height]))
        agents_view = specs.BoundedArray(
            shape=(self.num_bots, 2 * (self.num_bots + self.num_trees + self.num_apples)),
            dtype=jnp.int32,
            name="agents_view",
            minimum=-1,
            maximum=max_observation,
        )

        return specs.Spec(
            SimpleOrchardObservation,
            "SimpleOrchardObservationSpec",
            agents_view=agents_view,
            action_mask=self._action_mask_spec(),
            step_count=self._time_spec(time_limit),
        )

    # same
    def _action_mask_spec(self) -> specs.BoundedArray:
        """
        Returns the action spec for the environment.

        The action mask is a boolean array of shape (num_agents, 6). '6' is the number of actions.
        """
        return specs.BoundedArray(
            shape=(self.num_bots, 6),
            dtype=bool,
            minimum=False,
            maximum=True,
            name="action_mask"
        )

    # replaces `_step_count_spec`
    # same besides naming conventions
    def _time_spec(self, time_limit: int) -> specs.BoundedArray:
        """Returns the step count spec for the environment."""
        return specs.BoundedArray(
            shape=(),
            dtype=jnp.int32,
            minimum=0,
            maximum=time_limit,
            name="step_count",
        )

    # same
    def transform_positions(
        self, agent: SimpleOrchardEntity, items: SimpleOrchardEntity
    ) -> chex.Array:
        """
        Calculate the positions of items within the agent's field of view.

        Args:
            agent (Agent): The agent whose position is used as the reference point.
            items: The items (other Agents, trees, and apples) to be transformed.

        Returns:
            chex.Array: The transformed positions of the items.
        """

        # TODO: Understand why this is necessary
        min_x = jnp.minimum(self.fov, agent.position[0])
        min_y = jnp.minimum(self.fov, agent.position[1])
        return items.position - agent.position + jnp.array([min_x, min_y])

    # replaces `extract_food_info`
    # changed output to tuple of 2, due to removal of levels
    def extract_apples_info(
        self, agent: SimpleOrchardEntity, visible_apples: chex.Array, all_apples: SimpleOrchardApple
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Extract the positions of visible apples.

        Args:
            agent (Agent): The agent observing the apples.
            visible_apples (chex.Array): A boolean array indicating the visibility of apples.
            all_apples (SimpleOrchardApple): Containing information about all the apples.

        Returns:
            Tuple[chex.Array, chex.Array]: A tuple of 1D arrays of the Xs and Ys of the apple positions
        """
        transformed_positions = self.transform_positions(agent, all_apples)

        apple_xs = jnp.where(visible_apples, transformed_positions[:, 0], -1)
        apple_ys = jnp.where(visible_apples, transformed_positions[:, 1], -1)

        return apple_xs, apple_ys

    # new to our class
    # follows same logic as apples function
    def extract_trees_info(
        self, agent: SimpleOrchardEntity, visible_trees: chex.Array, all_trees: SimpleOrchardEntity
    ) -> Tuple[chex.Array, chex.Array]:
        """
        Extract the positions of visible apples.

        Args:
            agent (Agent): The agent observing the apples.
            visible_trees (chex.Array): A boolean array indicating the visibility of trees.
            all_trees (SimpleOrchardEntity): Containing information about all the trees.

        Returns:
            Tuple[chex.Array, chex.Array]: A tuple of 1D arrays of the Xs and Ys of the tree positions
        """
        transformed_positions = self.transform_positions(agent, all_trees)

        tree_xs = jnp.where(visible_trees, transformed_positions[:, 0], -1)
        tree_ys = jnp.where(visible_trees, transformed_positions[:, 1], -1)

        return tree_xs, tree_ys

    # output reduced to a tuple of 3 for removal of agent levels
    def extract_agents_info(
        self, agent: SimpleOrchardEntity, visible_agents: chex.Array, all_agents: SimpleOrchardEntity
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """
        Extract the positions and levels of visible agents excluding the current agent.

        Args:
            agent (Agent): The current agent.
            visible_agents (chex.Array): A boolean array indicating the visibility of other agents.
            all_agents (Agent): Containing information about all agents.

        Returns:
            Tuple[chex.Array, chex.Array, chex.Array]: A tuple of 1D arrays of the Xs and Ys of the agent positions
        """
        transformed_positions = self.transform_positions(agent, all_agents)
        agent_xs = jnp.where(visible_agents, transformed_positions[:, 0], -1)
        agent_ys = jnp.where(visible_agents, transformed_positions[:, 1], -1)

        # Remove the current agent's info from all agent's infos.
        agent_i_index = jnp.where(agent.id == all_agents.id, size=1)
        agent_i_infos = jnp.array(
            [
                agent_xs[agent_i_index],
                agent_ys[agent_i_index]
            ]
        ).ravel()

        other_agents_indices = jnp.where(
            agent.id != all_agents.id, size=self.num_bots - 1
        )
        agent_xs = agent_xs[other_agents_indices]
        agent_ys = agent_ys[other_agents_indices]

        return agent_i_infos, agent_xs, agent_ys


    # EXISTS ON LINE 211 OF SASHA REPO, NOT IN MAIN REPO
    # NEED TO REVIEW FURTHER
    def compute_action_mask(self, agent: SimpleOrchardEntity, state: SimpleOrchardState) -> chex.Array:
        """
        Calculate the action mask for a given agent based on the current state.

        Args:
            agent (Agent): The agent for which to calculate the action mask.
            state (State): The current state of the environment,
                        containing agent and food information.

        Returns:
            chex.Array: A boolean array representing the action mask for the given agent,
            where `True` indicates a valid action, and `False` indicates an invalid action.
        """
        # TODO: Actually implement a good action mask instead of letting the bot do whatever

        return jnp.ones((6,), dtype=bool)

    #  This replaces `make_agents_view`
    # remove levels
    # adjust for height and width
    def make_observation(self, agent: SimpleOrchardEntity, state: SimpleOrchardState) -> chex.Array:
        """
        Make an observation for a single agent based on the current state of the environment.

        Args:
            agent (Agent): The agent for which to make the observation and action mask.
        Returns:
            agent_view (chex.Array): The observation for the given agent.
        """
        INFO_PER_ENTITY = 2 # There are two floats passed into the network, the X and Y coord of the entity

        # Calculate which agents are within the field of view (FOV) of the current agent
        # and are not the current agent itself.
        visible_agents = jnp.all(
            jnp.abs(agent.position - state.bots.position) <= self.fov,
            axis=-1,
        )

        visible_trees = jnp.all(
            jnp.abs(agent.position - state.trees.position) <= self.fov,
            axis=-1,
        )

        # Calculate which apples are within the FOV of the current agent and are not eaten.
        visible_apples = (
            jnp.all(
                jnp.abs(agent.position - state.apples.position) <= self.fov,
                axis=-1,
            )
            & ~state.apples.collected
        )

        # Placeholder observation for apppes, trees, and agents
        # this is shown if an entity is not in view.
        init_vals = jnp.array([-1 for _ in range(INFO_PER_ENTITY)])
        agent_view = jnp.tile(init_vals, self.num_apples + self.num_trees + self.num_bots)

        # Extract the positions of visible apples.
        apple_xs, apple_ys = self.extract_apples_info(
            agent, visible_apples, state.apples
        )

        # Extract the positions and levels of visible agents.
        agent_i_infos, agent_xs, agent_ys = self.extract_agents_info(
            agent, visible_agents, state.bots
        )

        tree_xs, tree_ys = self.extract_trees_info(
            agent, visible_trees, state.trees
        )

        end_apple_idx = INFO_PER_ENTITY * self.num_apples
        end_tree_idx = end_apple_idx + INFO_PER_ENTITY * self.num_trees

        # Assign the foods and agents infos.
        agent_view = agent_view.at[jnp.arange(0, end_apple_idx, INFO_PER_ENTITY)].set(apple_xs)
        agent_view = agent_view.at[jnp.arange(1, end_apple_idx, INFO_PER_ENTITY)].set(apple_ys)
        agent_view = agent_view.at[jnp.arange(end_apple_idx, end_tree_idx, INFO_PER_ENTITY)].set(tree_xs)
        agent_view = agent_view.at[jnp.arange(end_apple_idx + 1, end_tree_idx, INFO_PER_ENTITY)].set(tree_ys)

        # Always place the current agent's info first.
        agent_view = agent_view.at[
            jnp.arange(end_tree_idx, end_tree_idx + INFO_PER_ENTITY)
        ].set(agent_i_infos)

        start_idx = end_tree_idx + INFO_PER_ENTITY
        end_idx = start_idx + INFO_PER_ENTITY * (self.num_bots - 1)
        agent_view = agent_view.at[jnp.arange(start_idx, end_idx, INFO_PER_ENTITY)].set(agent_xs)
        agent_view = agent_view.at[jnp.arange(start_idx + 1, end_idx, INFO_PER_ENTITY)].set(agent_ys)

        return agent_view