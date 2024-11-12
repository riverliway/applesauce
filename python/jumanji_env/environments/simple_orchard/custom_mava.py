# this contains modifications to mava functions necessary for us to work with our apple orchard environment. 

# importing packages
import chex
import jax.numpy as jnp
from typing import Tuple
from omegaconf import DictConfig
import jumanji
from jumanji.types import TimeStep
from mava.types import MarlEnv, Observation, State
from mava.wrappers import (
    RecordEpisodeMetrics
)
# pulling in the Jumanji mult-agent wrapper
from mava.wrappers.jumanji import JumanjiMarlWrapper

# our custom packages
from jumanji_env.environments.simple_orchard.generator import SimpleOrchardGenerator
from jumanji_env.environments.simple_orchard.env import SimpleOrchard

# This function replaces functionality of Mava's `mava.utils.make_env` functionality. 
# right now this just contains the RecordEpisodeMetrics wrapper and does not contain an equivalent
# to the LbfWrapper
def make_env(env_name: str, config: DictConfig) -> Tuple[MarlEnv, MarlEnv]:
    """
    Create Jumanji environments for training and evaluation.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
    -------
        A tuple of the environments.

    """
    # retreiving the attribuges from config file
    width = config.env.scenario.task_config.width
    height = config.env.scenario.task_config.height
    num_bots = config.env.scenario.task_config.num_bots
    num_trees = config.env.scenario.task_config.num_trees
    num_apples = config.env.scenario.task_config.num_apples

    # assigning the generator
    generator = SimpleOrchardGenerator(width=width, height=height, num_bots=num_bots, num_trees=num_trees, num_apples=num_apples)
 
    # making the environments
    train_env = jumanji.make(env_name, generator=generator)
    eval_env = jumanji.make(env_name, generator=generator)

    # adding the wrappers that log evaluation metrics.
    train_env = RecordEpisodeMetrics(train_env)
    eval_env = RecordEpisodeMetrics(eval_env)
    
    # adding the orchard specific wrapper below
    train_env = SimpleOrchardWrapper(train_env)
    eval_env = SimpleOrchardWrapper(eval_env)

    return train_env, eval_env

# This function replaces the functionality of Mavas `mava.wrappers.jumanji.LbfWrapper` functionality. 
# Only update necessary was to change the environment type to SimpleOrchard. 
class SimpleOrchardWrapper(JumanjiMarlWrapper):
    """
     Multi-agent wrapper for the Level-Based Foraging environment.

    Args:
        env (Environment): The base environment.
        use_individual_rewards (bool): If true each agent gets a separate reward,
        sum reward otherwise.
    """

    def __init__(self, 
                 env: SimpleOrchard, 
                 add_global_state: bool = False,
                 use_individual_rewards: bool = False,
                ):
        super().__init__(env, add_global_state)
        self._env: SimpleOrchard
        self._use_individual_rewards = use_individual_rewards

    def aggregate_rewards(
        self, timestep: TimeStep, observation: Observation
    ) -> TimeStep[Observation]:
        """Aggregate individual rewards across agents."""
        team_reward = jnp.sum(timestep.reward)

        # Repeat the aggregated reward for each agent.
        reward = jnp.repeat(team_reward, self._num_agents)
        return timestep.replace(observation=observation, reward=reward)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for Level-Based Foraging environment and update
        the reward based on the specified reward handling strategy."""

        # Create a new observation with adjusted step count
        modified_observation = Observation(
            agents_view=timestep.observation.agents_view.astype(float),
            action_mask=timestep.observation.action_mask,
            time=jnp.repeat(timestep.observation.time, self._num_agents),
        )
        if self._use_individual_rewards:
            # The environment returns a list of individual rewards and these are used as is.
            return timestep.replace(observation=modified_observation)

        # Aggregate the list of individual rewards and use a single team_reward.
        return self.aggregate_rewards(timestep, modified_observation)