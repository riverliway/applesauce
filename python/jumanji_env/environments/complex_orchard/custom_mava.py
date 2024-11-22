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
    RecordEpisodeMetrics,
    AutoResetWrapper,
)
# pulling in the Jumanji mult-agent wrapper
from mava.wrappers.jumanji import JumanjiMarlWrapper

# our custom packages
from jumanji_env.environments.complex_orchard.generator import ComplexOrchardGenerator
from jumanji_env.environments.complex_orchard.env import ComplexOrchard

# This function replaces functionality of Mava's `mava.utils.make_env` functionality. 
# right now this just contains the RecordEpisodeMetrics wrapper and does not contain an equivalent
# to the LbfWrapper
def make_env(env_name: str, config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
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
    num_picker_bots = config.env.scenario.task_config.num_picker_bots
    num_pusher_bots = config.env.scenario.task_config.num_pusher_bots
    num_baskets = config.env.scenario.task_config.num_baskets

    # assigning the generator
    generator = ComplexOrchardGenerator(width=width, height=height, num_picker_bots=num_picker_bots, num_pusher_bots=num_pusher_bots, num_baskets=num_baskets)
 
    # making the environments
    train_env = jumanji.make(env_name, generator=generator)
    eval_env = jumanji.make(env_name, generator=generator)
    
    # adding auto reset wrapper to train environment
    train_env = AutoResetWrapper(train_env)
    
    # adding the wrappers that log evaluation metrics.
    train_env = RecordEpisodeMetrics(train_env)
    eval_env = RecordEpisodeMetrics(eval_env)
    
    # adding the orchard specific wrapper below
    train_env = ComplexOrchardWrapper(train_env, add_global_state=add_global_state)
    eval_env = ComplexOrchardWrapper(eval_env, add_global_state=add_global_state)

    return train_env, eval_env

# This function replaces the functionality of Mavas `mava.wrappers.jumanji.LbfWrapper` functionality. 
# Only update necessary was to change the environment type to ComplexOrchard. 
class ComplexOrchardWrapper(JumanjiMarlWrapper):
    """
     Multi-agent wrapper for our ComplexOrchard environment.

    Args:
        env (Environment): The base environment.
        use_individual_rewards (bool): If true each agent gets a separate reward,
        sum reward otherwise.
    """

    def __init__(self, 
                 env: ComplexOrchard, 
                 add_global_state: bool = False,
                 use_individual_rewards: bool = False,
                ):
        super().__init__(env, add_global_state)
        self._env: ComplexOrchard
        self._use_individual_rewards = use_individual_rewards

    def aggregate_rewards(
        self, timestep: TimeStep, observation: Observation
    ) -> TimeStep[Observation]:
        """Aggregate individual rewards across agents."""
        team_reward = jnp.sum(timestep.reward)

        # Repeat the aggregated reward for each agent.
        reward = jnp.repeat(team_reward, self.num_agents)
        return timestep.replace(observation=observation, reward=reward)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for Level-Based Foraging environment and update
        the reward based on the specified reward handling strategy."""

        # Create a new observation with adjusted step count
        modified_observation = Observation(
            agents_view=timestep.observation.agents_view.astype(float),
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self.num_bots),
        )
        if self._use_individual_rewards:
            # The environment returns a list of individual rewards and these are used as is.
            return timestep.replace(observation=modified_observation)

        # Aggregate the list of individual rewards and use a single team_reward.
        return self.aggregate_rewards(timestep, modified_observation)