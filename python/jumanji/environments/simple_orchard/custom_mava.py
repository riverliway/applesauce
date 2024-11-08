# this contains modifications to mava functions necessary for us to work with our apple orchard environment. 


from omegaconf import DictConfig
from generator import SimpleOrchardGenerator
import jumanji
from mava.wrappers import (
    RecordEpisodeMetrics
)

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
    train_env = jumanji.make('SimpleOrchard-v0', generator=generator)
    eval_env = jumanji.make('SimpleOrchard-v0', generator=generator)

    # adding the wrappers that log evaluation metrics.
    train_env = RecordEpisodeMetrics(train_env)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env