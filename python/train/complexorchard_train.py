# for registering environments
from jumanji import register

# below are the needed mava packages within notebook
from mava.evaluator import get_eval_fn, make_ff_eval_act_fn

# other needed packages
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import shutil
import os
import sys
import importlib
import inspect
from IPython.display import Image, display

#importing our custom jumanji (and mava) packages
from jumanji_env.environments.complex_orchard.env import ComplexOrchard
from jumanji_env.environments.complex_orchard.orchard_types import ComplexOrchardObservation, ComplexOrchardState, ComplexOrchardEntity
from jumanji_env.environments.complex_orchard.generator import ComplexOrchardGenerator
from jumanji_env.environments.complex_orchard import custom_mava
importlib.reload(custom_mava)
from jumanji_env.environments.complex_orchard.custom_mava import make_env
from apple_mava.ff_networks import Actor, Critic
from apple_mava.render_logging import *
from apple_mava.ff_mappo import *
from config import config

# this is a workaround to get python to call the correct make_env. It was pulling the simple version first.
importlib.reload(custom_mava)
from jumanji_env.environments.complex_orchard.custom_mava import make_env
print(inspect.getfile(make_env))

# Convert the Python dictionary to a DictConfig
config: DictConfig = OmegaConf.create(config)

# Convert config to baseline for easy print out review
config = apply_baseline_config(config, use_baseline=True)

# File to store the current version
version_file = "orchard_version.txt"

# Function to get the next version number
def get_next_version():
    if os.path.exists(version_file):
        with open(version_file, "r") as file:
            current_version = int(file.read().strip())
    else:
        current_version = 0  # Default to version 0 if the file doesn't exist

    next_version = current_version + 1

    # Save the next version to the file
    with open(version_file, "w") as file:
        file.write(str(next_version))

    return next_version

# Get the next version number
version_number = get_next_version()

# Assign orchard name with the updated version
orchard_version_name = f'ComplexOrchard-v{version_number}'

# Register the orchard
register(
    id=orchard_version_name,
    entry_point='__main__:ComplexOrchard',
)

print(f"Registered orchard version: {orchard_version_name}")

# make the training and evaluation orchards
env, eval_env = make_env(orchard_version_name, config, add_global_state=True)

# PRNG keys.
key = jax.random.PRNGKey(config.system.seed)
key, key_e, actor_net_key, critic_net_key = jax.random.split(key, num=4)

# Setup learner.
learn, actor_network, learner_state = learner_setup(
    env, (key, actor_net_key, critic_net_key), config
)

eval_act_fn = make_ff_eval_act_fn(actor_network.apply, config)

# Setup evaluator.
evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)
absolute_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)

# Add total timesteps to the config and compute environment steps per rollout.
steps_per_rollout, config = compute_total_timesteps(config)

# Run experiment for a total number of evaluations.
ep_returns = []
start_time = time.time()
n_devices = len(jax.devices())

# exploring code for a single evaluation
# un-comment for multiple evaluations.
for _ in range(config["arch"]["num_evaluation"]):
    # Train.
    learner_output = learn(learner_state)
    jax.block_until_ready(learner_output)

    # collecting training data

    # Prepare for evaluation.
    trained_params = unreplicate_batch_dim(learner_state.params.actor_params)

    key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
    eval_keys = jnp.stack(eval_keys)
    eval_keys = eval_keys.reshape(n_devices, -1)

    # Evaluate.
    evaluator_output = evaluator(trained_params, eval_keys, {})
    jax.block_until_ready(evaluator_output)

    mean_episode_return = jnp.mean(evaluator_output["episode_return"])
    ep_returns = plot_performance(mean_episode_return, ep_returns, start_time, config)

    # Update runner state to continue training.
    learner_state = learner_output.learner_state

# Return trained params to be used for rendering or testing.
trained_params = unreplicate_n_dims(trained_params, unreplicate_depth=1)

print(f"{Fore.CYAN}{Style.BRIGHT}MAPPO experiment completed{Style.RESET_ALL}")

data = learner_state.timestep.extras

def table_episode_metrics(data):
  # Flattening the nested arrays
  episode_length = data['episode_metrics']['episode_length'].flatten()
  episode_return = data['episode_metrics']['episode_return'].flatten()
  is_terminal_step = data['episode_metrics']['is_terminal_step'].flatten()
  percent_collected = data['percent_collected'].flatten()

  # Creating a dictionary for Pandas
  flattened_data = {
      'episode_length': episode_length,
      'episode_return': episode_return,
      'is_terminal_step': is_terminal_step,
      'percent_collected': percent_collected
  }

  # Creating the DataFrame
  df = pd.DataFrame(flattened_data)

  # Display the DataFrame
  return df

table = table_episode_metrics(data)
table.describe()

render_data = render_one_episode_complex(orchard_version_name, config, trained_params, max_steps=450, verbose=False)

generate_gif(render_data, "rendered_simulation.gif")
