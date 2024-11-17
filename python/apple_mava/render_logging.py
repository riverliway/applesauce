import time  # For timing the execution in plot_performance
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting the performance
from IPython.display import clear_output  # For updating the plot dynamically
import jax  # For JAX utilities
import jax.numpy as jnp  # For JAX-based numerical operations
from colorama import Fore, Style  # For colored terminal output

# our custom implementations
from apple_mava.ff_networks import Actor
from jumanji_env.environments.simple_orchard.custom_mava import make_env

def render_one_episode(orchard_version_name, config, params, max_steps=100) -> None:
    """Rollout episodes of a trained MAPPO policy."""
    # Create envs
    env_config = {**config.env.kwargs, **config.env.scenario.env_kwargs}
    env, eval_env = make_env(orchard_version_name, config)

    # Create actor networks (We only care about the policy during rendering)
    actor_network = Actor(env.action_dim)
    apply_fn = actor_network.apply

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(config.system.seed)
    key, reset_key = jax.random.split(key)
    state, timestep = reset_fn(reset_key)

    states = [state]
    episode_return = 0
    episode_length = 0
    while not timestep.last():
        key, action_key = jax.random.split(key)
        pi = apply_fn(params, timestep.observation)

        if config["arch"]["evaluation_greedy"]:
            action = pi.mode()
        else:
            action = pi.sample(seed=action_key)
        state, timestep = step_fn(state, action)
        states.append(state)
        episode_return += jnp.mean(timestep.reward)
        episode_length += 1

    # Print out the results of the episode
    print(f"{Fore.CYAN}{Style.BRIGHT}EPISODE RETURN: {episode_return}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}EPISODE LENGTH:{episode_length}{Style.RESET_ALL}")

    # Limit the number of steps to record to the maximum number of steps
    steps = min([max_steps, len(states) - 1])
    states = states[:steps]

    # Assiging static values to add to each state record
    # size of environment
    width = env.width
    height = env.height
    # initial positions
    starting_bots = [tuple(pos) for pos in states[0].env_state.bots.position.tolist()]
    starting_trees = [tuple(pos) for pos in states[0].env_state.trees.position.tolist()]
    starting_apples = [tuple(pos) for pos in states[0].env_state.apples.position.tolist()]

    state_dicts = []
    for state in states:
      env_state = state.env_state
      record = {
          "width:": width,
          "height": height,
          "bots": [tuple(pos) for pos in env_state.bots.position.tolist()],
          "starting_bots": starting_bots,
          "trees": [tuple(pos) for pos in env_state.trees.position.tolist()],
          "starting_trees": starting_trees,
          "apples": [tuple(pos) for pos in env_state.apples.position.tolist()],
          "starting_apples": starting_apples,
          "time": int(env_state.step_count),
      }
      state_dicts.append(record)
    # Render the episode
    ##### Commenting out the rendering and returning the states. 
    # env.animate(states=states, interval=100, save_path="./applesauce.gif")
    return state_dicts

def plot_performance(mean_episode_return, ep_returns, start_time):
    plt.figure(figsize=(8, 4))
    clear_output(wait=True)

    # Plot the data
    ep_returns.append(mean_episode_return)
    plt.plot(
        np.linspace(0, (time.time() - start_time) / 60.0, len(list(ep_returns))), list(ep_returns)
    )
    plt.xlabel("Run Time [Minutes]")
    plt.ylabel("Episode Return")
    plt.title("Apple Orchard with {x} Agents")

    # Show the plot
    plt.show()
    return ep_returns