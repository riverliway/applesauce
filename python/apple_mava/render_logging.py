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
from jumanji_env.environments.complex_orchard.custom_mava import make_env as make_complex_env
from jumanji_env.environments.complex_orchard.constants import TICK_SPEED

def render_one_episode_simple(orchard_version_name, config, params, max_steps, verbose=True) -> None:
    """Simulates and visualises one episode from rolling out a trained MAPPO model that will be passed to the function using actors_params."""
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
    
    if verbose:
        action_types = ("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "LOAD")
        print("="*70)
        print("Apple locations:", state.env_state.apples.position.tolist())
        print("="*70)
    
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
        if verbose:
            print("Step:", episode_length)
            print("Bot Location Before Step:", state.env_state.bots.position.tolist())
        state, timestep = step_fn(state, action)
        states.append(state)
        episode_return += jnp.mean(timestep.reward)
        episode_length += 1
        if verbose:
            print("Action:", action)
            print("Bot Location After:", state.env_state.bots.position.tolist())
            print("Reward:", jnp.mean(timestep.reward))
            print("Accumulative Reward:", episode_return)
            print("Apples Collected:", sum(state.env_state.apples.collected.tolist()))
            print("-"*70)

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

def render_one_episode_complex(orchard_version_name, config, params, max_steps, verbose=True) -> None:
    """Simulates and visualises one episode from rolling out a trained MAPPO model that will be passed to the function using actors_params."""
    # Create envs
    env_config = {**config.env.kwargs, **config.env.scenario.env_kwargs}
    env, eval_env = make_complex_env(orchard_version_name, config)

    # Create actor networks (We only care about the policy during rendering)
    actor_network = Actor(env.action_dim)
    apply_fn = actor_network.apply

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    key = jax.random.PRNGKey(config.system.seed)
    key, reset_key = jax.random.split(key)
    state, timestep = reset_fn(reset_key)
    
    if verbose:
        action_types = ("NOOP", "UP", "DOWN", "LEFT", "RIGHT", "LOAD")
        print("="*70)
        apple_locations = state.env_state.apples.position.tolist()
        print("Number of Apples:", len(apple_locations))
        print("Apple locations:", apple_locations)
        print("="*70)
    
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
        if verbose:
            print("Step:", episode_length)
            print("Bot Location Before Step:", state.env_state.bots.position.tolist())
        state, timestep = step_fn(state, action)
        states.append(state)
        episode_return += jnp.mean(timestep.reward)
        episode_length += 1
        if verbose:
            print("Action:", action)
            print("Bot Location After:", state.env_state.bots.position.tolist())
            print("Reward:", jnp.mean(timestep.reward))
            print("Accumulative Reward:", episode_return)
            print("Apples Collected:", sum(state.env_state.apples.collected.tolist()))
            print("-"*70)

    # Print out the results of the episode
    print(f"{Fore.CYAN}{Style.BRIGHT}EPISODE RETURN: {episode_return}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}EPISODE LENGTH:{episode_length}{Style.RESET_ALL}")

    # Limit the number of steps to record to the maximum number of steps
    steps = min([max_steps, len(states) - 1])
    states = states[:steps]

    state_dicts = []
    for state in states:
      record = create_complex_dict(state, config.system.seed)
      state_dicts.append(record)
    # Render the episode
    ##### Commenting out the rendering and returning the states. 
    # env.animate(states=states, interval=100, save_path="./applesauce.gif")
    return state_dicts

def create_complex_dict(state, seed) -> dict:
    """
    Creates a dictionary from the state object.
    """

    bots = [{
        "x": position[0],
        "y": position[1],
        "diameter": diameter,
        "holding": holding,
        "job": 'picker' if job < 0.5 else 'pusher',
        "orientation": orientation
    }
    for position, orientation, holding, job, diameter in zip(
        state.env_state.bots.position.tolist(),
        state.env_state.bots.orientation.tolist(),
        state.env_state.bots.holding.tolist(),
        state.env_state.bots.job.tolist(),
        state.env_state.bots.diameter.tolist(),
    )]

    trees = [{
        "x": position[0],
        "y": position[1],
        "diameter": diameter
    }
    for position, diameter in zip(
        state.env_state.trees.position.tolist(),
        state.env_state.trees.diameter.tolist()
    )]

    baskets = [{
        "x": position[0],
        "y": position[1],
        "diameter": diameter
    }
    for position, diameter in zip(
        state.env_state.baskets.position.tolist(),
        state.env_state.baskets.diameter.tolist()
    )]

    apples = [{
        "x": position[0],
        "y": position[1],
        "diameter": diameter,
        "held": held,
        "collected": collected
    }
    for position, diameter, held, collected in zip(
        state.env_state.baskets.position.tolist(),
        state.env_state.baskets.diameter.tolist(),
        state.env_state.baskets.held.tolist(),
        state.env_state.baskets.collected.tolist()
    )]

    return {
        "width": state.env_state.width,
        "height": state.env_state.height,
        "seed": seed,
        "time": state.env_state.step_count,
        "bots": bots,
        "trees": trees,
        "baskets": baskets,
        "apples": apples,
        "TICK_SPEED": TICK_SPEED
    }

def plot_performance(mean_episode_return, ep_returns, start_time):
    """visualises the performance of the algorithm. This plot will be refreshed each time evaluation interval happens."""
    
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