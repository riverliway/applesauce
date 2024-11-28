import time  # For timing the execution in plot_performance
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting the performance
from matplotlib.animation import FuncAnimation
import imageio
import os

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
            print("Apples Picked:", state.env_state.bots.holding.tolist())
            print("Apples Collected:", state.env_state.apples.collected.tolist())
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
        state.env_state.apples.position.tolist(),
        state.env_state.apples.diameter.tolist(),
        state.env_state.apples.held.tolist(),
        state.env_state.apples.collected.tolist()
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

def plot_performance(mean_episode_return, ep_returns, start_time, config):
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
    width = config["env"]["scenario"]["task_config"]["width"]
    height = config["env"]["scenario"]["task_config"]["height"]
    pickers = config["env"]["scenario"]["task_config"]["num_picker_bots"]
    plt.title(f"{width}x{height} Apple Orchard with {pickers} Agents")

    # Show the plot
    plt.show()
    return ep_returns

def visualize_environment(data):
    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, data['width'])
    ax.set_ylim(0, data['height'])
    ax.set_aspect('equal', adjustable='box')

    # Plot bots
    for bot in data['bots']:
        bot_circle = plt.Circle((bot['x'], bot['y']), bot['diameter'] / 2, color='blue', alpha=0.6, label='Bot')
        ax.add_patch(bot_circle)
    
    # Plot trees
    for tree in data['trees']:
        tree_circle = plt.Circle((tree['x'], tree['y']), tree['diameter'] / 2, color='green', alpha=0.6, label='Tree')
        ax.add_patch(tree_circle)
    
    # Plot baskets
    for basket in data['baskets']:
        basket_circle = plt.Circle((basket['x'], basket['y']), basket['diameter'] / 2, color='orange', alpha=0.6, label='Basket')
        ax.add_patch(basket_circle)
    
    # Plot apples
    for apple in data['apples']:
        color = 'fushsia' if apple['collected'] else 'gold' if apple['held'] else 'red'
        apple_circle = plt.Circle((apple['x'], apple['y']), apple['diameter'] / 2, color=color, alpha=0.6, label='Apple')
        ax.add_patch(apple_circle)

    # Avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")
    
    # Set titles and labels
    ax.set_title("Environment Visualization")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    
    # Display the plot
    plt.gca().invert_yaxis()
    plt.show()
    
# Visualization function for a single step
def visualize_step(data, step_index, save_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, data['width'])
    ax.set_ylim(0, data['height'])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Environment Step {step_index}")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")

    # Plot bots
    for bot in data['bots']:
        bot_circle = plt.Circle((bot['x'], bot['y']), bot['diameter'] / 2, color='blue', alpha=0.6, label='Bot')
        ax.add_patch(bot_circle)
        
        # Calculate and draw the nose
        nose_x = bot['x'] + (bot['diameter'] / 2) * jnp.cos(bot['orientation'])
        nose_y = bot['y'] + (bot['diameter'] / 2) * jnp.sin(bot['orientation'])
        # Draw an arrow to represent the bot's orientation
        ax.arrow(
            bot['x'], bot['y'],  # Starting point (bot's center)
            nose_x - bot['x'], nose_y - bot['y'],  # Arrow vector
            head_width=bot['diameter'] * 0.2,  # Arrowhead width
            head_length=bot['diameter'] * 0.3,  # Arrowhead length
            fc='white', ec='white',  # Arrow color
            length_includes_head=True  # Include head in length calculation
        )
    
    # Plot trees
    for tree in data['trees']:
        tree_circle = plt.Circle((tree['x'], tree['y']), tree['diameter'] / 2, color='green', alpha=0.6, label='Tree')
        ax.add_patch(tree_circle)
    
    # Plot baskets
    for basket in data['baskets']:
        basket_circle = plt.Circle((basket['x'], basket['y']), basket['diameter'] / 2, color='orange', alpha=0.6, label='Basket')
        ax.add_patch(basket_circle)
    
    # Plot apples
    for apple in data['apples']:
        color = 'red' if apple['collected'] else 'yellow' if apple['held'] else 'brown'
        apple_circle = plt.Circle((apple['x'], apple['y']), apple['diameter'] / 2, color=color, alpha=0.6, label='Apple')
        ax.add_patch(apple_circle)

    # Avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc="upper right")

    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.close(fig)

# Function to generate .gif from a list of dictionaries
def generate_gif(data_list, output_gif_path, temp_dir="temp_frames"):
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_paths = []
    for i, step_data in enumerate(data_list):
        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        visualize_step(step_data, i, frame_path)
        frame_paths.append(frame_path)
    
    # Combine frames into a gif
    with imageio.get_writer(output_gif_path, mode='I', duration=0.2) as writer:
        for frame_path in frame_paths:
            image = imageio.imread(frame_path)
            writer.append_data(image)
    
    # Clean up temporary frame files
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(temp_dir)