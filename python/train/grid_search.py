import itertools
import json
import subprocess

# define config variables and values for grid
grid_search_params = {
    "system.actor_lr": [2.5e-4, 1e-4],
    "system.critic_lr": [2.5e-4, 1e-4],
    # "system.update_batch_size": [2, 4],
    # "system.rollout_length": [256, 128],
    # "rewards.REWARD_COST_OF_STEP": [-0.1, -0.05],
}

# Load the initial config
config_file = "config.py"

def load_config(file_path):
    with open(file_path, "r") as file:
        # Execute the file to load the Python dict
        config_globals = {}
        exec(file.read(), config_globals)
        return config_globals["config"]

def save_config(config, file_path):
    with open(file_path, "w") as file:
        file.write("# configuration for environment and models\n")
        file.write(f"config = {json.dumps(config, indent=4)}\n")

# Get the parameter keys for grid search
param_keys = grid_search_params.keys()

# Generate all combinations of hyperparameters
param_values = grid_search_params.values()
combinations = list(itertools.product(*param_values))

# Run the grid search
original_config = load_config(config_file)

for i, combination in enumerate(combinations):
    print(f"Running grid search iteration {i+1}/{len(combinations)} with params: {combination}")
    
    # Update config with the current combination of parameters
    updated_config = original_config.copy()
    for key, value in zip(param_keys, combination):
        # Navigate through nested keys
        keys = key.split(".")
        sub_config = updated_config
        for sub_key in keys[:-1]:
            sub_config = sub_config[sub_key]
        sub_config[keys[-1]] = value

    # Save the updated config
    save_config(updated_config, config_file)
    
    # Run the training script
    result = subprocess.run(["python", "complexorchard_train.py"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

print("Grid search completed.")