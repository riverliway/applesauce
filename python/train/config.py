# configuration for environment and models
config = {
    "system": {
        "actor_lr": 2.5e-4,
        "critic_lr": 2.5e-4,
        "update_batch_size": 2,
        "rollout_length": 128,
        "num_updates": 400,
        "ppo_epochs": 16,
        "num_minibatches":32,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "add_agent_id": True,
        "decay_learning_rates": False,
        "seed": 42,
    },
    "arch": {
        "num_envs": 512,
        "num_eval_episodes": 32,
        "num_evaluation": 20,
        "evaluation_greedy": False,
        "num_absolute_metric_eval_episodes": 32,
    },
    "env": {
        "eval_metric": "episode_return",
        "implicit_agent_id": False,
        "log_win_rate": False,
        "kwargs": {"time_limit": 500},
        "scenario": {
            "task_config": {
                "width": 500,
                "height": 500,
                "num_picker_bots": 2,
                "num_pusher_bots": 0,
                "num_baskets": 1,
            },
            "env_kwargs": {},
        },
    },
}