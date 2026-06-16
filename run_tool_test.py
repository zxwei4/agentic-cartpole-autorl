from tools.rl_tools import train_policy

config = {
    "algo": "PPO",
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "total_timesteps": 50_000,
    "n_eval_episodes": 20,
}

result = train_policy(config)

print(result)
