from tools.rl_tools import train_policy


search_space = [
    {
        "algo": "PPO",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "total_timesteps": 50_000,
        "n_eval_episodes": 20,
    },
    {
        "algo": "PPO",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "total_timesteps": 50_000,
        "n_eval_episodes": 20,
    },
    {
        "algo": "PPO",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "total_timesteps": 50_000,
        "n_eval_episodes": 20,
    },
    {
        "algo": "PPO",
        "learning_rate": 3e-4,
        "gamma": 0.995,
        "total_timesteps": 80_000,
        "n_eval_episodes": 20,
    },
    {
        "algo": "PPO",
        "learning_rate": 1e-4,
        "gamma": 0.995,
        "total_timesteps": 80_000,
        "n_eval_episodes": 20,
    },
]


def main():
    best_result = None

    for i, config in enumerate(search_space, start=1):
        print("=" * 80)
        print(f"Running experiment {i}/{len(search_space)}")
        print(config)

        result = train_policy(config)

        print("Result:")
        print(result)

        if best_result is None or result["mean_reward"] > best_result["mean_reward"]:
            best_result = result

    print("=" * 80)
    print("Best result:")
    print(best_result)


if __name__ == "__main__":
    main()
