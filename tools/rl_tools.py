from pathlib import Path
import json
import time
from typing import Dict, Any

import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy


def train_policy(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a reinforcement learning policy on CartPole-v1.

    Expected config:
    {
        "algo": "PPO",
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "total_timesteps": 50000,
        "n_eval_episodes": 20
    }
    """

    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    algo = config.get("algo", "PPO")
    learning_rate = config.get("learning_rate", 3e-4)
    gamma = config.get("gamma", 0.99)
    total_timesteps = config.get("total_timesteps", 50_000)
    n_eval_episodes = config.get("n_eval_episodes", 20)

    env = gym.make("CartPole-v1")

    start_time = time.time()

    if algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            verbose=0,
        )
    elif algo == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            verbose=0,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    model.learn(total_timesteps=total_timesteps)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )

    runtime = time.time() - start_time

    model_name = f"{algo.lower()}_cartpole_lr{learning_rate}_gamma{gamma}"
    model_path = f"models/{model_name}"
    model.save(model_path)

    result = {
        "config": config,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "runtime_seconds": runtime,
        "model_path": model_path,
    }

    with open("logs/experiment_history.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")

    env.close()

    return result
