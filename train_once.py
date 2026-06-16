from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def main():
    Path("models").mkdir(exist_ok=True)

    env = gym.make("CartPole-v1")

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=1024,
        batch_size=64,
        verbose=1,
    )

    model.learn(total_timesteps=50_000)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=20,
        deterministic=True,
    )

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    model.save("models/ppo_cartpole")
    env.close()


if __name__ == "__main__":
    main()