import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    history_path = Path("results/agentic_history.json")

    if not history_path.exists():
        raise FileNotFoundError("results/agentic_history.json not found")

    with open(history_path, "r") as f:
        history = json.load(f)

    rounds = [item["round_id"] for item in history]
    rewards = [item["mean_reward"] for item in history]

    Path("results").mkdir(exist_ok=True)

    plt.figure()
    plt.plot(rounds, rewards, marker="o")
    plt.xlabel("Agentic AutoRL Round")
    plt.ylabel("Mean Evaluation Reward")
    plt.title("Agentic PPO Tuning on CartPole-v1")
    plt.ylim(0, 520)
    plt.grid(True)

    output_path = "results/agentic_reward_curve.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()