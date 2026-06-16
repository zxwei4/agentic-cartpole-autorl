import json
from pathlib import Path

from tools.rl_tools import train_policy
from agent.rule_based_planner import propose_next_config


TARGET_REWARD = 475
MAX_ROUNDS = 6
MIN_ROUNDS = 5


def save_history(history):
    Path("results").mkdir(exist_ok=True)

    with open("results/agentic_history.json", "w") as f:
        json.dump(history, f, indent=2)


def main():
    history = []

    for round_id in range(1, MAX_ROUNDS + 1):
        print("=" * 80)
        print(f"Agentic AutoRL Round {round_id}")

        config = propose_next_config(history)

        if config.get("stop"):
            print("Agent decided to stop.")
            print("Reason:", config["reason"])
            break

        print("Agent proposed config:")
        print(json.dumps(config, indent=2))

        reason = config.pop("reason")

        result = train_policy(config)
        result["round_id"] = round_id
        result["agent_reason"] = reason

        history.append(result)
        save_history(history)

        print("Experiment result:")
        print(json.dumps(result, indent=2))

        if result["mean_reward"] >= TARGET_REWARD and round_id >= MIN_ROUNDS:
            print("=" * 80)
            print(f"Target reached: mean_reward = {result['mean_reward']}")
            break

    print("=" * 80)
    print("Final experiment history:")

    for item in history:
        print(
            f"Round {item['round_id']}: "
            f"reward={item['mean_reward']}, "
            f"lr={item['config']['learning_rate']}, "
            f"gamma={item['config']['gamma']}, "
            f"steps={item['config']['total_timesteps']}"
        )


if __name__ == "__main__":
    main()