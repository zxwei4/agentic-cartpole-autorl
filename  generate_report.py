import json
from pathlib import Path


def main():
    history_path = Path("results/agentic_history.json")

    if not history_path.exists():
        raise FileNotFoundError("results/agentic_history.json not found")

    with open(history_path, "r") as f:
        history = json.load(f)

    best = max(history, key=lambda x: x["mean_reward"])

    lines = []

    lines.append("# Agentic AutoRL Experiment Report")
    lines.append("")
    lines.append("## Task")
    lines.append("")
    lines.append("Train a PPO reinforcement learning policy to solve CartPole-v1 using an agentic hyperparameter tuning loop.")
    lines.append("")
    lines.append("## Experiment Summary")
    lines.append("")
    lines.append(f"- Total rounds: {len(history)}")
    lines.append(f"- Best mean reward: {best['mean_reward']}")
    lines.append(f"- Best std reward: {best['std_reward']}")
    lines.append(f"- Best model path: `{best['model_path']}`")
    lines.append("")
    lines.append("## Round-by-Round Results")
    lines.append("")
    lines.append("| Round | Learning Rate | Gamma | Timesteps | Mean Reward | Std Reward | Agent Reason |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    for item in history:
        config = item["config"]
        lines.append(
            f"| {item['round_id']} "
            f"| {config['learning_rate']} "
            f"| {config['gamma']} "
            f"| {config['total_timesteps']} "
            f"| {item['mean_reward']} "
            f"| {item['std_reward']} "
            f"| {item['agent_reason']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The agentic loop started from a weak baseline configuration and improved the policy by "
        "adjusting the learning rate, discount factor and training horizon based on previous reward feedback."
    )
    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append(
        "Replace the rule-based planner with an LLM planner that can read experiment history, "
        "analyze failure modes and propose the next hyperparameter configuration through tool calling."
    )

    output_path = Path("results/final_report.md")
    output_path.write_text("\n".join(lines))

    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()