# Agentic AutoRL for CartPole Control

This project builds an agentic reinforcement learning system for CartPole-v1 control using PPO, Gymnasium and Stable-Baselines3.

The system starts from weak PPO hyperparameters, trains a policy, evaluates its reward, records the experiment history, and uses a planner agent to propose the next configuration.

## Features

- PPO training pipeline for CartPole-v1
- Automated hyperparameter tuning loop
- Rule-based planner agent
- LLM-compatible planner interface
- Experiment logging
- Reward curve visualization
- Final experiment report generation

## Project Structure

```text
.
├── agent/
│   ├── rule_based_planner.py
│   └── llm_planner.py
├── tools/
│   └── rl_tools.py
├── results/
│   ├── agentic_history.json
│   ├── agentic_reward_curve.png
│   └── final_report.md
├── run_agentic_loop.py
├── run_llm_agentic_loop.py
├── run_autotune.py
├── plot_agentic_history.py
├── generate_report.py
└── requirements.txt