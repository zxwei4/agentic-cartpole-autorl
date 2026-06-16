from typing import Dict, Any, List


def propose_next_config(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    A simple planner agent that proposes the next PPO configuration
    based on previous experiment results.
    """

    # First round: intentionally weak baseline
    if not history:
        return {
            "algo": "PPO",
            "learning_rate": 1e-3,
            "gamma": 0.98,
            "total_timesteps": 5_000,
            "n_eval_episodes": 20,
            "reason": "Start with a short training horizon and relatively aggressive learning rate as a weak baseline."
        }

    last_result = history[-1]
    last_reward = last_result["mean_reward"]

    # If reward is very low, train longer and reduce learning rate
    if last_reward < 200:
        return {
            "algo": "PPO",
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "total_timesteps": 20_000,
            "n_eval_episodes": 20,
            "reason": "Reward is low, suggesting unstable or insufficient learning. Reduce learning rate and increase training steps."
        }

    # If reward is moderate, keep stable learning rate and train longer
    if last_reward < 400:
        return {
            "algo": "PPO",
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "total_timesteps": 50_000,
            "n_eval_episodes": 20,
            "reason": "Reward is improving but not solved. Continue with stable PPO parameters and longer training."
        }

    # If reward is high but not solved, increase discount factor and training time
    if last_reward < 475:
        return {
            "algo": "PPO",
            "learning_rate": 3e-4,
            "gamma": 0.995,
            "total_timesteps": 80_000,
            "n_eval_episodes": 20,
            "reason": "Policy is close to solving the task. Increase gamma and training horizon for better long-term balancing."
        }

    # Solved
    return {
        "stop": True,
        "reason": "Target performance reached. No further tuning required."
    }