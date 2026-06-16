import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _weak_initial_config() -> Dict[str, Any]:
    """
    Start from a deliberately weak PPO configuration so that
    the LLM planner has room to improve the policy.
    """
    return {
        "algo": "PPO",
        "learning_rate": 0.01,
        "gamma": 0.90,
        "total_timesteps": 1_000,
        "n_eval_episodes": 20,
        "reason": "Start from a weak baseline with high learning rate, low gamma, and short training horizon."
    }


def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clip LLM-generated hyperparameters to safe ranges.
    This prevents the LLM from producing invalid or extremely expensive configs.
    """

    algo = config.get("algo", "PPO")
    if algo != "PPO":
        algo = "PPO"

    learning_rate = float(config.get("learning_rate", 3e-4))
    gamma = float(config.get("gamma", 0.99))
    total_timesteps = int(config.get("total_timesteps", 50_000))
    n_eval_episodes = int(config.get("n_eval_episodes", 20))
    reason = str(config.get("reason", "LLM proposed this configuration based on previous experiment history."))

    learning_rate = min(max(learning_rate, 1e-5), 0.02)
    gamma = min(max(gamma, 0.85), 0.999)
    total_timesteps = min(max(total_timesteps, 500), 80_000)
    n_eval_episodes = min(max(n_eval_episodes, 5), 50)

    return {
        "algo": algo,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "total_timesteps": total_timesteps,
        "n_eval_episodes": n_eval_episodes,
        "reason": reason,
    }


def propose_next_config(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    LLM planner:
    - Reads previous experiment history
    - Analyzes reward, std, runtime, and hyperparameters
    - Proposes the next PPO configuration
    """

    if not history:
        return _weak_initial_config()

    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    compact_history = []

    for item in history:
        config = item["config"]
        compact_history.append({
            "round_id": item["round_id"],
            "learning_rate": config["learning_rate"],
            "gamma": config["gamma"],
            "total_timesteps": config["total_timesteps"],
            "mean_reward": item["mean_reward"],
            "std_reward": item["std_reward"],
            "runtime_seconds": item["runtime_seconds"],
            "agent_reason": item.get("agent_reason", ""),
        })

    system_prompt = """
You are an LLM planner for an AutoRL system.

Your job is to propose the next PPO hyperparameter configuration for CartPole-v1.

The Python training tool will execute your config.
You must return ONLY valid JSON.

Rules:
- Use only PPO.
- CartPole-v1 maximum reward is 500.
- If reward is low, increase training steps or stabilize learning rate.
- If reward is unstable, reduce learning rate.
- If reward is close to solved, fine-tune gamma and training horizon.
- Do not propose extremely expensive training.
- Output must contain:
  algo, learning_rate, gamma, total_timesteps, n_eval_episodes, reason.
"""

    user_prompt = f"""
Previous experiment history:

{json.dumps(compact_history, indent=2)}

Please propose the next PPO configuration.

Return only JSON in this format:

{{
  "algo": "PPO",
  "learning_rate": 0.0003,
  "gamma": 0.99,
  "total_timesteps": 50000,
  "n_eval_episodes": 20,
  "reason": "Explain why this config is appropriate."
}}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content

    try:
        config = json.loads(raw_text)
    except json.JSONDecodeError:
        print("LLM returned invalid JSON. Falling back to safe default.")
        config = {
            "algo": "PPO",
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "total_timesteps": 50_000,
            "n_eval_episodes": 20,
            "reason": "Fallback safe PPO configuration because LLM output was invalid."
        }

    return _validate_config(config)