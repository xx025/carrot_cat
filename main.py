import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


class CatPawPointingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=0, p_tissue=0.5, render_mode=None):
        super().__init__()

        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)

        self.seed_value = seed
        self.p_tissue = float(p_tissue)
        self.render_mode = render_mode

        self.np_random = None
        self.current_question = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.np_random is None or seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.current_question = 0 if self.np_random.random() < self.p_tissue else 1

        if self.render_mode == "human":
            print(f"\n[RESET] 人类提问对象：{self._decode_obj(self.current_question)}")

        return int(self.current_question), {"question": int(self.current_question)}

    def step(self, action):
        assert self.current_question is not None, "请先 reset()"

        action = int(action)
        correct_action = int(self.current_question)

        reward = 1.0 if action == correct_action else -1.0

        terminated = True
        truncated = False
        next_obs = 0

        info = {
            "question": int(self.current_question),
            "action": action,
            "correct_action": correct_action,
            "correct": bool(action == correct_action),
            "paw_target": action,
        }

        if self.render_mode == "human":
            q = self._decode_obj(self.current_question)
            paw = self._decode_obj(action)
            correct = self._decode_obj(correct_action)
            mark = "✅正确" if info["correct"] else "❌错误"
            print(f"[STEP] 问题={q} | 猫把脚掌放在={paw} | 正确应放在={correct} | {mark} | reward={reward}")

        self.current_question = None
        return next_obs, reward, terminated, truncated, info

    @staticmethod
    def _decode_obj(x: int) -> str:
        return "纸巾" if x == 0 else "萝卜"


def main():
    env = CatPawPointingEnv(seed=42, p_tissue=0.5, render_mode=None)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.0,
    )

    model.learn(total_timesteps=10_000)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/luobo_cat_ppo")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2000)
    print(f"\n[Eval] mean_reward={mean_reward:.3f} ± {std_reward:.3f}")

    demo_env = CatPawPointingEnv(seed=123, p_tissue=0.5, render_mode="human")
    for i in range(10):
        obs, _ = demo_env.reset(seed=1000 + i)
        action, _ = model.predict(obs, deterministic=True)
        demo_env.step(action)


if __name__ == "__main__":
    main()
