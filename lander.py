import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

ENV_ID = "LunarLander-v3"
TIME_FEE = 0.05          
PROGRESS_SCALE = 10.0    
HOVER_BONUS = 1.0        

class ShapedReward(gym.Wrapper):
    def __init__(self, env, mode="none"):
        super().__init__(env)
        self.mode = mode
        self.prev_phi = 0.0

    @staticmethod
    def _phi(obs):
        x, _, vx, vy, angle = obs[0], obs[1], obs[2], obs[3], obs[4]
        return -(abs(x) + abs(angle) + 0.5 * abs(vx) + 0.5 * abs(vy))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_phi = self._phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.mode == "time":
            reward -= TIME_FEE

        elif self.mode == "hover_bug":
            upright = abs(obs[4]) < 0.2
            slow = abs(obs[2]) < 0.3 and abs(obs[3]) < 0.3
            if upright and slow:
                reward += HOVER_BONUS

        elif self.mode == "fixed":
            phi = self._phi(obs)
            reward += PROGRESS_SCALE * (phi - self.prev_phi) - TIME_FEE
            self.prev_phi = phi

        return obs, reward, terminated, truncated, info


def model_path(mode, seed=0):
    return f"model_{mode}_s{seed}"


def run_random():
    env = gym.make(ENV_ID, render_mode="human")
    obs, _ = env.reset()
    print("Observation:", obs)
    for _ in range(1000):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        print([round(float(v), 2) for v in obs], round(r, 2))
        if term or trunc:
            obs, _ = env.reset()
    env.close()


def train(mode, steps, seed):
    env = make_vec_env(
        ENV_ID, n_envs=16, seed=seed,
        wrapper_class=ShapedReward, wrapper_kwargs={"mode": mode},
    )
    model = PPO(
        "MlpPolicy", env, verbose=1, seed=seed,
        n_steps=1024, batch_size=64, n_epochs=4,
        gamma=0.999, gae_lambda=0.98, ent_coef=0.01,
    )
    model.learn(total_timesteps=steps)
    model.save(model_path(mode, seed))
    print("Saved", model_path(mode, seed))


def evaluate(mode, seed=0, episodes=20):
    env = gym.make(ENV_ID)
    model = PPO.load(model_path(mode, seed))
    mean, std = evaluate_policy(model, env, n_eval_episodes=episodes)
    print(f"[{mode} seed {seed}] mean reward {mean:.1f} +/- {std:.1f} over {episodes} episodes "
          f"({'SOLVED' if mean >= 200 else 'not solved'})")


def fly(mode, record, seed=0):
    if record:
        env = gym.make(ENV_ID, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env, video_folder=f"videos/{mode}", episode_trigger=lambda i: True,
            name_prefix=f"{mode}_s{seed}",
        )
    else:
        env = gym.make(ENV_ID, render_mode="human")
    model = PPO.load(model_path(mode, seed))
    for _ in range(3):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(int(action))
            done = term or trunc
    env.close()
    if record:
        print(f"Videos saved in videos/{mode}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["random", "train", "eval", "watch", "record"])
    p.add_argument("--mode", default="none", choices=["none", "time", "hover_bug", "fixed"])
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    if a.cmd == "random":
        run_random()
    elif a.cmd == "train":
        train(a.mode, a.steps, a.seed)
    elif a.cmd == "eval":
        evaluate(a.mode, a.seed)
    elif a.cmd == "watch":
        fly(a.mode, record=False, seed=a.seed)
    elif a.cmd == "record":
        fly(a.mode, record=True, seed=a.seed)