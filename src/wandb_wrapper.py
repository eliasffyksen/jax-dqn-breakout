from typing import Any
import gymnasium as gym
from wandb.sdk.wandb_run import Run as WandbRun
import wandb
import numpy as np

class WandbWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, wdb: WandbRun, log_interval: int):
        self.wdb = wdb
        self.log_interval = log_interval
        self.scores = []
        self.current_score = 0
        self.current_episode = 0
        self.current_frame = 0

        self.best_episode_score = 0
        self.current_episode_observations = []

        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_episode_observations.append(obs)

        if self.current_frame % self.log_interval == 0 and len(self.scores) > 0:
            mean_score = sum(self.scores) / len(self.scores)
            self.scores = []

            self.wdb.log({
                'Mean Score': mean_score,
            }, commit=False)

        self.current_score += reward
        self.current_frame += 1

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        if self.current_score > self.best_episode_score:
            self.best_episode_score = self.current_score

            video: np.ndarray
            video = np.stack(self.current_episode_observations)
            video = video.transpose((0, 3, 1, 2))
            video = wandb.Video(
                video,
                caption=f'Score: {self.best_episode_score}, Frame: {self.current_frame}, Episode: {self.current_episode}',
                fps=15,
                format='mp4',
            )
            self.wdb.log({
                'Video': video,
            }, commit=False)

            print('New best score of:', self.best_episode_score)


        self.current_episode_observations = []

        self.scores.append(self.current_score)
        self.current_score = 0
        self.current_episode += 1

        return super().reset(seed=seed, options=options)