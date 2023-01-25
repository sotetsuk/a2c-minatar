from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def evaluate(
    env,
    model: nn.Module,
    deterministic: bool = False,
    num_episodes=100,
    time_limit=3000,  # for minatar/seaquest
) -> float:
    model.eval()
    num_envs = env.num_envs
    assert num_episodes % num_envs == 0
    R_seq = []
    for i in range(num_episodes // num_envs):
        obs = env.reset()  # (num_envs, obs_size)
        done = [False for _ in range(num_envs)]
        R = torch.zeros(num_envs)
        t = 0
        while not all(done):
            actions = act(model, obs, deterministic)
            obs, r, done, info = env.step(actions)
            R += r  # If some episode is terminated, all r is zero afterwards.
            t += 1
            if t >= time_limit:
                break
        R_seq.append(R.mean())

    return float(sum(R_seq) / len(R_seq))


def act(model: nn.Module, obs: np.ndarray, deterministic: bool = False) -> Union[int, np.ndarray]:
    logits, _ = model(obs)
    dist = Categorical(logits=logits)
    a = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
    return a

