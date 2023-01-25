from dataclasses import dataclass
from typing import List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.distributions import Categorical

import git


class MinAtarConfig(BaseModel):
    game: Literal["breakout", "asterix", "freeway", "seaquest", "space_invaders"] = "breakout"
    steps: int = int(5e6)
    eval_interval: int = int(1e5)
    eval_n_episodes: int = 64
    eval_deterministic=True
    seed: int = 1234
    num_envs: int = 64
    lr: float = 0.003
    ent_coef: float = 0.0
    gamma: float = 0.99
    value_coef: float = 1.0
    unroll_length: int = 5
    debug: bool = False
    githash: str = git.Repo().head.object.hexsha[:7]


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
    # TODO: continuous action space
    logits, _ = model(obs)
    dist = Categorical(logits=logits)
    a = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
    return a


def flat_str(v):
    strs = []
    for i in range(v.size(0)):
        strs.append(" ".join([f"{float(e.item()):.2f}" for e in v[i, :]]))
    return "\n".join(strs)
