from pydantic import BaseModel
from typing import Literal
import json
import random

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.optim as optim

from a2c import A2C
from minatar import Environment
from minatar_utils.models import ACNetwork
from minatar_utils.wrappers import MinAtarEnv
from utils import evaluate

import wandb
import git
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MinAtarConfig(BaseModel):
    game: Literal["breakout", "asterix", "freeway", "seaquest", "space_invaders"] = "breakout"
    steps: int = int(5e6)
    eval_interval: int = int(1e5)
    eval_n_episodes: int = 64
    eval_deterministic: bool = False
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
    logits, _ = model(obs)
    dist = Categorical(logits=logits)
    a = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
    return a

args = MinAtarConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args)

wandb.init(project=f"a2c-minatar", entity="sotetsuk", config=args.dict())


# fix seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

in_channels = Environment(args.game).state_shape()[2]
num_actions = len(Environment(args.game).minimal_action_set())

algo = A2C(config=args)
env = MinAtarEnv(game=args.game, num_envs=args.num_envs, seed=args.seed)
model = ACNetwork(in_channels, num_actions, args.game)
opt = optim.Adam(model.parameters(), lr=args.lr)

n_train = 0
log = {"steps": 0, "avg_prob": 1.0 / num_actions}
while True:
    log["eval_R"] = evaluate(
        MinAtarEnv(game=args.game, num_envs=args.num_envs, seed=args.seed+9999),  # TODO: fix seed
        model,
        deterministic=args.eval_deterministic,
        num_episodes=args.eval_n_episodes,
    )
    wandb.log({f"{args.game}/{k}": v for k, v in log.items()})
    print(json.dumps(log))
    if algo.n_steps >= args.steps:
        break
    log = algo.train(env, model, opt, n_steps_lim=(n_train + 1) * args.eval_interval)
    n_train += 1
