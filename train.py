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
from utils import MinAtarConfig, evaluate


args = MinAtarConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args)


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
    print(json.dumps(log))
    if algo.n_steps >= args.steps:
        break
    log = algo.train(env, model, opt, n_steps_lim=(n_train + 1) * args.eval_interval)
    n_train += 1
