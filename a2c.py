from typing import Dict, List

import torch
import torch.nn as nn
from torch.distributions import Categorical

from utils import MinAtarConfig


class A2C:
    def __init__(self, config: MinAtarConfig):
        self.config = config

        self.n_steps: int = 0
        self.n_episodes: int = 0
        self.data: Dict[str, List[torch.Tensor]] = {}
        self.env = None
        self.model = None
        self.opt = None

        # stats
        self.n_stats_update = 0
        self.avg_R = 0.0
        self.avg_ent = 0.0
        self.avg_seq_len = 0.0
        self.avg_prob = 0.0
        self.value = 0.0

        self.observations = None

    def train(
        self,
        env,
        model: nn.Module,
        opt,
        n_steps_lim: int = 100_000,
    ) -> Dict[str, float]:
        self.env, self.model, self.opt = env, model, opt

        if self.observations is None:
            self.observations = self.env.reset()  # (num_envs, obs_dim)

        while self.n_steps < n_steps_lim:
            # rollout data
            self.rollout()

            # compute loss and update gradient
            self.opt.zero_grad()
            loss = self.loss()
            loss.backward()
            self.opt.step()

            self.log()

        return {
            "steps": self.n_steps,
            "n_episodes": self.n_episodes,
            "avg_ent": self.avg_ent,
            "avg_prob": self.avg_prob,
            "value": self.value,
            "train_R": self.avg_R,
        }

    def rollout(self) -> None:
        assert self.env is not None and self.model is not None
        self.data = {}
        self.model.train()
        for unroll_ix in range(self.config.unroll_length):
            action, log_prob, entropy, value = self.act(self.observations)  # agent step
            self.observations, rewards, terminated, _ = self.env.step(action.numpy())  # env step
            self.n_steps +=  self.env.num_envs
            truncated = (
                int(unroll_ix == self.config.unroll_length - 1) * (1 - terminated.int())
            ).bool()
            with torch.no_grad():
                _, _, _, next_value = self.act(self.observations)
            self.push_data(
                terminated=terminated,
                truncated=truncated,
                log_prob=log_prob,
                entropy=entropy,
                value=value,
                next_value=next_value,
                rewards=rewards,
            )
            self.observations = self.env.init(terminated)  # auto reset

    def loss(self, reduce=True) -> torch.Tensor:
        v = torch.stack(self.data["value"]).t()  # (num_envs, max_seq_len + 1)
        with torch.no_grad():
            v_tgt = self.compute_return()
        # pg loss
        log_prob = torch.stack(self.data["log_prob"]).t()  # (n_env, seq_len)
        b = v.detach()
        loss = - (v_tgt - b) * log_prob
        # value loss
        value_loss = (v_tgt - v) ** 2
        # ent loss
        ent = torch.stack(self.data["entropy"]).t()  # (num_env, max_seq_len)
        ent_loss = -ent

        loss += self.config.ent_coef * ent_loss
        loss += self.config.value_coef * value_loss
        return loss.sum(dim=1).mean(dim=0) if reduce else loss

    def compute_return(self):
        """compute n-step return following A3C paper"""
        rewards = torch.stack(self.data["rewards"]).t()
        next_values = torch.stack(self.data["next_value"]).t()
        truncated = torch.stack(self.data["truncated"]).t()
        terminated = torch.stack(self.data["terminated"]).t()
        done = truncated | terminated
        R = rewards + self.config.gamma * next_values * truncated.float()
        seq_len = R.size(1)
        for i in reversed(range(seq_len - 1)):
            R[:, i] += self.config.gamma * R[:, i + 1] * (1 - done[:, i].float())
        return R

    def act(self, observations: torch.Tensor):
        assert self.model is not None
        logits, value = self.model(observations)  # (num_envs, action_dim)
        dist = Categorical(logits=logits)
        actions = dist.sample()  # (num_envs)
        log_prob = dist.log_prob(actions)  # (num_envs)
        entropy = dist.entropy()  # (num_envs)
        return actions, log_prob, entropy, value.squeeze()

    def push_data(self, **kwargs) -> None:
        for k, v in kwargs.items():
            assert isinstance(v, torch.Tensor)
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def log(self):
        self.n_stats_update += 1

        # logging
        prob = float(torch.exp(torch.stack(self.data["log_prob"])).mean())
        R = float(torch.stack(self.data["rewards"]).sum(dim=0).mean())
        ent = float(torch.stack(self.data["entropy"]).mean())
        v = float(torch.stack(self.data["value"]).mean())

        _avg = lambda x, y, n: (x * n + y * 1) / (n + 1)
        self.avg_R = _avg(self.avg_R, R, self.n_stats_update)
        self.avg_ent = _avg(self.avg_ent, ent, self.n_stats_update)
        self.avg_prob = _avg(self.avg_prob, prob, self.n_stats_update)
        self.value = _avg(self.value, v, self.n_stats_update)
