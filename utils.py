from copy import deepcopy

import numpy as np
import torch

from minatar import Environment


class RandomState:
    """Hotfix to replace legacy np.random.RandomState by np.random.default_rng.
    This is faster when using deepcopy."""

    def __init__(self, seed) -> None:
        self.rng = np.random.default_rng(seed)

    def choice(self, *args, **kwargs):
        self.rng.choice(args, kwargs)

    def rand(self):
        return self.rng.random()

    def randint(self, *args, **kwargs):
        return self.rng.integers(args, kwargs)

    @property
    def state(self):
        return self.rng.bit_generator.state

    @state.setter
    def state(self, s):
        self.rng.bit_generator.state = s


class MinAtarEnv:
    def __init__(self, game, num_envs: int, seed: int) -> None:
        super().__init__()
        self.game = game
        self.num_envs = num_envs
        self.seed = seed

        seeds = np.random.SeedSequence(seed).spawn(num_envs)
        self.envs = [Environment(game) for i in range(num_envs)]
        # Use np.random.default_rng instead of RandomState
        for i in range(num_envs):
            self.envs[i].random = RandomState(seeds[i])

        # define spaces
        self.action_set = self.envs[0].minimal_action_set()  # use minatar v1
        self.num_action = len(self.action_set)
        self.num_channels = {
            "breakout": 4,
            "asterix": 4,
            "seaquest": 10,
            "space_invaders": 6,
            "freeway": 7,
        }[game]
        self.obs_size = 10 * 10 * self.num_channels

        self.last_observations = self.reset()

    @staticmethod
    def _flatten(obs):
        # return np.array(obs, np.int8).transpose(2, 0, 1).flatten()
        # return np.array(obs, np.int8).transpose(3, 1, 2).flatten()
        return (
            obs.permute(0, 3, 1, 2).reshape(obs.size(0), -1).float()
        )  # (num_envs, 400) in breakout

    @staticmethod
    def _flatten_each(obs):
        return np.array(obs, np.int8).transpose(2, 0, 1).flatten()

    def reset(self):
        """
        >>> env = MinAtarEnv(game="breakout", num_envs=3, seed=0)
        >>> env.reset().size()
        torch.Size([3, 400])
        """
        observations = []
        for env in self.envs:
            env.reset()
            obs = env.state()
            obs = self._flatten_each(obs)
            observations.append(torch.from_numpy(obs))
        self.last_observations = torch.stack(observations).float()
        # observations = self._flatten(observations)
        return deepcopy(self.last_observations)  # (num_envs, 400)

    def step(self, actions):
        """
        >>> env = MinAtarEnv(game="breakout", num_envs=3, seed=0)
        >>> _ = env.reset()
        >>> obs, r, done, info = env.step([0, 1, 2])
        >>> obs.size()
        torch.Size([3, 400])
        >>> obs.type()
        'torch.FloatTensor'
        >>> r.size()
        torch.Size([3])
        >>> done.size()
        torch.Size([3])
        """
        observations = []
        rewards = torch.zeros((self.num_envs,)).float()
        dones = torch.zeros((self.num_envs,)).bool()
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            action = self.action_set[action]  # use minatar v1
            rewards[i], dones[i] = env.act(action)
            obs = env.state()
            obs = self._flatten_each(obs)
            observations.append(torch.from_numpy(obs))

        self.last_observations = torch.stack(observations).float()
        # observations = self._flatten(observations)
        return deepcopy(self.last_observations), rewards, dones, {}

    def init(self, where):
        for i, (should_be_reset, env) in enumerate(zip(where, self.envs)):
            if should_be_reset:
                env.reset()
                obs = env.state()
                obs = self._flatten_each(obs)
                self.last_observations[i, :] = torch.from_numpy(obs).float()
        return deepcopy(self.last_observations)

    def states(self):
        return [self.get_state(i) for i in range(self.num_envs)]

    def get_state(self, i):
        if self.game == "asterix":
            state = {
                "player_x": deepcopy(self.envs[i].env.player_x),
                "player_y": deepcopy(self.envs[i].env.player_y),
                "entities": deepcopy(self.envs[i].env.entities),
                "shot_timer": deepcopy(self.envs[i].env.shot_timer),
                "spawn_speed": deepcopy(self.envs[i].env.spawn_speed),
                "spawn_timer": deepcopy(self.envs[i].env.spawn_timer),
                "move_speed": deepcopy(self.envs[i].env.move_speed),
                "move_timer": deepcopy(self.envs[i].env.move_timer),
                "ramp_timer": deepcopy(self.envs[i].env.ramp_timer),
                "ramp_index": deepcopy(self.envs[i].env.ramp_index),
                "terminal": deepcopy(self.envs[i].env.terminal),
                "random": deepcopy(self.envs[i].random.state),
                "last_action": deepcopy(self.envs[i].last_action),
            }
        if self.game == "breakout":
            state = {
                "ball_y": deepcopy(self.envs[i].env.ball_y),
                "ball_x": deepcopy(self.envs[i].env.ball_x),
                "ball_dir": deepcopy(self.envs[i].env.ball_dir),
                "pos": deepcopy(self.envs[i].env.pos),
                "brick_map": deepcopy(self.envs[i].env.brick_map),
                "strike": deepcopy(self.envs[i].env.strike),
                "last_x": deepcopy(self.envs[i].env.last_x),
                "last_y": deepcopy(self.envs[i].env.last_y),
                "terminal": deepcopy(self.envs[i].env.terminal),
                "random": deepcopy(self.envs[i].random.state),
                "last_action": deepcopy(self.envs[i].last_action),
            }
        if self.game == "freeway":
            state = {
                "cars": deepcopy(self.envs[i].env.cars),
                "pos": deepcopy(self.envs[i].env.pos),
                "move_timer": deepcopy(self.envs[i].env.move_timer),
                "terminate_timer": deepcopy(self.envs[i].env.terminate_timer),
                "terminal": deepcopy(self.envs[i].env.terminal),
                "random": deepcopy(self.envs[i].random.state),
                "last_action": deepcopy(self.envs[i].last_action),
            }
        if self.game == "seaquest":
            state = {
                "oxygen": deepcopy(self.envs[i].env.oxygen),
                "diver_count": deepcopy(self.envs[i].env.diver_count),
                "sub_x": deepcopy(self.envs[i].env.sub_x),
                "sub_y": deepcopy(self.envs[i].env.sub_y),
                "sub_or": deepcopy(self.envs[i].env.sub_or),
                "f_bullets": deepcopy(self.envs[i].env.f_bullets),
                "e_bullets": deepcopy(self.envs[i].env.e_bullets),
                "e_fish": deepcopy(self.envs[i].env.e_fish),
                "e_subs": deepcopy(self.envs[i].env.e_subs),
                "divers": deepcopy(self.envs[i].env.divers),
                "e_spawn_speed": deepcopy(self.envs[i].env.e_spawn_speed),
                "e_spawn_timer": deepcopy(self.envs[i].env.e_spawn_timer),
                "d_spawn_timer": deepcopy(self.envs[i].env.d_spawn_timer),
                "move_speed": deepcopy(self.envs[i].env.move_speed),
                "ramp_index": deepcopy(self.envs[i].env.ramp_index),
                "shot_timer": deepcopy(self.envs[i].env.shot_timer),
                "surface": deepcopy(self.envs[i].env.surface),
                "terminal": deepcopy(self.envs[i].env.terminal),
                "random": deepcopy(self.envs[i].random.state),
                "last_action": deepcopy(self.envs[i].last_action),
            }
        if self.game == "space_invaders":
            state = {
                "pos": deepcopy(self.envs[i].env.pos),
                "f_bullet_map": deepcopy(self.envs[i].env.f_bullet_map),
                "e_bullet_map": deepcopy(self.envs[i].env.e_bullet_map),
                "alien_map": deepcopy(self.envs[i].env.alien_map),
                "alien_dir": deepcopy(self.envs[i].env.alien_dir),
                "enemy_move_interval": deepcopy(self.envs[i].env.enemy_move_interval),
                "alien_move_timer": deepcopy(self.envs[i].env.alien_move_timer),
                "alien_shot_timer": deepcopy(self.envs[i].env.alien_shot_timer),
                "ramp_index": deepcopy(self.envs[i].env.ramp_index),
                "shot_timer": deepcopy(self.envs[i].env.shot_timer),
                "terminal": deepcopy(self.envs[i].env.terminal),
                "random": deepcopy(self.envs[i].random.state),
                "last_action": deepcopy(self.envs[i].last_action),
            }
        return state

    def set_state(self, i, d):
        for k, v in d.items():
            if k == "random":
                self.envs[i].random.state = deepcopy(v)
            elif k == "last_action":
                setattr(self.envs[i], k, deepcopy(v))
            else:
                setattr(self.envs[i].env, k, deepcopy(v))
