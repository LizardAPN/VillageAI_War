"""MAPPO layout, GameEnv MAPPO helpers, MAPPOBotEnv, and MAPPOPolicy smoke tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")
pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from village_ai_war.agents.village_obs_builder import VillageObsBuilder
from village_ai_war.env.game_env import GameEnv
from village_ai_war.env.map_generator import generate_initial_state
from village_ai_war.models.mappo_critic import MAPPOCentralizedCritic
from village_ai_war.models.mappo_layout import (
    mappo_local_dim,
    mappo_obs_dim,
    pack_mappo_obs,
    pack_mappo_obs_slots,
)
from village_ai_war.models.mappo_policy import MAPPOPolicy
from village_ai_war.training.mappo_env import MAPPOBotEnv


def _tiny() -> dict[str, Any]:
    return {
        "map": {
            "size": 12,
            "seed": 0,
            "resource_density": 0.1,
            "mountain_density": 0.02,
            "resource_capacity": {"forest": 100, "stone": 50, "field": 999},
        },
        "economy": {
            "harvest_interval": 3,
            "harvest_amount": 5,
            "food_consumption": 1,
            "hunger_damage": 5,
            "bot_cost": {"wood": 50, "food": 100},
            "bot_spawn_delay": 2,
            "farm_food_bonus": 0.5,
        },
        "combat": {
            "stats": {
                "warrior": {"hp": 100, "damage": 10, "attack_range": 1},
                "gatherer": {"hp": 80, "damage": 8, "attack_range": 1},
                "farmer": {"hp": 70, "damage": 5, "attack_range": 1},
                "builder": {"hp": 80, "damage": 8, "attack_range": 1},
            },
            "tower_damage": 15,
            "tower_range": 3,
        },
        "buildings": {
            "townhall": {"hp": 500, "cost": {}},
            "barracks": {"hp": 100, "cost": {"wood": 100}},
            "storage": {"hp": 100, "cost": {"wood": 50}},
            "farm": {"hp": 100, "cost": {"wood": 80}},
            "tower": {"hp": 100, "cost": {"stone": 100}},
            "wall": {"hp": 100, "cost": {"stone": 30}},
            "citadel": {"hp": 100, "cost": {"stone": 200, "wood": 150}},
            "citadel_pop_bonus": 5,
        },
        "game": {
            "max_ticks": 50,
            "manager_interval": 5,
            "initial_resources": {"wood": 200, "stone": 100, "food": 500},
            "initial_bots": 1,
            "initial_buildings": ["barracks", "storage"],
            "blueprint_adjacent_to_townhall": True,
            "max_bots_for_role_change": 16,
        },
        "rewards": {
            "bot": {
                "alpha": 0.7,
                "warrior": {
                    "damage_dealt": 0.1,
                    "kill": 5.0,
                    "damage_taken": -0.05,
                    "death": -10.0,
                    "noop": -0.01,
                },
                "gatherer": {
                    "resource_collected": 0.5,
                    "damage_taken": -0.05,
                    "death": -10.0,
                    "noop": -0.01,
                },
                "farmer": {
                    "food_produced": 0.5,
                    "damage_taken": -0.05,
                    "death": -10.0,
                    "noop": -0.01,
                },
                "builder": {
                    "block_placed": 2.0,
                    "repair_pct": 0.1,
                    "damage_taken": -0.05,
                    "death": -10.0,
                    "noop": -0.01,
                },
                "global_modes": {"defend_coeff": -0.05, "attack_coeff": 0.05, "gather_coeff": 0.1},
            },
            "village": {
                "economy_coeff": 0.01,
                "kill_reward": 5.0,
                "loss_penalty": -3.0,
                "building_reward": 10.0,
                "stagnation_penalty": -0.05,
                "stagnation_threshold": 50,
                "win": 1000.0,
                "loss": -1000.0,
            },
        },
        "rendering": {"cell_size": 16, "fps": 60},
    }


def test_village_obs_build_map_and_vec_match_build() -> None:
    cfg = _tiny()
    rng = np.random.default_rng(0)
    st = generate_initial_state(cfg, rng)
    vb = VillageObsBuilder(int(cfg["map"]["size"]))
    full = vb.build(st, team=0)
    mp = vb.build_map(st, team=0)
    vec = vb.build_village_vec(st, team=0)
    np.testing.assert_array_equal(full["map"], mp)
    np.testing.assert_array_equal(full["village"], vec)


def test_bot_step_deterministic_regression() -> None:
    """Two fresh envs with the same seed should match for several bot-mode steps."""
    cfg = _tiny()
    a = GameEnv(cfg, mode="bot", team=0, render_mode=None)
    b = GameEnv(cfg, mode="bot", team=0, render_mode=None)
    o0, _ = a.reset(seed=99)
    o1, _ = b.reset(seed=99)
    np.testing.assert_array_equal(o0, o1)
    for _ in range(5):
        act = 0
        o0, r0, t0, tr0, i0 = a.step(act)
        o1, r1, t1, tr1, i1 = b.step(act)
        np.testing.assert_array_equal(o0, o1)
        assert r0 == r1 and t0 == t1 and tr0 == tr1
        assert i0["tick"] == i1["tick"]


def test_queue_and_simulation_tick_matches_step_with_opponent() -> None:
    cfg = _tiny()
    cfg["game"]["initial_bots"] = 1
    e1 = GameEnv(cfg, mode="bot", team=0, render_mode=None)
    e2 = GameEnv(cfg, mode="bot", team=0, render_mode=None)
    e1.reset(seed=7)
    e2.reset(seed=7)
    o_a, r_a, t_a, tr_a, i_a = e1.step_with_opponent(0, 0)

    e2.snapshot_bot_positions_for_tick()
    e2.begin_mappo_tick()
    e2.queue_bot_action(0, e2._controlled_bot_id, 0)
    e2.queue_bot_action(1, e2._opponent_controlled_bot_id, 0)
    o_b, r_b, t_b, tr_b, i_b = e2._simulation_tick(manager_action=None, learner_bot_action=0)

    np.testing.assert_array_equal(o_a, o_b)
    assert r_a == r_b and t_a == t_b and tr_a == tr_b
    assert i_a["tick"] == i_b["tick"]


def test_mappo_layout_and_critic_shapes() -> None:
    n = 12
    k = 16
    assert mappo_obs_dim(n) == mappo_local_dim() + n * n * 6 + 40
    assert mappo_obs_dim(n, k) == k * mappo_local_dim() + n * n * 6 + 40
    import torch

    crit = MAPPOCentralizedCritic(map_shape=(n, n, 6), village_vec_dim=40, hidden_dim=64)
    b = 3
    m = torch.zeros(b, n, n, 6)
    v = torch.zeros(b, 40)
    out = crit(m, v)
    assert out.shape == (b, 1)


def test_mappo_bot_env_and_policy(tmp_path: Path) -> None:
    cfg = _tiny()
    k = int(cfg["game"]["max_bots_for_role_change"])
    pool = tmp_path / "bot_pool"
    env = MAPPOBotEnv(cfg, team=0, opponent_pool_dir=str(pool))
    obs, info = env.reset(seed=0)
    assert obs.shape == (mappo_obs_dim(12, k),)
    assert env.action_space.shape == (k,)
    assert "global_state" in info
    assert info["global_state"]["map"].shape == (12, 12, 6)
    noop = np.zeros((k,), dtype=np.int64)
    obs2, _r, _t, _tr, info2 = env.step(noop)
    assert obs2.shape == obs.shape
    assert "global_state" in info2
    env.close()

    pool2 = tmp_path / "bot_pool2"
    pool2.mkdir()
    venv = DummyVecEnv([lambda: MAPPOBotEnv(_tiny(), opponent_pool_dir=str(pool2))])
    model = PPO(
        MAPPOPolicy,
        venv,
        n_steps=64,
        batch_size=32,
        verbose=0,
        policy_kwargs={"map_size": 12, "critic_hidden_dim": 64, "n_bot_slots": k},
    )
    obs = venv.reset()
    act, _ = model.predict(obs, deterministic=True)
    assert act.shape == (1, k)
    venv.close()


def test_pack_mappo_obs_roundtrip_dims() -> None:
    n = 12
    loc = np.zeros((mappo_local_dim(),), dtype=np.float32)
    mp = np.zeros((n, n, 6), dtype=np.float32)
    v0 = np.zeros((20,), dtype=np.float32)
    v1 = np.zeros((20,), dtype=np.float32)
    p = pack_mappo_obs(loc, mp, v0, v1)
    assert p.shape == (mappo_obs_dim(n),)
    k = 3
    locs = np.zeros((k, mappo_local_dim()), dtype=np.float32)
    ps = pack_mappo_obs_slots(locs, mp, v0, v1)
    assert ps.shape == (mappo_obs_dim(n, k),)


def test_game_env_terminal_info_keys() -> None:
    cfg = _tiny()
    cfg["game"]["max_ticks"] = 5
    e = GameEnv(cfg, mode="bot", team=0, render_mode=None)
    e.reset(seed=0)
    seen = False
    for _ in range(30):
        _o, _r, t, tr, info = e.step(0)
        if t or tr:
            assert "episode_outcome" in info
            assert "terminal_reason" in info
            seen = True
            break
    assert seen
    e.close()
