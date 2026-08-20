# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers to apply physics-backend presets and solver choices to resolved env configs.

Hydra-based entry points (e.g. RL train scripts) select backends with
``presets=newton_mjwarp`` on the CLI. Scripts that build their config with
:func:`~isaaclab_tasks.utils.parse_cfg.parse_env_cfg` (teleop recording, mimic
annotation, dataset generation) bypass Hydra, and ``parse_env_cfg`` resolves
every :class:`~isaaclab_tasks.utils.hydra.PresetCfg` to its default. These
helpers re-apply a named preset (and optionally a solver variant) after the
fact, using the raw registry config to recover the preset alternatives.
"""

from .hydra import apply_overrides, collect_presets
from .parse_cfg import load_cfg_from_registry

# Maps the --solver CLI choice onto the Newton solver-cfg latches read by
# NewtonMJWarpManager._build_solver (backend / adaptive / sap_adaptive).
PHYSICS_SOLVER_CHOICES: dict[str, dict] = {
    "mujoco": {"backend": "mujoco", "adaptive": False, "sap_adaptive": False},
    "mujoco-adaptive": {"backend": "mujoco", "adaptive": True, "sap_adaptive": False},
    "sap": {"backend": "sap", "adaptive": False, "sap_adaptive": False},
    "sap-adaptive": {"backend": "sap", "adaptive": False, "sap_adaptive": True},
    # ICF (Irrotational Contact Fields, Castro et al. 2023) -- SAP's successor,
    # from icf_warp. Fixed-step; compliant contact with a lagged Hunt & Crossley
    # dissipation law.
    "icf": {"backend": "icf", "adaptive": False, "sap_adaptive": False},
    # ICF under the CENIC per-world step-doubling controller. Same contact law,
    # same code path and the same IcfParams as "icf" -- the stepping scheme is
    # the only difference, which is what makes the icf/icf-adaptive pair the
    # single-variable contrast of the three-arm comparison.
    "icf-adaptive": {"backend": "icf", "adaptive": True, "sap_adaptive": False},
}


def apply_physics_preset(env_cfg, task_name: str, preset_name: str):
    """Apply a named physics preset to an already-resolved env config.

    **Contract:** Call this immediately after :func:`parse_env_cfg`, before any
    other config mutations. Other mutations (e.g., manual field updates) are
    discarded and must be applied after this function returns.

    Args:
        env_cfg: Env config returned by :func:`parse_env_cfg` (presets already
            collapsed to defaults).
        task_name: Registered gym task id used to reload the raw config.
        preset_name: Preset field name to select (e.g. ``"newton_mjwarp"``).

    Returns:
        The env config with the preset applied. ``scene.num_envs``,
        ``sim.device``, and ``sim.use_fabric`` are preserved from the input
        config.

    Raises:
        ValueError: If ``preset_name`` does not match any available preset
            alternative.
    """
    num_envs = env_cfg.scene.num_envs
    device = env_cfg.sim.device
    use_fabric = env_cfg.sim.use_fabric
    # parse_env_cfg already resolved every PresetCfg wrapper to its "default"
    # alternative, so re-running apply_overrides on that same object is a
    # no-op: the PresetCfg node it needs to walk no longer exists in the tree.
    # Reload the raw config (still carrying live PresetCfg nodes) and resolve
    # it with `preset_name` selected instead, mirroring what parse_env_cfg
    # would have produced had the preset been chosen from the start.
    raw_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    presets = {"env": collect_presets(raw_cfg), "agent": {}}
    # Validate that preset_name matches at least one preset alternative.
    available_names = set()
    for path_dict in presets["env"].values():
        available_names.update(path_dict.keys())
    available_names.discard("default")
    if preset_name not in available_names:
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {sorted(available_names)}")
    hydra_cfg = {"env": {}, "agent": None}
    env_cfg, _ = apply_overrides(raw_cfg, None, hydra_cfg, [preset_name], [], [], presets)
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    env_cfg.sim.use_fabric = use_fabric
    return env_cfg


def apply_solver_choice(env_cfg, solver: str) -> None:
    """Latch a Newton mjwarp solver variant onto ``env_cfg.sim.physics.solver_cfg``.

    Args:
        env_cfg: Env config whose physics is already a Newton config (apply a
            Newton preset first, e.g. via :func:`apply_physics_preset`).
        solver: One of :data:`PHYSICS_SOLVER_CHOICES`.

    Raises:
        ValueError: If ``solver`` is unknown or the physics config has no
            ``solver_cfg`` (a Newton preset was not applied first).
    """
    if solver not in PHYSICS_SOLVER_CHOICES:
        raise ValueError(f"Unknown solver '{solver}'. Choose from {sorted(PHYSICS_SOLVER_CHOICES)}.")
    solver_cfg = getattr(env_cfg.sim.physics, "solver_cfg", None)
    if solver_cfg is None:
        raise ValueError(
            "env_cfg.sim.physics has no solver_cfg — apply a Newton physics preset first "
            "(e.g. --physics_preset newton_mjwarp)."
        )
    for attr, value in PHYSICS_SOLVER_CHOICES[solver].items():
        setattr(solver_cfg, attr, value)
