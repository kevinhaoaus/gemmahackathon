"""Arm driver registry. Driver modules import their vendor SDK lazily (in connect()), so every
driver module imports fine on a machine without the SDK; only connecting needs it.

    mock    swarmmind.cell.drivers.mock.MockArm      no hardware
    kinova  swarmmind.cell.drivers.kinova.KinovaArm  Kinova Gen3 / Gen3 lite, Kortex API (kortex_api wheel)
    xarm    swarmmind.cell.drivers.xarm.XArm         UFactory xArm 5/6/7, Lite 6 (pip install xarm-python-sdk)
    ur      swarmmind.cell.drivers.ur.URArm          Universal Robots e-Series/CB3 (pip install ur_rtde)
    so101   swarmmind.cell.drivers.so101.SO101Arm    adapter onto capabilities/so101.py (named poses only)
    rebot   swarmmind.cell.drivers.rebot.RebotArm    Seeed reBot Arm B601-DM / B601-RS (pip install motorbridge)
"""
from __future__ import annotations

import importlib
from typing import Callable

DRIVERS: dict[str, str | Callable] = {
    "mock": "swarmmind.cell.drivers.mock:MockArm",
    "kinova": "swarmmind.cell.drivers.kinova:KinovaArm",
    "xarm": "swarmmind.cell.drivers.xarm:XArm",
    "ur": "swarmmind.cell.drivers.ur:URArm",
    "so101": "swarmmind.cell.drivers.so101:SO101Arm",
    "rebot": "swarmmind.cell.drivers.rebot:RebotArm",
}


def register_driver(name: str, factory: str | Callable) -> None:
    """factory: a class / callable taking cfg, or "module.path:ClassName"."""
    DRIVERS[name] = factory


def create(name: str, cfg: dict):
    f = DRIVERS.get(name)
    if f is None:
        raise ValueError(f"unknown arm driver {name!r}; known: {sorted(DRIVERS)}")
    if isinstance(f, str):
        mod, _, attr = f.partition(":")
        f = getattr(importlib.import_module(mod), attr)
    return f(cfg)


def missing_sdk(package: str, pip: str, err: Exception) -> RuntimeError:
    return RuntimeError(f"{package} is not installed ({type(err).__name__}: {err}). Install it on the arm "
                        f"host: {pip}")
