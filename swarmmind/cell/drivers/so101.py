"""SO-101 as a cell arm: an adapter onto capabilities/so101.py's ArmController.

LIMITATION, by design: capabilities/so101.py is joint-space only and has no inverse kinematics,
so this driver CANNOT do Cartesian moves. supports_cartesian = False: move_to, pick and place are
refused with a clear error before anything moves. What works: home, gripper (1 open .. 0 closed,
same convention), stop, and named poses ({"action": "named_pose", "name": "ready"}) from the
SO-101 calibration file's "poses". pose() reports joints (ticks), x/y/z/yaw are None.

If this node also runs the "arm.so101" capability, its controller (and serial port) is reused.
Otherwise this adapter opens its own controller from the same config keys as
capability_config["arm.so101"] (port, mock, calibration, ...; see capabilities/so101.py). Put them
directly in this arm's config, or under a nested `so101: {...}`.
"""
from __future__ import annotations

from typing import Optional

from ..arm import Arm


async def _quiet(_p: dict) -> None:
    return None


class SO101Arm(Arm):
    name = "so101"
    supports_cartesian = False

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.ctl = None
        self._own = False
        self.poses: list[str] = []

    def _conf(self) -> dict:
        nested = self.cfg.get("so101")
        return dict(nested) if isinstance(nested, dict) else dict(self.cfg)

    async def connect(self) -> None:
        from ...capabilities import so101 as so101_cap
        ctl = so101_cap.get_controller()
        if ctl is not None:
            self.ctl, self._own = ctl, False
        else:
            self.ctl = await so101_cap.build_controller(self._conf(), publish=None)
            self._own = True
        self.poses = sorted(self.ctl.poses)

    async def close(self) -> None:
        if self.ctl is not None and self._own:
            await self.ctl.stop_loops()
            self.ctl.close()
        self.ctl = None

    def _need(self):
        if self.ctl is None:
            raise RuntimeError("so101 adapter is not connected")
        return self.ctl

    async def _do(self, step: dict) -> dict:
        from ...capabilities.so101 import MotionAborted as SoAborted
        try:
            return await self._need().handle(step, _quiet)
        except SoAborted as e:
            raise RuntimeError(f"so101 motion aborted: {e}") from None

    def home_target(self) -> Optional[dict]:
        return None

    async def home(self, speed: float) -> None:
        await self._do({"action": "home"})

    async def move_to(self, x: float, y: float, z: float, yaw_deg: float, speed: float) -> None:
        raise RuntimeError("the so101 driver has no inverse kinematics, so Cartesian move_to/pick/place "
                           "are not supported; use home, gripper or named_pose")

    async def named_pose(self, name: str) -> dict:
        return await self._do({"action": "pose", "name": name})

    async def gripper(self, value: float) -> None:
        await self._do({"action": "gripper", "value": float(value)})
        self.grip = float(value)

    async def stop(self) -> None:
        if self.ctl is not None:
            self.ctl.hold("cell arm stop", estop=False)

    async def pose(self) -> dict:
        ctl = self._need()
        st = ctl.state()
        self.grip = st.get("gripper")
        return {"x": None, "y": None, "z": None, "yaw": None, "joints": st.get("joints"),
                "gripper": self.grip, "so101_mode": st.get("mode")}

    def info(self) -> dict:
        return {"driver": self.name, "cartesian": False, "named_poses": self.poses,
                "shared_controller": self.ctl is not None and not self._own}
