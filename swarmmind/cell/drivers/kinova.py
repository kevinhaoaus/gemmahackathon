"""Kinova Gen3 (7-DoF) / Gen3 lite (6-DoF) over the Kortex API (TCP port 10000).

Ported from the working GemmaHackathon/robot/arm.py + EdgeHand/arm.py KinovaArm: session setup,
ClearFaults + SINGLE_LEVEL_SERVOING, ExecuteAction + ACTION_END/ACTION_ABORT notification wait,
the factory "Home" action, GripperCommand in position mode with a settle wait, and (optional)
IK + joint-space moves for big wrist turns.

SDK: the kortex_api wheel (not on PyPI): pip install kortex_api-2.8.0.post5-py3-none-any.whl
(from Kinova's GitHub releases; a copy is in GemmaHackathonOld/.../GemmaHackathon/). It needs
protobuf 3.20.x.

Units: Kortex Cartesian poses are metres + degrees (theta_x/y/z, i.e. R = Rz(tz) Ry(ty) Rx(tx)),
speed.translation m/s, speed.orientation deg/s, joint angles degrees. Those are our units already;
nothing is scaled. The Kortex gripper finger value is 0.0 = OPEN .. 1.0 = CLOSED, the inverse of
our convention, so it is flipped here (finger = 1 - value) and when reading the feedback.

Config (besides the common keys in cell/arm.py):
    ip: "192.168.1.10"
    port: 10000
    username: admin, password: admin
    model: gen3 | gen3_lite             (speed cap 0.5 vs 0.25 m/s; joint mode is Gen3-only)
    tool_theta_x: 180.0, tool_theta_y: 0.0     tool pointing straight down
    yaw_offset_deg: 0.0                 theta_z = yaw + this. NOTE the old project's straight-down
                                        TOOL_ORIENT was (180, 0, 90): set 90 to reproduce its yaw 0.
    angular_speed: 20.0                 deg/s (also the fastest joint in joint-space moves)
    action_timeout_s: 30
    move_mode: cartesian | joints | auto      auto = joint-space (IK) when yaw changes by more than
                                        joint_mode_yaw_deg (a straight-line wrist turn can hit a
                                        singularity and fault the arm), else straight line
    joint_mode_yaw_deg: 45
    max_joint_swing_deg: 150            refuse IK solutions that swing a joint further (flipped arm)
    home_action: "Home"                 the robot's stored joint-angle action used for home
                                        (set `home: {x, y, z, yaw}` instead to home by pose)
    gripper_settle_s: 2.0
    session_inactivity_timeout_ms: 60000, connection_inactivity_timeout_ms: 2000
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from ..arm import Arm, wrap_deg
from ..arm import log as arm_log
from . import missing_sdk

SPEED_CAP = {"gen3": 0.5, "gen3_lite": 0.25}


class KinovaArm(Arm):
    name = "kinova"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        c = self.cfg
        self.ip = str(c.get("ip", "192.168.1.10"))
        self.port = int(c.get("port", 10000))
        self.username = str(c.get("username", "admin"))
        self.password = str(c.get("password", "admin"))
        self.model = str(c.get("model", "gen3")).lower().replace(" ", "_").replace("-", "_")
        if self.model not in SPEED_CAP:
            raise ValueError(f"kinova model must be gen3 | gen3_lite, got {c.get('model')!r}")
        self.theta_x = float(c.get("tool_theta_x", 180.0))
        self.theta_y = float(c.get("tool_theta_y", 0.0))
        self.yaw_offset = float(c.get("yaw_offset_deg", 0.0))
        self.angular_speed = float(c.get("angular_speed", 20.0))
        self.timeout_s = float(c.get("action_timeout_s", 30.0))
        self.move_mode = str(c.get("move_mode", "cartesian"))
        if self.move_mode not in ("cartesian", "joints", "auto"):
            raise ValueError("kinova move_mode must be cartesian | joints | auto")
        if self.move_mode != "cartesian" and self.model != "gen3":
            raise ValueError("kinova move_mode joints/auto is only ported for the 7-DoF Gen3; "
                             "use move_mode: cartesian on a Gen3 lite")
        self.joint_mode_yaw = float(c.get("joint_mode_yaw_deg", 45.0))
        self.max_swing = float(c.get("max_joint_swing_deg", 150.0))
        self.home_action = str(c.get("home_action", "Home"))
        self.gripper_settle_s = float(c.get("gripper_settle_s", 2.0))
        self.sess_timeout = int(c.get("session_inactivity_timeout_ms", 60000))
        self.conn_timeout = int(c.get("connection_inactivity_timeout_ms", 2000))
        self.B = None
        self.base = None
        self.cyclic = None
        self.session = None
        self.transport = None
        self.router = None
        self._last_yaw: Optional[float] = None

    # ------------------------------------------------------------------ lifecycle
    def _connect_sync(self) -> None:
        try:
            from kortex_api.RouterClient import RouterClient
            from kortex_api.SessionManager import SessionManager
            from kortex_api.TCPTransport import TCPTransport
            from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
            from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
            from kortex_api.autogen.messages import Base_pb2, Session_pb2
        except ImportError as e:
            raise missing_sdk("kortex_api (Kinova Kortex)",
                              "pip install kortex_api-2.8.0.post5-py3-none-any.whl  (Kinova's wheel, "
                              "from github.com/Kinovarobotics/kortex releases; needs protobuf 3.20.x)", e) from None
        self.B = Base_pb2
        self.transport = TCPTransport()
        self.router = RouterClient(self.transport, lambda e: arm_log.error("kortex router error: %s", e))
        self.transport.connect(self.ip, self.port)
        info = Session_pb2.CreateSessionInfo()
        info.username, info.password = self.username, self.password
        info.session_inactivity_timeout = self.sess_timeout
        info.connection_inactivity_timeout = self.conn_timeout
        self.session = SessionManager(self.router)
        self.session.CreateSession(info)
        self.base = BaseClient(self.router)
        self.cyclic = BaseCyclicClient(self.router)
        try:
            self.base.ClearFaults()            # e.g. left over from a crash; does not move the arm
        except Exception as e:  # noqa: BLE001
            arm_log.warning("kortex ClearFaults failed (ok if there were none): %s", e)
        time.sleep(0.5)
        mode = Base_pb2.ServoingModeInformation()
        mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(mode)
        p = self.base.GetMeasuredCartesianPose()
        self._last_yaw = wrap_deg(p.theta_z - self.yaw_offset)
        arm_log.info("kinova %s connected at %s:%d, tool at (%.3f, %.3f, %.3f) m",
                     self.model, self.ip, self.port, p.x, p.y, p.z)

    async def connect(self) -> None:
        try:
            await asyncio.to_thread(self._connect_sync)
        except Exception:
            await asyncio.to_thread(self._close_sync)
            raise

    def _close_sync(self) -> None:
        try:
            if self.session is not None:
                self.session.CloseSession()
        except Exception as e:  # noqa: BLE001
            arm_log.warning("kortex CloseSession: %s", e)
        try:
            if self.transport is not None:
                self.transport.disconnect()
        except Exception as e:  # noqa: BLE001
            arm_log.warning("kortex disconnect: %s", e)
        self.session = self.transport = self.base = self.cyclic = None

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _need(self):
        if self.base is None:
            raise RuntimeError(f"kinova at {self.ip} is not connected")
        return self.base

    async def reset(self) -> None:
        base = self._need()
        await asyncio.to_thread(base.ClearFaults)

    # ------------------------------------------------------------------ actions
    def _run(self, action) -> None:
        """ExecuteAction and block until ACTION_END / ACTION_ABORT (old KinovaArm._run)."""
        base, B = self._need(), self.B
        done = threading.Event()
        result: dict = {}

        def cb(n, e=done):
            if n.action_event in (B.ACTION_END, B.ACTION_ABORT):
                result["event"] = n.action_event
                result["why"] = B.SubErrorCodes.Name(n.abort_details) if n.abort_details else "no details"
                e.set()

        h = base.OnNotificationActionTopic(cb, B.NotificationOptions())
        try:
            base.ExecuteAction(action)
            if not done.wait(self.timeout_s):
                base.Stop()
                raise TimeoutError(f"Kinova action '{action.name}' timed out after {self.timeout_s:g}s")
        finally:
            try:
                base.Unsubscribe(h)
            except Exception:  # noqa: BLE001
                pass
        if result.get("event") == B.ACTION_ABORT:
            raise RuntimeError(f"Kinova aborted '{action.name}': {result['why']}. "
                               f"Check and clear faults in the Web App at http://{self.ip}")

    def _cart_sync(self, x: float, y: float, z: float, theta_z: float, speed: float) -> None:
        a = self.B.Action()
        a.name, a.application_data = "swarmmind_cart", ""
        p = a.reach_pose.target_pose
        p.x, p.y, p.z = float(x), float(y), float(z)                 # metres
        p.theta_x, p.theta_y, p.theta_z = self.theta_x, self.theta_y, float(theta_z)   # degrees
        s = a.reach_pose.constraint.speed
        s.translation, s.orientation = float(speed), self.angular_speed  # m/s, deg/s
        self._run(a)

    def _joints_sync(self, x: float, y: float, z: float, theta_z: float) -> None:
        """Same target solved with the arm's IK and driven joint by joint (old move_joints +
        reach_joints). Refuses a swing > max_joint_swing_deg (IK picked the mirrored arm)."""
        base, B = self._need(), self.B
        d = B.IKData()
        p = d.cartesian_pose
        p.x, p.y, p.z = float(x), float(y), float(z)
        p.theta_x, p.theta_y, p.theta_z = self.theta_x, self.theta_y, float(theta_z)
        cur = [j.value for j in base.GetMeasuredJointAngles().joint_angles]
        for v in cur:
            d.guess.joint_angles.add().value = v          # stay close to the current configuration
        sol = [j.value for j in base.ComputeInverseKinematics(d).joint_angles]
        # Continuous joints (1, 3, 5, 7 = index 0, 2, 4, 6): the equivalent angle nearest the current
        # one. Limited joints (2, 4, 6) stay in 0..360 like the Home action's values.
        target = [c + (s - c + 180.0) % 360.0 - 180.0 if k % 2 == 0 else s % 360.0
                  for k, (c, s) in enumerate(zip(cur, sol))]
        swing = max(abs((t - v + 180.0) % 360.0 - 180.0) for t, v in zip(target, cur))
        if swing > self.max_swing:
            raise ValueError(f"joint move would swing a joint {swing:.0f} deg (IK picked a flipped arm "
                             f"configuration?). Home the arm and retry.")
        a = B.Action()
        a.name, a.application_data = "swarmmind_joints", ""
        for k, v in enumerate(target):
            ja = a.reach_joint_angles.joint_angles.joint_angles.add()
            ja.joint_identifier, ja.value = k, v
        # JOINT_CONSTRAINT_SPEED makes the Gen3 abort (METHOD_FAILED) any move of the big joints,
        # so limit speed with a duration: the joint that moves furthest goes at angular_speed.
        c = a.reach_joint_angles.constraint
        c.type = B.JOINT_CONSTRAINT_DURATION
        c.value = max(1.5, swing / max(1e-3, self.angular_speed))
        self._run(a)

    async def move_to(self, x: float, y: float, z: float, yaw_deg: float, speed: float) -> None:
        self._need()
        speed = min(float(speed), SPEED_CAP[self.model])
        theta_z = wrap_deg(yaw_deg + self.yaw_offset)
        use_joints = self.move_mode == "joints" or (
            self.move_mode == "auto" and self._last_yaw is not None
            and abs(wrap_deg(yaw_deg - self._last_yaw)) > self.joint_mode_yaw)
        if use_joints:
            await asyncio.to_thread(self._joints_sync, x, y, z, theta_z)
        else:
            await asyncio.to_thread(self._cart_sync, x, y, z, theta_z, speed)
        self._last_yaw = wrap_deg(yaw_deg)

    def home_target(self):
        if isinstance(self.cfg.get("home"), dict):
            return super().home_target()
        return None                              # use the robot's stored "Home" action

    def _home_sync(self) -> None:
        base, B = self._need(), self.B
        req = B.RequestedActionType()
        req.action_type = B.REACH_JOINT_ANGLES
        for a in base.ReadAllActions(req).action_list:
            if a.name == self.home_action:
                self._run(a)
                return
        raise RuntimeError(f"no '{self.home_action}' joint action found on the Kinova at {self.ip}")

    async def home(self, speed: float) -> None:
        await asyncio.to_thread(self._home_sync)
        self._last_yaw = None

    # ------------------------------------------------------------------ gripper
    def _finger_pos(self) -> Optional[float]:
        """Measured Kortex finger position, 0 open .. 1 closed (None if no gripper feedback)."""
        m = self.cyclic.RefreshFeedback().interconnect.gripper_feedback.motor
        return m[0].position / 100.0 if m else None      # feedback is percent

    def _gripper_sync(self, value: float) -> None:
        self._need()
        B = self.B
        cmd = B.GripperCommand()
        cmd.mode = B.GRIPPER_POSITION
        f = cmd.gripper.finger.add()
        f.finger_identifier, f.value = 1, 1.0 - float(value)     # ours: 1 open; Kortex: 0 open
        self.base.SendGripperCommand(cmd)
        last, still, t0 = self._finger_pos() or 0.0, 0, time.time()
        while time.time() - t0 < self.gripper_settle_s:          # wait until the fingers stop
            time.sleep(0.05)
            pos = self._finger_pos() or 0.0
            still = still + 1 if abs(pos - last) < 0.002 else 0
            last = pos
            if still >= 6 and time.time() - t0 > 0.5:
                break
        self.grip = round(1.0 - last, 3)

    async def gripper(self, value: float) -> None:
        await asyncio.to_thread(self._gripper_sync, value)

    # ------------------------------------------------------------------ stop / state
    async def stop(self) -> None:
        if self.base is None:
            return
        try:
            # Called from the event loop while a _run thread waits: Stop() makes the running
            # action end with ACTION_ABORT, which releases that thread.
            await asyncio.to_thread(self.base.Stop)
        except Exception as e:  # noqa: BLE001
            arm_log.error("kortex Stop failed: %s", e)

    def _pose_sync(self) -> dict:
        p = self._need().GetMeasuredCartesianPose()
        out = {"x": p.x, "y": p.y, "z": p.z, "yaw": wrap_deg(p.theta_z - self.yaw_offset),
               "theta_x": p.theta_x, "theta_y": p.theta_y}
        try:
            fp = self._finger_pos()
            if fp is not None:
                self.grip = round(1.0 - fp, 3)
        except Exception:  # noqa: BLE001
            pass
        out["gripper"] = self.grip
        return out

    async def pose(self) -> dict:
        return await asyncio.to_thread(self._pose_sync)

    def info(self) -> dict:
        return {"driver": self.name, "model": self.model, "ip": self.ip, "move_mode": self.move_mode,
                "speed_cap": SPEED_CAP[self.model]}
