"""Seeed Studio reBot Arm B601 (6 joints + gripper): B601-DM (Damiao) or B601-RS (Robstride).

SDK: Seeed's "motorbridge" (pip install motorbridge; docs https://motorbridge.seeedstudio.com,
arm reference code https://github.com/Seeed-Projects/reBotArm_control_py). Only motorbridge is needed;
pinocchio is optional (ik_backend: pinocchio). Both are imported lazily, so this module imports on any
machine and only connect() needs the SDK.

Connection (verified against the motorbridge docs and reBotArm_control_py/actuator/rebotarm.py):
  * B601-DM: the Damiao "USB2CAN" serial bridge (HPMicro HPM5300, USB VID 0x2E88 PID 0x4603). It
    enumerates as a CDC serial port: "USB Serial Device (COMx)" on Windows, /dev/ttyACM* on Linux.
    CAN frames are tunnelled over it with Damiao's serial framing, which motorbridge implements:
    Controller.from_dm_serial(port, baud=921600). 921600 is the documented Damiao baud. The
    bridge's CAN side runs at 1 Mbit/s classic CAN.
  * B601-RS: a CAN interface. On Linux use SocketCAN ("can0", brought up with
    `ip link set can0 type can bitrate 1000000 && ip link set can0 up`) and Controller("can0").
    On Windows motorbridge maps "can0" to a PEAK PCAN adapter (PCANBasic.dll).
  * Motor ids (config/rebotarm_dm.yaml, rebotarm_rs.yaml): joints 1..6 = 0x01..0x06, gripper 0x07.
    DM feedback ids are id + 0x10 (0x11..0x17). Models: 4340P for j1-j3, 4310 for j4-j6 and the
    gripper. RS feedback (host) id is 0xFD; models rs-06 for j1-j3, rs-00 for j4-j6 and the gripper.
  * Motor units: rad, rad/s (motorbridge Motor API). Control mode: POS_VEL (Mode.POS_VEL,
    send_pos_vel(pos, vlim)), as in the official arm code (arm_control_mode: posvel). For DM the
    position/velocity loop gains are written to registers 25 vel_kp, 26 vel_ki, 27 pos_kp, 28 pos_ki
    with the values from rebotarm_dm.yaml before switching mode, as rebotarm.py does.

Kinematics: the joint chain of urdf/DM/urdf/ReBot_Arm_DM.urdf (base_link -> end_link) is embedded
below (REBOT_DM_URDF: joint origins, axes and limits only), so no file is needed for the DM arm.
Any other URDF (e.g. the RS one, urdf/RS/urdf/ReBot_Arm_RS.urdf with tool frame gripper_end) is
parsed with xml.etree. The inverse kinematics is damped least squares with joint limits, a full
tool-down orientation constraint and a reachability check (it refuses targets it cannot reach to
within ik_pos_tol / ik_rot_tol). The tool frame's +x axis is the approach direction (the URDF's
end_joint turns link6's z into end_link's x) and +y is the finger sliding axis.

Yaw (the cell's convention, cell/arm.py): the tool points straight down, and yaw is the direction,
in the arm base frame, of the finger (closing) axis, plus yaw_offset_deg.

Config (capability_config["arm.cell"] or ["arm.<id>"], besides the common keys in cell/arm.py; note
build_arm requires an explicit `workspace` for this driver):
    driver: rebot
    variant: dm | rs                  default dm
    port: "COM5"                      DM bridge serial port (Windows COMx, Linux /dev/ttyACM0), or a CAN
                                      channel ("can0") for RS. Default "COM5" (dm) / "can0" (rs)
    transport: auto | dm_serial | socketcan | socketcanfd   default auto: COMx, /dev/tty* -> dm_serial
    baud: 921600                      dm_serial only
    motors: [{name, id, feedback_id, model}, ...]   7 entries (6 joints + gripper); default per variant
    write_gains: true                 DM: write the POS_VEL loop gains (registers 25-28) at connect
    gains: [[vel_kp, vel_ki, pos_kp, pos_ki], ...]  7 rows; default the rebotarm_dm.yaml values
    joint_signs: [1, 1, 1, 1, 1, 1]   motor angle = sign * urdf angle + offset
    joint_offsets_rad: [0, 0, 0, 0, 0, 0]
    urdf_path: null                   null = the embedded DM chain
    base_frame: base_link             URDF root link of the chain
    tool_frame: end_link              URDF link used as the tool point (rs: gripper_end)
    tool_offset: [0, 0, 0]            extra offset (m) in the tool frame, e.g. to the finger-pad centre
    ik_backend: builtin | pinocchio   pinocchio needs urdf_path (pip install pin)
    ik_pos_tol: 0.0002                m, IK must converge to within this
    ik_rot_tol: 0.002                 rad
    yaw_offset_deg: 0
    joint_speed_max: 0.8              rad/s cap for every joint (on top of the Cartesian max_speed)
    yaw_speed: 60                     deg/s cap for pure wrist rotation
    control_hz: 50                    joint target streaming rate
    cart_step_m: 0.005                Cartesian path discretisation for straight-line moves
    max_joint_step_rad: 0.25          bigger jump between neighbouring path points = branch flip: refused
    tilt_tol_deg: 5                   start pose further than this from tool-down: joint-space move
    settle_tol_rad: 0.03              end of move: every joint within this of its target
    settle_timeout_s: 3.0
    home_joints: null                 [6 rad] native joint-space home (else the Cartesian `home`)
    park_joints: null                 [6 rad] moved to (slowly) before disabling on close
    disable_on_close: true            the arm goes limp when disabled: park it or support it first
    gripper_open_rad: (required)      gripper motor angle when fully open (rad, motor frame)
    gripper_closed_rad: 0.0           fully closed (the LeRobot guide's zero pose is gripper closed)
    gripper_vlim: 2.0                 rad/s
    gripper_mode: posvel | force_pos  force_pos limits grip current (send_force_pos ratio)
    gripper_force_ratio: 0.3          force_pos only, 0..1
    gripper_timeout_s: 3.0
    fault_codes: []                   MotorState.status_code values treated as a fault
    connect_timeout_s: 15             (the capability's timeout around connect())

Read-only first test, which enables no motor and moves nothing. It registers the motors, asks each
for a status frame and prints position, velocity, torque, temperature and status:
    .venv/Scripts/python -m swarmmind.cell.drivers.rebot --probe --port COM5
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any, Optional, Sequence

import numpy as np

from ..arm import Arm, ArmConfigError, num, wrap_deg

# ------------------------------------------------------------------ embedded robot data
# Kinematic elements of urdf/DM/urdf/ReBot_Arm_DM.urdf (Seeed-Projects/reBotArm_control_py), copied
# from the published file: origins, axes and limits only (inertias, meshes and fingers dropped).
REBOT_DM_URDF = """<robot name="ReBot_Arm_DM">
  <link name="base_link"/><link name="link1"/><link name="link2"/><link name="link3"/>
  <link name="link4"/><link name="link5"/><link name="link6"/><link name="end_link"/>
  <joint name="joint1" type="revolute"><parent link="base_link"/><child link="link1"/>
    <origin xyz="-8.416E-05 0 0.08465" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-2.8" upper="2.8" effort="27" velocity="50"/></joint>
  <joint name="joint2" type="revolute"><parent link="link1"/><child link="link2"/>
    <origin xyz="0.020084 0.031625 0.05555" rpy="-1.5708 0 0"/><axis xyz="0 0 -1"/>
    <limit lower="-3.14" upper="0" effort="27" velocity="50"/></joint>
  <joint name="joint3" type="revolute"><parent link="link2"/><child link="link3"/>
    <origin xyz="-0.264 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="0" effort="27" velocity="50"/></joint>
  <joint name="joint4" type="revolute"><parent link="link3"/><child link="link4"/>
    <origin xyz="0.2426 -0.054 -0.001625" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.87" upper="1.57" effort="7" velocity="200"/></joint>
  <joint name="joint5" type="revolute"><parent link="link4"/><child link="link5"/>
    <origin xyz="0.078308 -0.0375 -0.03" rpy="-1.5708 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="7" velocity="200"/></joint>
  <joint name="joint6" type="revolute"><parent link="link5"/><child link="link6"/>
    <origin xyz="0.028008 0 0.04" rpy="0 1.5708 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="7" velocity="200"/></joint>
  <joint name="end_joint" type="fixed"><parent link="link6"/><child link="end_link"/>
    <origin xyz="0 0 0.15539" rpy="0 -1.5708 3.1415"/></joint>
</robot>"""

# config/rebotarm_dm.yaml / rebotarm_rs.yaml: (name, motor_id, feedback_id, model)
MOTORS = {
    "dm": [("joint1", 0x01, 0x11, "4340P"), ("joint2", 0x02, 0x12, "4340P"), ("joint3", 0x03, 0x13, "4340P"),
           ("joint4", 0x04, 0x14, "4310"), ("joint5", 0x05, 0x15, "4310"), ("joint6", 0x06, 0x16, "4310"),
           ("gripper", 0x07, 0x17, "4310")],
    "rs": [("joint1", 0x01, 0xFD, "rs-06"), ("joint2", 0x02, 0xFD, "rs-06"), ("joint3", 0x03, 0xFD, "rs-06"),
           ("joint4", 0x04, 0xFD, "rs-00"), ("joint5", 0x05, 0xFD, "rs-00"), ("joint6", 0x06, 0xFD, "rs-00"),
           ("gripper", 0x07, 0xFD, "rs-00")],
}
# rebotarm_dm.yaml POS_VEL gains [vel_kp, vel_ki, pos_kp, pos_ki] per motor (joint1..6, gripper)
DM_GAINS = [[0.0125, 0.004, 150.0, 0.5]] * 3 + [[0.0008, 0.002, 70.0, 1.0]] + \
    [[0.0008, 0.002, 60.0, 1.0]] * 2 + [[0.0008, 0.002, 50.0, 1.0]]
DM_GAIN_REGS = (25, 26, 27, 28)            # vel_kp, vel_ki, pos_kp, pos_ki (Damiao register ids)
N_JOINTS = 6

# A few elbow-up, tool-down-ish seed postures for the IK's restarts (radians, URDF frame).
IK_SEEDS = [
    [0.0, -1.2, -1.6, 0.0, 0.0, 0.0],
    [0.0, -0.8, -2.0, 0.6, 0.0, 0.0],
    [0.0, -1.6, -1.2, -0.5, 0.0, 0.0],
    [0.0, -0.4, -2.4, 1.2, 0.0, 0.0],
    [0.0, -2.0, -0.8, -1.2, 0.0, 0.0],
]


def missing_motorbridge(err: Exception) -> RuntimeError:
    return RuntimeError(
        f"the reBot driver needs Seeed's motorbridge SDK ({type(err).__name__}: {err}). Install it on the arm "
        f"host: pip install motorbridge  (docs: https://motorbridge.seeedstudio.com, arm code: "
        f"https://github.com/Seeed-Projects/reBotArm_control_py)")


# ------------------------------------------------------------------ geometry helpers
def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    """URDF rpy: fixed-axis roll about x, then pitch about y, then yaw about z (R = Rz Ry Rx)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis: np.ndarray, q: float) -> np.ndarray:
    x, y, z = axis
    c, s, v = math.cos(q), math.sin(q), 1.0 - math.cos(q)
    return np.array([[c + x * x * v, x * y * v - z * s, x * z * v + y * s],
                     [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
                     [z * x * v - y * s, z * y * v + x * s, c + z * z * v]])


def tool_down_rotation(yaw_deg: float) -> np.ndarray:
    """Tool frame columns: x (approach) straight down, y (finger axis) at yaw in the base xy plane."""
    a = math.radians(yaw_deg)
    xt = np.array([0.0, 0.0, -1.0])
    yt = np.array([math.cos(a), math.sin(a), 0.0])
    return np.column_stack([xt, yt, np.cross(xt, yt)])


def rotation_error(R: np.ndarray, Rd: np.ndarray) -> np.ndarray:
    """Small-angle orientation error (world frame) that turns R towards Rd."""
    return 0.5 * sum(np.cross(R[:, i], Rd[:, i]) for i in range(3))


def rotation_angle(R: np.ndarray, Rd: np.ndarray) -> float:
    c = (np.trace(R.T @ Rd) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


# ------------------------------------------------------------------ kinematic chain
class Chain:
    """Serial chain base -> tool from a URDF: fixed transforms and revolute joints."""

    def __init__(self, urdf_xml: str, base: str = "base_link", tool: str = "end_link",
                 tool_offset: Sequence[float] = (0.0, 0.0, 0.0)):
        root = ET.fromstring(urdf_xml)
        by_child = {}
        for j in root.iter("joint"):
            child = j.find("child")
            if child is not None:
                by_child[child.get("link")] = j
        path = []
        link = tool
        while link != base:
            j = by_child.get(link)
            if j is None:
                raise ArmConfigError(f"URDF: no chain from {base!r} to {tool!r} (stuck at link {link!r})")
            path.append(j)
            link = j.find("parent").get("link")
        path.reverse()
        # steps: ("fixed", 4x4) or ("rev", axis); joints collect names and limits
        self.steps: list[tuple[str, Any]] = []
        self.names: list[str] = []
        lo, hi = [], []
        for j in path:
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3] = rpy_matrix(*rpy)
            T[:3, 3] = xyz
            self.steps.append(("fixed", T))
            jt = j.get("type")
            if jt in ("revolute", "continuous"):
                ax = j.find("axis")
                a = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
                self.steps.append(("rev", a / np.linalg.norm(a)))
                self.names.append(j.get("name"))
                lim = j.find("limit")
                if jt == "revolute" and lim is not None:
                    lo.append(float(lim.get("lower", -math.pi)))
                    hi.append(float(lim.get("upper", math.pi)))
                else:
                    lo.append(-2 * math.pi)
                    hi.append(2 * math.pi)
            elif jt != "fixed":
                raise ArmConfigError(f"URDF joint {j.get('name')!r} of type {jt!r} in the arm chain is not supported")
        T = np.eye(4)
        T[:3, 3] = [float(v) for v in tool_offset]
        self.steps.append(("fixed", T))
        self.lower, self.upper = np.array(lo), np.array(hi)
        self.n = len(self.names)
        # reach bound for the quick reachability check: sum of every link translation
        self.reach = sum(float(np.linalg.norm(s[1][:3, 3])) for s in self.steps if s[0] == "fixed")

    def fk(self, q: Sequence[float]) -> np.ndarray:
        return self._walk(q)[0]

    def _walk(self, q):
        T = np.eye(4)
        origins, axes = [], []
        k = 0
        for kind, v in self.steps:
            if kind == "fixed":
                T = T @ v
            else:
                origins.append(T[:3, 3].copy())
                axes.append(T[:3, :3] @ v)
                R = np.eye(4)
                R[:3, :3] = axis_angle(v, float(q[k]))
                T = T @ R
                k += 1
        return T, origins, axes

    def fk_jac(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Tool pose and the 6 x n geometric Jacobian (rows: linear, angular; world frame)."""
        T, origins, axes = self._walk(q)
        p = T[:3, 3]
        J = np.zeros((6, self.n))
        for i, (o, a) in enumerate(zip(origins, axes)):
            J[:3, i] = np.cross(a, p - o)
            J[3:, i] = a
        return T, J


class PinocchioChain:
    """Same interface as Chain, on pinocchio (pip install pin). Needs the URDF file."""

    def __init__(self, urdf_path: str, joint_names: Sequence[str], tool: str,
                 tool_offset: Sequence[float] = (0.0, 0.0, 0.0)):
        try:
            import pinocchio as pin  # noqa: PLC0415 - optional dependency
        except ImportError as e:
            raise RuntimeError(f"ik_backend: pinocchio needs the pinocchio package ({e}); pip install pin, "
                               f"or use ik_backend: builtin") from None
        self.pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.fid = self.model.getFrameId(tool)
        if self.fid >= self.model.nframes:
            raise ArmConfigError(f"URDF {urdf_path}: no frame {tool!r}")
        self.names = list(joint_names)
        ids = [self.model.getJointId(n) for n in self.names]
        self.iq = [self.model.joints[i].idx_q for i in ids]
        self.iv = [self.model.joints[i].idx_v for i in ids]
        self.lower = np.array([self.model.lowerPositionLimit[i] for i in self.iq])
        self.upper = np.array([self.model.upperPositionLimit[i] for i in self.iq])
        self.n = len(self.names)
        self.off = np.array([float(v) for v in tool_offset])
        self.reach = float("inf")

    def fk_jac(self, q):
        pin = self.pin
        qf = pin.neutral(self.model)
        for i, v in zip(self.iq, q):
            qf[i] = v
        pin.forwardKinematics(self.model, self.data, qf)
        pin.updateFramePlacements(self.model, self.data)
        M = self.data.oMf[self.fid]
        Jf = pin.computeFrameJacobian(self.model, self.data, qf, self.fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        J = np.array(Jf)[:, self.iv]
        T = np.array(M.homogeneous)
        if np.any(self.off):
            r = T[:3, :3] @ self.off                  # offset point: v = v_frame + w x r
            T[:3, 3] = T[:3, 3] + r
            J[:3, :] = J[:3, :] + np.cross(J[3:, :].T, r).T
        return T, J

    def fk(self, q):
        return self.fk_jac(q)[0]


class IKError(ValueError):
    """Target not reachable (outside the arm's reach, joint limits or tool-down constraint)."""


def solve_ik(chain, p_target: Sequence[float], R_target: np.ndarray, seeds: Sequence[Sequence[float]],
             pos_tol: float = 2e-4, rot_tol: float = 2e-3, max_iter: int = 150,
             damping: float = 0.02, max_step: float = 0.2, rot_weight: float = 0.3) -> tuple[np.ndarray, float, float]:
    """Damped least squares from each seed in turn (first success wins). Joints are clamped to their
    limits every iteration. Returns (q, pos_err_m, rot_err_rad); raises IKError if no seed converges."""
    p_t = np.asarray(p_target, float)
    if np.linalg.norm(p_t) > chain.reach + 1e-6:      # quick bound: the sum of every link length
        raise IKError(f"target {np.round(p_t, 4).tolist()} is beyond the arm's reach ({chain.reach:.3f} m)")
    W = np.diag([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight])
    best = (None, float("inf"), float("inf"))
    for seed in seeds:
        q = np.clip(np.asarray(seed, float), chain.lower, chain.upper)
        for _ in range(max_iter):
            T, J = chain.fk_jac(q)
            ep = p_t - T[:3, 3]
            er = rotation_error(T[:3, :3], R_target)
            pe, re = float(np.linalg.norm(ep)), rotation_angle(T[:3, :3], R_target)
            if pe + 0.1 * re < best[1] + 0.1 * best[2]:
                best = (q.copy(), pe, re)
            if pe <= pos_tol and re <= rot_tol:
                return q, pe, re
            e = W @ np.concatenate([ep, er])
            Jw = W @ J
            lam2 = (damping * (1.0 + 10.0 * min(1.0, pe))) ** 2
            dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam2 * np.eye(6), e)
            m = float(np.max(np.abs(dq)))
            if m > max_step:
                dq *= max_step / m
            q_new = np.clip(q + dq, chain.lower, chain.upper)
            if float(np.max(np.abs(q_new - q))) < 1e-10:
                break                                # pinned at a limit: try the next seed
            q = q_new
    q, pe, re = best
    raise IKError(f"target {np.round(p_t, 4).tolist()} is not reachable tool-down within the joint limits "
                  f"(best: {pe * 1000:.1f} mm, {math.degrees(re):.1f} deg off)")


# ------------------------------------------------------------------ port presence (unplug detection)
def port_present(port: str) -> Optional[bool]:
    """True/False if we can tell whether the device exists, None if we can't (e.g. PCAN on Windows).
    Windows: the live COM list in HKLM\\HARDWARE\\DEVICEMAP\\SERIALCOMM (a USB CDC port disappears
    from it the moment the cable is pulled). Linux: the /dev node, or /sys/class/net/<can>."""
    p = str(port or "")
    if sys.platform.startswith("win"):
        if not p.upper().startswith("COM"):
            return None
        try:
            import winreg  # noqa: PLC0415
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM") as k:
                i, names = 0, set()
                while True:
                    try:
                        names.add(str(winreg.EnumValue(k, i)[1]).upper())
                    except OSError:
                        break
                    i += 1
            return p.upper() in names
        except OSError:
            return False                         # the key is absent when no COM port exists at all
    if p.startswith("/dev/"):
        return os.path.exists(p)
    if os.path.isdir("/sys/class/net"):
        return os.path.exists(f"/sys/class/net/{p}")
    return None


# ------------------------------------------------------------------ the driver
class RebotArm(Arm):
    name = "rebot"
    supports_cartesian = True

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        c = self.cfg
        self.variant = str(c.get("variant", "dm")).lower()
        if self.variant not in MOTORS:
            raise ArmConfigError(f"rebot variant must be dm | rs, got {self.variant!r}")
        self.port = str(c.get("port") or ("COM5" if self.variant == "dm" else "can0"))
        self.transport = str(c.get("transport", "auto"))
        if self.transport == "auto":
            self.transport = "dm_serial" if (self.port.upper().startswith("COM") or self.port.startswith("/dev/tty")) \
                else "socketcan"
        if self.transport not in ("dm_serial", "socketcan", "socketcanfd"):
            raise ArmConfigError(f"rebot transport must be auto | dm_serial | socketcan | socketcanfd, got {self.transport!r}")
        self.baud = int(c.get("baud", 921600))
        motors = c.get("motors")
        if motors:
            if len(motors) != N_JOINTS + 1:
                raise ArmConfigError("rebot motors: give 7 entries (joint1..joint6, gripper)")
            self.motor_spec = [(str(m["name"]), int(m["id"]), int(m["feedback_id"]), str(m["model"])) for m in motors]
        else:
            self.motor_spec = list(MOTORS[self.variant])
        self.vendor = "damiao" if self.variant == "dm" else "robstride"
        self.write_gains = bool(c.get("write_gains", self.variant == "dm"))
        self.gains = [list(map(float, g)) for g in (c.get("gains") or DM_GAINS)]
        self.signs = np.array([float(v) for v in c.get("joint_signs", [1.0] * N_JOINTS)])
        self.offsets = np.array([float(v) for v in c.get("joint_offsets_rad", [0.0] * N_JOINTS)])
        if len(self.signs) != N_JOINTS or len(self.offsets) != N_JOINTS:
            raise ArmConfigError("joint_signs and joint_offsets_rad need 6 values")
        self.yaw_offset = num(c.get("yaw_offset_deg", 0.0), "yaw_offset_deg")
        self.joint_speed_max = num(c.get("joint_speed_max", 0.8), "joint_speed_max")
        self.yaw_speed = num(c.get("yaw_speed", 60.0), "yaw_speed")
        self.control_hz = max(5.0, min(200.0, num(c.get("control_hz", 50.0), "control_hz")))
        self.cart_step = num(c.get("cart_step_m", 0.005), "cart_step_m")
        self.max_joint_step = num(c.get("max_joint_step_rad", 0.25), "max_joint_step_rad")
        self.tilt_tol = num(c.get("tilt_tol_deg", 5.0), "tilt_tol_deg")
        self.settle_tol = num(c.get("settle_tol_rad", 0.03), "settle_tol_rad")
        self.settle_timeout = num(c.get("settle_timeout_s", 3.0), "settle_timeout_s")
        self.pos_tol = num(c.get("ik_pos_tol", 2e-4), "ik_pos_tol")
        self.rot_tol = num(c.get("ik_rot_tol", 2e-3), "ik_rot_tol")
        self.home_joints = self._joints_opt("home_joints")
        self.park_joints = self._joints_opt("park_joints")
        self.disable_on_close = bool(c.get("disable_on_close", True))
        if c.get("gripper_open_rad") is None:
            raise ArmConfigError("rebot: gripper_open_rad is required (gripper motor angle in rad when fully open; "
                                 "read it with --probe after opening the gripper by hand, motors disabled)")
        self.g_open = num(c["gripper_open_rad"], "gripper_open_rad")
        self.g_closed = num(c.get("gripper_closed_rad", 0.0), "gripper_closed_rad")
        self.g_vlim = num(c.get("gripper_vlim", 2.0), "gripper_vlim")
        self.g_mode = str(c.get("gripper_mode", "posvel"))
        if self.g_mode not in ("posvel", "force_pos"):
            raise ArmConfigError("gripper_mode must be posvel | force_pos")
        self.g_ratio = min(1.0, max(0.0, num(c.get("gripper_force_ratio", 0.3), "gripper_force_ratio")))
        self.g_timeout = num(c.get("gripper_timeout_s", 3.0), "gripper_timeout_s")
        # on contact, re-aim this far past where the fingers stopped (rad) instead of keeping the
        # original (closed) target, which would squeeze at full stiffness. Small = gentle.
        self.g_squeeze = abs(num(c.get("gripper_squeeze_rad", 0.03), "gripper_squeeze_rad"))
        self.fault_codes = {int(v) for v in (c.get("fault_codes") or [])}
        # kinematics
        tool = str(c.get("tool_frame") or ("end_link" if self.variant == "dm" else "gripper_end"))
        base = str(c.get("base_frame") or "base_link")
        off = [float(v) for v in c.get("tool_offset", [0.0, 0.0, 0.0])]
        urdf_path = c.get("urdf_path")
        if urdf_path is None and self.variant != "dm":
            raise ArmConfigError("rebot variant rs needs urdf_path (urdf/RS/urdf/ReBot_Arm_RS.urdf from "
                                 "Seeed-Projects/reBotArm_control_py); the embedded chain is the DM arm's")
        if str(c.get("ik_backend", "builtin")) == "pinocchio":
            if not urdf_path:
                raise ArmConfigError("ik_backend: pinocchio needs urdf_path")
            names = Chain(open(urdf_path, encoding="utf-8").read(), base, tool).names
            self.chain = PinocchioChain(str(urdf_path), names, tool, off)
        else:
            xml = open(urdf_path, encoding="utf-8").read() if urdf_path else REBOT_DM_URDF
            self.chain = Chain(xml, base, tool, off)
        if self.chain.n != N_JOINTS:
            raise ArmConfigError(f"URDF chain {base}->{tool} has {self.chain.n} joints, the reBot has {N_JOINTS}")
        # runtime
        self.ctrl = None
        self.motors: list = []
        self.Mode = None
        self.q: Optional[np.ndarray] = None            # last measured joint angles, URDF frame
        self.status: list = []
        self._gen = 0
        self._pool: Optional[concurrent.futures.ThreadPoolExecutor] = None

    def _joints_opt(self, key: str) -> Optional[np.ndarray]:
        v = self.cfg.get(key)
        if v is None:
            return None
        if len(v) != N_JOINTS:
            raise ArmConfigError(f"{key} needs 6 joint angles (rad)")
        return np.array([float(x) for x in v])

    # ---------------------------------------------------------------- SDK plumbing (one bus thread)
    async def _call(self, fn, *a):
        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="rebot-bus")
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *a)

    def _open_sync(self) -> None:
        try:
            import motorbridge  # noqa: PLC0415
            from motorbridge import Controller, Mode  # noqa: PLC0415
        except ImportError as e:
            raise missing_motorbridge(e) from None
        if self.transport == "dm_serial":
            ctrl = Controller.from_dm_serial(self.port, baud=self.baud)
        elif self.transport == "socketcanfd":
            ctrl = Controller.from_socketcanfd(self.port)
        else:
            ctrl = Controller(self.port)
        add = ctrl.add_damiao_motor if self.vendor == "damiao" else ctrl.add_robstride_motor
        self.motors = [add(mid, fid, model) for _n, mid, fid, model in self.motor_spec]
        self.ctrl, self.Mode, self._mb = ctrl, Mode, motorbridge

    def _read_sync(self) -> np.ndarray:
        """Request one status frame from every motor and return all 7 motor angles (rad)."""
        for m in self.motors:
            m.request_feedback()
        self.ctrl.poll_feedback_once()
        out, status = [], []
        for (name, *_), m in zip(self.motor_spec, self.motors):
            st = m.get_state()
            if st is None:
                raise ConnectionError(f"rebot: no feedback from {name} (motor not powered, wrong id, or 24 V off?)")
            out.append(float(st.pos))
            status.append(int(getattr(st, "status_code", 0)))
        self.status = status
        bad = [(self.motor_spec[i][0], s) for i, s in enumerate(status) if s in self.fault_codes]
        if bad:
            raise RuntimeError(f"rebot motor fault: {bad}")
        return np.array(out)

    def _enable_sync(self) -> np.ndarray:
        for i, m in enumerate(self.motors):
            try:
                m.clear_error()
            except Exception:  # noqa: BLE001 - not every firmware has a clearable error state
                pass
            if self.vendor == "damiao" and self.write_gains:
                for reg, val in zip(DM_GAIN_REGS, self.gains[i]):
                    m.write_register_f32(reg, float(val))
            gm = self.Mode.FORCE_POS if (i == N_JOINTS and self.g_mode == "force_pos") else self.Mode.POS_VEL
            m.ensure_mode(gm, 1000)
        pos = self._read_sync()
        # Hold where we are BEFORE enabling, so the POS_VEL loop never chases a stale target.
        # Check on the real arm: that a send_pos_vel to a disabled DM motor is accepted (it is
        # resent right after enable below in any case).
        self._hold_sync(pos, 0.3)
        for m in self.motors:
            m.enable()
        self._hold_sync(pos, 0.3)
        return pos

    def _hold_sync(self, motor_pos: np.ndarray, vlim: float) -> None:
        for i, m in enumerate(self.motors[:N_JOINTS]):
            m.send_pos_vel(float(motor_pos[i]), float(vlim))

    def _send_sync(self, motor_q: np.ndarray, vlim: np.ndarray) -> None:
        for i, m in enumerate(self.motors[:N_JOINTS]):
            m.send_pos_vel(float(motor_q[i]), float(vlim[i]))

    def _grip_send_sync(self, target: float) -> None:
        m = self.motors[N_JOINTS]
        if self.g_mode == "force_pos":
            m.send_force_pos(float(target), float(self.g_vlim), float(self.g_ratio))
        else:
            m.send_pos_vel(float(target), float(self.g_vlim))

    def _close_sync(self) -> None:
        if self.ctrl is None:
            return
        try:
            if self.disable_on_close:
                self.ctrl.disable_all()
        finally:
            for fn in (self.ctrl.shutdown, self.ctrl.close):
                try:
                    fn()
                except Exception:  # noqa: BLE001 - port already gone after an unplug
                    pass
            self.ctrl, self.motors = None, []

    # ---------------------------------------------------------------- frames
    def motor_to_urdf(self, motor: np.ndarray) -> np.ndarray:
        return (np.asarray(motor[:N_JOINTS], float) - self.offsets) * self.signs

    def urdf_to_motor(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, float) * self.signs + self.offsets

    def tool_pose(self, q: Sequence[float]) -> dict:
        T = self.chain.fk(q)
        yt, xt = T[:3, 1], T[:3, 0]
        yaw = math.degrees(math.atan2(yt[1], yt[0])) - self.yaw_offset
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -xt[2]))))
        return {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3]), "yaw": wrap_deg(yaw),
                "tilt_deg": tilt}

    def ik(self, x: float, y: float, z: float, yaw_deg: float, seed: Optional[Sequence[float]] = None) -> np.ndarray:
        """Joint angles (URDF frame) for the tool-down pose; raises IKError if unreachable."""
        Rd = tool_down_rotation(yaw_deg + self.yaw_offset)
        seeds = ([list(seed)] if seed is not None else []) + IK_SEEDS
        # joint1 points the arm at the target: seed it there to save iterations
        az = math.atan2(y, x)
        seeds += [[az] + s[1:] for s in IK_SEEDS]
        q, _pe, _re = solve_ik(self.chain, (x, y, z), Rd, seeds, self.pos_tol, self.rot_tol)
        return q

    def check_reachable(self, x: float, y: float, z: float, yaw_deg: float) -> None:
        """Raise IKError if (x, y, z, yaw) is unreachable tool-down. Used by arm.cell before any motion."""
        self.ik(x, y, z, yaw_deg, self.q)

    def plan_line(self, q0: np.ndarray, x: float, y: float, z: float, yaw_deg: float) -> list[np.ndarray]:
        """Joint waypoints for a straight tool line from FK(q0) to the target (tool-down all the way).
        If the start is not tool-down (e.g. a folded rest pose) the path is one joint-space segment."""
        start = self.tool_pose(q0)
        q_goal = self.ik(x, y, z, yaw_deg, q0)
        if start["tilt_deg"] > self.tilt_tol:
            return [np.asarray(q0, float), q_goal]
        d = math.dist((start["x"], start["y"], start["z"]), (x, y, z))
        dyaw = wrap_deg(yaw_deg - start["yaw"])
        n = max(1, math.ceil(d / self.cart_step), math.ceil(abs(dyaw) / 3.0))
        path, q = [np.asarray(q0, float)], np.asarray(q0, float)
        for k in range(1, n + 1):
            f = k / n
            qk = self.ik(start["x"] + (x - start["x"]) * f, start["y"] + (y - start["y"]) * f,
                         start["z"] + (z - start["z"]) * f, start["yaw"] + dyaw * f, q)
            if float(np.max(np.abs(qk - q))) > self.max_joint_step:
                raise IKError(f"straight-line path flips the arm configuration at {100 * f:.0f}% "
                              f"(joint jump {float(np.max(np.abs(qk - q))):.2f} rad); choose another target")
            path.append(qk)
            q = qk
        return path

    def timing(self, path: list[np.ndarray], speed: float) -> float:
        """Duration (s) of a path: Cartesian speed, joint speed cap and wrist yaw speed, whichever is slowest."""
        cart = sum(float(np.linalg.norm(self.chain.fk(b)[:3, 3] - self.chain.fk(a)[:3, 3]))
                   for a, b in zip(path, path[1:]))
        joint = sum(float(np.max(np.abs(b - a))) for a, b in zip(path, path[1:]))
        yaw = abs(math.degrees(float(path[-1][5] - path[0][5])))
        return max(cart / max(speed, 1e-6), joint / self.joint_speed_max, yaw / self.yaw_speed, 0.1)

    # ---------------------------------------------------------------- Arm interface
    async def connect(self) -> None:
        present = port_present(self.port)
        if present is False:
            raise ConnectionError(f"rebot: {self.port} is not present (USB adapter unplugged?)")
        await self._call(self._close_sync)
        await self._call(self._open_sync)
        try:
            motor = await self._call(self._enable_sync)
        except Exception:
            await self._call(self._close_sync)
            raise
        self.q = self.motor_to_urdf(motor)
        self.grip = self._grip_value(motor[N_JOINTS])

    async def close(self) -> None:
        self._gen += 1
        if self.ctrl is not None and self.park_joints is not None:
            try:
                await self._move_joints([self.q, self.park_joints] if self.q is not None else [self.park_joints],
                                        speed=0.03, gen=self._gen)
            except Exception:  # noqa: BLE001 - closing must go on
                pass
        await self._call(self._close_sync)
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

    async def ping(self) -> dict:
        """Health check (called about once a second by arm.cell): port present and every motor answers."""
        if port_present(self.port) is False:
            raise ConnectionError(f"{self.port} disappeared (USB unplugged)")
        if self.ctrl is None:
            raise ConnectionError("rebot not connected")
        motor = await self._call(self._read_sync)
        self.q = self.motor_to_urdf(motor)
        self.grip = self._grip_value(motor[N_JOINTS])
        return {"status": list(self.status)}

    async def pose(self) -> dict:
        if self.ctrl is None:
            raise ConnectionError("rebot not connected")
        motor = await self._call(self._read_sync)
        self.q = self.motor_to_urdf(motor)
        self.grip = self._grip_value(motor[N_JOINTS])
        p = self.tool_pose(self.q)
        return {**p, "joints": [round(float(v), 4) for v in self.q], "gripper": self.grip,
                "status": list(self.status)}

    def home_target(self) -> Optional[dict]:
        if self.home_joints is not None:
            return None
        return super().home_target()

    async def home(self, speed: float) -> None:
        q0 = await self._measured()
        await self._move_joints([q0, self.home_joints], speed, self._gen)

    async def move_to(self, x: float, y: float, z: float, yaw_deg: float, speed: float) -> None:
        gen = self._gen
        q0 = await self._measured()
        path = self.plan_line(q0, x, y, z, yaw_deg)        # every waypoint solved before anything moves
        await self._move_joints(path, speed, gen)

    async def stop(self) -> None:
        self._gen += 1
        try:
            if self.ctrl is not None:
                motor = await self._call(self._read_sync)
                await self._call(self._hold_sync, motor, 0.5)
        except Exception:  # noqa: BLE001 - stop must not raise
            pass

    async def gripper(self, value: float) -> None:
        gen = self._gen
        target = self.g_closed + (self.g_open - self.g_closed) * float(value)
        await self._call(self._grip_send_sync, target)
        t0, last, still = time.monotonic(), None, 0.0
        dt = 0.05
        while time.monotonic() - t0 < self.g_timeout:
            await asyncio.sleep(dt)
            if gen != self._gen:
                return
            motor = await self._call(self._read_sync)
            g = float(motor[N_JOINTS])
            if abs(g - target) < self.settle_tol:
                break
            # fingers blocked by the part: position stops changing -> grasped
            still = still + dt if last is not None and abs(g - last) < 0.002 else 0.0
            if still >= 0.25:
                # hold just past the contact point, not the original target: limits the squeeze
                direction = 1.0 if target > g else -1.0
                await self._call(self._grip_send_sync, g + direction * self.g_squeeze)
                break
            last = g
        self.grip = float(value)

    def _grip_value(self, motor_g: float) -> float:
        span = self.g_open - self.g_closed
        return float(min(1.0, max(0.0, (motor_g - self.g_closed) / span))) if span else 0.0

    async def _measured(self) -> np.ndarray:
        motor = await self._call(self._read_sync)
        self.q = self.motor_to_urdf(motor)
        return self.q

    async def _move_joints(self, path: list, speed: float, gen: int) -> None:
        """Stream the path at control_hz, eased in and out, with per-joint vlim from the planned rate."""
        path = [np.clip(np.asarray(p, float), self.chain.lower, self.chain.upper) for p in path]
        if len(path) < 2:
            return
        dur = self.timing(path, speed)
        seg = np.array([0.0] + [float(np.max(np.abs(b - a))) + 1e-9 for a, b in zip(path, path[1:])])
        cum = np.cumsum(seg) / np.sum(seg)
        dt = 1.0 / self.control_hz
        n = max(1, math.ceil(dur / dt))
        peak = 1.5 * max(float(np.max(np.abs(path[-1] - path[0]))), float(np.sum(seg))) / dur   # smoothstep peak
        vlim = np.full(N_JOINTS, min(self.joint_speed_max, max(0.1, 1.5 * peak)))
        t0 = time.monotonic()
        for k in range(1, n + 1):
            if gen != self._gen:
                return                                       # stopped: stop() already holds position
            s = k / n
            s = s * s * (3 - 2 * s)                          # ease in/out
            i = int(np.searchsorted(cum, s, side="right")) - 1
            i = min(max(i, 0), len(path) - 2)
            f = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)
            q = path[i] + (path[i + 1] - path[i]) * min(1.0, max(0.0, f))
            await self._call(self._send_sync, self.urdf_to_motor(q), vlim)
            delay = t0 + k * dt - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        goal = path[-1]
        t1 = time.monotonic()
        while True:
            if gen != self._gen:
                return
            q = await self._measured()
            err = float(np.max(np.abs(q - goal)))
            if err <= self.settle_tol:
                return
            if time.monotonic() - t1 > self.settle_timeout:
                raise RuntimeError(f"rebot did not reach the target (max joint error {err:.3f} rad after "
                                   f"{self.settle_timeout:g} s; blocked, overloaded or gains too soft?)")
            await asyncio.sleep(0.05)

    def info(self) -> dict:
        return {"driver": self.name, "variant": self.variant, "port": self.port, "transport": self.transport,
                "joint_speed_max": self.joint_speed_max}


# ------------------------------------------------------------------ read-only probe
def probe(port: str, variant: str = "dm", baud: int = 921600) -> int:
    """Open the bus, register the 7 motors, request one status frame from each and print it.
    Enables nothing and commands nothing: no mode change, no register write, no position command.
    (request_feedback sends a status query frame on the CAN bus; that is the only traffic.)"""
    try:
        from motorbridge import Controller  # noqa: PLC0415
    except ImportError as e:
        print(missing_motorbridge(e))
        return 2
    print(f"port {port}: present={port_present(port)}")
    spec = MOTORS[variant]
    serial = port.upper().startswith("COM") or port.startswith("/dev/tty")
    ctrl = Controller.from_dm_serial(port, baud=baud) if serial else Controller(port)
    try:
        add = ctrl.add_damiao_motor if variant == "dm" else ctrl.add_robstride_motor
        motors = [add(mid, fid, model) for _n, mid, fid, model in spec]
        for _ in range(3):
            for m in motors:
                m.request_feedback()
            ctrl.poll_feedback_once()
            time.sleep(0.05)
        ok = 0
        for (name, mid, fid, model), m in zip(spec, motors):
            st = m.get_state()
            if st is None:
                print(f"{name:8s} id 0x{mid:02X} fb 0x{fid:02X} {model:6s}: NO FEEDBACK")
                continue
            ok += 1
            print(f"{name:8s} id 0x{mid:02X} fb 0x{fid:02X} {model:6s}: pos {st.pos:+.4f} rad "
                  f"({math.degrees(st.pos):+7.2f} deg) vel {st.vel:+.3f} torq {st.torq:+.3f} "
                  f"t_mos {getattr(st, 't_mos', float('nan')):.0f}C status {getattr(st, 'status_code', '?')}")
        print(f"{ok}/{len(spec)} motors answered")
        return 0 if ok == len(spec) else 1
    finally:
        for fn in (ctrl.shutdown, ctrl.close):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass


def _main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="reBot B601 read-only probe (enables nothing, moves nothing)")
    ap.add_argument("--probe", action="store_true", required=True)
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--variant", default="dm", choices=sorted(MOTORS))
    ap.add_argument("--baud", type=int, default=921600)
    a = ap.parse_args(argv)
    return probe(a.port, a.variant, a.baud)


if __name__ == "__main__":
    raise SystemExit(_main())
