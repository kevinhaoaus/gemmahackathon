"""SO-101 kinematics: forward kinematics from the URDF, top-down inverse kinematics, and the
joint-radian <-> STS3215 servo-tick mapping.

Model source (numbers embedded below, copied from the real file):
    TheRobotStudio/SO-ARM100, Simulation/SO101/so101_new_calib.urdf
    commit 385e8d7c68e24945df6c60d9bd68837a4b7411ae (2025-07-02)
"new_calib" means URDF joint zero = the middle of each joint's range, which is LeRobot's
calibration convention (and capabilities/so101.py's default home = middle of the range).
Pass `urdf="path/to/so101.urdf"` (driver config `kinematics: {urdf: ...}`) to use another file.

Frames
------
URDF base_link: the frame the URDF is written in (origin under the base, z up).
ARM frame (what this module and swarmmind.cell.arm speak): metres, z up, origin on the
shoulder_pan axis at base_link z = 0, +x = the direction the arm reaches at shoulder_pan = 0.
So shoulder_pan is (up to sign) atan2(y, x). The arm frame is base_link translated and rotated
about z; it is derived from the model, not hand-entered.

Tool (TCP) = URDF gripper_frame_link (between the jaws, near the fingertips). Its z axis is the
approach direction. "Top-down" = TCP z axis pointing straight down (arm frame -z).
Yaw (degrees) = rotation of the tool about the vertical, 0 when shoulder_pan = 0 and
wrist_roll = 0 with the tool pointing down; then yaw = pan + roll (each with the model's sign).

Joint model: 5 revolute arm joints (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll); the gripper jaw is separate (it does not move the TCP).

Ticks: tick = home + sign * rad * 4096 / (2 pi)   (STS3215: 4096 ticks/rev = 0.088 deg/tick)
home = the raw tick at URDF zero. Calibration (per joint): {home, min, max, sign}.
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

URDF_SOURCE = ("TheRobotStudio/SO-ARM100 Simulation/SO101/so101_new_calib.urdf "
               "@ 385e8d7c68e24945df6c60d9bd68837a4b7411ae")
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TICKS_PER_REV = 4096
TICKS_PER_RAD = TICKS_PER_REV / (2.0 * math.pi)
DEG_PER_TICK = 360.0 / TICKS_PER_REV          # 0.0879 deg


@dataclass(frozen=True)
class JointSpec:
    name: str
    xyz: tuple[float, float, float]           # origin in the parent link frame (m)
    rpy: tuple[float, float, float]           # origin rotation (rad, URDF fixed-axis roll-pitch-yaw)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lower: float = -math.pi
    upper: float = math.pi
    fixed: bool = False


# Verbatim from so101_new_calib.urdf (base_link -> ... -> gripper_frame_link).
URDF_CHAIN: tuple[JointSpec, ...] = (
    JointSpec("shoulder_pan", (0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159),
              (0, 0, 1), -1.91986, 1.91986),
    JointSpec("shoulder_lift", (-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0),
              (0, 0, 1), -1.74533, 1.74533),
    JointSpec("elbow_flex", (-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708),
              (0, 0, 1), -1.69, 1.69),
    JointSpec("wrist_flex", (-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708),
              (0, 0, 1), -1.65806, 1.65806),
    JointSpec("wrist_roll", (5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159),
              (0, 0, 1), -2.74385, 2.84121),
    JointSpec("gripper_frame_joint", (-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0),
              fixed=True),
)


class Unreachable(ValueError):
    """IK refused a target. `code` is one of: out_of_reach, below_table, joint_limit, invalid."""

    def __init__(self, code: str, reason: str):
        super().__init__(f"unreachable ({code.replace('_', ' ')}): {reason}")
        self.code = code
        self.reason = reason


# ====================================================================== math helpers
def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(a: np.ndarray, t: float) -> np.ndarray:
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)


def _tf(rot: np.ndarray, pos: Sequence[float]) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = pos
    return t


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _floats(s: Optional[str], n: int = 3) -> tuple[float, ...]:
    if not s:
        return (0.0,) * n
    v = tuple(float(t) for t in s.split())
    if len(v) != n:
        raise ValueError(f"expected {n} numbers, got {s!r}")
    return v


def load_urdf_chain(path: str | Path, tip_link: str = "gripper_frame_link") -> tuple[JointSpec, ...]:
    """Read the serial chain base -> tip_link from a URDF. It must contain the 5 SO-101 arm joints
    as revolute joints; fixed joints in between are kept."""
    root = ET.parse(Path(path)).getroot()
    by_child: dict[str, ET.Element] = {}
    for j in root.findall("joint"):
        c = j.find("child")
        if c is not None:
            by_child[c.get("link", "")] = j
    chain: list[JointSpec] = []
    link = tip_link
    while link in by_child:
        j = by_child[link]
        o = j.find("origin")
        ax = j.find("axis")
        lim = j.find("limit")
        jtype = j.get("type", "fixed")
        fixed = jtype == "fixed"
        if not fixed and jtype not in ("revolute", "continuous"):
            raise ValueError(f"{path}: joint {j.get('name')} has unsupported type {jtype!r}")
        lo, hi = -math.pi, math.pi
        if lim is not None and jtype == "revolute":
            lo, hi = float(lim.get("lower", -math.pi)), float(lim.get("upper", math.pi))
        chain.append(JointSpec(j.get("name", "?"),
                               _floats(o.get("xyz") if o is not None else None),
                               _floats(o.get("rpy") if o is not None else None),
                               _floats(ax.get("xyz")) if ax is not None and not fixed else (0.0, 0.0, 1.0),
                               lo, hi, fixed))
        link = j.find("parent").get("link", "")
    chain.reverse()
    names = [c.name for c in chain if not c.fixed]
    if names != list(ARM_JOINTS):
        raise ValueError(f"{path}: revolute joints from base to {tip_link} are {names}, want {list(ARM_JOINTS)}")
    return tuple(chain)


# ====================================================================== calibration / ticks
@dataclass
class JointCal:
    home: float = 2048.0          # raw tick at URDF zero
    sign: int = 1                 # +1: ticks increase with the URDF joint angle
    min: int = 0                  # tick bounds
    max: int = TICKS_PER_REV - 1

    def to_ticks(self, rad: float) -> int:
        return int(round(self.home + self.sign * rad * TICKS_PER_RAD))

    def to_rad(self, ticks: float) -> float:
        return (float(ticks) - self.home) * self.sign / TICKS_PER_RAD

    def rad_limits(self) -> tuple[float, float]:
        a, b = self.to_rad(self.min), self.to_rad(self.max)
        return (min(a, b), max(a, b))


def parse_calibration(raw: dict, bounds: Optional[dict] = None, require_home: bool = False) -> dict[str, JointCal]:
    """Per-joint {home, min, max, sign} for the 5 arm joints.

    raw: our format {"joints": {j: {home, min, max, sign}}} or a LeRobot calibration file
         {j: {range_min, range_max, drive_mode, ...}} (home = middle of the range, which is URDF
         zero for so101_new_calib; drive_mode 1 -> sign -1). An explicit "sign" always wins.
    bounds: capabilities/so101.py's resolved bounds {j: {min, max, home}}; used for any value the
         raw file leaves out, so the IK limits match what the controller will actually clamp to.
    require_home: the home tick (URDF zero) must come from `raw` itself, never from `bounds`
         (whose home may just be a guess such as the boot pose).
    Raises ValueError if a joint has no home tick.
    """
    raw = raw or {}
    jraw = raw.get("joints") if isinstance(raw.get("joints"), dict) else raw
    out = {}
    for j in ARM_JOINTS:
        c = jraw.get(j) if isinstance(jraw.get(j), dict) else {}
        b = (bounds or {}).get(j) or {}
        lo = c.get("min", c.get("range_min", b.get("min")))
        hi = c.get("max", c.get("range_max", b.get("max")))
        home = c.get("home")
        if home is None and "range_min" in c and "range_max" in c:
            home = (float(c["range_min"]) + float(c["range_max"])) / 2.0
        if home is None and not require_home:
            home = b.get("home")
        if home is None:
            raise ValueError(f"calibration has no home tick (URDF zero) for {j}: give joints.{j}.home "
                             f"or range_min/range_max")
        sign = c.get("sign")
        if sign is None:
            sign = -1 if int(c.get("drive_mode", 0) or 0) == 1 else 1
        if int(sign) not in (1, -1):
            raise ValueError(f"calibration {j}.sign must be 1 or -1, got {sign!r}")
        lo = 0 if lo is None else int(lo)
        hi = TICKS_PER_REV - 1 if hi is None else int(hi)
        out[j] = JointCal(float(home), int(sign), min(lo, hi), max(lo, hi))
    return out


def load_calibration_file(path: str | Path, bounds: Optional[dict] = None,
                          require_home: bool = True) -> dict[str, JointCal]:
    with open(Path(path), "r", encoding="utf-8") as f:
        return parse_calibration(json.load(f) or {}, bounds, require_home)


def joints_to_ticks(cal: dict[str, JointCal], q: Sequence[float]) -> dict[str, int]:
    return {j: cal[j].to_ticks(float(v)) for j, v in zip(ARM_JOINTS, q)}


def ticks_to_joints(cal: dict[str, JointCal], ticks: dict) -> np.ndarray:
    return np.array([cal[j].to_rad(ticks[j]) for j in ARM_JOINTS], dtype=float)


# ====================================================================== kinematics
@dataclass
class IKResult:
    q: np.ndarray                 # radians, ARM_JOINTS order
    pos_err_m: float
    rot_err_deg: float
    iters: int
    elbow: str = ""               # "elbow+" | "elbow-" | "seed": which starting guess converged

    def joints(self) -> dict[str, float]:
        return {j: float(v) for j, v in zip(ARM_JOINTS, self.q)}


@dataclass
class SO101Kinematics:
    chain: tuple[JointSpec, ...] = URDF_CHAIN
    limits: Optional[dict[str, tuple[float, float]]] = None   # rad; default the URDF limits
    table_z: float = 0.0          # arm-frame z of the table; TCP and joint centres must stay >= it
    rot_weight: float = 0.1       # m per rad when mixing position and orientation errors
    tol_m: float = 5e-5           # refinement stops below these
    tol_rad: float = math.radians(0.02)
    accept_m: float = 5e-4        # a solution must be at least this good (spec: < 0.5 mm)
    accept_rad: float = math.radians(0.5)
    _base: np.ndarray = field(init=False, repr=False)

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None) -> "SO101Kinematics":
        """cfg (driver `kinematics:` block): {urdf: path, table_z: 0.0, limits_deg: {joint: [lo, hi]}}."""
        cfg = dict(cfg or {})
        chain = load_urdf_chain(cfg["urdf"]) if cfg.get("urdf") else URDF_CHAIN
        lim = None
        if isinstance(cfg.get("limits_deg"), dict):
            lim = {j: (math.radians(float(v[0])), math.radians(float(v[1]))) for j, v in cfg["limits_deg"].items()}
        return cls(chain=chain, limits=lim, table_z=float(cfg.get("table_z", 0.0)))

    def __post_init__(self) -> None:
        self._joints = [c for c in self.chain if not c.fixed]
        if [c.name for c in self._joints] != list(ARM_JOINTS):
            raise ValueError(f"chain joints {[c.name for c in self._joints]} != {list(ARM_JOINTS)}")
        self._origins = [_tf(_rpy(*c.rpy), c.xyz) for c in self.chain]
        self._axes = [np.array(c.axis, dtype=float) / np.linalg.norm(c.axis) for c in self.chain]
        self._rev_axes = [a for a, c in zip(self._axes, self.chain) if not c.fixed]
        lim = {c.name: (c.lower, c.upper) for c in self._joints}
        lim.update(self.limits or {})
        self.limits = {j: (float(lim[j][0]), float(lim[j][1])) for j in ARM_JOINTS}
        self._base = np.eye(4)
        self._base = self._derive_arm_frame()
        self._derive_planar_model()

    # ---------------------------------------------------------------- FK
    def _frames(self, q: Sequence[float]) -> tuple[list[np.ndarray], np.ndarray]:
        """World (arm-frame) transform of each revolute joint (after its origin, before its
        rotation) and the TCP transform."""
        t = self._base.copy()
        joint_tf = []
        k = 0
        for c, o, a in zip(self.chain, self._origins, self._axes):
            t = t @ o
            if not c.fixed:
                joint_tf.append(t.copy())
                t = t @ _tf(_axis_angle(a, float(q[k])), (0, 0, 0))
                k += 1
        return joint_tf, t

    def fk(self, q: Sequence[float]) -> np.ndarray:
        """4x4 TCP transform in the arm frame."""
        return self._frames(q)[1]

    def points(self, q: Sequence[float]) -> list[np.ndarray]:
        """Joint centres + TCP in the arm frame (for the table-clearance check)."""
        jt, tcp = self._frames(q)
        return [f[:3, 3] for f in jt] + [tcp[:3, 3]]

    def tool_yaw(self, rot: np.ndarray) -> float:
        """Tool yaw in degrees (from the TCP x axis) relative to the pan=0, roll=0 reference."""
        return math.degrees(_wrap(math.atan2(rot[1, 0], rot[0, 0]) - self._yaw_ref))

    def pose(self, q: Sequence[float]) -> dict:
        """{x, y, z (m), yaw (deg), tilt (deg from straight down)}."""
        t = self.fk(q)
        down = float(np.clip(-t[2, 2], -1.0, 1.0))
        return {"x": float(t[0, 3]), "y": float(t[1, 3]), "z": float(t[2, 3]),
                "yaw": self.tool_yaw(t[:3, :3]), "tilt": math.degrees(math.acos(down))}

    def jacobian(self, q: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Geometric Jacobian (6x5: linear, angular) and the TCP transform."""
        jt, tcp = self._frames(q)
        p = tcp[:3, 3]
        jac = np.zeros((6, len(jt)))
        for i, (f, a) in enumerate(zip(jt, self._rev_axes)):
            w = f[:3, :3] @ a
            jac[:3, i] = np.cross(w, p - f[:3, 3])
            jac[3:, i] = w
        return jac, tcp

    # ---------------------------------------------------------------- frame / planar model
    def _derive_arm_frame(self) -> np.ndarray:
        """Arm frame = base_link moved onto the pan axis and turned so +x is the reach direction
        at pan = 0 (perpendicular to the pitch axes, away from the base)."""
        q0 = np.zeros(5)
        jt, _ = self._frames(q0)
        pan_axis = jt[0][:3, :3] @ self._axes[0]
        if abs(abs(pan_axis[2]) - 1.0) > 1e-4:
            raise ValueError("shoulder_pan axis is not vertical in base_link; unsupported model")
        origin = jt[0][:3, 3].copy()
        origin[2] = 0.0
        pitch = jt[1][:3, :3] @ self._axes[1]
        fwd = np.cross(pitch, [0.0, 0.0, 1.0])
        fwd[2] = 0.0
        fwd /= np.linalg.norm(fwd)
        # Sign: the side the arm reaches to at q = 0 (TCP, else the wrist, ahead of the pan axis;
        # for so101_new_calib q = 0 is upper arm up, forearm and gripper straight forward).
        _, tcp0 = self._frames(q0)
        reach = tcp0[:3, 3] - origin
        if abs(float(np.dot(fwd[:2], reach[:2]))) < 1e-3:
            reach = jt[3][:3, 3] - origin
        if float(np.dot(fwd[:2], reach[:2])) < 0:
            fwd = -fwd
        left = np.cross([0.0, 0.0, 1.0], fwd)
        rot = np.array([fwd, left, [0.0, 0.0, 1.0]])            # rows: arm axes in base_link
        return _tf(rot, -rot @ origin)

    def _derive_planar_model(self) -> None:
        """Planar parameters for the analytic seed, measured from the model itself."""
        q0 = np.zeros(5)
        jt, tcp = self._frames(q0)
        up = np.array([0.0, 0.0, 1.0])
        fwd = np.array([1.0, 0.0, 0.0])
        n = np.cross(fwd, up)                                    # +rotation about n: fwd -> up
        axes = [f[:3, :3] @ a for f, a in zip(jt, self._rev_axes)]
        self._d = [float(np.sign(np.dot(a, n))) for a in axes[1:4]]   # lift, elbow, wrist_flex signs
        self._d_pan = float(np.sign(axes[0][2]))

        def inplane(v: np.ndarray) -> tuple[float, float]:
            return float(v[0]), float(v[2])

        S, E, W, T = jt[1][:3, 3], jt[2][:3, 3], jt[3][:3, 3], tcp[:3, 3]
        self._S = inplane(S)
        self._lat = float(T[1])                                  # lateral offset of the TCP plane
        self._links = []
        for a, b in ((S, E), (E, W), (W, T)):
            dx, dz = inplane(b - a)
            self._links.append((math.hypot(dx, dz), math.atan2(dz, dx)))
        tz = tcp[:3, 2]
        self._beta0 = math.atan2(float(tz[2]), float(tz[0]))     # tool approach angle at q = 0
        # Yaw reference and roll sign. Rotations about parallel pitch axes compose, so every
        # pan = roll = 0 posture with the tool vertical has the same tool orientation; use
        # lift = elbow = 0 and only wrist_flex turning the tool down.
        q_ref = np.array([0.0, 0.0, 0.0, (-math.pi / 2 - self._beta0) / self._d[2], 0.0])
        t_ref = self.fk(q_ref)
        if t_ref[2, 2] > -0.999:
            raise ValueError("could not find the tool-down reference posture; unsupported model")
        self._yaw_ref = 0.0
        self._yaw_ref = math.atan2(t_ref[1, 0], t_ref[0, 0])
        jt_ref, _ = self._frames(q_ref)
        roll_axis = jt_ref[4][:3, :3] @ self._rev_axes[4]
        self._d_roll = float(np.sign(roll_axis[2])) or 1.0       # +roll turns the yaw by this sign

    def _seed_planar(self, r: float, h: float, elbow: int) -> Optional[np.ndarray]:
        """Analytic lift/elbow/wrist_flex for an in-plane TCP target (r forward, h up) with the
        tool vertical; pan = roll = 0. None if the 2-link triangle doesn't close."""
        (l1, a1), (l2, a2), (l3, a3) = self._links
        th3 = -math.pi / 2 + a3 - self._beta0                    # link-3 angle when the tool points down
        wx, wz = r - l3 * math.cos(th3) - self._S[0], h - l3 * math.sin(th3) - self._S[1]
        d2 = wx * wx + wz * wz
        c = (d2 - l1 * l1 - l2 * l2) / (2 * l1 * l2)
        if abs(c) > 1.1:
            return None
        c = max(-1.0, min(1.0, c))          # just outside: let the refinement decide
        g = elbow * math.acos(c)
        th1 = math.atan2(wz, wx) - math.atan2(l2 * math.sin(g), l1 + l2 * math.cos(g))
        th2 = th1 + g
        dl, de, dw = self._d
        ql = (th1 - a1) / dl
        qe = (th2 - a2 - dl * ql) / de
        qw = (th3 - a3 - dl * ql - de * qe) / dw
        return np.array([0.0, _wrap(ql), _wrap(qe), _wrap(qw), 0.0])

    # ---------------------------------------------------------------- IK
    def _target_rot(self, yaw_deg: float) -> np.ndarray:
        psi = math.radians(yaw_deg) + self._yaw_ref
        x = np.array([math.cos(psi), math.sin(psi), 0.0])
        z = np.array([0.0, 0.0, -1.0])
        return np.column_stack([x, np.cross(z, x), z])

    def _err(self, tcp: np.ndarray, p: np.ndarray, rd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = tcp[:3, :3]
        ep = p - tcp[:3, 3]
        er = 0.5 * sum(np.cross(r[:, i], rd[:, i]) for i in range(3))
        return ep, er

    def _refine(self, q: np.ndarray, p: np.ndarray, rd: np.ndarray, use_limits: bool,
                iters: int = 200) -> tuple[np.ndarray, float, float, int]:
        lo = np.array([self.limits[j][0] for j in ARM_JOINTS])
        hi = np.array([self.limits[j][1] for j in ARM_JOINTS])
        q = q.copy()
        if use_limits:
            q = np.clip(q, lo, hi)
        w = self.rot_weight
        lam = 1e-3
        ep = er = np.zeros(3)
        best, best_it = float("inf"), 0
        for it in range(iters):
            jac, tcp = self.jacobian(q)
            ep, er = self._err(tcp, p, rd)
            if np.linalg.norm(ep) < self.tol_m and np.linalg.norm(er) < self.tol_rad:
                return q, float(np.linalg.norm(ep)), math.degrees(float(np.linalg.norm(er))), it
            cost = float(np.linalg.norm(ep) + w * np.linalg.norm(er))
            if cost < 0.99 * best:
                best, best_it = cost, it
            elif it - best_it > 15:                  # stalled (on a limit or out of reach)
                break
            e = np.concatenate([ep, w * er])
            jw = jac.copy()
            jw[3:] *= w
            dq = jw.T @ np.linalg.solve(jw @ jw.T + (lam ** 2) * np.eye(6), e)
            step = float(np.max(np.abs(dq)))
            if step > 0.2:
                dq *= 0.2 / step
            q = q + dq
            if use_limits:
                q = np.clip(q, lo, hi)
        return q, float(np.linalg.norm(ep)), math.degrees(float(np.linalg.norm(er))), iters

    def _seeds(self, x: float, y: float, z: float, yaw_deg: float) -> list[tuple[str, np.ndarray]]:
        rho2 = x * x + y * y - self._lat ** 2
        if rho2 <= 0:
            return []
        r = math.sqrt(rho2)
        pan = (math.atan2(y, x) - math.atan2(self._lat, r)) * self._d_pan
        out = []
        for elbow, name in ((+1, "elbow+"), (-1, "elbow-")):
            s = self._seed_planar(r, z, elbow)
            if s is None:
                continue
            s[0] = _wrap(pan)
            # measure the yaw this seed actually gives and correct the roll with its sign
            got = math.radians(self.tool_yaw(self.fk(s)[:3, :3]))
            s[4] = _wrap(s[4] + self._d_roll * _wrap(math.radians(yaw_deg) - got))
            out.append((name, s))
        return out

    def _roll_variants(self, q: np.ndarray) -> list[np.ndarray]:
        lo, hi = self.limits["wrist_roll"]
        out = []
        for k in (0.0, -2 * math.pi, 2 * math.pi):
            v = q.copy()
            v[4] = q[4] + k
            if lo - 1e-9 <= v[4] <= hi + 1e-9:
                out.append(v)
        return out

    def ik(self, x: float, y: float, z: float, yaw_deg: float = 0.0,
           seed: Optional[Sequence[float]] = None) -> IKResult:
        """Top-down IK. Raises Unreachable(code, reason)."""
        for v, n in ((x, "x"), (y, "y"), (z, "z"), (yaw_deg, "yaw")):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise Unreachable("invalid", f"{n} must be a finite number, got {v!r}")
        if z < self.table_z:
            raise Unreachable("below_table", f"z={z:.4f} m is below the table (table_z={self.table_z:.4f} m)")
        p = np.array([x, y, z], dtype=float)
        rd = self._target_rot(yaw_deg)
        seeds = self._seeds(x, y, z, yaw_deg)
        if seed is not None:
            s = np.array(seed, dtype=float)
            seeds.insert(0, ("seed", s))
        cands: list[tuple[str, np.ndarray]] = []
        for name, s in seeds:
            for v in self._roll_variants(s) or [s]:
                cands.append((name, v))
        solutions: list[IKResult] = []
        blocked: list[tuple[np.ndarray, str]] = []       # in limits but a link under the table
        over: list[np.ndarray] = []                      # converged only outside the limits
        for name, s in cands:
            for use_limits in (True, False):
                q, pe, re, it = self._refine(s, p, rd, use_limits=use_limits)
                if pe >= self.accept_m or math.radians(re) >= self.accept_rad:
                    continue
                fits = [v for v in self._roll_variants(np.concatenate([q[:4], [_wrap(q[4])]]))
                        if self._within(v)]
                if not fits:
                    over.append(q)
                    continue
                low = self._lowest_point(fits[0])
                if low is None:
                    solutions.append(IKResult(fits[0], pe, re, it, name))
                    break
                blocked.append((fits[0], low))
        if solutions:
            ref = np.array(seed, dtype=float) if seed is not None else np.zeros(5)
            return min(solutions, key=lambda r: float(np.sum(np.abs(r.q - ref))))
        if blocked:
            raise Unreachable("below_table", f"every solution puts a link under the table ({blocked[0][1]})")
        if over:
            q = min(over, key=self._limit_excess)
            q = q.copy()
            q[4] = _wrap(q[4])
            bad = [f"{j} needs {math.degrees(v):.1f} deg, limit [{math.degrees(self.limits[j][0]):.1f}, "
                   f"{math.degrees(self.limits[j][1]):.1f}]"
                   for j, v in zip(ARM_JOINTS, q) if not self.limits[j][0] - 1e-6 <= v <= self.limits[j][1] + 1e-6]
            raise Unreachable("joint_limit", "; ".join(bad) or "no in-limit solution")
        reach = math.hypot(x, y)
        raise Unreachable("out_of_reach", f"({x:.3f}, {y:.3f}, {z:.3f}) m, {reach:.3f} m from the pan axis, "
                                          f"is outside the SO-101's top-down reach")

    def _within(self, q: Sequence[float]) -> bool:
        return all(self.limits[j][0] - 1e-6 <= v <= self.limits[j][1] + 1e-6 for j, v in zip(ARM_JOINTS, q))

    def _limit_excess(self, q: Sequence[float]) -> float:
        return sum(max(0.0, self.limits[j][0] - v, v - self.limits[j][1]) for j, v in zip(ARM_JOINTS, q))

    def _lowest_point(self, q: Sequence[float]) -> Optional[str]:
        """None if every joint centre and the TCP are at or above the table, else a description."""
        names = list(ARM_JOINTS) + ["tcp"]
        for n, pt in zip(names, self.points(q)):
            if pt[2] < self.table_z - 1e-6:
                return f"{n} at z={pt[2]:.4f} m"
        return None

    def check_clear(self, q: Sequence[float]) -> Optional[str]:
        return self._lowest_point(q)

    # ---------------------------------------------------------------- utilities
    def clip_limits(self, cal: dict[str, JointCal]) -> None:
        """Narrow the joint limits to what the servo tick bounds allow."""
        for j in ARM_JOINTS:
            a, b = cal[j].rad_limits()
            lo, hi = self.limits[j]
            nlo, nhi = max(lo, a), min(hi, b)
            if nlo >= nhi:
                raise ValueError(f"{j}: URDF limits and tick bounds do not overlap")
            self.limits[j] = (nlo, nhi)


def top_down_workspace(kin: SO101Kinematics, z_levels: Iterable[float], step: float = 0.01,
                       yaws: Sequence[float] = (0.0,), relative_yaw: bool = True) -> dict:
    """Sample the top-down reachable set and return the largest axis-aligned (x, y) box that is
    reachable at EVERY z in z_levels (and every yaw in `yaws`; relative_yaw: yaw measured from
    the pan direction, i.e. roll = yaw).

    Returns {"x": [lo, hi], "y": [lo, hi], "z": [zmin, zmax], "area_m2", "reach_r": [rmin, rmax]}.
    """
    zs = list(z_levels)
    xs = np.arange(-0.10, 0.50 + 1e-9, step)
    ys = np.arange(-0.45, 0.45 + 1e-9, step)
    ok = np.ones((len(xs), len(ys)), dtype=bool)
    rmin, rmax = float("inf"), 0.0
    for i, x in enumerate(xs):
        for k, y in enumerate(ys):
            for z in zs:
                if not ok[i, k]:
                    break
                for yw in yaws:
                    yaw = math.degrees(math.atan2(y, x)) + yw if relative_yaw else yw
                    try:
                        kin.ik(float(x), float(y), float(z), yaw)
                    except Unreachable:
                        ok[i, k] = False
                        break
            if ok[i, k]:
                r = math.hypot(x, y)
                rmin, rmax = min(rmin, r), max(rmax, r)
    # largest rectangle of True cells (histogram method over rows of x)
    best = (0, None)
    h = np.zeros(len(ys), dtype=int)
    for i in range(len(xs)):
        h = np.where(ok[i], h + 1, 0)
        stack: list[int] = []
        for k in range(len(ys) + 1):
            cur = h[k] if k < len(ys) else 0
            while stack and h[stack[-1]] >= cur:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                area = int(h[top]) * (k - left)
                if area > best[0]:
                    best = (area, (i - int(h[top]) + 1, i, left, k - 1))
            stack.append(k)
    if best[1] is None:
        return {"x": None, "y": None, "z": [min(zs), max(zs)], "area_m2": 0.0, "reach_r": None}
    i0, i1, k0, k1 = best[1]
    return {"x": [round(float(xs[i0]), 3), round(float(xs[i1]), 3)],
            "y": [round(float(ys[k0]), 3), round(float(ys[k1]), 3)],
            "z": [min(zs), max(zs)],
            "area_m2": round(float((xs[i1] - xs[i0]) * (ys[k1] - ys[k0])), 4),
            "reach_r": [round(rmin, 3), round(rmax, 3)]}
