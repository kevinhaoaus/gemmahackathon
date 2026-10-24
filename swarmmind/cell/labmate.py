"""LabMate brain (docs/LABMATE.md): inspect the bench, clear hazards, identify the experiment, fetch.

Rule: Gemma names things, code places things. Gemma labels the numbered marks of the `cell.blobs`
frame from the fixed catalog (kind "label" jobs), and turns the operator's reply into fields (kind
"answer") and a spoken summary (kind "report"). Ranking, use-case scoring, every position and every
arm task come from code: blob positions from `cell.blobs`, fixtures (safe_tray, handover) from
`cell.state`, arm work as `move_object` tasks on capability "arm.cell" in TABLE coordinates.

LabMate plans are ordinary replicated plans of the PlanManager (`Plan.kind == "lab"`, state in
`Plan.lab`), so they share plan.new / plan.update replication, sync, resume on failover and
`POST /cell/plans/{id}/cancel`. Plan states (leader-driven, every transition broadcast):

    inspect:  labelling (2 passes, frames >= pass_gap_s apart, keep what agrees)
                -> per movable hazard, most severe first:
                     confirming (announce, wait confirm_s for a cancel)
                     -> moving (move_object to the safe tray) -> verifying (re-label one fresh frame)
                -> concluding (use case in code) -> awaiting_answer (question; idle: nothing moves)
              POST /lab/answer -> answering (kind "answer") -> reporting (kind "report") -> done
    fetch:    locating (agreed labels older than label_max_age_s, or the blob vanished: labelling)
                -> moving (move_object to the handover tag) -> done
                   | failed "I can't see a <item>"

Resume: job ids and task ids are deterministic per (plan, round/step, attempt), so a new leader
waits for a replicated in-flight job (mid-sentence handoff) or move_object task instead of
submitting a second one. A confirming step restarts its full countdown on the new leader.
awaiting_answer survives in every replica.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..logutil import get_logger
from ..protocol import ANSWER_SYSTEM_PROMPT, LABEL_SYSTEM_PROMPT, PLAN_NEW, REPORT_SYSTEM_PROMPT
from . import planner as P
from .planner import Plan, PlanBusy, PlanError, extract_json

if TYPE_CHECKING:
    from ..agent import Node
    from .planner import PlanManager

log = get_logger("labmate")

KIND = "lab"
ARM_CAP = "arm.cell"
BLOBS_TOPIC = "cell.blobs"
SAFE_TRAY = "safe_tray"
HANDOVER = "handover"
# Prefix of the ground-truth line added to label jobs ONLY when cell.blobs carries sim_truth (the
# synthetic scene). Real cameras never publish sim_truth, so real Gemma never sees this line.
SIM_TRUTH_TAG = "[SIMULATOR ONLY: ground truth of the synthetic scene, absent on real hardware]"

ITEM_KINDS = ("tool", "part", "consumable", "equipment")
GRASPS = ("across", "center", "none")
PHASES = ("first_test", "final_data", "unknown")
UNKNOWN = "unknown"

LABEL_RETRIES = 2         # re-ask a label pass this many times after unusable output
ANSWER_RETRIES = 1
MOVE_RETRIES = 1          # retry a failed move_object once (with a fresh blob position)
BLOBS_WAIT_S = 6.0        # wait this long for (a new enough) cell.blobs before failing
FRESH_SETTLE_S = 0.3      # verify only trusts frames captured this long after the move ended
FRESH_TIMEOUT_S = 3.0     # ... and waits at most this long for one
REFIND_M = 0.04           # a blob within this distance of the expected point is the same object

DEFAULTS = {
    "confirm_s": 5.0,          # announce a hazard move, then wait this long for a cancel
    "pass_gap_s": 1.0,         # the two labelling passes use frames at least this far apart
    "label_max_age_s": 10.0,   # fetch re-labels when the agreed labels are older than this
    "tray_radius_m": 0.06,     # a blob this close to the safe_tray point is inside the tray
    "slot_m": 0.035,           # spacing of the per-item drop points in the tray / at the handover
    "min_use_case_score": 0.5,  # below this the experiment is "unknown" (general notes only)
}

_STATE_WORDS = {"open": "open", "opened": "open", "uncapped": "open", "unsealed": "open",
                "closed": "closed", "capped": "closed", "sealed": "closed", "shut": "closed"}
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_NAME_LEN = 40         # longest free-form item name Gemma may give (anything else is "unknown")
OTHER_HEIGHT_M = 0.03     # assumed grip height of an item that is not in the catalog
SELF_ITEMS = ("robot_arm", "robot", "gripper")   # the arm seeing itself: never picked up
LONG_ASPECT = 2.5        # a blob this much longer than wide is gripped "across" its width


class LabError(PlanError):
    """A LabMate request that cannot be served (HTTP status in .status) or unusable Gemma output."""

    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


class CatalogError(ValueError):
    pass


# ============================================================================ catalog
@dataclass(frozen=True)
class Catalog:
    items: dict[str, dict]        # name -> {kind, height, grasp}
    hazards: dict[str, dict]      # type -> {movable, severity, fix}
    use_cases: dict[str, dict]    # id -> {title, needs, optional}

    def fingerprint(self) -> str:
        """Short hash of the canonical catalog: every node must report the same one."""
        blob = json.dumps({"items": self.items, "hazards": self.hazards, "use_cases": self.use_cases},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    def summary(self) -> dict:
        return {"fingerprint": self.fingerprint(), "items": sorted(self.items),
                "hazards": {k: dict(v) for k, v in self.hazards.items()},
                "use_cases": {k: {"title": v["title"], "needs": v["needs"]} for k, v in self.use_cases.items()}}


def _num(v: Any, what: str, errs: list[str], lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        errs.append(f"{what} must be a number")
        return 0.0
    if not lo <= float(v) <= hi:
        errs.append(f"{what} = {v} is outside [{lo}, {hi}]")
    return float(v)


def load_catalog(cell: dict | None) -> Catalog:
    """Validate `capability_config.cell.catalog` (docs/LABMATE.md). Raises CatalogError listing
    every problem; the result is canonical (sorted lists, floats), so equal configs give equal
    fingerprints on every node."""
    cat = (cell or {}).get("catalog")
    if not isinstance(cat, dict):
        raise CatalogError("capability_config.cell.catalog is missing")
    errs: list[str] = []
    items: dict[str, dict] = {}
    raw_items = cat.get("items")
    if not isinstance(raw_items, dict) or not raw_items:
        errs.append("catalog.items must be a non-empty mapping")
        raw_items = {}
    for name, it in raw_items.items():
        n = str(name)
        if not _NAME_RE.match(n) or n == UNKNOWN:
            errs.append(f"item name {n!r} must be lower_snake_case and not 'unknown'")
        it = it if isinstance(it, dict) else {}
        kind = it.get("kind")
        if kind not in ITEM_KINDS:
            errs.append(f"item {n}: kind {kind!r} is not one of {', '.join(ITEM_KINDS)}")
        grasp = it.get("grasp")
        if grasp not in GRASPS:
            errs.append(f"item {n}: grasp {grasp!r} is not one of {', '.join(GRASPS)}")
        height = _num(it.get("height"), f"item {n}: height", errs, 0.0, 0.5)
        items[n] = {"kind": kind, "height": height, "grasp": grasp}
    hazards: dict[str, dict] = {}
    raw_h = cat.get("hazards")
    if not isinstance(raw_h, dict) or not raw_h:
        errs.append("catalog.hazards must be a non-empty mapping")
        raw_h = {}
    for name, hz in raw_h.items():
        n = str(name)
        if not _NAME_RE.match(n):
            errs.append(f"hazard type {n!r} must be lower_snake_case")
        hz = hz if isinstance(hz, dict) else {}
        if not isinstance(hz.get("movable"), bool):
            errs.append(f"hazard {n}: movable must be true or false")
        sev = hz.get("severity")
        if isinstance(sev, bool) or not isinstance(sev, int) or not 1 <= sev <= 5:
            errs.append(f"hazard {n}: severity must be an integer 1..5")
        fix = hz.get("fix")
        if not isinstance(fix, str) or not fix.strip():
            errs.append(f"hazard {n}: fix must be a non-empty text")
        hazards[n] = {"movable": bool(hz.get("movable")), "severity": int(sev) if isinstance(sev, int) else 0,
                      "fix": str(fix or "").strip()}
    use_cases: dict[str, dict] = {}
    raw_u = cat.get("use_cases")
    if not isinstance(raw_u, dict):
        errs.append("catalog.use_cases must be a mapping")
        raw_u = {}
    for uid, uc in raw_u.items():
        u = str(uid)
        if not _NAME_RE.match(u) or u == UNKNOWN:
            errs.append(f"use case id {u!r} must be lower_snake_case and not 'unknown'")
        uc = uc if isinstance(uc, dict) else {}
        title = uc.get("title")
        if not isinstance(title, str) or not title.strip():
            errs.append(f"use case {u}: title must be a non-empty text")
        lists = {}
        for key in ("needs", "optional"):
            v = uc.get(key, [] if key == "optional" else None)
            if not isinstance(v, list) or (key == "needs" and not v):
                errs.append(f"use case {u}: {key} must be a {'non-empty ' if key == 'needs' else ''}list of item names")
                v = []
            bad = [x for x in v if str(x) not in raw_items]
            if bad:
                errs.append(f"use case {u}: {key} names unknown item(s) {', '.join(map(str, bad))}")
            lists[key] = sorted(dict.fromkeys(str(x) for x in v))
        use_cases[u] = {"title": str(title or "").strip(), **lists}
    if errs:
        raise CatalogError("invalid LabMate catalog: " + "; ".join(errs))
    return Catalog(items=items, hazards=hazards, use_cases=use_cases)


def settings(cell: dict | None) -> dict:
    s = dict(DEFAULTS)
    for k, v in ((cell or {}).get("labmate") or {}).items():
        if k in s and isinstance(v, (int, float)) and not isinstance(v, bool):
            s[k] = float(v)
    return s


# ============================================================================ label parsing
def norm_name(v: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(v or "").strip().lower()).strip("_")


def resolve(v: Any, names: Any) -> str:
    """Normalised name, matched to `names` ignoring case, spaces/underscores and a plural s."""
    n = norm_name(v)
    if n in names:
        return n
    squash = {k.replace("_", ""): k for k in names}
    key = n.replace("_", "")
    return squash.get(key) or squash.get(key[:-1] if key.endswith("s") else key) or n


def item_spec(catalog: Catalog, name: str, blob: dict | None = None) -> dict:
    """Catalog entry for `name`. An item Gemma named that is not in the catalog gets a default:
    kind "other", a mid-size grip height, and a grasp taken from its blob (long and thin: across)."""
    spec = catalog.items.get(name)
    if spec is not None:
        return spec
    if name in SELF_ITEMS:
        return {"kind": "equipment", "height": 0.0, "grasp": "none"}
    L, W = float((blob or {}).get("length_m") or 0.0), float((blob or {}).get("width_m") or 0.0)
    grasp = "across" if W > 1e-6 and L / W >= LONG_ASPECT else "center"
    return {"kind": "other", "height": OTHER_HEIGHT_M, "grasp": grasp}


def _canon(name: str) -> str:
    """Name key that ignores word order and plurals, so tape_measure == measuring_tape-ish variants."""
    return "_".join(sorted(t[:-1] if len(t) > 3 and t.endswith("s") else t for t in name.split("_") if t))


def same_item(a: str, b: str) -> bool:
    return a == b or (UNKNOWN not in (a, b) and _canon(a) == _canon(b))


def _mark(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        m = re.fullmatch(r"\s*#?\s*(\d+)\s*", v)
        if m:
            return int(m.group(1))
    return None


def _coord_problem(entry: dict) -> str | None:
    keys = {str(k).lower() for k in entry}
    coords = sorted(keys & P._COORD_KEYS)
    if coords:
        return ", ".join(coords)
    for k, v in entry.items():
        if str(k).lower() not in ("mark", "marks", "id", "confidence") and P._has_number(v):
            return f"numeric value in {k!r}"
    return None


def parse_label(text: str, marks: set[int] | list[int], catalog: Catalog) -> dict:
    """Validate a label job's answer against this pass's marks and the catalog.

    -> {"items": [{"mark", "item", "state"}], "hazards": [{"type", "marks"}], "use_case", "warnings"}
    Names outside the catalog become "unknown" and marks that are not on the table are dropped
    (both listed in warnings). Coordinates, or anything that is not the JSON schema, raise LabError
    with a message meant to be shown back to Gemma."""
    marks = set(int(m) for m in marks)
    try:
        obj = extract_json(text)
    except PlanError as e:
        raise LabError(str(e)) from None
    if not isinstance(obj, dict) or not isinstance(obj.get("items"), list):
        raise LabError('expected {"items": [...], "hazards": [...], "use_case": "..."}')
    raw_h = obj.get("hazards", [])
    if not isinstance(raw_h, list):
        raise LabError('"hazards" must be a list')
    for key in obj:
        if str(key).lower() in P._COORD_KEYS:
            raise LabError(f"coordinates are not allowed ({key}); name marks only, the robot measures positions")
    warnings: list[str] = []
    items: dict[int, dict] = {}
    for i, e in enumerate(obj["items"], 1):
        if not isinstance(e, dict):
            raise LabError(f"items[{i}] must be an object")
        bad = _coord_problem(e)
        if bad:
            raise LabError(f"items[{i}]: coordinates/numbers are not allowed ({bad}); give mark, item and state only")
        m = _mark(e.get("mark", e.get("id")))
        if m is None:
            raise LabError(f"items[{i}] needs an integer \"mark\"")
        if m not in marks:
            warnings.append(f"mark #{m} is not on the table; ignored")
            continue
        name = resolve(e.get("item", e.get("name", e.get("label"))), catalog.items)
        if name not in catalog.items:
            name = re.sub(r"[^a-z0-9_]+", "", name)         # free-form names are kept (open vocabulary)
            if name != UNKNOWN and (not _NAME_RE.match(name) or len(name) > MAX_NAME_LEN):
                warnings.append(f"#{m}: {name or 'empty name'!r} is not a usable item name; using unknown")
                name = UNKNOWN
        state = _STATE_WORDS.get(norm_name(e.get("state")), "")
        if m in items:
            if items[m]["item"] != name:
                warnings.append(f"#{m} labelled twice ({items[m]['item']}, {name}); kept the first")
            continue
        items[m] = {"mark": m, "item": name, "state": state}
    hazards: dict[str, set[int]] = {}
    for i, e in enumerate(raw_h, 1):
        if not isinstance(e, dict):
            raise LabError(f"hazards[{i}] must be an object")
        bad = _coord_problem({k: v for k, v in e.items() if k not in ("marks", "mark")})
        if bad:
            raise LabError(f"hazards[{i}]: coordinates/numbers are not allowed ({bad})")
        typ = resolve(e.get("type", e.get("hazard")), catalog.hazards)
        if typ not in catalog.hazards:
            warnings.append(f"hazard type {typ or 'empty'!r} is not in the list; ignored")
            continue
        raw_m = e.get("marks", e.get("mark"))
        raw_m = raw_m if isinstance(raw_m, list) else [raw_m]
        ms = set()
        for v in raw_m:
            m = _mark(v)
            if m is None or m not in marks:
                warnings.append(f"hazard {typ}: mark {v!r} is not on the table; ignored")
            else:
                ms.add(m)
        if not ms:
            warnings.append(f"hazard {typ} names no mark on the table; ignored")
            continue
        hazards.setdefault(typ, set()).update(ms)
    uc = resolve(obj.get("use_case"), catalog.use_cases)
    if uc not in catalog.use_cases:
        if uc and uc != UNKNOWN:
            warnings.append(f"use case {uc!r} is not in the list; using unknown")
        uc = UNKNOWN
    return {"items": [items[m] for m in sorted(items)],
            "hazards": [{"type": t, "marks": sorted(ms)} for t, ms in sorted(hazards.items())],
            "use_case": uc, "warnings": warnings}


def _derived_hazards(items: list[dict], catalog: Catalog) -> dict[str, set[int]]:
    """Hazards that follow from labels: an item labelled "open" is an open_container."""
    out: dict[str, set[int]] = {}
    if "open_container" in catalog.hazards:
        for it in items:
            if it.get("state") == "open":
                out.setdefault("open_container", set()).add(int(it["mark"]))
    return out


def label_hazards(parsed: dict, catalog: Catalog) -> dict[str, set[int]]:
    hz = {h["type"]: set(h["marks"]) for h in parsed.get("hazards") or []}
    for t, ms in _derived_hazards(parsed.get("items") or [], catalog).items():
        hz.setdefault(t, set()).update(ms)
    return hz


def agree(p1: dict, p2: dict, catalog: Catalog) -> dict:
    """Two sightings: keep only (mark -> item) labels both passes gave (known items only), the
    hazard marks both passes reported per type, and the use case if both guessed the same.
    A state ("open") counts only if both passes saw it."""
    a = {it["mark"]: it for it in p1.get("items") or []}
    b = {it["mark"]: it for it in p2.get("items") or []}
    items = []
    for m in sorted(set(a) & set(b)):
        if same_item(a[m]["item"], b[m]["item"]) and a[m]["item"] != UNKNOWN:
            st = a[m]["state"] if a[m]["state"] == b[m]["state"] else ""
            name = next((x for x in (a[m]["item"], b[m]["item"]) if x in catalog.items), a[m]["item"])
            items.append({"mark": m, "item": name, "state": st})
    h1, h2 = label_hazards(p1, catalog), label_hazards(p2, catalog)
    hazards = {t: sorted(h1[t] & h2[t]) for t in sorted(set(h1) & set(h2)) if h1[t] & h2[t]}
    for t, ms in _derived_hazards(items, catalog).items():          # an agreed "open" state
        hazards[t] = sorted(set(hazards.get(t, [])) | ms)
    uc = p1.get("use_case") if p1.get("use_case") == p2.get("use_case") else UNKNOWN
    return {"items": items, "hazards": [{"type": t, "marks": ms} for t, ms in sorted(hazards.items())],
            "use_case": uc or UNKNOWN}


# ============================================================================ geometry / ranking
def compact_blobs(blobs: list[dict]) -> list[dict]:
    out = []
    for b in blobs or []:
        try:
            xy = [round(float(b["table_xy"][0]), 4), round(float(b["table_xy"][1]), 4)]
            out.append({"id": int(b["id"]), "table_xy": xy, "yaw_deg": round(float(b.get("yaw_deg") or 0.0), 1),
                        "length_m": round(float(b.get("length_m") or 0.0), 4),
                        "width_m": round(float(b.get("width_m") or 0.0), 4)})
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return sorted(out, key=lambda b: b["id"])


def table_bounds(cell: dict | None) -> tuple[float, float, float, float] | None:
    """(x0, x1, y0, y1) of the bench from the table tags (the tags sit at the bench corners)."""
    tt = (cell or {}).get("table_tags") or {}
    pts = [v for v in tt.values() if isinstance(v, (list, tuple)) and len(v) >= 2]
    if len(pts) < 2:
        return None
    xs, ys = [float(p[0]) for p in pts], [float(p[1]) for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def edge_distance(xy: list[float], bounds: tuple[float, float, float, float] | None) -> float:
    if bounds is None:
        return math.inf
    x0, x1, y0, y1 = bounds
    x, y = float(xy[0]), float(xy[1])
    return max(0.0, min(x - x0, x1 - x, y - y0, y1 - y))


def _dist(a: list[float], b: list[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def rank_hazards(hazards: list[dict], blobs: list[dict], catalog: Catalog,
                 bounds: tuple[float, float, float, float] | None) -> list[dict]:
    """Deterministic, safety first: severity (high first), then distance of the hazard's blob to
    the bench edge (near first), then type and mark. A movable hazard on several marks becomes one
    entry per mark (one move each); a report-only hazard stays one entry."""
    by_id = {b["id"]: b for b in blobs}
    out = []
    for h in hazards:
        spec = catalog.hazards[h["type"]]
        groups = [[m] for m in h["marks"]] if spec["movable"] else [list(h["marks"])]
        for ms in groups:
            d = min((edge_distance(by_id[m]["table_xy"], bounds) for m in ms if m in by_id), default=math.inf)
            out.append({"type": h["type"], "marks": ms, "severity": spec["severity"],
                        "movable": spec["movable"], "fix": spec["fix"], "status": "found",
                        "edge_m": None if math.isinf(d) else round(d, 3)})
    out.sort(key=lambda h: (-h["severity"], math.inf if h["edge_m"] is None else h["edge_m"],
                            h["type"], min(h["marks"])))
    return out


def score_use_cases(seen: list[str] | set[str], guess: str, catalog: Catalog,
                    min_score: float = DEFAULTS["min_use_case_score"]) -> dict:
    """Experiment from code: score = 0.6 * share of its needs seen on the bench
    + 0.2 * share of the bench's items it explains (needs + optional)
    + 0.2 if Gemma's agreed guess names it. Ties: more needs seen, then id. Below min_score the
    result is "unknown" (general notes only). missing = needs - seen."""
    seen = set(seen)
    cands = []
    for uid, uc in catalog.use_cases.items():
        needs, opt = uc["needs"], uc["optional"]
        have = [n for n in needs if n in seen]
        explained = len(seen & (set(needs) | set(opt))) / len(seen) if seen else 0.0
        score = 0.6 * len(have) / len(needs) + 0.2 * explained + (0.2 if guess == uid else 0.0)
        cands.append({"id": uid, "score": round(score, 3), "needs_seen": len(have)})
    cands.sort(key=lambda c: (-c["score"], -c["needs_seen"], c["id"]))
    if not cands or cands[0]["score"] < min_score:
        return {"id": UNKNOWN, "title": "an experiment I don't know", "confidence": 0.0, "missing": [],
                "candidates": cands[:3]}
    best = cands[0]
    uc = catalog.use_cases[best["id"]]
    return {"id": best["id"], "title": uc["title"], "confidence": round(min(1.0, best["score"]), 2),
            "missing": [n for n in uc["needs"] if n not in seen], "candidates": cands[:3]}


def parse_answer(text: str) -> dict:
    """The answer job's JSON -> {"experiment", "confirmed", "phase", "noise_ok"}; raises LabError."""
    try:
        obj = extract_json(text)
    except PlanError as e:
        raise LabError(str(e)) from None
    if not isinstance(obj, dict):
        raise LabError("expected one JSON object")
    exp = obj.get("experiment")
    if not isinstance(exp, str) or not exp.strip():
        raise LabError('"experiment" must be a non-empty text')

    def flag(v: Any, allow_none: bool) -> bool | None:
        if isinstance(v, bool):
            return v
        if isinstance(v, str) and v.strip().lower() in ("true", "yes", "false", "no"):
            return v.strip().lower() in ("true", "yes")
        if v is None or (isinstance(v, str) and v.strip().lower() in ("null", "none", "")):
            if allow_none:
                return None
        raise LabError(f"expected true/false{' or null' if allow_none else ''}, got {v!r}")

    if "confirmed" not in obj:
        raise LabError('missing "confirmed"')
    confirmed = flag(obj.get("confirmed"), False)
    phase = norm_name(obj.get("phase")) or UNKNOWN
    phase = {"first": "first_test", "test": "first_test", "final": "final_data"}.get(phase, phase)
    if phase not in PHASES:
        phase = UNKNOWN
    return {"experiment": exp.strip()[:80], "confirmed": bool(confirmed), "phase": phase,
            "noise_ok": flag(obj.get("noise_ok"), True)}


# ============================================================================ prompts (user turns)
def label_context(blobs: list[dict], catalog: Catalog, sim_truth: dict | None = None) -> str:
    lines = ["Marks on the bench (mark: size, aspect ratio = length / width):"]
    for b in blobs:
        L, W = float(b.get("length_m") or 0.0), float(b.get("width_m") or 0.0)
        aspect = L / W if W > 1e-6 else 1.0
        lines.append(f"- #{b['id']}: {L * 100:.1f} x {W * 100:.1f} cm, aspect {aspect:.1f}")
    if not blobs:
        lines.append("- (none: the bench looks empty)")
    lines.append("Catalog items (use these names exactly when one fits): " + ", ".join(catalog.items))
    lines.append("Anything else: name it yourself in 1 to 3 lowercase words joined by underscores "
                 "(for example scissors, tape_measure, beaker). unknown only if you cannot tell.")
    lines.append("Hazard types (use these names exactly): " + ", ".join(catalog.hazards))
    lines.append("Use cases (id: title; needs):")
    for uid, uc in catalog.use_cases.items():
        lines.append(f"- {uid}: {uc['title']}; needs {', '.join(uc['needs'])}")
    if sim_truth:
        lines.append(f"{SIM_TRUTH_TAG} sim_truth: {json.dumps(sim_truth, sort_keys=True)}")
    return "\n".join(lines)


def _name(item: str) -> str:
    return item.replace("_", " ")


def _article(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def items_phrase(items: list[dict]) -> str:
    if not items:
        return "nothing I recognise on the bench"
    counts = Counter((it["item"], it.get("state") == "open") for it in items)
    parts = []
    for (item, is_open), n in sorted(counts.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        word = ("open " if is_open else "") + _name(item)
        parts.append(_article(word) if n == 1 else f"{n} x {word}")
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def build_question(items: list[dict], use_case: dict) -> str:
    if use_case.get("id") == UNKNOWN:
        return (f"I see {items_phrase(items)}; I don't recognise this setup. "
                "What experiment is this, and what is this run for?")
    return f"I see {items_phrase(items)}; I think this is {use_case['title']}. Is that right, and what is this run for?"


def _hazard_line(h: dict, items_by_mark: dict[int, dict]) -> str:
    what = ", ".join(f"#{m} {_name(items_by_mark[m]['item'])}" if m in items_by_mark else f"#{m}"
                     for m in h["marks"])
    st = h.get("status")
    if st == "fixed":
        res = "FIXED: moved to the safe tray" + (f" ({h['verify']})" if h.get("verify") else "")
    elif st == "unverified":
        res = "moved to the safe tray, but the camera could not confirm it"
    elif st == "contained":
        res = "already inside the safe tray"
    elif st in ("not_fixed", "move_failed", "needs_human", "cancelled"):
        res = f"NOT fixed ({h.get('error') or st}); a human must {h['fix']}"
    else:
        res = f"report only; a human must {h['fix']}"
    return f"- {_name(h['type'])} on {what} (severity {h['severity']}): {res}"


def report_facts(lab: dict) -> str:
    items_by_mark = {it["mark"]: it for it in lab.get("items") or []}
    uc = lab.get("use_case") or {"id": UNKNOWN, "title": "an experiment I don't know", "missing": []}
    ans = lab.get("answer") or {}
    lines = ["Hazards found (most severe first):"]
    lines += [_hazard_line(h, items_by_mark) for h in lab.get("hazards") or []] or ["- none"]
    lines.append(f"Items on the bench: {items_phrase(lab.get('items') or [])}")
    if uc.get("id") == UNKNOWN:
        lines.append("Experiment: not one LabMate knows; give general bench notes only")
    else:
        lines.append(f"Experiment I proposed: {uc['title']} (confidence {uc.get('confidence', 0):.2f})")
    if ans:
        lines.append(f"Operator says: {ans.get('experiment')} (agrees with my proposal: "
                     f"{'yes' if ans.get('confirmed') else 'no'}); reply was: {ans.get('text', '')!r}")
        noise = {True: "yes", False: "no", None: "not said"}[ans.get("noise_ok")]
        lines.append(f"Phase: {ans.get('phase', UNKNOWN)}; noisy results acceptable: {noise}")
        if ans.get("phase") == "first_test" and ans.get("noise_ok") is True:
            lines.append("Design notes may be relaxed for this first test; safety items never change")
        else:
            lines.append("Keep every design note for this run; safety items never change")
    missing = uc.get("missing") or []
    lines.append("Missing tools: " + (", ".join(_name(m) for m in missing) if missing else "none"))
    return "\n".join(lines)


# ============================================================================ the manager
_STEP_STATE = {"announce": "confirming", "move": "moving", "verify": "verifying"}
STEP_STATES = tuple(_STEP_STATE.values())


class LabMate:
    """Leader-side LabMate flows on top of the node's PlanManager (shared replication/resume)."""

    def __init__(self, node: "Node", plans: "PlanManager"):
        self.node = node
        self.pm = plans
        self.catalog: Catalog | None = None
        self.catalog_error: str | None = None
        self.reload()
        plans.register_kind(KIND, self._drive, self._on_cancel)

    # ------------------------------------------------------------------ config / perception
    @property
    def cell(self) -> dict:
        return dict((self.node.cfg.capability_config or {}).get("cell") or {})

    def reload(self) -> None:
        try:
            self.catalog = load_catalog(self.cell)
            self.catalog_error = None
        except CatalogError as e:
            self.catalog, self.catalog_error = None, str(e)
            if (self.cell or {}).get("catalog") is not None:
                log.error("%s", e)

    @property
    def conf(self) -> dict:
        return settings(self.cell)

    def _cat(self) -> Catalog:
        if self.catalog is None:
            raise LabError(f"LabMate unavailable: {self.catalog_error}", status=503)
        return self.catalog

    def blobs_state(self) -> dict | None:
        """Newest cell.blobs (topic cache or what local perception set), by frame_ts."""
        cands = []
        sample = self.node.topics.latest.get(BLOBS_TOPIC)
        if isinstance(sample, dict) and isinstance(sample.get("data"), dict):
            cands.append(sample["data"])
        local = getattr(self.node, "cell_blobs", None)
        if isinstance(local, dict):
            cands.append(local)
        if not cands:
            return None
        return max(cands, key=lambda d: float(d.get("frame_ts") or 0.0))

    def _frame_ts(self) -> float:
        try:
            return float((self.blobs_state() or {}).get("frame_ts") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def _wait_blobs(self, term: int, min_ts: float = 0.0, timeout: float = BLOBS_WAIT_S) -> dict:
        end = time.monotonic() + timeout
        while True:
            st = self.blobs_state()
            if st and float(st.get("frame_ts") or 0.0) >= min_ts and st.get("calibrated", True):
                return st
            if time.monotonic() > end:
                if st is None:
                    raise LabError("no cell.blobs from the camera (is vision.cell running?)", 503)
                if not st.get("calibrated", True):
                    raise LabError("the bench camera is not calibrated (table tags not seen)", 503)
                raise LabError(f"no new camera frame for {timeout:.0f} s (camera stalled?)", 503)
            self.pm._guard(term)
            await asyncio.sleep(P.POLL_S)

    async def _fresh_blobs(self, term: int) -> dict | None:
        """cell.blobs captured after the arm finished (camera clock), or None: judging the bench
        from a frame taken before the move (e.g. the camera died with the old leader) would give a
        wrong verdict, so the caller reports the hazard as unverified instead."""
        st = self.blobs_state()
        if st is None:
            return None
        after = float(st.get("frame_ts") or 0.0) + FRESH_SETTLE_S
        end = time.monotonic() + FRESH_TIMEOUT_S
        while self._frame_ts() < after and time.monotonic() < end:
            self.pm._guard(term)
            await asyncio.sleep(P.POLL_S)
        if self._frame_ts() < after:
            log.warning("verify: no camera frame newer than the move after %.1f s", FRESH_TIMEOUT_S)
            return None
        return self.blobs_state()

    async def _marked_frame(self) -> bytes | None:
        fn = getattr(self.node, "cell_marked", None)
        if fn is not None:
            try:
                return await fn()
            except Exception as e:  # noqa: BLE001 - reported as "no marked frame" by the caller
                log.warning("marked frame unavailable: %s", e)
                return None
        return getattr(self.node, "cell_marked_jpeg", None)

    def _objects(self) -> dict[str, dict]:
        return self.pm._state_objects()

    def _fixture(self, name: str) -> tuple[list[float], float, float] | None:
        """(table_xy, yaw_deg, z) of a tagged fixture from the latest cell.state, if visible."""
        o = self._objects().get(name)
        if not o or not o.get("visible") or not isinstance(o.get("table"), (list, tuple)) or len(o["table"]) < 2:
            return None
        t = [float(v) for v in o["table"]]
        return t[:2], float(o.get("yaw_deg") or 0.0), (t[2] if len(t) > 2 else 0.0)

    def _in_tray(self, xy: list[float]) -> bool:
        tray = self._fixture(SAFE_TRAY)
        return tray is not None and _dist(xy, tray[0]) <= self.conf["tray_radius_m"]

    # ------------------------------------------------------------------ queries
    def lab_plans(self) -> list[Plan]:
        return [self.pm.plans[p] for p in self.pm.order if self.pm.plans[p].kind == KIND]

    def latest_labels(self, exclude: str | None = None) -> tuple[list[dict], float, str | None]:
        """(items, labels_ts, plan_id) of the newest agreed labels in any replicated LabMate plan."""
        best: tuple[list[dict], float, str | None] = ([], 0.0, None)
        for p in self.lab_plans():
            ts = float(p.lab.get("labels_ts") or 0.0)
            if p.plan_id != exclude and ts > best[1]:
                best = ([dict(i) for i in p.lab.get("items") or []], ts, p.plan_id)
        return best

    def awaiting(self) -> Plan | None:
        return next((p for p in reversed(self.lab_plans()) if p.state == "awaiting_answer"), None)

    def state(self) -> dict:
        plans = self.lab_plans()
        items, ts, src = self.latest_labels()
        blobs = self.blobs_state()
        aw = self.awaiting()
        return {"self": self.node.cfg.node_id, "leader_id": self.node.election.leader_id,
                "catalog": self.catalog.summary() if self.catalog else None,
                "catalog_error": self.catalog_error, "settings": self.conf,
                "plan": plans[-1].wire() if plans else None,
                "plans": [p.wire() for p in plans[-5:]],
                "awaiting_answer": aw.plan_id if aw else None,
                "question": aw.lab.get("question") if aw else None,
                "labels": {"items": items, "labels_ts": ts or None, "from_plan": src,
                           "age_s": round(time.time() - ts, 1) if ts else None},
                "blobs": None if blobs is None else {
                    "frame_ts": blobs.get("frame_ts"), "calibrated": blobs.get("calibrated"),
                    "background": blobs.get("background"), "count": len(blobs.get("blobs") or []),
                    "simulated": bool(blobs.get("sim_truth"))}}

    # ------------------------------------------------------------------ leader API
    def _require_leader(self) -> None:
        if not self.node.election.is_leader:
            raise LabError("not the leader", 503)

    def _new_plan(self, goal: str, state: str, lab: dict) -> Plan:
        n = self.node
        cat = self._cat()
        plan = Plan(plan_id=uuid.uuid4().hex[:8], goal=goal, state=state, kind=KIND, created_ts=time.time(),
                    lab={"mode": "", "items": [], "hazards": [], "use_case": None, "question": None,
                         "answer": None, "report": None, "catalog": cat.fingerprint(), **lab})
        self.pm._store(plan)
        self.pm._update(plan, n.election.term, f"LabMate: {goal}", type_=PLAN_NEW)
        self.pm._start(plan)
        return plan

    def _supersede_waiting(self, term: int, by: str) -> None:
        for p in self.lab_plans():
            if p.state == "awaiting_answer" and p.lab.get("mode") == "inspect":
                p.state, p.error = "cancelled", f"superseded by {by}"
                self.pm._update(p, term, f"superseded by {by} (question left unanswered)")

    async def inspect(self) -> Plan:
        self._require_leader()
        self._cat()
        act = self.pm.active()
        if act:
            raise PlanBusy(act.plan_id)
        self._supersede_waiting(self.node.election.term, "a new inspection")
        return self._new_plan("inspect the bench", "labelling",
                              {"mode": "inspect", "label": _new_round(0, "inspect")})

    async def fetch(self, item: str) -> Plan:
        self._require_leader()
        cat = self._cat()
        items, ts, src = self.latest_labels()
        name = resolve(item, set(cat.items) | {it["item"] for it in items})
        if not name:
            raise LabError("item is required")
        if not _NAME_RE.match(name) or len(name) > MAX_NAME_LEN:
            raise LabError(f"{item!r} is not a usable item name")
        if item_spec(cat, name)["grasp"] == "none":
            raise LabError(f"I never pick up a {_name(name)} (grasp: none)")
        act = self.pm.active()
        if act:
            raise PlanBusy(act.plan_id)
        return self._new_plan(f"fetch {name}", "locating",
                              {"mode": "fetch", "fetch": name, "items": items, "labels_ts": ts or None,
                               "labels_from": src, "relabelled": False})

    async def answer(self, text: str) -> Plan:
        self._require_leader()
        text = str(text or "").strip()
        if not text:
            raise LabError("text is required")
        plan = self.awaiting()
        if plan is None:
            raise LabError("no LabMate question is waiting for an answer", 409)
        plan.lab["answer_text"] = text[:500]
        plan.lab["answer_attempt"] = 0
        plan.state = "answering"
        self.pm._update(plan, self.node.election.term, f"LabMate: operator answered: {text[:120]!r}")
        self.pm._start(plan)
        return plan

    def _on_cancel(self, plan: Plan) -> None:
        for h in plan.lab.get("hazards") or []:
            if h.get("status") in ("found", "confirming", "moving"):
                h["status"] = "cancelled"
                h["error"] = "cancelled by operator"

    # ------------------------------------------------------------------ driver
    async def _drive(self, plan: Plan, term: int) -> None:
        st = plan.state
        try:
            if st == "labelling":
                await self._labelling(plan, term)
            elif st == "locating":
                self._locating(plan, term)
            elif st in STEP_STATES or st == "running":
                await self._run_step(plan, term)
            elif st == "concluding":
                self._conclude(plan, term)
            elif st == "answering":
                await self._answering(plan, term)
            elif st == "reporting":
                await self._reporting(plan, term)
            else:
                self.pm._fail(plan, term, f"unknown LabMate state {st!r}")
        except LabError as e:
            if not plan.finished:
                self.pm._fail(plan, term, str(e))

    def _set_state(self, plan: Plan, term: int, state: str, msg: str | None = None, level: str = "info") -> None:
        plan.state = state
        self.pm._update(plan, term, f"LabMate: {msg}" if msg else None, level=level)

    # ---- labelling (two passes, or one verify pass)
    async def _label_job(self, plan: Plan, term: int, rec: dict, jid: str, min_ts: float,
                         errors: list[str], fresh: bool = False) -> None:
        """Start (or re-find) one label job; records frame_ts / blobs / job_id in `rec`."""
        n = self.node
        cat = self._cat()
        if jid not in n.jobs.jobs:
            st = await self._fresh_blobs(term) if fresh else await self._wait_blobs(term, min_ts)
            if st is None:
                raise LabError("no camera frame since the move" if fresh else "no cell.blobs from the camera", 503)
            frame = await self._marked_frame()
            if not frame:
                raise LabError("no marked camera frame (GET /cell/marked.jpg failed)", 503)
            blobs = compact_blobs(st.get("blobs") or [])
            truth = st.get("sim_truth") if isinstance(st.get("sim_truth"), dict) else None
            q = "Label every numbered mark in the image."
            if errors:
                q += (f"\n\nYour previous answer was rejected: {errors[-1]}. "
                      "Reply again with ONLY the JSON object and follow every rule.")
            self.pm._guard(term)
            rec.update(frame_ts=float(st.get("frame_ts") or 0.0), blobs=blobs)
            await n.jobs.create(q, kind="label", system=LABEL_SYSTEM_PROMPT, image=frame,
                                context=label_context(blobs, cat, truth), job_id=jid)
        elif not rec.get("blobs"):
            # the job was created (and replicated) but the leader died before the plan update
            # carrying its blob list: the stable track ids of the newest frame stand in for it
            st = self.blobs_state() or {}
            rec.update(frame_ts=float(st.get("frame_ts") or 0.0), blobs=compact_blobs(st.get("blobs") or []))
        rec.update(job_id=jid, state="running")
        self.pm._update(plan, term)

    async def _labelling(self, plan: Plan, term: int) -> None:
        L = plan.lab["label"]
        passes: list[dict] = L["passes"]
        if len(passes) < 2 or passes[-1]["state"] != "done":
            if not passes or passes[-1]["state"] == "done":
                passes.append({"k": len(passes), "attempt": 0, "state": "pending", "job_id": None,
                               "frame_ts": None, "blobs": [], "parsed": None, "errors": []})
            rec = passes[-1]
            k = rec["k"]
            jid = rec.get("job_id") or f"{plan.plan_id}-L{L['round']}p{k}a{rec['attempt']}"
            if rec.get("job_id") is None or rec["job_id"] not in self.node.jobs.jobs:
                min_ts = float(passes[0]["frame_ts"] or 0.0) + self.conf["pass_gap_s"] if k == 1 else 0.0
                if rec.get("job_id") is None:
                    self.node.event(f"plan {plan.plan_id}: LabMate labelling pass {k + 1}/2 (job {jid})")
                await self._label_job(plan, term, rec, jid, min_ts, rec["errors"])
            try:
                text = await self.pm._wait_job(jid, term)
                rec["parsed"] = parse_label(text, [b["id"] for b in rec["blobs"]], self._cat())
            except (P._JobFailed, LabError) as e:
                rec["errors"].append(str(e))
                if rec["attempt"] >= LABEL_RETRIES:
                    raise LabError(f"labelling failed after {rec['attempt'] + 1} attempts: {e}") from None
                rec.update(attempt=rec["attempt"] + 1, job_id=None, state="pending")
                self.pm._update(plan, term, f"LabMate: unusable labels ({e}); asking again", level="warning")
                return
            rec["state"] = "done"
            self.pm._update(plan, term)
            return
        agreed = agree(passes[0]["parsed"], passes[1]["parsed"], self._cat())
        self._on_labels(plan, term, agreed, passes[1]["blobs"])

    def _on_labels(self, plan: Plan, term: int, agreed: dict, blobs: list[dict]) -> None:
        lab = plan.lab
        by_id = {b["id"]: b for b in blobs}
        lab["items"] = [{**it, "table_xy": by_id[it["mark"]]["table_xy"], "yaw_deg": by_id[it["mark"]]["yaw_deg"]}
                        for it in agreed["items"] if it["mark"] in by_id]
        lab["labels_ts"] = time.time()
        lab["use_case_guess"] = agreed["use_case"]
        n_items = len(lab["items"])
        if lab["mode"] == "fetch":
            lab["relabelled"] = True
            self._set_state(plan, term, "locating", f"labels agreed for {n_items} item(s)")
            return
        cat = self._cat()
        hazards = rank_hazards(agreed["hazards"], blobs, cat, table_bounds(self.cell))
        items_by_mark = {it["mark"]: it for it in lab["items"]}
        steps: list[dict] = []
        moved: dict[int, int] = {}                      # mark -> hazard index that moves it
        for hi, h in enumerate(hazards):
            if not h["movable"]:
                h["status"] = "report_only"
                continue
            m = h["marks"][0]
            it = items_by_mark.get(m)
            if m in moved:
                h["status"], h["via"] = "found", moved[m]   # the same object is already being moved
                continue
            if it is None or item_spec(cat, it["item"])["grasp"] == "none":
                h["status"] = "needs_human"
                h["error"] = "I can't pick it up" if it else "I don't know what it is"
                continue
            if self._in_tray(it["table_xy"]):
                h["status"] = "contained"
                continue
            moved[m] = hi
            for op in ("announce", "move", "verify"):
                steps.append(P._new_step({"op": op, "hazard": hi, "mark": m, "item": it["item"],
                                          "target": SAFE_TRAY}))
        lab["hazards"] = hazards
        plan.steps, plan.step = steps, 0
        summary = (", ".join(f"{h['type']} #{'/#'.join(map(str, h['marks']))}" for h in hazards) or "no hazards")
        self._set_state(plan, term, _STEP_STATE[steps[0]["op"]] if steps else "concluding",
                        f"labels agreed: {items_phrase(lab['items'])}; {summary}")

    # ---- fetch
    def _locating(self, plan: Plan, term: int) -> None:
        lab = plan.lab
        name = lab["fetch"]
        age = time.time() - float(lab.get("labels_ts") or 0.0)
        blobs = {b["id"]: b for b in compact_blobs((self.blobs_state() or {}).get("blobs") or [])}
        cands = [it for it in lab.get("items") or [] if same_item(it["item"], name)]
        visible = [it for it in cands if it["mark"] in blobs]
        stale = not lab.get("labels_ts") or age > self.conf["label_max_age_s"] or (cands and not visible)
        if stale and not lab.get("relabelled"):
            lab["label"] = _new_round(0, "fetch")
            why = "no labels yet" if not lab.get("labels_ts") else (
                f"labels are {age:.0f} s old" if age > self.conf["label_max_age_s"] else "the labelled blob moved")
            self._set_state(plan, term, "labelling", f"re-labelling before the fetch ({why})")
            return
        if not visible:
            raise LabError(f"I can't see a {_name(name)}")
        # prefer one that is not in the safe tray (it was put there for a reason)
        visible.sort(key=lambda it: (self._in_tray(blobs[it["mark"]]["table_xy"]), it["mark"]))
        it = visible[0]
        plan.steps = [P._new_step({"op": "move", "mark": it["mark"], "item": name, "target": HANDOVER})]
        plan.step = 0
        self._set_state(plan, term, "moving", f"fetching the {_name(name)} (#{it['mark']}) to the handover")

    # ---- steps
    async def _run_step(self, plan: Plan, term: int) -> None:
        i = plan.step
        if i >= len(plan.steps):
            if plan.lab.get("mode") == "fetch":
                plan.lab["report"] = f"Here is the {_name(plan.lab['fetch'])}."
                self._set_state(plan, term, "done", f"DONE: the {_name(plan.lab['fetch'])} is at the handover")
            else:
                self._set_state(plan, term, "concluding")
            return
        s = plan.steps[i]
        if s.get("state") in ("done", "skipped"):
            plan.step = i + 1
            nxt = plan.steps[i + 1]["op"] if i + 1 < len(plan.steps) else None
            if nxt:
                plan.state = _STEP_STATE[nxt]
            self.pm._update(plan, term)
            return
        want = _STEP_STATE[s["op"]]
        if plan.state != want:
            plan.state = want
        if s["op"] == "announce":
            await self._announce(plan, i, term)
        elif s["op"] == "move":
            await self._move(plan, i, term)
        else:
            await self._verify(plan, i, term)

    def _hazard(self, plan: Plan, s: dict) -> dict | None:
        hi = s.get("hazard")
        hz = plan.lab.get("hazards") or []
        return hz[hi] if isinstance(hi, int) and 0 <= hi < len(hz) else None

    def _set_hazard(self, plan: Plan, s: dict, status: str, **kw: Any) -> None:
        h = self._hazard(plan, s)
        if h is None:
            return
        h["status"] = status
        h.update(kw)
        for other in plan.lab.get("hazards") or []:          # hazards fixed by the same move
            if other is not h and other.get("via") == s.get("hazard"):
                other["status"] = status
                other.update(kw)

    async def _announce(self, plan: Plan, i: int, term: int) -> None:
        s = plan.steps[i]
        h = self._hazard(plan, s) or {}
        wait = self.conf["confirm_s"]
        it = next((x for x in plan.lab.get("items") or [] if x["mark"] == s["mark"]), {})
        what = ("open " if it.get("state") == "open" else "") + _name(s["item"])
        text = (f"{_name(h.get('type', 'hazard'))}: the {what} (#{s['mark']}). "
                f"Moving it to the safe tray in {wait:g} s unless you cancel.")
        s.update(state="running", result={"announce": text, "confirm_s": wait,
                                          "until": round(time.time() + wait, 2)})
        plan.lab["announce"] = text
        self._set_hazard(plan, s, "confirming")
        self.pm._update(plan, term, f"LabMate: {text}", level="warning")
        end = time.monotonic() + wait                  # a new leader restarts the full countdown
        while time.monotonic() < end:
            self.pm._guard(term)
            await asyncio.sleep(min(P.POLL_S, max(0.0, end - time.monotonic())))
        s["state"] = "done"
        plan.step = i + 1
        plan.state = _STEP_STATE[plan.steps[i + 1]["op"]] if i + 1 < len(plan.steps) else plan.state
        self.pm._update(plan, term)

    def _drop_point(self, target: str, exclude_mark: int) -> tuple[list[float], float, float]:
        """A free drop point at the fixture: the fixture point plus a small per-item offset, the
        first of a fixed pattern that is at least slot_m from every blob already there."""
        fx = self._fixture(target)
        if fx is None:
            raise LabError(f"the {_name(target)} tag is not visible")
        (fx_xy, fyaw, fz) = fx
        d = self.conf["slot_m"]
        r = self.conf["tray_radius_m"]
        occupied = [b["table_xy"] for b in compact_blobs((self.blobs_state() or {}).get("blobs") or [])
                    if b["id"] != exclude_mark and _dist(b["table_xy"], fx_xy) <= r + d]
        c, s_ = math.cos(math.radians(fyaw)), math.sin(math.radians(fyaw))
        pattern = [(0, 0), (d, 0), (-d, 0), (0, d), (0, -d), (d, d), (-d, d), (d, -d), (-d, -d)]
        pts = [[fx_xy[0] + c * ox - s_ * oy, fx_xy[1] + s_ * ox + c * oy] for ox, oy in pattern]
        best = next((p for p in pts if all(_dist(p, o) >= d for o in occupied)), None)
        if best is None:                          # full: the point farthest from everything there
            best = max(pts, key=lambda p: min(_dist(p, o) for o in occupied))
        return [round(best[0], 4), round(best[1], 4)], fyaw, fz

    def compile_move(self, s: dict, plan: Plan) -> dict:
        """move_object args in TABLE coordinates (docs/LABMATE.md). The arm re-finds the blob in
        the latest cell.blobs (<= 4 cm) and converts with its own base calibration."""
        cat = self._cat()
        st = self.blobs_state() or {}
        blobs = {b["id"]: b for b in compact_blobs(st.get("blobs") or [])}
        b = blobs.get(s["mark"])
        if b is None:
            known = next((x for x in plan.lab.get("items") or [] if x["mark"] == s["mark"]), None)
            near = [x for x in blobs.values() if known and _dist(x["table_xy"], known["table_xy"]) <= REFIND_M]
            if not near:
                raise LabError(f"the {_name(s['item'])} (#{s['mark']}) is no longer visible")
            b = min(near, key=lambda x: _dist(x["table_xy"], known["table_xy"]))
        to_xy, _, to_z = self._drop_point(s["target"], b["id"])
        spec = item_spec(cat, s["item"], b)
        return {"action": "move_object", "item": s["item"], "blob_id": int(b["id"]),
                "table_xy": list(b["table_xy"]), "yaw_deg": float(b["yaw_deg"]),
                "height": float(spec["height"]), "grasp": spec["grasp"],
                # to.z = grip height at the drop point in the table frame: fixture surface + item height
                "to": {"table_xy": to_xy, "yaw_deg": float(b["yaw_deg"]),
                       "z": round(float(to_z) + float(spec["height"]), 4)},
                "plan_id": plan.plan_id, "purpose": s["target"]}

    async def _move(self, plan: Plan, i: int, term: int) -> None:
        n = self.node
        s = plan.steps[i]
        tid = s.get("task_id") or f"{plan.plan_id}-s{i}-a{s.get('attempt', 0)}"
        task = n.tasks.tasks.get(tid)
        if task is None and (s.get("task_id") or s.get("state") == "running"):
            task = await self.pm._lookup_task(tid, term)       # replica may still be syncing
        if task is None:
            try:
                if not await self._arm_alive(term):
                    raise LabError(f"no {ARM_CAP} is connected")
                args = self.compile_move(s, plan)
            except LabError as e:
                self._move_failed(plan, i, term, str(e), retry=False)
                return
            self.pm._guard(term)
            task = n.tasks.submit(ARM_CAP, args, tid)
            s.update(task_id=tid, state="running", arm=ARM_CAP, args=args, error=None)
            self._set_hazard(plan, s, "moving")
            self.pm._update(plan, term, f"LabMate: move_object {s['item']} #{s['mark']} -> {s['target']} "
                                        f"(task {tid}, assignee {task.assignee or 'none yet'})")
        elif s.get("state") != "running" or s.get("task_id") != tid:
            s.update(task_id=tid, state="running")
            self.pm._update(plan, term, f"LabMate: waiting for move task {tid}")
        else:
            log.info("plan %s: waiting for existing move task %s (%s on %s)", plan.plan_id, tid,
                     task.state, task.assignee)
        final = await self.pm._wait_task(tid, term)
        if plan.finished:
            return
        if final is not None and final.state == "done":
            s.update(state="done", result=final.result, error=None, done_frame_ts=self._frame_ts())
            self._set_hazard(plan, s, "moved")
            plan.step = i + 1
            if i + 1 < len(plan.steps):
                plan.state = _STEP_STATE[plan.steps[i + 1]["op"]]
            self.pm._update(plan, term, f"LabMate: {s['item']} moved to the {_name(s['target'])} "
                                        f"(on {final.assignee})")
        else:
            why = "task vanished" if final is None else (final.error or final.state)
            self._move_failed(plan, i, term, f"task {tid} {getattr(final, 'state', 'missing')}: {why}")

    async def _arm_alive(self, term: int) -> bool:
        end = time.monotonic() + P.ARM_WAIT_S
        while not self.pm._cap_alive(ARM_CAP):
            if time.monotonic() > end:
                return False
            self.pm._guard(term)
            await asyncio.sleep(0.2)
        return True

    def _move_failed(self, plan: Plan, i: int, term: int, err: str, retry: bool = True) -> None:
        s = plan.steps[i]
        if retry and s.get("retries", 0) < MOVE_RETRIES:
            s.update(retries=s.get("retries", 0) + 1, attempt=s.get("attempt", 0) + 1, task_id=None,
                     state="pending", error=err)
            self.pm._update(plan, term, f"LabMate: move failed ({err}); retrying", level="warning")
            return
        s.update(state="failed", error=err)
        if plan.lab.get("mode") == "fetch":
            self.pm._fail(plan, term, f"could not fetch the {_name(s['item'])}: {err}")
            return
        self._set_hazard(plan, s, "move_failed", error=err)
        for later in plan.steps[i + 1:]:                       # its verify step has nothing to check
            if later.get("hazard") == s.get("hazard") and later.get("state") == "pending":
                later["state"] = "skipped"
        plan.step = i + 1
        self.pm._update(plan, term, f"LabMate: could not move the {_name(s['item'])}: {err}", level="error")

    async def _verify(self, plan: Plan, i: int, term: int) -> None:
        s = plan.steps[i]
        h = self._hazard(plan, s)
        jid = s.get("job_id") or f"{plan.plan_id}-v{i}a{s.get('attempt', 0)}"
        verdict: tuple[str, str] | None = None
        rec = s.setdefault("check", {"blobs": [], "frame_ts": None})
        if s.get("job_id") is None or jid not in self.node.jobs.jobs:
            try:
                await self._label_job(plan, term, rec, jid, 0.0, [s["error"]] if s.get("error") else [], fresh=True)
            except LabError as e:                  # camera gone (e.g. it died with the old leader)
                verdict = ("unverified", str(e))
            else:
                s.update(job_id=jid, state="running")
        if verdict is None:
            parsed = None
            try:
                text = await self.pm._wait_job(jid, term)
                parsed = parse_label(text, [b["id"] for b in rec["blobs"]], self._cat())
            except (P._JobFailed, LabError) as e:
                if s.get("retries", 0) < 1:
                    s.update(retries=s.get("retries", 0) + 1, attempt=s.get("attempt", 0) + 1,
                             job_id=None, error=str(e))
                    rec.update(blobs=[], frame_ts=None)
                    self.pm._update(plan, term, f"LabMate: verify labels unusable ({e}); asking again",
                                    level="warning")
                    return
            verdict = self._resolved(plan, s, h or {}, rec["blobs"], parsed)
        status, why = verdict
        self._refresh_positions(plan, rec.get("blobs") or [])
        self._set_hazard(plan, s, status, verify=why)
        s.update(state="done", result={"status": status, "reason": why})
        plan.step = i + 1
        if i + 1 < len(plan.steps):
            plan.state = _STEP_STATE[plan.steps[i + 1]["op"]]
        self.pm._update(plan, term, f"LabMate: {s['item']} #{s['mark']}: {status} ({why})",
                        level="info" if status == "fixed" else "warning")

    def _resolved(self, plan: Plan, s: dict, h: dict, blobs: list[dict], parsed: dict | None) -> tuple[str, str]:
        """Fixed if the hazard's blob now lies within the tray area, or the hazard label no longer
        appears on any blob outside the tray."""
        tray = self._fixture(SAFE_TRAY)
        r = self.conf["tray_radius_m"]
        by_id = {b["id"]: b for b in blobs}
        in_tray = lambda xy: tray is not None and _dist(xy, tray[0]) <= r  # noqa: E731
        b = by_id.get(s["mark"])
        if b is not None and in_tray(b["table_xy"]):
            return "fixed", f"#{s['mark']} is now in the safe tray"
        i_move = self._move_index(plan, s)
        dest = None
        if i_move is not None:
            dest = ((plan.steps[i_move].get("args") or {}).get("to") or {}).get("table_xy")
        if b is None and dest and any(_dist(x["table_xy"], dest) <= REFIND_M for x in blobs):
            return "fixed", "an object now sits at the tray drop point"
        if parsed is not None:
            typ = h.get("type")
            marks = label_hazards(parsed, self._cat()).get(typ, set())
            outside = sorted(m for m in marks if m in by_id and not in_tray(by_id[m]["table_xy"]))
            if not outside:
                return "fixed", f"no {_name(typ or 'hazard')} outside the tray any more"
            return "not_fixed", f"{_name(typ or 'hazard')} still seen on #{', #'.join(map(str, outside))}"
        if not blobs:
            return "unverified", "no camera frame to check"
        return "not_fixed", f"#{s['mark']} is not in the safe tray"

    @staticmethod
    def _move_index(plan: Plan, s: dict) -> int | None:
        return next((j for j, x in enumerate(plan.steps)
                     if x["op"] == "move" and x.get("hazard") == s.get("hazard")), None)

    @staticmethod
    def _refresh_positions(plan: Plan, blobs: list[dict]) -> None:
        by_id = {b["id"]: b for b in blobs}
        for it in plan.lab.get("items") or []:
            b = by_id.get(it["mark"])
            if b is not None:
                it["table_xy"], it["yaw_deg"] = b["table_xy"], b["yaw_deg"]
        if blobs:
            plan.lab["labels_ts"] = time.time()

    # ---- use case, question, answer, report
    def _conclude(self, plan: Plan, term: int) -> None:
        lab = plan.lab
        seen = sorted({it["item"] for it in lab.get("items") or []})
        uc = score_use_cases(seen, lab.get("use_case_guess") or UNKNOWN, self._cat(),
                             self.conf["min_use_case_score"])
        lab["use_case"] = uc
        lab["question"] = build_question(lab.get("items") or [], uc)
        self._set_state(plan, term, "awaiting_answer", f"asks: {lab['question']}")

    async def _answering(self, plan: Plan, term: int) -> None:
        lab = plan.lab
        att = int(lab.get("answer_attempt") or 0)
        jid = f"{plan.plan_id}-ans-a{att}"
        uc = lab.get("use_case") or {}
        if jid not in self.node.jobs.jobs:
            ctx = (f"Proposed experiment: {uc.get('title')} ({uc.get('id')})\n"
                   f"LabMate asked: {lab.get('question')}")
            q = f"Operator's reply: {lab.get('answer_text', '')}"
            if lab.get("answer_error"):
                q += (f"\n\nYour previous answer was rejected: {lab['answer_error']}. "
                      "Reply again with ONLY the JSON object.")
            self.pm._guard(term)
            await self.node.jobs.create(q, kind="answer", system=ANSWER_SYSTEM_PROMPT, context=ctx, job_id=jid)
        if plan.job_id != jid:
            plan.job_id = jid
            self.pm._update(plan, term)
        try:
            ans = parse_answer(await self.pm._wait_job(jid, term))
        except (P._JobFailed, LabError) as e:
            if att < ANSWER_RETRIES:
                lab.update(answer_attempt=att + 1, answer_error=str(e))
                self.pm._update(plan, term, f"LabMate: unusable answer fields ({e}); asking again", level="warning")
                return
            ans = {"experiment": uc.get("title") or UNKNOWN, "confirmed": False, "phase": UNKNOWN,
                   "noise_ok": None, "error": str(e)}
        ans["text"] = lab.get("answer_text", "")
        lab["answer"] = ans
        self._set_state(plan, term, "reporting", f"answer understood: {ans['experiment']}, phase {ans['phase']}, "
                                                 f"confirmed {ans['confirmed']}, noise_ok {ans['noise_ok']}")

    async def _reporting(self, plan: Plan, term: int) -> None:
        lab = plan.lab
        jid = f"{plan.plan_id}-rep"
        if jid not in self.node.jobs.jobs:
            self.pm._guard(term)
            await self.node.jobs.create("Give the operator your spoken summary now.", kind="report",
                                        system=REPORT_SYSTEM_PROMPT, context=report_facts(lab), job_id=jid)
        if plan.job_id != jid:
            plan.job_id = jid
            self.pm._update(plan, term)
        try:
            text = (await self.pm._wait_job(jid, term)).strip()
        except P._JobFailed as e:
            text = ""
            lab["report_error"] = str(e)
        lab["report"] = text or report_facts(lab)
        self._set_state(plan, term, "done", "DONE: report ready")


def _new_round(r: int, purpose: str) -> dict:
    return {"round": r, "purpose": purpose, "passes": []}
