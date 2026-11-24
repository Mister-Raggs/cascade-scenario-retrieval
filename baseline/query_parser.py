"""Rule-based translation of templated CASCADE queries into the devkit DSL (frozen 2026-10-14 fidelity version).

Ambiguous phrase mappings carry several candidate fragments; `choice` selects one per key
(chosen on training queries only). Unparsed clauses raise `Unparsed`.
"""
import re
from .query_frames import split

class Unparsed(ValueError):
    pass

# Subjects: phrase -> ('ego'|'agent'|'obj'|'light', filters)
AGENT_SUBJECTS = [
    (r"road user", {}),
    (r"vehicle", {"type": "vehicle"}),
    (r"car", {"type": "car"}),
    (r"heavy-duty vehicle", {"type": "truck"}),
    (r"truck", {"type": "truck"}),
    (r"bus", {"type": "bus"}),
    (r"public bus", {"type": "bus"}),
    (r"motorcycle", {"type": "motorcycle"}),
    (r"motorcyclist", {"type": "motorcycle"}),
    (r"bicycle", {"type": "bicycle"}),
    (r"cyclist", {"type": "bicycle"}),
    (r"scooter", {"type": "scooter"}),
    (r"emergency vehicle", {"type": "emergency"}),
    (r"adult pedestrian", {"type": "ped_adult"}),
    (r"pedestrian", {"type": "ped"}),
    (r"person", {"type": "ped"}),
    (r"child", {"type": "ped_kid"}),
    (r"road worker", {"type": "ped_personnel"}),
    (r"police officer", {"type": "ped_officer"}),
    (r"officer", {"type": "ped_officer"}),
    (r"animal", {"type": "animal"}),
]
MODIFIERS = {"signaling": ("prop", "signal"), "slow": ("prop", "slow"), "fast": ("prop", "fast"),
             "erratic": ("prop", "erratic"), "aggressive": ("prop", "aggressive"),
             "double-parked": ("prop", "double_parked")}

AGENT_PREDICATES = {
    "is visible": None,
    "stops or is stopped but is not parked": ["stop"],
    "is stationary but not parked": ["stop", "not_move"],
    "stops": ["stop"],
    "parks": ["park"], "is parked": ["park"],
    "yields": ["yield"],
    "turns": ["turn"], "turns left": ["turn_left"], "turns right": ["turn_right"], "makes a u-turn": ["uturn"],
    "changes lanes": ["change_lane", "change_lane_left|change_lane_right"],
    "changes to the left lane": ["change_lane_left"], "changes to the right lane": ["change_lane_right"],
    "travels in their lane": ["drive"], "drives in its lane": ["drive"], "drives in their lane": ["drive"],
    "follows another road user": ["follow"],
    "walks": ["walk"], "stands": ["stand"], "runs": ["run"],
    "jaywalks": ["jaywalk_action", "jaywalk_flag", "jaywalk_any"],
    "slows down": ["decel"], "reverses": ["reverse"], "overtakes": ["overtake"],
    "is signaling": "PROP:signal",
    "rides in their lane": ["drive"], "rides in its lane": ["drive"],
    "comes to a full stop": ["stop"], "is stopped": ["stop"],
    "makes a small lateral move": ["nudge", "nudge_in|nudge_out"],
    "moves slightly within its lane": ["nudge_in"], "moves slightly out of its lane": ["nudge_out"],
    "is double-parked": "PROP:double_parked", "is moving slowly": "PROP:slow", "is moving fast": "PROP:fast",
    "crosses the road": ["walk"], "crosses": ["walk"],
}
EGO_PREDICATES = {
    "yields": ["yield"], "stops": ["stop"], "follows another road user": ["follow"],
    "drives in its lane": ["drive"], "turns": ["turn"], "turns left": ["turn_left"], "turns right": ["turn_right"],
    "makes a u-turn": ["uturn"], "slows down": ["decel"], "changes lanes": ["change_lane", "change_lane_left|change_lane_right"],
    "changes to the left lane": ["change_lane_left"], "changes to the right lane": ["change_lane_right"],
    "makes a small lateral move": ["nudge", "nudge_in|nudge_out"],
    "moves slightly within its lane": ["nudge_in"], "moves slightly out of its lane": ["nudge_out"],
    "reverses": ["reverse"], "overtakes": ["overtake"], "creeps forward": ["creep"], "creeps": ["creep"],
    "enters": ["enter"], "aborts a maneuver": ["abort"],
}
OBJECTS = {
    "a traffic cone or bollard": "cone|bollard", "a traffic cone": "cone", "a bollard": "bollard",
    "a stop sign": "stop_sign", "a yield sign": "yield_sign", "a speed-limit sign": "speed_limit_sign",
    "a do-not-enter sign": "do_not_enter", "a barrier": "barrier", "a warning sign": "warning_sign",
    "a temporary road marker": "portable_indicator", "a traffic sign": "traffic_sign", "a roadblock": "roadblock",
    "a boom gate": "boom_gate", "a merge-ahead sign": "merge_ahead", "a toll plaza": "toll_plaza",
    "a portable display": "portable_display", "debris": "other_debris", "road debris": "other_debris", "fallen object": "fallen_object",
}
LIGHTS = [
    (r"a traffic light shows (red|yellow|green)", lambda m: f"light.color = {m.group(1)}"),
    (r"a yellow traffic light is on the ego vehicle's path with the ego vehicle already in the intersection",
     lambda m: "light(color = yellow, ego_in_on_yellow = true)"),
    (r"a (red|yellow|green) traffic light is on the ego vehicle's path", lambda m: f"light(color = {m.group(1)}, affects_ego = true)"),
    (r"a (red|yellow|green) (left|right)-arrow signal is active", lambda m: f"light(color = {m.group(1)}, shape = arrow_{m.group(2)})"),
    (r"a (red|yellow|green) upward-arrow signal is active", lambda m: f"light(color = {m.group(1)}, shape = arrow_up)"),
    (r"a flashing (red|yellow|green) traffic light is active", lambda m: f"light(color = {m.group(1)}, state = flashing)"),
]

def _action(alias, kind):
    if alias == "jaywalk_flag":
        return "action.type = \"jaywalk\"" if kind == "agent" else None
    if alias == "jaywalk_any":
        return "action.type in (jaywalk_action, \"jaywalk\")"
    values = alias.split("|")
    return f"action.type = {values[0]}" if len(values) == 1 else f"action.type in ({', '.join(values)})"

SUFFIX = re.compile(r" (at the same time|at some point|only)$")
REWRITES = [
    (r"\bis turning\b", "turns"), (r"\bis yielding\b", "yields"), (r"\bis standing\b", "stands"),
    (r"\bis changing\b", "changes"), (r"\bis reversing\b", "reverses"), (r"\bis walking\b", "walks"),
    (r"\bis stopping\b", "stops"), (r"\bis parking\b", "parks"), (r"\bis following\b", "follows"),
    (r"\bis slowing down\b", "slows down"), (r"\bis jaywalking\b", "jaywalks"), (r"\bis running\b", "runs"),
    (r"\bfinishes turning\b", "turns"), (r"\bfinishes changing\b", "changes"),
    (r"\bgives way\b", "yields"), (r"\bis in view\b", "is visible"), (r"\bis present\b", "is visible"),
    (r"\bmoves slightly sideways\b", "makes a small lateral move"),
]

def normalize(t):
    t = SUFFIX.sub("", t.strip())
    for pattern, repl in REWRITES:
        t = re.sub(pattern, repl, t)
    return t

def clause(text, choice):
    t = normalize(text)
    if t.startswith("the ego vehicle "):
        pred = t[len("the ego vehicle "):]
        if pred not in EGO_PREDICATES:
            raise Unparsed(t)
        cands = EGO_PREDICATES[pred]; alias = cands[choice.get(("ego", pred), 0)]
        values = alias.split("|")
        return f"ego.action = {values[0]}" if len(values) == 1 else f"ego.action in ({', '.join(values)})"
    for pattern, build in LIGHTS:
        m = re.fullmatch(pattern, t)
        if m:
            return build(m)
    m = re.fullmatch(r"(a .+?|debris|road debris|fallen object) is visible", t)
    if m and m.group(1) in OBJECTS:
        values = OBJECTS[m.group(1)].split("|")
        return f"obj.type = {values[0]}" if len(values) == 1 else f"obj.type in ({', '.join(values)})"
    pred = next((p for p in sorted(AGENT_PREDICATES, key=len, reverse=True) if t.endswith(" " + p)), None)
    if pred is None:
        raise Unparsed(t)
    words = t[: -len(pred) - 1].split()
    if not words or words[0] not in ("a", "an", "one", "another"):
        raise Unparsed(t)
    single = words[0] == "one"; words = words[1:]
    subject = None
    for k in range(len(words)):
        hit = next((f for p, f in AGENT_SUBJECTS if p == " ".join(words[k:])), None)
        if hit is not None:
            subject, mods = hit, words[:k]; break
    if subject is None:
        raise Unparsed(t)
    parts = [f"type = {subject['type']}"] if "type" in subject else []
    for word in mods:
        if word not in MODIFIERS:
            raise Unparsed(t)
        parts.append(f"prop.type = {MODIFIERS[word][1]}")
    if single and choice.get(("det", "one"), 0) == 1:
        parts.append("amount = single")
    spec = AGENT_PREDICATES[pred]
    if isinstance(spec, str):
        parts.append(f"prop.type = {spec.split(':')[1]}")
    elif spec is not None:
        parts.append(_action(spec[choice.get(("agent", pred), 0)], "agent"))
    if not parts:
        return "agent.type = road_user"
    return f"agent({', '.join(parts)})"

def conjunction(text, choice):
    try:
        return clause(text, choice)
    except Unparsed:
        pieces = re.split(r",? and ", text)
        if len(pieces) < 2:
            raise
        op = " while " if any(p.endswith(" at the same time") for p in pieces[1:]) else " and "
        fragments = []
        for piece in pieces:
            same = re.match(r"(?:that|the) same (.+)$", piece)
            if same and fragments and fragments[-1].startswith("agent("):
                extra = clause("a " + same.group(1), choice)
                inner = [x for x in extra[len("agent("):-1].split(", ") if not x.startswith("type = ")]
                fragments[-1] = fragments[-1][:-1] + "".join(", " + x for x in inner) + ")"
            else:
                fragments.append(clause(piece, choice))
        return op.join(f"({f})" for f in fragments) if len(fragments) > 1 else fragments[0]

# Choices selected on training queries in research/parser-2026-10-14/fidelity.py; validation scored once.
FROZEN_CHOICE = {("op", "then"): 1}


def translate(text, choice=None):
    choice = FROZEN_CHOICE if choice is None else choice
    rel, a, b = split(text)
    if rel == "single":
        return conjunction(a, choice)
    op = "while" if rel == "while" else ["then", "before"][choice.get(("op", "then"), 0)]
    return f"({conjunction(a, choice)}) {op} ({conjunction(b, choice)})"
