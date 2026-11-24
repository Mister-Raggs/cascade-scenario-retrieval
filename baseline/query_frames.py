"""Split templated CASCADE queries into (relation, clause A, clause B)."""
import re

def norm(s):
    return re.sub(r"\s+", " ", s.strip().strip(".").strip()).lower()

# Order matters: more specific frames first. Relation tells how clause A relates to clause B in time.
FRAMES = [
    (r"^it is only after (?P<a>.+?) that (?P<b>.+)$", "a_then_b"),
    (r"^(?P<a>.+?); at a later point, (?P<b>.+)$", "a_then_b"),
    (r"^(?P<a>.+?), and later (?P<b>.+)$", "a_then_b"),
    (r"^(?P<a>.+?); after (?:that|this), (?P<b>.+)$", "a_then_b"),
    (r"^(?P<a>.+?); at the same time, (?P<b>.+)$", "while"),
    (r"^(?P<a>.+?), and at the same time (?P<b>.+)$", "while"),
    (r"^(?P<a>.+?) at the same time as (?P<b>.+)$", "while"),
    (r"^at the same time as (?P<a>.+?), (?P<b>.+)$", "while"),
    (r"^while (?P<a>.+?), (?P<b>.+)$", "while"),
    (r"^(?P<a>.+?) while (?P<b>.+)$", "while"),
    (r"^after (?P<a>.+?), (?P<b>.+)$", "a_then_b"),
    (r"^(?P<b>.+?) after (?P<a>.+)$", "a_then_b"),
    (r"^before (?P<b>.+?), (?P<a>.+)$", "a_then_b"),
    (r"^(?P<a>.+?) before (?P<b>.+)$", "a_then_b"),
    (r"^(?:here|in this scene), (?P<a>.+)$", "single"),
    (r"^(?P<a>.+)$", "single"),
]

def split(text):
    t = norm(text)
    for pattern, rel in FRAMES:
        m = re.match(pattern, t)
        if m:
            g = m.groupdict()
            return rel, norm(g["a"]), norm(g["b"]) if g.get("b") else None
    raise ValueError(text)
