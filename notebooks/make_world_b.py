"""make_world_b.py - builds worlds/component_b.wbt (4 TurtleBot3 robots with Emitter/Receiver)
by reusing the header, background, light and arena of component_a.wbt."""
import math
import re

src = open("worlds/component_a.wbt").read().splitlines()
header = [l for l in src if l.startswith("#VRML") or ("EXTERNPROTO" in l and not l.startswith("#"))]
keep = {"WorldInfo", "Viewpoint", "TexturedBackground", "TexturedBackgroundLight", "RectangleArena"}

blocks, cur, depth, name = [], [], 0, None
for line in src:
    if depth == 0:
        m = re.match(r"^([A-Za-z0-9_]+)\s*\{", line)
        if not m:
            continue
        name, cur = m.group(1), []
    cur.append(line)
    depth += line.count("{") - line.count("}")
    if depth == 0:
        if name in keep:
            blocks.append("\n".join(cur))
        cur = []
print("copied blocks:", len(blocks), "(expected 5)")

POSITIONS = [(-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5)]
robots = []
for i, (x, y) in enumerate(POSITIONS):
    heading = math.atan2(-y, -x)                        # initially facing the arena centre
    robots.append(f'''TurtleBot3Burger {{
  translation {x} {y} 0
  rotation 0 0 1 {heading:.4f}
  name "robot{i}"
  customData "{x} {y}"
  controller "swarm_controller"
  extensionSlot [
    RobotisLds01 {{
    }}
    Emitter {{
    }}
    Receiver {{
    }}
  ]
}}''')

with open("worlds/component_b.wbt", "w") as f:
    f.write("\n".join(header) + "\n\n" + "\n".join(blocks) + "\n" + "\n".join(robots) + "\n")
print("wrote worlds/component_b.wbt")
