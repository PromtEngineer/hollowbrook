# Reusable code

Lifted from the shipped project. `rng.js` and `contracts.js` transfer almost verbatim;
`audit.js` needs the porting notes below.

---

## `rng.js` — deterministic randomness

Copy as-is. mulberry32 + a string hasher + 2D value noise + fBm.

The part that matters is `Rng.fork(tag)`. Without it, adding one `rng.next()` call to a
generator reshuffles everything downstream of it, so a small change to one building silently
re-rolls the whole village. Fork a child generator per subsystem and per object:

```js
const rng = new Rng(plot.seed);
const trim = rng.fork('trim');    // adding calls here cannot disturb `rng`
```

Never `Math.random()`. A defect you cannot reproduce is a defect you cannot fix, and every
review loop depends on the world being byte-identical between reloads.

---

## `contracts.js` — the interface template

The load-bearing file. Adapt the *names* to your world; keep the *shape*.

Keep:
- **`ColliderSpec`** and its builders (`boxCollider`, `sphereCollider`, `capsuleCollider`,
  `trimeshColliderFromObject`, `boxColliderFromObject`). Geometry emits these; physics consumes
  them. Neither imports the other.
- **`InteractableSpec`** — object3D + body type + local collider + mass + damping + optional
  joint. This is the whole physics-object protocol.
- **`tagInteractive`**, **`setShadow`**, **`disposeGroup`** — small shared helpers that stop
  nine modules inventing nine conventions.
- **`variantKey`** — stable cache key so material variants still batch.

Replace:
- **`MATERIAL_KEYS`** / **`TEXTURE_KEYS`** with your world's surfaces. This list is a contract:
  the materials stream must produce every name, and geometry may only ask for names on it.
- **`RoomSpec`** / **`OpeningSpec`** / **`ROOM_USES`** — these are the interiors hand-off. Keep
  the *pattern* (a producer publishes its own interior; consumers never re-derive geometry) even
  if your world has no rooms.

Always set `tag` on colliders. It is what turns a raycast into a diagnosis.

---

## `audit.js` — the verification harness

Run every review loop. Catches what screenshots cannot.

**Porting:**

1. It reads a global handle — rename `window.HOLLOWBROOK` to your own, or pass `H` explicitly.
   Expose `{ THREE, scene, camera, renderer, physics, terrain, player, frameCtx, failures }`.
2. `VIEWPOINTS` is this village's tour. Replace with your own; keep the `look` target
   convention (yaw is derived, so the stops stay correct when the layout moves).
3. `auditInteriors()` assumes `buildings.interiors.rooms` publishes `RoomSpec`s. Delete it if
   your world has no interiors, or repoint it at your equivalent.
4. The ground-agreement check imports layout predicates to sample only open ground. It now falls
   back to sampling everywhere if that import is missing — see the note in the file.

**Generic checks worth keeping whatever you build:** NaN transforms; magenta debug materials in
use (a missing material name); foliage/emissive cards with `transparent && alphaTest === 0`
(hard-edged rectangles — this shipped twice); geometry without normals; objects fallen below the
world; `scene.environment` null; collision ground disagreeing with the analytic ground; draw and
triangle budgets.

**The `tour()` and `goto()` helpers** matter more than they look — a repeatable set of camera
stops makes visual regressions across review loops comparable instead of anecdotal.

---

## Not included, and why

The world-building modules themselves (buildings, terrain, vegetation, textures, lighting) are
~24,000 lines and specific to a thatched village. Their *techniques* are extracted into
`../references/` — recipes and traps rather than code you would have to gut. Copying a
2,000-line thatch generator into a project about a space station is not reuse.
