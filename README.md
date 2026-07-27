# Hollowbrook

An explorable first-person 3D village, built with Three.js and Rapier, recreated from a
single reference image of a thatched half-timbered village square.

Everything you see is generated in code at load time. There are no texture files, no
model files, no HDR files — no assets of any kind on disk. The village, its materials,
its sky and its lighting are all procedural.

---

## Running it

```bash
npm install
```

```bash
npm run dev
```

Then open **http://127.0.0.1:5173** and click **Enter the village**.

For a production build:

```bash
npm run build && npm run preview
```

Force a quality tier with a URL parameter — useful for testing:
`http://127.0.0.1:5173/?quality=low` (`low` | `medium` | `high` | `ultra`).
Without it, the tier is auto-detected from your GPU and adapts at runtime.

**Requirements:** any WebGL2 browser (Chrome, Edge, Firefox, Safari 16+) and Node 18+
to run the dev server.

---

## Controls

| | |
|---|---|
| `W` `A` `S` `D` / arrows | walk |
| mouse | look |
| `Shift` | sprint |
| `Ctrl` / `C` | crouch |
| `Space` | jump |
| `E` | pick up / drop |
| left click | throw the held object |
| `F` | lantern |
| `Tab` | performance overlay |
| `Esc` | settings |

The settings panel has quality, adaptive resolution, mouse sensitivity, field of view,
**time of day**, bloom, exposure, ambient occlusion, head bob, and a physics debug
wireframe. Time of day re-bakes the environment map, so the whole village relights —
try dusk with the lanterns.

---

## What's in the box

**Rendering.** Physically-based materials throughout, lit by an image-based lighting
environment that is itself procedural: a custom atmospheric-scattering sky shader with
raymarched cumulus is rendered into a float cubemap and prefiltered with `PMREMGenerator`
to produce a real HDRI. A single texel-snapped directional light provides the sun and the
shadows. The frame runs through an `EffectComposer` — ambient occlusion, HDR bloom above a
threshold, subtle grain and vignette, SMAA, and exactly one ACES filmic tone map in the
final `OutputPass`.

**Physics.** Rapier runs on a fixed 1/60 timestep with interpolated transform sync. The
player is a swept capsule driven by Rapier's kinematic character controller — it cannot
tunnel through geometry, it steps over kerbs, it slides along walls and refuses slopes
that are too steep. Barrels, crates, buckets, pots and apples are real dynamic bodies you
can shove, carry and throw; the well bucket hangs on a rope joint and the shop signs swing
on hinges.

**Content.** Eleven buildings, each generated from a plot description in
`src/world/layout.js`: real timber framing with the plaster inset behind it, jettied upper
floors, deep thatch with rounded eaves, brick chimneys, dormers, leaded windows and plank
doors. Around them, a cobbled plaza, a dry-stone well, market stalls, carts, fences,
hundreds of flowers, and a valley of instanced trees and grass that moves in the wind.

**You can go inside all of them.** Every front door is a real hinged body on a revolute
joint, hanging open, with the wall collider broken into segments around the opening. Inside
are 18 rooms — board or flagstone floors, lath-and-plaster between exposed studs, oak joists
overhead, rafters and the underside of the thatch in the top storey, and a staircase to the
upper floor in the seven buildings that have one. Each room is furnished for its trade: a
taproom with a bar and benches, a bakery with a domed brick oven, a stable with stalls and
straw, bedrooms with rope beds and washstands. Hearths hold a real fire that lights the room
and flickers, and there are tankards, loaves, crocks and candlesticks you can pick up and
throw. Interiors are built once and then culled to the room you are in: standing in the inn's
taproom measures 634 draw calls against 502–512 outdoors, rather than eleven buildings' worth.

**Performance.** Everything repeated is instanced or merged. The performance manager
watches real frame timings and adapts resolution first, then ambient occlusion, bloom and
shadow distance, with hysteresis so it never oscillates.

---

## Layout

```
src/
  main.js              integration — the only file that knows the whole system
  config.js            quality presets and every tunable
  contracts.js         the interfaces every subsystem is written against
  core/                renderer, scene, camera, HUD
  world/
    layout.js          THE VILLAGE PLAN — where every building and prop lives
    terrain.js         ground, plaza, roads, hills, height field
    buildings.js       the eleven buildings — and it publishes its own interior
    interiors.js       floors, wall linings, joists, stairs, upper floors
    furnishings.js     furniture, hearths, and the things you can pick up
    props.js           the well, carts, barrels, lamps, flowers — everything physical
    vegetation.js      trees, bushes, grass, wind
  materials/           procedural GPU-baked PBR textures and the material library
  lighting/            sky shader, HDRI bake, sun, shadows, lantern light pool
  physics/             Rapier world, colliders, character controller, grab/throw
  player/              first-person controls and camera
  post/                the composer chain
  perf/                frame timing, adaptive quality, the overlay
```

`src/contracts.js` is the load-bearing file: geometry modules never touch physics or
create materials, they emit collider descriptions and ask for materials by name. That
separation is what let the subsystems be built independently and integrated without
rewiring. Interiors work the same way — `buildings.js` publishes a `RoomSpec` per storey
with its real clear dimensions and every door and window opening, and the two interior
modules build against that rather than re-deriving where a wall is.

---

## Debugging

`window.HOLLOWBROOK` is exposed in the console: the scene, every subsystem, live
`stats()`, and `teleport(x, y, z, yaw)` for jumping around the village without walking.
`Tab` shows draw calls, triangles, physics step time and the adaptive resolution state.
