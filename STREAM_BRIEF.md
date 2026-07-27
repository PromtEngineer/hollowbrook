# Hollowbrook — stream brief

You are building one stream of an explorable first-person 3D village in Three.js,
recreated from a reference image: **a cobbled plaza with a round stone well, ringed
by half-timbered cottages with steep thatched roofs and red-brick chimneys, ivy,
flower boxes and terracotta pots, green hills and conifers beyond, big cumulus
clouds, warm late-afternoon sun.**

You own **only the files listed in your task**. Do not create, edit or delete any
other file. Other agents are editing the rest of the project *at the same time*.

---

## Hard rules

1. **No network. No external assets.** No `fetch`, no CDN, no `.hdr`/`.glb`/`.png`
   files, no `TextureLoader` on a URL. Everything is generated in code. The only
   dependencies are `three` (0.185.1) and `@dimforge/rapier3d-compat` (0.19.3),
   both already installed.
2. **Read `src/contracts.js` first.** It defines the collider format, the
   interactable format, and the canonical material names. Import helpers from it
   rather than reinventing them.
3. **Read `src/world/layout.js`.** It is the single source of truth for where
   every building, road, prop site, tree and lamp post lives. Never hard-code a
   position that layout.js already provides.
4. **Read `src/main.js`** to see exactly how your factory is called and what it
   must return. The signature there is law — if you need a different one, you are
   wrong; adapt.
5. **Stay in budget.** Target 60 fps on a mid-range GPU at 1080p. Concretely:
   whole scene under **≈450 draw calls** and **≈2.2 M triangles** at `high`.
   Anything repeated more than ~8 times must be an `InstancedMesh` or merged
   geometry. Nothing per-frame may allocate — hoist `Vector3`/`Quaternion`
   scratch objects to module scope.
6. **Everything deterministic.** Use `Rng` from `src/util/rng.js` seeded from a
   constant or from `plot.seed`. Never `Math.random()`.
7. **Scale is metres.** Player eye height 1.68 m, doors ~2.05 m, ground floor
   ~2.9 m. Sanity-check every number against a human.
8. **Never throw at module top level.** Fail soft: log a warning and return a
   valid (possibly emptier) result.
9. ESM only, no TypeScript, no build step beyond Vite. Match the house style of
   the existing files: JSDoc on exported factories, terse comments that explain
   *why*, no decorative banners beyond what already exists.

## Verified API notes (three 0.185.1 — do not guess)

- Tone mapping is applied **once**, by `OutputPass`. Materials rendering into the
  composer's render target skip tone mapping automatically. Do not add another.
- `new SMAAPass()` takes **no arguments**. `new OutputPass()` takes none.
- `new UnrealBloomPass(resolutionVec2, strength, radius, threshold)`.
- `new GTAOPass(scene, camera, width, height)`.
- `EffectComposer` targets are `HalfFloatType` already.
- `import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js'`.
- Addons resolve through the `three/addons/*` path (works with Vite + three 0.185).
- Colour management is on: author colours as sRGB hex via `new THREE.Color(0x…)`;
  set `texture.colorSpace = THREE.SRGBColorSpace` on albedo maps only.
- Rapier: `import RAPIER from '@dimforge/rapier3d-compat'; await RAPIER.init();`
  The WASM is inlined as base64 — no asset config needed.

## Frame order (fixed by main.js)

```
player.update(dt)        // look, camera, interaction ray, HUD
physics.update(dt)       // fixed 1/60 substeps + interpolated transform sync
<world>.update(dt, ctx)  // terrain, buildings, props, vegetation, sky, lighting
post.render(dt, ctx)     // composer
```

`ctx` is `{ elapsed, dt, camera, playerPosition, quality, paused }`.

## Quality presets

`quality` is one of the objects in `src/config.js` plus `.name`. Fields you will
use: `textureSize`, `envSize`, `shadowMapSize`, `shadowDistance`, `anisotropy`,
`geometryDetail`, `vegetationDensity`, `thatchDetail`, `pixelRatioCap`,
`antialias`, `bloom`, `gtao`, `softShadows`, `cullDistance`, `grassDistance`,
`maxDynamicBodies`.

Only the knobs listed in `RUNTIME_KNOBS` may change after load; everything else
is baked once.

## What to return

Your factory returns the object described in your task. Geometry streams return a
`WorldChunk`:

```js
{
  group,          // THREE.Group — main.js adds it to the scene
  colliders,      // ColliderSpec[]  (world space, static)
  interactables,  // InteractableSpec[] (optional)
  lightAnchors,   // [{position, colour, intensity, kind}] (optional)
  update(dt,ctx), // optional
  dispose(),      // optional
  stats: {},      // free-form, shown in the perf overlay
}
```

## Deliverable report

When you finish, report (as your final message, plain text — it is the return
value, not a chat message):

1. **Built** — what exists now, file by file, with the exported API.
2. **Numbers** — draw calls, triangles, texture memory, build time your stream adds.
3. **Weak** — everything still weak, missing, faked or likely to look wrong, and
   what you would do next. Be blunt; this list feeds the review pass.
4. **Assumptions** — anything you assumed about another stream's behaviour.

Do not report success for something you did not verify. You can sanity-check your
module in isolation with `node --input-type=module -e "..."` for pure logic, but
anything needing WebGL will be verified by the integrator in a browser.
