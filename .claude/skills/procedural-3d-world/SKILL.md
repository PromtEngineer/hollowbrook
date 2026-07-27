---
name: procedural-3d-world
description: Build an explorable, photoreal 3D world in Three.js from a reference image or brief — fully procedural, with no external assets of any kind. Every texture, mesh, and the HDRI environment are generated in code at load time. Use when asked to create an explorable 3D scene, world, environment, level, or walkable space with first-person controls, PBR materials, real shadows, physics, interiors, and a frame-rate budget. Also use when asked to recreate a place from an image in 3D, to build a Three.js world with no art assets, or to review/debug an existing Three.js scene that looks flat, dark, washed out, or runs slowly.
---

# Procedural 3D world

Distilled from a shipped project: an explorable first-person village, ~26,000 lines, built by
nine parallel agents against a shared contract and driven through five measured review loops.
No texture files, no model files, no `.hdr` — the sky, its image-based lighting, every material
and every mesh are generated at load. 60 fps, 11 enterable buildings with furnished interiors.

The village is not the point. **The method and the measured facts are.** Most of what follows
is transferable to a space station, a jungle, or a city block.

---

## The one-paragraph method

Author a **contract** first, then build subsystems **in parallel** against it, then **measure**
the result in a browser and fix what the measurements say — not what the screenshots suggest.
Every hard defect in this project was found by measurement and would have been missed, or
misattributed, by looking.

---

## Rules that are not negotiable

1. **Contract before code.** Geometry modules never touch physics and never create materials.
   They emit collider *descriptions* and ask for materials *by name*. This is what lets several
   agents build simultaneously without stepping on each other, and it is what made integration
   a wiring job rather than a rewrite. See `references/architecture.md`.
2. **A single source of truth for placement.** One layout file owns where every building, road,
   prop and light lives. Nothing re-derives a position that the layout already publishes. When
   a shell has an interior, the shell publishes its own rooms; the furnisher never re-derives
   where a wall is.
3. **Measure, don't eyeball.** Screenshots tell you something looks wrong. They do not tell you
   why, and they routinely point at the wrong subsystem. See `references/verification.md`.
4. **Verify library APIs against `node_modules`, not memory.** Several facts that most training
   data gets wrong cost this project real time. See `references/three-api-notes.md`.
5. **Determinism.** Seeded RNG everywhere, never `Math.random()`. A defect you cannot reproduce
   is a defect you cannot fix.
6. **Everything repeated more than ~8 times is instanced or merged.** Budget draw calls from the
   start; retrofitting batching is painful.
7. **No allocation in a per-frame path.** Hoist scratch vectors to module scope.
8. **Fail soft at the seams.** Each subsystem is built behind a stage that catches, logs, and
   substitutes an inert stub. One broken module must not produce a black screen — it must
   produce a world with a hole in it and a named failure.
9. **Metres, and sanity-check against a human.** Eye height 1.68, door head 2.05, riser 0.18.
   Scale errors are extremely hard to see in isolation and obvious in situ.
10. **State what you did not verify.** A report that marks its unverified claims is worth far
    more than one that presents everything with equal confidence.

---

## Workflow

### 1. Author the contract and the layout (do this yourself, first)

Write, before any subsystem exists:

- **`contracts.js`** — collider spec, interactable spec, canonical material names, and any
  hand-off type (e.g. a `RoomSpec` describing an interior). Start from `assets/contracts.js`.
- **`layout.js`** — the authored plan. Positions, rotations, footprints, roads, sun bearing.
  Authored, not random, so the composition is under your control.
- **`config.js`** — quality presets and every tunable. Distinguish **bake-time** knobs (texture
  size, mesh density) from **runtime** knobs (pixel ratio, AO on/off); only the latter may be
  changed by an adaptive performance manager.
- **`main.js`** — the integration, written *before* the modules it imports, so each module's
  call signature is fixed by how it is actually used.

Getting this right is most of the job. A vague contract produces subsystems that cannot be
integrated; a precise one produces subsystems that snap together.

### 2. Build the subsystems in parallel

One agent per stream, each owning **its own files exclusively**. The split that worked:

| Stream | Owns |
|---|---|
| materials | procedural texture bakery + material library |
| lighting | sky shader, HDRI bake, sun, shadows, light pooling |
| terrain | ground, roads, height field, collision field |
| buildings | the structures — and they publish their own interiors |
| props | hero objects and everything with physics |
| vegetation | trees, ground cover, wind |
| physics | the world, colliders, character controller, grab/throw |
| controls | first-person camera, movement, interaction |
| performance | post-processing chain, adaptive quality, overlay |

Give every agent: the contract files, the reference image description, the budgets, the
verified API notes, and an instruction to **report what is still weak**. That last list is what
feeds the first review pass.

### 3. Integrate, then run the measured review loop

Load it, then for each loop:

1. Run the automated scene audit (`assets/audit.js`) — it catches what screenshots cannot:
   NaN transforms, missing materials, blended-instead-of-alpha-tested foliage, collision that
   disagrees with the visual ground, budget overruns.
2. Walk the world and capture several viewpoints.
3. For each visible defect, **isolate it by measurement before assigning it** — toggle the
   suspect subsystem off and re-measure, don't guess.
4. Dispatch one fix agent per affected file, handing it the measured evidence and an explicit
   list of what is already good and must not regress.

Repeat until a full pass is clean. Five loops was enough here; the last two were polish.

---

## What actually goes wrong

The full catalogue with root causes and detection recipes is in
`references/defect-catalogue.md`. Read it before building — most of these are cheap to avoid
and expensive to find. The five worth knowing before you write a line:

- **A surface that looks flat may be inside-out.** Backface culling hides the top and shows you
  the underside. Invisible in code review; found by ray-casting front-vs-double sided.
- **A dark scene is usually albedo, not lighting.** Raising exposure to fix it makes everything
  milky. Measure the baked albedo.
- **A normal map cannot carry detail finer than ~5 texels.** Finer features cancel to flat —
  which is why one roof read as plastic despite a beautifully authored fibre pattern.
- **Bloom thresholds are relative to your sky's radiance, not to 1.0.** A procedural sky can
  sit well above 1 everywhere, putting the entire sky over threshold.
- **A backtick inside a GLSL string terminates the JavaScript template literal.** Shipped
  twice. Costs a syntax error at import time.

---

## Reference material

| File | What's in it |
|---|---|
| `references/architecture.md` | the contract, the stream split, integration, how to parallelise |
| `references/verification.md` | the measurement harness — how to prove a claim in a browser |
| `references/defect-catalogue.md` | every real bug, its root cause, and how it was detected |
| `references/measured-baseline.md` | every measured number from the shipped build, and what was *not* measured |
| `references/three-api-notes.md` | verified three.js API facts + how to re-verify on a new version |
| `references/procedural-textures.md` | the GPU bake system, packing, tiling, and the sampling traps |
| `references/material-calibration.md` | measured albedo table and the calibrate-by-measurement loop |
| `references/lighting-and-sky.md` | procedural HDRI, sun as composition, the indoor IBL problem |
| `references/geometry-recipes.md` | silhouette, real depth, merging, LOD, interiors |
| `references/physics-and-performance.md` | character controller, doors, budgets, adaptive quality |

## Reusable code

`assets/` holds the parts that transfer directly:

- **`contracts.js`** — the interface template. Start here.
- **`audit.js`** — the scene audit and walkthrough harness.
- **`rng.js`** — deterministic seeded RNG, value noise, fBm.

---

## Scope honesty

This method suits a **hand-authored, bounded scene** — a village, a station, a street — where
composition matters and the whole world fits in memory. It is not a streaming open-world
recipe, and it says nothing about networking, animation rigs, or authored narrative. The
procedural-everything constraint is a real constraint: it buys zero asset pipeline and total
determinism, and it costs you the ability to art-direct a specific object by hand.
