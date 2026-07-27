# Physics, Controls & Performance

From Hollowbrook (three 0.185.1 + `@dimforge/rapier3d-compat` 0.19.3; 11 buildings, 18 rooms,
all enterable). Rules first, then the number they came from. Library claims are cited to
`node_modules`, not memory. Anything I could not confirm is in **Unverified** at the end.

---

## 1. Frame order

`main.js:212-239` is the only file that knows it: `perf.begin() → player.update(dt) →
physics.update(dt) → world updates → renderer.info.reset() → post.render(dt) → perf.end(dt)`.

- **`renderer.info.autoReset = false` + exactly one `reset()` per frame** (`engine.js:39`,
  `main.js:236`). This is what makes per-pass attribution free: counters accumulate across every
  nested `renderer.render`, so a before/after delta around a pass *is* that pass's cost.
- **A subsystem that throws gets an inert stub, not a black screen** (`main.js:26-45`); failures
  land on `window.HOLLOWBROOK.failures` and in the perf overlay. With N independently-built
  streams, one will throw during integration.

---

## 2. Fixed step + interpolated transform sync

`physics.js:1053-1100`.

```js
accumulator += Math.min(dt, 0.25);          // 1066 — clamp the tab-switch spike first
while (accumulator >= step && steps < maxSub) {
  savePrevious();                            // snapshot BEFORE stepping
  for (cb of stepCallbacks) cb(step, ctx);   // 1072 — player movement runs here
  applyCharacters(step); updateHeld(step);
  world.step(); accumulator -= step; steps++;
}
if (accumulator > step) accumulator = step * 0.999;   // 1086 spiral-of-death guard
writeTransforms(accumulator / step);                  // 1095 interpolation alpha
```

`fixedStep` 1/60, `maxSubSteps` 5 (`config.js:17-18`), gravity **-19.6** (2 g, deliberately not
9.81 — "reads better for a first-person game", `config.js:15`).

**Failure mode.** Without the alpha blend, props visibly stutter at *any* refresh rate that is
not exactly 60 Hz. Detection: on a 120 Hz display only *dynamic* bodies shimmer while the camera
stays smooth — that is the missing interpolation, not the solver.

Traps inside it: skip sleeping bodies in `savePrevious` (prev already equals cur) and write each
sleeping body exactly once via a `settled` flag (`physics.js:1108`, `1123-1129`);
`object3D.position` is **local to its parent**, so cache `parent.matrixWorld.invert()` once at
spawn and only when non-identity (`398-412`); **throw the backlog away** past `maxSubSteps` —
simulating slower than real time beats a tab that never catches up; and **prime the broad phase**
with one `world.step()` after adding colliders (`1045`), or the session's first raycast misses
geometry that is demonstrably there.

---

## 3. Movement inside the fixed step

`controls.js` registers via `physics.onFixedStep` (1031-1042) and self-steps at 1/60 only if the
hook is absent (1182-1192). Identical feel at 30 and 144 fps.

Two invariants worth copying verbatim (`controls.js:10-18`):
1. **The physics body is never smoothed.** Head bob, landing dip, strafe roll, breathing and
   sprint FOV all live in the *eye* transform layered on the interpolated body position
   (1197-1254). Smoothing the body makes collision resolution lie.
2. **Head bob is parameterised by distance travelled, not time** (stride 1.62 m,
   `controls.js:1022`, `1215-1218`). A time-driven bob keeps swaying after you stop.

| | value | |
|---|---|---|
| capsule | r 0.32, total 1.80 (crouch 1.15) | 0.64 m diameter — governs door design |
| walk / sprint / crouch | 3.0 / 5.9 / 1.45 m/s | |
| accel ground / air | 46 / 7 | air control ~15% of ground |
| jump vel / coyote / buffer | 6.2 m/s / 0.12 s / 0.14 s | all three needed to feel honest |
| `groundStick` | 0.25 m/s downward bias while grounded | `controls.js:93` |

`groundStick` is measured: Rapier's autostep refuses to lift once the desired motion has much
downward component, and the ceiling falls off fast — against autostep 0.45 m / snap 0.4 m, **a
0.35 m step needs ≤0.5 m/s and a 0.44 m step needs ≤0.25 m/s**, while snap-to-ground still holds
at 0.1 m/s walking off a 0.35 m ledge.

**The autostep/velocity interaction that eats a stride** (`controls.js:979-1003`). While the
sweep is lifting you it deliberately trades horizontal travel for height, so both the slide
projection and the "rebuild velocity from actual displacement" reconcile must be skipped:

```js
const stepping = res.movement.y > 0.004 && grounded && velocity.y <= 0;
```

Without it, the reconcile clamps velocity to the reduced horizontal movement and a stride over a
doorstep becomes a half-second crawl. **Rule: any reconcile-from-reality logic needs an explicit
exemption for step-up frames.**

---

## 4. The kinematic character controller is the no-clipping guarantee

`physics.js:544-573` — a swept capsule on a `kinematicPositionBased` body. Skin 0.02 m,
`enableAutostep(0.45, 0.12, true)`, `enableSnapToGround(0.4)`, climb angle 48°, **slide angle also
48°**, `setApplyImpulsesToDynamicBodies(true)`, `setCharacterMass(78)`, `setSlideEnabled(true)`.

`minSlopeSlideAngle == maxSlopeClimbAngle` is what stops the player parking on a 60° thatch roof.
`setCharacterMass(78)` is what lets a walk into a barrel move it.

**Never teleport** (`physics.js:617-638`): `computeColliderMovement` → `computedMovement()` →
`setNextKinematicTranslation`, wrapped in try/catch falling back to the uncorrected motion — a
solver hiccup must not freeze the player.

**Crouch/stand without clipping your head** (`physics.js:666-716`). Growing is refused unless a
shape-cast proves headroom. The trick: *the swept volume of the current capsule moved up by
`rise` is exactly the volume the taller capsule will occupy* (same radius, same feet), so **one**
cast is a complete test. Ignore hits with `normal1.y >= 0.6` — that is a floor you grazed, and
refusing on it traps the player in a permanent crouch.

---

## 5. THE AUTOSTEP TRAP — and the detection method, which generalises

**Symptom.** Buildings that could not be entered: the player *stalled on the threshold and could
only get in by jumping*, while the doorway itself measured completely clear (`config.js:44-52`).

**Cause.** `enableAutostep(maxHeight, minWidth, includeDynamicBodies)` — `minWidth` is *"the
minimum width of free space that must be available after stepping on a stair"*
(`control/character_controller.d.ts:96,110`). The first value was **0.25 m**, wider than the flat
top of a worn doorstep. Autostep silently refuses and the capsule stops dead against a 30 mm lip.

**Fix.** `PLAYER.stepMinWidth = 0.12` — "still half a bootprint, so it will not let the player
walk up a fence rail" (`config.js:44-53`). `stepHeight` stayed 0.45.

> **Diagnostic — jump vs walk.** When a character stalls somewhere geometrically clear, test
> jump against walk. Jumping bypasses autostep entirely. Jump in, walk blocked ⇒ not a collider,
> it is autostep refusing to lift. Clear approach rays *confirm* this rather than refuting it:
> rays test occupancy, autostep tests free space **on top of** the step. Different questions.

**Corollary.** Measure the narrowest horizontal top surface in your entry path and set `minWidth`
under it. This path stacks three ledges: steps with a 0.36 m nominal run (`buildings.js:1764-1770`),
a `building-threshold` slab topping 12 mm above the finished floor (`2447-2450`), and a
`building-weatherbar` collider only **0.090 m deep** × 0.060 m tall (`2453-2456`). Any of those can
undercut a default `minWidth`. `stepHeight` likewise bounds your authored stair: rise here is
`min(0.155, 0.42/count)` m (`1381`).

---

## 6. Doors: three findings, all measurement beating specification

`buildings.js:1000-1245`. Each leaf is its own dynamic body on a **vertical revolute joint**
(`{type:'revolute', anchorWorld, axis:[0,1,0]}`), translated at `physics.js:462-467`.

**(a) A kinematic capsule pushes but cannot pull.** A door with a real doorstop on its outboard
face opens inward from the plaza and then cannot be opened from inside — the first player to shut
one behind them is walled into the taproom forever. Everything limiting the swing is therefore
geometry only; the leaf stops at ~±170° on piers it cannot rotate through (`1010-1017`).

**(b) A 35 kg leaf is not shoveable.** A real 40 mm oak leaf that size *is* ~35 kg, but the player
pushes dynamic bodies through the character controller's impulse path, and at 35 kg on 2.4 angular
damping a walking shove barely moves it. Now **24 kg (barn) / 17 (paired) / 19 (single)**,
`linearDamping 0.8`, `angularDamping 2.4` (`buildings.js:1206-1212`).

**(c) "Ajar looks like 60°" is wrong on measurement** (`buildings.js:1031-1036`). At 60° a **1.08 m
leaf in a 1.20 m opening** puts its tip within 20 mm of the centre line 0.94 m inside the room:
**the straight corridor measures 0.64 m — exactly the capsule diameter**, so it cannot be used.
Only past ~85° is the opening a straight tube. The upper bound is *derived*, not chosen:

```js
const cLim    = clamp((HINGE_INSET * 0.95 - hT) / clearIn, 0, 1);
const openMax = Math.PI/2 + Math.asin(cLim);                     // ≈105°
const open    = Math.min(openMax - 0.12, rng.range(1.50, 1.60)); // 85.9°–92.8°
```

If `WALL_COL_IN`, `HINGE_INSET` or leaf thickness move, `openMax` moves with them, and the code
warns if the window closes (`1085-1090`).

**(d) The spawn-overlap bug behind it — the more general lesson.** `HINGE_INSET` was 0.045 m, so
the leaf hung on a plane 0.17 m inboard while the pier collider reaches `WALL_COL_IN = 0.43` m
inboard. Rapier resolved that spawn overlap the only way a body on a vertical revolute joint can:
**it spun the leaf.** Measured on the first step, *twelve of fifteen leaves picked up 2.0–6.2 rad/s
of yaw and ended 110–173° from where they were authored* (`98-115`); four of eleven buildings
became unenterable.

> **Rule: a jointed body must spawn in a pose where its collider touches nothing.** A spawn
> overlap does not error — it converts into velocity, and on a hinge that reads as "the art is
> wrong". Detection: log `body.angvel()` after the first step for every jointed body; anything
> non-zero at t=0 is a spawn penetration.

**(e) Survive the dynamic-body cap.** `physics.js:317-338` sorts specs so grabbables (0) and
jointed bodies (1) claim the budget before plain props (2); anything past `maxDynamicBodies`
(40/70/110/150 by preset) silently becomes a `fixed` body **with no joint**. A shut fixed leaf
makes its building unenterable — the third reason every leaf spawns ajar.

---

## 7. Grab / carry / throw by velocity, never by transform

`physics.js:750-863`, tunables at `77-81`: close 0.32 of the gap to the hold point per step via
`setLinvel`, clamp to 14 m/s, damp angular velocity ×0.82/step, release past 2.6 m.

**Rule: velocity-drive a held body, never write its transform.** A teleported body has no
velocity: it passes through walls and pushes nothing. A velocity-driven one still collides, still
shoves crates, and breaks its own hold when wedged — which is why the controller polls
`heldObject()` every frame instead of trusting its own state (`controls.js:1133`). On grab:
damping 2.4/6.0, `enableCcd(true)`; restore the spec's values on release. `maxCarryMass` 40 kg,
`carryDistance` 1.55 m.

`throwHeld` reads its argument as a **target speed** and multiplies by mass
(`physics.js:823-833`), so a barrel and a bucket leave the hand at the same speed (+ `m*1.2` of
loft). For thin props generally: `setSoftCcdPrediction(0.5)` on every dynamic body
(`physics.js:368`) is cheap and stops tunnelling at throw speed; full CCD only while held.

---

## 8. Collider specs as data — the decoupling that allowed parallel agents

`contracts.js:9-15` rule 1: geometry modules never talk to the physics engine. They emit
`ColliderSpec` / `InteractableSpec` POJOs; `physics.js:145-206` is the single translator and
`main.js:145-162` the single hand-off where six streams' arrays are concatenated.

One shared `RigidBodyDesc.fixed()` body carries **every** static collider (`physics.js:104`).
**Rapier colliders have no `userData`** — keep your own `Map<colliderHandle, {tag, object3D, spec,
kind}>` (`107`) so a raycast can name what it hit. A spec that cannot be honoured warns and is
skipped; the batch is never lost (`188-191`). Every collider carries a free-form `tag`
(`building-wall`, `building-threshold`, `door`…), which is what makes a console audit possible.

### Rapier 0.19.3 facts, verified in `node_modules`

| claim | file |
|---|---|
| `enableAutostep(maxHeight, minWidth, includeDynamicBodies)`; `minWidth` = free space *after* the step | `control/character_controller.d.ts:96,110` |
| autostep is **off** until `enableAutostep` is called | `control/character_controller.d.ts:102` |
| **Ray** hits expose `timeOfImpact` (camelCase) | `geometry/ray.d.ts:36,63,94` |
| **Shape-cast** hits expose `time_of_impact` (snake_case) | `geometry/toi.d.ts:12,44` |
| `castShape(pos,rot,vel,shape,targetDistance,maxToi,stopAtPenetration,flags,groups,exclCollider,exclBody,pred)` | `pipeline/world.d.ts:418` |
| `castRayAndGetNormal(ray,maxToi,solid,flags,groups,exclCollider,exclBody,pred)` | `pipeline/world.d.ts:343` |
| `world.lengthUnit` exists and scales every internal tolerance | `pipeline/world.d.ts:105,121` |

The camelCase/snake_case split is a real footgun: both compile, one silently yields `undefined`.
`physics.js:938` and `:688` use each correctly.

**Heightfield convention** (`physics.js:26-47`, established empirically — a transposed heightfield
is a silent, world-breaking bug): `nrows`/`ncols` are **cell** counts; `heights.length ===
(nrows+1)*(ncols+1)`; buffer is **column-major**, `index = row + col*(nrows+1)`; rows run along
**Z**, columns along **X**; centred on the collider translation. Re-pack a differently-ordered
producer buffer **once at build time** (`257-269`), never per frame.

**Heightfield raycast artefact.** A ray landing exactly on a cell boundary slips between cells and
reports nothing — verified against 0.19.3: *a downward ray at x=30.0 misses, x=30.01 hits*.
Measure-zero, except spawn points are round numbers (`playerStart = [0,0,34]`). Fix: retry once on
a miss, nudged 7.1e-5 m along **both** perpendicular axes (`919-935`) — nudging one leaves a
vertical ray on the same X cell edge.

---

## 9. Budgets that were actually used

`dev/audit.js:194-196` is the enforced gate.

| budget | threshold |
|---|---|
| draw calls | warn over **520**; budget **450** (includes shadow + AO passes) |
| triangles rendered | **2.6 M** |
| lights in scene | 12 |
| physics step (`stats.stepMs`, EMA) | 6 ms |
| interiors resident at once | 4 |
| collider sanity | ≥50, else "geometry is probably not solid" |

**The count is not the object count.** Every eligible mesh is submitted up to three times:
(1) the colour pass; (2) the sun's shadow map — a full scene re-render from the light; (3) GTAO's
normal+depth gbuffer — `_overrideVisibility()` → `_renderOverride(normalMaterial, …)` →
`_restoreVisibility()`, verified at `three/examples/jsm/postprocessing/GTAOPass.js:502-508`. Plus
6 cube faces per shadow-casting point light. `info.render.calls` counts all of it: `WebGLInfo`
increments it in `update()` (`three/src/renderers/webgl/WebGLInfo.js:18-20`), reached from
`renderBufferDirect`, which the shadow and override renders also use.

> **Rule: budget in *submissions*, not objects. Expect ≈3× the visible mesh count, and spend
> your effort shrinking passes 2 and 3, not pass 1.**

Both extra passes are cut by **distance**, not by quality. Sun shadow distance is a runtime knob
(55/72/95/120 m by preset). GTAO's gbuffer hides everything past **55 m** plus named subtrees
`sky`, `terrain.far`, `terrain.mountains` (`post.js:111-114`, `514-544`) — justified by
measurement, not taste: *a 0.35 m occlusion radius at 55 m is far below one pixel*, so those
objects can only cost draw calls, and their depth stays cleared, which reads as "unoccluded", what
they would have been. The hook is sanctioned, not a hack: `_restoreVisibility` un-hides everything
in `_visibilityCache`, so appending to it is safe (`GTAOPass.js:651-681`).

Point-light shadows refresh on demand: `shadow.autoUpdate = false` at construction, then
`needsUpdate = true` at **1.1 Hz indoors / 5 Hz outdoors**, never while the light is dark or its
room is not the player's (`lighting.js:557-559`, `1197-1221`) — a point shadow is view-independent
and a hearth does not move; flicker changes intensity, not geometry. And `light.distance` is the
**only** thing bounding a point-light cube render (WebGLShadowMap overwrites `shadow.camera.far`
with it), which is why indoor fires were pulled 16 m → 11 m (`lighting.js:1215-1219`).

**Per-pass attribution beats guessing** (`post.js:305-415`, `460-480`). Wrap each pass's `render`,
delta the info counters, and read GPU time from `EXT_disjoint_timer_query_webgl2`. Only one
`TIME_ELAPSED` query may be open at a time, so passes are profiled **round-robin, one per frame** —
each number is a few frames stale, fine at a 5 Hz overlay. Label the line `gpu` or `cpu`, so a
0.02 ms "cost" for a full-screen pass is not read as free. Overlay format
(`perf.js:541-566`): `pass 326d  render 214d/2.10M  gtao 96d/0.82M  bloom 12d  aa 3d  grade 1d`.

---

## 10. Adaptive resolution, and the bug that made it eat itself

`perf/perf.js`. Resolution first (cheapest knob, easiest to reverse), then a fixed ladder
**gtao → bloom → shadowDistance** (`199-222`). Only keys in `RUNTIME_KNOBS` may be written;
everything else is baked at load and `setKnob` refuses it (`180-186`, `config.js:158-166`).

### THE BUG

The slow-frame threshold was **16.6 ms**, below the **16.67 ms** vsync period of a 60 Hz display.
Every healthy frame counted as slow and the pixel ratio walked to its floor on hardware that was
never dropping a frame. Fixed at `slowMs = 19.5` (≈51 fps — genuinely missing), `perf.js:47-55`.

> **Rule: on a vsync-locked display the frame period can never fall below the refresh interval,
> so a "slow" threshold at or under it is always true.** Put the down-trigger a real margin above
> the period and add a dead band, so 16.67 ms provokes nothing at all.

Mirror image: such a machine can never measure *below* ~16.7 ms either, so resolution could never
be won back on period alone. `hasHeadroom()` (`387-390`) counts "at the vsync cap **and** cheap on
the CPU" as fast: `emaMs < 14.0 || (emaMs < 17.6 && cpuEmaMs < 10.0)`.

### Warm-up and hitch rules — the other half of the same collapse

Ignore the first **90 frames** (~1.5 s) after the world appears, re-armed 45 frames on pointer
lock and 30 on tab return (`perf.js:69`, `651-660`). Treat any single frame over **100 ms** as a
compile hitch — keep it in the history so `worst` stays honest, exclude it from the EMA *and* the
streaks — and any frame over **200 ms** as a stall, discarded outright. EMA α is 0.12 (~16-frame
constant). "Feeding a 300 ms compile into a 0.12 EMA poisons the average for 30+ frames, which is
exactly how the collapse to the floor happened" (`perf.js:73-78`). On leaving warm-up, reseed
`emaMs` from a clean frame (`359`), or the governor inherits a load-inflated average and steps
down on its first eligible frame.

### Hysteresis is the whole design

first down-step 180 slow frames (3 s) · later down-steps 60 · any up-step 180 fast frames ·
oscillation penalty ×1.6 per down-after-up, cap 900 · cooldown 60 frames after any change ·
pixel-ratio step 0.1, floor **0.65** · 2 slow streaks *at the floor* before dropping a tier ·
3 tier drops before that tier locks off for the session.

"A renderer that flips between 0.9 and 1.0 every second looks far worse than one that sits at
0.8." (`perf.js:38-41`)

**Rule: make the governor auditable.** `perf.js:381-419` carries a written proof, from the code,
that a steady 60 fps machine can never be stepped down — `slowStreak++` appears exactly once,
gated on one comparison; `emaMs` has exactly two writers, both guarded against hitches. If you
cannot write that paragraph about your own governor, it has a path you have not found.

---

## 11. Post chain order, and the bloom trap

`RenderPass → GTAOPass? → UnrealBloomPass → SMAA|FXAA → GradePass → OutputPass`
(`post.js:9-18`; nothing outside `post/` touches the composer, `contracts.js:14`).

- **One tone map, at the end.** `renderer.toneMapping = ACESFilmic`, but three skips the tone map
  in material shaders whenever it renders into a render target, so the chain is genuine linear HDR
  and `OutputPass` does the single conversion (`engine.js:1-9`). Never set `toneMapping` on a pass.
- **SMAA must precede OutputPass** — it works in linear-srgb, per three's own JSDoc (`SMAAPass.js:15`).
- **Grade before the tone map on purpose**: a vignette is lens falloff (a scene-referred multiply)
  and multiplicative grain reads as film response, not noise pasted on a finished image.
- Full-screen quads get `depthTest = depthWrite = false` — the composer's ping-pong targets carry
  stale depth from the RenderPass (`post.js:626-628`).

### THE BLOOM TRAP

Threshold was **1.25**, **below the sky's own HDR radiance** (sun disc at 46, lit cumulus deck
steadily 1.5–4). The whole sky was over threshold — and three's high pass is a **mix, not a
subtract**:

```glsl
float alpha = smoothstep( luminosityThreshold, luminosityThreshold + smoothWidth, v );
gl_FragColor = mix( outputColor /* black */, texel, alpha );
```
(`three/examples/jsm/shaders/LuminosityHighPassShader.js:60-62`)

Over threshold ⇒ **the whole texel passes at full value**. Every roofline seen against the sky
grew a white halo the width of the blur kernel.

**Fix** (`post.js:49-75`): threshold raised **above the sky dome** — 2.4 / 2.5 / **2.6** / 2.7 by
preset — with a soft knee of 2.2–2.4 fed to `smoothWidth`, so a cloud at 3 contributes ~2%, the
brightest cloud edge near the sun ~50%, the sun disc fully. Radius 0.5 → **0.35**.

`UnrealBloomPass.render` rewrites only `luminosityThreshold` per frame (`UnrealBloomPass.js:309`),
never `smoothWidth` — so a `smoothWidth` set externally sticks; its constructor sets it to 0.01,
i.e. a cliff (`:137`). **Cost, stated up front:** the lantern emissives are only ~1.7 in luminance
(0xffb45a at intensity 3.2) and no longer bloom on their own. That fix belongs in the lantern
material (a hotter core), *not* in a threshold that also catches 100% of the sky.

> **Rule: set the bloom threshold against your sky's actual radiance, not against 1.0.** Print the
> max luminance your sky shader writes, put the threshold above it, and use a knee, not a cliff.
> Detection: a halo of constant width along *every* silhouette against the bright region is a
> threshold problem, not a blur-radius problem.

GTAO tuning that mattered (`post.js:77-103`): radius **0.35 m** (crevice occlusion, not fake GI),
`distanceExponent` 1.6, blend **0.85** (<1 keeps AO off silhouettes, where the dark halo comes
from), internal resolution **0.5/0.6/0.75/0.85** of the framebuffer — *0.75 costs 56% of the fill
and is invisible, because the result is Poisson-denoised (deliberately blurred) immediately after*.

---

## 12. Build once, show few

`interiors.js:29-36`, `95-102`, `2003-2036`. Eighteen rooms built once at load, then hidden;
`update()` shows only the building the player is inside or standing in the doorway of.

show 7.5 m · hide 10.5 m (hysteresis) · hard cap **3** resident · re-check every 0.11 s or 0.4 m
of movement · plots ~16 m apart, which is what makes those radii work.

- **Toggle a *unit*, not a room.** You can see the upper storey up the stairwell; hiding only the
  room you stand in opens a hole in the ceiling.
- **Show and hide radii must differ**, or you flicker on the boundary; and the show radius must
  exceed the distance at which you can first see in, or you get pop-in instead of culling.
- **Keep the hard cap even when you believe it cannot bite** — "a budget you only hope you are
  inside is not a budget" (`interiors.js:2018`).
- **`visible = false` on a Group is a real saving:** `projectObject` returns immediately on
  `object.visible === false`, skipping the whole subtree
  (`three/src/renderers/WebGLRenderer.js:1831-1833`). Resident cost is exactly the visible units'
  mesh/triangle counts, and 0 outdoors (`interiors.js:2029-2037`).
- **Don't re-derive geometry across module boundaries.** Interiors build against the shell's
  published `RoomSpec` (`contracts.js:178-200`). Same pattern for vegetation LOD radii and grass
  fade, which derive from the runtime `cullDistance`/`grassDistance` knobs
  (`vegetation.js:1977-2004`), so the governor's ladder reaches them without any module knowing
  the governor exists.

**And the opposite move, for shader permutations** (`controls.js:686-695`): the hand lantern stays
`visible = true` with `intensity = 0` rather than being hidden, because three keys its program
cache on the count of **visible** lights — toggling visibility mid-play recompiles every material
in view. Baking the permutation in at load costs one zero-contribution spot evaluation per
fragment and buys a hitch-free toggle key. **Prefer changing a uniform over changing anything that
alters the shader permutation.**

---

## Unverified

- **"Four of eleven buildings unenterable from autostep."** `config.js:48-51` names **two**
  (weaver, granary) for the autostep cause. *Four of eleven* belongs to the separate
  `HINGE_INSET` spawn-overlap defect (§6d) — a different bug with a different fix. Both are real;
  do not merge them.
- **"The player stopped 0.92 m out"** / **"approach rays clear at every height and offset"** — not
  recorded in code. The jump-vs-walk discriminator *is* implied by `config.js:49-50`.
- **"0.25 m is Rapier's default `minWidth`."** In 0.19.3 autostep is off until `enableAutostep` is
  called, so 0.25 was this project's chosen first value; no JS-side default confirmed. (Rapier's
  Rust default is expressed as a *relative* length, not metres.)
- **"Draw calls ≈3.3× objects."** No measured figure in the repo. 3× is derivable (colour + sun
  shadow + GTAO gbuffer); the extra ~0.3 is consistent with point-light cube faces and post's
  full-screen quads, but the multiplier is not recorded. `perf.js:545-547` documents the overlay
  *format*, not a capture.
- No frame-time, fps or draw-call capture is stored anywhere in the repo — every number above is a
  constant, a threshold, or a comment recording a past measurement.
