# Lighting and sky, without an .hdr file

Reference implementation: `src/lighting/sky.js` (1278 lines), `src/lighting/lighting.js` (1357 lines).
three.js **0.185.1**, verified against `node_modules/three/build/three.module.js` and `three.core.js`.
Every library claim below cites the line that proves it.

Nothing in this project loads an asset. The environment map, the fog colour, the sun tint and the
ground-bounce colour are all read out of **one analytic atmosphere model evaluated twice** — once in
GLSL for the visible dome, once in JS for everything that needs a number. That duplication is the
core architectural idea: *if the fog is sampled from a different model than the sky, they drift, and
the drift is visible as a seam at the horizon.*

---

## 1. Sun geometry is art direction, not mood

**The transferable rule: elevation is a composition control, azimuth is a lighting control.**
Pick them in that order, and pick them from the frame you want, not from a time of day.

### Shadow length

Shadow length on flat ground is `casterHeight / tan(elevation)`. That is the whole reason to care
about elevation.

| Elevation | shadow / caster height | 10 m eave throws |
|---|---|---|
| 20° | 2.75 | 27.5 m |
| **28°** | **1.88** | **18.8 m** ← shipped first, wrong |
| 32° | 1.60 | 16.0 m |
| **36°** | **1.38** | **13.8 m** ← shipped |
| 45° | 1.00 | 10.0 m |
| 60° | 0.58 | 5.8 m |

**Failure mode actually hit** (`src/world/layout.js:623-628`): at 28° the ~10 m eaves on the east
side of a plaza of radius 26 m laid 19 m of shade across it and swallowed the centre of the square —
including the well, the hero prop. Detected by walking to the plaza centre and finding the hero prop
in shadow at the authored hero hour. Fix: 36°, shadow 13.8 m, clears the centre by ~13 m.

Procedure for a new world:
1. Measure the tallest caster near the hero framing (`h`) and the open radius that must stay lit (`R`).
2. Require `h / tan(elev) < R`, i.e. `elev > atan(h / R)`. Plaza: `atan(10/26) = 21°` bare minimum;
   36° gives a comfortable margin and still reads as afternoon.
3. Only *then* worry about whether the elevation feels golden.

### Bearing decides which facades exist

`SUN.azimuthDeg = 57` is the compass bearing the light comes **from**
(`src/lighting/sky.js:201-206`: `x = sin(a)cos(e), y = sin(e), z = -cos(a)cos(e)`, so `+z` is south).

A wall whose outward normal has bearing `β` receives `N·L = cos(elevation) · cos(β − azimuth)`.

**Failure mode:** the first pass shipped azimuth 118° (south-east). Every plaza-facing facade fell
into shade and the square read flat. The hero cottage's front (measured normal bearing **56.0°**)
went from **N·L 0.40 → 0.88** when the sun moved to bearing 57 (`sky.js:146-151`,
`layout.js:615-620`). At 36° elevation a wall pointing straight at the sun caps at
`cos(36°) = 0.809`, which is why 36 is also the *ceiling* here: 37° would drop it to 0.799 and break
the project's "plaza-facing facades stay above 0.8" guarantee (`sky.js:157-158`).

Procedure: **list the walls that must be sunlit, average their normals' bearings, put the sun
there.** Then choose elevation from shadow length. Realism of the resulting sun path is the last
thing to give up, not the first — see below.

### The consequence: an impossible sun, deliberately

No real latitude puts the sun in the north-east at 15:50. Rather than compromise the composition,
the *direction of travel* was sacrificed: in Hollowbrook the sun rises in the south (05:05, bearing
171), transits at 72.9° in the south-east (bearing 117) and sets ENE (18:47, bearing 63) — a 13.6 h
day (`sky.js:160-166`). The hero hour is exact; only sweeping the whole time-of-day slider reveals
the cheat. **Rule: art-direct the hero frame exactly and let the parts nobody looks at absorb the
error.**

---

## 2. The pole-solve trap — check a solver's feasible range before touching its input

The daily sun path is a circle of angular radius `SUN_RHO` about a pole, and the pole is **solved at
load** from exactly one constraint: at `t = DEFAULT_TIME_OF_DAY` the direction must equal the
authored hero sun (`solveSunPole`, `sky.js:250-304`). The solver is a 360-step sign-change scan on
the phase error plus 60 bisection steps; residual ~1e-9 rad, cost well under a millisecond, once.
Two roots exist; it takes the one with the higher transit (`sky.js:289`).

**The trap.** The hero sits at phase `(0.66 − 0.5)·2π = 57.6°` past the transit — well down the arc —
so the higher the hero elevation, the higher the transit must be, and past some elevation **no pole
on the candidate ring satisfies the phase at all**. At the originally shipped `SUN_RHO = 80°` the
ceiling is a hero elevation of **33°**: scanned across rho 50–89°, 80° finds no pole from 34° up
(`sky.js:169-186`). Raising elevation from 28 to 36 while leaving rho at 80 would have hit the
`if (!best)` branch, fallen back to a **baked hard-coded axis** (`sky.js:301`), and thrown the entire
composition away with only a `console.warn` to show for it.

Fix: `SUN_RHO = 56°`, solvable for hero elevations **5–45.5°**, so 36 sits with 9.5° of margin, and
the transit stays a believable 72.9° instead of the near-vertical 86–90° that rho values just under
the old ceiling produce.

**Transferable rules:**
- When a hand-authored parameter is fed into a solver, **find the solver's feasible range before
  changing the parameter**. A scan across the free knob is cheap and is the only way to see a cliff.
- A solver's failure branch must be **loud and must land somewhere defensible**. Here the fallback
  vector is literally the pole the scan produces for the shipped values, so a failure degrades to
  "correct but frozen" rather than "silently different world".
- **Clamp the authored input to the proven range at the boundary.** `SUN_ELEVATION_WINDOW = [22, 40]`
  and `SUN_AZIMUTH_WINDOW = [40, 78]` (`sky.js:198-199`) gate what `layout.js` may assert; 40 < 45.5,
  so no value the window admits can fail the solve. Out-of-window values are overridden **with a
  console.warn naming the exact edit to make** (`heroSunAngles`, `sky.js:214-228`) — not silently
  obeyed, not silently clamped.
- **Assert the invariant after the solve.** `sky.js:336-342` re-evaluates `sunDirectionAt(0.66)` and
  warns if it is more than 0.05° off the authored hero direction. Two lines; catches every future
  regression in this file.

---

## 3. Procedural HDRI: sky shader → CubeCamera → PMREM

The pipeline, all in `createSky()`:

```
ShaderMaterial (Preetham + raymarched cumulus, toneMapped:false)
  → rendered by CubeCamera into WebGLCubeRenderTarget { HalfFloatType, LinearSRGBColorSpace }
  → PMREMGenerator.fromCubemap(cubeTarget.texture, pmremTarget)
  → scene.environment
```

Sizes come from the quality preset (`src/config.js`): `envSize` 128 / 256 / 512 / 512 for
low / medium / high / ultra. `new THREE.CubeCamera(0.05, 20, cubeTarget)` (`sky.js:925`).

### The seven things that go wrong

| # | Failure | Fix in code |
|---|---|---|
| 1 | **Village bakes itself into its own env map** — lights, shadow maps, geometry all end up in the "sky". | A private `bakeScene` holding a second `Mesh` sharing the same geometry+material (`sky.js:929-932`). |
| 2 | **`fromCubemap(cube, target)` throws on the first call.** `_allocateTargets()` builds the ping-pong target, `_lodMeshes`, blur and GGX materials — and `_fromTexture` **skips it entirely when you pass a target** (`three.module.js:2876`: `const cubeUVRenderTarget = renderTarget \|\| this._allocateTargets();`). Pass a target on call one → empty `_lodMeshes` → throw. | `let pmremTarget = null;` and always `pmremTarget = pmrem.fromCubemap(tex, pmremTarget)` (`sky.js:940-948, 1029`). First call allocates; every later call reuses. |
| 3 | **Leaking a render target per rebake**, and materials holding a stale `envMap` reference. | Reusing one target keeps the **texture object identity stable** across time-of-day changes, so `scene.environment` and every cached `material.envMap` stay valid (`sky.js:14-19, 1029-1033`). |
| 4 | **The solar disc becomes a blocky hot square** after PMREM downsampling — it is sub-texel at 256². | `uSunDisc` forced to 0 for the duration of the bake, restored in `finally` (`sky.js:1020-1042`). The direct term is the `DirectionalLight`'s job anyway. |
| 5 | **A throw inside the bake leaves a render target bound → black screen.** | `catch { renderer.setRenderTarget(null) }` (`sky.js:1034-1038`). |
| 6 | **~10 ms hitch on the first bake** (shader compile). | `pmrem.compileCubemapShader()` up front, wrapped in try/catch so a hostile driver only costs speed (`sky.js:937-939`; the method exists at `three.module.js:2777`). |
| 7 | **Rebake storm while a time-of-day slider is dragged.** | `setTimeOfDay` → `scheduleBake()`, a `setTimeout` debounce with `REBAKE_INTERVAL_MS = 260` (`sky.js:115, 1049-1058`). Deliberately a **timer, not an `update()` check**, because the slider lives in the pause menu and `main.js` does not call `update()` while paused. `update()` carries a belt-and-braces `if (envDirty && !bakeTimer) scheduleBake()`. |

`dispose()` (`sky.js:1242-1255`) clears the timer, nulls `scene.environment` if it still points at
our texture, and disposes geometry, material, noise texture, cube target, PMREM target and the
generator. Note the `CubeCamera` is never added to a scene; `CubeCamera.update()` calls
`updateMatrixWorld()` itself when `parent === null` (`three.core.js:50440`).

### Env intensity has to live in the bake

`main.js:103-104` sets `scene.environmentIntensity = 1.0` and then
`materials.setEnvironment(sky.envMap, 1.0)` assigns `envMap` **per material**. Verified: three only
overwrites `envMapIntensity` from `scene.environmentIntensity` when `material.envMap === null`
(`three.module.js:18688-18690`). So once you assign `envMap` explicitly, **`scene.environmentIntensity`
is a dead knob**. The only global handle left is the level the cube is *captured* at:

```js
const ENV_BAKE_GAIN = 0.98;      // sky.js:83 — uEnvGain during the bake only
uniforms.uEnvGain.value = ENV_BAKE_GAIN;   // 1.0 for the visible dome
```

Measured effect of taking it from 0.72 → 0.98, alongside the ground-albedo correction: shadowed plaza
cobble 0.233 → 0.341 sRGB luminance, shadowed plaster wall 0.243 → 0.385, while
indirect/direct on the ground only moved 0.580 → 0.625 — the sun still models the square.
The 0.72 was compensation for the wrong problem (near-black ground textures).

**Rule: decide up front whether `envMap` is per-material or scene-level, because that choice
determines which global intensity knob is even connected.**

### Calibrating the direct term against it

`SUN_GAIN = 1.22` on `layout.js`'s authored `SUN.intensity = 3.6` → effective 4.39 (`sky.js:64`).
Solved, not guessed: it puts sunlit plaza cobble at **0.549 sRGB luminance** (target band 0.50–0.62)
and stays in band across the plausible albedo range (0.511 at cobble albedo 0.24, 0.582 at 0.32).
`toneMappingExposure` stays at 1.0, so sunlit plaster peaks at 218/255 rather than clipping.
An earlier pass used 1.6 and the square went flat. **The check that matters is not brightness but
the linear lit:shadow irradiance ratio on the ground: 2.60 here, 2.72 before — a scale, not a
compression.** Tone mapping is ACES, applied exactly once, in `OutputPass` (`src/core/engine.js:28`,
`src/post/post.js:18`); everything upstream including the sky is genuine linear HDR
(`toneMapped: false` on the sky material, `sky.js:897`).

### Sky shader details worth stealing

- **Pin the dome to the far plane**: `gl_Position.z = gl_Position.w` in the vertex shader
  (`sky.js:552`). The dome's radius then no longer matters and `camera.far` can never clip it.
  Radius is 1, `renderOrder = -1000`, `depthWrite: false`, `frustumCulled = false`, and `update()`
  copies the camera position into the group so the eye is always inside it (`sky.js:1230`).
- **Give the lower hemisphere a real ground colour** (`uGroundColour`, `sky.js:697-700`). Preetham
  clamps to the horizon value below `y = 0`; without this the PMREM lights the whole village from
  underneath. The ground colour is computed as `albedo × (direct + sky)` in JS
  (`sky.js:1147-1160`) with `GROUND_ALBEDO = [0.24, 0.225, 0.155]` — an area-weighted mix of the
  actual baked ground textures (setts ~0.28 linear, grass ~0.13, stone/soil ~0.20). **Without it,
  plaster shadows go acid blue.**
- **Preetham goes black several degrees before sunset.** Its earth-shadow hack collapses `sunE` from
  about 5° elevation down. Put twilight back with a broad night floor plus a gaussian bump on the
  horizon: `NIGHT_SKY = [0.0040, 0.0060, 0.0160]`, `DUSK_SKY = [0.0260, 0.0240, 0.0550]`,
  blended by `smoothstep(0.22, -0.25, sun.y)` and `exp(-(sun.y/0.10)²)` (`sky.js:420-427, 1075-1079`).
- **Clamp the solar disc.** Physical solar radiance is ~6e6; `SUN_DISC_RADIANCE = 46.0` (`sky.js:94`).
  Anything near the real value turns bloom into a white sheet and saturates the half-float composer
  target.
- **Sun tint must carry hue only.** `sunColour` is the atmospheric extinction normalised so the
  largest channel is 1; all brightness lives in `sunLevel` (`sky.js:1090-1108`). Fold brightness into
  the colour and **noon comes out dimmer than late afternoon**, because normalising a whiter tint
  throws the energy away. Extinction elevation is floored at ~1.1° (`Math.max(0.02, sun.y)`,
  `sky.js:382`) or the path length runs away and the tint degenerates to pure red.
- **Pin the authored colour at the hero hour.** Before the first `applyTime`, the boot block
  (`sky.js:1259-1271`) computes `SUN_CORRECTION = SUN.colour / normalisedExtinction(t=0.66)`, so at
  the hero hour the `DirectionalLight` gets *exactly* `layout.js`'s `0xffd9a0`, and the physical model
  supplies only the variation away from it.
- **Cloud size is controlled by octave count, not base frequency.** `periodicFbm(4, 3, …)` — three
  octaves, not five. At 5 octaves the mean contiguous cloud island measured **0.27 shell units** (a
  field of small puffs, whatever the base scale); at 3 it is **0.6–0.8** and reads as cumulus
  (`sky.js:503-513`). Extra octaves add detail *at the scale of the coverage threshold*, which chops
  the contour into fragments. Measured result: `uCoverage 0.51` + `uClusterBias 0.26` → 26 % of the
  sky above 5° under cloud, masses averaging 16° of arc, gaps 15° (`sky.js:860-867`).
- **Never guard a texture fetch behind a density test inside the sun-march loop.** A fetch in
  non-uniform control flow has undefined derivatives → wrong mip → shimmering blocks along every
  cloud silhouette. Four wasted taps on clear sky is the cheaper mistake (`sky.js:750-762`).
- **Opacity from optical thickness, not coverage**: `aT = 1 − exp(−2.6·(d0 + 0.85·d1 + 0.70·d2))`
  (`sky.js:772`). Summing coverages left every mass a 40 %-alpha veil the sky showed straight through.

---

## 4. The indoor IBL problem — every enclosed space in three.js hits this

**`scene.environment` has no occlusion whatsoever.** The interior face of a sealed wall receives
exactly the same sky irradiance as the exterior face. Left alone, every room reads as a brightly,
flatly lit box and the hearth fire does nothing. The `DirectionalLight` *is* correctly occluded, so
sunlight shafts through a window survive — which makes the mismatch worse, not better: the shaft is
invisible against an equally bright wall.

There is exactly one clean handle, and it is **per-material `envMapIntensity`**
(`lighting.js:107-158`). Hollowbrook's values, absolute (they *replace* the material definition's own
`env`, they do not multiply it):

| key | value | applies to | why |
|---|---|---|---|
| `surface` | **0.16** | walls, main interior surfaces | env term = 0.16/0.85 = **19 %** of what the same plaster takes outdoors → lands at ~0.20–0.21 sRGB luminance (~52/255) |
| `deep` | **0.09** | ceiling boards, joist undersides, stair soffits, cupboard interiors, attics | roughly half the wall level, which is what a real room does above head height |
| `reveal` | **0.45** | door/window reveals, sills, thresholds, inner faces of an open leaf | these genuinely see a big lump of sky; **starve them and the openings read as holes cut in cardboard** |

Exterior calibration for scale: sunlit ground irradiance ~4.3 (direct 2.58, env ~1.61, hemisphere
0.099); shadowed exterior plaster 0.385 sRGB luminance.

**A physically honest number would be wrong.** A room with two small leaded windows has a real
daylight factor of 1–3 %, i.e. `surface ≈ 0.03`, which renders black. 0.16 is the cinematic answer:
unmistakably indoors, still readable, and comfortably below the 0.55–0.60 a sunlit window shaft keeps.

**Failure mode to expect and its retrofit.** If any stream builds interior walls from
`materials.get('lathPlaster')` instead of `materials.variant(key, { envMapIntensity: … })`, those
surfaces get the full unoccluded sky and that room is a daylit box. `applyInteriorEnv()`
(`lighting.js:189-202`) stamps `userData.envScale` onto the canonical instances of
`INTERIOR_ONLY_MATERIALS = ['lathPlaster', 'floorBoard', 'ceilingBeam', 'soot']` and re-runs
`materials.setEnvironment()`. It is a **fallback, not the main path** — and note the explicit warning
not to widen the key list to `flagstone`, `sackcloth`, `strawLitter`, which legitimately appear
outdoors (doorstep, market stall, stable yard) where dimming them is a visible exterior regression.

**Transferable rule:** the interior/exterior split must be expressed in the *material set*, not in
scene-level state, because scene-level IBL state is global and geometry is not. Budget for it in the
material system from day one: a `variant()` that can override `envMapIntensity` is the whole ask.

---

## 5. Ambient is the wrong tool — use a tilted hemisphere

**The failure:** interiors were dark, so an `AmbientLight` was raised to intensity **0.52**
(~0.0159 of luminance-weighted irradiance, ~9 % of the interior fill). It worked, and it flattened
everything: a flat term added to every normal in the world with no shape and no occlusion is the
single most reliable way to make a room look like a lit box (`lighting.js:204-247`).

**The fix, and the trick worth remembering:**

> **three derives a `HemisphereLight`'s axis from its world *position*.** Verified:
> `WebGLLights` does `uniforms.direction.setFromMatrixPosition(light.matrixWorld); uniforms.direction.transformDirection(viewMatrix);`
> — `three.module.js:8923-8928`. Length is irrelevant (it is normalised in the shader path); it must
> not be zero.

So a hemisphere light is a **free directional handle**. Hollowbrook folds the indoor fill into it:

```js
const INDOOR_HEMI_CUT      = 0.40;   // level: most of the sky is walled off
const INDOOR_HEMI_SKY      = 0x93b4e0;  // cool — stands in for the leaded glazing
const INDOOR_HEMI_GROUND   = 0x6d4a2a;  // warm — boards, straw litter, hearth bounce
const INDOOR_HEMI_TINT_MIX = 0.70;
const INDOOR_HEMI_TILT     = 0.55;   // axis lerped toward the sun direction
const INDOOR_AMBIENT       = 0.05;   // was 0.52 — now a black-floor guard and nothing else
```
(`lighting.js:217-250`, applied in `applyIndoor()`, `lighting.js:828-851`.)

At `INDOOR_HEMI_TILT = 0.55` the bright/cool lobe faces the window wall and the warm lobe faces the
back of the room: **the only cheap source of horizontal directionality an occlusion-free fill can be
given.** The tilt axis is the sun direction itself (`hemiTilt.copy(dir)`, `lighting.js:776`), so the
cool lobe automatically tracks whichever wall the windows are actually taking light from.

Numbers: `0.15 × hemiBase = +0.033` intensity, worth about the same irradiance as the 0.52 ambient it
replaced — same energy, now shaped. `INDOOR_AMBIENT = 0.05` leaves ~0.0015, under 1 % of the interior
level: a corner behind a dresser is dark rather than dead.

Outdoors the hemisphere is weak on purpose. `HEMI_DAY_INTENSITY = 0.22` (night 0.035) contributes
**0.099 of the 3.08 total irradiance reaching sunlit ground — 3.2 %** (`lighting.js:81-93`). It
*shapes*; the measured job of lifting shadows belongs to the env bake. At 0.30 it was adding a flat
8 % of the sun's irradiance to every surface regardless of what it could see — the classic way to
make a sunlit square look ambient-lit.

**Crucially, everything in `applyIndoor()` is multiplied by `indoorMix`, so at `indoorMix = 0` it
writes bit-for-bit the values the exterior was signed off with**: hemi 0.22 on
(`HEMI_SKY 0x9fc4ff`, `HEMI_GROUND 0x6b5a3c`), axis exactly `(0,1,0)`, ambient 0.02 on `0x2a2f38`.
That is what makes a global indoor mode safe — the exterior seen through a doorway moves by ~0.5 % of
its irradiance, under the visible threshold, which is why one material set suffices instead of two.
`INDOOR_RAMP = 0.42 s` crossfade, long enough to read as eye adaptation.

---

## 6. Light pooling: a fixed pool, faded, and room-aware

**The hard constraint that shapes everything:** the number of lights in the scene must **never change
after construction**. Adding, removing, or even toggling `visible` on one rewrites
`NUM_POINT_LIGHTS` and recompiles every material in the world mid-frame (`lighting.js:28-30`). Same
for `castShadow`, which rewrites `NUM_POINT_LIGHT_SHADOWS` — so `castShadow` is fixed per slot for
the lifetime of the run (`lighting.js:548`).

Architecture: `N` `PointLight`s created up front, reassigned to the nearest *anchors* each frame.
Pool size by preset: low 3, medium 5, high 7, ultra 8. Shadow casters: low 0, medium 1, high 1,
ultra 2, at 384 / 512 / 768 texels (`lighting.js:359-395`).

### Selection

Insertion into a fixed-size top-K array — **no sort, no allocation** (`lighting.js:1040-1072`).
Score is **penalised squared distance**, and the penalties multiply `d²`, so a penalty of 16 means
"count this anchor as four times further away":

| constant | value | case |
|---|---|---|
| `PENALTY_CROSS_ROOM` | 16 | player and anchor in different rooms — walls both ways |
| `PENALTY_INSIDE_OUT` | 9 | player indoors, anchor outdoors |
| `PENALTY_OUTSIDE_IN` | 4 | player outdoors, anchor indoors — a fire glimpsed through an open door is worth a slot; it is the cue that tells you the building has an inside |
| `CROSS_ROOM_RANGE_SQ` | 36 | hard cull for a different-room anchor beyond 6 m |
| `dy² > 2.25` | — | hard cull vertically across rooms: **sideways through a doorway yes, through a floor never** |

**The case this exists for:** a hearth 5 m away *in the room you are standing in* scores 25; a
lantern 4 m away *through a wall* scores `16 × 16 = 256`. The hearth wins, which is correct and is
exactly what pure distance gets wrong.

Ranges are on **real metres**, separately from the penalty — `ANCHOR_RANGE = 35 m` outdoors,
`INDOOR_ANCHOR_RANGE = 20 m` indoors (`lighting.js:95-105`) — because ranking with the penalty would
let an indoor anchor cull itself.

Per-kind `weight` multiplies `d²` before the room penalty. `fire.weight = 0.25` means "count a fire as
half as far as it is": **without it a furnished taproom with three table candles crowds its own hearth
out of a 3-slot pool purely on metres**, which is precisely backwards — the hearth is the light that
tells you what room you are in (`lighting.js:286-294`).

### Popping

Slots cross-fade over `FADE_TIME = 0.25 s`. A slot leaving the top-K is marked `retiring` and fades
out; an unclaimed anchor is only handed to a slot **after that slot's fade has reached ~0**
(`lighting.js:1091-1105`, `1151-1167`). Never reassign an on-screen light instantly.

### Rooms

`setRoomResolver(fn)` takes `roomAt(x,y,z) → identity | falsy` from the interiors module; identities
are interned to small integers for `===` comparison (`internRoom`, `lighting.js:604-613`). Absent a
resolver, everything degrades to pure distance — the behaviour the file shipped with. Anchors may
also stamp `room:` / `indoor:` on themselves and skip the resolver entirely.

Two probing subtleties, both learned the hard way (`lighting.js:615-648`):
- A hearth's flame anchor sits *in the firebox*, inside the chimney breast, legitimately **outside**
  the room's clear volume. Fires get a wide probe set: centre, +0.35 m up, and ±0.6 m on both
  horizontal axes.
- **Nothing else gets the wide probe.** A street lantern bracketed 0.3 m off a cottage wall would
  probe straight through 0.4 m of plaster, be declared indoors, inherit `fire.indoor.dayFloor` and
  **start burning at 72 % in broad daylight on a signed-off exterior.**

Cost control: anchor rooms are resolved once per epoch, never per frame. The player probe runs at
20 Hz while moving (>0.15 m) and 6 Hz standing still, and probes **the eye, not the feet** —
`playerPosition.y + 1.5` — because a player in a doorway has their feet on the threshold and their
head unambiguously in one room (`lighting.js:1236-1260`). A resolver that throws is caught, nulled,
and the system falls back to distance-only with a warning rather than dying.

### Indoor light specs

Each kind carries an `indoor` override merged at anchor-add time (`lighting.js:276-329`). Two things
always change inside: **reach comes in** (a room is 3–5 m and a 16 m fire spills across the plaza
through walls that cannot occlude it) and **`dayFloor` goes near 1** (an indoor fire burns as brightly
at 16:00 as at midnight; a street lantern does not).

| kind | outdoor candela / dist / dayFloor | indoor |
|---|---|---|
| lantern | 9.0 / 14 / 0.00 | 7.0 / 8.5 / 0.72 |
| window | 6.5 / 12 / 0.16 | 4.0 / 7.0 / 0.30 |
| fire | 16.0 / 16 / 0.22 | **6.0** / 11 / 0.88 |
| candle | 3.5 / 7 / 0.00 | 3.4 / 5.5 / 0.80 |

**The hearth clipping failure, in full** (`lighting.js:295-321`) — this is the best worked example of
calibrating an emissive against a scene's own scale. Indoor fire candela was 17.0, "solved for
brightest thing in the room" and overshot into "only thing in the room": with the taproom hearth's
authored scale 1.1, day floor 0.88 and flare peak 1.45, the `PointLight` measured **20.6**, putting
hearth stone 0.5 m away at linear radiance **5.5**. **ACES clips past 2.87**, so fire and surround
were one white blob with no flame structure. At 6.0 the peak is `6.0 × 1.1 × 0.88 × 1.45 = 8.4`
(nominal 5.8), giving measured:

| surface | distance | linear | sRGB |
|---|---|---|---|
| hearth stone | 0.5 m | 1.72 (2.49 on a flare) | 0.951 (0.974) |
| plaster | 1.0 m | 0.81 | 0.883 |
| oak settle | 1.5 m | 0.35 | 0.727 |
| far wall | 3.0 m | 0.09 | 0.365 |

against an env-lit interior wall at 0.20. **25× a candle, 9× the room's own fill at 2 m** — so the
room reads as firelit rather than evenly lit — while peaking just under sunlit exterior plaster
(0.858 sRGB). *Nothing indoors should out-expose the exterior the same frame can see through the
doorway.*

### Fire colour and flicker

Blackbody (Tanner Helland approximation, valid well below 6600 K) baked to a **9-entry LUT over
1850–2250 K** as linear colours, lerped per frame — allocates nothing, costs no `log()`
(`lighting.js:429-471`). Red channel pinned at 1 so candela stays the only brightness knob. Colour
temperature **rides the flicker**: hotter/less red on a flare.

Fire flicker is three bands, all deterministic off the anchor's `Rng` phase (`lighting.js:1002-1018`):
`wander` 0.90 / 1.63 / 2.71 Hz (mutually incommensurate — this carries most of the amplitude, it is
what the eye reads as "a fire"), `jitter` 11.3 / 17.9 Hz at a sixth amplitude, and `flare`, a
rectified 0.37 Hz beat `max(0, sin(...) − 0.86) × 7.1` that **spends ~92 % of its time at zero** and
then throws a short sharp lift. **Without the flare a fire animates evenly and reads as a sine, not a
flame.** Range for amp 0.30 is roughly [0.74, 1.45].

### Point-light shadows

- **Only `light.distance` bounds the cost.** Verified: `WebGLShadowMap` does
  `const far = light.distance || camera.far;` for every point light (`three.module.js:9369`) —
  it *overwrites* whatever you set on `shadow.camera.far`. This is why the indoor fire spec pulls
  distance from 16 m to 11 m: it is the only lever on how much of the village six cube renders walk.
- `shadow.autoUpdate = false` plus explicit `needsUpdate`. A point-light shadow is **view-independent
  and a hearth does not move** — the flicker changes intensity, not geometry — so: two quick refreshes
  at 0.09 s to catch interior geometry that streamed in after assignment, then **1.1 Hz indoors**
  (0.9 s) versus 5 Hz outdoors (0.2 s, where props are throwable) (`lighting.js:1197-1221`).
- Skipped entirely while the light is dark or its room is not the player's.
- Because `castShadow` cannot be toggled, **the shadow is moved by swapping the anchor into the slot
  that already casts** — a state swap of `{anchor, fade, retiring, worth}`, seamless because position,
  colour and intensity are recomputed from the anchor every frame. Reconciled every 0.45 s with
  **+0.6 hysteresis so it does not chatter** (`lighting.js:1110-1146`).

---

## 7. Directional shadow: fit a sphere, snap to texels

`fitShadow()` (`lighting.js:855-913`).

1. **Fit a sphere, not a box.** `shadowExtent = sqrt(R² + halfH²)` where `R = distance/2` and
   `halfH = SHADOW_WORLD_HEIGHT/2 = 17` (ground to tallest chimney). A sphere is invariant under the
   sun's rotation, so **the cascade does not change size as the day advances**. A fitted box breathes.
2. **Snap the centre to whole shadow-map texels, in the shadow camera's own basis.** Reconstruct
   exactly the basis `Matrix4.lookAt()` will build — `z = f`, `x = normalize(cross(up, f))`,
   `y = cross(f, x)` — round the `x` and `y` components to multiples of `texel = 2·extent/mapSize`,
   leave `z` free. **Snapping in any other frame does nothing.** Skip this and every shadow edge
   crawls as the player walks: "the single most visible shadow defect there is."
3. Degenerate guard: if `|f.y| > 0.9995` the cross product with world-up collapses, so nudge
   `f` to `(0.02, f.y, 0.02)`.
4. **Never flip `castShadow` to save cost.** Below the horizon, set `sun.shadow.autoUpdate = false`
   instead (`lighting.js:749-755`) — flipping `castShadow` rewrites `NUM_DIR_LIGHT_SHADOWS` and
   recompiles the world.
5. Bias is **both** kinds: `bias = -0.0005` **and** `normalBias = 0.03`. The constant alone
   peter-pans thatch eaves; the normal offset alone leaves acne on the near-tangent setts of the plaza.
6. The `DirectionalLight`'s `target` **must be added to the scene** or its `matrixWorld` is never
   refreshed and `DirectionalLightShadow.updateMatrices()` reads a stale position
   (`lighting.js:515-519`; the read is `three.core.js:45948`).

**Indoor cascade clamp.** Outdoors at `high`, `shadowDistance = 95 m` → ortho box 100.9 m across a
4096 map → **40.6 texels/m, 2.46 cm/texel** — resolves a doorway, not a 4 cm leaded glazing bar.
Indoors nothing beyond the room matters, so `INDOOR_SHADOW_DISTANCE = 48 m` → 60.4 m across →
**67.8 texels/m, 1.5 cm/texel**, which does resolve the bars (`lighting.js:252-262`). The blend is
**quantised to 0.5 m** so the 0.42 s ramp re-fits the ortho box ~20 times rather than 25/s, and so the
texel size holds still once the ramp lands. (At `low`, 1024 texels is the binding constraint and this
is not enough — unverified beyond the code comment.)

---

## 8. Fog is aerial perspective, and near-white fog starting too close eats the mid-ground

`THREE.Fog` is **linear in view distance with no height term**, so the only way to keep a village
crisp is to start it past the far edge of the village.

```js
const FOG_NEAR = 170;   // lighting.js:56 — was 90 here, 60 in config.js. Both ate the mid-ground.
const FOG_FAR  = 1500;
```

170 m clears the whole plaza (radius 26), every building (< 40 m), the hero trees (< 60 m) and the
first rise of the hills (nothing above ground before ~138 m). Resulting fog factor:

| distance | fog factor |
|---|---|
| 250 m | 0.06 |
| 400–620 m (hill crests) | 0.17–0.34 |
| 940–1060 m (far ridge) | 0.60 |

**The white-band failure.** Fog colour was tracking the analytic horizon, which runs above 1.0
radiance — after ACES that is **0.94 sRGB, a white band that swallowed the hills.** Two independent
clamps fixed it (`lighting.js:58-76`, applied at `lighting.js:791-801`):

- `FOG_MAX_LUMINANCE = 0.46` linear, hard cap → tone-maps to about **0.78**: unmistakably haze, still
  brighter than the hills behind it, never brighter than the sky above it.
- `FOG_LUMINANCE_SCALE = 0.62` — the fog tracks only 62 % of the sky's own horizon luminance.

**Hue and level are decided separately, and that separation is the technique.** Hue comes from
`sky.horizonColour` normalised to unit max (so the haze can never disagree with the band it blends
into), then pulled `FOG_HAZE_MIX = 0.32` toward a warm `0xffd2a6`, scaled by `dayMix`: **the raw
analytic horizon reads slightly green, and a near-neutral fog is what made the whole picture milky.**
Level is then imposed by rescaling to the capped target luminance.

The horizon colour itself is an 8-direction average of the JS atmosphere at `y = 0.06`, weighted
`1 + 1.6 × side` toward the sun and carrying the same authored warm band the fragment shader adds
(`sky.js:1110-1145`) — otherwise the fog reads cooler than the sky it blends into. The band strength
is gated by `smoothstep(-0.10, 0.07, sun.y)` so it does not keep glowing orange all night and drag
the fog with it.

**Colour space gotcha, verified:** three reads `fog.color.getRGB(uniforms.fogColor.value, getUnlitUniformColorSpace(renderer))`
(`three.module.js:14992`), so when rendering into a linear target these are **linear** values and no
conversion happens. Set them with `setRGB(r, g, b, THREE.LinearSRGBColorSpace)`.

---

## Checklist for a new world

1. List the surfaces that must be sunlit → sun **azimuth**. List the tallest caster and the open
   radius that must stay lit → sun **elevation** via `atan(h/R)`.
2. If a solver sits between your authored angles and the runtime direction, **scan its free knob for
   the feasible range first**, clamp the authored window inside it, and assert the hero direction
   after the solve.
3. Bake the env from a **private scene containing only the sky**, into a HalfFloat cube, through
   PMREM into a **reused** target (first call must pass `null`). Kill the sun disc during the bake.
   Debounce rebakes on a timer, not in `update()`.
4. Decide per-material vs scene-level `envMap` **before** tuning intensity — it decides which global
   knob is connected. If per-material, the env level must live in the bake gain.
5. Calibrate the direct term by the **lit:shadow linear irradiance ratio**, not by brightness.
6. Interiors: a per-material `envMapIntensity` tier (surface / deep / reveal ≈ 0.16 / 0.09 / 0.45 of
   an exterior 0.85). Physically honest daylight factors render black; pick cinematic.
7. Fill goes through a **tilted `HemisphereLight`** (axis = world position), never an `AmbientLight`.
   Keep ambient at ~0.02–0.05 as a black-floor guard only.
8. Fixed light pool, fixed `castShadow` per slot, cross-faded reassignment, room-penalised `d²`
   ranking, and a per-kind `weight` so the storytelling light outranks nearer clutter.
9. Fog near past the far edge of the playable set; cap fog luminance well under the horizon's; take
   hue from the sky and level from a scaled cap, separately.

### Not verified here

- The "26 % sky coverage / 16° masses" and per-surface sRGB tables are quoted from the source
  comments as measured by the review loops; I did not re-measure them.
- The `low`-preset shadow claim ("1024 texels is the binding constraint and this is not enough")
  is asserted in a comment referencing an external report.
- The claimed sunrise/transit/sunset times (05:05 / 72.9° / 18:47) follow from the solve but were
  not recomputed.
- `stats.lastBakeMs` is recorded at runtime; no baked figure for PMREM cost at `envSize 512` appears
  in the source, so the "rebaking costs milliseconds" claim is the author's, unquantified.
