# Procedural PBR textures baked on the GPU

From `src/materials/textures.js` (2154 lines) and `src/materials/materials.js` (465 lines). API claims
verified against `three@0.185.1` in `node_modules`. **37 texture sets, 24 GLSL programs, 106 render
passes**, zero external files; 27 sets are exterior, the interiors pass added 10 sets and 5 new families.

## 1. The deliverable

Every set is the same record (`bakeSet`, textures.js:2035-2049):

```js
{ map, normalMap, roughnessMap, metalnessMap, aoMap, ormMap,   // the last five are ONE texture
  alphaMap: null, hasAlphaInMap, worldScale, relief, atlas, directional, sizes }
```

Two fields carry the authoring contract out to geometry streams:

| field | meaning | why it exists |
|---|---|---|
| `worldScale` | metres covered by one uv tile | streams author UVs in metres and divide, so texel density is uniform world-wide and no material needs a bespoke `repeat` (materials.js:12-18) |
| `directional` | axis note, e.g. `'fibres along V (V runs down the pitch)'` | a `repeat` **cannot rotate** a texture; if the fibre must run the other way the *geometry* must swap u and v (materials.js:80-83) |

**Rule.** Publish physical scale and anisotropy axis with the maps. A set that says only "here is a map"
forces every consumer to re-derive tiling by eye, and they will disagree.

## 2. The bake pipeline

```
PlaneGeometry(2,2) + OrthographicCamera(-1,1,1,-1,0,1)   [textures.js:1892-1897]
  -> ShaderMaterial(PRELUDE + FAMILY[x] + SURFACE_MAIN)  [1961-1976]
  -> render into a pooled WebGLRenderTarget, one per size [1900-1916]
  -> readRenderTargetPixels into a Uint8Array            [1994-1997]
  -> new THREE.DataTexture(buf, ...)                     [2001-2016]
```

Vertex shader writes `gl_Position = vec4(position.xy, 0, 1)` (textures.js:219-225) — no matrices, the quad
is already in clip space.

**Three passes, one program.** `uPass` branches at the end of `SURFACE_MAIN` (textures.js:181-198): 0
albedo, 1 normal, 2 ORM. A family body supplies only `hgt()`, `alb()`, `orm3()`; the template supplies
`main()`. Atlas families supply `hgt()` + `atlasRGBA()` and get `ATLAS_MAIN` (textures.js:202-217), which
has no ORM pass at all.

### Why read back at all

The non-obvious part; the header explains it (textures.js:9-17). Verified in the library:

- `WebGLTextures.setTexture2D` skips `uploadTexture` when `texture.isRenderTargetTexture === true` and
  binds `textureProperties.__webglTexture` straight from the *texture's own* properties
  (three/src/renderers/webgl/WebGLTextures.js:559, 584).
- For a normal texture the GL object lives in a `_sources` WeakMap keyed by `texture.source`, sub-keyed by
  sampler state (`initTexture`, WebGLTextures.js:722-784; `getTextureCacheKey`, :528-549).
- `Texture.copy` assigns `this.source = source.source` (three/src/textures/Texture.js:478), and
  **`repeat`/`offset` are not in the cache key** — so a clone that only changes the uv transform reuses the
  same `WebGLTexture`.

Consequence: a render-target texture cannot be cloned (the clone gets fresh properties, no
`__webglTexture`, renders black), while a `DataTexture` clones for one JS object and zero GPU memory. That
is exactly what `materials.repeatTexture` exploits (materials.js:318-331) to give `thatchRidge` a stretched
repeat off the shared thatch bake.

**Rule.** If anything downstream will clone your baked textures to retint or re-tile them, pay the readback
and hand out `DataTexture`s. If nothing clones them, keep the render target and skip it.

**Costs, recomputed from the plan** (high, base 1024): 106 synchronous `readRenderTargetPixels` calls, each
a full pipeline stall, and **68.0 MB of Uint8Array retained on the JS heap** for the world's lifetime —
the DataTexture *is* the buffer. Budget ~3/4 of the GPU figure.

Three things worth stealing:
- `await new Promise(r => setTimeout(r, 0))` every 4 sets (textures.js:1939) — the loading bar cannot paint
  inside a synchronous bake loop, and "a 1 s frozen tab reads as a hang".
- `renderer.initTexture(t)` per finished texture (textures.js:2014) forces the upload during load instead
  of stuttering on the frame that first shows the surface.
- Renderer state is **borrowed**: render target, `autoClear`, clear colour and clear alpha are saved and
  restored in a `finally`, and every program and render target disposed (textures.js:1918-1924, 1941-1950).
  A bake that leaves the renderer clearing to black is a bug that surfaces three files away.

## 3. Families vs sets: one shader, many materials

`FAMILY` is a map of GLSL strings (24 entries). `SETS` is a flat table of 37 records
(textures.js:1648-1812), each naming a family plus four `vec4` uniforms:

```js
{ key: 'thatch', family: 'thatch', tier: 'hero', worldScale: 1.6, relief: 0.056,
  uA: [96, 4, 0.45, 1.0], uB: [0.55, 0.60, 16, 0], directional: '...' }
```

`material(spec)` caches by **family name**, not set key (textures.js:1956-1977), so `cobble` and
`cobbleWorn` share one program and differ only in `uA`/`uB`. Six sets ride `FAMILY.wood`, four ride
`FAMILY.fibre`, four ride `FAMILY.metal`.

**Rule — compile cost dominates, and it scales with families, not sets.** Adding a set to an existing
family costs three fullscreen passes at a few hundred thousand pixels: microseconds. Adding a family costs
a compile + link: tens to hundreds of milliseconds. Design the parameter vectors first; resist forking a
family for a variation a uniform could reach. The interiors pass was slow because it introduced **5 new
families** (`flagstone`, `lathPlaster`, `hearthStone`, `soot`, `strawLitter`), not 10 new sets. *(The
reported ~350 ms → ~4 s wall-clock is not recorded anywhere in the repo — §11.)*

Two conventions that keep it maintainable: every family header comments its own packing
(`/* uA = (cells, wear, damp, jointWidth) uB = (_, polish, gritty, _) */`, textures.js:234), without which
the SETS table is unreadable; and a signed-off family string is **frozen** — "a comment edit is a different
shader source and a different program" (textures.js:474-480), so dead uniform slots are documented rather
than deleted. (That reasoning is right; that particular comment is now stale — §11.)

## 4. The resolution planner and its budget

Three knobs (textures.js:1815-1858):

```js
TIER_FRACTION = {            // [albedo, normal, orm] as fractions of base
  hero: [1.0, 0.5, 0.5],  mid: [0.5, 0.5, 0.25],  low: [0.25, 0.25, 0.25],
  atlas: [0.5, 0.25, 0.0], interior: [0.375, 0.375, 0.1875], interiorLow: [0.25, 0.25, 0.125] }
BUDGET_MB = { low: 26, medium: 92, high: 92, ultra: 224 }
snap = n => clamp(round(n/64)*64, 64, 2048)
mib  = size => size*size*4 * 4/3 / 1MiB      // +33% for the mip chain
```

`planSizes` computes the total, and if it exceeds the budget multiplies `base` by 0.75 and retries, up to 8
times or until `base <= 192`.

Resolved plans (recomputed from the real table):

| preset | requested | planned base | estimate | hero A/N/O | interior A/N/O |
|---|---|---|---|---|---|
| low | 512 | 512 | 22.8 MB | 512/256/256 | 192/192/128 |
| medium, high | 1024 | 1024 | 90.6 MB | 1024/512/512 | 384/384/192 |
| ultra | 2048 | **1536** | 204.3 MB | 1536/768/768 | 576/576/320 |

1. **Snap to 64, not to powers of two.** WebGL2 filters and mips NPOT fine, which lets the planner land on
   1536 instead of falling 2048 → 1024 (textures.js:1835-1838).
2. **Normal and ORM do not need albedo's resolution.** Every tier halves or quarters them; atlases bake **no
   ORM at all** (fraction 0.0), because a cut-out card has no meaningful cavity AO.
3. **A cheap tier is how you add content without degrading the calibrated stuff.** Exterior-only at high is
   80.3 MB of the 92 MB budget (recomputed). Putting the 10 interior sets in at `mid` would have cost
   ~30 MB, blown the budget, and forced `base` to 768 — silently halving every already colour-calibrated
   hero set. The two interior tiers cost **10.3 MB** and fit (textures.js:1728-1734, 1820-1824). `ultra`'s
   budget went 200 → 224 for the same reason: at 200 the planner drops 1536 → 1152 (recomputed).

**Rule.** Tier by how close and how often the camera sees a surface, then let a planner shrink globally.
Never let a new content pass shrink an already-signed-off surface — give it its own cheaper tier.

**Fallback.** No renderer or no viable plan → `flatSet()` builds 4×4 constant textures with the same record
shape (textures.js:2092-2132); per-set bake failures are caught individually and substituted the same way
(textures.js:1930-1940). A GPU failure degrades to ugly, never to dead.

## 5. Packing: albedo / derived normal / ORM

**Normal from the analytic height field, not from the albedo.** `SURFACE_MAIN` runs a 3×3 Sobel over
`hgt()` sampled at ±`uTexel` (textures.js:187-192); because `hgt()` is evaluated live, the gradient is not
quantised by an 8-bit intermediate.

**Normal strength is physical** (textures.js:2026-2029):

```js
strength = Math.min(64, spec.relief * nSize / worldScale);   // relief in METRES
n = normalize(vec3(-gx * uNormal, -gy * uNormal, 1.0));
```

`relief` is peak-to-peak height in metres and `worldScale` is metres per uv tile, so `relief/worldScale` is
a slope and `× nSize` converts it to rise-per-texel. Resolution-independent except that a finer texel
resolves a steeper local slope. Worked values at base 1024: thatch `0.056·512/1.6 = 17.9`, cobble
`0.036·512/2.0 = 9.2`, plaster `0.008·512/2.5 = 1.6`, grass `0.024·512/4.6 = 2.7`.

**Rule.** Express relief in world units and derive the normal-map scalar. Hand-tuning a unitless "normal
strength" per material is how a library ends up with one surface at 5 mm of relief and its neighbour at 5 cm.

**ORM is glTF layout: r = AO, g = roughness, b = metalness** (textures.js:22-25). Verified in three:
- `roughnessmap_fragment.glsl.js:8` — `roughnessFactor *= texelRoughness.g`
- `metalnessmap_fragment.glsl.js:9` — `metalnessFactor *= texelMetalness.b`
- `aomap_fragment.glsl.js:5` — `texture2D(aoMap, vAoMapUv).r`

Because those are **multiplies**, `material.roughness`/`material.metalness` become multipliers and must
default to **1.0**, not 0.85/0.0 (materials.js:230-231); with no ORM map they revert to absolutes (:233-234).

> **`metalness: 1` with an ORM map is correct, not a bug.** `pewter` ships `rough: 1.0, metal: 1.0`
> (materials.js:154) because the ORM's blue channel is already 1 for the metal and 0 elsewhere. "Fixing" it
> to 0 turns every metal in the world into grey plastic. Comment it where the value lives.

**aoMap needs no uv1 in three ≥ r151.** `Texture.channel` defaults to 0 (Texture.js:118) and
`getChannel(0)` returns `'uv'` (WebGLPrograms.js:46-52, 273), so one ORM texture binds to all three slots
with no second uv set (materials.js:23). `aoMapIntensity` is the per-material dial (materials.js:228;
`lathPlaster` runs 0.65 because an interior bounce wall must not be pre-darkened, :144).

**Albedo is written as sRGB bytes.** The render target is `NoColorSpace`/`RGBAFormat`/`UnsignedByteType`
(textures.js:1903-1912); only the albedo DataTexture is tagged `SRGBColorSpace` (:2007, called with
`srgb=true` at :2024), which three uploads as `SRGB8_ALPHA8` (WebGLTextures.js:234). So **a GLSL value `x`
in `alb()` is linear `≈ x^2.2`** — every palette comment in the file is written in those terms
(textures.js:262-266, 436-439, 1088-1090). Normal and ORM stay `NoColorSpace`. `flipY = false` throughout
(:2008): GL rows come back bottom-up, which already equals v=0 at the bottom.

**The cheap AO.** `cavity()` (textures.js:168-179) averages `hgt()` at six fixed uv offsets (±0.007, ±0.005)
and compares to the centre: below local mean = occluded. The offsets are in **uv**, not texels — a
deliberate feature-scale blur that keeps the same physical radius at every resolution.

## 6. Tileability

Every noise primitive takes a lattice period and wraps it:

```glsl
float vnoise(vec2 x, vec2 per) { vec2 i = floor(x); ... hash12(mod(i, per)) ... }   // :72-80
vec4  voro (vec2 x, vec2 per, float jitter) { vec2 c = mod(n + g, per); ... }       // :123-153
```

Callers always pass a period equal to the frequency: `fbm(uv * 44.0, vec2(44.0))`. `fbm`/`fbm2`/`fbm6`
double period alongside frequency per octave (`q *= 2.0; pe *= 2.0`, :86-103), so all octaves wrap together.

Three hard constraints, all violated at some point in this file:

1. **Periods must be integers.** `FAMILY.wood`'s `wdSmoke` carries the scar: `mod(cell, 2.5)` "put a hard
   seam down every tile edge (measured: 6× the interior gradient)" (textures.js:534-537).
2. **Rotation does not tile; integer shear does.** `FAMILY.grass` (:824-835) builds its second blade layer
   as `uv.x*70 + uv.y*70` — leaning exactly one tile per tile. `FAMILY.strawLitter` (:1266-1270) builds its
   three crossing layers from `(u)`, `(u+v)`, `(u−v)` for the same reason, and needs an even straw count so
   the warp's `n/2` period stays whole.
3. **A per-row half-cell shift must repeat over an even number of rows**, or a running bond does not tile in
   V (textures.js:666-668, 1064-1066).

### What a visible repeat actually looks like

`FAMILY.grass` (:836-843) names it precisely: terrain lays this at one tile per few metres over a hillside
filling a quarter of the screen, and "the eye locks onto the per-Voronoi-cell tone and the fbm blotches and
sees a **hexagonal lattice marching across the hill**".

The fix was *not* more noise. It was **less low-frequency contrast**:
- the tonal fbm went `uv*4` → `uv*11` — a 1 m blotch at meadow scale became sub-metre (:863-865);
- the per-cell tone multiplier went `0.72..1.06` → `0.90..1.04` and is annotated `// was 0.72..1.06: THE hex lattice` (:873).

**Rule.** A repeat is visible when the pattern carries contrast at frequencies the mip chain does *not*
average away — anything at or below the tile period. Push variation up in frequency and narrow its range;
distance then flattens it to a mean instead of stamping it. Parallel fix: push the tile period past the
object — `lathPlaster` went `worldScale` 2.5 → 3.2 partly so the repeat exceeds most wall panels (:1758-1761).

## 7. The traps

### 7.1 Nyquist — the single most valuable lesson in the file

**Symptom.** Thatched roofs read as flat olive plastic. The albedo looked like straw; the surface didn't.

**Cause.** 220 straws per uv tile against a hero set's normal pass of `base/2 = 512` texels is **2.3 texels
per straw**. The 3×3 Sobel in `SURFACE_MAIN` straddles a whole fibre period, so the gradient cancels to
nearly flat and the normal map carries no fibre at all (textures.js:315-322, 1652-1656).

**Fix.** *Fewer, larger* features: 96 straws per tile = **5.3 texels each** (`uA: [96, 4, 0.45, 1.0]`,
textures.js:1657-1658).

**The general rule.** *A normal map cannot carry detail finer than ~5 texels. Pick feature frequency from
the resolution of the NORMAL pass, not from what looks right in the albedo.* The albedo pass is 2× the
normal pass in every tier, and it stays legible at 2 texels/feature — which is exactly why this failure is
invisible while you are authoring colour.

**Detection.** Compute `normalSize / featuresPerTile` before authoring, and write the arithmetic into the
SETS comment — the codebase now does, for every fine-featured set (`lathPlaster` 72 laths / 384 = 5.3
texels, :1753-1754; `floorBoard` ringFreq 26 / 384 = 14.8, :1743-1744).

**Companion move.** One frequency is never enough. Thatch runs **three** scales (:316-323): straws (96/tile,
the raking-light read up close), wads (16/tile ≈ 6 fibres each, "the scale that survives mip-mapping and
still reads at 20 m"), clump (6/tile, the coat's thickness). And relief is a *budget*: of the set's 0.056 m
a straw owns ~0.15. Giving the straws half the field — "the obvious thing to write" — makes every reed a
3 cm half-pipe and the normal saturates into noise (:349-355).

### 7.2 Domain warp

**Symptom.** Oak floorboards and ceiling joists read as swirling marble, worst under raking hearth light
1–3 m away.

**Cause.** `wdGrain` adds `warp * 2.8` plus `(fbm2-0.5) * 1.4` to the ring coordinate (textures.js:494-503)
— up to **~3.5 ring periods** of lateral wander. The grain therefore wanders across `3.5 / ringFreq` of the
board's width:

| ringFreq | wander | reads as |
|---|---|---|
| 6 (old `ceilingBeam`) | 58 % | dark swirling liquid |
| 9 (old `floorBoard`) | 39 % | marble |
| 22 (`ceilingBeam` now) | 16 % | quarter-sawn oak |
| 26 (`floorBoard` now) | 13 % | quarter-sawn oak |

**Fix.** Raise the feature frequency until the warp is a small fraction of it (textures.js:1738-1744,
1765-1767). The *frequency* moved, not the warp, because the frozen family string could not change — and
26 rings still clears the Nyquist floor from §7.1 at 14.8 texels.

**Rule.** Warp amplitude must be a small fraction of the feature period. Express it as the ratio
`warpAmplitude / featurePeriod` and keep it under ~0.2; authored in absolute units, nobody notices when
someone changes the frequency.

### 7.3 Voronoi damage terms saturate

**Symptom.** Interior lath-and-plaster walls read as crazed dried mud — "the single worst thing about the
interiors on first look".

**Cause.** `lpFlake` is a Voronoi field (`istep(0.34, 0.12, v.x)`, textures.js:1125-1129) scaled by
`uA.y`. At 0.45 the distemper scabs **join up** into a connected network of dark irregular polygons
(textures.js:1755-1759).

**Fix.** `flaking` 0.45 → **0.10**, and the cobweb term came down with it (`uA: [72, 0.10, 0.32, 0.16]`,
textures.js:1762-1763).

**Rule.** Cell-noise damage terms (flaking, crazing, cracking, spalling) have a percolation threshold: they
go from "scattered scabs" to "one crack network" over a narrow amplitude range, and the eye reads the
network as a completely different material. Sweep the amplitude, find where isolated cells first touch,
use half of it. Contrast `hsCraze` (textures.js:1189-1193), which is *deliberately* a network —
`istep(0.018, 0.004, v.z)` on the border distance, gated by a low-frequency mask so it only appears over
the hottest patches.

The vocabulary that makes this controllable: `voro()` returns distance to the feature point (`.x` — round
blobs) *and* distance to the cell **border** (`.z`, Inigo Quilez's second pass — joints of even width,
textures.js:115-153). Mortar, crazing and grout use `.z`; scabs, bubbles and clods use `.x`.

### 7.4 Atlas grid published in one place, read from another

Two separate real bugs, same root cause.

**(a) Hard-coded grid.** `buildings.js` hard-coded 4×2 against the 2×2 ivy atlas, so every ivy card sampled
**half a cluster** — "which is what made the ivy read as pale confetti". Fixed by importing the published
layout with a defensive fallback (buildings.js:70-85), via a *namespace* import, because a named import of
an export that does not exist yet is a link error that takes the whole module down (buildings.js:53-56).

**(b) Published on one object, read from another.** `materials.js` publishes the grid on
`material.userData.atlas` (materials.js:239); `vegetation.js`'s `atlasGrid()` probed
`texture.userData.atlasCols` — which nothing ever set — so **every atlas silently fell back to 2×2**
(vegetation.js:527-547). `leaf`, `flower` and `grassBlade` are all 4×2, so each card sampled two columns:
two half leaves, or a slice through two blossoms. Silent, because alpha coverage is per unit area, so the
measured luminance of the meadow did not move at all.

**Rule.** Publish the layout **once**, export it (`ATLAS_LAYOUT`, textures.js:2135-2144), import it. Never
hard-code a grid; never invent a second place to look. If you must have a fallback, make it `warn` — a
silent fallback to a plausible-looking grid is undetectable by any luminance or performance metric and will
survive several review rounds. Give each entry a note describing what a cell *contains*, since that sets
the card size a consumer should pick: `ivy: { cols: 2, rows: 2, note: 'all four cells are 5-9 leaf clusters; no up axis' }`.

### 7.5 Alpha cut-outs: `alphaTest`, never blending

Foliage config (materials.js:260-267): `side = DoubleSide`, `alphaTest = 0.42`, `transparent = false`,
`depthWrite = true`, `shadowSide = DoubleSide` ("cards must cast from both faces or half the canopy drops
its shadow").

- Alpha lives in **`map.a`**, not `alphaMap`: three reads `alphaMap` from the **green** channel —
  `alphamap_fragment.glsl.js:4`, `diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g`.
- `transparent: true` would force back-to-front sorting and full fill-rate cost for thousands of cards.
- Cut-outs get **no ORM** (materials.js:222-223).
- The shadow depth material needs the same `map` + `alphaTest` + `side`, or the shadow silhouette is a solid
  rectangle (vegetation.js:486-491).

**The failure that shipped, twice** — two meshes went out with `alphaTest = 0` and rendered as hard-edged
opaque rectangles. (1) The wildflower variant nulled its `map` and forced `alphaTest = 0` (defect V2,
vegetation.js:1276-1284); it also left the flower atlas's *normal* map driving a petal fan whose UVs were a
radial disc — wrong normals on top of a hard silhouette. (2) `asCutout`'s no-map path deliberately sets
`alphaTest = 0` (vegetation.js:519-523) — right as a last resort, same visual failure, so it warns.

**Detection.** `src/dev/audit.js:107-110` walks the live scene graph for
`m.transparent && m.alphaTest === 0 && /leaf|needle|ivy|flower|grass/i.test(m.name)`. Ship an auditor with
rules like this; it catches the class, not the instance.

**Atlas hygiene at low mips** needs two independent insets: in the *shader*, keep shapes off the cell border
(`needleAtlas` restricts needles to the middle of the stem "so none of them can reach past the top of the
atlas cell and bleed into its neighbour at low mips", textures.js:1495-1508; `grassAtlas` clamps blade
height below 1, :1600-1602); and in the *consumer*, inset the UV rect by a texel (`iu = 0.004 / cols`,
vegetation.js:379-390). Atlas sets bake `ClampToEdgeWrapping`, everything else `RepeatWrapping` (:2021).

## 8. Per-material recipes — the shape of the noise, not the constants

| surface | what makes it read | key mechanism |
|---|---|---|
| cobble setts | one **stone type hashed per Voronoi cell** (grey / brown / ochre / slate), never per pixel — otherwise it is "one sandy polygon field"; joints stay dark because "they are the whole reason the relief reads at all"; a polish term only on the crown | domain-warped `voro`, joint from `.z`; textures.js:236-304 |
| thatch | three nested scales (straw / wad / clump) + a per-course `lap` crease + jittered broken butt ends; albedo darkens in the gaps but the **relief itself lives only in the normal map** | :325-412 |
| plaster | broad trowel sweep at 3/tile + daub at 15 + grit at 80 + sparse aggregate specks, then rain streaks elongated in v (`uv.x*26, uv.y*3`) | :417-463 |
| timber / oak | ring bands from `fract()` of a warped coordinate, medullary **flecks** as a separate high-frequency term, per-board id offsetting phase and value | :481-606 |
| brick | row-offset lattice (not Voronoi — bricks are regular); `edge` distance perturbed by fine fbm so no two bricks chip alike; burnt headers get a *roughness drop* as well as a colour | :611-657 |
| rubble stone | Voronoi with running bond and high jitter (0.85) so no two stones share a shape; **two** lichen scales, both gated by the joint mask so lichen never grows in mortar | :662-709 |
| flagstone (interior) | flags wear **dished, not domed** (`fsDish` = distance from the edge) and the dished middle is the only part with any sheen | :1060-1112 |

Four cross-cutting notes that generalise:
- **Never bake curvature shading into albedo.** `roofTile`: contact shade under the lap only, "the barrel
  curvature belongs to the normal map, baking it into albedo would shade the tile twice" (:742-744).
- **Interiors invert the priorities.** With one window and one fire, albedo does most of the work:
  `lathPlaster` is deliberately the brightest set in the library (0.44-0.50 linear), `soot` the darkest
  (0.02-0.035), and interior AO is weak because "baked cavity shade on a nearly flat wall only ever makes an
  interior murkier" (:1045-1056, 1168-1170).
- **Paint-ready is a mode, not a colour.** `woodPainted` sets `uD.x = 1`, collapsing albedo to a mid-grey
  luminance so `material.color` can put any paint on it — tinting the brown map directly "would go muddy"
  (:583-588). Same for `flowerAtlas`: petals bake near-white, tinted per variant (materials.js:170-173).
- **`material.color` is a multiplier, so it can only ever darken.** `dirtPath` used `0xd8c8a8` meaning "pale
  warm earth" and measured `soil × 0.58 linear`, making the lanes the darkest ground in the village. A pale
  lane has to come from the *bake* (materials.js:119-123).

## 9. Material-library conventions worth copying

`materials.js` is the only file allowed to construct a `THREE.Material`.

- Unknown name → **magenta debug material** + one warning (materials.js:285-304): "a typo shows up as a
  screaming surface in the world instead of a thrown exception".
- All materials are built up front from the contract key list (materials.js:441-449) — free, because a
  material is not a shader until something renders with it, and a missing definition is then reported at
  load rather than the first time a wall faces the camera.
- `variant(name, opts)` is **cached by key**, so the same (name, opts) returns the same instance and variants
  still batch (materials.js:349-398).
- `applyRepeat` walks a fixed `MAP_SLOTS` list (materials.js:333) so a re-tile hits *every* map; missing one
  slot is a normal map that slides off its albedo.
- **`setEnvironment` assigns `envMap` per material rather than relying on `scene.environment`.** It has to:
  `WebGLRenderer.js:2693-2698` overwrites `envMapIntensity` with `scene.environmentIntensity` whenever
  `material.envMap === null`, so without an explicit envMap the per-material intensity knob does nothing.

## 10. Porting checklist for a new world

1. Decide `worldScale` (metres/tile) and `relief` (metres) per surface **first**; they drive texel density
   and normal strength, and nothing else then needs tuning.
2. Group surfaces into families by *shader shape*, not by name; aim for ≤ ~20 families however many sets.
3. Per family write only `hgt()` / `alb()` / `orm3()`; keep the pass switch, Sobel and cavity AO shared.
4. Before authoring fine detail compute `normalPassSize / featuresPerTile`; below ~5, cut the feature count.
   Write the arithmetic into the set's comment.
5. Every noise call gets an explicit integer period equal to its frequency. No rotations — integer shears.
6. Keep damage/flaking amplitudes below the percolation point.
7. Bake albedo as sRGB bytes, tag only the albedo `SRGBColorSpace`, write palette comments in linear
   (≈ value^2.2).
8. Pack ORM r/g/b = AO/rough/metal, bind one texture to three slots, default `roughness`/`metalness` to 1.0.
9. Read back into `DataTexture`s if anything downstream clones them; expect ~0.75× the GPU figure on the heap.
10. Export the atlas layout table; consumers import it; the fallback warns.
11. Ship a flat-texture fallback and a per-set try/catch, plus a scene-graph auditor with rules for blended
    foliage, debug materials and untextured fully-rough materials.

## 11. Unverified / stale in the source

- **The ~350 ms → ~4 s bake timings are not recorded anywhere in the repo.** `stats.bakeMs` is computed
  (textures.js:2053) but never persisted or asserted. The structural claim (compile cost scales with
  families) is supported by `material()` caching on `spec.family` (:1957-1958) and by the interiors pass
  adding 5 families / 10 sets / 30 passes; the wall-clock numbers are reported, not measured here.
- **`FAMILY.wood`'s header is stale.** It claims the interior sets moved to a `FAMILY.sawnOak` and that
  "every remaining set passes 0" for `uD.y/z/w` (textures.js:474-480). No `FAMILY.sawnOak` exists (that
  comment is its only occurrence), and `floorBoard` (`uD:[0,0.85,0,0]`) and `ceilingBeam`
  (`uD:[0,0,1.0,0.35]`) both use `family:'wood'` with those slots live (:1745-1771). The frozen-string
  discipline is sound; the comment is wrong — which is the lesson: an invariant asserted in a comment should
  be checkable by grep.
- **The header's "~15 shader compiles instead of ~60"** (textures.js:20) predates the interiors pass;
  recomputed: 24 programs, 106 passes, 37 sets.
- **`vegetation.js:502-509` claims `aoMap` needs a uv1 the cards lack.** Not true in three ≥ r151
  (`Texture.channel` defaults to 0 → `'uv'`). Harmless — foliage takes no ORM — but the reason is wrong.
- **`buildings.js:32-36` says UVs are authored "in metres"**, caveated "if the material library bakes its
  tiling into `texture.repeat` (as `worldScale` implies)". It does not — only explicit `def.repeat` entries
  are applied (materials.js:240); `terrain.js` divides explicitly (`uvScaleFor`, :628-636, with
  `TILE.grass = 4.60` matching the grass set's `worldScale` 4.6). Whether the building meshes land at the
  intended texel density is unverified: pin the metres-vs-metres/worldScale convention down explicitly.
- Draw calls, triangle counts and the 60 fps figure are outside these two files and unverified here.
