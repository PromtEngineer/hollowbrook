# Geometry recipes

Transferable construction rules extracted from a shipped procedural village (11 buildings,
~15 k lines of world geometry, zero external assets). Every rule is followed by the failure
that produced it and the file:line where the code lives. Read the rule; the village-specific
numbers are there so you can re-derive the ratio, not copy the metre.

Source files: `src/world/buildings.js` (2981), `interiors.js` (2101), `props.js` (2138),
`vegetation.js` (2122), `terrain.js` (1931). Library versions verified: three **0.185.1**,
`@dimforge/rapier3d-compat` **0.19.3**.

---

## 1. Silhouette over texture, for any soft material

**Rule.** A "soft" material — thatch, snow, moss, sand, fur, drifted dust — is sold by its
*edge*, not its surface. Give it (a) real thickness, (b) a rounded terminating roll where it
overhangs, (c) rounded returns at the cut edges, (d) sag under its own weight, (e) low-frequency
waviness in **both** surface directions. Texture is the last 10 %.

**Failure.** Three passes tried to fix "the roof reads as folded card" in the *material* (normal
strength, roughness, albedo). None worked: a zero-thickness slab with a knife eave has no
silhouette to fix. Detected by looking at the roofline against sky, not at the roof.

**Recipe** — `thatchPatch()`, `buildings.js:498-645`. Sweep a cross-section along a lip line:

| Shaping | Constant | Value | What it buys |
|---|---|---|---|
| slab thickness on the normal | `THATCH_T` | 0.46 m | the eave is a *band*, not a line |
| eave roll radius up-slope | `rc = T * 0.60` | 0.276 m | half-round bull-nose at the lip |
| swell (fattens above the slope plane) | `swell = T * 0.26` | 0.120 m | the lip bulges instead of tapering |
| bell-cast lift at the lip | `BELL_FRAC * eave` | 0.42 × eave | sprocketed curl, kills the straight rake |
| ridge sag at mid-span | `THATCH_SAG` | 0.085 m | pure −Y, shared by both faces |
| surface waviness | `wave` | 0.090 (tile: 0.010) | two octaves, ~3 m and ~1.5 m |
| gable verge roll radius | `clamp(T*0.5, 0.11, 0.23)` | 0.23 m | no thin edge anywhere |

Four constructions that make this work and generalise:

1. **Author the cross-section symmetric about its medial surface** (`buildings.js:546-554`):
   profile entry `j` mirrors `P-1-j` about `n = -T/2`. That single invariant is what lets
   `vergeRoll()` (`:694`) close a raked edge by lerping `ring[j] → midpoint(ring[j], ring[P-1-j])`
   while sweeping outward on a quarter-circle — a true bull-nose with no seam and no thin edge.
   Without symmetry you need a separate cap mesh and it will crack.
2. **Fade every deformation to zero at the shared seam** (`ridgeFade`, `:577`). Both roof faces
   run up to the same ridge line, so waviness, bell and swell are multiplied by
   `smoothstep(1-t, 0, 0.26)`. Sag is exempt because it is applied identically to both faces —
   a shared displacement never cracks a shared edge, an independent one always does.
3. **Confine the "fattening" term to one half of the section** (`topF`, `:580`, squared ramp on
   `(n+T)/T`). Applying swell to the whole section translates the slab; applying it to the top
   half only makes a roll.
4. **Bite the soft slab into whatever it lands on.** `ROOF_BITE = 0.22` (`:122`) drops the thatch
   underside 0.22 m *inside* the wall head so waviness can never open a light slot at the
   junction. Generalise: a wavy surface meeting a straight one needs an interpenetration budget
   ≥ its own peak-to-peak amplitude.

Weathering is vertex colour, streaked along the *gradient* direction: `NOISE(u*1.25, d*0.05)` —
high frequency across the slope, near-constant down it, gathered toward the eave (`:605-607`).
Same trick works for rust runs, snow melt lines, water staining.

---

## 2. Winding, and why you cannot code-review it

**Rule.** Any hand-indexed surface has a 50 % chance of being inside-out, and the compiler,
the type system and code review will all pass it. Decide winding **by construction** — from a
measurable property of the geometry — never by reasoning about vertex order in your head.

**Failure.** Every swept thatch surface was wound inside-out. `FrontSide` culled the top face,
so the visible surface was the underside 0.46 m below, shaded by an up-pointing normal. This is
why the roofs looked flat and thin. See `references/defect-catalogue.md` for the detection
recipe (down-rays, `FrontSide` vs `DoubleSide`, per-band up/down = 0/660).

**Three constructive fixes, in order of preference:**

| Situation | Tool | Where |
|---|---|---|
| closed shell (soup of triangles) | `orientClosed(pos)` — signed volume; flip all if negative | `buildings.js:653` |
| open strip, known outward reference | `pushTri(arr, a, b, c, ref)` — swaps b/c if `n·ref < 0` | `buildings.js:671` |
| swept grid with a computed frame | detect frame handedness, branch the index order | `buildings.js:529-542, 616-635` |

The swept-grid case is the subtle one. `thatchPatch` builds a `(sweep, slope, normal)` frame per
ring and forces `N` to point up — which on hip ends and back slopes **flips the handedness of
the frame** and with it the correct index order. So it accumulates a `hand` score
(`hand += N.y < 0 ? -1 : 1` *before* negating) and emits `(a, a+1, c)` or `(a, c, a+1)`
accordingly. Any "make the normal point up/outward" normalisation in a sweep is a winding hazard.

Verified against the library:

- `Ray.intersectTriangle(a,b,c,backfaceCulling,target)` returns `null` when `D·N > 0` and
  culling is on — `node_modules/three/src/math/Ray.js`.
- `Mesh.raycast` passes `backfaceCulling = (material.side === FrontSide)` —
  `src/objects/Mesh.js:419`. **This is what makes the FrontSide/DoubleSide ray diff a valid
  test.**
- `BufferGeometry.computeVertexNormals()` derives the face normal from `cross(pC-pB, pA-pB)` —
  `src/core/BufferGeometry.js`. So bad winding also produces **inverted normals**: even
  `DoubleSide` will not save you, it will just light the surface backwards.
- Default cull face is `gl.BACK`; `flipSided` only when `material.side === BackSide` —
  `src/renderers/webgl/WebGLState.js:760, 1239`.

---

## 3. Real depth beats textured depth

**Rule.** Where a style depends on one layer standing proud of another, model the offset in
geometry at 4-6 cm and let the sun do the work. A normal map fakes the shading but not the
occlusion, not the contact shadow, and not the parallax as you walk past.

**Recipe** — half-timber framing, `buildings.js:1535-1620`:

- Plaster is a real `ExtrudeGeometry` from a `Shape` with `shape.holes` for every opening,
  extruded `depth: WALL_T = 0.34` (`:1582`). Because the hole goes through the slab, window and
  door **reveals are the genuine wall thickness**, not a decal.
- The slab is translated back by `TIMBER_INSET = 0.055` (`faceN = -TIMBER_INSET`, `:1386`).
- Timber members are boxes of depth `TIMBER_T = 0.16` centred at `n = -TIMBER_T/2`, so their
  outer face sits at `n = 0` — **55 mm proud of the plaster**, self-shadowing in raking light.
- Everything downstream is derived from the sum, not re-guessed: the wall collider reaches
  `WALL_COL_IN = WALL_T + 0.09 = 0.43` inboard (`:126`), which is 35 mm proud of the plaster's
  inner face at `0.395` — the exact place `interiors.js` puts its lining.

Generalises to panel lines on a hull, mortar joints, lighting coves, grout, inset screens. The
transferable number is the **ratio**: proud offset ≈ 1/3 of member depth, ≈ 1/6 of slab thickness.

Companion trick — the framing is authored as **2D layout in a wall-local `(u, v, n)` frame**
carried by one orthonormal `Matrix4` per wall (`makeWall`, `:265`; `wallBox`, `:379`). Studs,
rails, braces, jambs and lintels then become interval arithmetic: `freeIntervals(min, max,
blocked)` (`:398`) subtracts opening spans from a run, and `blockedU`/`blockedV` (`:414`, `:423`)
project openings onto the axis. This is why the frame never crosses a window. Build the local
frame first; it is what makes complex facade detail tractable in any style.

---

## 4. Merging vs culling — pick per spatial extent

**Rule.** Merge until the merged mesh's bounding sphere stops being a useful cull unit, then
stop. The break-even is: *can the player ever have this whole volume off-screen?*

**The trade, measured on this project:**

| Strategy | Meshes | Cull granularity |
|---|---|---|
| one mesh per (building, material) | 139 building meshes (`post.js:505`) | per building — good |
| one mesh per (cluster, material), 3 clusters | "around 40 instead of 140" (`buildings.js:2841`) | per third-of-ring |
| one mesh per material, whole village | ~14 | none — never off-screen |

Reported for the full scene: 139 → 54 meshes, 458 → ~178 draw calls after clustering
*(reported by the project brief; the in-repo comment says "around 40 instead of 140" and
`STREAM_BRIEF.md:30` targets ≈450 draws / ≈2.2 M tris at `high` — treat the exact 54/178 as
unverified here).*

**Recipe** — `createBuildings`, `buildings.js:2809-2870`:

1. Author each object in **local space**, accumulate into a per-material bucket (`Builder`,
   `:238`).
2. Merge the buckets, then **bake the world matrix into the vertices** (`merged.applyMatrix4(world)`,
   `:2729`) instead of carrying a `Group` transform — otherwise the second, cross-object merge
   is impossible.
3. Group objects into contiguous **spatial** clusters (`CLUSTERS = 3`, index-based because
   `PLOTS` is authored going round the ring, so contiguous slices are contiguous arcs, `:2816`).
4. Merge again per `(cluster, material)`. Call `computeBoundingSphere()` on the result (`:2862`)
   — `Frustum.intersectsObject` falls back to `geometry.boundingSphere` for a plain `Mesh`
   (`three/src/math/Frustum.js`).

**Merging preconditions.** `mergeGeometries` requires all inputs indexed (or none) and
**identical attribute sets and counts**; on violation it returns `null` and `console.error`s — it
does **not** throw (`three/examples/jsm/utils/BufferGeometryUtils.js`), so `try/catch` alone is not
the safety net; check the result (`:1161`, `:2851`). Hence `normalizeGeo()` (`buildings.js:214`) at
*push* time: compute normals if absent, project UVs if absent, synthesise a white/tinted `color`
attribute, **delete every other attribute**, force an index, `clearGroups()`.

**Shadow flags force bucket splits.** `castShadow`/`receiveShadow` are per-mesh, so a decal that
must not cast needs its own bucket or it silently disables casting for everything sharing that
material — `props.js:328-333` keys buckets `${key}#${cast?'':'c'}${receive?'':'r'}`.

**aoMap needs a second UV set.** Alias it rather than generating one:
`merged.setAttribute('uv1', merged.getAttribute('uv'))` (`:2859`), otherwise AO samples at (0,0).

---

## 5. Instancing, per-instance colour, and the recycled carpet

**Per-instance colour.** `InstancedMesh.setColorAt` lazily allocates `instanceColor`
(`three/src/objects/InstancedMesh.js:320`). Two traps, both verified in r185:

- `color_vertex.glsl.js` does `vColor.rgb *= color;` under `USE_COLOR`. An **unbound `color`
  attribute reads as 0 on WebGL2 → every vertex black.** If you set `vertexColors: true` on a
  shared material, every geometry drawn with it must carry a white `color` attribute.
  `withInstanceColor()` (`vegetation.js:372`) does exactly that.
- `WebGLProgram.js:737` defines `USE_COLOR` in the **fragment** prefix when
  `vertexColors || instancingColor`, but line 567 defines it in the **vertex** prefix only for
  `vertexColors`. Pairing the flag with a real white attribute is the version-proof form.

**The recycled carpet** — `GrassCarpet`, `vegetation.js:1070-1192`. Ground cover that follows the
player with **zero allocation and zero per-frame CPU when standing still**:

- A fixed `n × n` lattice of instances in world cells of size `cell`. Slot index is
  `(iz*n + ix)`; the world cell it currently holds is `cx = s + mod(ix - s, n)` — a torus wrap,
  so a one-cell step rewrites one row and one column, not the whole lattice (`:1129-1140`).
- `refresh()` early-outs when the player has not crossed a cell boundary (`:1125`).
- Content is a **pure hash of the world cell**, not a stream: `_hash(cx, cz)` (`:1114`), five
  decorrelated draws for jitter/rotation/scale/tone/density. Same cell always grows the same
  tuft, so there is no popping when a slot is recycled.
- Rejected cells are **zero-scaled, not removed** (`:1177`). A zero-scale instance costs the same
  as a live one, so density becomes a free dial: raising `densityBias` stands more of the same
  lattice up without changing the buffer.
- `mesh.boundingSphere` is set manually and its centre moved with the lattice (`:1109`, `:1144`).
  `Frustum.intersectsObject` prefers `object.boundingSphere` when it exists
  (`three/src/math/Frustum.js`) — an `InstancedMesh` has that field, so you must maintain it.
- `setUsage(THREE.DynamicDrawUsage)` on both `instanceMatrix` and `instanceColor` (`:1100`, `:1108`).

Measured tuning: the far carpet went 9 216 tufts over a 58 m disc (147 k tris, the largest item
in the module) → 68² tufts at one per 2.9 m², scaled ~1.45×, `segs: 1` blades at 8 tris instead of
16 → **37 k tris**, visually equivalent past 13 m (`vegetation.js:1713-1726`). Rule: past the
distance where a feature is under ~2 px, trade count for size — the ink integral is what reads.

**When to turn culling off.** `props.js:400` sets `frustumCulled = false` on village-wide
`InstancedMesh`es: the extent covers the whole playable area, so the test can only ever return
true and you pay for it every frame.

**Partial-buffer upload.** When only the first `k` instances are live, set `mesh.count = k` and
upload just that range (`publish()`, `vegetation.js:1902`):
`a.clearUpdateRanges(); a.addUpdateRange(0, count*16); a.needsUpdate = true;`
Verified: `BufferAttribute.updateRanges` / `addUpdateRange` / `clearUpdateRanges` exist in
`three/src/core/BufferAttribute.js:119, 181, 190`. **Trap:** an empty range list means "upload
everything", so return early at `count === 0` rather than publishing an empty list.

---

## 6. Three-tier LOD, and the strict-subset trick

**Failure.** A two-tier system (real geometry near, crossed cards far) put the crossover at 80 m.
Measured from the spawn eye, that is exactly where the treeline behind the rooflines sits: a
raycast into what looked like a spiky bare tree resolved to `conifer-billboard` at **79.5 m**,
and the near bucket held **19 trees against 3 393 billboards** — essentially the whole visible
forest was cards. (`vegetation.js:52-66`. Detection: name your meshes, raycast from the eye, read
back `intersect.object.name`. Do this from the *actual* viewpoints, not from an orbit camera.)

**Rule.** Put the LOD crossovers where the *frame* is, not where the maths is convenient. Find
the distance band that occupies most pixels — usually the mid-ground behind your foreground
silhouettes — and give that band real geometry.

**Tiers** (`vegetation.js:68-70`): full ≤ 60 m (`TREE_NEAR_M`, trunk + foliage, conifer 2 324 tris);
mid 60-180 m (`TREE_MID_M`, same tree with cards decimated and no trunk, 575 tris); billboard
180-760 m (`TREE_FAR_CULL_M`, crossed alpha cards, ~36 tris).

**The strict-subset trick** (`buildConifer`, `:558-566`; built from the same fork at `:1336-1348`):

1. The mid variant is generated from **the same seeded generator** (`speciesRng.fork('conifer0')`
   twice — `fork` is pure, so both calls return an identical stream).
2. Every rng call happens in the **same order and the same number of times** in both passes. The
   card randoms are hoisted out of the emit call *specifically* so the `continue` that skips a
   card still consumes its randoms (`:628-635`). Same skeleton: height, lean, whorl heights,
   branch angles.
3. Only the **emission** is skipped — mid keeps card 0 on every branch and card 1 on alternate
   branches, 1.5 of 3, at `segs: 1` instead of 2.
4. Survivors are widened to hold the silhouette: `cardGain = 1.36` (conifer), `1.36 × 1.16`
   (broadleaf, `:723`).

Result: the mid geometry is a strict subset of the full tree's cards at the same transform, so
the 60 m crossover **changes density, not shape** — no pop. Both tiers share `nearMat`, the same
baked matrix array (`:1487-1490`).

**Supporting decisions worth stealing:**

- The mid tier does **not** cast shadows (`cast: false`, `:1461-1470`). Its band starts at 60 m,
  shadow distance is 95 m, and three does not cull instances individually — a casting mid mesh
  re-submits ~170 trees into the shadow map to light a handful.
- Re-bucketing is **allocation-free**: hand-copy 16 floats per matrix rather than `subarray()`,
  which allocates a view per tree (`:1936`).
- Deep-field decimation is keyed on the object's distance **from the world centre**, not from the
  player (`TREE_DECIM_R = 420`, keep 0.65, widen 1.50 → ink-neutral at 0.98). A player-relative
  test makes objects blink along the boundary on every refresh (`:71-80`).
- Crossover radii scale with the quality preset's cull distance but clamp to the authored values:
  `nearDist = min(60, cull*0.42)`, `midDist = min(180, cull*0.95)` (`:1981`).
- Billboards: card **overlap** is the design variable, not card count. With an atlas cell only
  ~10 % opaque above a 0.42 alpha test, crossed-card coverage = `cards × (bandHeight / totalSpan)`.
  16 cards at 0.40/1.08 ≈ 5.9 crossed ≈ 46 % read as a bottle brush; 18 at 0.50/1.07 ≈ 8.4 ≈ 59 %
  reads as a mass, for +4 triangles (`:824-838`). Also offset alternate planes sideways by
  `w * 0.12` — cards sharing an axis pile their transparent margins and open a hole down the
  middle of the silhouette (`:868`).

---

## 7. Interiors: publish the shell's own interior

**Rule.** The module that *cuts* an opening publishes its result. A consumer that re-derives
wall positions from the same layout data will drift the moment either side changes, and the
drift shows up as a lining floating in mid-air or a stair through a chimney.

**Recipe** — `buildings.js` fills a `RoomSpec[]` while building the shell (`:2660-2699`) and
returns it on `chunk.interiors`; `interiors.js` reads it and *never re-derives a wall*
(`interiors.js:12-14`). Contract in `contracts.js:185-215`:

- `RoomSpec`: `plotId, storey, floorY, ceilingY, width, depth, centre[3], rotation,
  wallThickness, openings[], openToRoof, ridgeY?, use`. `width`/`depth` are **clear interior
  dimensions to the inner face of the shell's own plaster** — say which face, always.
- `OpeningSpec`: `kind ('door'|'window'), wall, lx, sillY, width, height, centreWorld[3],
  inward[3], primary?`. Publishing `centreWorld` *and* `inward` *and* the wall-local `lx` means
  the consumer never has to reconstruct the frame.

**Insets, so nothing z-fights** (`interiors.js:55-56`): `GAP = 0.02` of air, then `LIN_T = 0.035`
of lining. Never let two coplanar faces point the same way — around every opening the lining
returns *back to the shell face* (`n` from `0` to `nIn`) over a `jw = 0.06` strip while flat panels
run `GAP → nIn`, so panel and return are never coplanar (`:695-701`). Floors tuck **under** the
lining (`hw = room.width/2 - GAP*0.5`, `:1057`); thresholds sit **6 mm proud** of the finished floor
rather than flush, which is nothing against a 0.45 m autostep (`:1113-1115`).

**A box collider cannot have a hole.** Rapier's `Cuboid` is a convex primitive
(`@dimforge/rapier3d-compat/geometry/shape.d.ts:72, 189`; only `TriMesh`, `Polyline` and `Voxels`
are non-convex). So a floor with a stairwell emits **four box colliders around the opening**
(`buildFloor`, `interiors.js:1051-1090`) — left slab, right slab, near strip, far strip — each
0.25 m thick so nothing can tunnel. Same pattern for a wall with a doorway: break the collider
into spans (`freeSpans(-half, half, doorSpans)`, `:756`).

**Publish keep-out volumes.** Before this existed, the furnishing module was *guessing* where
stairs and chimney breasts were — measured `roomsWithGuessedStair 14, roomsWithPublishedKeepOuts 0`,
and furniture went through flights of stairs (`interiors.js:2060-2066`). Now every room publishes
boxes tagged `'stair' | 'stairwell' | 'hearth' | 'threshold' | 'door-swing'` (`keep()`, `:2067`),
each carrying the **same box in three shapes** — world `centre`+`halfExtents`+`rotation`,
room-local `x0/z0/x1/z1`, and `lx/lz/halfW/halfD` — so it does not matter which convention the
consumer was written against. Redundant encoding is cheap; a mismatched convention is a bug you
find by screenshot.

Two derived rules: the swept volume of a flight includes its **foot landing** (`+0.78 m` past the
hole, `:2073`); the stairwell keep-out includes the **landing you step off onto** (`-0.72 m`,
`:2081`).

**Visibility.** Toggle interiors **per building, not per room** — you can see the upper storey up
the stairwell, so hiding just the room you stand in opens a hole in the ceiling
(`interiors.js:30-36`). Hysteresis: show at 7.5 m, hide at 10.5 m, cap 3 visible, re-check every
0.11 s or 0.4 m of movement (`:98-103`).

---

## 8. Deterministic generation and fork discipline

**Rule.** Every generator takes an `Rng` instance, never `Math.random`. A defect you cannot
reproduce is a defect you cannot fix (`util/rng.js:2-4`).

**The discipline that matters is forking.** `Rng.fork(tag)` returns
`new Rng((this.seed ^ hashSeed(tag)) >>> 0)` (`rng.js:53`) — a *pure function of parent seed and
tag*, not a draw from the parent stream. Consequences:

- Adding a feature that consumes randomness inside `fork('thatch')` cannot reshuffle
  `fork('interior')`. Without forking, one extra `rng.next()` anywhere re-rolls the entire village
  downstream, and every screenshot in your review loop becomes non-comparable.
- Because `fork` is pure, you can call it **twice with the same tag** to get two identical
  streams — this is the mechanism behind the LOD strict-subset trick (§6) and it is only safe
  because forking does not advance the parent.

Practice from the codebase:

- One `Rng` seeded per object: `new Rng(plot.seed)` (`buildings.js:1255`), then
  `rng.fork('interior')` (`:1260`), `rng.fork('thatch')` (`:1853`).
- Module-level named roots: `new Rng('hollowbrook-props')` (`props.js:1695`),
  `new Rng('hollowbrook-vegetation')` (`vegetation.js:1352`) — string seeds go through FNV-1a
  (`rng.js:20`), so the tag *is* the seed and reordering modules changes nothing.
- Sub-tags per feature and per index: `fork('well')`, `fork(\`stall${idx}\`)`,
  `fork(\`plot${plot.seed}\`)` (`props.js:614, 1015, 1924`).
- Spatial content that must be stable under recycling uses a **positional hash**, not a stream —
  `GrassCarpet._hash(cx, cz)` (§5).
- Avoid clumping when you draw one value per object: `plot.seed * 0.6180339887498949 % 1` jittered
  by the Rng, because a plain `rng.next()` per plot put 8 of 11 roofs in the same half of the age
  range — "a village of identically drab roofs" (`buildings.js:1854-1857`). Golden-ratio (additive
  recurrence) sequences are low-discrepancy; independent uniforms are not.

---

## 9. Ground: authored flat core, blended surround, analytic normals

**Rule for a hub world.** Make the playable core **exactly flat and exactly authored**, then blend
to procedural relief over a band wide enough that no single frame contains the whole transition.
Everything else places objects by dropping them on the flat plane, which removes an entire class
of "prop floating / prop buried" defects.

`terrain.js:5-8, 43-47`: ground is **exactly y = 0** inside `VILLAGE_R = 46`, blends to full hill
strength by `BLEND_R = 70` via `shapedHeight(x,z) * smooth01((r - 46) / (70 - 46))`
(`heightAt`, `:340-341`). Detailed mesh ends at `MEADOW_R = 262`, coarse skirt at `FAR_R = 1500`,
invisible bound ring at `BOUND_R = 300`.

**Layer stack, ordered so nothing can z-fight** (`terrain.js:11-24`) — separate co-located
surfaces with *real geometry*, not depth bias:

```
meadow      y = heightAt + gutter        opaque, depth-writes
wear decal  y = meadow + 0.012           transparent, depthWrite:false
roads       y = heightAt + 0.040 crown   opaque, edges dive to -0.090
plaza       y = 0.013 ± 0.010            opaque, own annulus hole
```

The meadow's inner ring is the plaza polygon pulled 0.25 m inward and dropped 30 mm, so the
cobbles always overlap it: no gap *and* no coincident surfaces. Roads bury their outer lanes
90 mm **below** the terrain so their outline is a real intersection curve rather than an alpha
edge. Generalise: prefer a buried intersection to a coplanar seam anywhere two authored surfaces
meet.

**Analytic normals, always.** `writeNormalAt()` (`:353-367`) samples `heightAt(x±e, z)` /
`heightAt(x, z±e)` and writes the normal directly, with `e ≈ 0.85 × local ring spacing`. Reason:
`computeVertexNormals()` makes shading resolution a function of *mesh* resolution, so a coarse
band shades every quad as a flat facet — the visible triangular faceting on the hills
(`:1062-1064`). Same normals are reused by the wear decal (`:1247`) and the far skirt (`:1330`)
so the layers shade identically. **Rule: if you have an analytic height field, never let the
triangles decide the normals.**

Mesh density is authored against a *sag bound*, not a triangle budget: geometric ring growth
capped at 2.4 m out to 135 m (roads live there and a corridor cut into a hillside is the one
feature a coarse mesh cannot follow), 3.6 m to 180 m, 4.6 m beyond (`meadowRings`, `:971-987`).
Lifting the cap entirely past 180 m produced 10 m rings — "long thin triangles that faceted
visibly"; 4.6 m costs ~13 extra rings (~7 k tris) and reads smooth. Angular divisions `288 × detail`
keep cells square at 230 m (5.0 m arc vs 4.6 m ring step).

**Vertex-colour wear.** Bare-earth fields (road proximity, building proximity, plaza rim, road-mouth
aprons) are combined with `unionWear()` (`:522-533`), **not** `Math.max`:

```js
u = 1 - (1-a)(1-b)(1-c)(1-d);   m = max(a,b,c,d);   return m + (u - m) * 0.70;
```

`max` left a pale wedge of untouched turf where three surfaces terminate within a few metres
(road edge / plaza arc / meadow). Probabilistic union pulled 70 % back toward the max makes
overlaps darken without turning the whole village to mud. **Rule: overlapping reasons for the
same effect should compound, and a soft-union with a pullback is the knob.**

Large-scale tonal variation needs **octaves the player can see across**, and they must be *added*,
not averaged. Inside the flat core `n·l` is constant, so one 22 m octave was the only variation and
the meadow read as painted. Adding ~60 m and ~9 m octaves at full weight
(`0.5 + 0.55(broad-0.5) + 1.00(patch-0.5) + 0.55(mid-0.5)`) raised large-scale vertex-luminance sd
0.0221 → 0.0258 at 12 m cells and 0.0208 → 0.0247 at 24 m cells, mean albedo effectively unmoved
(0.7260 → 0.7235) because each octave's mean is exactly 0.5 and everything downstream is linear in
it (`:1085-1108`). **Averaging measurably reduced the spread** — the 60 m octave is near-constant
over the 110 m the player can see, so it contributes nothing locally while diluting the one that
did.

---

## 10. Small recipes worth keeping

| Recipe | Where | Rule |
|---|---|---|
| UVs in metres | `applyLocalUV`, `buildings.js:192` | dominant-axis planar projection of local position; adjacent merged pieces then share a continuous texture and nothing stretches. Pair with a `worldScale` on the texture set. |
| Swept tube | `tubeAlong`, `buildings.js:731` | ridge liggers, hip rolls, bracket arms, cables, pipes. Frame is `(cross(up,tan), cross(tan,nx), tan)`; index `(a,b,c)` — right-handed frame, normals out. |
| Extrude with holes | `buildings.js:1536-1582` | `Shape` + `shape.holes.push(Path)`, `bevelEnabled:false, steps:1`. Wind the outer contour CCW and `ExtrudeGeometry` normalises for you (`:2125`). |
| Atlas cell cropping | `ivyLeafGeometry`, `buildings.js:2748` | **Read the grid from the texture library, never hard-code it.** A 4×2 crop against a 2×2 atlas samples half a cluster per card — the ivy read as pale confetti. Inset by `pad = 0.03` of a cell to keep the bilinear filter off the seam. |
| Radial keep-outs for scatter | `KEEP_OUT_R`, `props.js:426` | hero props register a radius (cart 2.5 m, trough 1.75, bench 1.25); scatter passes test against it. Cheaper than collision and deterministic. |
| Cubed lerp for scatter scale | `vegetation.js:1186` | `lerp(min, max, r³)` — most instances near the low end, a few proud. A linear ramp made every tuft the same height ("neon spikes"). |
| Fail-soft material resolution | `makeMatResolver`, `buildings.js:165` | try `variant()`, fall back to `get()`, fall back to magenta debug + one warning. A missing material must not take the module down. |
| Namespace import for cross-stream exports | `buildings.js:44-53` | a named import of an export that does not exist yet is a **link error that kills the whole module**; `import * as X` + `X?.THING ?? fallback` degrades. Matters when modules are built in parallel. |

---

## Verified vs unverified

**Verified in `node_modules` (three 0.185.1 / rapier3d-compat 0.19.3):** `mergeGeometries` returns
`null` + `console.error` on attribute mismatch and requires uniform indexing; `Mesh.raycast` passes
`backfaceCulling = (material.side === FrontSide)` to `Ray.intersectTriangle`;
`computeVertexNormals` takes the face normal from winding; `Frustum.intersectsObject` prefers
`object.boundingSphere` over `geometry.boundingSphere`; `InstancedMesh.setColorAt` lazily allocates
`instanceColor`; `color_vertex.glsl` does `vColor.rgb *= color` under `USE_COLOR`, and
`WebGLProgram` defines `USE_COLOR` in the fragment prefix for `instancingColor` but not in the
vertex prefix; `BufferAttribute.addUpdateRange` / `clearUpdateRanges` exist; Rapier `Cuboid` is a
convex shape type with no hole representation.

**Reported, not re-measured here:** the 139 → 54 mesh and 458 → ~178 draw-call figures (repo says
"139 building meshes" and "around 40 instead of 140"; `STREAM_BRIEF.md:30` targets ≈450 draws /
≈2.2 M tris at `high`); 60 fps; the per-band `up/down = 0/660` winding audit — that one-off script
is not in `src/`, though `src/dev/audit.js` is the surviving harness of its class.
