# Measured baseline

Every number here was measured in a browser during the build, not estimated. It exists because a
reviewer of this skill correctly objected that the performance claims had no committed source —
they lived only in a chat transcript. Anything not measured is marked as such.

**Hardware and settings.** Apple M2 Max, Chrome, canvas 1033×666 to 1280×800 at pixel ratio 1.0–1.5,
quality preset `high`. A mid-range discrete GPU or an integrated GPU will differ substantially;
treat these as *relative* figures and re-measure on your target.

**Method.** `renderer.info` reset before a single render (it accumulates otherwise — see
`verification.md`); simulation stepped ~180 frames to let culling, light pools and shadow extents
settle before sampling, because mid-transition readings are wildly wrong (one read 6.2 M triangles
where the settled value was 2.2 M).

---

## Frame cost, by viewpoint

| Viewpoint | Draw calls | Triangles | Notes |
|---|---:|---:|---|
| Plaza centre | 512 | 2.42 M | typical outdoor |
| At the well | 502 | 2.41 M | typical outdoor |
| Spawn, looking into the village | 718 | 3.99 M | worst outdoor — whole village + treeline in frustum |
| Interior, inn taproom | 634 | 3.56 M | shadow extent shrinks indoors, offsetting interior geometry |
| Interior doorway, looking out | 768 | 3.86 M | worst case: a full interior *and* the exterior |

Unique scene triangles: **828 k**. Rendered triangles are ~3× that because the shadow and GTAO
passes re-render the scene — which is why **object count**, not mesh complexity, is the lever.

Frame rate held **60 fps** at ceiling pixel ratio with adaptive resolution enabled, measured over
a 4-second `requestAnimationFrame` sample.

## Attribution, by subsystem

Measured by toggling each group's visibility and re-measuring per frame:

| Group | Draw calls | Triangles | Objects |
|---|---:|---:|---|
| Buildings | 458 | 1.02 M | 139 meshes + 1 instanced |
| Props | 269 | 254 k | 76 meshes + 8 instanced |
| Vegetation | 80 | 741 k | 25 instanced |
| Terrain | 14 | 187 k | 7 meshes |
| Sky | 2 | 4 k | 1 mesh |

This is the measurement that produced the ~3.3 draw calls per object figure, and it directed the
optimisation at merging (buildings 139 meshes → 54, draws 458 → ~178) rather than at geometry
detail.

## Build time

| Stage | Measured |
|---|---:|
| Texture bake, 27 sets, base 1024 | **350 ms** |
| Texture bake, 37 sets / +5 GLSL families — cold shader cache | **4.0 s** (observed twice) |
| Texture bake, 37 sets — warm shader cache, later session | **460 ms** |
| Interiors build (18 rooms), first version | 905 ms |
| Interiors build, after optimisation | **179–249 ms** |
| Furnishings build | 286 ms |
| Vegetation build | 77–98 ms |

The bake jump is the headline: the new interior textures were *smaller* (384/256 vs 1024) than the
existing ones, and the cost was **shader compilation** — one synchronous compile per GLSL family.
Count programs, not pixels.

The 4.0 s / 460 ms spread for the *same* 37 sets is itself the lesson: browsers cache compiled
programs, so a bake time measured after several reloads is a warm-cache number and understates
what a first-time visitor pays. **Measure cold**, in a fresh profile, before believing a load-time
figure. The compile-bound diagnosis stands either way — the pixel work did not change.

## Physics

| | |
|---|---|
| Static colliders | 998 (347 world + 217 shell + 280 interior + 220 furniture) |
| Dynamic bodies | 89 (47 exterior + 42 interior grabbables) |
| Step time | **0.16 ms** (EMA, fixed 1/60) |

Physics was never close to being the bottleneck at this scale.

## Rendering calibration

| | Measured |
|---|---|
| Ground-half mean screen luminance, spawn | **0.451–0.457** (target band 0.42–0.48) |
| Sunlit plaza cobble | **0.586** sRGB (target 0.50–0.62) |
| Worst-case sunlit plaster peak | 233 per channel (no clipping) |
| Clipped pixels, outdoors | **0.00%** |
| Clipped pixels, interior with lit hearth | 2.34% |

Before calibration the ground read **0.243** with cobble albedo at 0.099 linear. See
`material-calibration.md`.

## Behavioural verification

Measured by driving the character controller with synthetic key events at a fixed 1/60 step:

| Check | Result |
|---|---|
| Walk speed | 4.63 m in 1.5 s = **3.09 m/s** vs 3.0 configured |
| Jump apex | **0.98 m** vs `v²/2g` = 0.98 m for 6.2 m/s at 19.6 m/s² |
| Wall collision | stopped at 5.39 m from plot centre (half-depth 4.75 + radius + skin) — no penetration |
| Buildings enterable | **11 / 11** |
| Staircases climbable | **7 / 7** |
| Interior objects grabbable | 5 / 6 attempted (the miss was test aim, not a defect) |

## Explicitly not measured

- Performance on any GPU other than an M2 Max. **The 60 fps figure does not transfer.**
- Whether the first-bake hitch is really hidden by `PMREMGenerator.compileCubemapShader()`
  (~10 ms is a code comment, not a measurement).
- GTAO memory footprint (~50 MB at 1080p is an estimate in a comment).
- The quality cost of running AA before `OutputPass` rather than after.
- Cobble specular clipping in a low ground-facing close-up: a 24-view sweep found no clipped
  pixel after the fix, and the baked roughness floor provably rose, but the original 0.19%
  figure was never reproduced at the same window size.
