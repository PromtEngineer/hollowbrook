# Defect catalogue

Every one of these is a real defect from a shipped procedural Three.js world, with the root
cause and the detection method. They are grouped by how they present, because that is how you
will meet them. Most are cheap to avoid and expensive to find.

The pattern worth internalising: **the visible symptom almost never names the responsible
subsystem.** "The scene is dark" was albedo. "The roof is flat" was winding. "The player can't
get in" was a step-height parameter, not the door.

---

## Presents as: a surface looks flat or wrong

### Inside-out winding — the most instructive bug in the project
Every swept roof surface was wound inside-out. The renderer backface-culled the top, so what you
saw was the *underside*, 0.46 m lower and lit by an up-pointing normal. The roofs read as thin
folded card no matter how the material was tuned, and three separate passes tried to fix it in
the texture.

- **Detect:** cast rays down over the surface and compare hits with `FrontSide` against
  `DoubleSide`. A per-band audit gave top-surface up/down = 0/660 — conclusive.
- **Also check:** closed shells (signed volume), swept tubes (index order `a,b,c` vs `a,c,b`),
  and any open band you generate by hand.
- **Rule:** if a surface looks flat or oddly lit and the material is fine, suspect winding before
  you touch the material again.

### Normal-map detail below the sampling rate cancels to flat
A thatch texture with 220 straw fibres per tile was baked into a 512-texel normal pass. The
Sobel operator straddled a whole fibre period, the gradient cancelled, and the mean facet tilt
came out near zero — a beautifully authored pattern rendering as smooth plastic.

- **Fix:** *fewer, larger* features. 96 fibres/tile (~5 texels each) raised mean facet tilt to
  0.591, by far the strongest relief in the library.
- **Rule:** a normal map cannot carry detail finer than about 5 texels. Choose feature frequency
  from the **normal pass resolution**, not from what looks good in the albedo.

### Domain warp swamping the feature period
Wood grain read as swirling marble. The grain shader added ~3.5 ring periods of domain warp to
the ring coordinate; at a ring frequency of 6 the grain wandered across 58% of the beam's width.

- **Fix:** raise the feature frequency so warp is a small fraction of a period (6 → 22 took the
  wander to 16%). Same fix on the floor (9 → 26).
- **Rule:** warp amplitude must be small *relative to the feature period*, not small in absolute
  terms.

### Cell noise saturating into a network
Lath plaster read as crazed dried mud. A Voronoi "distemper flaking" term at strength 0.45 had
joined up into a continuous crack network.

- **Rule:** cell-noise damage terms go from "sparse scabs" to "cracked mud" over a very narrow
  range. Keep them low and check at the scale the surface is actually viewed from.

### Visible tiling
A hill showed an obvious repeating hex pattern; a wall panel showed the same blob cluster several
times. Interior surfaces are viewed from 1–3 m — a `worldScale` tuned for exterior distance
repeats within a single panel.

- **Rule:** pick tile scale so the repeat is larger than the surface it lands on, and break up
  low-frequency structure with a second, much lower-frequency layer.

---

## Presents as: the scene is dark, milky, or flat

### It's the albedo, not the lights
Ground luminance 0.243 with everything correct. Disabling shadows moved it 4%; AO 1%; forcing
`metalness = 0` on 33 materials moved it 0%. Exposure was the only large lever — which means the
surfaces were too dark. Baked cobble measured **0.099 linear** where weathered stone is 0.24–0.34.

- **Fix:** recalibrate albedo (see `material-calibration.md`), and **lower** the too-bright
  surfaces at the same time — plaster came down 0.551 → 0.421 so the cobbles could come up
  without the frame going milky.
- **Trap:** raising `toneMappingExposure` "fixes" the dark thing and clips the bright thing.

### `metalness: 1` with an ORM map is not a bug
A material reading `metalness: 1` looks alarming. With a packed ORM texture it is the correct
glTF-style multiplier and the blue channel carries the real value. Forcing it to 0 changed
nothing — verify before "fixing".

### Flat ambient light kills the image
A flat `AmbientLight` at 0.52, added to lift dark interiors, applies equally to every normal and
removes all shape. Fold interior fill into a **tilted hemisphere light** instead — three derives
a `HemisphereLight`'s axis from its world position, which is a free directional handle.

### Sun elevation is a composition control
Shadow length is `1 / tan(elevation)`. At 28° shadows were 1.88× caster height and the buildings
on one side threw the entire square into shade; at 36° they are 1.38× and the centre stays lit.
Bearing decides *which* façades are lit at all — moving to azimuth 57° took the hero building's
front from N·L 0.40 to 0.88.

- **Rule:** choose bearing from which walls must be sunlit, elevation from how much shadow the
  ground should carry. Not from "golden hour".

### Parameters feeding a solver have a feasible range
Raising the sun elevation would have silently failed: the daily-path pole solve had **no solution
above 33°** at the shipped radius, and would have fallen back to a baked axis, destroying the
composition without an error. Check a solver's feasible range before changing its inputs.

---

## Presents as: something glows, halos, or looks like a stray polygon

### Bloom threshold is relative to your sky, not to 1.0
Threshold 1.25 sat *below* the procedural sky's HDR radiance, so the whole sky was over
threshold. three's `LuminosityHighPassShader` passes the **whole texel** once it is over, so
every roofline got a halo the width of the blur kernel.

- **Fix:** threshold ~2.6 with a soft knee (`smoothWidth`), radius down. Measure your sky's
  actual radiance first.

### Emissive/foliage cards with `alphaTest: 0`
Shipped twice, in two different files: an ember bed and a wildflower field, both rendering as
hard-edged opaque rectangles because the cut-out alpha was never applied.

- **Detect:** audit every material for `transparent === false && alphaTest === 0` on anything
  card-shaped. Add it to the automated pass.
- **Fix for a soft glow with no texture:** a fan geometry with a 4-component colour attribute
  whose rim alpha is 0. With `transparent: true`, final alpha scales emissive too, so the glow
  dies out instead of stopping at an edge.

### An over-bright light reads as a white blob, not as fire
A hearth point light at intensity 20.63 put nearby stone at 5.5 linear — solid white, no flame
structure. ACES clips past ~2.87 linear.

- **Rule:** size a light against the scene's own scale (here: sun 4.39, candle ~0.3) and against
  where the tone curve clips, not against "bright".

---

## Presents as: the player can't move somewhere

### Autostep `minWidth`, not the obstruction
4 of 11 buildings were unenterable. Approach rays were clear at every height *and* every lateral
offset, and the player stalled 0.92 m short. **Jumping always worked.** That single differential
isolated it: not an obstruction, but a step the controller refused to climb — Rapier's autostep
`minWidth` of 0.25 m was wider than a worn doorstep's top tread. 0.12 m fixed both.

- **Rule:** when a character stalls somewhere geometrically clear, test **jump vs walk** before
  investigating geometry.

### A hinged door is not reliably shoveable
Doors as revolute-jointed dynamic bodies feel great and do not work: a kinematic capsule cannot
dependably push a ~35 kg leaf. Worse, the three narrow doors that *did* work only worked because
their leaves happened to hang open.

- **Geometry trap:** "ajar" intuitively means 60°, but a 1.08 m leaf at 60° in a 1.20 m opening
  leaves a **0.64 m** straight corridor — exactly the capsule diameter. Doors must rest at ~85°.
- **Rule:** verify a clear corridor by measurement, not by nominal opening width.

### Ray height matters
Rays cast from *floor* height reported a doorway "completely clear" while a threshold step 8 cm
below was blocking the capsule. Cast from the heights the character actually occupies.

---

## Presents as: it runs badly, or the resolution collapses

### A slow-frame threshold below the vsync period
Adaptive resolution collapsed to its floor on a machine holding a steady 60 fps. The threshold
was **16.6 ms** — below the 16.67 ms vsync period — so *every healthy frame* counted as slow.

- **Also:** exclude frames over ~100 ms (shader-compilation hitches poison an EMA for 30+ frames)
  and ignore the first ~90 frames after load entirely.

### Draw calls are ~3.3× your object count
The colour pass is not the whole story — shadow and AO passes re-render the scene. The lever is
**object count**, so merge by material across spatial clusters (139 meshes → 54; 458 draws → 178).
The cost is coarser frustum culling; for a bounded scene where most things are usually in view,
that trade is correct.

### Bake time is compile-bound, not pixel-bound
Adding ten GLSL families took texture baking from ~350 ms to ~4 s while the new textures were
*smaller* than the existing ones. Each distinct family is a synchronous shader compile.

- **Rule:** share one family across many sets via uniforms; count programs, not pixels.

### LOD crossover too close
Tree billboards engaged at 79 m, right behind the buildings, reading as flat spiky cards. Fixed
with a three-tier system (full ≤ 60 m, mid 60–180 m, billboard > 180 m).

- **Trick:** derive the mid LOD from the *same seed and RNG call order* as the full model, so it
  is a strict subset and cannot pop.

---

## Presents as: it doesn't build or load at all

### A backtick inside a GLSL string
Shipped twice. A backtick in a shader comment terminates the JavaScript template literal holding
it. `node --check` catches it instantly; a code reviewer will not.

### `requestAnimationFrame` stalls in a background tab
A loader that yielded via rAF between stages froze permanently if the tab lost focus mid-load.
Race rAF against a timer:

```js
await new Promise((r) => { let d = false; const f = () => { if (!d) { d = true; r(); } };
  requestAnimationFrame(f); setTimeout(f, 60); });
```

---

## Presents as: two modules disagree

These only happen when subsystems are built independently — and they are the price of
parallelism. All were caught by measurement, none by code review.

### A hard-coded copy of a published constant
An atlas was rebuilt as 2×2; a consumer still cropped it as 4×2, sampling half-clusters. Import
the layout; never hard-code a grid.

### Published on one object, read from another
An atlas grid was published on `material.userData` and read from `texture.userData`, so **every**
atlas silently fell back to a 2×2 default. No error, no warning — just subtly wrong UVs
everywhere.

### A tracker that never updates
A room-occupancy tracker read the same room permanently, so the furniture shown always belonged
to a building 55 m away. The equivalent tracker in another module was correct in the same frame,
which localised it immediately.

- **Rule:** when two modules compute the same thing, compare them at runtime — the disagreement
  is the diagnosis.

### Consumers guessing what a producer knows
A furnisher guessed staircase positions for 14 rooms because the module that *built* the stairs
published no keep-out volumes. Publish what you know; don't make the consumer re-derive it.
