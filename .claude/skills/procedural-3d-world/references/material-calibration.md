# Material calibration by measurement

How to make a procedural PBR scene look right by measuring albedo instead of tuning it by eye.
This was the single largest visual change in Hollowbrook; everything else (sky, grade, AO,
shadows) turned out to be innocent. All numbers below are from
`/Users/prompt/videos/world` unless marked otherwise. Library versions verified:
three **0.185.1**, @dimforge/rapier3d-compat **0.19.3**.

---

## 1. The symptom, and why it lies to you

The village looked **dark and muddy**. Raising `toneMappingExposure` made it **milky** — the
darks lifted into grey without the picture gaining any contrast, and the plaster went to white.

That pair of symptoms — *dark* that goes *milky* rather than *bright* when you push exposure —
is the signature of **albedo that is too low across the board while one or two surfaces are too
high**. It is not a lighting bug. Learn to recognise it: if a global gain makes the picture
worse in a *different* way instead of better, the fault is in the ratios, not the level.

## 2. Diagnose by elimination, in the browser console, before touching anything

Every candidate got disabled or forced to a null value, and the ground luminance was re-measured.
The point is not the numbers; the point is that **four of the five suspects moved the image less
than the measurement noise of a repositioned camera**, which is what proves the fifth.

| Suspect disabled / forced                  | Ground luminance | Δ      | Verdict |
| ------------------------------------------ | ---------------- | ------ | ------- |
| Shadows off                                | 0.243 → 0.253    | +4%    | innocent |
| `metalness = 0` forced on 33 materials     | no change        | 0%     | innocent — the suspected PBR bug does not exist |
| AO (GTAO + `aoMap`) off                    | —                | ~1%    | innocent |
| Grade pass flattened (contrast/sat/toe off)| 0.242 → 0.276    | 13%    | contributor, not cause |
| Exposure 1.0 → 1.6                         | —                | +55%   | the only big lever — and the wrong one (§3) |

The grade row is recorded in `src/post/post.js:120-128`. The other four rows come from the
review log; I could not re-derive them from source (see *Unverified*). The **method** is the
transferable part:

- Change exactly one thing per measurement, and measure the same pixels each time.
- Prefer a **region mean** (e.g. the ground half of the frame) over a spot sample; a spot sample
  moves 10% if the camera shifts 20 cm.
- A suspect that cannot move the number by more than a few percent **cannot be the cause of a
  problem you can see**. Stop investigating it.
- The "force it to a null value" test is stronger than the "tweak it" test. Forcing metalness to
  0 on every material at once is one console line and it permanently retired a theory.

Conclusion: the light was fine. **The albedo was wrong.**

## 3. Why exposure is the wrong lever

Exposure is a multiply *upstream of the tone curve*. It moves everything, including the things
that were already correct, and the tone curve has a fixed ceiling — so the bright end compresses
while the dark end merely slides up the same shape. That is the definition of "milky".

Verified against three 0.185.1:
`node_modules/three/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js:46-72` —
`ACESFilmicToneMapping` does `color *= toneMappingExposure / 0.6` and ends in `saturate()`. So

- exposure 1.0 is **already a 1.667× pre-scale**; "1.0" is not "no gain".
- the output is hard-clamped to [0,1]; there is no highlight rolloff past the fit.

Reproduced numerically (neutral grey, exposure 1.0 vs 1.6, through the exact ACES fit + sRGB OETF):

| scene-referred linear | sRGB @ exp 1.0 | sRGB @ exp 1.6 |
| --------------------- | -------------- | -------------- |
| 0.174                 | 0.489          | 0.625          |
| 0.248                 | 0.592          | 0.719          |

Raising exposure buys ~28% on the ground and pushes lit plaster toward clipping at the same time.
Hollowbrook shipped with **`RENDER.toneMappingExposure = 1.0`** (`src/config.js:169`,
applied at `src/core/engine.js:29`) and never moved it. Exposure stayed a user setting
(`src/main.js:355`), not a fix.

> **Transferable rule.** Exposure and a global light gain are *level* controls. A muddy picture
> is almost never a level problem — it is a *ratio* problem. Fix ratios in albedo first, then
> set level once, at the end.

## 4. The measurement: integrate luminance over the baked bytes

The bakery renders each albedo pass into an RGBA8 target, reads it back once
(`src/materials/textures.js:1994-1997`) and uploads it as a `DataTexture`
(`textures.js:2000-2016`). `DataTexture` keeps the array on the CPU
(`node_modules/three/src/textures/DataTexture.js:50`: `this.image = { data, width, height }`),
so **measuring costs zero GL calls** — you read the same bytes the GPU samples.

```js
// Browser console. window.HOLLOWBROOK is published at src/main.js:255.
const { textures } = window.HOLLOWBROOK;

// three's EXACT sRGB EOTF — node_modules/three/src/math/ColorManagement.js:205-207,
// and the same constants appear in the shader chunk colorspace_pars_fragment.glsl.js:7-9.
// Do NOT use x^2.2 here (see Trap 1).
const eotf = (c) => (c < 0.04045
  ? c * 0.0773993808
  : Math.pow(c * 0.9478672986 + 0.0521327014, 2.4));

/** Mean Rec.709 LINEAR luminance of a baked albedo map. */
function albedo(key, alphaTest = 0) {
  const d = textures.get(key).map.image.data;      // Uint8Array, RGBA8, sRGB-encoded
  let sum = 0, n = 0;
  for (let i = 0; i < d.length; i += 4) {
    if (alphaTest && d[i + 3] / 255 < alphaTest) continue;   // see below
    sum += 0.2126 * eotf(d[i]     / 255)
         + 0.7152 * eotf(d[i + 1] / 255)
         + 0.0722 * eotf(d[i + 2] / 255);
    n++;
  }
  return { key, albedo: +(sum / n).toFixed(4), coverage: +(n / (d.length / 4)).toFixed(3) };
}

const CUTOUT = /leaf|ivy|needle|flower|grassBlade/;
console.table(textures.keys().map((k) => albedo(k, CUTOUT.test(k) ? 0.42 : 0)));
```

Four things that make this measurement mean something:

1. **Alpha atlases must be averaged only over texels that pass `alphaTest`.** The RGB under a
   transparent texel is never shaded, so anything else is meaningless. Cut-out threshold here is
   `alphaTest = 0.42` (`materials.js:262`); the atlas `uC` values are documented as measured that
   way (`textures.js:1792-1796`). Report **coverage** alongside the mean — at coverage 0.08 you
   are averaging 8% of the image and the number is noisy.
2. **The map is not the material.** `material.color` multiplies the albedo and so **can only ever
   darken**. `materials.js:119-123` records the trap: `dirtPath` used `0xd8c8a8` because it "reads
   as pale warm earth" and measured as `soil × 0.58 linear`, making the lanes the *darkest* ground
   in the village. To brighten, change the bake. `dim(k)` (`materials.js:40-44`) returns the sRGB
   grey whose linear value is exactly `k`, for the honest case.
3. **Vertex colours multiply too.** `terrain.js:142-144` clamps them to ≤ 1 (a multiplier above 1
   is unjustifiable energy gain); `PLAZA_FLOOR = 0.72` (`terrain.js:85-92`) floors the wear
   multiplier's *luminance* — 0.37 on a 0.28-albedo cobble is "not wear, it is a stain".
4. **Record before and after for every edit.** Every albedo comment in `textures.js` carries the
   old value, the new value and the reason. That is what made §5 reconstructable at all.

## 5. The table — physically sane albedos to start a new world from

Linear (scene-referred) diffuse albedo, Rec.709-weighted, measured over the whole baked map
(alpha sets: over texels passing `alphaTest = 0.42`).

| set | before | after | real-world band used as the target | source |
| --- | ------ | ----- | ---------------------------------- | ------ |
| `cobble` (granite setts) | 0.099 | **0.280** | 0.24–0.34 (weathered granite) = sRGB 0.50–0.66 | `textures.js:256-266`, `sky.js:44-45,109` |
| `cobbleWorn` | — | ≈0.31 | ~0.03 above the plaza set, from thinner joints + polish, not a tint | `textures.js:284-288,1670-1671` |
| `plaster` (limewashed daub, exterior) | 0.551 | **0.421** | 0.38–0.48 = sRGB 0.66–0.74 | `textures.js:436-439` |
| `lathPlaster` (limewash, interior) | — | 0.44–0.50 | brightest albedo in the library, deliberately above exterior plaster | `textures.js:1146-1150` |
| `grass` (meadow) | 0.074 | **0.127** | 0.10–0.16 = sRGB 0.36–0.44 mid tone | `textures.js:856-858`, `sky.js:109` |
| `grassBlade` (atlas) | 0.061 | matched to `grass` | a tuft darker than the ground it sits on reads as a dark spike | `textures.js:1621-1627` |
| `timber` (dark oak frame) | 0.038 | **0.0615** | deliberately below the oak band; dark oak that still shows grain | `textures.js:1661-1667,1690` |
| `woodBeam` (oak) | — | ~0.10 | 0.09–0.13 (oak beams) | `textures.js:1690-1694` |
| `woodPlank` (dry sawn softwood) | 0.075 | **0.143** | 0.15–0.22 — "the single biggest miss in the library" | `textures.js:1681-1689` |
| `soil` (dry topsoil) | 0.036 | **0.060** | 0.07–0.11 | `textures.js:802-804`, `materials.js:122` |
| `bark` | 0.049 | ~0.077 | 0.07–0.11 (tree bark) | `textures.js:1705-1709` |
| `roofTile` (weathered pantile) | 0.090 | **0.109** | 0.10–0.14; the fix was ×1.089 in sRGB = ×1.21 linear | `textures.js:734-736` |
| `stone` (grey ashlar/rubble) | — | 0.213 | 0.20–0.24 = sRGB 0.50–0.56 for the faces | `textures.js:685`, `1088-1089` |
| `flagstone` (indoor) | — | 0.16–0.20 | a shade under exterior ashlar: boots, ash, grease seal them | `textures.js:1088-1089` |
| `hearthStone` | — | 0.08–0.12 | dressed sandstone dragged down by smoke | `textures.js:1211-1212` |
| `soot` | — | 0.02–0.035 | blackest common surface; ~6× darker than anything outdoors | `textures.js:1248-1249` |
| `strawLitter` | — | 0.18–0.24 | bright straw over dark trodden earth; the mean lands between | `textures.js:1299-1300` |
| `leaf` (broadleaf, atlas) | — | 0.09–0.14 | leaves are far more reflective than they look — a lit canopy is mostly transmission | `textures.js:1792-1798` |
| `ivy` (atlas) | — | ~0.11 | ≥ leaf | `textures.js:1802-1803` |
| `needle` (conifer, atlas) | 0.032 | **0.060** | 0.06–0.10; at 0.032 it was darker than wrought iron and every conifer was a flat black cut-out | `textures.js:1804-1807` |

Reference bands worth carrying to any world (from the comments above, and consistent with
standard PBR albedo charts): fresh snow 0.8+, limewash/white plaster 0.38–0.48, dry concrete
0.25–0.35, weathered granite 0.24–0.34, grey ashlar 0.20–0.24, dry grass/meadow 0.10–0.16,
clay tile 0.10–0.14, oak 0.09–0.13, bark 0.07–0.11, dry topsoil 0.07–0.11, conifer needle
0.06–0.10, fresh asphalt 0.04–0.06, soot/charcoal 0.02–0.035.

**Nothing in a natural exterior scene should measure below ~0.03, and almost nothing above
~0.50.** The failing bake had six surfaces under 0.08 — the whole village lived in the bottom
one-eighth of the range, which is why no amount of light helped.

## 6. The ordering rule

Absolute values matter, but the eye reads **relative** brightness. Two orderings were written
into the code as invariants that must survive any future edit:

```
timber (0.0615) < woodBeam (~0.10) < woodPlank (0.143)      textures.js:1690-1691
needle (0.060)  <  leaf (0.09-0.14) <= ivy (~0.11)          textures.js:1792-1807
```

Also enforced by construction: `lathPlaster` (interior limewash) **above** `plaster` (exterior,
forty years of rain streaks) with an explicit "do not dirty this up to match"
(`textures.js:1146-1150`); `flagstone` **below** exterior `stone`; `strawLitter` between its own
straw and its own dirt.

> **Transferable rule.** Write the ordering constraints down next to the constants, as prose, with
> the reason. A future agent will otherwise "fix" one surface in isolation and silently invert a
> pair. A surface that is correct in absolute terms but inverted against its neighbour looks
> *more* wrong than both being off by 20%.

Corollary seen twice in this project: when two surfaces of the same family end up at different
levels for map reasons, cancel it with a **measured compensation constant**, not by re-tinting.
`ROAD_LEVEL = { cobble: 0.885, dirt: 1.0 }` (`terrain.js:101-123`) exists because the road bakes
on `cobbleWorn` and measured 0.670 screen luminance against the plaza's 0.586 (+14.3%);
0.670 × 0.885 = 0.593, within 1.2% of the plaza. The comment states explicitly that vertex
colours were already equal at the join, so "the residual is the map".

## 7. The counter-lesson: bring the bright thing DOWN

Raising the dark surfaces is only half the fix. `plaster` had to go **0.551 → 0.421** —
*downward* — before the cobbles could come up.

Why (`textures.js:436-439`): at 0.55 linear the plaster was the brightest thing in the village
**by a factor of five**, and it clipped as soon as the ground was given a realistic albedo. The
tone curve has a fixed ceiling; the only way to raise the ground *without* the frame going milky
is to make room at the top.

Consequences that had to be tracked through the same edit:

- `plasterWorn` grime 0.75 → 0.52 and cracks 0.95 → 0.82 (`textures.js:1672-1676`): with the
  palette dropped, the old grime load put the worn set a *fifth* below `plaster` instead of just
  under it. **Weathering parameters are relative to the base palette; move one, re-solve the
  other.**
- `cobble` variation had to be carried **downward**. With a mean near sRGB 0.59 there is only
  1.7× of headroom to 1.0, so the palette's variety lives in the dark ends of each stone type and
  the multiplier maxima are capped (`textures.js:258-266`). *"Do not scale the palette up without
  pulling the multiplier maxima down by the same factor."*
- The **grade pivot** had to move: `PIVOT` 0.18 → 0.14 in `src/post/post.js:181-195`. Mid grey is
  the textbook pivot, but the textbook assumes the subject sits at mid grey — the sunlit setts
  need 0.174–0.248 scene-referred, so with a 0.18 pivot every ground pixel was on the *falling*
  side of the contrast curve.
- The **shadow toe** had to move: `SHADOW_TOE` 0.16 → 0.06 and `shadows` 0.10–0.14 → 0.05–0.06
  (`post.js:116-137,197-203`). 0.16 is mid grey, so "shadows" was rolling down the sunlit ground
  — worth 13% of the ground luminance.
- **Roughness floors** had to rise where a specular lobe now sat on 3× the diffuse energy. The
  cobble polished-crown roughness floor went 0.30 → 0.42 (`textures.js:295-301`): *"at 0.28 the
  same lobe carries 2.8× the energy over a diffuse term that only tripled, and near-camera
  ground-facing highlights clipped (0.19% of pixels ≥ 250 in a low close-up)."*

> **Transferable rule.** An albedo change is an *energy* change, and every specular/emissive
> constant tuned against the old diffuse level is now wrong. Re-check: roughness floors on
> polished surfaces, emissive intensities, bloom thresholds, and any tint that was compensating
> for the old level.

## 8. What must move *with* the albedo (propagation checklist)

Recalibrating albedo alone leaves the picture correctly-proportioned but under-lit, because the
direct term was tuned against the old dark ground. Hollowbrook moved five constants together:

| constant | value | why it moved with albedo | file |
| -------- | ----- | ------------------------ | ---- |
| `SUN_GAIN` | 1.22 (was 1.0) | with the albedos corrected the whole picture moves up the ACES curve; the direct term has to come with it or the sun stops reading as a sun | `sky.js:39-64` |
| `ENV_BAKE_GAIN` | 0.98 (was 0.72) | 0.72 was compensation for the wrong problem — with a near-black ground the indirect term was the only thing keeping shadows alive | `sky.js:66-83` |
| `GROUND_ALBEDO` | `[0.24, 0.225, 0.155]` (was ~0.20 mean) | area-weighted mix of the corrected setts (~0.28), meadow (~0.13) and stone/soil (~0.20) | `sky.js:101-112` |
| `HEMI_DAY_INTENSITY` | 0.22 (was 0.16, earlier 0.30) | still only 0.099 of the 3.08 total irradiance on sunlit ground (3.2%) — it *shapes*, the env bake *fills* | `lighting.js:81-92` |
| grade `PIVOT` / `SHADOW_TOE` | 0.14 / 0.06 | §7 | `post.js:181-203` |

`SUN_GAIN` was **solved, not guessed**: it is the gain that puts sunlit plaza cobble at 0.549 sRGB
— the middle of the 0.50–0.62 target — and stays in band across the plausible albedo range
(0.511 at cobble 0.24, 0.582 at 0.32). The stated guard against over-gaining is the **linear
lit:shadow irradiance ratio** on the ground: 2.60 after vs 2.72 before. *"This scales the
picture, it does not compress it."* Use that ratio as the acceptance test whenever you raise a
key light — it distinguishes "brighter" from "flatter", which exposure alone cannot.

Measured irradiance budget after calibration, recorded at `lighting.js:128-133`:
sunlit ground ≈ **4.3 total** = direct 2.58 + env ~1.61 + hemisphere 0.099; shadowed plaster wall
0.385 sRGB. Shadowed plaza cobble went 0.233 → 0.341 and shadowed plaster 0.243 → 0.385 purely
from the env bake gain, while indirect/direct on the ground only went 0.580 → 0.625
(`sky.js:75-81`).

## 9. Verification targets, and how to measure them off the composited canvas

Acceptance criteria used for the exterior hero view:

| target | value | evidence in code |
| ------ | ----- | ---------------- |
| sunlit ground, screen luminance | **0.50–0.62 sRGB** | `sky.js:49-51`, `post.js:184-188` |
| ground-half frame mean | **≥ 0.36** | brief; measured 0.474 after calibration (`terrain.js:685,1106`), 0.242–0.276 before (`post.js:120-123`) |
| clipped pixels | **zero** | the working metric is "% of pixels ≥ 250" — 0.19% was treated as a defect (`textures.js:299`) |
| shadowed ground under 0.20 luminance | was 46.6% of ground pixels — used as the "shadows are dead" detector | `sky.js:76-78` |
| sunlit plaster peak | 218/255 ungraded (`sky.js:52`); 226,223,216 graded (`post.js:193-194`) | both independently reproduced below |

**Reading the frame.** `src/core/engine.js:22` sets `preserveDrawingBuffer: false`, so the default
framebuffer is undefined once the JS task ends — `canvas.toDataURL()` / `drawImage` from the
console return black. Render and read **inside one task**:

```js
const { renderer, post, frameCtx } = window.HOLLOWBROOK;
const gl = renderer.getContext();
const w = renderer.domElement.width, h = renderer.domElement.height;

post.render(0, frameCtx);                       // src/main.js:237 calls it exactly this way
const px = new Uint8Array(w * h * 4);
gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);   // still valid: same task

// OutputPass has already tone-mapped and sRGB-encoded, so these bytes ARE screen luminance.
let sum = 0, n = 0, clipped = 0;
for (let y = 0; y < h / 2; y++) {               // bottom half of the frame = "ground half"
  for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4;
    const L = (0.2126 * px[i] + 0.7152 * px[i+1] + 0.0722 * px[i+2]) / 255;
    if (px[i] >= 250 || px[i+1] >= 250 || px[i+2] >= 250) clipped++;
    sum += L; n++;
  }
}
console.log({ groundHalfMean: sum / n, clippedPct: 100 * clipped / n });
```

Row 0 of `readPixels` is the **bottom** of the image, which is why the loop over `y < h/2` is the
ground half. Do **not** linearise these bytes — the target band is defined in sRGB screen space.

**Solving offline instead of guessing.** The chain is deterministic, so invert it in Node and
solve for the albedo you need. This script reproduces `post.js`'s claims exactly (graded plaster
→ 226,223,216; pivot deltas +4.9/+3.5/+2.3/+1.3% at scene 0.03/0.08/0.20/0.40 vs the comment's
+5.0/+3.9/+2.3/+1.3%; scene 0.174 → 0.500 and 0.248 → 0.620 sRGB, exactly the 0.50–0.62 band):

```js
const IN=[[0.59719,0.07600,0.02840],[0.35458,0.90834,0.13383],[0.04823,0.01566,0.83777]];
const OUT=[[1.60475,-0.10208,-0.00327],[-0.53108,1.10813,-0.07276],[-0.07367,-0.00605,1.07602]];
const mul=(m,v)=>[0,1,2].map(j=>m[0][j]*v[0]+m[1][j]*v[1]+m[2][j]*v[2]);   // three stores columns
const fit=v=>v.map(x=>(x*(x+0.0245786)-0.000090537)/(x*(0.983729*x+0.4329510)+0.238081));
const aces=(c,exp=1)=>mul(OUT,fit(mul(IN,c.map(x=>x*exp/0.6)))).map(x=>Math.min(1,Math.max(0,x)));
const oetf=x=>x<=0.0031308?x*12.92:1.055*Math.pow(x,1/2.4)-0.055;
const L=[0.2126,0.7152,0.0722];
function grade(c,{contrast=1.17,sat=1.10,shadows=0.06,PIVOT=0.14,TOE=0.06}={}){
  let v=c.map(x=>PIVOT*Math.pow(x/PIVOT,contrast));
  const sl=v.reduce((a,x,i)=>a+x*L[i],0), t=Math.min(1,Math.max(0,sl/TOE)), s=t*t*(3-2*t);
  v=v.map(x=>x*((1-shadows)+shadows*s));
  const lu=v.reduce((a,x,i)=>a+x*L[i],0);
  return v.map(x=>lu+(x-lu)*sat);
}
const screen = c => aces(grade(c)).map(oetf);
```

Useful invariants that fall out of it (all recomputed here, exposure 1.0):

- neutral input that **rounds to 255**: **12.15 linear**; ACES `saturate()` first bites at 15.40.
- a **saturated** hue clips far earlier: the 1900 K fire colour (sRGB 1, 0.517, 0 — from
  `kelvinToColour`, `lighting.js:429-435`) has its red channel hit 1.0 at scene luminance **0.76**.
  Clipping is per-channel and chroma-dependent; a neutral headroom figure will mislead you on
  anything coloured.
- the ungraded/graded gap is large: 0.74/0.69/0.58 → 215,213,206 ungraded but 226,223,216 graded.
  **Always state which one a measurement is.** `src/perf/perf.js:569-590` prints the live grade
  state next to the frame precisely so "a flattened measurement is never mistaken for a graded one".

## 10. Traps

1. **`x^2.2` is not the sRGB EOTF, and the error is worst exactly where it hurts.** Against the
   exact transfer function it under-reads by **37% at sRGB 0.10**, 12.4% at 0.20, 3.4% at 0.30,
   and only ≈2% above 0.5. Dark surfaces — soot, needle, timber, soil — are precisely the ones you
   are trying to measure. `textures.js:259` uses "about x^2.2" as shorthand in prose; use the real
   function in the measuring code (`ColorManagement.js:205-207`).
2. **Authored constant ≠ effective value.** `SUN.intensity = 3.6` (`src/world/layout.js:630`), but
   `lighting.js:503` and `:749` both construct/update the light as
   `SUN.intensity * SUN_GAIN` = 3.6 × 1.22 = **4.392**. During integration this was reported as
   drift between streams; it is not — it is one authored value and one deliberate, documented
   correction, kept as a named export specifically so lighting and the sky's ground-bounce term
   agree (`sky.js:61-63`). **Before filing a "constant drift" defect, grep for every multiplier
   applied to the constant at its use site.** Name such gains and export them; never inline a
   bare number that silently redefines an authored one.
3. **A tint can only darken.** See §4.2. The correct move for "make this brighter" is always the
   bake.
4. **Averaging an alpha atlas over all texels is meaningless.** See §4.1.
5. **`scene.environmentIntensity` is a dead knob if you assign `envMap` per material.** three's
   renderer overwrites `envMapIntensity` from `scene.environmentIntensity` only when
   `material.envMap === null` (`materials.js:29-33`, `lighting.js:115-122`). In Hollowbrook the
   real IBL handle is per-material `envMapIntensity` (`INTERIOR_ENV = { surface: 0.16, deep: 0.09,
   reveal: 0.45 }`, `lighting.js:159-163`).
6. **Recolouring for chroma must be luminance-neutral, and you can prove it.** `terrain.js:660-685`
   splits one warm base tone into three sett families normalised to exactly `lum = 0.90138` —
   "luminance is a linear functional, so any mix of them has that luminance too". Measured over
   17 689 plaza vertices: mean luminance 0.81359 → 0.81363 (unchanged, as predicted), mean (r−b)
   −49%, sd (r−b) +45%. Same trick at `terrain.js:1088-1110` with a noise octave of mean exactly
   0.5 (ground-half 0.474 before and after). **Design edits so the thing you are not trying to
   change is provably invariant, then measure to confirm it.**
7. **AO is not a fix for a dark scene.** GTAO measured ~1% here; it is deliberately short-radius
   contact occlusion (`GTAO_AO.radius = 0.35 m`, `post.js:77-85`) and `aoMapIntensity` defaults to
   0.9 (`materials.js:228`), lowered to 0.62–0.75 on interior walls that act as bounce cards
   (`textures.js:1168-1170`). If you find yourself raising AO to add depth, your albedo ratios are
   flat.

## 11. Procedure to port to a different world

1. Bake the library, run the §4 snippet, print the whole table sorted by albedo.
2. Write a target band next to every set, sourced from a real material, **as a code comment** —
   including the sRGB equivalent, because that is what you will be typing.
3. Flag anything under 0.03 or over 0.50 in a natural exterior. Outliers first.
4. Write down the ordering constraints (§6) before editing anything.
5. Bring the too-bright surfaces **down** in the same pass as the too-dark ones up (§7).
6. Re-solve direct/indirect gains against the new albedos (§8) — solve for a screen-luminance
   target, do not eyeball; check the **lit:shadow linear irradiance ratio** did not fall.
7. Re-check every specular floor, emissive and bloom threshold tuned against the old diffuse (§7).
8. Verify with §9 at both quality extremes and in at least one shaded close-up — the cobble
   roughness clip only appeared in a *low* preset close-up.
9. Leave every before/after number in a comment.

---

## Unverified

- The four elimination rows in §2 (shadows 0.243→0.253, metalness-0 on **33 materials**, AO 1%,
  exposure 1.0→1.6 = **+55%**) are from the review log. Only the grade row (13%, ground-half
  0.242→0.276) appears in source (`post.js:120-123`). The +55% is not reproducible as an sRGB
  delta on a neutral patch — my recomputation gives +27.8% at scene 0.174 — so it presumably
  refers to a different statistic (region mean, or linear rather than sRGB).
- The **ground-half ≥ 0.36** threshold is from the brief; the code records measurements (0.474)
  but not the threshold itself.
- "ACES clips anything past 2.87" (`lighting.js:300`) — I get **12.15** neutral (rounds to
  255) and **0.76** luminance for the 1900 K fire hue. 2.87 is neither; it is probably specific to
  a chromaticity or a pre-grade value. Treat the *rule* (clipping is per-channel and
  chroma-dependent) as sound and re-derive the number for your own palette.
- `textures.js:1685` says `woodPlank` uses `(0.536, 0.386, 0.252)` ≈ 0.145 linear, but the SETS
  entry at `textures.js:1689` is `uC: [0.561, 0.404, 0.264]` — a base-tone luminance of 0.160, not
  0.145. The comment drifted from the constant during a later tweak. **Real defect pattern**: when
  a calibrated value is documented in one place and defined in another, they diverge. Prefer
  putting the measured number *in* the table entry.
- Per-set "after" values not stated in the code (`cobbleWorn`, `bark`, `ivy`, `woodBeam`) are
  derived here from base-tone luminance of the `uC`/palette constants, which sits **above** the
  whole-map mean (multipliers and dark joints pull the mean down). Re-measure with §4 rather than
  trusting them.
