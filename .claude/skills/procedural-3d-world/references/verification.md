# Verification — how to prove a claim about a 3D scene

The central discipline of this project. Every hard defect was found by measurement; several were
first *mis*-attributed by looking at a screenshot. Screenshots tell you something is wrong. They
do not tell you which subsystem, and they are actively misleading about lighting.

Everything here runs in a browser console against a global handle the app exposes:

```js
window.MYWORLD = { THREE, scene, camera, renderer, physics, materials, terrain, player, /* … */ };
```

Expose that handle. It costs nothing and it is the difference between debugging and guessing.

---

## 1. Isolate before you attribute

The most valuable single habit. When something looks wrong, do not reason about which subsystem
is responsible — **turn candidates off and re-measure.**

Worked example. The ground looked dark and muddy. Four hypotheses, four measurements, on the
ground half of the frame:

| Change | Mean screen luminance |
|---|---|
| baseline | 0.243 |
| `sun.castShadow = false` | 0.253 (shadowing accounts for 4%) |
| `metalness = 0` forced on 33 materials | 0.242 (no change — ruled out a suspected PBR bug) |
| ambient occlusion off | 0.244 (1%) |
| grade contrast/toe flattened | 0.276 (13%) |
| `toneMappingExposure` 1.0 → 1.6 | 0.375 (+55%) |

Conclusion: the lighting was fine. Exposure was the only big lever, which meant the surfaces
themselves were too dark — an **albedo** problem. Reading the baked texture bytes confirmed
cobble at 0.099 linear where weathered stone should be 0.24–0.34.

Had this been "fixed" by raising exposure, the plaster (already 0.551) would have clipped and
the frame would have gone milky — which is exactly what an earlier pass did.

**Rule:** a measurement that *rules something out* is as valuable as one that finds the cause.

---

## 2. Sample the composited canvas, not your impression

```js
function frameLuminance(H, { rows = [14, 30], w = 48, h = 30 } = {}) {
  H.renderer.info.reset();
  H.post.render(1 / 60, H.frameCtx);            // or renderer.render(scene, camera)
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').drawImage(H.renderer.domElement, 0, 0, w, h);
  const d = c.getContext('2d').getImageData(0, 0, w, h).data;
  let sum = 0, n = 0, clipped = 0;
  for (let y = rows[0]; y < rows[1]; y++) for (let x = 0; x < w; x++) {
    const i = (y * w + x) * 4;
    sum += (0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]) / 255;
    n++;
    if (d[i] >= 250 || d[i + 1] >= 250 || d[i + 2] >= 250) clipped++;
  }
  return { mean: +(sum / n).toFixed(3), clipPct: +(100 * clipped / n).toFixed(2) };
}
```

Two caveats that cost real time here:

- **`drawImage` from a WebGL canvas only works in the same task as the render.** Render, then
  sample, synchronously.
- **Downsampling averages away peaks.** A 320-px-wide sample will not find a 0.2%-of-pixels
  clipping problem. Sample at native resolution when hunting highlights.

An ASCII luminance grid is a fast way to see *where* a frame is dark:

```js
const ch = ' .:-+*#@';                       // darkest → brightest
// …per pixel: ch[Math.min(7, Math.floor(L * 8))]
```

---

## 3. Drive the simulation by hand

Do not rely on the render loop being alive. In a background or hidden tab, `requestAnimationFrame`
stops firing, the world freezes, and every behavioural test silently returns a false negative.
(This also caught a real bug: a loader that yielded via rAF stalled forever if the user switched
tabs mid-load. Race rAF against a timer.)

Step the simulation yourself:

```js
window.step = (dt) => {
  H.player.update(dt, H.frameCtx);
  H.frameCtx.playerPosition.copy(H.player.position);
  H.physics.update(dt, H.frameCtx);
};
// 3.5 simulated seconds, deterministic, visibility-independent
for (let i = 0; i < 210; i++) step(1 / 60);
```

Now behavioural tests are reproducible: synthesise key events, step a fixed number of frames,
assert on the resulting position.

```js
window.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyW', bubbles: true }));
```

---

## 4. Assert on physics, not on pixels

These are the checks that caught the real gameplay defects.

**Movement matches config.** Walk for a known time, divide. 4.63 m in 1.5 s = 3.09 m/s against a
configured 3.0 — correct.

**Jump matches `v²/2g`.** Peak 0.98 m for 6.2 m/s at 19.6 m/s² — correct. If this is wrong, your
fixed-step integration is wrong.

**Walls actually stop the player.** Run at a wall for 3 s and assert the final distance from the
plot centre is at least half-depth + capsule radius.

**Can the player reach the place they must reach?** For each room, teleport outside the door,
hold forward, step, and assert the final position is inside the room's footprint in the room's
own frame:

```js
const dx = p.x - centre.x, dz = p.z - centre.z;
const lx = dx * cos - dz * sin, lz = dx * sin + dz * cos;   // into room-local
const inside = Math.abs(lx) < width / 2 && Math.abs(lz) < depth / 2;
```

This is the test that found 4 of 11 buildings unenterable. No screenshot would have shown it.

---

## 5. Ray-cast to find what is blocking

When a character stalls somewhere that looks clear, cast rays from the stall point at a ladder of
heights and report what each one hits, by collider tag:

```js
for (const h of [0.05, 0.15, 0.25, 0.35, 0.5, 0.8, 1.2, 1.7, 2.0, 2.2]) {
  const hit = H.physics.raycast(new THREE.Vector3(o.x, floorY + h, o.z), inward, 3.2, {});
  console.log(h, hit ? `${hit.tag} @ ${hit.distance.toFixed(2)}` : 'clear');
}
```

**Tag every collider.** Being told `door-weaver-0` blocks from 0.15 m to 2.0 m is a diagnosis;
being told "something hit at 1.62" is not.

Two traps this exposed:

- Cast from the height you care about. Rays cast from *floor* height flew straight over a
  threshold step that was blocking a capsule whose feet were 8 cm lower — the doorway measured
  "completely clear" while being impassable.
- Cast laterally too. A ray straight down the centre line misses an obstruction 0.4 m to one
  side, and the first walk test aimed exactly between two barrels and concluded the raycast was
  broken.

**Behavioural differential:** when geometry is clear but the player still cannot pass, test
**jump vs walk**. If jumping works and walking does not, it is not an obstruction — it is a step
the character controller refuses to climb. That one comparison isolated an autostep parameter in
a single test.

---

## 6. Attribute cost before optimising

Toggle groups off and re-measure per frame. Note `renderer.info.autoReset = false` means counts
accumulate — reset before each single render or you will measure nonsense (a 40-frame total once
read as 23,224 draw calls).

```js
const perFrame = () => { H.renderer.info.reset(); H.post.render(1/60, H.frameCtx);
  return { draws: H.renderer.info.render.calls, tris: H.renderer.info.render.triangles }; };
const base = perFrame();
for (const [name, g] of Object.entries(groups)) {
  g.visible = false; const off = perFrame(); g.visible = true;
  console.log(name, base.draws - off.draws, base.tris - off.tris);
}
```

This gave: buildings 458 draws, props 269, vegetation 80, terrain 14 — and the realisation that
each object costs ~3.3 draw calls once the shadow and AO passes are counted, so the lever is
**object count**, not mesh complexity.

**Let it settle before you measure.** Culling, light pools and shadow extents converge over
frames. A measurement taken immediately after a teleport read 6.2 M triangles; the settled value
was 2.2 M. Step ~180 frames first, and be suspicious of any number you took mid-transition.

---

## 7. Automated scene audit

Run a structural pass every loop; it catches classes of defect that are invisible in a
screenshot. See `assets/audit.js`. What it checks, and why each earned its place:

| Check | Caught |
|---|---|
| NaN transforms | — (cheap insurance) |
| magenta debug material in use | a missing material name silently rendering |
| foliage `transparent` with `alphaTest === 0` | cards rendering as hard-edged opaque rectangles |
| geometry without normals | flat/black shading |
| objects below the world | props that fell through |
| `scene.environment` null | nothing image-based lit |
| collision ground vs analytic ground | a heightfield disagreeing with the visual mesh by 1.26 m |
| draw / triangle budget | regressions, per loop |

Two lessons from writing it. **Sample only where the check is meaningful** — an early version
dropped rays onto rooftops and reported a 12.6 m "ground mismatch", a false positive that cost
an investigation. And **make it async-safe and re-runnable**, because you will run it dozens of
times.

---

## 8. Report honestly

- Distinguish measured from inferred. If you could not reproduce a number, say so and state what
  your conclusion actually rests on.
- Prefer "the peak is provably down 45–62% and a 24-view sweep found no clipped pixel" over "the
  clipping is fixed".
- When a subagent reports success, verify the claim, not the report. Several agents in this
  project were killed mid-task by transient API errors *after* writing their files — the work was
  on disk and correct while the report was missing entirely. Others reported work that had landed
  only partially.
- Correct your own misattributions explicitly. Two in this project: a `DirectionalLight`'s angle
  read from `position` alone (its direction is `position − target`, and the target was following
  the player), and an "effective" light intensity mistaken for drift when it was
  `authored × documented gain`.
