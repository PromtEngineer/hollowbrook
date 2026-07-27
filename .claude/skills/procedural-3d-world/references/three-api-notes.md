# three.js API facts that training data gets wrong

Verified against `node_modules/three` **0.185.1** in this project. Every claim below has a
`library-file:line`. Project usages cite `src/...`. Re-verification greps at the end — run them
before trusting this on a different three version.

Convention: **Rule** = transferable. *This project* = the instance it was found in.

---

## 1. Tone mapping is SKIPPED when rendering into a render target

**Rule.** If you use an `EffectComposer` (or any offscreen target), material shaders are compiled
with `NoToneMapping` regardless of `renderer.toneMapping`. The whole pass chain therefore sees
genuine scene-referred HDR, and the tone map must happen exactly once, at the end, in `OutputPass`.
Any bloom/AO/grade threshold you tune is a threshold in **linear HDR**, not in 0–1 display values.

Library proof, duplicated in two places (program build + per-object cache check), both identical:

```js
let toneMapping = NoToneMapping;
if ( material.toneMapped ) {
    if ( currentRenderTarget === null || currentRenderTarget.isXRRenderTarget === true ) {
        toneMapping = renderer.toneMapping;
    }
}
```
`src/renderers/webgl/WebGLPrograms.js:176-186` and `src/renderers/WebGLRenderer.js:2351-2361`.
So `currentRenderTarget !== null` ⇒ no tone map in the material. Not a bug; deliberate.

The same "am I rendering to a target?" branch governs **unlit uniform colour space**:
`getUnlitUniformColorSpace()` (`src/renderers/shaders/UniformsUtils.js:128-145`) returns
`renderer.outputColorSpace` only when the target is `null`, otherwise the working space
(LinearSRGB). It is used for `fogColor` (`src/renderers/webgl/WebGLMaterials.js:27`) and the
background colour (`src/renderers/webgl/WebGLBackground.js:241`).

> **Trap.** When you move a scene onto a composer, `scene.fog.color` silently changes meaning:
> off-composer it is converted to sRGB output, on-composer it is passed through as linear.
> A fog colour authored by eye without a composer will be visibly wrong once you add one.
> *This project* keeps fog colours linear on purpose — `src/lighting/lighting.js:788-790`.

**Consequences designed around in this project**

- `renderer.toneMapping = ACESFilmicToneMapping` is still set (`src/core/engine.js:28`) because
  `OutputPass` *reads it off the renderer* — the pass has no tone-mapping argument.
- `new OutputPass()` last in the chain, one and only tone map (`src/post/post.js:637, 645`).
- Bloom thresholds are HDR luminances, not 0–1: `BLOOM_THRESHOLD = { low: 2.4, medium: 2.5,
  high: 2.6, ultra: 2.7 }` (`src/post/post.js:72`).
- Grade runs **before** `OutputPass` so contrast is a power in linear light and the vignette is a
  scene-referred multiply (`src/post/post.js:29-31`).
- Sky material is `toneMapped: false` anyway (`src/lighting/sky.js:897`), making the HDR intent
  explicit for anyone who later renders it without a composer.

### Failure mode that was actually hit: bloom threshold below the sky

`UnrealBloomPass`'s high pass is a **mix, not a subtract**:

```glsl
float alpha = smoothstep( luminosityThreshold, luminosityThreshold + smoothWidth, v );
gl_FragColor = mix( outputColor, texel, alpha );
```
`examples/jsm/shaders/LuminosityHighPassShader.js:60-62`

**Rule.** Once a texel is over threshold, its *entire* value is blurred — there is no
"excess only" term. So a threshold below your sky luminance means the sky blooms 100%, and every
silhouette against it grows a white halo the width of the blur kernel.

*This project*: sky sun disc ≈ 46, lit cumulus 1.5–4; the old threshold of 1.25 put the whole sky
over the line. Detected as a white halo along every roofline. Fix: threshold above the sky
(2.4–2.7) + a soft knee. Documented `src/post/post.js:49-71`.

The knee is `smoothWidth`, which the pass sets to `0.01` in its constructor
(`examples/jsm/postprocessing/UnrealBloomPass.js:137`) and — critically — **only re-writes
`luminosityThreshold` per frame** (`:309`), never `smoothWidth`. So a value you poke into
`bloomPass.highPassUniforms.smoothWidth.value` sticks. `src/post/post.js:592-596`.

Cost accepted: lantern emissives at ~1.7 luminance no longer bloom. The lesson recorded is
"fix the emitter, not the threshold" — a threshold that catches four lanterns also catches 100% of
the sky.

---

## 2. `PCFSoftShadowMap` is deprecated and silently downgraded

```js
if ( this.type === PCFSoftShadowMap ) {
    warn( 'WebGLShadowMap: PCFSoftShadowMap has been deprecated. Using PCFShadowMap instead.' );
    this.type = PCFShadowMap;
}
```
`src/renderers/webgl/WebGLShadowMap.js:99-103` (inside `render()`, so it fires on the first
shadow frame; default is already `PCFShadowMap` at `:89`).

**Rule.** In r185 `PCFShadowMap` *is* the soft path, and softness is a property of the **light**,
not the renderer. Asking for `PCFSoftShadowMap` buys you a console warning and nothing else.

What `PCFShadowMap` compiles to (`src/renderers/shaders/ShaderChunk/shadowmap_pars_fragment.glsl.js`):

| element | line | detail |
|---|---|---|
| hardware comparison sampler | `:20, :46, :70` | `sampler2DShadow` / `samplerCubeShadow` under `SHADOWMAP_TYPE_PCF` |
| interleaved gradient noise | `:94-98` | `fract( 52.9829189 * fract( dot( pos, vec2(0.06711056, 0.00583715) ) ) )` → per-pixel rotation `phi` |
| Vogel disk | `:101-107` | golden angle `2.399963229728653`, `r = sqrt((i+0.5)/n)` |
| the filter | `:129-142` | **5** Vogel taps × 4-tap hardware PCF ≈ "20 filtered taps"; `* 0.2` |
| the softness knob | `:131` | `float radius = shadowRadius * texelSize.x;` |

`shadowRadius` is `light.shadow.radius`, uploaded per light. And the depth texture is only given
`compareFunction` + `LinearFilter` **when `type === PCFShadowMap`** — otherwise `NearestFilter` and
`compareFunction = null` (`src/renderers/webgl/WebGLShadowMap.js:261-271`). That is why the
hardware 4-tap only exists on the PCF path.

> **Failure mode, and it lasted the project's entire life.** `light.shadow.radius` defaults to `1`,
> which is one texel — visually a hard shadow. Hollowbrook shipped hard shadows for months because
> everyone assumed "soft" was a `renderer.shadowMap.type` choice. Detected only in a late review
> loop when someone read the shader. Fix is one line:
> `lighting.sun.shadow.radius = quality.softShadows ? 3.5 : 1.0;` (`src/main.js:96`), with the
> reasoning left in place at `src/core/engine.js:31-37` and `src/main.js:91-95`.

**Rule.** Shadow softness budget = `radius` (texels) × texel size (metres). Keep the texel size a
published stat so the two can be reasoned about together: this project prints
`shadowTexelsPerMetre` and `shadowTexelCm` (`src/lighting/lighting.js:883-885`).

---

## 3. Constructor signatures that changed (all no-arg or re-ordered)

| class | r185 signature | library:line | old signature you probably remember |
|---|---|---|---|
| `OutputPass` | `new OutputPass()` | `examples/jsm/postprocessing/OutputPass.js:36` | `new OutputPass( toneMapping, outputColorSpace )` — now read off the renderer |
| `SMAAPass` | `new SMAAPass()` | `examples/jsm/postprocessing/SMAAPass.js:30` | `new SMAAPass( width, height )` — sizing is via `setSize()` |
| `FXAAPass` | `new FXAAPass()` | `examples/jsm/postprocessing/FXAAPass.js:19` | used to be `new ShaderPass( FXAAShader )` + manual `resolution` uniform |
| `UnrealBloomPass` | `new UnrealBloomPass( resolution:Vector2, strength = 1, radius, threshold )` | `examples/jsm/postprocessing/UnrealBloomPass.js:46` | unchanged, but `resolution` is a **Vector2**, not two numbers |
| `GTAOPass` | `new GTAOPass( scene, camera, width = 512, height = 512, parameters, aoParameters, pdParameters )` | `examples/jsm/postprocessing/GTAOPass.js:56` | — |

Project usage: `src/post/post.js:585-590` (bloom), `:605` (SMAA), `:604` (FXAA), `:549` (GTAO),
`:637` (output).

**AA ordering rule, from three's own doc comment:**
> "`SMAAPass` operates in `linear-srgb` so this pass must be executed before `OutputPass`."
> `examples/jsm/postprocessing/SMAAPass.js:14-15`

and `OutputPass`'s own comment: "If a pass requires sRGB input (e.g. like FXAA), the pass must
follow `OutputPass`" (`examples/jsm/postprocessing/OutputPass.js:20-21`).

*This project* puts both AA modes before `OutputPass` (`src/post/post.js:644`), which is correct
for SMAA and a small compromise for FXAA — worth knowing if edges look under-resolved on the FXAA
preset. **Unverified**: no measurement of the FXAA-before-output quality loss was found in the repo.

### GTAOPass specifics

- `GTAOPass.OUTPUT` is a static enum: `{ Off:-1, Default:0, Diffuse:1, Depth:2, Normal:3, AO:4,
  Denoise:5 }` — `examples/jsm/postprocessing/GTAOPass.js:717-725`. `Default` is the composited
  result; anything else is a debug view that replaces the frame.
- `blendIntensity` defaults to `1.` (`:114`) and is only applied on the `Default` path (`:577`).
- **Trap:** `composer.setSize()` / `insertPass()` hand every pass the *full* framebuffer size, so a
  fractional-resolution AO must override `setSize`, not just the constructor args — otherwise the
  first resize silently restores full res. `src/post/post.js:551-557`.
- `_overrideVisibility()` / `_visibilityCache` (`GTAOPass.js:651, 106`) is the sanctioned hook for
  hiding geometry during the normal+depth gbuffer pass. The pass re-renders the **whole scene** to
  build that gbuffer, so it is the most expensive thing in the chain.

**Rule.** An SSAO-family pass costs a second full scene submit. Cull it aggressively: anything
whose projected occlusion radius is sub-pixel cannot change the result. This project hides the sky,
far terrain and anything beyond `GTAO_CULL_DISTANCE = 55 m` for a `radius = 0.35 m` effect
(`src/post/post.js:78-114, 517-542`). Measured tuning: `GTAO_SCALE.high = 0.75` "costs 56% of the
fill and is invisible" because the result is Poisson-denoised immediately anyway
(`src/post/post.js:96-103`).

---

## 4. `three/addons/*` resolution

`node_modules/three/package.json:8-19` exports map:

```json
"./examples/jsm/*": "./examples/jsm/*",
"./addons":         "./examples/jsm/Addons.js",
"./addons/*":       "./examples/jsm/*",
"./src/*":          "./src/*",
"./webgpu":         "./build/three.webgpu.js",
"./tsl":            "./build/three.tsl.js"
```

**Rule.** `three/addons/postprocessing/EffectComposer.js` is a real, first-class specifier — no
Vite alias, no `resolve.alias`, no deep `three/examples/jsm/...` path needed. This project's
`vite.config.js` contains **no** three aliasing at all. Addons `import { ... } from 'three'`, so
they always share the core module instance; there is no dual-instance hazard via this path.

Note `./src/*` is exported too: `three/src/renderers/...` is importable, which is how you can grep
GLSL chunks from a build if you ever need to.

---

## 5. `PMREMGenerator.fromCubemap` — building an HDRI with no `.hdr` file

Pipeline used (`src/lighting/sky.js:914-1047`):

```
ShaderMaterial sky (HDR, toneMapped:false)
  → private bakeScene (dome only, no village, no lights)
  → CubeCamera(0.05, 20) into WebGLCubeRenderTarget(envSize, HalfFloatType, LinearSRGBColorSpace)
  → pmrem.fromCubemap( cubeTarget.texture, pmremTarget )
  → scene.environment
```

**Verified API facts**

- `fromCubemap( cubemap, renderTarget = null )` — `src/extras/PMREMGenerator.js:169-173`.
  Doc there: ideal input cube is 256×256, minimum 16×16 per face.
- **The first call MUST pass `null`.** `_fromTexture` does
  `const cubeUVRenderTarget = renderTarget || this._allocateTargets();`
  (`src/extras/PMREMGenerator.js:279`), and `_allocateTargets()` is the *only* thing that builds
  `_lodMeshes`, `_sizeLods`, `_sigmas`, `_blurMaterial`, `_ggxMaterial`
  (`:316-319`). Hand it a target on call one and `_textureToCubeUV` dereferences
  `this._lodMeshes[ 0 ]` on an empty array (`:474`) and throws.
  This is exactly what `src/lighting/sky.js:940-948` documents.
- **Reuse the target on every rebake after that.** It keeps the output `texture` object identity
  stable, so `scene.environment` and every material that cached `envMap` stay valid across a
  time-of-day change, and nothing leaks. `src/lighting/sky.js:14-19, 1029-1033`.
- `compileCubemapShader()` (`:179-189`) precompiles; wrap in try/catch and fail soft
  (`src/lighting/sky.js:937-939`). *Unverified*: the "~10 ms first-bake hitch" is the code
  comment's figure, not reproduced here.
- Allocated target: `HalfFloatType`, `RGBAFormat`, `LinearSRGBColorSpace`, `depthBuffer: false`,
  size `3*max(cubeSize,112) × 4*cubeSize` (`:288-302`).
- PMREM restores the previously bound target in `_cleanup`; only a **throw** can leave one bound,
  and a stuck render target is a black screen — so catch and `renderer.setRenderTarget(null)`
  (`src/lighting/sky.js:1034-1038`).

**Rule (bake hygiene).** Bake the environment from a *private* scene containing only the sky, and
suppress the sun disc during the bake — a sub-texel solar disc becomes a blocky hot square after
prefiltering, and the `DirectionalLight` already carries the direct term
(`src/lighting/sky.js:1020-1026`). Capture at a lower gain than you draw (`ENV_BAKE_GAIN`) because
what you are capturing is the *indirect* level.

### IBL has no occlusion — the interiors trap

**Rule.** `scene.environment` lights the interior face of a sealed wall exactly as much as the
exterior face. Every enclosed room reads as a flat bright box and your fire does nothing.

The only clean handle is **per-material `envMapIntensity`**, and here is the API gotcha:

```js
if ( ( material.isMeshStandardMaterial || ... ) && material.envMap === null && scene.environment !== null ) {
    m_uniforms.envMapIntensity.value = scene.environmentIntensity;
}
```
`src/renderers/WebGLRenderer.js:2693-2697`

So `scene.environmentIntensity` **only** applies to materials whose own `envMap` is `null`. This
project assigns `envMap` explicitly on every standard material (`src/materials/materials.js:401-406`),
which makes `scene.environmentIntensity` a dead knob project-wide — documented at
`src/lighting/lighting.js:110-122`. Setting it in `main.js` changes nothing.

**Rule.** Decide once whether env comes from `scene.environment` (scene-level intensity works) or
from per-material `envMap` (only `material.envMapIntensity` works). Mixing them produces a knob
that appears to exist and does nothing.

Also: `material.envMap` going from null→texture changes the shader program, so it needs
`needsUpdate = true` (`src/materials/materials.js:403-404`).

---

## 6. `DirectionalLight` direction is `position − target`

```js
uniforms.direction.setFromMatrixPosition( light.matrixWorld );
vector3.setFromMatrixPosition( light.target.matrixWorld );
uniforms.direction.sub( vector3 );
```
`src/renderers/webgl/WebGLLights.js:509-511` (spot lights: same, `:523-525`).

The shadow camera does the same thing geometrically —
`shadowCamera.position.copy( lightPositionWorld ); shadowCamera.lookAt( targetWorld )`
(`src/lights/LightShadow.js:206-210`).

**Rule.** `light.position` alone tells you **nothing** about sun angle when the target moves.
Always compute `normalize(position − target.getWorldPosition())`.

*This project* — the sun's position is derived, not authored. Every frame
(`src/lighting/lighting.js:887-912`):

```js
_f.copy(sky.sunDirection).normalize();           // the actual sun direction
...
sun.target.position.copy(_snapped);               // snapped to a shadow texel, follows the player
sun.target.updateMatrixWorld();
sun.position.copy(_snapped).addScaledVector(_f, shadowExtent + 20);
```

So `sun.position` is `target + direction × (extent + 20)` and wanders around the map with the
player. **Failure mode:** during integration the sun angle was diagnosed by printing
`sun.position` and reported as wrong; it was correct, the reader was subtracting nothing.

Two more directional-light facts this code depends on:

- `light.target` must be **in the scene graph** or its `matrixWorld` is never refreshed —
  `scene.add(sun.target)` at `src/lighting/lighting.js:519`, plus an explicit
  `sun.target.updateMatrixWorld()` because the light is moved after the scene's traversal
  (`:910`).
- **Texel snapping must happen in the shadow camera's own basis.** `LightShadow.updateMatrices`
  builds the basis via `lookAt`, i.e. `z = f`, `x = normalize(cross(up, f))`, `y = cross(f, x)`.
  Snapping the centre in world XZ instead does nothing for shadow crawl. Reconstructed exactly at
  `src/lighting/lighting.js:890-907`. Guard for the degenerate case: if `|f.y| > 0.9995`, perturb
  (`:888`) — `cross(up, f)` is zero at the pole.
- **Rule.** Fit the shadow camera to a **sphere**, not a box: a sphere is invariant under the sun's
  rotation, so the cascade does not resize as the day advances and the texel size holds still
  (`src/lighting/lighting.js:871-874`).

---

## 7. Vertex colours: 3 vs 4 components are different shader programs

Selection happens on `itemSize`:

```js
vertexAlphas: material.vertexColors === true && !! geometry.attributes.color && geometry.attributes.color.itemSize === 4
```
`src/renderers/webgl/WebGLPrograms.js:309` → `#define USE_COLOR_ALPHA`
(`src/renderers/webgl/WebGLProgram.js:568, 738`).

| define | vertex chunk (`color_vertex.glsl.js`) | fragment (`color_fragment.glsl.js`) |
|---|---|---|
| `USE_COLOR` (itemSize 3) | `vColor.rgb *= color;` | `diffuseColor *= vColor;` (alpha stays 1) |
| `USE_COLOR_ALPHA` (itemSize 4) | `vColor *= color;` | `diffuseColor *= vColor;` — **alpha included** |

And `diffuseColor.a` becomes the fragment's output alpha:
`gl_FragColor = vec4( outgoingLight, diffuseColor.a );`
(`src/renderers/shaders/ShaderChunk/opaque_fragment.glsl.js:10`), where
`outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance`
(`src/renderers/shaders/ShaderLib/meshphysical.glsl.js:198`).

**Rule.** With `transparent: true` and standard SrcAlpha blending, the per-vertex alpha scales
**everything the fragment emits, emissive included**. That is a free, geometry-driven emissive
falloff with no texture and no extra draw.

*This project* — the hearth ember bed. Three rings, alpha `1.0 / 0.78 / 0.0`, colour warm→cool, one
14-segment fan (`EMBER_TRIS = 42`):

```js
col.push(ring.warm, ring.warm * 0.42, ring.warm * 0.12, ring.a);
g.setAttribute('color', new THREE.Float32BufferAttribute(col, 4));
```
`src/world/furnishings.js:315, 331`, material at `:2637-2640`
(`transparent: true, depthWrite: false, vertexColors: true`). Rationale in the doc comment at
`:285-291`. `depthWrite: false` is required or the overlapping cards punch holes in each other.

### The black-mesh trap

**Rule.** `vertexColors: true` with **no** `color` attribute renders every vertex **black** on
WebGL2. The vertex prefix declares `attribute vec3 color;`
(`src/renderers/webgl/WebGLProgram.js:657`; the vec4 form at `:653`), and an unbound generic vertex
attribute reads `(0,0,0,1)` — so `vColor.rgb *= color` zeroes it.

Two mitigations, both in this repo:

1. Strip the flag from materials whose geometry can't honour it —
   `sanitize()` at `src/world/vegetation.js:498-508` clears `vertexColors`, `aoMap`, `lightMap`
   (the map slots also need a `uv1` set the cards don't have).
2. Or attach a real all-white attribute so the multiply is a no-op —
   `withInstanceColor()` at `src/world/vegetation.js:372-376`.

**A related asymmetry worth knowing** (verified, and it is why option 2 is called "version-proof"):
the **fragment** prefix defines `USE_COLOR` for `vertexColors || instancingColor`
(`WebGLProgram.js:737`) while the **vertex** prefix defines it for `vertexColors` only (`:567`) and
handles instancing via `USE_INSTANCING_COLOR` (`:486`, chunk `color_vertex.glsl.js`). So an
`InstancedMesh` with `instanceColor` gets per-instance tint even with `vertexColors: false` — but
that is an implementation detail of r185, not a contract. The project notes this at
`src/world/vegetation.js:366-371` and still pairs the flag with a white attribute.

---

## 8. Colour construction: hex is sRGB, three floats are LINEAR

| call | interpreted as | library:line |
|---|---|---|
| `new THREE.Color( 0xffb45a )` → `setHex` | **sRGB** (`colorSpace = SRGBColorSpace` default) | `src/math/Color.js:204` |
| `new THREE.Color( 1, 0.5, 0.2 )` → `setRGB` | **working space = LinearSRGB** | `src/math/Color.js:173, 227` |
| `color.setRGB( r, g, b, colorSpace )` | 4th arg | `src/math/Color.js:227` |
| `color.getRGB( target, colorSpace )` | 2nd arg | `src/math/Color.js:619` |

`ColorManagement.enabled = true` and `workingColorSpace = LinearSRGBColorSpace` by default
(`src/math/ColorManagement.js:21, 23`).

**Rule.** The same triple written two ways gives two different colours. Always pass the colour space
explicitly for HDR/authored-linear values. This project does, everywhere it matters —
`new THREE.Color().setRGB(FLAME_GAIN, FLAME_GAIN, FLAME_GAIN, THREE.LinearSRGBColorSpace)`
(`src/world/furnishings.js:2618`), `fog.color.setRGB(0.40, 0.38, 0.34, THREE.LinearSRGBColorSpace)`
(`src/lighting/lighting.js:800`).

Textures: `colorSpace = SRGBColorSpace` for albedo, `NoColorSpace` for normal/roughness/metalness/AO
(`src/materials/textures.js:2007, 2110`), and `flipY = false` when the data came back from
`readRenderTargetPixels` because GL rows are already bottom-up (`:2008`).

---

## 9. Point/spot attenuation clamps — don't light your own emitter

```glsl
float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
```
`src/renderers/shaders/ShaderChunk/lights_pars_begin.glsl.js:61`

**Rule.** At `decay = 2` the falloff saturates at **100×** for anything within 0.1 m of the light.
A lit material sitting inside its own light source gets a ~100× diffuse boost and blows past any
bloom threshold. The fix is not a smaller number — it is to stop lighting it: use
`MeshBasicMaterial` (unlit), so the pixel is exactly texel × gain.

*This project*: the hearth flame card was a lit `lanternEmissive` variant 0.05–0.3 m from its own
`PointLight`; measured ~(5.3, 2.6, 1.2) diffuse on top of ~(3.6, 1.6, 0.4) emissive, luminance ≈ 5.0
— **twice** the bloom threshold, i.e. the white blob. Now `MeshBasicMaterial` + an alpha teardrop
texture. Full write-up `src/world/furnishings.js:377-390`.

Note lights in r185 are physical units: `intensity: 18` on the lantern spot is candela
(`src/player/controls.js:77-84`).

---

## 10. Smaller verified facts

| fact | library / project |
|---|---|
| `EffectComposer` allocates its own targets as `HalfFloatType` when you don't supply one | `examples/jsm/postprocessing/EffectComposer.js:69` — this is what makes HDR-through-the-chain work by default |
| Composer ping-pong targets carry **stale depth** from the `RenderPass`; a full-screen quad must set `depthTest = false, depthWrite = false` | `src/post/post.js:625-627` |
| `mergeGeometries` returns **`null`** (with a `console.error`) on mismatched attribute sets or mixed indexed/non-indexed input — it does not throw | `examples/jsm/utils/BufferGeometryUtils.js:156-172`; project strips to position-only before merging and null-checks: `src/contracts.js:100-107` |
| `Object3D.rotateX(+t)` sends `+Z` to `(0, −sin t, cos t)` — it tips `+Z` **down** | derived from `Matrix4.makeRotationX`; project note `src/world/interiors.js:984-985` (roof rafters had the wrong sign) |
| `renderer.info` does not auto-reset when you drive a composer; set `info.autoReset = false` and call `info.reset()` yourself immediately before `composer.render()` | `src/core/engine.js:39`, `src/main.js:236-237`, `src/perf/perf.js:323-324` |
| An `InstancedMesh`'s **geometry** boundingSphere covers ONE instance; the **mesh** carries `.boundingSphere` over all of them | `src/post/post.js:523-533` — getting this wrong culls or fails to cull the whole herd |
| three does **not** frustum-cull instances individually; one casting `InstancedMesh` re-submits all its instances into the shadow map | project consequence: mid-LOD trees are built `cast: false` (`src/world/vegetation.js:1461-1465`) |
| Alpha-tested foliage needs `shadowSide = DoubleSide` or half the canopy drops no shadow | `src/materials/materials.js:260-266` |
| A wind/vertex-displacement `onBeforeCompile` patch must be applied to a matching `customDepthMaterial` too, or shadows detach from the leaves | `src/world/vegetation.js:483-493, 1314`; the depth material must also mirror `map`/`alphaMap`/`alphaTest`/`side` |
| `onBeforeCompile` needs a `customProgramCacheKey` or variants share a compiled program | `src/world/vegetation.js:484, 492` |
| Sky-dome far-plane pin: `gl_Position.z = gl_Position.w;` after the MVP multiply — the dome can never be clipped by `camera.far` whatever its radius | `src/lighting/sky.js:549-552` |
| GPU timing: `EXT_disjoint_timer_query_webgl2`, and **only one `TIME_ELAPSED` query may be open at a time** — profile passes round-robin, one per frame | `src/post/post.js:299-323` |

---

## How to re-verify all of this on a new three version — 60 seconds

Run from the project root. Each grep either reproduces the quoted line or tells you the fact has
moved.

```bash
T=node_modules/three; S=$T/src; X=$T/examples/jsm; C=$S/renderers/shaders/ShaderChunk

# §1 tone map skipped into a target (expect the currentRenderTarget === null guard, twice)
grep -n "toneMapping = NoToneMapping" -A 8 $S/renderers/webgl/WebGLPrograms.js $S/renderers/WebGLRenderer.js
grep -n "getUnlitUniformColorSpace" -A 8 $S/renderers/shaders/UniformsUtils.js

# §2 is PCFSoft still deprecated, and what does PCF compile to?
grep -n "PCFSoftShadowMap\|this.type = " $S/renderers/webgl/WebGLShadowMap.js
grep -n "vogelDiskSample\|interleavedGradientNoise\|shadowRadius \* texelSize" $C/shadowmap_pars_fragment.glsl.js

# §3 pass constructors + ordering docs + the bloom high-pass mix
grep -n "constructor(" $X/postprocessing/{OutputPass,SMAAPass,FXAAPass,UnrealBloomPass,GTAOPass}.js
grep -n "must be executed before\|requires sRGB input" $X/postprocessing/{SMAAPass,OutputPass}.js
grep -n "smoothWidth\|luminosityThreshold" $X/postprocessing/UnrealBloomPass.js $X/shaders/LuminosityHighPassShader.js
grep -n "GTAOPass.OUTPUT" -A 10 $X/postprocessing/GTAOPass.js

# §4 addons resolution
node -e "console.log(require('$PWD/$T/package.json').exports)"

# §5 fromCubemap first-call-null, and scene.environmentIntensity vs envMap===null
grep -n "fromCubemap\|renderTarget || this._allocateTargets\|_lodMeshes\[ 0 \]" $S/extras/PMREMGenerator.js
grep -n "envMapIntensity.value = scene.environmentIntensity" -B 3 $S/renderers/WebGLRenderer.js

# §6 directional direction = position - target
grep -n "uniforms.direction" -A 3 $S/renderers/webgl/WebGLLights.js
grep -n "_lookTarget\|shadowCamera.lookAt" $S/lights/LightShadow.js

# §7 vertex colour alpha       §8 colour spaces       §9 attenuation clamp
grep -n "itemSize === 4" $S/renderers/webgl/WebGLPrograms.js
grep -n "USE_COLOR_ALPHA\|USE_COLOR'" $S/renderers/webgl/WebGLProgram.js
cat $C/color_vertex.glsl.js $C/opaque_fragment.glsl.js
grep -n "setHex( hex\|setRGB( r, g, b" -A 2 $S/math/Color.js
grep -n "enabled:\|workingColorSpace:" $S/math/ColorManagement.js
grep -n "distanceFalloff = 1.0 / max" $C/lights_pars_begin.glsl.js
```

Runtime check that costs one frame and catches a silent shadow-type downgrade:

```js
console.assert(renderer.shadowMap.type === THREE.PCFShadowMap, 'shadow type was substituted');
console.log('sun dir', sun.position.clone().sub(sun.target.getWorldPosition(new THREE.Vector3())).normalize());
```

**If a fact has moved, distrust the transferable rule too, not just the line number.** The rules in
sections 1, 2, 5 and 7 are all consequences of implementation choices three has changed before.
