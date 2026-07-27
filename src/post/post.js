/**
 * ============================================================================
 *  POST — the composer chain
 * ============================================================================
 * Nothing outside this file touches the EffectComposer (contracts.js rule 4).
 *
 * Chain (in order):
 *
 *   RenderPass          scene -> HDR half-float target, NO tone map
 *   GTAOPass            optional; short-radius crevice occlusion, rendered at a
 *                       fraction of the framebuffer with the far half of the
 *                       village culled out of its gbuffer
 *   UnrealBloomPass     highlight bloom, threshold ABOVE the sky, soft knee
 *   SMAAPass | FXAAPass antialias (SMAA must run before OutputPass — it works
 *                       in linear-srgb; three's own docs say so)
 *   GradePass           contrast + saturation + shadow toe + vignette + grain
 *                       + chromatic aberration
 *   OutputPass          THE tone map + sRGB encode. Exactly one, here.
 *
 * Every pass reports its own draw calls, triangles, CPU submit time and (where
 * the driver allows) GPU time into `stats.passCost`, which the perf overlay
 * prints. Attribution beats guessing when the draw-call budget goes over.
 *
 * Tone mapping: `renderer.toneMapping` is ACESFilmic, but three skips the tone
 * map in material shaders whenever it renders into a render target, so the
 * whole chain is genuine linear HDR and OutputPass does the single conversion
 * at the end. Never set `toneMapping` on a pass and never add a second map.
 *
 * The grade pass sits BEFORE the tone map deliberately: a vignette is lens
 * falloff, which is a scene-referred multiply, and multiplicative grain reads
 * as film response rather than as noise pasted onto a finished image.
 * ============================================================================
 */

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { SMAAPass } from 'three/addons/postprocessing/SMAAPass.js';
import { FXAAPass } from 'three/addons/postprocessing/FXAAPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

/* -------------------------------------------------------------------------- */
/* Tuning                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * Bloom threshold, in linear scene-referred luminance.
 *
 * MEASURED, not guessed: the sky shader writes `SKY_RADIANCE_SCALE`-scaled
 * Preetham radiance with a sun disc at 46 and a lit cumulus deck that sits
 * comfortably between 1.5 and 4. The old 1.25 threshold therefore put the
 * ENTIRE sky over the line — and because three's luminosity high pass passes
 * the whole texel through once it is over threshold (it is a mix, not a
 * subtract), every roofline seen against the sky grew a white halo the width of
 * the blur kernel.
 *
 * So the threshold now sits ABOVE the sky dome, and `knee` is fed to the high
 * pass's `smoothWidth` so the transition is a ramp rather than a cliff: a cloud
 * at 3 contributes ~2%, the brightest cloud edge near the sun ~50%, the sun
 * disc fully. Radius is down from 0.5 to 0.35 to stop what is left from
 * creeping across a silhouette.
 *
 * Cost of being honest: the lantern emissives are only ~1.7 in luminance
 * (0xffb45a at intensity 3.2), so they no longer bloom on their own. Killing
 * the halo is worth more than a glow on four lanterns, and the fix belongs in
 * the lantern material (a hotter core), not in a threshold that also catches
 * 100% of the sky.
 */
const BLOOM_THRESHOLD = { low: 2.4, medium: 2.5, high: 2.6, ultra: 2.7 };
/** Soft knee width, added to the threshold: full bloom only at t + knee. */
const BLOOM_KNEE = { low: 2.2, medium: 2.3, high: 2.4, ultra: 2.4 };
const BLOOM_RADIUS = 0.35;

/** GTAO. Short radius — this is crevice occlusion, not a fake GI darkening. */
const GTAO_AO = {
  radius: 0.35,          // metres. Cobble gaps, window reveals, under eaves.
  distanceExponent: 1.6, // falls off fast, so distant geometry contributes ~0
  thickness: 0.45,
  scale: 1.0,
  distanceFallOff: 1.0,
  screenSpaceRadius: false,
};
const GTAO_PD = {        // Poisson denoise — tight, so AO stays a contact effect
  lumaPhi: 6,
  depthPhi: 1.5,
  normalPhi: 3.5,
  radius: 3.5,
  rings: 2,
};
/** < 1 keeps AO off the silhouettes, which is where the dark halo comes from. */
const GTAO_BLEND = 0.85;

/**
 * GTAO internal resolution, as a fraction of the framebuffer. The pass owns
 * three targets AND re-renders the whole scene into a normal+depth gbuffer, so
 * it is the single most expensive thing in the chain. A 0.75 scale costs 56% of
 * the fill and is invisible: the effect is a 0.35 m contact darkening that is
 * Poisson-denoised (i.e. deliberately blurred) immediately afterwards.
 */
const GTAO_SCALE = { low: 0.5, medium: 0.6, high: 0.75, ultra: 0.85 };

/**
 * Metres. Geometry further away than this is hidden during GTAO's gbuffer pass:
 * a 0.35 m occlusion radius at 55 m is far below one pixel, so those objects
 * can only cost draw calls, never change the image. Their depth stays cleared,
 * which reads as "unoccluded" — exactly what they would have been.
 */
const GTAO_CULL_DISTANCE = 55;

/** Named subtrees that can never contribute contact AO. Skipped in the gbuffer. */
const GTAO_SKIP_NAMES = new Set(['sky', 'terrain.far', 'terrain.mountains']);

/**
 * Grade. Runs after bloom, before the tone map, so `contrast` is a power
 * function in linear light — a straight line in log-log, which is what a film
 * characteristic curve actually is.
 *
 * `shadows` was 0.10-0.14 and it was measurably too much: with contrast,
 * saturation and the toe all flattened the ground-half mean screen luminance
 * went 0.242 -> 0.276, and the toe was most of that 13%. It was too much
 * because of WHERE it applied, not only how hard — the old SHADOW_TOE of 0.16
 * is scene-referred mid grey, so "shadows" was rolling down the entire lower
 * half of the picture, the sunlit setts included. Toe is now 0.06 (about 1.6
 * stops under mid grey: eave undersides, doorways, interiors) and the depth is
 * 0.05-0.06, which still reads as a toe and no longer taxes the ground.
 *
 * Contrast and saturation are unchanged — they are what makes the picture snap.
 */
const GRADE_DEFAULTS = {
  low:    { vignette: 0.20, grain: 0.020, aberration: 0.0,    contrast: 1.10, saturation: 1.06, shadows: 0.05 },
  medium: { vignette: 0.22, grain: 0.030, aberration: 0.0012, contrast: 1.14, saturation: 1.09, shadows: 0.055 },
  high:   { vignette: 0.24, grain: 0.034, aberration: 0.0016, contrast: 1.17, saturation: 1.10, shadows: 0.06 },
  ultra:  { vignette: 0.25, grain: 0.036, aberration: 0.0018, contrast: 1.18, saturation: 1.11, shadows: 0.06 },
};

/* -------------------------------------------------------------------------- */
/* Grade shader                                                                */
/* -------------------------------------------------------------------------- */

const GradeShader = {
  name: 'HollowbrookGrade',

  uniforms: {
    tDiffuse: { value: null },
    uResolution: { value: new THREE.Vector2(1, 1) },
    uTime: { value: 0 },
    uVignette: { value: 0.24 },
    uGrain: { value: 0.034 },
    uAberration: { value: 0.0016 },
    uContrast: { value: 1.17 },
    uSaturation: { value: 1.10 },
    uShadows: { value: 0.06 },
  },

  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
    }
  `,

  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse;
    uniform vec2  uResolution;
    uniform float uTime;
    uniform float uVignette;
    uniform float uGrain;
    uniform float uAberration;
    uniform float uContrast;
    uniform float uSaturation;
    uniform float uShadows;

    varying vec2 vUv;

    const vec3  LUMA  = vec3( 0.2126, 0.7152, 0.0722 );

    // Contrast pivot: the one luminance the curve leaves exactly where it is.
    //
    // It was 0.18 — scene-referred mid grey, the textbook answer — but the
    // textbook assumes the subject sits AT mid grey. It does not here: the
    // sunlit plaza setts need to be 0.174-0.248 scene-referred to land in the
    // 0.50-0.62 sRGB target, and the ground-half average sits well below that,
    // so with a 0.18 pivot every ground pixel was on the falling side of the
    // curve. 0.14 puts the pivot at the bottom of the ground's range instead.
    //
    // Measured through three's ACES fit + sRGB encode, contrast 1.17: the pivot
    // move alone is +5.0% at scene 0.03, +3.9% at 0.08, +2.3% at 0.20, and
    // +1.3% at 0.40 — it lifts the ground and barely touches the highlights,
    // which is the whole point. Sunlit plaster at scene 0.74/0.69/0.58 still
    // encodes to 226,223,216, so nothing new clips.
    const float PIVOT = 0.14;

    // Luminance below which the shadow roll-down applies in full.
    //
    // Was 0.16, which is mid grey — so the "shadow" toe was rolling down the
    // whole lower half of the frame, sunlit ground included, and cost ~13% of
    // the ground luminance. At 0.06 it is a genuine toe: full effect only in
    // eave undersides, doorways and interiors, nothing at all on lit setts.
    const float SHADOW_TOE = 0.06;

    // Cheap 3D hash -> [0,1). Good enough for grain; no texture fetch.
    float hash13( vec3 p ) {
      p = fract( p * 0.1031 );
      p += dot( p, p.yzx + 33.33 );
      return fract( ( p.x + p.y ) * p.z );
    }

    void main() {
      vec2 uv = vUv;
      vec2 c  = uv - 0.5;

      // Aspect-correct radius so the vignette is round, not oval.
      float aspect = uResolution.x / max( uResolution.y, 1.0 );
      vec2  ca = vec2( c.x * aspect, c.y );
      float r2 = dot( ca, ca );

      vec4 texel;
      // uAberration is a uniform, so this branch is coherent across the draw.
      if ( uAberration > 0.0 ) {
        // Transverse CA: the sampling scale grows with r^2, like a real lens.
        vec2 offs = c * ( r2 * uAberration );
        vec4 g = texture2D( tDiffuse, uv );
        texel = vec4(
          texture2D( tDiffuse, uv + offs ).r,
          g.g,
          texture2D( tDiffuse, uv - offs ).b,
          g.a
        );
      } else {
        texel = texture2D( tDiffuse, uv );
      }

      vec3 color = max( texel.rgb, vec3( 0.0 ) );

      // --- filmic contrast: a power about PIVOT. Cheap (one pow3), stable in
      //     HDR, and it cannot clip because the tone map still follows.
      if ( uContrast != 1.0 ) {
        color = PIVOT * pow( color / PIVOT, vec3( uContrast ) );
      }

      // --- shadows: pull the toe down a little further than the curve does, so
      //     the deep interiors and the undersides of the eaves read as dark
      //     rather than as flat grey. Smooth, so nothing crushes to black.
      //     SHADOW_TOE is deliberately far below the ground's luminance range:
      //     this must shape the interiors, never tax the lit plaza.
      if ( uShadows > 0.0 ) {
        float sl = dot( color, LUMA );
        color *= mix( 1.0 - uShadows, 1.0, smoothstep( 0.0, SHADOW_TOE, sl ) );
      }

      // --- saturation: a small lift about luminance. The setts and the thatch
      //     are close in hue, and a flat grade lets them merge.
      if ( uSaturation != 1.0 ) {
        float lumS = dot( color, LUMA );
        color = max( mix( vec3( lumS ), color, uSaturation ), vec3( 0.0 ) );
      }

      // --- vignette: scene-referred multiply (lens falloff, pre tone map).
      float v = smoothstep( 1.30, 0.18, r2 );
      color *= mix( 1.0, v, uVignette );

      // --- grain: multiplicative, biased into the shadows and midtones where
      //     film actually shows it. gl_FragCoord keeps it pixel-locked, and a
      //     24 Hz time quantisation stops it shimmering at high frame rates.
      if ( uGrain > 0.0 ) {
        float n = hash13( vec3( gl_FragCoord.xy, floor( uTime * 24.0 ) ) ) - 0.5;
        float luma = dot( color, LUMA );
        float resp = 1.0 - smoothstep( 0.0, 1.5, luma );
        color *= 1.0 + n * uGrain * ( 0.30 + 0.70 * resp );
      }

      gl_FragColor = vec4( max( color, vec3( 0.0 ) ), texel.a );
    }
  `,
};

/* -------------------------------------------------------------------------- */
/* Per-pass accounting                                                         */
/* -------------------------------------------------------------------------- */

/* Scratch — the frame path must never allocate. */
const _camPos = new THREE.Vector3();
const _centre = new THREE.Vector3();
const _scale3 = new THREE.Vector3();

/**
 * Attributes draw calls, triangles, CPU submit time and (where the driver
 * exposes timer queries) real GPU time to each pass, so a draw-call regression
 * can be blamed on the stream that caused it instead of on "post".
 *
 * Draw calls and triangles are exact and free: `engine.js` sets
 * `renderer.info.autoReset = false` and `main.js` resets once per frame, so the
 * counters accumulate across every nested `renderer.render` and a before/after
 * delta round a pass is precisely that pass's cost.
 *
 * GPU time uses EXT_disjoint_timer_query_webgl2 when present. Only one
 * TIME_ELAPSED query may be open at a time, so passes are measured round-robin,
 * one per frame; each number is therefore a few frames stale, which is fine for
 * an overlay refreshed at 5 Hz.
 */
function createPassProfiler(renderer) {
  const recs = [];
  const byKey = new Map();

  let gl = null;
  let ext = null;
  try {
    gl = renderer.getContext();
    if (gl && typeof gl.createQuery === 'function') {
      ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
    }
  } catch {
    ext = null;
  }

  const inFlight = [];
  let active = null;
  let targetKey = null;
  let cursor = 0;

  function add(key) {
    let rec = byKey.get(key);
    if (!rec) {
      rec = { key, draws: 0, tris: 0, cpuMs: 0, gpuMs: 0, ran: false, enabled: true };
      byKey.set(key, rec);
      recs.push(rec);
    }
    return rec;
  }

  function beginFrame() {
    if (!ext || !recs.length) return;
    // Only ever profile a pass that actually ran last frame.
    for (let i = 0; i < recs.length; i++) {
      const rec = recs[(cursor + i) % recs.length];
      if (rec.enabled) { targetKey = rec.key; cursor = (cursor + i + 1) % recs.length; return; }
    }
    targetKey = null;
  }

  function beginPass(key) {
    if (!ext || active || key !== targetKey) return;
    try {
      const query = gl.createQuery();
      gl.beginQuery(ext.TIME_ELAPSED_EXT, query);
      active = { key, query };
    } catch {
      active = null;
      ext = null;           // a driver that throws once will throw forever
    }
  }

  function endPass() {
    if (!active) return;
    try {
      gl.endQuery(ext.TIME_ELAPSED_EXT);
      inFlight.push(active);
      if (inFlight.length > 8) {
        const drop = inFlight.shift();
        gl.deleteQuery(drop.query);
      }
    } catch {
      ext = null;
    }
    active = null;
  }

  function poll() {
    if (!ext || !inFlight.length) return;
    for (let i = inFlight.length - 1; i >= 0; i--) {
      const item = inFlight[i];
      let done = false;
      try {
        done = gl.getQueryParameter(item.query, gl.QUERY_RESULT_AVAILABLE);
      } catch {
        ext = null;
        return;
      }
      if (!done) continue;
      const disjoint = gl.getParameter(ext.GPU_DISJOINT_EXT);
      if (!disjoint) {
        const ns = gl.getQueryParameter(item.query, gl.QUERY_RESULT);
        const rec = byKey.get(item.key);
        if (rec) {
          const ms = ns / 1e6;
          // Light smoothing: one sample every recs.length frames is noisy.
          rec.gpuMs = rec.gpuMs > 0 ? rec.gpuMs + (ms - rec.gpuMs) * 0.4 : ms;
        }
      }
      gl.deleteQuery(item.query);
      inFlight.splice(i, 1);
    }
  }

  function dispose() {
    if (!gl) return;
    for (const item of inFlight) { try { gl.deleteQuery(item.query); } catch { /* gone */ } }
    inFlight.length = 0;
  }

  return {
    add,
    beginFrame,
    beginPass,
    endPass,
    poll,
    dispose,
    recs,
    get hasGpuTiming() { return !!ext; },
  };
}

/* -------------------------------------------------------------------------- */
/* Factory                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Build the post chain.
 *
 * @param {Object}   o
 * @param {THREE.WebGLRenderer} o.renderer
 * @param {THREE.Scene}         o.scene
 * @param {THREE.PerspectiveCamera} o.camera
 * @param {Object}   o.quality   active preset from config.js (has `.name`)
 * @param {Object}   o.engine    core/engine.js — used for onResize
 * @returns {{composer:EffectComposer, render:Function, setSize:Function,
 *            setBloom:Function, setAO:Function, setQuality:Function,
 *            setGrade:Function, passes:Object, dispose:Function, stats:Object}}
 */
export function createPostProcessing({ renderer, scene, camera, quality, engine }) {
  const q = quality || {};
  const name = q.name || 'medium';

  let width = Math.max(1, window.innerWidth);
  let height = Math.max(1, window.innerHeight);
  let pixelRatio = renderer.getPixelRatio();

  const composer = new EffectComposer(renderer);
  composer.setPixelRatio(pixelRatio);
  composer.setSize(width, height);

  const ew = () => Math.max(1, Math.round(width * pixelRatio));
  const eh = () => Math.max(1, Math.round(height * pixelRatio));

  /* ------------------------------------------------------- instrumentation */
  const profiler = createPassProfiler(renderer);
  // Registered up front so the overlay prints them in chain order rather than
  // in the order they happened to be constructed. A pass that never runs is
  // marked idle on its first frame and drops out of the report.
  for (const k of ['render', 'gtao', 'bloom', 'aa', 'grade', 'output']) profiler.add(k);

  /**
   * Wrap a pass's `render` so it accounts for itself. Idempotent, and it must
   * stay allocation-free: it runs 6x per frame.
   */
  function instrument(pass, key) {
    if (!pass || pass.__hbProfiled) return pass;
    const rec = profiler.add(key);
    const inner = pass.render.bind(pass);
    pass.__hbProfiled = key;
    pass.render = function profiledRender(r, writeBuffer, readBuffer, deltaTime, maskActive) {
      const info = renderer.info.render;
      const d0 = info.calls;
      const t0 = info.triangles;
      const c0 = performance.now();
      profiler.beginPass(key);
      inner(r, writeBuffer, readBuffer, deltaTime, maskActive);
      profiler.endPass();
      rec.cpuMs = performance.now() - c0;
      rec.draws = info.calls - d0;
      rec.tris = info.triangles - t0;
      rec.ran = true;
      return undefined;
    };
    return pass;
  }

  /** A pass that did not run this frame must report zero, not its last value. */
  function clearIdle() {
    for (const rec of profiler.recs) {
      if (rec.ran) { rec.ran = false; rec.enabled = true; continue; }
      rec.enabled = false;
      rec.draws = 0; rec.tris = 0; rec.cpuMs = 0; rec.gpuMs = 0;
    }
  }

  /* ------------------------------------------------------------ 1. render */
  const renderPass = new RenderPass(scene, camera);
  instrument(renderPass, 'render');

  /* --------------------------------------------------------------- 2. AO */
  // Built lazily: GTAO owns three full-resolution half-float targets plus a
  // depth texture (~50 MB at 1080p), which is not worth reserving on a preset
  // that has AO switched off and may never switch it on.
  let gtaoPass = null;
  let gtaoBroken = false;
  const aoScale = GTAO_SCALE[name] ?? 0.75;

  /**
   * GTAOPass re-renders the entire scene into a normal+depth gbuffer every
   * frame — that is where its draw calls go, and at 139 building meshes it is
   * not small. Nothing beyond `GTAO_CULL_DISTANCE` can change the result of a
   * 0.35 m occlusion radius, and the sky and the far scenery can never change
   * it at all, so both are hidden for the duration of that pass only.
   *
   * `_overrideVisibility` already exists to hide points and lines, and
   * `_restoreVisibility` un-hides everything its cache holds — so appending to
   * that same cache is the sanctioned hook, not a hack around one.
   */
  function cullGBuffer(cache) {
    _camPos.setFromMatrixPosition(camera.matrixWorld);
    const cut = GTAO_CULL_DISTANCE;
    scene.traverse((o) => {
      if (!o.visible) return;
      if (o.userData.noAO || GTAO_SKIP_NAMES.has(o.name)) {
        o.visible = false;
        cache.push(o);
        return;
      }
      if (!o.isMesh && !o.isInstancedMesh) return;
      // An InstancedMesh's geometry sphere covers ONE instance; the mesh keeps
      // its own sphere over all of them.
      let bs = null;
      if (o.isInstancedMesh) {
        if (!o.boundingSphere) o.computeBoundingSphere?.();
        bs = o.boundingSphere;
      } else {
        if (!o.geometry.boundingSphere) o.geometry.computeBoundingSphere?.();
        bs = o.geometry.boundingSphere;
      }
      if (!bs) return;
      _centre.copy(bs.center).applyMatrix4(o.matrixWorld);
      _scale3.setFromMatrixScale(o.matrixWorld);
      const r = bs.radius * Math.max(_scale3.x, _scale3.y, _scale3.z);
      if (_centre.distanceTo(_camPos) - r > cut) {
        o.visible = false;
        cache.push(o);
      }
    });
  }

  function ensureGTAO() {
    if (gtaoPass || gtaoBroken) return gtaoPass;
    try {
      const p = new GTAOPass(scene, camera, Math.round(ew() * aoScale), Math.round(eh() * aoScale));
      p.output = GTAOPass.OUTPUT.Default;
      p.blendIntensity = GTAO_BLEND;
      const samples = name === 'ultra' ? 16 : 12;
      p.updateGtaoMaterial({ ...GTAO_AO, samples });
      p.updatePdMaterial({ ...GTAO_PD, samples });

      // composer.setSize / insertPass hand every pass the full framebuffer size,
      // so the scale has to live in setSize or the first resize undoes it.
      const baseSetSize = p.setSize.bind(p);
      p.setSize = (w, h) => baseSetSize(
        Math.max(1, Math.round(w * aoScale)),
        Math.max(1, Math.round(h * aoScale)),
      );

      const baseOverride = p._overrideVisibility.bind(p);
      p._overrideVisibility = () => {
        baseOverride();
        try { cullGBuffer(p._visibilityCache); } catch (err) {
          console.warn('[post] GTAO gbuffer cull failed; falling back to the full scene:', err);
          p._overrideVisibility = baseOverride;
        }
      };

      // The pass copies camera.near/far/projection into its uniforms on every
      // render, so its depth reconstruction can never drift from the camera.
      gtaoPass = instrument(p, 'gtao');
    } catch (err) {
      gtaoBroken = true;
      console.warn('[post] GTAOPass unavailable, continuing without AO:', err);
    }
    return gtaoPass;
  }

  /* ------------------------------------------------------------ 3. bloom */
  const bloomStrength = Number.isFinite(q.bloom) ? q.bloom : 0.5;
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(ew(), eh()),
    bloomStrength,
    BLOOM_RADIUS,
    BLOOM_THRESHOLD[name] ?? 2.6,
  );
  bloomPass.enabled = bloomStrength > 0.001;
  // `smoothWidth` is the high pass's knee. UnrealBloomPass only ever re-writes
  // `luminosityThreshold` per frame, so a value set here sticks.
  let bloomKnee = BLOOM_KNEE[name] ?? 2.4;
  bloomPass.highPassUniforms.smoothWidth.value = bloomKnee;
  instrument(bloomPass, 'bloom');

  /* --------------------------------------------------------------- 4. AA */
  let aaMode = q.antialias === 'fxaa' ? 'fxaa' : q.antialias === 'none' ? 'none' : 'smaa';
  let aaPass = makeAA(aaMode);

  function makeAA(mode) {
    try {
      if (mode === 'fxaa') return instrument(new FXAAPass(), 'aa');
      if (mode === 'smaa') return instrument(new SMAAPass(), 'aa');
    } catch (err) {
      console.warn('[post] antialias pass failed, running without:', err);
    }
    return null;
  }

  /* ------------------------------------------------------------ 5. grade */
  /**
   * Mirror of the clamped grade uniforms, published on `stats.grade` so the perf
   * overlay can print the live tuning state without reaching into the shader.
   * `setGrade` is the only writer, so this can never drift from the uniforms.
   * Declared here because `setGrade` runs before `stats` exists.
   */
  const gradeState = {
    vignette: 0, grain: 0, aberration: 0,
    contrast: 1, saturation: 1, shadows: 0, enabled: true,
  };

  const gradePass = new ShaderPass(GradeShader);
  // Full-screen quads have no business testing or writing depth; the composer's
  // ping-pong targets carry stale depth from the RenderPass.
  gradePass.material.depthTest = false;
  gradePass.material.depthWrite = false;
  gradePass.uniforms.uResolution.value.set(ew(), eh());
  // `setGrade` is a hoisted declaration, so the preset is applied through the
  // same path the settings panel uses — one place that clamps, one place to fix.
  setGrade(GRADE_DEFAULTS[name] || GRADE_DEFAULTS.high);
  instrument(gradePass, 'grade');

  /* ----------------------------------------------------------- 6. output */
  // Reads renderer.toneMapping / outputColorSpace. The only tone map.
  const outputPass = new OutputPass();
  instrument(outputPass, 'output');

  /* ---------------------------------------------------------- assembly */
  composer.addPass(renderPass);
  let aoEnabled = false;
  if (q.gtao && ensureGTAO()) { composer.addPass(gtaoPass); aoEnabled = true; }
  composer.addPass(bloomPass);
  if (aaPass) composer.addPass(aaPass);
  composer.addPass(gradePass);
  composer.addPass(outputPass);

  /* -------------------------------------------------------------------- */
  /* Public surface                                                        */
  /* -------------------------------------------------------------------- */

  const stats = {
    passes: composer.passes.length,
    gtao: aoEnabled,
    bloom: bloomPass.strength,
    bloomThreshold: bloomPass.threshold,
    aa: aaPass ? aaMode : 'none',
    cpuMs: 0,
    width: ew(),
    height: eh(),
    pixelRatio,
    aoScale,
    /**
     * Tone-map exposure, refreshed every frame in `render`. The lighting stream
     * owns `renderer.toneMappingExposure`, so this is a read-only mirror — but it
     * belongs in the overlay, because it and `grade` together are the entire
     * reason a ground measurement comes out where it does.
     */
    exposure: renderer.toneMappingExposure,
    /** Live grade values, clamped as the shader sees them. Read, never mutate. */
    grade: gradeState,
    /** Live per-pass accounting — see createPassProfiler. Read, never mutate. */
    passCost: profiler.recs,
    gpuTiming: profiler.hasGpuTiming,
  };

  let broken = false;   // one render throw is enough; fall back forever after
  let time = 0;

  function syncStats() {
    stats.passes = composer.passes.length;
    stats.gtao = aoEnabled;
    stats.bloom = bloomPass.enabled ? bloomPass.strength : 0;
    stats.bloomThreshold = bloomPass.threshold;
    stats.bloomKnee = bloomKnee;
    stats.aa = aaPass && aaPass.enabled ? aaMode : 'none';
    stats.width = ew();
    stats.height = eh();
    stats.pixelRatio = pixelRatio;
    stats.gpuTiming = profiler.hasGpuTiming;
  }

  /** AO is always re-inserted immediately after the render pass. */
  function insertAO() {
    if (aoEnabled || !ensureGTAO()) return;
    const i = composer.passes.indexOf(renderPass);
    composer.insertPass(gtaoPass, i < 0 ? 1 : i + 1);   // insertPass sizes it
    aoEnabled = true;
    syncStats();
  }

  function removeAO() {
    if (!gtaoPass || !aoEnabled) return;
    composer.removePass(gtaoPass);
    aoEnabled = false;
    syncStats();
  }

  /**
   * Resize everything. `composer.setSize` forwards to every pass currently in
   * the chain; a detached pass is re-sized by `insertPass` when it comes back.
   */
  function setSize(w, h) {
    width = Math.max(1, w | 0);
    height = Math.max(1, h | 0);
    const pr = renderer.getPixelRatio();
    if (Math.abs(pr - pixelRatio) > 1e-4) {
      pixelRatio = pr;
      composer.setPixelRatio(pr);   // this also re-sizes at the old dimensions
    }
    composer.setSize(width, height);
    gradePass.uniforms.uResolution.value.set(ew(), eh());
    syncStats();
  }

  engine?.onResize?.((w, h) => setSize(w, h));

  /** Bloom strength. 0 removes the cost entirely rather than blurring for free. */
  function setBloom(strength) {
    const s = Math.max(0, Math.min(3, Number(strength) || 0));
    bloomPass.strength = s;
    bloomPass.enabled = s > 0.001;
    syncStats();
  }

  /**
   * @param {number} t     linear luminance where bloom starts
   * @param {number} [knee] width of the ramp to full bloom (defaults to the
   *                        preset's knee; 0 restores the hard cliff)
   */
  function setBloomThreshold(t, knee) {
    bloomPass.threshold = Math.max(0, Number(t) || 0);
    if (knee !== undefined) {
      bloomKnee = Math.max(0.001, Number(knee) || 0.001);
      bloomPass.highPassUniforms.smoothWidth.value = bloomKnee;
    }
    syncStats();
  }

  function setAO(on) {
    if (on) insertAO(); else removeAO();
  }

  /**
   * Every grade knob, all optional and all clamped to a range that cannot
   * produce a broken image: `contrast` 0.5-2, `saturation` 0-2, `shadows` 0-0.5.
   */
  function setGrade(opts = {}) {
    const u = gradePass.uniforms;
    if (opts.vignette !== undefined) u.uVignette.value = clamp(opts.vignette, 0, 1);
    if (opts.grain !== undefined) u.uGrain.value = Math.max(0, Number(opts.grain) || 0);
    if (opts.aberration !== undefined) u.uAberration.value = Math.max(0, Number(opts.aberration) || 0);
    if (opts.contrast !== undefined) u.uContrast.value = clamp(opts.contrast, 0.5, 2);
    if (opts.saturation !== undefined) u.uSaturation.value = clamp(opts.saturation, 0, 2);
    if (opts.shadows !== undefined) u.uShadows.value = clamp(opts.shadows, 0, 0.5);
    if (opts.enabled !== undefined) gradePass.enabled = !!opts.enabled;

    // Republish the clamped values, not the requested ones — the overlay should
    // show what the shader is actually doing.
    gradeState.vignette = u.uVignette.value;
    gradeState.grain = u.uGrain.value;
    gradeState.aberration = u.uAberration.value;
    gradeState.contrast = u.uContrast.value;
    gradeState.saturation = u.uSaturation.value;
    gradeState.shadows = u.uShadows.value;
    gradeState.enabled = gradePass.enabled !== false;
  }

  function clamp(v, lo, hi) {
    const n = Number(v);
    return !Number.isFinite(n) ? lo : n < lo ? lo : n > hi ? hi : n;
  }

  /**
   * Re-apply a whole preset. Only touches things that are cheap to change at
   * runtime — swapping the AA pass rebuilds one pass, never the chain.
   */
  function setQuality(next) {
    if (!next) return;
    if (Number.isFinite(next.bloom)) setBloom(next.bloom);
    if (next.name && BLOOM_THRESHOLD[next.name]) {
      setBloomThreshold(BLOOM_THRESHOLD[next.name], BLOOM_KNEE[next.name]);
    }
    if (next.gtao !== undefined) setAO(!!next.gtao);

    const wantAA = next.antialias === 'fxaa' ? 'fxaa'
      : next.antialias === 'none' ? 'none' : 'smaa';
    if (wantAA !== aaMode) {
      const idx = aaPass ? composer.passes.indexOf(aaPass) : composer.passes.indexOf(gradePass);
      if (aaPass) {
        composer.removePass(aaPass);
        aaPass.dispose?.();
      }
      aaMode = wantAA;
      aaPass = makeAA(aaMode);
      if (aaPass) composer.insertPass(aaPass, Math.max(0, idx));
    }

    const g = GRADE_DEFAULTS[next.name];
    if (g) setGrade(g);
    syncStats();
  }

  /**
   * main.js wires the settings panel by firing every control's handler once,
   * and the AO checkbox in index.html starts unchecked regardless of preset —
   * so a `high` load would silently lose its AO before the first frame.
   * `quality.gtao` is the source of truth, so re-assert it once, here.
   */
  let firstFrame = true;
  function healInitialState() {
    firstFrame = false;
    if (q.gtao && !aoEnabled) {
      insertAO();
      if (aoEnabled) console.info('[post] re-asserted GTAO from the quality preset');
    }
  }

  /** Drive the chain. No allocation here. */
  function render(dt) {
    if (firstFrame) healInitialState();
    const d = Number.isFinite(dt) ? dt : 0;
    time += d;
    if (time > 3600) time -= 3600;         // keep float precision in the grain
    gradePass.uniforms.uTime.value = time;
    // One property read per frame. The lighting stream can change exposure at
    // any time and there is no event for it, so poll rather than cache.
    stats.exposure = renderer.toneMappingExposure;

    profiler.beginFrame();
    const t0 = performance.now();
    if (broken) {
      renderer.setRenderTarget(null);
      renderer.render(scene, camera);
    } else {
      try {
        composer.render(d);
      } catch (err) {
        broken = true;
        console.error('[post] composer.render threw; falling back to a direct render:', err);
        renderer.setRenderTarget(null);
        renderer.render(scene, camera);
      }
    }
    stats.cpuMs = performance.now() - t0;
    clearIdle();
    profiler.poll();
  }

  function dispose() {
    profiler.dispose();
    for (const p of composer.passes.slice()) composer.removePass(p);
    renderPass.dispose?.();
    gtaoPass?.dispose?.();
    bloomPass.dispose?.();
    aaPass?.dispose?.();
    gradePass.dispose?.();
    outputPass.dispose?.();
    composer.dispose?.();
  }

  return {
    composer,
    render,
    setSize,
    setBloom,
    setBloomThreshold,
    setAO,
    setGrade,
    setQuality,
    dispose,
    stats,
    passes: {
      render: renderPass,
      // A getter, because GTAO is built lazily: a plain reference captured here
      // would stay null for the whole session on a preset that enables AO later.
      get gtao() { return gtaoPass; },
      bloom: bloomPass,
      get aa() { return aaPass; },
      grade: gradePass,
      output: outputPass,
    },
  };
}
