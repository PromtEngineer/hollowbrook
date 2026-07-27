/**
 * ============================================================================
 *  PERF — frame budget watchdog, adaptive resolution, overlay
 * ============================================================================
 * Two jobs:
 *
 *  1. Measure. Real wall-clock frame period (begin -> begin), CPU cost of the
 *     frame (begin -> end), CPU cost of the composer (post.stats.cpuMs), and
 *     whatever `renderer.info` and `physics.stats` will tell us.
 *
 *  2. Defend the budget. Resolution is scaled first, because it is the cheapest
 *     knob to move and the easiest to reverse. Only when it bottoms out do we
 *     start removing effects, in a fixed ladder, one step at a time, each step
 *     logged exactly once.
 *
 * Hysteresis is the whole design. A renderer that flips between 0.9 and 1.0
 * every second looks far worse than one that sits at 0.8 — so the trigger to
 * step DOWN is sustained (180 frames for the first step, 60 after), the trigger
 * to step UP is long (180 frames comfortably under), there is a dead band
 * between the two so a frame time that sits at 16.7 ms provokes nothing at all,
 * every change starts a 60-frame cooldown, and every oscillation makes the next
 * recovery stricter.
 *
 * Three rules exist purely so a fast machine is never punished for loading:
 *   - a warm-up of ~90 frames after the world appears is ignored outright, and
 *     re-armed when the player enters (pointer lock) or returns to the tab;
 *   - any single frame over 100 ms is a compilation hitch, not a slow GPU: it
 *     is kept in the history but excluded from the average and the streaks;
 *   - "slow" means 19.5 ms, not 16.6 — a vsync-locked 60 fps machine measures a
 *     16.67 ms period, and treating that as a miss walked the pixel ratio to
 *     the floor on hardware that was never dropping a frame.
 *
 * Only knobs listed in RUNTIME_KNOBS are ever written.
 * ============================================================================
 */

import { RUNTIME_KNOBS } from '../config.js';

/* -------------------------------------------------------------------------- */
/* Thresholds                                                                  */
/* -------------------------------------------------------------------------- */

const TUNE = {
  /** EMA weight for each new sample. ~16-frame time constant. */
  emaAlpha: 0.12,

  /**
   * Above this smoothed frame time we are missing 60 fps.
   *
   * NOT 16.6. A machine locked to a 60 Hz vsync measures a frame PERIOD of
   * 16.67 ms with jitter either side, so a 16.6 threshold declared every
   * perfectly healthy frame "slow" and walked the pixel ratio to the floor on a
   * machine sitting at a solid 60 fps. 19.5 ms ≈ 51 fps: genuinely missing.
   */
  slowMs: 19.5,
  /** Consecutive slow frames before stepping down. 60 = 1 s. */
  slowFrames: 60,
  /**
   * The FIRST down-step must be earned over 3 s. Early frames are polluted by
   * shader compilation, the first shadow bake and PMREM work, and a resolution
   * drop is the one change the player sees immediately.
   */
  firstSlowFrames: 180,

  /**
   * Frames ignored after the world becomes visible. ~1.5 s at 60 fps: long
   * enough to cover first-draw program compilation and the first shadow bake.
   */
  warmupFrames: 90,

  /**
   * A single frame this long is a compilation hitch, not a slow GPU. It is kept
   * in the history (so `worst` stays honest) but excluded from the EMA and from
   * the down-step streak — feeding a 300 ms compile into a 0.12 EMA poisons the
   * average for 30+ frames, which is exactly how the collapse to the floor
   * happened.
   */
  hitchMs: 100,
  /** Frames of quiet after a hitch before the governor is trusted again. */
  hitchCooldownFrames: 20,

  /** Comfortably under budget — ~5 ms of headroom before we risk spending it. */
  fastMs: 14.0,

  /**
   * A vsync-locked machine can never measure below ~16.7 ms however much
   * headroom it has, so resolution could never be won back at 60 Hz on frame
   * period alone. Treat "at the vsync period AND cheap on the CPU" as fast too.
   */
  vsyncMs: 17.6,
  cpuHeadroomMs: 10.0,
  /** Consecutive fast frames before stepping up. 180 ≈ 3 s. */
  fastFrames: 180,
  /** Each down-immediately-after-up multiplies the step-up requirement. */
  fastFramesGrowth: 1.6,
  fastFramesMax: 900,

  /** Frames ignored after any change (shader recompiles, render-target realloc). */
  cooldownFrames: 60,

  pixelRatioStep: 0.1,
  pixelRatioFloor: 0.65,

  /** A frame longer than this was a stall (tab switch, GC, module load) — discard. */
  outlierMs: 200,

  /** Slow streaks required *at the resolution floor* before dropping a tier. */
  floorStreaksPerTier: 2,
  /** A tier dropped more than this many times stays down for the session. */
  tierDropsBeforeLock: 3,

  historyLength: 120,
  sparkLength: 40,
  /** Overlay refresh. Writing a text node every frame is itself measurable. */
  overlayMs: 200,
};

const SPARK = ' ▁▂▃▄▅▆▇█';
/** Sparkline top of scale: 33 ms = two dropped frames at 60 Hz. */
const SPARK_MAX_MS = 33;

/* -------------------------------------------------------------------------- */

/**
 * @param {Object} o
 * @param {THREE.WebGLRenderer} o.renderer
 * @param {Object} o.engine    core/engine.js — setPixelRatio re-sizes everything
 * @param {Object} o.post      post/post.js  — setAO / setBloom / stats
 * @param {Object} o.quality   the live preset object (identical to ctx.quality)
 * @param {Object} o.hud       core/hud.js
 * @param {THREE.Scene} o.scene
 * @param {Object} o.physics   physics/physics.js — stats
 * @returns {{begin:Function, end:Function, toggle:Function, setAdaptive:Function,
 *            quality:Object, stats:Object, dispose:Function}}
 */
export function createPerformanceManager({ renderer, engine, post, quality, hud, scene, physics }) {
  const q = quality || {};

  /* --------------------------------------------------------------- state */

  const history = new Float32Array(TUNE.historyLength);
  const sparkCells = new Array(TUNE.sparkLength).fill(' ');
  let historyIdx = 0;
  let historyFilled = 0;

  let emaMs = 16.6;          // starts inside the dead band: provoke nothing
  let cpuEmaMs = 8;
  let lastBegin = 0;
  let beginAt = 0;

  let slowStreak = 0;
  let fastStreak = 0;
  let floorStreaks = 0;      // slow streaks accumulated while pinned at the floor
  let cooldown = TUNE.cooldownFrames * 2;   // let the first ~2 s settle
  let fastNeeded = TUNE.fastFrames;
  let lastChangeWasDown = true;             // first down-step is not an oscillation
  let warmup = TUNE.warmupFrames;           // frames still to ignore entirely
  let downSteps = 0;                        // resolution steps taken this session
  let hitches = 0;                          // frames discarded as compile stalls

  let adaptive = true;
  let visible = false;
  let disposed = false;
  let lastOverlayAt = 0;

  const dpr = window.devicePixelRatio || 1;
  const ceilingRatio = Math.min(dpr, q.pixelRatioCap || 1);
  let ratio = renderer.getPixelRatio() || ceilingRatio;

  /* --------------------------------------------------- degradation ladder */

  // What the preset asked for. Restoring means going back to exactly this.
  const baseline = {
    gtao: !!q.gtao,
    bloom: Number.isFinite(q.bloom) ? q.bloom : 0.5,
    shadowDistance: Number.isFinite(q.shadowDistance) ? q.shadowDistance : 80,
  };

  /** Guard: never write a knob the contract says is baked at load. */
  function setKnob(key, value) {
    if (!RUNTIME_KNOBS.includes(key)) {
      console.warn(`[perf] refusing to change bake-time knob "${key}"`);
      return;
    }
    q[key] = value;
  }

  /**
   * Shadow distance has no direct setter here — nothing outside lighting/ owns
   * lights. Writing `quality.shadowDistance` is the contract (it is the same
   * object as `ctx.quality`, which the lighting stream reads); the optional
   * call is a courtesy for a lighting module that caches instead of polling.
   */
  function pushShadowDistance(v) {
    setKnob('shadowDistance', v);
    try { window.HOLLOWBROOK?.lighting?.setShadowDistance?.(v); } catch { /* optional */ }
  }

  const ladder = [
    {
      name: 'gtao',
      applicable: () => baseline.gtao,
      down() { setKnob('gtao', false); post?.setAO?.(false); },
      up() { setKnob('gtao', baseline.gtao); post?.setAO?.(baseline.gtao); },
    },
    {
      name: 'bloom',
      applicable: () => baseline.bloom > 0.01,
      down() {
        const v = Math.max(0.12, baseline.bloom * 0.45);
        setKnob('bloom', v);
        post?.setBloom?.(v);
      },
      up() { setKnob('bloom', baseline.bloom); post?.setBloom?.(baseline.bloom); },
    },
    {
      name: 'shadowDistance',
      applicable: () => baseline.shadowDistance > 45,
      down() { pushShadowDistance(Math.max(38, Math.round(baseline.shadowDistance * 0.62))); },
      up() { pushShadowDistance(baseline.shadowDistance); },
    },
  ];

  let tier = 0;                                  // ladder steps currently down
  const dropCount = new Int8Array(ladder.length);
  const locked = new Int8Array(ladder.length);   // dropped too often — stays down

  /* --------------------------------------------------------- static facts */

  const gpuName = (() => {
    try {
      const gl = renderer.getContext();
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const s = (ext && gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) ||
        gl.getParameter(gl.RENDERER) || 'unknown gpu';
      return String(s).replace(/\s+/g, ' ').trim().slice(0, 52);
    } catch {
      return 'unknown gpu';
    }
  })();

  // Counted once — the scene is fully built by the time this factory runs.
  const sceneCount = (() => {
    let meshes = 0;
    let instanced = 0;
    let instances = 0;
    try {
      scene?.traverse?.((o) => {
        if (o.isInstancedMesh) { instanced++; instances += o.count; }
        else if (o.isMesh) meshes++;
      });
    } catch { /* an unbuilt scene is not worth failing over */ }
    return { meshes, instanced, instances };
  })();

  /* -------------------------------------------------------------- stats */

  const stats = {
    fps: 0,
    frameMs: 0,
    worstMs: 0,
    cpuMs: 0,
    postMs: 0,
    physicsMs: 0,
    draws: 0,
    tris: 0,
    programs: 0,
    geometries: 0,
    textures: 0,
    bodies: 0,
    pixelRatio: ratio,
    pixelRatioCeiling: ceilingRatio,
    tier: 0,
    adaptive: true,
    warmup: TUNE.warmupFrames,
    hitches: 0,
    downSteps: 0,
    gpu: gpuName,
    scene: sceneCount,
  };

  /* --------------------------------------------------------------------- */
  /* Measurement — no allocation in here                                    */
  /* --------------------------------------------------------------------- */

  function begin() {
    beginAt = performance.now();
  }

  function end(dt, ctx) {
    if (disposed) return;
    const now = performance.now();

    const frameMs = lastBegin ? beginAt - lastBegin : 16.6;
    lastBegin = beginAt;

    const cpuMs = now - beginAt;
    cpuEmaMs += (cpuMs - cpuEmaMs) * TUNE.emaAlpha;

    const stalled = frameMs > TUNE.outlierMs || frameMs <= 0;
    // Shorter than a stall but far longer than a frame: a shader compile, a
    // shadow bake, a texture upload. Real, worth showing, and no evidence at
    // all about how fast this GPU renders a steady frame.
    const hitch = !stalled && frameMs > TUNE.hitchMs;

    if (frameMs > 0 && !stalled) {
      history[historyIdx] = frameMs;
      historyIdx = (historyIdx + 1) % TUNE.historyLength;
      if (historyFilled < TUNE.historyLength) historyFilled++;
    }

    if (stalled || hitch) {
      // A stall or a hitch poisons the streaks; start counting again from clean.
      if (hitch) hitches++;
      slowStreak = 0;
      fastStreak = 0;
      const quiet = hitch ? TUNE.hitchCooldownFrames : TUNE.cooldownFrames;
      if (cooldown < quiet) cooldown = quiet;
    } else {
      emaMs += (frameMs - emaMs) * TUNE.emaAlpha;
    }

    // renderer.info was reset by main.js immediately before post.render().
    const info = renderer.info;
    stats.frameMs = emaMs;
    stats.fps = emaMs > 0 ? 1000 / emaMs : 0;
    stats.cpuMs = cpuEmaMs;
    stats.postMs = post?.stats?.cpuMs || 0;
    stats.draws = info.render.calls;
    stats.tris = info.render.triangles;
    stats.programs = info.programs ? info.programs.length : 0;
    stats.geometries = info.memory.geometries;
    stats.textures = info.memory.textures;
    stats.pixelRatio = renderer.getPixelRatio();
    stats.tier = tier;
    stats.adaptive = adaptive;
    stats.warmup = warmup;
    stats.hitches = hitches;
    stats.downSteps = downSteps;

    const ps = physics?.stats;
    if (ps) {
      stats.physicsMs = firstNum(ps.stepMs, ps.stepMS, ps.step, ps.ms);
      stats.bodies = firstNum(ps.bodies, ps.bodyCount, ps.dynamicBodies, ps.rigidBodies);
    }

    // The settings menu pauses the world; frame times there mean nothing.
    const measuring = !stalled && !hitch && !(ctx && ctx.paused);

    // Warm-up: the world has only just appeared. Programs are still compiling
    // and the first shadow map is still being baked, so nothing measured here
    // says anything about the steady state. Count it down on real frames only.
    if (warmup > 0) {
      if (measuring) warmup--;
      slowStreak = 0;
      fastStreak = 0;
      // Leaving warm-up with an average dragged up by load frames would hand the
      // governor a slow streak on its first eligible frame, so start clean.
      if (warmup === 0 && frameMs > 0 && !stalled && !hitch) emaMs = frameMs;
    } else if (cooldown > 0) cooldown--;
    else if (adaptive && measuring) governor();

    if (visible && now - lastOverlayAt >= TUNE.overlayMs) {
      lastOverlayAt = now;
      paint();
    }
  }

  function firstNum(a, b, c, d) {
    if (Number.isFinite(a)) return a;
    if (Number.isFinite(b)) return b;
    if (Number.isFinite(c)) return c;
    if (Number.isFinite(d)) return d;
    return 0;
  }

  /* --------------------------------------------------------------------- */
  /* The governor                                                           */
  /* --------------------------------------------------------------------- */

  /**
   * Headroom test. Below `fastMs` is unambiguous. At a hard vsync cap the frame
   * period cannot fall any further no matter how much headroom there is, so a
   * period pinned at the cap with a cheap CPU frame counts as fast as well —
   * without it, a machine that dropped a step at 60 Hz could never earn it back.
   */
  function hasHeadroom() {
    if (emaMs < TUNE.fastMs) return true;
    return emaMs < TUNE.vsyncMs && cpuEmaMs < TUNE.cpuHeadroomMs;
  }

  /**
   * WHY A STEADY 60 fps CAN NEVER BE STEPPED DOWN — the audit, from the code, so
   * the next person does not have to re-derive it:
   *
   *  1. Both degradations (`applyRatio(..., 'down')` and `dropTier()`) sit inside
   *     `if (slowStreak >= slowNeeded)` below. Nothing else in this file calls
   *     them except `setTier` / `setPixelRatio`, which are the explicit manual
   *     API. So `slowStreak` reaching its threshold is a NECESSARY condition for
   *     any automatic loss of quality.
   *  2. `slowStreak++` appears exactly once — the first line of this function —
   *     and is gated on `emaMs > TUNE.slowMs` (19.5 ms). Every other mention of
   *     `slowStreak` in this file assigns it 0.
   *  3. `emaMs` has exactly two writers: the EMA update in `end()` and the
   *     one-shot reseed at warm-up exit. BOTH are guarded by `!stalled && !hitch`,
   *     so neither can ever be fed a sample above `TUNE.hitchMs` (100 ms).
   *  4. Therefore a machine whose real frame period is 16.67 ms drives `emaMs` to
   *     16.67, which is 2.8 ms below the 19.5 ms trigger, and `slowStreak` is
   *     pinned at 0 forever. The dead band at 16.67 falls into case (2)-false and
   *     case `hasHeadroom()`, so the machine either steps UP or sits still.
   *
   *  The one transient worth checking is the warm-up reseed: it can legitimately
   *  set `emaMs` as high as 100 ms. It cannot cause a step down either, because
   *  `cooldown` is only ever decremented on the `warmup === 0` branch, so it is
   *  still at its initial `cooldownFrames * 2` = 120 frames when warm-up ends,
   *  and 100 ms decays under 19.5 ms in 27 frames at alpha 0.12 — the governor
   *  does not run until ~93 frames after the average is already healthy.
   *
   *  Conclusion: no remaining step-down path. Nothing to fix here.
   */
  function governor() {
    if (emaMs > TUNE.slowMs) { slowStreak++; fastStreak = 0; }
    else if (hasHeadroom()) { fastStreak++; slowStreak = 0; }
    else { slowStreak = 0; fastStreak = 0; }   // dead band — leave it alone

    /* ------------------------------------------------------------ slower */
    // The first drop has to be earned over three seconds, later ones over one.
    const slowNeeded = downSteps === 0 ? TUNE.firstSlowFrames : TUNE.slowFrames;
    if (slowStreak >= slowNeeded) {
      slowStreak = 0;

      if (ratio > TUNE.pixelRatioFloor + 1e-4) {
        applyRatio(Math.max(TUNE.pixelRatioFloor, round1(ratio - TUNE.pixelRatioStep)), 'down');
        return;
      }

      // Resolution is spent. Require a second full slow streak at the floor so
      // one bad corner of the village doesn't strip the picture permanently.
      floorStreaks++;
      if (floorStreaks >= TUNE.floorStreaksPerTier) {
        floorStreaks = 0;
        dropTier();
      }
      return;
    }

    /* ------------------------------------------------------------ faster */
    if (fastStreak >= fastNeeded) {
      fastStreak = 0;
      floorStreaks = 0;

      // Resolution comes back first — it is what the eye notices most.
      if (ratio < ceilingRatio - 1e-4) {
        applyRatio(Math.min(ceilingRatio, round1(ratio + TUNE.pixelRatioStep)), 'up');
        return;
      }
      if (tier > 0) restoreTier();
    }
  }

  function round1(v) { return Math.round(v * 10) / 10; }

  function applyRatio(next, dir) {
    if (Math.abs(next - ratio) < 1e-4) return;
    // Stepping down straight after a step up means we over-reached: make the
    // next recovery attempt materially harder to earn.
    if (dir === 'down' && !lastChangeWasDown) {
      fastNeeded = Math.min(TUNE.fastFramesMax, Math.round(fastNeeded * TUNE.fastFramesGrowth));
    }
    if (dir === 'down') downSteps++;
    lastChangeWasDown = dir === 'down';
    ratio = next;
    engine?.setPixelRatio?.(ratio);   // re-sizes the renderer, and the composer via onResize
    cooldown = TUNE.cooldownFrames;
    console.info(
      `[perf] pixel ratio ${dir} -> ${ratio.toFixed(2)}  (${emaMs.toFixed(1)} ms; ` +
      `recovery needs ${fastNeeded} frames under ${TUNE.fastMs} ms)`,
    );
  }

  function dropTier() {
    while (tier < ladder.length) {
      const i = tier;
      const step = ladder[i];
      tier++;
      if (!step.applicable()) continue;         // nothing to give up here
      try { step.down(); } catch (err) { console.warn(`[perf] tier "${step.name}" failed:`, err); }
      dropCount[i]++;
      if (dropCount[i] >= TUNE.tierDropsBeforeLock && !locked[i]) {
        locked[i] = 1;
        console.info(`[perf] "${step.name}" has cost the budget ${dropCount[i]}x — locked off for this session`);
      }
      cooldown = TUNE.cooldownFrames;
      console.info(`[perf] degraded: ${step.name}  (${emaMs.toFixed(1)} ms at the pixel-ratio floor)`);
      return;
    }
  }

  function restoreTier() {
    while (tier > 0) {
      const i = tier - 1;
      const step = ladder[i];
      tier--;
      if (locked[i]) continue;                  // stays down, keep unwinding
      try { step.up(); } catch (err) { console.warn(`[perf] restoring "${step.name}" failed:`, err); }
      cooldown = TUNE.cooldownFrames;
      console.info(`[perf] restored: ${step.name}`);
      return;
    }
  }

  /* --------------------------------------------------------------------- */
  /* Overlay                                                                */
  /* --------------------------------------------------------------------- */

  function spark() {
    const n = Math.min(TUNE.sparkLength, historyFilled);
    const pad = TUNE.sparkLength - n;
    for (let i = 0; i < pad; i++) sparkCells[i] = ' ';
    for (let i = 0; i < n; i++) {
      const idx = (historyIdx - n + i + TUNE.historyLength * 2) % TUNE.historyLength;
      let s = Math.round((history[idx] / SPARK_MAX_MS) * (SPARK.length - 1));
      if (s < 0) s = 0; else if (s > SPARK.length - 1) s = SPARK.length - 1;
      sparkCells[pad + i] = SPARK[s];
    }
    return sparkCells.join('');
  }

  function worst() {
    let m = 0;
    for (let i = 0; i < historyFilled; i++) if (history[i] > m) m = history[i];
    return m;
  }

  function fmtCount(t) {
    if (t >= 1e6) return `${(t / 1e6).toFixed(2)}M`;
    if (t >= 1e3) return `${(t / 1e3).toFixed(0)}k`;
    return String(t);
  }

  /**
   * Per-pass attribution, straight from post.stats.passCost. Two lines:
   *
   *   pass   render 214d/2.10M  gtao 96d/0.82M  bloom 12d  aa 3d  grade 1d
   *   gpu    render 6.21  gtao 2.40  bloom 0.90  aa 0.31  grade 0.12  (ms)
   *
   * The second line is real GPU time where the driver exposes timer queries and
   * CPU submit time where it does not — labelled either way, because a 0.02 ms
   * "cost" for a full-screen pass means the number is CPU-side, not free.
   */
  function passLines() {
    const cost = post?.stats?.passCost;
    if (!Array.isArray(cost) || !cost.length) return '';
    const gpu = !!post.stats.gpuTiming;
    let draws = '';
    let times = '';
    let totalDraws = 0;
    for (const p of cost) {
      if (!p.enabled) continue;
      totalDraws += p.draws;
      draws += `  ${p.key} ${p.draws}d`;
      if (p.tris > 0) draws += `/${fmtCount(p.tris)}`;
      times += `  ${p.key} ${(gpu ? p.gpuMs : p.cpuMs).toFixed(2)}`;
    }
    if (!draws) return '';
    return `pass ${totalDraws}d${draws}\n${gpu ? 'gpu ' : 'cpu '}${times}  (ms)\n`;
  }

  /**
   * The tone/grade state line:
   *
   *   exp 1.00  grade c1.17 s1.10 sh0.06 v0.24 gr0.034 ca0.0016
   *
   * Cheap (post.stats.grade is a live mirror, already clamped) and it exists so
   * a ground-luminance measurement can be read alongside the tuning that
   * produced it — without poking at objects in the console. `off` when the grade
   * pass is disabled, so a flattened measurement is never mistaken for a graded
   * one.
   */
  function gradeLine() {
    const ps = post?.stats;
    if (!ps) return '';
    const exp = Number.isFinite(ps.exposure) ? ps.exposure : 1;
    const g = ps.grade;
    if (!g) return `exp ${exp.toFixed(2)}\n`;
    if (g.enabled === false) return `exp ${exp.toFixed(2)}   grade off\n`;
    return `exp ${exp.toFixed(2)}   grade c${g.contrast.toFixed(2)} s${g.saturation.toFixed(2)} ` +
      `sh${g.shadows.toFixed(3)} v${g.vignette.toFixed(2)} gr${g.grain.toFixed(3)} ` +
      `ca${g.aberration.toFixed(4)}\n`;
  }

  function names(mask) {
    let s = '';
    for (let i = 0; i < ladder.length; i++) {
      if (!mask(i)) continue;
      s += (s ? ',' : '') + ladder[i].name;
    }
    return s;
  }

  function paint() {
    stats.worstMs = worst();

    const w = renderer.domElement.width;
    const h = renderer.domElement.height;

    const fails = window.HOLLOWBROOK?.failures;
    const failText = fails && fails.length
      ? `\nFAILED  ${fails.map((f) => f.label || String(f)).join(', ')}`
      : '';

    const degraded = names((i) => i < tier);
    const lockedNames = names((i) => locked[i]);
    const ladderText =
      (degraded ? `  down[${degraded}]` : '') +
      (lockedNames ? `  locked[${lockedNames}]` : '');

    hud?.setPerf?.(
      `HOLLOWBROOK  ${q.name || '?'}${adaptive ? '' : '  adaptive:off'}` +
      `${warmup > 0 ? `  warmup ${warmup}` : ''}${hitches ? `  hitches ${hitches}` : ''}\n` +
      `${stats.fps.toFixed(0).padStart(3)} fps   ${emaMs.toFixed(1)} ms   worst ${stats.worstMs.toFixed(1)} ms\n` +
      `${spark()}\n` +
      `cpu ${cpuEmaMs.toFixed(1)}   post ${stats.postMs.toFixed(2)}   phys ${stats.physicsMs.toFixed(2)}  (ms)\n` +
      `draws ${stats.draws}   tris ${fmtCount(stats.tris)}   progs ${stats.programs}\n` +
      passLines() +
      `geos ${stats.geometries}   texs ${stats.textures}   ` +
      `mesh ${sceneCount.meshes}+${sceneCount.instanced}i/${fmtCount(sceneCount.instances)}\n` +
      `bodies ${stats.bodies}   pxr ${stats.pixelRatio.toFixed(2)}/${ceilingRatio.toFixed(2)}   ${w}x${h}\n` +
      `aa ${post?.stats?.aa || '-'}   ` +
      `ao ${post?.stats?.gtao ? `on x${(post?.stats?.aoScale ?? 1).toFixed(2)}` : 'off'}   ` +
      `bloom ${(post?.stats?.bloom ?? 0).toFixed(2)}@${(post?.stats?.bloomThreshold ?? 0).toFixed(2)}` +
      `+${(post?.stats?.bloomKnee ?? 0).toFixed(2)}${ladderText}\n` +
      gradeLine() +
      `${gpuName}${failText}`,
    );
  }

  /* --------------------------------------------------------------------- */

  function toggle(force) {
    visible = hud?.togglePerf?.(force) ?? false;
    if (visible) { lastOverlayAt = 0; paint(); }
    return visible;
  }

  /**
   * Entering the world (pointer lock) and coming back to the tab both trigger a
   * fresh round of program compilation and shadow work. Re-arm a short warm-up
   * so those frames can never be read as a slow GPU.
   */
  function rearmWarmup(frames) {
    if (warmup < frames) warmup = frames;
    slowStreak = 0;
    fastStreak = 0;
  }

  const onPointerLock = () => { if (document.pointerLockElement) rearmWarmup(45); };
  const onVisibility = () => { if (!document.hidden) rearmWarmup(30); };
  document.addEventListener('pointerlockchange', onPointerLock);
  document.addEventListener('visibilitychange', onVisibility);

  function setAdaptive(on) {
    adaptive = !!on;
    slowStreak = 0;
    fastStreak = 0;
    floorStreaks = 0;
    cooldown = TUNE.cooldownFrames;
    stats.adaptive = adaptive;
  }

  function dispose() {
    disposed = true;
    document.removeEventListener('pointerlockchange', onPointerLock);
    document.removeEventListener('visibilitychange', onVisibility);
  }

  return {
    begin,
    end,
    toggle,
    setAdaptive,
    quality: q,
    stats,
    dispose,
    /** Exposed so the review pass can inspect and poke the governor live. */
    tune: TUNE,
    setTier(n) {
      const target = Math.max(0, Math.min(ladder.length, n | 0));
      while (tier < target) dropTier();
      while (tier > target) restoreTier();
    },
    setPixelRatio(r) {
      applyRatio(Math.max(TUNE.pixelRatioFloor, Math.min(ceilingRatio, round1(r))), 'manual');
    },
  };
}
