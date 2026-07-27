/**
 * ============================================================================
 *  HOLLOWBROOK — integration
 * ============================================================================
 * This file owns nothing except the order things happen in. Every subsystem is
 * built by its own module behind the interface described in `contracts.js`.
 *
 * A module that throws is isolated: it is replaced by an inert stub, the world
 * still loads, and the failure is reported in the perf overlay and the console
 * rather than showing a black screen.
 * ============================================================================
 */

import * as THREE from 'three';
import { detectQuality, presetFor, WORLD, CAMERA, RENDER, PLAYER } from './config.js';
import { createEngine } from './core/engine.js';
import { createHud } from './core/hud.js';
import { SPAWN } from './world/layout.js';

/* -------------------------------------------------------------------------- */

const hud = createHud();
const failures = [];

/** Load a module and call its factory; on failure, substitute a stub. */
async function stage(fraction, label, fn, stub) {
  hud.progress(fraction, label);
  // Yield so the browser can paint the progress bar between heavy stages.
  // rAF alone is wrong here: a backgrounded tab stops firing it, so switching
  // away mid-load would stall the world at whatever stage it had reached. Race
  // it against a timer and take whichever fires first.
  await new Promise((r) => {
    let done = false;
    const finish = () => { if (!done) { done = true; r(); } };
    requestAnimationFrame(finish);
    setTimeout(finish, 60);
  });
  try {
    return await fn();
  } catch (err) {
    console.error(`[hollowbrook] stage "${label}" failed:`, err);
    failures.push({ label, err });
    return typeof stub === 'function' ? stub() : stub;
  }
}

const emptyChunk = () => ({
  group: new THREE.Group(),
  colliders: [],
  interactables: [],
  lightAnchors: [],
  update: null,
  dispose: null,
  stats: {},
});

/* -------------------------------------------------------------------------- */

/**
 * Two capability checks worth making before we spend four seconds building a
 * world the visitor cannot use. This is a public link: a good share of traffic
 * is phones, and iOS Safari has no Pointer Lock at all, so the start button
 * would simply do nothing and look broken.
 */
function checkCapabilities() {
  let webgl2 = false;
  try {
    const c = document.createElement('canvas');
    webgl2 = !!c.getContext('webgl2');
  } catch { /* treated as unsupported */ }

  const pointerLock = 'requestPointerLock' in HTMLElement.prototype
    || 'mozRequestPointerLock' in HTMLElement.prototype
    || 'webkitRequestPointerLock' in HTMLElement.prototype;
  const coarse = matchMedia('(pointer: coarse)').matches
    && !matchMedia('(pointer: fine)').matches;

  return { webgl2, pointerLock, touchOnly: coarse };
}

async function boot() {
  const caps = checkCapabilities();
  if (!caps.webgl2) {
    hud.progress(1, 'unsupported browser');
    hud.fail('This world needs WebGL2, which this browser does not provide.\n\n'
      + 'Chrome, Edge, Firefox or Safari 16+ on a desktop will run it.');
    return;
  }

  const qualityName = detectQuality();
  const quality = presetFor(qualityName);
  console.info(`[hollowbrook] quality preset: ${quality.name}`);

  const canvas = document.getElementById('viewport');
  const engine = createEngine(canvas, quality);
  const { renderer, scene, camera } = engine;

  /* ---------------------------------------------------------------- physics */
  const { createPhysics } = await import('./physics/physics.js');
  const physics = await stage(0.05, 'starting the physics solver', () =>
    createPhysics({ gravity: [0, WORLD.gravity, 0], quality, scene }));

  /* --------------------------------------------------------------- textures */
  const { createTextureLibrary } = await import('./materials/textures.js');
  const textures = await stage(0.14, 'weaving cloth, straw and stone', () =>
    createTextureLibrary(renderer, { quality, anisotropy: engine.maxAnisotropy }));

  const { createMaterialLibrary } = await import('./materials/materials.js');
  const materials = await stage(0.24, 'mixing lime wash and pigment', () =>
    createMaterialLibrary(textures, { quality, renderer }));

  /* --------------------------------------------------------------- lighting */
  const { createSky } = await import('./lighting/sky.js');
  const sky = await stage(0.32, 'raising the sky', () =>
    createSky({ renderer, scene, quality }));

  const { createLighting } = await import('./lighting/lighting.js');
  const lighting = await stage(0.40, 'letting the afternoon in', () =>
    createLighting({ scene, sky, quality, renderer }));

  // Shadow softness lives on the light, not on the renderer: with PCFShadowMap
  // the shader offsets its Vogel disk by shadow.radius texels. Left at the
  // default 1 every shadow in the village is pin-sharp, which is wrong for a
  // hazy afternoon.
  if (lighting?.sun?.shadow) {
    lighting.sun.shadow.radius = quality.softShadows ? 3.5 : 1.0;
  }

  // The environment map is the project's HDRI: a float cube capture of the
  // procedural sky, prefiltered by PMREM. Everything is lit from it.
  if (sky?.envMap) {
    scene.environment = sky.envMap;
    scene.environmentIntensity = 1.0;
    materials?.setEnvironment?.(sky.envMap, 1.0);
  }

  /* ---------------------------------------------------------------- terrain */
  const { createTerrain } = await import('./world/terrain.js');
  const terrain = await stage(0.50, 'laying cobbles and turf', () =>
    createTerrain({ materials, quality, sky }), emptyChunk);
  scene.add(terrain.group);

  /* -------------------------------------------------------------- buildings */
  const { createBuildings } = await import('./world/buildings.js');
  const buildings = await stage(0.62, 'raising timber frames', () =>
    createBuildings({ materials, terrain, quality }), emptyChunk);
  scene.add(buildings.group);

  /* ------------------------------------------------------------------ props */
  const { createProps } = await import('./world/props.js');
  const props = await stage(0.72, 'setting out barrels and blossom', () =>
    createProps({ materials, terrain, buildings, quality }), emptyChunk);
  scene.add(props.group);

  /* ------------------------------------------------------------- vegetation */
  const { createVegetation } = await import('./world/vegetation.js');
  const vegetation = await stage(0.78, 'planting the valley', () =>
    createVegetation({ materials, terrain, quality }), emptyChunk);
  scene.add(vegetation.group);

  /* -------------------------------------------------------------- interiors */
  // The shell publishes its own rooms (see RoomSpec in contracts.js); these two
  // modules furnish them and never re-derive a wall position.
  const { createInteriors } = await import('./world/interiors.js');
  const interiors = await stage(0.83, 'flooring and framing the rooms', () =>
    createInteriors({ materials, buildings, terrain, quality }), emptyChunk);
  scene.add(interiors.group);

  const { createFurnishings } = await import('./world/furnishings.js');
  const furnishings = await stage(0.87, 'laying the hearths', () =>
    createFurnishings({ materials, buildings, interiors, terrain, quality }), emptyChunk);
  scene.add(furnishings.group);

  /* ------------------------------------------------------ hand it to physics */
  await stage(0.90, 'making the world solid', () => {
    const statics = [
      ...(terrain.colliders || []),
      ...(buildings.colliders || []),
      ...(props.colliders || []),
      ...(vegetation.colliders || []),
      ...(interiors.colliders || []),
      ...(furnishings.colliders || []),
    ];
    physics.addStaticColliders(statics);
    physics.addInteractables([
      ...(props.interactables || []),
      ...(buildings.interactables || []),
      ...(interiors.interactables || []),
      ...(furnishings.interactables || []),
    ]);
    return true;
  });

  // Lanterns, window glow and hearth fire all come from anchors the geometry
  // emits. Indoors the pool has to prefer the room the player is standing in
  // over a nearer light through a wall, so hand the lighting the room resolver.
  await stage(0.93, 'lighting the lanterns', () => {
    lighting?.addLightAnchors?.([
      ...(buildings.lightAnchors || []),
      ...(props.lightAnchors || []),
      ...(interiors.lightAnchors || []),
      ...(furnishings.lightAnchors || []),
    ]);
    if (interiors.roomAt) lighting?.setRoomResolver?.(interiors.roomAt);
    return true;
  });

  /* ------------------------------------------------------------------- post */
  const { createPostProcessing } = await import('./post/post.js');
  const post = await stage(0.95, 'grading the picture', () =>
    createPostProcessing({ renderer, scene, camera, quality, engine }));

  /* --------------------------------------------------------------- controls */
  const { createPlayerController } = await import('./player/controls.js');
  const player = await stage(0.97, 'finding your feet', () =>
    createPlayerController({
      camera, renderer, physics, terrain, hud, quality, scene,
      spawn: { position: SPAWN.position, yaw: SPAWN.yaw },
    }));

  /* ------------------------------------------------------------ performance */
  const { createPerformanceManager } = await import('./perf/perf.js');
  const perf = await stage(0.99, 'tuning', () =>
    createPerformanceManager({ renderer, engine, post, quality, hud, scene, physics }));

  /* ------------------------------------------------------------------- loop */
  const clock = new THREE.Clock();
  const frameCtx = {
    elapsed: 0,
    dt: 0,
    camera,
    playerPosition: new THREE.Vector3(),
    quality,
    paused: false,
  };

  const updatables = [terrain, buildings, props, vegetation, interiors, furnishings, sky, lighting]
    .filter((m) => m && typeof m.update === 'function');

  let running = true;

  function frame() {
    if (!running) return;
    requestAnimationFrame(frame);

    const dt = Math.min(clock.getDelta(), 0.1);
    frameCtx.dt = dt;
    frameCtx.elapsed += dt;

    perf?.begin?.();

    if (!frameCtx.paused) {
      player?.update?.(dt, frameCtx);
      frameCtx.playerPosition.copy(player?.position || camera.position);
      physics?.update?.(dt, frameCtx);
      for (const m of updatables) m.update(dt, frameCtx);
    } else {
      frameCtx.playerPosition.copy(player?.position || camera.position);
      // Still tick the sky and the sun, with dt 0 so nothing animates. The sky
      // shader refreshes per-frame uniforms in its update(); skipping it entirely
      // while the settings menu is open rendered the whole dome black.
      sky?.update?.(0, frameCtx);
      lighting?.update?.(0, frameCtx);
    }

    renderer.info.reset();
    post?.render?.(dt, frameCtx);
    perf?.end?.(dt, frameCtx);
  }

  hud.progress(1, 'ready');
  hud.finishLoading();

  /* --------------------------------------------------------------- settings */
  wireSettings({ hud, engine, post, sky, lighting, player, perf, physics, quality, frameCtx });

  // Say so plainly rather than handing a phone a button that does nothing.
  if (caps.touchOnly || !caps.pointerLock) {
    const note = document.getElementById('start-note');
    if (note) {
      note.innerHTML = caps.touchOnly
        ? '<b>This needs a keyboard and mouse.</b> Walking uses WASD and looking '
          + 'around uses pointer lock, neither of which a touchscreen provides. '
          + 'You can still look at the village below, but open it on a desktop to explore it.'
        : '<b>This browser has no Pointer Lock.</b> You can enter, but the mouse '
          + 'will not turn the camera — try Chrome, Edge or Firefox.';
      note.classList.remove('hidden');
    }
    if (caps.touchOnly && hud.el.startBtn) hud.el.startBtn.textContent = 'Look around anyway';
  }

  hud.el.startBtn?.addEventListener('click', () => {
    hud.el.start.classList.add('hidden');
    hud.showHud(true);
    player?.requestPointerLock?.();
  });

  // Expose for the review pass — walking the world from the console beats
  // guessing at what a screenshot is showing.
  window.HOLLOWBROOK = {
    THREE, engine, scene, camera, renderer, physics, materials, textures,
    terrain, buildings, props, vegetation, interiors, furnishings,
    sky, lighting, post, player, perf,
    frameCtx, failures,
    teleport(x, y, z, yaw) { player?.teleport?.(x, y, z, yaw); },
    stats() {
      return {
        failures: failures.map((f) => f.label),
        draws: renderer.info.render.calls,
        tris: renderer.info.render.triangles,
        programs: renderer.info.programs?.length,
        textures: renderer.info.memory.textures,
        geometries: renderer.info.memory.geometries,
        quality: quality.name,
      };
    },
  };

  if (failures.length) {
    console.warn('[hollowbrook] subsystems that failed to build:',
      failures.map((f) => f.label));
  }

  frame();
}

/* -------------------------------------------------------------------------- */

function wireSettings(ctx) {
  const { hud, engine, post, sky, lighting, player, perf, physics, frameCtx } = ctx;
  const $ = (id) => document.getElementById(id);

  const open = () => {
    frameCtx.paused = true;
    hud.el.menu.classList.remove('hidden');
    player?.exitPointerLock?.();
  };
  const close = () => {
    frameCtx.paused = false;
    hud.el.menu.classList.add('hidden');
    // Chrome refuses a pointer-lock request issued too soon after the user
    // pressed Escape to leave it, so give it a beat. Clicking the canvas is
    // the fallback path either way.
    setTimeout(() => player?.requestPointerLock?.(), 260);
  };

  $('resume-btn')?.addEventListener('click', close);

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Escape') {
      if (hud.el.menu.classList.contains('hidden')) open(); else close();
    }
    if (e.code === 'Tab') {
      e.preventDefault();
      perf?.toggle?.();
    }
  });

  // Pointer lock loss should raise the menu, not silently freeze input.
  document.addEventListener('pointerlockchange', () => {
    if (!document.pointerLockElement &&
        hud.el.start.classList.contains('hidden') &&
        hud.el.menu.classList.contains('hidden')) {
      open();
    }
  });

  // Reflect the active preset in the controls BEFORE binding, so the initial
  // apply does not clobber it with the markup's default value.
  const gtaoEl = $('opt-gtao');
  if (gtaoEl) gtaoEl.checked = !!ctx.quality.gtao;
  const bloomEl = $('opt-bloom');
  if (bloomEl) bloomEl.value = String(ctx.quality.bloom);

  const bind = (id, label, fn, fmt = (v) => v.toFixed(2)) => {
    const el = $(id);
    const out = $(label);
    if (!el) return;
    const apply = () => {
      const v = el.type === 'checkbox' ? el.checked : parseFloat(el.value);
      if (out && typeof v === 'number') out.textContent = fmt(v);
      fn(v);
    };
    el.addEventListener('input', apply);
    apply();
  };

  bind('opt-sens', 'val-sens', (v) => player?.setSensitivity?.(v));
  // The controller owns camera.fov (it widens it while sprinting), so route
  // the slider through it rather than writing the camera directly.
  bind('opt-fov', 'val-fov', (v) => player?.setFov?.(v), (v) => v.toFixed(0));
  bind('opt-tod', 'val-tod', (v) => {
    sky?.setTimeOfDay?.(v);
    lighting?.setTimeOfDay?.(v);
  }, (v) => {
    const h = v * 24;
    return `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor((h % 1) * 60)).padStart(2, '0')}`;
  });
  bind('opt-bloom', 'val-bloom', (v) => post?.setBloom?.(v));
  bind('opt-exp', 'val-exp', (v) => { engine.renderer.toneMappingExposure = v; });
  bind('opt-gtao', null, (v) => post?.setAO?.(v));
  bind('opt-bob', null, (v) => player?.setHeadBob?.(v));
  bind('opt-adaptive', null, (v) => perf?.setAdaptive?.(v));
  bind('opt-physdebug', null, (v) => physics?.setDebug?.(v));

  const qsel = $('opt-quality');
  if (qsel) {
    qsel.value = ctx.quality.name;
    qsel.addEventListener('change', () => {
      // Texture and env-map resolution are baked at load, so a tier change is a
      // reload rather than a live swap.
      const url = new URL(location.href);
      url.searchParams.set('quality', qsel.value);
      location.href = url.toString();
    });
  }
}

/* -------------------------------------------------------------------------- */

boot().catch((err) => {
  console.error('[hollowbrook] fatal:', err);
  hud.fail(err);
});
