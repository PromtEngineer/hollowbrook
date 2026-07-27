/**
 * ============================================================================
 *  FIRST-PERSON CONTROLS & CAMERA
 * ============================================================================
 * Owns: pointer-lock mouse look, fixed-step character movement, the camera
 * rig (eye height, head bob, strafe roll, sprint FOV, landing dip, breathing),
 * the interaction ray + carry/throw, and the hand lantern.
 *
 * Two invariants drive most of the design here:
 *
 *   1. The physics body is never smoothed. Every soft, springy thing lives in
 *      the *eye* transform, layered on top of an interpolated body position.
 *      Smoothing the body would make collision resolution lie.
 *   2. Movement integrates at a fixed 1/60 (driven by `physics.onFixedStep`
 *      when the physics stream offers it, self-stepped otherwise) so the feel
 *      of the character is identical at 30 fps and 144 fps.
 *
 * This module is deliberately defensive about the physics stream's surface: it
 * probes for a character-controller factory, a raycast and grab/release under
 * several plausible names, and falls back to a terrain-following kinematic
 * capsule if none of them exist. A broken physics module should degrade to
 * "you can still walk the village", never to "you cannot move".
 * ============================================================================
 */

import * as THREE from 'three';
import { PLAYER, WORLD, CAMERA } from '../config.js';

/* -------------------------------------------------------------------------- */
/* Tunables that are ours, not config.js's                                     */
/* -------------------------------------------------------------------------- */

const TUNING = {
  /** rad per pixel of raw mouse movement at sensitivity 1.0. */
  baseSensitivity: 0.0022,
  /** Anything larger than this in one event is a driver / pointer-lock artefact. */
  maxMouseDelta: 260,
  pitchLimit: 89 * (Math.PI / 180),

  /** Metres of travel per full head-bob cycle (one stride = two footfalls). */
  bobStride: 1.62,
  bobVertical: 0.036,
  bobLateral: 0.028,
  bobRoll: 0.006,         // rad at full amplitude
  bobAmpLambda: 9,        // how fast bob amplitude follows speed

  strafeRoll: 1.2 * (Math.PI / 180),
  strafeRollLambda: 7,

  sprintFov: 4.0,         // degrees added at full sprint
  fovLambda: 5.5,

  landDipPerSpeed: 0.017, // metres of dip per m/s of impact
  landDipMax: 0.21,
  landSpring: 165,
  landDampingRatio: 0.78,
  landMinSpeed: 2.6,      // softer impacts do not register

  breathAmp: 0.0065,
  breathPitch: 0.0011,
  breathYaw: 0.0016,

  eyeLambda: 13,          // crouch / stand eye-height smoothing

  /**
   * Ray origin is nudged this far forward. The physics stream already excludes
   * the player capsule, so this only has to clear floating-point contact with
   * whatever we are standing against.
   */
  rayOriginOffset: 0.04,

  carrySpin: 0.55,        // rad/s while carried
  carryLambda: 18,        // how fast a carried object's rotation catches up

  lantern: {
    colour: 0xffd2a1,
    intensity: 18,        // candela — three 0.185 lights are physical
    distance: 22,
    angle: 0.62,
    penumbra: 0.5,
    decay: 2,
  },

  /**
   * Downward bias kept while grounded so snap-to-ground stays engaged.
   * MEASURED, not guessed: rapier's autostep refuses to lift the capsule once
   * the desired motion has much downward component, and the ceiling it can
   * still climb falls off fast — against this project's physics settings
   * (autostep 0.45 m, snap 0.4 m) a 0.35 m step needs <= 0.5 m/s and a 0.44 m
   * step needs <= 0.25 m/s. Snap-to-ground still holds at 0.1 m/s walking off
   * a 0.35 m ledge (0 airborne steps), so the small value costs nothing.
   */
  groundStick: 0.25,
  /** Fixed steps to ignore ground contact after a jump. */
  jumpUngroundSteps: 4,
  /** Below this the fall is not a fall. */
  terminalVelocity: 60,
};

/** Every key code the movement system consumes. */
const MOVEMENT_CODES = new Set([
  'KeyW', 'KeyA', 'KeyS', 'KeyD',
  'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
  'Space', 'ShiftLeft', 'ShiftRight', 'ControlLeft', 'ControlRight',
]);

/* -------------------------------------------------------------------------- */
/* Module-scope scratch — nothing in the frame path allocates                  */
/* -------------------------------------------------------------------------- */

const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _wish = new THREE.Vector3();
const _delta = new THREE.Vector3();
const _tmpA = new THREE.Vector3();
const _tmpB = new THREE.Vector3();
const _tmpC = new THREE.Vector3();
const _rayOrigin = new THREE.Vector3();
const _rayDir = new THREE.Vector3();
const _rayUp = new THREE.Vector3(0, 1, 0);
const _carryPos = new THREE.Vector3();
const _quatA = new THREE.Quaternion();
const _quatB = new THREE.Quaternion();
const _spinAxis = new THREE.Vector3(0.22, 1, 0.13).normalize();

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const round2 = (v) => Math.round(v * 100) / 100;
const damp = THREE.MathUtils.damp;

let warned = {};
function warnOnce(key, ...msg) {
  if (warned[key]) return;
  warned[key] = true;
  console.warn('[controls]', ...msg);
}

/* -------------------------------------------------------------------------- */
/* Adapters over the physics / terrain streams                                 */
/* -------------------------------------------------------------------------- */

/** First callable member of `obj` from `names`, bound. Resolved once, never per frame. */
function pickFn(obj, names) {
  if (!obj) return null;
  for (const n of names) {
    if (typeof obj[n] === 'function') return obj[n].bind(obj);
  }
  return null;
}

function readVecInto(src, out) {
  if (!src) return null;
  if (typeof src.x === 'number') return out.set(src.x, src.y || 0, src.z || 0);
  if (Array.isArray(src) && src.length >= 3) return out.set(src[0], src[1], src[2]);
  return null;
}

/**
 * Terrain height lookup, used by the fallback character and by `teleport`. The
 * terrain stream may name this whatever it likes; we try the plausible set and
 * fall back to the flat village floor (y = 0) that config.WORLD guarantees.
 */
function terrainHeightFn(terrain) {
  const fn = pickFn(terrain, ['heightAt', 'sampleHeight', 'getHeight', 'heightAtWorld', 'elevationAt']);
  if (!fn) {
    warnOnce('terrainHeight', 'terrain exposes no height sampler — assuming a flat floor at y=0');
    return () => 0;
  }
  return (x, z) => {
    let h;
    try { h = fn(x, z); } catch { h = 0; }
    return Number.isFinite(h) ? h : 0;
  };
}

/**
 * A raycast closure with a normalised result. Handles both the positional
 * signature `raycast(origin, dir, maxDist, opts)` and the object signature
 * `raycast({origin, direction, maxDistance})`.
 */
function makeRaycast(physics) {
  const raw = pickFn(physics, ['raycast', 'castRay', 'rayCast']);
  if (!raw) {
    warnOnce('raycast', 'physics exposes no raycast — interaction and headroom checks are disabled');
    return () => null;
  }
  let objectForm = false;
  // `excludeCollider` must stay UNDEFINED when we have nothing to exclude:
  // the physics stream reads `undefined` as "use the default player exclusion"
  // but `null` as "exclude nothing", which would make every ray hit our own
  // capsule from the inside.
  const opts = { excludeCollider: undefined, excludePlayer: true, skipPlayer: true, solid: true };
  const objArg = {
    origin: null, direction: null, dir: null,
    maxDistance: 0, maxDist: 0, maxToi: 0,
    exclude: null, excludePlayer: true,
  };

  return function cast(origin, dir, maxDist, exclude) {
    if (!(maxDist > 0)) return null;
    try {
      if (!objectForm) {
        opts.excludeCollider = exclude || undefined;
        return raw(origin, dir, maxDist, opts) || null;
      }
      objArg.origin = origin;
      objArg.direction = dir;
      objArg.dir = dir;
      objArg.maxDistance = maxDist;
      objArg.maxDist = maxDist;
      objArg.maxToi = maxDist;
      objArg.exclude = exclude || null;
      return raw(objArg) || null;
    } catch (err) {
      if (!objectForm) {
        objectForm = true;
        warnOnce('raycastForm', 'physics.raycast rejected the positional form, switching to the object form:', err?.message);
        return cast(origin, dir, maxDist, exclude);
      }
      warnOnce('raycastFail', 'physics.raycast threw:', err?.message);
      return null;
    }
  };
}

/** Pull an Object3D out of whatever shape the raycast hit is. */
function hitObject(hit) {
  if (!hit) return null;
  return (
    hit.object3D || hit.object || hit.mesh || hit.target ||
    hit.collider?.userData?.object3D ||
    hit.body?.userData?.object3D ||
    hit.userData?.object3D ||
    null
  );
}

function hitDistance(hit) {
  if (!hit) return Infinity;
  const d = hit.distance ?? hit.toi ?? hit.t ?? hit.timeOfImpact;
  return Number.isFinite(d) ? d : Infinity;
}

/**
 * Walk up to the object the interaction system considers "the thing".
 * `tagInteractive` in contracts.js stamps `interactiveRoot` on every child.
 */
function interactiveRootOf(obj) {
  if (!obj) return null;
  if (obj.userData?.interactiveRoot) return obj.userData.interactiveRoot;
  let o = obj;
  while (o) {
    if (o.userData?.interactive) return o;
    o = o.parent;
  }
  return null;
}

/* ------------------------------ character --------------------------------- */

/** Preallocated move result — `move()` fills this in place, every step. */
function makeMoveResult() {
  const normals = [];
  for (let i = 0; i < 8; i++) normals.push(new THREE.Vector3());
  return {
    movement: new THREE.Vector3(),   // the CORRECTED translation actually taken
    position: new THREE.Vector3(),   // where the capsule centre ends up
    grounded: false,
    normalCount: 0,
    normals,
  };
}

/**
 * Wrap whatever the physics stream handed back so the rest of this file only
 * ever talks to one shape. Returns null if the object has no move method.
 */
function wrapCharacter(raw, opts) {
  const result = makeMoveResult();
  const before = new THREE.Vector3();
  const after = new THREE.Vector3();
  const scratchN = new THREE.Vector3();

  const fnMove = pickFn(raw, ['move', 'moveBy', 'step', 'computeMovement', 'translate', 'update']);
  const fnGetT = pickFn(raw, ['translation', 'getTranslation', 'getPosition']);
  const fnSetT = pickFn(raw, ['setTranslation', 'setPosition', 'teleport', 'setNextKinematicTranslation']);
  const fnGrounded = pickFn(raw, ['computedGrounded', 'isGrounded', 'grounded']);
  const fnMovement = pickFn(raw, ['computedMovement', 'movement', 'correctedMovement']);
  const fnNumCol = pickFn(raw, ['numComputedCollisions', 'numCollisions']);
  const fnCol = pickFn(raw, ['computedCollision', 'collision', 'getCollision']);
  const fnHeight = pickFn(raw, ['setHeight', 'setCapsule', 'resize', 'setDimensions']);
  const fnCrouch = pickFn(raw, ['setCrouched', 'setCrouch']);
  const fnStand = pickFn(raw, ['canStand', 'hasHeadroom', 'checkHeadroom', 'canStandUp']);
  const fnSlope = pickFn(raw, ['setMaxSlopeClimbAngle', 'setMaxSlope']);
  const fnDispose = pickFn(raw, ['dispose', 'destroy', 'remove', 'free']);

  // The project's physics stream keeps the raw Rapier controller on `.controller`
  // and that is the only place the per-collision normals live.
  const ctl = raw.controller || null;
  const fnNumCol2 = fnNumCol || pickFn(ctl, ['numComputedCollisions']);
  const fnCol2 = fnCol || pickFn(ctl, ['computedCollision']);
  let colScratch = null;   // reused CharacterCollision, captured on first use

  /**
   * A kinematic-position body only lands on its new translation when
   * `world.step()` runs, so `translation()` still reads the OLD value right
   * after the sweep. Detect that and trust the reported movement instead of
   * hard-teleporting the body (which would zero the kinematic velocity the
   * solver uses to push dynamic bodies).
   */
  const deferredCommit = !!(raw.body && typeof raw.body.setNextKinematicTranslation === 'function');

  if (!fnMove) return null;
  if (fnSlope) { try { fnSlope(opts.maxSlopeDeg * (Math.PI / 180)); } catch { /* optional */ } }

  function getTranslation(out) {
    if (fnGetT) {
      const t = fnGetT();
      if (readVecInto(t, out)) return out;
    }
    if (readVecInto(raw.position, out)) return out;
    if (raw.body && typeof raw.body.translation === 'function') {
      if (readVecInto(raw.body.translation(), out)) return out;
    }
    return out;
  }

  function setTranslation(v) {
    if (fnSetT) { fnSetT(v); return true; }
    if (raw.position && typeof raw.position.copy === 'function') { raw.position.copy(v); return true; }
    if (raw.body && typeof raw.body.setTranslation === 'function') { raw.body.setTranslation(v, true); return true; }
    warnOnce('setT', 'character controller has no way to set its translation — teleport will not work');
    return false;
  }

  return {
    raw,
    kind: 'physics',
    collider: raw.collider || raw.rawCollider || null,
    getTranslation,
    setTranslation,

    /**
     * @param {THREE.Vector3} desired translation for this fixed step
     * @param {number} h step length in seconds
     */
    move(desired, h) {
      getTranslation(before);
      let out = null;
      try {
        out = fnMove(desired, h);
      } catch (err) {
        warnOnce('moveThrow', 'character move() threw:', err?.message);
      }

      // Corrected translation: whatever move() returned, else a query method.
      let reported = !!readVecInto(out, result.movement);
      if (!reported && out) reported = !!readVecInto(out.movement, result.movement);
      if (!reported && fnMovement) reported = !!readVecInto(fnMovement(), result.movement);

      // Did the controller already commit the translation, or must we?
      getTranslation(after);
      const committed = after.distanceToSquared(before) > 1e-12;
      if (!reported) {
        result.movement.copy(after).sub(before);
        result.position.copy(after);
      } else if (committed) {
        // Prefer the real delta so we can never double-apply.
        result.movement.copy(after).sub(before);
        result.position.copy(after);
      } else if (deferredCommit) {
        // Already queued for the solver — just predict where we will land.
        result.position.copy(before).add(result.movement);
      } else {
        setTranslation(_tmpC.copy(before).add(result.movement));
        result.position.copy(_tmpC);
      }

      let g = null;
      if (out && typeof out.grounded === 'boolean') g = out.grounded;
      else if (fnGrounded) { const r = fnGrounded(); if (typeof r === 'boolean') g = r; }
      else if (typeof raw.grounded === 'boolean') g = raw.grounded;
      result.grounded = !!g;

      // Collision normals, world space, pointing out of the obstacle.
      result.normalCount = 0;
      const list = (out && out.collisions) || raw.collisions;
      if (Array.isArray(list)) {
        for (let i = 0; i < list.length && result.normalCount < 8; i++) {
          const n = list[i]?.normal1 || list[i]?.normal || list[i];
          if (readVecInto(n, scratchN)) result.normals[result.normalCount++].copy(scratchN);
        }
      } else if (fnNumCol2 && fnCol2) {
        const n = fnNumCol2() | 0;
        for (let i = 0; i < n && result.normalCount < 8; i++) {
          // Passing `out` keeps rapier from allocating a CharacterCollision.
          const c = colScratch ? fnCol2(i, colScratch) : fnCol2(i);
          if (c && !colScratch) colScratch = c;
          const nv = c?.normal1 || c?.normal;
          if (readVecInto(nv, scratchN)) result.normals[result.normalCount++].copy(scratchN);
        }
      }
      return result;
    },

    /**
     * Swap the capsule between standing and crouched.
     * @returns {boolean|null} true applied · false refused (no headroom) ·
     *                         null the controller cannot resize at all
     */
    setHeight(totalHeight, radius) {
      if (fnHeight) {
        try {
          const r = fnHeight(totalHeight, radius);
          return r === false ? false : true;
        } catch (err) {
          warnOnce('setHeight', 'character setHeight() threw:', err?.message);
          return false;
        }
      }
      if (fnCrouch) {
        try { fnCrouch(totalHeight < opts.height - 0.01); return true; }
        catch { /* fall through */ }
      }
      warnOnce('noCrouchCapsule', 'character controller cannot resize — crouch only lowers the camera');
      return null;
    },

    /** true / false / null (= unknown, caller should raycast). */
    canStand(height) {
      if (!fnStand) return null;
      try {
        const r = fnStand(height);
        return typeof r === 'boolean' ? r : null;
      } catch { return null; }
    },

    dispose() { try { fnDispose && fnDispose(); } catch { /* best effort */ } },
  };
}

/** Ask the physics stream for a character. Null if it has no factory we know. */
function adoptCharacter(physics, opts) {
  const names = [
    'createCharacter', 'createCharacterController', 'createKinematicCharacter',
    'createPlayerBody', 'createPlayer', 'addCharacter', 'spawnCharacter', 'character',
  ];
  for (const n of names) {
    if (typeof physics?.[n] !== 'function') continue;
    try {
      const raw = physics[n](opts);
      if (!raw) continue;
      const wrapped = wrapCharacter(raw, opts);
      if (wrapped) {
        console.info(`[controls] player capsule from physics.${n}()`);
        return wrapped;
      }
      warnOnce('badChar', `physics.${n}() returned an object with no move() — falling back`);
    } catch (err) {
      warnOnce('charThrow', `physics.${n}() threw:`, err?.message);
    }
  }
  return null;
}

/**
 * Last-resort character: a terrain-following capsule that probes for walls
 * with the physics raycast. Worse than a real character controller (no
 * autostep, coarse wall response, no dynamic-body pushing) but it keeps the
 * village walkable if the physics stream fails to build or names its factory
 * something this file does not recognise.
 */
function makeFallbackCharacter(opts, cast, heightAt) {
  const pos = new THREE.Vector3().copy(opts.position);
  const result = makeMoveResult();
  const probeDir = new THREE.Vector3();
  const probeOrigin = new THREE.Vector3();
  const nrm = new THREE.Vector3();
  const start = new THREE.Vector3();
  const radius = opts.radius;
  const probeFractions = [0.22, 0.55, 0.88]; // of capsule height, feet-relative
  let height = opts.height;

  function wallProbe(from, dir, dist, feetY, h) {
    let closest = Infinity;
    nrm.set(0, 0, 0);
    for (let i = 0; i < probeFractions.length; i++) {
      probeOrigin.set(from.x, feetY + probeFractions[i] * h, from.z);
      const hit = cast(probeOrigin, dir, dist, null);
      if (!hit) continue;
      const d = hitDistance(hit);
      if (d < closest) {
        closest = d;
        if (!readVecInto(hit.normal || hit.normal1, nrm)) nrm.copy(dir).negate();
      }
    }
    return closest;
  }

  return {
    raw: null,
    kind: 'fallback',
    collider: null,
    getTranslation(out) { return out.copy(pos); },
    setTranslation(v) { pos.copy(v); },

    move(desired) {
      start.copy(pos);
      const feetY = pos.y - height / 2;

      // --- horizontal, with one slide retry --------------------------------
      _tmpA.set(desired.x, 0, desired.z);
      let len = _tmpA.length();
      for (let pass = 0; pass < 2 && len > 1e-6; pass++) {
        probeDir.copy(_tmpA).multiplyScalar(1 / len);
        const d = wallProbe(pos, probeDir, len + radius + 0.05, feetY, height);
        if (d > len + radius) break;
        const into = _tmpA.dot(nrm);
        if (into < 0) _tmpA.addScaledVector(nrm, -into);
        len = _tmpA.length();
      }
      pos.x += _tmpA.x;
      pos.z += _tmpA.z;

      // --- vertical ---------------------------------------------------------
      pos.y += desired.y;
      const ground = heightAt(pos.x, pos.z);
      const newFeet = pos.y - height / 2;
      if (newFeet <= ground + (desired.y <= 0 ? PLAYER.snapToGround : 0)) {
        pos.y = ground + height / 2;
        result.grounded = true;
      } else {
        result.grounded = false;
      }

      result.movement.copy(pos).sub(start);
      result.position.copy(pos);
      result.normalCount = 0;
      if (nrm.lengthSq() > 0.5 && Math.abs(nrm.y) < 0.7) {
        result.normals[result.normalCount++].copy(nrm);
        nrm.set(0, 0, 0);
      }
      return result;
    },

    setHeight(totalHeight) {
      const feet = pos.y - height / 2;
      height = totalHeight;
      pos.y = feet + height / 2;
      return true;
    },
    canStand() { return null; },
    dispose() {},
  };
}

/* -------------------------------------------------------------------------- */
/* The controller                                                              */
/* -------------------------------------------------------------------------- */

/**
 * @param {Object} deps
 * @param {THREE.PerspectiveCamera} deps.camera
 * @param {THREE.WebGLRenderer} deps.renderer
 * @param {Object} deps.physics
 * @param {Object} deps.terrain
 * @param {Object} deps.hud
 * @param {Object} deps.quality
 * @param {THREE.Scene} deps.scene
 * @param {{position:number[], yaw:number}} deps.spawn
 */
export function createPlayerController({ camera, renderer, physics, terrain, hud, quality, scene, spawn }) {
  warned = {};

  const dom = renderer?.domElement || document.body;
  const heightAt = terrainHeightFn(terrain);
  const cast = makeRaycast(physics);

  // Resolved once — probing for these per frame would allocate bound functions.
  const fnGrab = pickFn(physics, ['grab', 'pickUp', 'attach']);
  const fnRelease = pickFn(physics, ['release', 'drop', 'detach']);
  const fnThrow = pickFn(physics, ['throwHeld', 'throwObject', 'throwGrabbed']);
  const fnHeld = pickFn(physics, ['heldObject', 'getHeld']);
  const fnCarryTarget = pickFn(physics, ['setGrabTarget', 'setCarryTarget', 'moveGrabbed', 'updateGrab']);
  // If the physics stream velocity-drives the held body itself (it derives the
  // hold point from ctx.camera every frame), writing the transform from here
  // would only fight it.
  const physicsDrivesHold = !!(fnThrow || fnHeld);

  /* ----------------------------------------------------------------- state */

  const spawnPos = spawn?.position || WORLD.playerStart;
  const spawnYaw = spawn?.yaw ?? WORLD.playerStartYaw ?? 0;

  const standHeight = PLAYER.height;
  const crouchHeight = PLAYER.crouchHeight;
  const radius = PLAYER.radius;
  const maxSlopeCos = Math.cos(PLAYER.maxSlopeDeg * (Math.PI / 180));

  let capsuleHeight = standHeight;
  let capsuleResizable = true;

  // The body position is the CAPSULE CENTRE. `position`, exported to main.js,
  // is the interpolated render position — what culling and audio actually want.
  const bodyPos = new THREE.Vector3(
    spawnPos[0],
    (heightAt(spawnPos[0], spawnPos[2]) || spawnPos[1] || 0) + standHeight / 2,
    spawnPos[2],
  );
  const prevPos = bodyPos.clone();
  const position = bodyPos.clone();
  const velocity = new THREE.Vector3();

  let yaw = spawnYaw;
  let pitch = 0;
  let mouseDX = 0;
  let mouseDY = 0;
  let ignoreNextMouse = 0;
  let sensitivity = 1.0;
  let baseFov = camera?.fov ?? CAMERA.fov;

  let grounded = false;
  let coyote = 0;
  let jumpBuffered = -1;
  let ungroundSteps = 0;
  let crouching = false;
  let sprinting = false;
  let fallSpeed = 0;
  let lastLandImpact = 0;

  let bobDistance = 0;
  let bobAmp = 0;
  let eyeCurrent = PLAYER.eyeHeight;
  let rollCurrent = 0;
  let fovBoost = 0;
  let landOffset = 0;
  let landVel = 0;
  let breathT = 0;
  let stepAccum = 0;
  let selfStepAccum = 0;
  let headBobEnabled = true;

  let carryHandle = null;
  let carryObject = null;
  let carryName = null;
  let carrySpin = 0;
  let lookHit = null;
  let lookRoot = null;
  const promptCache = new WeakMap();

  /* ------------------------------------------------------------- character */

  const charOpts = {
    radius,
    height: standHeight,
    halfHeight: (standHeight - radius * 2) / 2,   // Rapier capsule: caps excluded
    crouchHeight,
    crouchHalfHeight: Math.max(0.02, (crouchHeight - radius * 2) / 2),
    position: bodyPos.clone(),
    maxSlopeDeg: PLAYER.maxSlopeDeg,
    maxSlope: PLAYER.maxSlopeDeg * (Math.PI / 180),
    stepHeight: PLAYER.stepHeight,
    skin: PLAYER.skin,
    offset: PLAYER.skin,
    snapToGround: PLAYER.snapToGround,
    up: [0, 1, 0],
    tag: 'player',
  };

  let character = adoptCharacter(physics, charOpts);
  if (!character) {
    warnOnce('fallbackChar', 'no character controller from the physics stream — using the terrain-following fallback');
    character = makeFallbackCharacter(charOpts, cast, heightAt);
  }
  character.setTranslation(bodyPos);

  /* --------------------------------------------------------------- lantern */

  // The one light this stream owns: parented to the camera, shadows off, so it
  // costs a single forward light and no shadow pass. Makes dusk usable.
  //
  // It stays `visible` with intensity 0 rather than being hidden: three keys
  // its program cache on the count of *visible* lights, so toggling visibility
  // mid-play would recompile every material in view. Baking the permutation in
  // at load costs one extra (zero-contribution) spot evaluation per fragment
  // and buys a hitch-free F key.
  const lantern = new THREE.SpotLight(
    TUNING.lantern.colour,
    0,
    TUNING.lantern.distance,
    TUNING.lantern.angle,
    TUNING.lantern.penumbra,
    TUNING.lantern.decay,
  );
  lantern.castShadow = false;
  lantern.visible = true;
  let lanternOn = false;
  lantern.position.set(0.16, -0.14, 0.05);
  const lanternTarget = new THREE.Object3D();
  lanternTarget.position.set(0.05, -0.06, -1);
  lantern.target = lanternTarget;
  camera.add(lantern);
  camera.add(lanternTarget);
  // Children of the camera only render if the camera itself is in the graph.
  if (!camera.parent && scene) scene.add(camera);

  camera.rotation.order = 'YXZ';

  /* ------------------------------------------------------------------ input */

  const keys = Object.create(null);
  let mouseThrow = false;
  let interactPressed = false;
  let jumpQueued = false;

  function formFocused() {
    const a = document.activeElement;
    if (!a || a === document.body) return false;
    if (a.isContentEditable) return true;
    const t = a.tagName;
    return t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT' || t === 'OPTION';
  }

  function onKeyDown(e) {
    if (e.repeat || formFocused()) return;
    const c = e.code;
    if (!MOVEMENT_CODES.has(c) && c !== 'KeyE' && c !== 'KeyF' && c !== 'KeyC') return;
    keys[c] = true;
    // Queue separately: a tap that starts and ends inside one frame must still
    // jump, and the fixed step may not have run in between.
    if (c === 'Space') { jumpQueued = true; e.preventDefault(); }   // else the page scrolls
    if (c === 'KeyE') interactPressed = true;
    if (c === 'KeyF') setLantern(!lanternOn);
  }

  function onKeyUp(e) {
    keys[e.code] = false;
  }

  function releaseAllKeys() {
    for (const k in keys) keys[k] = false;
    mouseThrow = false;
    interactPressed = false;
    jumpQueued = false;
  }

  function onMouseMove(e) {
    if (document.pointerLockElement !== dom) return;
    const dx = e.movementX || 0;
    const dy = e.movementY || 0;
    // Chrome emits one enormous movementX on the first event after lock.
    if (ignoreNextMouse > 0) {
      ignoreNextMouse--;
      if (Math.abs(dx) > 90 || Math.abs(dy) > 90) return;
    }
    if (Math.abs(dx) > TUNING.maxMouseDelta || Math.abs(dy) > TUNING.maxMouseDelta) return;
    mouseDX += dx;
    mouseDY += dy;
  }

  function onMouseDown(e) {
    if (document.pointerLockElement !== dom) {
      // Click-to-look, but only when nothing is covering the canvas.
      if (e.target === dom) requestPointerLock();
      return;
    }
    if (e.button === 0) mouseThrow = true;
  }

  function onPointerLockChange() {
    if (document.pointerLockElement === dom) {
      ignoreNextMouse = 2;
      mouseDX = 0;
      mouseDY = 0;
    } else {
      releaseAllKeys();
    }
  }

  function onPointerLockError() {
    warnOnce('lockErr', 'pointer lock refused (browser rate limit, or no user gesture)');
  }

  function onBlur() { releaseAllKeys(); }
  function onVisibility() { if (document.hidden) releaseAllKeys(); }

  window.addEventListener('keydown', onKeyDown, { passive: false });
  window.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', onBlur);
  document.addEventListener('visibilitychange', onVisibility);
  document.addEventListener('mousemove', onMouseMove);
  document.addEventListener('mousedown', onMouseDown);
  document.addEventListener('pointerlockchange', onPointerLockChange);
  document.addEventListener('pointerlockerror', onPointerLockError);

  function requestPointerLock() {
    if (document.pointerLockElement === dom) return;
    ignoreNextMouse = 2;
    try {
      // unadjustedMovement takes OS pointer acceleration out of the aim.
      const p = dom.requestPointerLock({ unadjustedMovement: true });
      if (p && typeof p.catch === 'function') {
        p.catch(() => { try { dom.requestPointerLock(); } catch { /* refused */ } });
      }
    } catch {
      try { dom.requestPointerLock(); } catch { /* refused */ }
    }
  }

  /** F. Intensity, not visibility — see the note where the light is built. */
  function setLantern(on) {
    lanternOn = !!on;
    lantern.intensity = lanternOn ? TUNING.lantern.intensity : 0;
  }

  function exitPointerLock() {
    if (document.pointerLockElement === dom) document.exitPointerLock();
    releaseAllKeys();
  }

  function keyDown(a, b, c) {
    return !!(keys[a] || (b && keys[b]) || (c && keys[c]));
  }

  /* -------------------------------------------------------------- crouching */

  /**
   * Where is the floor under (x,z)? A downward ray finds cobbles, steps and
   * rooftops, which the terrain height field alone does not know about.
   */
  function probeFloor(x, z, aroundY) {
    _tmpA.set(x, aroundY + 1.2, z);
    _tmpB.set(0, -1, 0);
    const hit = cast(_tmpA, _tmpB, 80, character.collider);
    if (hit) {
      const p = hit.point;
      if (p && Number.isFinite(p.y)) return p.y;
      const d = hitDistance(hit);
      if (Number.isFinite(d)) return _tmpA.y - d;
    }
    return heightAt(x, z);
  }

  /** Can we grow back to full height here? */
  function hasHeadroom() {
    const asked = character.canStand(standHeight);
    if (typeof asked === 'boolean') return asked;
    const feetY = bodyPos.y - capsuleHeight / 2;
    const gap = standHeight - crouchHeight + 0.08;
    _tmpA.set(bodyPos.x, feetY + crouchHeight - radius * 0.5, bodyPos.z);
    const hit = cast(_tmpA, _rayUp, gap + radius * 0.5, character.collider);
    return !hit || hitDistance(hit) > gap;
  }

  /**
   * Resize the capsule about its feet.
   * @returns {boolean} false when growing was refused for lack of headroom —
   *          the caller must stay crouched.
   */
  function setCapsule(totalHeight) {
    if (Math.abs(totalHeight - capsuleHeight) < 1e-4) return true;
    const growing = totalHeight > capsuleHeight;

    // Two independent headroom tests, and both must agree. Ours is a single
    // ray straight up (it under-detects, never over-detects); the controller's
    // is a shape cast. Neither alone is trustworthy for every controller.
    if (growing && !hasHeadroom()) return false;

    // The controller cannot resize at all: only the eye moves.
    if (!capsuleResizable) return true;

    const applied = character.setHeight(totalHeight, radius);
    if (applied === null) { capsuleResizable = false; return true; }
    if (applied === false) return false;   // controller refused: no headroom

    // Feet stay put; the capsule centre moves. A good controller repositions
    // itself — read it back rather than assuming.
    const oldCentre = bodyPos.y;
    const feet = oldCentre - capsuleHeight / 2;
    capsuleHeight = totalHeight;
    character.getTranslation(bodyPos);
    if (Math.abs(bodyPos.y - oldCentre) < 1e-6) {
      bodyPos.y = feet + capsuleHeight / 2;
      character.setTranslation(bodyPos);
    }
    // Shift the interpolation history too, or the camera pops for one frame.
    const dy = bodyPos.y - oldCentre;
    prevPos.y += dy;
    position.y += dy;
    return true;
  }

  /* ------------------------------------------------------------- fixed step */

  function fixedStep(h) {
    if (!(h > 0)) h = WORLD.fixedStep;
    stepAccum = Math.max(0, stepAccum - h);
    prevPos.copy(bodyPos);

    /* ----- intent -------------------------------------------------------- */
    const fwdIn = (keyDown('KeyW', 'ArrowUp') ? 1 : 0) - (keyDown('KeyS', 'ArrowDown') ? 1 : 0);
    const sideIn = (keyDown('KeyD', 'ArrowRight') ? 1 : 0) - (keyDown('KeyA', 'ArrowLeft') ? 1 : 0);
    const hasInput = fwdIn !== 0 || sideIn !== 0;

    const wantsCrouch = keyDown('ControlLeft', 'ControlRight', 'KeyC');
    if (wantsCrouch) {
      if (!crouching) { setCapsule(crouchHeight); crouching = true; }
    } else if (crouching && setCapsule(standHeight)) {
      crouching = false;   // stays true while something is over our head
    }

    sprinting = !crouching && fwdIn > 0 && keyDown('ShiftLeft', 'ShiftRight');

    const targetSpeed = crouching ? PLAYER.crouchSpeed
      : sprinting ? PLAYER.sprintSpeed
        : PLAYER.walkSpeed;

    // Wish direction in camera-yaw space, normalised so diagonals are not faster.
    _fwd.set(-Math.sin(yaw), 0, -Math.cos(yaw));
    _right.set(Math.cos(yaw), 0, -Math.sin(yaw));
    _wish.set(0, 0, 0);
    if (hasInput) _wish.addScaledVector(_fwd, fwdIn).addScaledVector(_right, sideIn).normalize();

    /* ----- horizontal: friction only when idle, accel toward the target --- */
    _tmpA.set(velocity.x, 0, velocity.z);
    if (grounded && !hasInput) {
      const speed = _tmpA.length();
      if (speed > 1e-4) {
        const drop = Math.max(speed * PLAYER.friction * h, PLAYER.friction * h * 0.35);
        _tmpA.multiplyScalar(Math.max(0, speed - drop) / speed);
      } else {
        _tmpA.set(0, 0, 0);
      }
    }
    if (hasInput) {
      _tmpB.copy(_wish).multiplyScalar(targetSpeed).sub(_tmpA);   // velocity error
      const err = _tmpB.length();
      const maxDelta = (grounded ? PLAYER.accelGround : PLAYER.accelAir) * h;
      if (err > 1e-5) _tmpA.addScaledVector(_tmpB, Math.min(1, maxDelta / err));
    }
    velocity.x = _tmpA.x;
    velocity.z = _tmpA.z;

    /* ----- vertical ------------------------------------------------------- */
    // Holding Space keeps the buffer topped up, so landing re-jumps cleanly.
    if (jumpQueued || keys.Space) { jumpBuffered = PLAYER.jumpBuffer; jumpQueued = false; }
    else if (jumpBuffered >= 0) jumpBuffered -= h;

    if (grounded && ungroundSteps === 0) {
      coyote = PLAYER.coyoteTime;
      if (velocity.y < 0) velocity.y = -TUNING.groundStick;   // keep snap engaged
    } else {
      coyote -= h;
      velocity.y += WORLD.gravity * h;
    }

    if (jumpBuffered >= 0 && coyote > 0 && !crouching) {
      velocity.y = PLAYER.jumpVelocity;
      jumpBuffered = -1;
      coyote = -1;
      grounded = false;
      ungroundSteps = TUNING.jumpUngroundSteps;
    }
    if (ungroundSteps > 0) ungroundSteps--;

    if (velocity.y < -TUNING.terminalVelocity) velocity.y = -TUNING.terminalVelocity;

    /* ----- move ----------------------------------------------------------- */
    _delta.copy(velocity).multiplyScalar(h);
    const res = character.move(_delta, h);
    bodyPos.copy(res.position);

    const wasGrounded = grounded;
    grounded = !!res.grounded && ungroundSteps === 0;

    // Is the controller lifting us over something (autostep)? If so the sweep
    // deliberately trades horizontal travel for height, and neither the slide
    // projection nor the reconcile below may touch the velocity — killing the
    // push is what turns a stride over a doorstep into a half-second crawl.
    const stepping = res.movement.y > 0.004 && grounded && velocity.y <= 0;

    // Slide: project the velocity out of any surface it is pushing into, so we
    // glide along a wall instead of sticking to it.
    for (let i = 0; !stepping && i < res.normalCount; i++) {
      const n = res.normals[i];
      const into = velocity.dot(n);
      if (into >= 0) continue;
      if (n.y > maxSlopeCos && grounded) continue;   // walkable floor
      velocity.addScaledVector(n, -into);
      // A face too steep to walk is a wall: never let it be climbed.
      if (n.y > 0.05 && n.y <= maxSlopeCos && velocity.y > 0) velocity.y = 0;
    }

    // Reconcile: if the controller moved us less than asked, the velocity is
    // lying. Rebuild it from the real displacement so speed cannot accumulate
    // against a wall.
    //
    const invH = 1 / h;
    if (!stepping) {
      const actualX = res.movement.x * invH;
      const actualZ = res.movement.z * invH;
      if (Math.abs(actualX) < Math.abs(velocity.x)) velocity.x = actualX;
      if (Math.abs(actualZ) < Math.abs(velocity.z)) velocity.z = actualZ;
    }
    if (ungroundSteps === 0 && velocity.y > 0 && res.movement.y < velocity.y * h * 0.5) {
      velocity.y = 0;   // bonked our head
    }

    /* ----- landing -------------------------------------------------------- */
    if (grounded && !wasGrounded) {
      const impact = Math.max(0, -fallSpeed);
      if (impact > TUNING.landMinSpeed) {
        landOffset -= Math.min(TUNING.landDipMax, impact * TUNING.landDipPerSpeed);
        landVel -= impact * 0.05;
        lastLandImpact = impact;
      }
    }
    fallSpeed = grounded ? 0 : velocity.y;
    if (grounded && velocity.y < 0) velocity.y = -TUNING.groundStick;

    /* ----- head bob is driven by distance, not time ----------------------- */
    if (grounded) {
      bobDistance = (bobDistance + Math.hypot(res.movement.x, res.movement.z)) % TUNING.bobStride;
    }
  }

  /* ------------------------------------------------ fixed-step registration */

  const stepCallback = (h) => fixedStep(h);
  let externallyDriven = false;
  let unsubscribe = null;
  if (typeof physics?.onFixedStep === 'function') {
    try {
      const r = physics.onFixedStep(stepCallback);
      externallyDriven = true;
      if (typeof r === 'function') unsubscribe = r;
    } catch (err) {
      warnOnce('onFixedStep', 'physics.onFixedStep threw — self-stepping instead:', err?.message);
    }
  }
  if (!externallyDriven) {
    warnOnce('selfStep', 'no physics.onFixedStep — movement self-steps at 1/60');
  }

  /* ---------------------------------------------------------- interaction */

  function updateInteraction() {
    lookHit = null;
    lookRoot = null;
    if (carryObject) { hud?.setPrompt?.(null); return; }

    camera.getWorldDirection(_rayDir);
    _rayOrigin.copy(camera.position).addScaledVector(_rayDir, TUNING.rayOriginOffset);
    const hit = cast(_rayOrigin, _rayDir, PLAYER.reach - TUNING.rayOriginOffset, character.collider);
    const root = interactiveRootOf(hitObject(hit));
    if (!root) { hud?.setPrompt?.(null); return; }

    lookHit = hit;
    lookRoot = root;
    hud?.setPrompt?.(promptFor(root));
  }

  /**
   * Interactables normally carry their own prompt. For the ones that only say
   * "grabbable", synthesise one ONCE and cache it — building the string every
   * frame would allocate in the frame path and defeat the HUD's dedupe.
   */
  function promptFor(root) {
    const ud = root.userData || {};
    if (ud.prompt) return ud.prompt;
    if (!ud.grabbable) return null;
    let s = promptCache.get(root);
    if (s === undefined) {
      s = `[E] pick up ${ud.name || ud.kind || 'it'}`;
      promptCache.set(root, s);
    }
    return s;
  }

  function tryGrab() {
    if (!lookRoot) return;
    if (lookRoot.userData?.grabbable === false) return;
    if (!fnGrab) { warnOnce('noGrab', 'physics exposes no grab() — pick-up is disabled'); return; }
    let handle = null;
    try {
      handle = fnGrab(lookRoot, {
        hit: lookHit,
        maxMass: PLAYER.maxCarryMass,
        distance: PLAYER.carryDistance,
        camera,
      });
    } catch (err) {
      warnOnce('grabThrow', 'physics.grab() threw:', err?.message);
      return;
    }
    if (handle === false || handle === null || handle === undefined) return;
    carryHandle = handle === true ? lookRoot : handle;
    carryObject = lookRoot;
    carryName = lookRoot.userData?.name || lookRoot.userData?.kind || lookRoot.name || 'it';
    carrySpin = 0;
    hud?.setPrompt?.(null);
    hud?.setCarrying?.(carryName);
  }

  function doRelease(thrown) {
    if (!carryHandle) return;
    camera.getWorldDirection(_rayDir);
    // Released objects inherit the player's velocity; thrown ones add the
    // throw on top, along the aim.
    _tmpA.copy(velocity);
    if (thrown) _tmpA.addScaledVector(_rayDir, PLAYER.throwImpulse);
    _tmpB.copy(_rayDir).multiplyScalar(thrown ? PLAYER.throwImpulse : 0);
    try {
      if (thrown && fnThrow) fnThrow(PLAYER.throwImpulse, { velocity: _tmpA, direction: _rayDir });
      else if (fnRelease) fnRelease(carryHandle, { velocity: _tmpA, impulse: _tmpB, thrown });
    } catch (err) {
      warnOnce('releaseThrow', 'physics release/throw threw:', err?.message);
    }
    clearCarry();
  }

  function clearCarry() {
    carryHandle = null;
    carryObject = null;
    carryName = null;
    hud?.setCarrying?.(null);
  }

  function updateCarry(dt) {
    if (!carryObject) return;

    // The physics stream breaks the hold on its own if the object gets wedged.
    // Poll it so the HUD cannot lie about what we are carrying.
    if (fnHeld && !fnHeld()) { clearCarry(); return; }
    if (physicsDrivesHold && !fnCarryTarget) return;   // it owns the transform

    camera.getWorldDirection(_rayDir);
    _carryPos.copy(camera.position).addScaledVector(_rayDir, PLAYER.carryDistance);
    carrySpin += TUNING.carrySpin * dt;
    _quatA.setFromAxisAngle(_spinAxis, carrySpin);
    _quatB.copy(camera.quaternion).multiply(_quatA);

    if (fnCarryTarget) {
      try { fnCarryTarget(carryHandle, _carryPos, _quatB); return; }
      catch (err) { warnOnce('carryTarget', 'physics carry-target hook threw:', err?.message); }
    }
    // No target hook: drive the transform ourselves. Correct if the physics
    // stream switched the body to kinematic on grab, harmless (overwritten) if
    // it drives the body itself.
    carryObject.position.copy(_carryPos);
    carryObject.quaternion.slerp(_quatB, 1 - Math.exp(-TUNING.carryLambda * dt));
  }

  /* -------------------------------------------------------------- the frame */

  const api = {
    position,
    velocity,
    grounded: false,
    stats: {
      mode: character.kind,
      x: 0, y: 0, z: 0, speed: 0,
      grounded: false, crouching: false, sprinting: false,
      carrying: null, lantern: false, fov: baseFov, yaw: 0, lastLandImpact: 0,
    },

    update(dt) {
      if (!(dt > 0)) dt = WORLD.fixedStep;

      /* ---- look ---------------------------------------------------------- */
      if (mouseDX !== 0 || mouseDY !== 0) {
        const s = TUNING.baseSensitivity * sensitivity;
        yaw -= mouseDX * s;
        pitch -= mouseDY * s;
        pitch = clamp(pitch, -TUNING.pitchLimit, TUNING.pitchLimit);
        if (yaw > Math.PI * 8 || yaw < -Math.PI * 8) yaw %= Math.PI * 2;  // keep precision
        mouseDX = 0;
        mouseDY = 0;
      }

      /* ---- movement ------------------------------------------------------ */
      stepAccum += dt;
      if (!externallyDriven) {
        const h = WORLD.fixedStep;
        selfStepAccum += dt;
        let n = 0;
        while (selfStepAccum >= h && n < WORLD.maxSubSteps) {
          selfStepAccum -= h;
          fixedStep(h);
          n++;
        }
        if (selfStepAccum > h * WORLD.maxSubSteps) selfStepAccum = 0;   // give up catching up
      }

      /* ---- interpolated body position ------------------------------------ */
      position.lerpVectors(prevPos, bodyPos, clamp(stepAccum / WORLD.fixedStep, 0, 1));

      /* ---- eye ------------------------------------------------------------ */
      eyeCurrent = damp(eyeCurrent, crouching ? PLAYER.crouchEyeHeight : PLAYER.eyeHeight,
        TUNING.eyeLambda, dt);

      // Landing dip — a damped spring back to zero.
      const k = TUNING.landSpring;
      const c = 2 * TUNING.landDampingRatio * Math.sqrt(k);
      landVel += (-k * landOffset - c * landVel) * dt;
      landOffset += landVel * dt;
      if (Math.abs(landOffset) < 1e-4 && Math.abs(landVel) < 1e-3) { landOffset = 0; landVel = 0; }

      const speed = Math.hypot(velocity.x, velocity.z);
      const speedRatio = clamp(speed / PLAYER.walkSpeed, 0, 1.7);

      // Head bob: a 2:1 Lissajous parameterised by DISTANCE TRAVELLED, so it
      // freezes the instant you stop instead of drifting on a timer.
      const targetAmp = (headBobEnabled && grounded) ? speedRatio * (crouching ? 0.55 : 1) : 0;
      bobAmp = damp(bobAmp, targetAmp, TUNING.bobAmpLambda, dt);
      const phase = (bobDistance / TUNING.bobStride) * Math.PI * 2;
      const bobY = Math.sin(phase * 2) * TUNING.bobVertical * bobAmp;
      const bobX = Math.sin(phase) * TUNING.bobLateral * bobAmp;
      const bobRoll = Math.sin(phase) * TUNING.bobRoll * bobAmp;

      // Idle breathing.
      breathT += dt;
      const idle = clamp(1 - speed / 0.9, 0, 1);
      const breathY = Math.sin(breathT * 1.15) * TUNING.breathAmp * idle;
      const breathPitch = Math.sin(breathT * 0.83) * TUNING.breathPitch * idle;
      const breathYaw = Math.sin(breathT * 0.47) * TUNING.breathYaw * idle;

      // Strafe roll.
      const strafeIn = (keys.KeyD || keys.ArrowRight ? 1 : 0) - (keys.KeyA || keys.ArrowLeft ? 1 : 0);
      rollCurrent = damp(rollCurrent,
        -strafeIn * TUNING.strafeRoll * clamp(speed / PLAYER.walkSpeed, 0, 1),
        TUNING.strafeRollLambda, dt);

      // Sprint FOV.
      const sprintAmount = sprinting ? clamp(speed / PLAYER.sprintSpeed, 0, 1) : 0;
      fovBoost = damp(fovBoost, sprintAmount * TUNING.sprintFov, TUNING.fovLambda, dt);
      const wantFov = baseFov + fovBoost;
      if (Math.abs(camera.fov - wantFov) > 0.01) {
        camera.fov = wantFov;
        camera.updateProjectionMatrix();
      }

      /* ---- commit the camera ---------------------------------------------- */
      camera.rotation.set(pitch + breathPitch, yaw + breathYaw, rollCurrent + bobRoll, 'YXZ');
      const feetY = position.y - capsuleHeight / 2;
      camera.position.set(
        position.x,
        feetY + eyeCurrent + bobY + breathY + landOffset,
        position.z,
      );
      if (bobX !== 0) {
        _right.set(Math.cos(yaw), 0, -Math.sin(yaw));
        camera.position.addScaledVector(_right, bobX);
      }
      camera.updateMatrixWorld();

      /* ---- interaction ----------------------------------------------------- */
      updateInteraction();
      if (interactPressed) {
        interactPressed = false;
        if (carryObject) doRelease(false); else tryGrab();
      }
      if (mouseThrow) {
        mouseThrow = false;
        if (carryObject) doRelease(true);
      }
      updateCarry(dt);

      /* ---- exported state --------------------------------------------------- */
      api.grounded = grounded;
      const st = api.stats;
      st.x = round2(position.x);
      st.y = round2(position.y);
      st.z = round2(position.z);
      st.speed = round2(speed);
      st.grounded = grounded;
      st.crouching = crouching;
      st.sprinting = sprinting;
      st.carrying = carryName;
      st.lantern = lanternOn;
      st.fov = round2(wantFov);
      st.yaw = round2(yaw);
      st.lastLandImpact = round2(lastLandImpact);
    },

    requestPointerLock,
    exitPointerLock,

    setSensitivity(v) { if (Number.isFinite(v) && v > 0) sensitivity = v; },

    setFov(v) {
      if (!Number.isFinite(v)) return;
      baseFov = v;
      camera.fov = baseFov + fovBoost;
      camera.updateProjectionMatrix();
    },

    setHeadBob(on) {
      headBobEnabled = !!on;
      if (!headBobEnabled) bobAmp = 0;
    },

    setLantern,

    setPitch(rad) {
      if (Number.isFinite(rad)) pitch = clamp(rad, -TUNING.pitchLimit, TUNING.pitchLimit);
    },

    setYaw(rad) { if (Number.isFinite(rad)) yaw = rad; },

    /**
     * Move the physics BODY, not just the camera.
     *
     * `y` is interpreted generously, because callers disagree about it: the
     * ground height (`teleport(x, 0, z)`) and the EYE height
     * (`teleport(x, 1.68, z)`, which is what the audit tool's viewpoints use)
     * both mean "stand here". Anything more than an eye height above the floor
     * is taken literally, so you can still teleport onto a roof.
     */
    teleport(x, y, z, yawRad) {
      const floor = probeFloor(x, z, Number.isFinite(y) ? y : 0);
      let feet = Number.isFinite(y) ? y : floor;
      if (feet - floor <= PLAYER.eyeHeight + 0.05) feet = floor;
      bodyPos.set(x, feet + capsuleHeight / 2, z);
      prevPos.copy(bodyPos);
      position.copy(bodyPos);
      velocity.set(0, 0, 0);
      landOffset = 0; landVel = 0; bobAmp = 0; fallSpeed = 0;
      grounded = false;
      coyote = 0;
      jumpBuffered = -1;
      ungroundSteps = 0;
      character.setTranslation(bodyPos);
      if (Number.isFinite(yawRad)) yaw = yawRad;
      camera.position.set(x, feet + eyeCurrent, z);
      camera.rotation.set(pitch, yaw, 0, 'YXZ');
      camera.updateMatrixWorld();
      return api;
    },

    /** Escape hatch for the review pass. */
    debug: {
      character,
      body: bodyPos,
      get yaw() { return yaw; },
      get pitch() { return pitch; },
      set yaw(v) { yaw = v; },
      set pitch(v) { pitch = clamp(v, -TUNING.pitchLimit, TUNING.pitchLimit); },
    },

    dispose() {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onBlur);
      document.removeEventListener('visibilitychange', onVisibility);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mousedown', onMouseDown);
      document.removeEventListener('pointerlockchange', onPointerLockChange);
      document.removeEventListener('pointerlockerror', onPointerLockError);
      try { unsubscribe && unsubscribe(); } catch { /* best effort */ }
      if (typeof physics?.offFixedStep === 'function') {
        try { physics.offFixedStep(stepCallback); } catch { /* best effort */ }
      }
      camera.remove(lantern);
      camera.remove(lanternTarget);
      lantern.dispose?.();
      character.dispose?.();
      exitPointerLock();
    },
  };

  // Face the spawn direction before the first frame, so the hand-over from the
  // loading screen shows the composed view rather than a random heading.
  camera.rotation.set(0, yaw, 0, 'YXZ');
  camera.position.set(bodyPos.x, bodyPos.y - standHeight / 2 + PLAYER.eyeHeight, bodyPos.z);
  camera.updateMatrixWorld();

  return api;
}
