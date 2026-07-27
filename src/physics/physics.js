/**
 * ============================================================================
 *  PHYSICS  —  Rapier world, fixed timestep, character controller, grabbing
 * ============================================================================
 * Everything that is solid in Hollowbrook goes through this module. Geometry
 * streams never touch Rapier: they emit `ColliderSpec` / `InteractableSpec`
 * (see `contracts.js`) and this file turns them into colliders and bodies.
 *
 * Three things here are worth knowing before you change anything:
 *
 *  1. THE STEP IS FIXED. `update(dt)` accumulates real time and steps the world
 *     at exactly WORLD.fixedStep. Rendering interpolates between the previous
 *     and current body transforms with the leftover accumulator — without that,
 *     props visibly stutter at any refresh rate that is not exactly 60 Hz.
 *
 *  2. RAPIER COLLIDERS HAVE NO userData. We keep our own
 *     `Map<colliderHandle, {tag, object3D, spec}>` so a raycast can name what
 *     it hit. Never attach state to a Rapier object; it will not survive.
 *
 *  3. THE PLAYER IS SWEPT, NEVER TELEPORTED. The kinematic character
 *     controller shape-casts the capsule along the desired motion and slides
 *     it along contacts. That is what guarantees "no clipping through walls".
 *     Held objects are moved by VELOCITY for the same reason.
 *
 * ---------------------------------------------------------------------------
 * HEIGHTFIELD CONVENTION  (verified empirically against rapier3d-compat 0.19.3,
 * not guessed — a transposed heightfield is a silent, world-breaking bug)
 *
 *   RAPIER.ColliderDesc.heightfield(nrows, ncols, heights, scale)
 *
 *   - `nrows` / `ncols` are CELL counts, not vertex counts.
 *     `heights.length` must be `(nrows + 1) * (ncols + 1)`.
 *   - The buffer is COLUMN-MAJOR:   index = row + col * (nrows + 1)
 *   - ROWS run along local **Z**, COLUMNS run along local **X**:
 *         x = (-0.5 + col / ncols) * scale.x
 *         z = (-0.5 + row / nrows) * scale.z
 *         y = heights[row + col * (nrows + 1)] * scale.y
 *     The field is centred on the collider's translation.
 *
 *   Two escape hatches for a terrain stream that packs it differently. Both are
 *   re-packed once, here, at build time — never per frame:
 *     field.order = 'row-major'   buffer index is row*(ncols+1) + col
 *     field.rowsAlongX = true     rows index X and columns index Z, so `nrows`
 *                                 counts cells along X and `ncols` along Z
 *   `field.scale` always means world extents on [X, Y, Z] whichever is set.
 *   Verified equivalent by ray-probing all three layouts of the same surface.
 * ============================================================================
 */

import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';
import { WORLD, PLAYER } from '../config.js';
import { DEFAULT_FRICTION, DEFAULT_RESTITUTION } from '../contracts.js';

/* -------------------------------------------------------------------------- */
/* Scratch — module scope, never allocated per frame                           */
/* -------------------------------------------------------------------------- */

const _v3a = new THREE.Vector3();
const _v3b = new THREE.Vector3();
const _v3c = new THREE.Vector3();
const _qa = new THREE.Quaternion();
const _qb = new THREE.Quaternion();
const _mat = new THREE.Matrix4();
const ONE = new THREE.Vector3(1, 1, 1);

/** Plain {x,y,z} scratch — Rapier wants POJOs, not THREE.Vector3. */
const _rv = { x: 0, y: 0, z: 0 };
const _rv2 = { x: 0, y: 0, z: 0 };
const _rq = { x: 0, y: 0, z: 0, w: 1 };
const IDENTITY_ROT = Object.freeze({ x: 0, y: 0, z: 0, w: 1 });
const UP = Object.freeze({ x: 0, y: 1, z: 0 });

const DEG = Math.PI / 180;
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/** Hold tuning. Velocity control, so a carried crate still collides. */
const HOLD_RESPONSE = 0.32;      // fraction of the gap closed per step
const HOLD_MAX_SPEED = 14;       // m/s — stops a stuck object from launching
const HOLD_ANGULAR_DAMP = 0.82;  // per step multiplier on angular velocity
const HOLD_BREAK_DISTANCE = 2.6; // metres past the hold point before we let go

/* -------------------------------------------------------------------------- */

/**
 * Build the physics world.
 *
 * @param {Object} opts
 * @param {[number,number,number]} opts.gravity
 * @param {Object} opts.quality        active quality preset (uses maxDynamicBodies)
 * @param {THREE.Scene} [opts.scene]   where the debug LineSegments is parented
 */
export async function createPhysics({ gravity = [0, WORLD.gravity, 0], quality = {}, scene = null } = {}) {
  await RAPIER.init();

  const g = gravity && gravity.length === 3 ? gravity : [0, WORLD.gravity, 0];
  const world = new RAPIER.World({ x: g[0], y: g[1], z: g[2] });
  world.timestep = WORLD.fixedStep;
  // The village is human-scaled; 1 unit == 1 metre, which is Rapier's default,
  // but say so explicitly because every tolerance is derived from it.
  world.lengthUnit = 1;

  /** One shared fixed body carries every static collider in the village. */
  const staticBody = world.createRigidBody(RAPIER.RigidBodyDesc.fixed());

  /** colliderHandle -> { tag, object3D, spec, kind } */
  const colliderInfo = new Map();
  /** object3D -> interactable record */
  const byObject = new Map();

  /** @type {Array} interactable records */
  const records = [];
  /** @type {Array} character records */
  const characters = [];

  const stepCallbacks = [];

  let dynamicCount = 0;
  const maxDynamicBodies = Number.isFinite(quality.maxDynamicBodies) ? quality.maxDynamicBodies : 90;

  let accumulator = 0;
  let primed = false;
  let disposed = false;

  const stats = {
    bodies: 0,
    colliders: 0,
    active: 0,
    stepMs: 0,
    substeps: 0,
    staticColliders: 0,
    interactables: 0,
    cappedBodies: 0,
    heldMass: 0,
  };

  /* ------------------------------------------------------------------ */
  /* Shape translation                                                   */
  /* ------------------------------------------------------------------ */

  /**
   * ColliderSpec -> RAPIER.ColliderDesc. Returns null (and warns) for a spec we
   * cannot honour; the caller skips it rather than losing the whole batch.
   */
  function descFor(spec) {
    if (!spec || !spec.shape) return null;
    let desc = null;
    try {
      switch (spec.shape) {
        case 'box': {
          const h = spec.halfExtents || [0.5, 0.5, 0.5];
          desc = RAPIER.ColliderDesc.cuboid(
            Math.max(1e-3, h[0]), Math.max(1e-3, h[1]), Math.max(1e-3, h[2]));
          break;
        }
        case 'sphere':
          desc = RAPIER.ColliderDesc.ball(Math.max(1e-3, spec.radius || 0.5));
          break;
        case 'capsule':
          desc = RAPIER.ColliderDesc.capsule(
            Math.max(1e-3, spec.halfHeight ?? 0.5), Math.max(1e-3, spec.radius || 0.3));
          break;
        case 'cylinder':
          desc = RAPIER.ColliderDesc.cylinder(
            Math.max(1e-3, spec.halfHeight ?? 0.5), Math.max(1e-3, spec.radius || 0.3));
          break;
        case 'cone':
          desc = RAPIER.ColliderDesc.cone(
            Math.max(1e-3, spec.halfHeight ?? 0.5), Math.max(1e-3, spec.radius || 0.3));
          break;
        case 'trimesh': {
          if (!spec.vertices || !spec.indices || spec.indices.length < 3) return null;
          const verts = spec.vertices instanceof Float32Array
            ? spec.vertices : new Float32Array(spec.vertices);
          const idx = spec.indices instanceof Uint32Array
            ? spec.indices : new Uint32Array(spec.indices);
          desc = RAPIER.ColliderDesc.trimesh(verts, idx);
          break;
        }
        case 'heightfield': {
          desc = heightfieldDesc(spec);
          break;
        }
        default:
          console.warn(`[physics] unknown collider shape "${spec.shape}"`);
          return null;
      }
    } catch (err) {
      console.warn('[physics] could not build collider', spec.shape, spec.tag, err);
      return null;
    }
    if (!desc) return null;

    const p = spec.position || [0, 0, 0];
    desc.setTranslation(p[0] || 0, p[1] || 0, p[2] || 0);
    const q = spec.quaternion;
    if (q && (q[0] || q[1] || q[2] || q[3] !== 1)) {
      _rq.x = q[0] || 0; _rq.y = q[1] || 0; _rq.z = q[2] || 0;
      _rq.w = q[3] === undefined ? 1 : q[3];
      desc.setRotation(_rq);
    }
    desc.setFriction(spec.friction === undefined ? DEFAULT_FRICTION : spec.friction);
    desc.setRestitution(spec.restitution === undefined ? DEFAULT_RESTITUTION : spec.restitution);
    if (spec.sensor) desc.setSensor(true);
    return desc;
  }

  /**
   * Heightfield, with the convention documented at the top of this file.
   *
   * The terrain stream may describe its buffer with two optional flags:
   *   field.order      'column-major' (default) | 'row-major'
   *   field.rowsAlongX false (default: rows run along Z) | true
   * Anything other than the default pair is re-packed ONCE, here, at build
   * time. `scale` always means world extents on X / Y / Z regardless.
   */
  function heightfieldDesc(spec) {
    const f = spec.field;
    if (!f || !f.heights) {
      console.warn('[physics] heightfield collider with no field');
      return null;
    }
    let nrows = f.nrows | 0;
    let ncols = f.ncols | 0;
    const src = f.heights instanceof Float32Array ? f.heights : new Float32Array(f.heights);
    const s = f.scale || [1, 1, 1];
    const scale = Array.isArray(s)
      ? { x: s[0] === undefined ? 1 : s[0], y: s[1] === undefined ? 1 : s[1], z: s[2] === undefined ? 1 : s[2] }
      : { x: s.x === undefined ? 1 : s.x, y: s.y === undefined ? 1 : s.y, z: s.z === undefined ? 1 : s.z };

    // Rapier's nrows/ncols are CELL counts; the buffer holds the vertex grid.
    // Tolerate a terrain stream that passed vertex counts instead.
    const wantCells = (nrows + 1) * (ncols + 1);
    const wantVerts = nrows * ncols;
    if (heightsFit(src.length, wantCells)) {
      /* already right */
    } else if (heightsFit(src.length, wantVerts) && nrows > 1 && ncols > 1) {
      console.info('[physics] heightfield: nrows/ncols look like vertex counts ' +
        `(${nrows}x${ncols}, ${src.length} samples) — reading them as ${nrows - 1}x${ncols - 1} cells.`);
      nrows -= 1; ncols -= 1;
    } else {
      console.warn(`[physics] heightfield size mismatch: ${nrows}x${ncols} cells needs ` +
        `${wantCells} samples, got ${src.length}. Skipping this collider.`);
      return null;
    }

    const R = nrows + 1;           // producer rows
    const C = ncols + 1;           // producer columns
    const rowMajor = f.order === 'row-major' || f.rowMajor === true;
    const rowsAlongX = !!f.rowsAlongX;

    if (!rowMajor && !rowsAlongX) {
      // Exactly what Rapier wants — no copy.
      return RAPIER.ColliderDesc.heightfield(nrows, ncols, src, scale);
    }

    // Re-pack into Rapier's layout: out[i + j*Rout], i indexes Z, j indexes X.
    const Rout = rowsAlongX ? C : R;
    const Cout = rowsAlongX ? R : C;
    const out = new Float32Array(Rout * Cout);
    for (let r = 0; r < R; r++) {
      for (let c = 0; c < C; c++) {
        const h = rowMajor ? src[r * C + c] : src[r + c * R];
        const i = rowsAlongX ? c : r;    // Z index
        const j = rowsAlongX ? r : c;    // X index
        out[i + j * Rout] = h;
      }
    }
    return RAPIER.ColliderDesc.heightfield(Rout - 1, Cout - 1, out, scale);
  }

  function heightsFit(len, want) { return want > 0 && len === want; }

  /* ------------------------------------------------------------------ */
  /* Static colliders                                                    */
  /* ------------------------------------------------------------------ */

  /**
   * @param {import('../contracts.js').ColliderSpec[]} specs  world space
   * @returns {number} how many colliders were actually created
   */
  function addStaticColliders(specs) {
    if (!specs || !specs.length) return 0;
    let made = 0;
    for (let i = 0; i < specs.length; i++) {
      const spec = specs[i];
      const desc = descFor(spec);
      if (!desc) continue;
      let col;
      try {
        col = world.createCollider(desc, staticBody);
      } catch (err) {
        console.warn('[physics] createCollider failed for', spec.tag || spec.shape, err);
        continue;
      }
      colliderInfo.set(col.handle, {
        tag: spec.tag || spec.shape,
        object3D: spec.object3D || null,
        spec,
        kind: 'static',
      });
      made++;
    }
    stats.staticColliders += made;
    prime();
    return made;
  }

  /* ------------------------------------------------------------------ */
  /* Interactables                                                       */
  /* ------------------------------------------------------------------ */

  /**
   * @param {import('../contracts.js').InteractableSpec[]} specs
   * @returns {Array} handles, in the order the specs were given
   */
  function addInteractables(specs) {
    if (!specs || !specs.length) return [];

    // Grabbable and jointed things get first claim on the dynamic budget;
    // everything past `maxDynamicBodies` becomes a solid but immobile prop.
    const order = specs.map((s, i) => i);
    order.sort((a, b) => priority(specs[a]) - priority(specs[b]));

    const out = new Array(specs.length);
    for (const i of order) {
      out[i] = addInteractable(specs[i]);
    }
    prime();
    return out.filter(Boolean);
  }

  function priority(s) {
    if (!s) return 3;
    if (s.grabbable) return 0;
    if (s.joint) return 1;
    return 2;
  }

  function addInteractable(spec) {
    if (!spec || !spec.object3D || !spec.collider) return null;
    const obj = spec.object3D;
    obj.updateWorldMatrix(true, false);
    obj.getWorldPosition(_v3a);
    obj.getWorldQuaternion(_qa);

    let type = spec.body || 'dynamic';
    let capped = false;
    if (type === 'dynamic' && dynamicCount >= maxDynamicBodies) {
      type = 'fixed';
      capped = true;
      stats.cappedBodies++;
    }

    let bodyDesc;
    if (type === 'dynamic') bodyDesc = RAPIER.RigidBodyDesc.dynamic();
    else if (type === 'kinematic') bodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased();
    else bodyDesc = RAPIER.RigidBodyDesc.fixed();

    bodyDesc.setTranslation(_v3a.x, _v3a.y, _v3a.z);
    _rq.x = _qa.x; _rq.y = _qa.y; _rq.z = _qa.z; _rq.w = _qa.w;
    bodyDesc.setRotation(_rq);
    if (type === 'dynamic') {
      bodyDesc.setLinearDamping(spec.linearDamping === undefined ? 0.12 : spec.linearDamping);
      bodyDesc.setAngularDamping(spec.angularDamping === undefined ? 0.35 : spec.angularDamping);
      if (spec.ccd) bodyDesc.setCcdEnabled(true);
      // Soft CCD is cheap and stops thin props from tunnelling at throw speed.
      bodyDesc.setSoftCcdPrediction(0.5);
    }

    let body;
    try {
      body = world.createRigidBody(bodyDesc);
    } catch (err) {
      console.warn('[physics] createRigidBody failed for', spec.tag, err);
      return null;
    }

    const desc = descFor(spec.collider);
    if (!desc) {
      world.removeRigidBody(body);
      return null;
    }
    const mass = spec.mass === undefined ? 8 : spec.mass;
    if (type === 'dynamic') desc.setMass(Math.max(0.05, mass));

    let collider;
    try {
      collider = world.createCollider(desc, body);
    } catch (err) {
      console.warn('[physics] interactable collider failed for', spec.tag, err);
      world.removeRigidBody(body);
      return null;
    }

    if (type === 'dynamic') dynamicCount++;

    // Where do we write the transform back to? object3D.position is LOCAL to
    // its parent; cache the parent's inverse world matrix once (parents are
    // static groups) so the common "parent at origin" case costs nothing.
    let parentInverse = null;
    const parent = obj.parent;
    if (parent) {
      parent.updateWorldMatrix(true, false);
      const e = parent.matrixWorld.elements;
      const isIdentity =
        e[0] === 1 && e[1] === 0 && e[2] === 0 &&
        e[4] === 0 && e[5] === 1 && e[6] === 0 &&
        e[8] === 0 && e[9] === 0 && e[10] === 1 &&
        e[12] === 0 && e[13] === 0 && e[14] === 0;
      if (!isIdentity) parentInverse = new THREE.Matrix4().copy(parent.matrixWorld).invert();
    }

    const rec = {
      spec,
      object3D: obj,
      body,
      collider,
      type,
      capped,
      mass,
      tag: spec.tag || 'prop',
      grabbable: !!spec.grabbable && type === 'dynamic',
      joint: null,
      parentInverse,
      // interpolation state
      px: _v3a.x, py: _v3a.y, pz: _v3a.z,
      pqx: _qa.x, pqy: _qa.y, pqz: _qa.z, pqw: _qa.w,
      settled: false,
      removed: false,
    };

    colliderInfo.set(collider.handle, {
      tag: rec.tag,
      object3D: obj,
      spec,
      kind: 'interactable',
      record: rec,
    });
    byObject.set(obj, rec);
    records.push(rec);
    stats.interactables++;

    if (spec.joint && type !== 'fixed') {
      rec.joint = makeJoint(spec.joint, body, _v3a, _qa, rec);
    }

    return makeHandle(rec);
  }

  /** Attach a body to the world at `joint.anchorWorld`. */
  function makeJoint(j, body, spawnPos, spawnQuat, rec) {
    try {
      const aw = j.anchorWorld || [spawnPos.x, spawnPos.y, spawnPos.z];
      // anchor1 is in staticBody space, which is the identity — world space.
      const a1 = { x: aw[0], y: aw[1], z: aw[2] };
      // anchor2 is in the body's local space.
      _v3b.set(aw[0], aw[1], aw[2]).sub(spawnPos).applyQuaternion(_qb.copy(spawnQuat).invert());
      const a2 = { x: _v3b.x, y: _v3b.y, z: _v3b.z };

      let data = null;
      switch (j.type) {
        case 'revolute': {
          const ax = j.axis || [0, 1, 0];
          data = RAPIER.JointData.revolute(a1, a2, { x: ax[0], y: ax[1], z: ax[2] });
          break;
        }
        case 'spherical':
          data = RAPIER.JointData.spherical(a1, a2);
          break;
        case 'fixed':
          // frame1 carries the spawn orientation so the body keeps its pose.
          data = RAPIER.JointData.fixed(
            a1, { x: spawnQuat.x, y: spawnQuat.y, z: spawnQuat.z, w: spawnQuat.w },
            a2, IDENTITY_ROT);
          break;
        case 'rope': {
          const len = j.length === undefined
            ? Math.max(0.05, _v3c.set(aw[0], aw[1], aw[2]).distanceTo(spawnPos))
            : j.length;
          data = RAPIER.JointData.rope(len, a1, a2);
          break;
        }
        case 'spring': {
          const rest = j.length === undefined
            ? Math.max(0.02, _v3c.set(aw[0], aw[1], aw[2]).distanceTo(spawnPos))
            : j.length;
          data = RAPIER.JointData.spring(
            rest, j.stiffness === undefined ? 220 : j.stiffness,
            j.damping === undefined ? 14 : j.damping, a1, a2);
          break;
        }
        default:
          console.warn(`[physics] unknown joint type "${j.type}"`);
          return null;
      }
      return world.createImpulseJoint(data, staticBody, body, true);
    } catch (err) {
      console.warn('[physics] joint creation failed for', rec.tag, err);
      return null;
    }
  }

  function makeHandle(rec) {
    return {
      body: rec.body,
      collider: rec.collider,
      object3D: rec.object3D,
      tag: rec.tag,
      grabbable: rec.grabbable,
      capped: rec.capped,
      mass: rec.mass,
      joint: rec.joint,
      get record() { return rec; },
      remove() { removeRecord(rec); },
    };
  }

  function removeRecord(rec) {
    if (!rec || rec.removed) return;
    rec.removed = true;
    if (held === rec) release();
    colliderInfo.delete(rec.collider.handle);
    byObject.delete(rec.object3D);
    const i = records.indexOf(rec);
    if (i >= 0) records.splice(i, 1);
    try {
      if (rec.joint) world.removeImpulseJoint(rec.joint, true);
      world.removeRigidBody(rec.body); // removes its colliders too
    } catch { /* already gone */ }
    if (rec.type === 'dynamic') dynamicCount--;
    stats.interactables--;
  }

  /* ------------------------------------------------------------------ */
  /* Character controller                                                */
  /* ------------------------------------------------------------------ */

  /**
   * A swept capsule on a KinematicPositionBased body. This is the only thing
   * standing between the player and the inside of a wall, so it is deliberately
   * conservative: autostep on, snap-to-ground on, slide on.
   */
  function createCharacter({
    radius = PLAYER.radius,
    halfHeight = Math.max(0.01, PLAYER.height / 2 - PLAYER.radius),
    position = [0, 2, 0],
  } = {}) {
    const p = Array.isArray(position)
      ? { x: position[0], y: position[1], z: position[2] }
      : { x: position.x, y: position.y, z: position.z };

    const body = world.createRigidBody(
      RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(p.x, p.y, p.z));

    const colDesc = RAPIER.ColliderDesc.capsule(halfHeight, radius)
      .setFriction(0)
      .setRestitution(0)
      .setActiveCollisionTypes(RAPIER.ActiveCollisionTypes.ALL);
    const collider = world.createCollider(colDesc, body);

    const controller = world.createCharacterController(PLAYER.skin);
    controller.setUp(UP);
    controller.enableAutostep(PLAYER.stepHeight, PLAYER.stepMinWidth ?? 0.25, true);
    controller.enableSnapToGround(PLAYER.snapToGround);
    controller.setMaxSlopeClimbAngle(PLAYER.maxSlopeDeg * DEG);
    // Anything we cannot climb, we slide off — keeps the player from parking
    // on a 60 degree thatch roof.
    controller.setMinSlopeSlideAngle(PLAYER.maxSlopeDeg * DEG);
    controller.setApplyImpulsesToDynamicBodies(true);
    controller.setCharacterMass(78);
    controller.setSlideEnabled(true);

    const state = {
      radius,
      halfHeight,
      totalHeight: (halfHeight + radius) * 2,
      pendingX: 0, pendingY: 0, pendingZ: 0,
      hasPending: false,
      collisions: 0,
    };

    const corrected = new THREE.Vector3();
    const translationOut = new THREE.Vector3();
    const feetOut = new THREE.Vector3();

    // Hoisted so computeMovement never allocates.
    const filterPredicate = (c) => {
      if (c.handle === collider.handle) return false;
      if (held && c.handle === held.collider.handle) return false;
      return true;
    };

    const handle = {
      body,
      collider,
      controller,
      grounded: false,
      radius,
      get halfHeight() { return state.halfHeight; },
      get totalHeight() { return state.totalHeight; },

      /** Queue a translation delta to be swept at the next fixed step. */
      setDesiredMovement(v) {
        state.pendingX = v.x || 0;
        state.pendingY = v.y || 0;
        state.pendingZ = v.z || 0;
        state.hasPending = true;
        return handle;
      },

      /**
       * Sweep `desired` (a translation delta, metres) against the world, move
       * the body to the corrected position, and return the correction.
       * The returned vector is REUSED — copy it if you need to keep it.
       */
      computeMovement(desired, dt = WORLD.fixedStep) {
        _rv.x = desired.x || 0; _rv.y = desired.y || 0; _rv.z = desired.z || 0;
        state.hasPending = false;
        try {
          controller.computeColliderMovement(
            collider, _rv, RAPIER.QueryFilterFlags.EXCLUDE_SENSORS, undefined, filterPredicate);
        } catch (err) {
          // Never let a solver hiccup freeze the player.
          console.warn('[physics] character sweep failed', err);
          corrected.set(_rv.x, _rv.y, _rv.z);
          return corrected;
        }
        const mv = controller.computedMovement();
        corrected.set(mv.x, mv.y, mv.z);
        handle.grounded = controller.computedGrounded();
        state.collisions = controller.numComputedCollisions();

        const t = body.translation();
        _rv2.x = t.x + mv.x; _rv2.y = t.y + mv.y; _rv2.z = t.z + mv.z;
        body.setNextKinematicTranslation(_rv2);
        return corrected;
      },

      /** Capsule CENTRE, world space. Reused vector, distinct from feet(). */
      translation() {
        const t = body.translation();
        return translationOut.set(t.x, t.y, t.z);
      },

      /** Sole of the capsule, world space. Reused vector. */
      feet() {
        const t = body.translation();
        return feetOut.set(t.x, t.y - state.totalHeight * 0.5, t.z);
      },

      setTranslation(v) {
        _rv.x = v.x || 0; _rv.y = v.y || 0; _rv.z = v.z || 0;
        body.setTranslation(_rv, true);
        body.setNextKinematicTranslation(_rv);
        return handle;
      },

      /**
       * Crouch / stand. Keeps the FEET fixed and resizes the capsule about
       * them. Growing is refused unless a shape-cast straight up proves there
       * is room, so standing up under a beam can never push the head through
       * geometry.
       * @returns {boolean} true if the new height was applied
       */
      setHeight(totalHeight) {
        const minTotal = state.radius * 2 + 0.02;
        const want = Math.max(minTotal, totalHeight);
        const old = state.totalHeight;
        if (Math.abs(want - old) < 1e-4) return true;

        const t = body.translation();
        const feetY = t.y - old * 0.5;

        if (want > old) {
          const rise = want - old;
          // The swept volume of the CURRENT capsule moved up by `rise` is
          // exactly the volume the TALLER capsule will occupy (same radius,
          // same feet). So one shape-cast is a complete headroom test.
          let blocked = false;
          try {
            _rv.x = t.x; _rv.y = t.y; _rv.z = t.z;
            const shape = new RAPIER.Capsule(state.halfHeight, state.radius);
            const hit = world.castShape(
              _rv, IDENTITY_ROT, UP, shape, 0, rise, false,
              RAPIER.QueryFilterFlags.EXCLUDE_SENSORS, undefined, collider, body,
              filterPredicate);
            if (hit && hit.time_of_impact <= rise) {
              // Ignore anything whose outward normal points up — that is a
              // floor we grazed, not a ceiling, and refusing on it would trap
              // the player in a crouch forever.
              const n = hit.normal1;
              if (!n || n.y < 0.6) blocked = true;
            }
          } catch (err) {
            console.warn('[physics] headroom cast failed, refusing to grow', err);
            blocked = true;
          }
          if (blocked) return false;
        }

        const newHalf = Math.max(0.01, want * 0.5 - state.radius);
        try {
          collider.setShape(new RAPIER.Capsule(newHalf, state.radius));
        } catch (err) {
          console.warn('[physics] capsule resize failed', err);
          return false;
        }
        state.halfHeight = newHalf;
        state.totalHeight = (newHalf + state.radius) * 2;

        _rv.x = t.x; _rv.y = feetY + state.totalHeight * 0.5; _rv.z = t.z;
        body.setTranslation(_rv, true);
        body.setNextKinematicTranslation(_rv);
        return true;
      },

      /** Number of obstacles the last sweep touched — useful for footstep FX. */
      get collisions() { return state.collisions; },

      dispose() {
        const i = characters.indexOf(entry);
        if (i >= 0) characters.splice(i, 1);
        try { world.removeCharacterController(controller); } catch { /* ignore */ }
        try { world.removeRigidBody(body); } catch { /* ignore */ }
        colliderInfo.delete(collider.handle);
      },
    };

    const entry = { handle, state, body, collider, controller };
    characters.push(entry);
    colliderInfo.set(collider.handle, { tag: 'player', object3D: null, spec: null, kind: 'character' });
    return handle;
  }

  /** Run the queued sweep for every character, just before world.step(). */
  function applyCharacters(step) {
    for (let i = 0; i < characters.length; i++) {
      const c = characters[i];
      if (!c.state.hasPending) continue;
      _v3c.set(c.state.pendingX, c.state.pendingY, c.state.pendingZ);
      c.handle.computeMovement(_v3c, step);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Grabbing                                                            */
  /* ------------------------------------------------------------------ */

  let held = null;
  const holdTarget = new THREE.Vector3();
  const cameraForward = new THREE.Vector3(0, 0, -1);
  let holdValid = false;

  function resolveRecord(x) {
    if (!x) return null;
    if (x.record) return x.record;                       // one of our handles
    if (x.isObject3D) {
      let o = x;
      while (o) {
        const r = byObject.get(o);
        if (r) return r;
        o = o.parent;
      }
      const root = x.userData && x.userData.interactiveRoot;
      if (root && root !== x) return byObject.get(root) || null;
      return null;
    }
    if (typeof x === 'number') {
      const info = colliderInfo.get(x);
      return info ? info.record || null : null;
    }
    if (x.handle !== undefined && colliderInfo.has(x.handle)) {
      const info = colliderInfo.get(x.handle);
      return info.record || null;
    }
    return null;
  }

  /**
   * Pick something up. Refused for anything that is not a live dynamic body,
   * or heavier than PLAYER.maxCarryMass.
   * @returns {boolean} whether the grab succeeded
   */
  function grab(x) {
    const rec = resolveRecord(x);
    if (!rec || rec.removed) return false;
    if (rec.type !== 'dynamic') return false;
    if (rec.joint) return false;                 // bolted to the world
    const m = rec.body.mass ? rec.body.mass() : rec.mass;
    if (m > PLAYER.maxCarryMass) return false;
    if (held) release();

    held = rec;
    rec.body.setLinearDamping(2.4);
    rec.body.setAngularDamping(6.0);
    rec.body.enableCcd(true);            // carried objects move fast; do not tunnel
    rec.body.wakeUp();
    stats.heldMass = m;
    holdValid = false;
    return true;
  }

  function release() {
    if (!held) return null;
    const rec = held;
    held = null;
    stats.heldMass = 0;
    try {
      rec.body.setLinearDamping(rec.spec.linearDamping === undefined ? 0.12 : rec.spec.linearDamping);
      rec.body.setAngularDamping(rec.spec.angularDamping === undefined ? 0.35 : rec.spec.angularDamping);
      rec.body.enableCcd(!!rec.spec.ccd);
      rec.body.wakeUp();
    } catch { /* body already gone */ }
    return rec.object3D;
  }

  /**
   * Throw whatever is held along the camera forward.
   * `impulse` is read as a TARGET SPEED in m/s and scaled by the body's mass,
   * so a barrel and a bucket leave the hand at the same speed.
   */
  function throwHeld(impulse = PLAYER.throwImpulse) {
    if (!held) return false;
    const rec = held;
    const m = rec.body.mass ? Math.max(0.05, rec.body.mass()) : rec.mass;
    release();
    const speed = typeof impulse === 'number' ? impulse : PLAYER.throwImpulse;
    _v3a.copy(cameraForward).normalize().multiplyScalar(speed * m);
    _rv.x = _v3a.x; _rv.y = _v3a.y + m * 1.2; _rv.z = _v3a.z; // a little loft
    try { rec.body.applyImpulse(_rv, true); } catch { /* body already gone */ }
    return true;
  }

  function heldObject() { return held ? held.object3D : null; }

  /** Velocity-drive the held body toward the hold point. Runs per fixed step. */
  function updateHeld(step) {
    if (!held || !holdValid) return;
    const b = held.body;
    const t = b.translation();
    const dx = holdTarget.x - t.x;
    const dy = holdTarget.y - t.y;
    const dz = holdTarget.z - t.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (dist > HOLD_BREAK_DISTANCE) { release(); return; }

    const k = HOLD_RESPONSE / step;
    let vx = dx * k, vy = dy * k, vz = dz * k;
    const sp = Math.sqrt(vx * vx + vy * vy + vz * vz);
    if (sp > HOLD_MAX_SPEED) {
      const s = HOLD_MAX_SPEED / sp;
      vx *= s; vy *= s; vz *= s;
    }
    _rv.x = vx; _rv.y = vy; _rv.z = vz;
    b.setLinvel(_rv, true);

    const av = b.angvel();
    _rv2.x = av.x * HOLD_ANGULAR_DAMP;
    _rv2.y = av.y * HOLD_ANGULAR_DAMP;
    _rv2.z = av.z * HOLD_ANGULAR_DAMP;
    b.setAngvel(_rv2, false);
  }

  /* ------------------------------------------------------------------ */
  /* Raycast                                                             */
  /* ------------------------------------------------------------------ */

  const _ray = new RAPIER.Ray({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: -1 });
  const _hitOut = {
    point: new THREE.Vector3(),
    normal: new THREE.Vector3(),
    distance: 0,
    collider: null,
    object3D: null,
    userData: null,
    tag: '',
    body: null,
    record: null,
  };

  /**
   * Cast a ray. The player capsule is excluded by default.
   * The returned object is REUSED between calls — read it immediately.
   *
   * A ray that lands EXACTLY on a heightfield cell boundary slips between the
   * cells and reports nothing (verified against rapier 0.19.3: a downward ray
   * at x=30.0 misses, x=30.01 hits). It is a measure-zero artefact, but the
   * player spawns at x=0, z=34 — exactly the kind of round number that lands
   * on a grid line. So on a miss we retry once, nudged sideways by a tenth of
   * a millimetre. Misses are the rare path; the extra cast is free in practice.
   *
   * @param {THREE.Vector3} origin
   * @param {THREE.Vector3} dir             need not be normalised
   * @param {number} maxToi                 metres
   * @param {Object} [opts] {solid, excludeHeld, excludeCollider, excludeBody,
   *                         filter, sensors, retryOnMiss}
   */
  function raycast(origin, dir, maxToi = PLAYER.reach, opts) {
    let dx = dir.x, dy = dir.y, dz = dir.z;
    const len = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
    dx /= len; dy /= len; dz /= len;
    _ray.dir.x = dx; _ray.dir.y = dy; _ray.dir.z = dz;
    _ray.origin.x = origin.x; _ray.origin.y = origin.y; _ray.origin.z = origin.z;

    const o = opts || EMPTY_OPTS;
    const flags = o.sensors ? 0 : RAPIER.QueryFilterFlags.EXCLUDE_SENSORS;
    let exclude = o.excludeCollider !== undefined
      ? o.excludeCollider
      : (characters.length ? characters[0].collider : undefined);
    if (exclude === null) exclude = undefined;
    const excludeBody = o.excludeBody === null ? undefined : o.excludeBody;
    const solid = o.solid === undefined ? true : o.solid;

    rayExcludeHeld = o.excludeHeld === false ? null : held;
    rayUserFilter = o.filter || null;

    let hit = castOnce(maxToi, solid, flags, exclude, excludeBody);
    if (!hit && o.retryOnMiss !== false) {
      // Offset perpendicular to dir, on BOTH perpendicular axes — nudging on
      // only one would leave a vertical ray sitting on the same x cell edge.
      _v3b.set(dx, dy, dz);
      const ax = Math.abs(dx), ay = Math.abs(dy), az = Math.abs(dz);
      if (ax <= ay && ax <= az) _v3c.set(1, 0, 0);
      else if (ay <= az) _v3c.set(0, 1, 0);
      else _v3c.set(0, 0, 1);
      _v3c.cross(_v3b).normalize();          // first perpendicular
      _v3b.cross(_v3c).normalize();          // second, orthogonal to both
      _v3c.add(_v3b).multiplyScalar(7.1e-5);
      _ray.origin.x = origin.x + _v3c.x;
      _ray.origin.y = origin.y + _v3c.y;
      _ray.origin.z = origin.z + _v3c.z;
      hit = castOnce(maxToi, solid, flags, exclude, excludeBody);
      _ray.origin.x = origin.x; _ray.origin.y = origin.y; _ray.origin.z = origin.z;
    }
    if (!hit) return null;

    const toi = hit.timeOfImpact;
    _hitOut.point.set(origin.x + dx * toi, origin.y + dy * toi, origin.z + dz * toi);
    _hitOut.normal.set(hit.normal.x, hit.normal.y, hit.normal.z);
    _hitOut.distance = toi;
    _hitOut.collider = hit.collider;
    const info = hit.collider ? colliderInfo.get(hit.collider.handle) : null;
    _hitOut.object3D = info ? info.object3D : null;
    _hitOut.record = info ? info.record || null : null;
    _hitOut.tag = info ? info.tag : '';
    _hitOut.userData = _hitOut.object3D ? _hitOut.object3D.userData : (info ? info.spec : null);
    _hitOut.body = _hitOut.record ? _hitOut.record.body : null;
    return _hitOut;
  }

  function castOnce(maxToi, solid, flags, exclude, excludeBody) {
    try {
      return world.castRayAndGetNormal(
        _ray, maxToi, solid, flags, undefined, exclude, excludeBody, rayPredicate);
    } catch (err) {
      console.warn('[physics] raycast failed', err);
      return null;
    }
  }

  const EMPTY_OPTS = {};
  let rayExcludeHeld = null;
  let rayUserFilter = null;
  const rayPredicate = (c) => {
    if (rayExcludeHeld && c.handle === rayExcludeHeld.collider.handle) return false;
    if (rayUserFilter) return rayUserFilter(c, colliderInfo.get(c.handle));
    return true;
  };

  /* ------------------------------------------------------------------ */
  /* Debug draw                                                          */
  /* ------------------------------------------------------------------ */

  let debugOn = false;
  let debugLines = null;
  let debugPos = null;
  let debugCol = null;
  let debugCapacity = 0;

  function setDebug(on) {
    debugOn = !!on;
    if (!debugOn) {
      if (debugLines) debugLines.visible = false;
      return;
    }
    if (!debugLines) {
      if (!scene) { console.warn('[physics] setDebug: no scene to draw into'); debugOn = false; return; }
      const geo = new THREE.BufferGeometry();
      const mat = new THREE.LineBasicMaterial({
        vertexColors: true, toneMapped: false, depthTest: true, depthWrite: false,
      });
      debugLines = new THREE.LineSegments(geo, mat);
      debugLines.frustumCulled = false;
      debugLines.renderOrder = 999;
      debugLines.name = 'physics-debug';
      scene.add(debugLines);
    }
    debugLines.visible = true;
    drawDebug();
  }

  function drawDebug() {
    if (!debugOn || !debugLines) return;
    const buf = world.debugRender();
    const verts = buf.vertices;
    const cols = buf.colors;
    const n = verts.length / 3; // vertices

    if (n > debugCapacity) {
      // Grow in chunks so we are not reallocating every frame the world changes.
      debugCapacity = Math.max(1024, Math.ceil(n * 1.5));
      debugPos = new Float32Array(debugCapacity * 3);
      debugCol = new Float32Array(debugCapacity * 3);
      const geo = debugLines.geometry;
      geo.setAttribute('position', new THREE.BufferAttribute(debugPos, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(debugCol, 3));
    }
    debugPos.set(verts);
    // Rapier hands back RGBA; three's colour attribute here is RGB.
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      debugCol[i * 3] = cols[j];
      debugCol[i * 3 + 1] = cols[j + 1];
      debugCol[i * 3 + 2] = cols[j + 2];
    }
    const geo = debugLines.geometry;
    geo.attributes.position.needsUpdate = true;
    geo.attributes.color.needsUpdate = true;
    geo.setDrawRange(0, n);
  }

  /* ------------------------------------------------------------------ */
  /* Step loop                                                           */
  /* ------------------------------------------------------------------ */

  function onFixedStep(fn) {
    if (typeof fn === 'function' && stepCallbacks.indexOf(fn) === -1) stepCallbacks.push(fn);
  }
  function offFixedStep(fn) {
    const i = stepCallbacks.indexOf(fn);
    if (i >= 0) stepCallbacks.splice(i, 1);
  }

  /** One step, so the broad phase is populated before the first query. */
  function prime() {
    if (primed) return;
    try { world.step(); primed = true; } catch (err) { console.warn('[physics] prime step failed', err); }
  }

  const step = WORLD.fixedStep;
  const maxSub = Math.max(1, WORLD.maxSubSteps | 0);

  function update(dt, ctx) {
    if (disposed) return;
    if (!Number.isFinite(dt) || dt <= 0) dt = 0;

    // Where should a held object sit? Derived from the camera, once per frame.
    const cam = ctx && ctx.camera;
    if (cam) {
      cam.getWorldDirection(cameraForward);
      cam.getWorldPosition(_v3a);
      holdTarget.copy(cameraForward).multiplyScalar(PLAYER.carryDistance).add(_v3a);
      holdValid = true;
    }

    accumulator += Math.min(dt, 0.25);

    const t0 = now();
    let steps = 0;
    while (accumulator >= step && steps < maxSub) {
      savePrevious();
      for (let i = 0; i < stepCallbacks.length; i++) {
        try { stepCallbacks[i](step, ctx); }
        catch (err) { console.warn('[physics] fixed-step callback threw', err); }
      }
      applyCharacters(step);
      updateHeld(step);
      try { world.step(); }
      catch (err) { console.error('[physics] world.step failed', err); accumulator = 0; break; }
      accumulator -= step;
      steps++;
    }
    // Spiral-of-death guard: if we are still behind after maxSubSteps, throw the
    // backlog away. Simulating slower than real time is better than never
    // catching up and locking the tab.
    if (accumulator > step) accumulator = step * 0.999;

    stats.substeps = steps;
    if (steps > 0) {
      const ms = now() - t0;
      stats.stepMs = stats.stepMs === 0 ? ms : stats.stepMs * 0.9 + ms * 0.1;
      primed = true;
    }

    writeTransforms(accumulator / step);

    stats.bodies = world.bodies.len();
    stats.colliders = world.colliders.len();
    if (debugOn) drawDebug();
  }

  /** Snapshot body transforms before a step, for interpolation. */
  function savePrevious() {
    for (let i = 0; i < records.length; i++) {
      const r = records[i];
      if (r.type === 'fixed' || r.removed) continue;
      if (r.body.isSleeping()) continue;   // prev already equals cur
      const t = r.body.translation();
      const q = r.body.rotation();
      r.px = t.x; r.py = t.y; r.pz = t.z;
      r.pqx = q.x; r.pqy = q.y; r.pqz = q.z; r.pqw = q.w;
    }
  }

  /** Blend prev -> cur by `alpha` and push onto the Object3Ds. */
  function writeTransforms(alpha) {
    const a = alpha < 0 ? 0 : alpha > 1 ? 1 : alpha;
    let active = 0;
    for (let i = 0; i < records.length; i++) {
      const r = records[i];
      if (r.type === 'fixed' || r.removed) continue;
      const sleeping = r.body.isSleeping();
      if (sleeping) {
        if (r.settled) continue;
        r.settled = true;
      } else {
        r.settled = false;
        active++;
      }
      const w = sleeping ? 1 : a;

      const t = r.body.translation();
      const q = r.body.rotation();
      _v3a.set(
        r.px + (t.x - r.px) * w,
        r.py + (t.y - r.py) * w,
        r.pz + (t.z - r.pz) * w);
      _qa.set(r.pqx, r.pqy, r.pqz, r.pqw);
      _qb.set(q.x, q.y, q.z, q.w);
      _qa.slerp(_qb, w);

      const obj = r.object3D;
      if (r.parentInverse) {
        _mat.compose(_v3a, _qa, ONE);
        _mat.premultiply(r.parentInverse);
        _mat.decompose(obj.position, obj.quaternion, _v3b);
      } else {
        obj.position.copy(_v3a);
        obj.quaternion.copy(_qa);
      }
    }
    stats.active = active;
  }

  /* ------------------------------------------------------------------ */

  function dispose() {
    if (disposed) return;
    disposed = true;
    stepCallbacks.length = 0;
    held = null;
    if (debugLines) {
      debugLines.parent && debugLines.parent.remove(debugLines);
      debugLines.geometry.dispose();
      debugLines.material.dispose();
      debugLines = null;
    }
    colliderInfo.clear();
    byObject.clear();
    records.length = 0;
    characters.length = 0;
    try { world.free(); } catch { /* already freed */ }
  }

  return {
    RAPIER,
    world,
    staticBody,

    addStaticColliders,
    addInteractables,
    createCharacter,

    onFixedStep,
    offFixedStep,
    update,

    raycast,
    grab,
    release,
    throwHeld,
    heldObject,

    setDebug,
    stats,
    dispose,

    /* --- extras the review pass will want from the console --------------- */
    /** What did we hit? colliderHandle -> {tag, object3D, spec, kind}. */
    info: (colliderHandle) => colliderInfo.get(colliderHandle) || null,
    /** Live interactable records (read-only in spirit). */
    records,
    /** How the heightfield buffer is interpreted; see the header comment. */
    heightfieldConvention:
      'nrows/ncols are CELL counts; heights.length === (nrows+1)*(ncols+1); ' +
      'column-major, index = row + col*(nrows+1); rows run along Z, columns along X',
    get accumulator() { return accumulator; },
    get held() { return held; },
  };
}

export default createPhysics;
