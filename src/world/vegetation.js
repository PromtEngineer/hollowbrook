/**
 * ============================================================================
 *  VEGETATION — trees, bushes, hedgerows, grass, wildflowers, ground clutter
 * ============================================================================
 * Everything here is instanced and wind-animated in the vertex shader. The CPU
 * never touches a vertex after build: `update()` only writes uniforms and, when
 * the player has actually moved, re-shuffles instance matrices that already
 * exist (LOD buckets for trees, a recycled lattice for the grass carpets).
 *
 * Trees run THREE LODs — full geometry to 60 m, the same tree with its cards
 * decimated to 180 m, crossed cards beyond — because the band right behind the
 * rooflines is most of the frame and cards do not survive there. See
 * `TREE_NEAR_M` and `buildConifer`.
 *
 * The wind patch is injected into BOTH the surface material and a matching
 * `customDepthMaterial`, otherwise the shadows detach from the leaves.
 * ============================================================================
 */

import * as THREE from 'three';
import { Rng, valueNoise2D, fbm2D } from '../util/rng.js';
import { cylinderCollider } from '../contracts.js';
import { WORLD } from '../config.js';
import {
  TREE_SITES, HILLS, PLOTS, ROADS, WELL,
  isClearGround, insideAnyBuilding, insidePlaza, plazaRadiusAt,
  plotToWorld,
} from './layout.js';

/* -------------------------------------------------------------------------- */
/* Scratch — hoisted so nothing allocates per frame                            */
/* -------------------------------------------------------------------------- */

const _m4 = new THREE.Matrix4();
const _m4b = new THREE.Matrix4();
const _nm3 = new THREE.Matrix3();
const _q = new THREE.Quaternion();
const _qb = new THREE.Quaternion();
const _v = new THREE.Vector3();
const _vb = new THREE.Vector3();
const _vc = new THREE.Vector3();
const _scl = new THREE.Vector3();
const _col = new THREE.Color();
/** Meadow variation sample — one object, refilled in place, never allocated. */
const _field = { patch: 0.5, dry: 0.5, shade: 0 };
const _UP = new THREE.Vector3(0, 1, 0);
const _sphere = new THREE.Sphere(new THREE.Vector3(), 1200);

const TAU = Math.PI * 2;
const GOLDEN = 2.39996323;

/*
 * THREE tree LODs, not two. The old two-tier split put the billboard crossover
 * at 80 m, and measured from the spawn eye that is exactly where the treeline
 * behind the rooflines sits: a raycast into what looked like a spiky bare tree
 * resolved to 'conifer-billboard' at 79.5 m. Worse, the near bucket only ever
 * held 19 trees against 3 393 billboards, so essentially the whole visible
 * forest was cards.
 *
 * The band that matters is 60-180 m — trees directly behind the rooflines, a
 * large part of the frame — so it gets a real mid LOD: the SAME tree, built from
 * the same rng stream, with the cards decimated to ~1.5 per branch at one
 * segment each. A mid conifer is 575 triangles against the full tree's 2 324,
 * and because the mid geometry is a strict subset of the full tree's cards the
 * 60 m crossover changes density, not shape.
 */
const TREE_NEAR_M = 60;     // full detail inside this
const TREE_MID_M = 180;     // reduced-card real geometry out to this
const TREE_FAR_CULL_M = 760; // billboards stop here (farthest hill summit 669 m)

/*
 * Deep-field billboard decimation. Past this radius FROM THE VILLAGE CENTRE a
 * fixed 35 % of the forest is dropped and the survivors are widened, which is
 * ink-neutral (0.65 x 1.50 = 0.98) and saves ~800 billboards. Keyed on the
 * tree's own distance from the centre rather than from the player on purpose:
 * a player-relative test would make trees blink in and out along the boundary
 * every time the walk-refresh fires. The player cannot leave the 300 m bound
 * ring, so a decimated tree is never closer than 120 m.
 */
const TREE_DECIM_R = 420;
const TREE_DECIM_KEEP = 0.65;
const TREE_DECIM_WIDTH = 1.50;
const TREE_DECIM_HEIGHT = 1.02;

/* -------------------------------------------------------------------------- */
/* Terrain adapter — the terrain stream owns the real height field, but this    */
/* module must still build if it failed or names its sampler differently.       */
/* -------------------------------------------------------------------------- */

const _fallbackNoise = valueNoise2D(90210);

/** Rough stand-in for the terrain height field, derived from layout's HILLS. */
function fallbackHeight(x, z) {
  let h = 0;
  for (let i = 0; i < HILLS.length; i++) {
    const hl = HILLS[i];
    const d = Math.hypot(x - hl.x, z - hl.z);
    if (d >= hl.radius) continue;
    const t = 1 - d / hl.radius;
    h += hl.height * Math.pow(t * t * (3 - 2 * t), hl.sharpness);
  }
  // Gentle rolling on top, then flattened over the village bowl.
  h += fbm2D(_fallbackNoise, x * 0.006, z * 0.006, 3) * 3.2;
  const r = Math.hypot(x, z);
  const flat = THREE.MathUtils.smoothstep(r, WORLD.villageRadius, WORLD.villageRadius * 2.4);
  return h * flat;
}

/**
 * Road keep-out. `layout.roadDistance` returns an object and this runs inside
 * the per-frame grass refresh, so it is reimplemented here allocation-free —
 * and per-road, which is also more correct than testing the nearest road's
 * width against a different road's carriageway.
 */
let _hitRoadSurface = 'dirt';
function nearRoad(x, z, clear) {
  for (let i = 0; i < ROADS.length; i++) {
    const road = ROADS[i];
    const lim = road.width * 0.5 + clear;
    const lim2 = lim * lim;
    const pts = road.points;
    for (let j = 0; j < pts.length - 1; j++) {
      const ax = pts[j][0], az = pts[j][1];
      const vx = pts[j + 1][0] - ax, vz = pts[j + 1][1] - az;
      const len2 = vx * vx + vz * vz || 1;
      let t = ((x - ax) * vx + (z - az) * vz) / len2;
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const dx = x - (ax + vx * t), dz = z - (az + vz * t);
      if (dx * dx + dz * dz < lim2) { _hitRoadSurface = road.surface; return true; }
    }
  }
  return false;
}

/**
 * Stand-in for `terrain.surfaceAt` — mirrors the terrain stream's own rules
 * closely enough that a scatter never plants on paving if terrain failed.
 * @returns {'cobble'|'dirt'|'grass'|'stone'}
 */
function fallbackSurface(x, z) {
  if (insidePlaza(x, z)) return 'cobble';
  if (nearRoad(x, z, 0.5)) return _hitRoadSurface === 'cobble' ? 'cobble' : 'dirt';
  const r = Math.hypot(x, z);
  if (r < WORLD.villageRadius) {
    if (insideAnyBuilding(x, z, 1.8)) return 'dirt';
    if (r < plazaRadiusAt(Math.atan2(z, x)) + 2.4) return 'dirt';
  }
  return 'grass';
}

/**
 * How far up a hill's flank (x,z) sits — 0 at the foot, 1 at the crown, from
 * layout's HILLS rather than the height field so it is independent of whatever
 * rolling noise the terrain adds. Used to thin the forest toward the ridges: a
 * real tree line stops short of the top.
 */
function hillRelief(x, z) {
  let best = 0, rel = 0;
  for (let i = 0; i < HILLS.length; i++) {
    const hl = HILLS[i];
    const d = Math.hypot(x - hl.x, z - hl.z);
    if (d >= hl.radius) continue;
    const t = 1 - d / hl.radius;
    const c = hl.height * Math.pow(t * t * (3 - 2 * t), hl.sharpness);
    if (c > best) { best = c; rel = c / hl.height; }
  }
  return rel;
}

function makeTerrainAdapter(terrain) {
  const h =
    (terrain && typeof terrain.heightAt === 'function' && terrain.heightAt.bind(terrain)) ||
    (terrain && typeof terrain.getHeight === 'function' && terrain.getHeight.bind(terrain)) ||
    (terrain && typeof terrain.sampleHeight === 'function' && terrain.sampleHeight.bind(terrain)) ||
    null;

  const heightAt = h
    ? (x, z) => { const y = h(x, z); return Number.isFinite(y) ? y : 0; }
    : fallbackHeight;

  const s =
    (terrain && typeof terrain.slopeAt === 'function' && terrain.slopeAt.bind(terrain)) || null;

  // Some terrain implementations return radians, some return a 0..1 gradient.
  // Finite differences are cheap and unambiguous, so prefer them when the
  // terrain does not expose a slope of its own.
  const slopeAt = s
    ? (x, z) => {
      const v = s(x, z);
      if (!Number.isFinite(v)) return 0;
      return v > Math.PI ? Math.atan(v) : v; // guard against degree-returning impls
    }
    : (x, z) => {
      const e = 1.5;
      const dx = (heightAt(x + e, z) - heightAt(x - e, z)) / (2 * e);
      const dz = (heightAt(x, z + e) - heightAt(x, z - e)) / (2 * e);
      return Math.atan(Math.hypot(dx, dz));
    };

  // The surface classifier is the gate that keeps plants off the setts, so it
  // must never be missing: fall back to layout-derived rules.
  const sf =
    (terrain && typeof terrain.surfaceAt === 'function' && terrain.surfaceAt.bind(terrain)) || null;
  const surfaceAt = sf
    ? (x, z) => { const s = sf(x, z); return typeof s === 'string' ? s : 'grass'; }
    : fallbackSurface;

  return { heightAt, slopeAt, surfaceAt, synthetic: !h, syntheticSurface: !sf };
}

/* -------------------------------------------------------------------------- */
/* Geometry builder                                                            */
/* -------------------------------------------------------------------------- */

/**
 * A tiny triangle-soup accumulator. Build time only — this never runs per
 * frame, so plain arrays are fine and readable beats clever here.
 */
class Geo {
  constructor() {
    this.pos = [];
    this.nor = [];
    this.uv = [];
    this.flex = [];
    this.idx = [];
  }

  get vertexCount() { return this.pos.length / 3; }

  vert(px, py, pz, nx, ny, nz, u, v, f) {
    this.pos.push(px, py, pz);
    this.nor.push(nx, ny, nz);
    this.uv.push(u, v);
    this.flex.push(f);
    return this.vertexCount - 1;
  }

  tri(a, b, c) { this.idx.push(a, b, c); }
  quad(a, b, c, d) { this.idx.push(a, b, c, a, c, d); }

  build(name) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.pos), 3));
    g.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(this.nor), 3));
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(this.uv), 2));
    g.setAttribute('aFlex', new THREE.BufferAttribute(new Float32Array(this.flex), 1));
    g.setIndex(new THREE.BufferAttribute(
      this.pos.length / 3 > 65535 ? new Uint32Array(this.idx) : new Uint16Array(this.idx), 1));
    g.computeBoundingSphere();
    g.name = name || 'hb-veg';
    return g;
  }
}

/**
 * One foliage / grass card. Local space: spine along +Y from the origin, width
 * along X, bending along +Z. Everything else in this file is built from this.
 *
 * @param {Geo} g
 * @param {THREE.Matrix4} mtx     placement
 * @param {Object} o              width, height, segs, bend, taper, uv, flexBase,
 *                                flexTip, flexPow, upBlend, radial (Vector3|null)
 */
function addCard(g, mtx, o) {
  const segs = o.segs || 2;
  const w = o.width;
  const h = o.height;
  const bend = o.bend || 0;
  const taper = o.taper === undefined ? 0.25 : o.taper;
  const uv = o.uv || UV_FULL;
  const flexBase = o.flexBase || 0;
  const flexTip = o.flexTip === undefined ? 1 : o.flexTip;
  const flexPow = o.flexPow || 1.6;
  const upBlend = o.upBlend === undefined ? 0.4 : o.upBlend;
  _nm3.getNormalMatrix(mtx);

  const start = g.vertexCount;
  for (let iy = 0; iy <= segs; iy++) {
    const t = iy / segs;
    const y = t * h;
    const z = bend * t * t;
    const dz = (2 * bend * t) / (h || 1);
    // Perpendicular to the spine, in the YZ plane.
    const nl = 1 / Math.hypot(dz, 1);
    const nyL = -dz * nl;
    const nzL = 1 * nl;
    const halfW = (w * 0.5) * (1 - taper * t);
    const f = flexBase + (flexTip - flexBase) * Math.pow(t, flexPow);
    for (let ix = 0; ix <= 1; ix++) {
      _v.set((ix - 0.5) * 2 * halfW, y, z).applyMatrix4(mtx);
      _vb.set(0, nyL, nzL).applyMatrix3(_nm3).normalize();
      if (o.radial) {
        _vc.copy(_v).sub(o.radial);
        if (_vc.lengthSq() > 1e-6) _vb.lerp(_vc.normalize(), 0.7).normalize();
      } else if (upBlend > 0) {
        _vb.lerp(_UP, upBlend).normalize();
      }
      g.vert(_v.x, _v.y, _v.z, _vb.x, _vb.y, _vb.z,
        uv.u0 + ix * uv.du, uv.v0 + t * uv.dv, f);
    }
  }
  for (let iy = 0; iy < segs; iy++) {
    const a = start + iy * 2;
    g.quad(a, a + 1, a + 3, a + 2);
  }
}

/**
 * A tapered tube along a poly-line — trunks and limbs. `pts` are object-space
 * Vector3, `radii` matches, `flexes` matches (0 keeps the vertex nailed down).
 */
function addTube(g, pts, radii, flexes, radialSegs, uvRepeat) {
  const n = pts.length;
  if (n < 2) return;
  // Parallel transport so the rings do not spin around the trunk.
  const tangent = new THREE.Vector3();
  const prevT = new THREE.Vector3();
  const normal = new THREE.Vector3();
  tangent.copy(pts[1]).sub(pts[0]).normalize();
  normal.set(1, 0, 0);
  if (Math.abs(normal.dot(tangent)) > 0.9) normal.set(0, 0, 1);
  normal.projectOnPlane(tangent).normalize();
  prevT.copy(tangent);

  let vAcc = 0;
  const rings = [];
  for (let i = 0; i < n; i++) {
    if (i > 0) {
      const a = i === n - 1 ? i : i;
      tangent.copy(pts[Math.min(a + 1, n - 1)]).sub(pts[a - 1]).normalize();
      if (tangent.lengthSq() < 1e-8) tangent.copy(prevT);
      _q.setFromUnitVectors(prevT, tangent);
      normal.applyQuaternion(_q).projectOnPlane(tangent).normalize();
      prevT.copy(tangent);
      vAcc += pts[i].distanceTo(pts[i - 1]);
    }
    _vb.copy(tangent).cross(normal).normalize(); // binormal
    const ring = [];
    for (let k = 0; k <= radialSegs; k++) {
      const a = (k / radialSegs) * TAU;
      const ca = Math.cos(a), sa = Math.sin(a);
      _v.copy(normal).multiplyScalar(ca).addScaledVector(_vb, sa);
      const nx = _v.x, ny = _v.y, nz = _v.z;
      ring.push(g.vert(
        pts[i].x + nx * radii[i], pts[i].y + ny * radii[i], pts[i].z + nz * radii[i],
        nx, ny, nz,
        k / radialSegs, vAcc * uvRepeat, flexes[i]));
    }
    rings.push(ring);
  }
  for (let i = 0; i < n - 1; i++) {
    for (let k = 0; k < radialSegs; k++) {
      g.quad(rings[i][k], rings[i][k + 1], rings[i + 1][k + 1], rings[i + 1][k]);
    }
  }
}

const UV_FULL = { u0: 0, v0: 0, du: 1, dv: 1 };

/**
 * Give a geometry a white `color` attribute, so a `vertexColors: true` material
 * can be used on it safely: three's vertex shader does `vColor.rgb *= color`
 * under `USE_COLOR`, and an unbound attribute reads as 0 on WebGL2 — i.e. every
 * vertex black. White makes the multiply a no-op.
 *
 * (r185 also defines `USE_COLOR` in the *fragment* prefix whenever an
 * InstancedMesh has an `instanceColor`, so per-instance tint would survive
 * without `vertexColors`. Pairing the flag with a real white attribute is the
 * version-proof form: it is correct either way, and one vec3 per tuft vertex on
 * a 30-vertex card is not a cost worth the fragility.)
 */
function withInstanceColor(geo) {
  geo.setAttribute('color', new THREE.BufferAttribute(
    new Float32Array(geo.getAttribute('position').count * 3).fill(1), 3));
  return geo;
}

/** Pick an atlas cell. Inset a texel so mip-mapping cannot bleed neighbours. */
function atlasCell(rng, cols, rows) {
  if (cols <= 1 && rows <= 1) return UV_FULL;
  const cx = rng.int(0, cols - 1);
  const cy = rng.int(0, rows - 1);
  const iu = 0.004 / cols;
  const iv = 0.004 / rows;
  return {
    u0: cx / cols + iu,
    v0: cy / rows + iv,
    du: 1 / cols - 2 * iu,
    dv: 1 / rows - 2 * iv,
  };
}

/** Quaternion that rotates +Y onto `dir`, then rolls `roll` about it. */
function orientY(dir, roll, out) {
  _v.copy(dir).normalize();
  out.setFromUnitVectors(_UP, _v);
  if (roll) { _qb.setFromAxisAngle(_v, roll); out.premultiply(_qb); }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Materials — clone what the library hands us, then patch the clone.          */
/* The contract forbids mutating a shared material, and onBeforeCompile is a    */
/* mutation, so a private clone is the only correct move.                       */
/* -------------------------------------------------------------------------- */

let _windMatId = 0;

function baseMaterial(materials, name) {
  try {
    const m = materials && typeof materials.get === 'function' ? materials.get(name) : null;
    if (m && m.isMaterial) return m.clone();
  } catch (err) {
    console.warn(`[vegetation] materials.get('${name}') failed:`, err);
  }
  console.warn(`[vegetation] material '${name}' unavailable — using a flat stand-in.`);
  // Stand-in only. Matched to the raised foliage albedos so a missing material
  // shows up as "flat", not as "black hole".
  return new THREE.MeshStandardMaterial({ color: 0x62804a, roughness: 0.92, metalness: 0 });
}

/**
 * Patch a material (and a matching depth material) with the wind vertex
 * displacement. Returns `{ material, depthMaterial }`.
 *
 * @param {Object} o  amp, freq, phase, dirX, dirZ, bow, fade(bool), shared uniforms
 */
function makeWindMaterials(src, uniforms, o) {
  const amp = o.amp.toFixed(4);
  const freq = o.freq.toFixed(4);
  const bow = (o.bow === undefined ? 0.35 : o.bow).toFixed(3);
  const px = (o.phaseX || 0.31).toFixed(4);
  const pz = (o.phaseZ || 0.24).toFixed(4);
  const dx = (o.dirX === undefined ? 0.82 : o.dirX).toFixed(3);
  const dz = (o.dirZ === undefined ? 0.57 : o.dirZ).toFixed(3);
  const id = `hbwind${_windMatId++}`;

  const decls = /* glsl */`
    #include <common>
    attribute float aFlex;
    uniform float uWindTime;
    uniform float uWindStrength;
    uniform vec3 uPlayerPos;
    uniform vec2 uFadeRange;
  `;

  const fade = o.fade ? /* glsl */`
    float hbFade = 1.0 - smoothstep( uFadeRange.x, uFadeRange.y, distance( hbOrigin, uPlayerPos ) );
  ` : 'float hbFade = 1.0;';

  const body = /* glsl */`
    #include <begin_vertex>
    {
      #ifdef USE_INSTANCING
        vec3 hbOrigin = ( modelMatrix * instanceMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
      #else
        vec3 hbOrigin = ( modelMatrix * vec4( 0.0, 0.0, 0.0, 1.0 ) ).xyz;
      #endif
      ${fade}
      float hbPh = hbOrigin.x * ${px} + hbOrigin.z * ${pz};
      float hbT = uWindTime * ${freq};
      float hbS = sin( hbT + hbPh ) * 0.62 + sin( hbT * 1.71 + hbPh * 2.17 + 1.3 ) * 0.38;
      float hbA = hbS * uWindStrength * ${amp} * aFlex;
      transformed.x += hbA * ${dx};
      transformed.z += hbA * ${dz};
      transformed.y -= abs( hbA ) * ${bow};
      transformed *= hbFade;
    }
  `;

  const patch = (shader) => {
    shader.uniforms.uWindTime = uniforms.uWindTime;
    shader.uniforms.uWindStrength = uniforms.uWindStrength;
    shader.uniforms.uPlayerPos = uniforms.uPlayerPos;
    shader.uniforms.uFadeRange = uniforms.uFadeRange;
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', decls)
      .replace('#include <begin_vertex>', body);
  };

  const material = src;
  material.onBeforeCompile = patch;
  material.customProgramCacheKey = () => id;

  const depthMaterial = new THREE.MeshDepthMaterial();
  // Matching alpha state, or the shadow silhouette will be a solid rectangle.
  depthMaterial.map = material.map || null;
  depthMaterial.alphaMap = material.alphaMap || null;
  depthMaterial.alphaTest = material.alphaTest || 0;
  depthMaterial.side = material.side;
  depthMaterial.onBeforeCompile = patch;
  depthMaterial.customProgramCacheKey = () => `${id}d`;

  return { material, depthMaterial };
}

/**
 * Strip map slots this module's geometry cannot honour. `aoMap`/`lightMap`
 * need a uv1 set the cards do not have, and `vertexColors` without a `color`
 * attribute makes every vertex black on WebGL2 (unbound attributes read 0).
 */
function sanitize(m) {
  m.vertexColors = false;
  m.aoMap = null;
  m.lightMap = null;
  m.displacementMap = null;
  return m;
}

/** Configure a cloned material as an alpha-tested cut-out. */
function asCutout(m, alphaTest = 0.42) {
  sanitize(m);
  m.transparent = false;
  m.depthWrite = true;
  m.alphaTest = alphaTest;
  m.side = THREE.DoubleSide;
  m.shadowSide = THREE.DoubleSide;
  if (!m.map && !m.alphaMap) {
    // No cut-out texture available — opaque cards would read as flat plates,
    // so drop the alpha test and let the flat colour carry it.
    m.alphaTest = 0;
    console.warn('[vegetation] foliage material has no map/alphaMap; cards will be opaque.');
  }
  return m;
}

/**
 * How the atlas is subdivided.
 *
 * The material library publishes the real grid on `material.userData.atlas`
 * ({cols, rows}); it does NOT copy it onto the texture, so the old
 * texture-userData probe never found anything and every atlas fell back to 2x2.
 * That is wrong for three of the five: `leaf`, `flower` and `grassBlade` are all
 * 4x2, so a card was sampling two columns at once — two half leaves instead of
 * one leaf, and for the flowers a slice through two blossoms. Alpha coverage is
 * per unit area so the ink (and therefore the meadow's measured luminance) is
 * unchanged; the cards are simply the shape the bakery drew.
 */
function atlasGrid(m) {
  const a = m && m.userData && m.userData.atlas;
  if (a && a.cols > 0 && a.rows > 0) return { cols: a.cols | 0, rows: a.rows | 0 };
  const t = m && (m.map || m.alphaMap);
  const ud = t && t.userData;
  const cols = ud && (ud.atlasCols || ud.cols);
  const rows = ud && (ud.atlasRows || ud.rows);
  return { cols: cols || 2, rows: rows || 2 };
}

/* -------------------------------------------------------------------------- */
/* Species geometry                                                            */
/* -------------------------------------------------------------------------- */

/**
 * Tall dark spruce: a tapered trunk plus stacked drooping whorls. Returns
 * `{ trunk, foliage, height, radius, trunkRadius }` — two geometries because a
 * single InstancedMesh cannot carry two materials.
 *
 * `o.mid` builds the MID LOD of the very same tree. Two rules make that work:
 *
 *   1. every rng call happens in the same order and the same number of times as
 *      the full build (the card randoms are hoisted out of the addCard literal
 *      for exactly this reason), so a mid variant forked from the same seed has
 *      an identical skeleton — same height, lean, whorl heights, branch angles;
 *   2. only the EMISSION is skipped. The mid tree is therefore a strict subset
 *      of the full tree's cards, widened to hold the silhouette, which is why
 *      the 60 m crossover is invisible.
 *
 * The mid tier also drops the trunk: a spruce stem is behind its own skirt, and
 * at 60 m+ it is a 4 px sliver that would otherwise cost a second InstancedMesh
 * and a bark draw call per variant.
 */
function buildConifer(rng, detail, grid, o = {}) {
  const mid = !!o.mid;
  const trunkG = mid ? null : new Geo();
  const foliG = new Geo();

  const H = rng.range(11, 16.5);
  const trunkR = H * rng.range(0.028, 0.038);
  const lean = rng.range(-0.035, 0.035);
  const leanDir = rng.range(0, TAU);

  const segs = Math.max(5, Math.round(9 * detail));
  const pts = [], radii = [], flexes = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const y = t * H;
    const off = lean * H * t * t;
    pts.push(new THREE.Vector3(Math.cos(leanDir) * off, y, Math.sin(leanDir) * off));
    radii.push(trunkR * (1 - 0.88 * Math.pow(t, 0.78)) + 0.012);
    flexes.push(Math.pow(t, 1.8) * 0.55);
  }
  if (!mid) addTube(trunkG, pts, radii, flexes, Math.max(5, Math.round(7 * detail)), 0.45);

  // Whorls are packed tight and overlap by design: a spruce is a solid dark
  // mass from 100 m, not a pole with tufts on it.
  const whorls = Math.round(THREE.MathUtils.lerp(15, 21, rng.next()) * Math.max(0.65, detail));
  const maxR = H * rng.range(0.30, 0.38);
  const cardSegs = mid ? 1 : (detail >= 0.9 ? 2 : 1);
  const cardsPerBranch = detail >= 0.8 ? 3 : 2;
  // Fewer cards have to cover the same crown, so the survivors grow.
  const cardGain = mid ? 1.36 : 1;
  let maxSpread = 0;

  for (let i = 0; i < whorls; i++) {
    const t = 0.05 + 0.93 * (i / Math.max(1, whorls - 1));
    const y = t * H;
    const off = lean * H * t * t;
    const cx = Math.cos(leanDir) * off;
    const cz = Math.sin(leanDir) * off;
    // Classic spruce profile: widest at ~20% height, tapering to a spire. The
    // floor keeps the upper whorls broad enough to close the silhouette.
    const prof = Math.pow(1 - t, 0.62) * (0.42 + 0.58 * THREE.MathUtils.smoothstep(t, 0.0, 0.20));
    // Alternate whorls breathe in and out so the edge is ragged, not conical.
    const jitter = 1 + (i % 2 === 0 ? 0.09 : -0.07);
    const R = maxR * prof * jitter + 0.35;
    maxSpread = Math.max(maxSpread, R);
    const branches = Math.max(6, Math.round(rng.int(9, 12) * Math.max(0.7, detail)));
    const trunkFlex = Math.pow(t, 1.8) * 0.55;
    for (let b = 0; b < branches; b++) {
      const a = i * GOLDEN + (b / branches) * TAU + rng.sym(0.14);
      // Droop deepens toward the bottom of the tree.
      const pitch = THREE.MathUtils.lerp(-0.58, -0.16, t) + rng.sym(0.09);
      const len = R * rng.range(0.85, 1.15);
      // Some branches start a little above the whorl plane, which is what makes
      // neighbouring whorls read as one continuous skirt.
      const yOff = rng.sym(H / whorls * 0.55);
      for (let c = 0; c < cardsPerBranch; c++) {
        // Hoisted in the original evaluation order (width, bend, uv) so the mid
        // pass consumes an identical rng stream even when it emits nothing.
        const cw = len * rng.range(0.72, 0.98);
        const cbend = -len * rng.range(0.20, 0.34);
        const cuv = atlasCell(rng, grid.cols, grid.rows);
        // Mid: card 0 on every branch, card 1 on every other branch — 1.5 cards
        // where the full tree has 3, at one segment instead of two.
        if (mid && !(c === 0 || (c === 1 && (b & 1) === 0))) continue;
        // Re-seed the direction each pass: addCard() reuses the same scratch.
        _vb.set(Math.cos(a) * Math.cos(pitch), Math.sin(pitch), Math.sin(a) * Math.cos(pitch));
        orientY(_vb, c * 0.95, _q);
        _scl.set(1, 1, 1);
        _v.set(cx + Math.cos(a) * radii[Math.min(segs, Math.round(t * segs))] * 0.8,
          y + yOff, cz + Math.sin(a) * radii[Math.min(segs, Math.round(t * segs))] * 0.8);
        _m4.compose(_v, _q, _scl);
        addCard(foliG, _m4, {
          width: cw * cardGain,
          height: len,
          segs: cardSegs,
          bend: cbend,
          taper: 0.28,
          uv: cuv,
          flexBase: trunkFlex,
          flexTip: trunkFlex + 0.45,
          flexPow: 1.5,
          upBlend: 0.32,
        });
      }
    }
  }

  // Spire — a few near-vertical cards so the top is not a bald stick. Kept in
  // full at every tier: 8 triangles, and it is the tip of the silhouette.
  for (let c = 0; c < 4; c++) {
    _vb.set(rng.sym(0.25), 1, rng.sym(0.25));
    orientY(_vb, c * 1.1, _q);
    _v.set(Math.cos(leanDir) * lean * H, H * (0.90 + c * 0.015), Math.sin(leanDir) * lean * H);
    _m4.compose(_v, _q, _scl.set(1, 1, 1));
    addCard(foliG, _m4, {
      width: H * 0.085 * cardGain, height: H * 0.13, segs: 1, bend: 0, taper: 0.7,
      uv: atlasCell(rng, grid.cols, grid.rows),
      flexBase: 0.55, flexTip: 1.0, upBlend: 0.3,
    });
  }

  return {
    trunk: mid ? null : trunkG.build('conifer-trunk'),
    foliage: foliG.build(mid ? 'conifer-foliage-mid' : 'conifer-foliage'),
    height: H,
    radius: maxSpread,
    trunkRadius: trunkR,
  };
}

/**
 * Gnarled broadleaf: forking limbs with clustered leaf cards. `o.mid` builds the
 * mid LOD under the same two rules as `buildConifer` — identical rng stream,
 * emission-only decimation. Unlike the conifer this one KEEPS its trunk mesh:
 * the bare forking limbs are most of a broadleaf's silhouette. They are just
 * coarser tubes (3-sided instead of 8, 4 trunk sections instead of 8) and the
 * fork twigs are dropped while the leaf clusters they carried stay.
 */
function buildBroadleaf(rng, detail, grid, o = {}) {
  const mid = !!o.mid;
  const trunkG = new Geo();
  const foliG = new Geo();

  const H = rng.range(8.5, 13.5);
  const trunkR = H * rng.range(0.045, 0.062);
  const crotch = H * rng.range(0.32, 0.44);
  const gnarl = valueNoise2D(rng.int(1, 1e6));

  const segs = mid ? 4 : Math.max(5, Math.round(8 * detail));
  const pts = [], radii = [], flexes = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs;
    const y = t * crotch;
    const bx = fbm2D(gnarl, t * 2.6, 0.5, 2) * crotch * 0.16;
    const bz = fbm2D(gnarl, 4.1, t * 2.6, 2) * crotch * 0.16;
    pts.push(new THREE.Vector3(bx, y, bz));
    radii.push(trunkR * (1 - 0.42 * t) * (1 + (i === 0 ? 0.35 : 0)));
    flexes.push(Math.pow(y / H, 1.9) * 0.5);
  }
  const radial = mid ? 3 : Math.max(5, Math.round(8 * detail));
  addTube(trunkG, pts, radii, flexes, radial, 0.5);

  const limbCount = rng.int(4, 6);
  const clusterSize = H * rng.range(0.13, 0.18);
  const cardSegs = mid ? 1 : (detail >= 0.9 ? 2 : 1);
  // Three cards instead of six per cluster, modestly enlarged. The gain is held
  // down deliberately: measured against the full build, 2 cards at 1.70x grew the
  // crown's bounding radius from 6.37 m to 7.40 m, and a 16 % size step at the
  // 60 m crossover is exactly the pop this LOD exists to remove. At 3 x 1.36 the
  // crown lands within ~4 %.
  const cardKeep = mid ? 3 : 99;
  const cardGainW = mid ? 1.36 : 1;
  const cardGainH = mid ? 1.16 : 1;
  let maxSpread = 0;

  const addCluster = (px, py, pz, scale, flexBase) => {
    const n = detail >= 0.9 ? 6 : 4;
    for (let c = 0; c < n; c++) {
      // Same hoisting rule as the conifer: consume the rng, then decide.
      _vb.set(rng.sym(0.9), rng.range(0.25, 1.0), rng.sym(0.9));
      const roll = rng.range(0, TAU);
      const cw = clusterSize * scale * rng.range(1.0, 1.45);
      const ch = clusterSize * scale * rng.range(0.9, 1.3);
      const cuv = atlasCell(rng, grid.cols, grid.rows);
      if (c >= cardKeep) continue;
      orientY(_vb, roll, _q);
      _v.set(px, py, pz);
      _m4.compose(_v, _q, _scl.set(1, 1, 1));
      addCard(foliG, _m4, {
        width: cw * cardGainW,
        height: ch * cardGainH,
        segs: cardSegs,
        bend: -clusterSize * scale * 0.2,
        taper: 0.12,
        uv: cuv,
        flexBase,
        flexTip: flexBase + 0.5,
        flexPow: 1.35,
        upBlend: 0.42,
      });
    }
  };

  for (let l = 0; l < limbCount; l++) {
    const a = (l / limbCount) * TAU + rng.sym(0.4);
    const rise = rng.range(0.55, 1.0);
    const len = H * rng.range(0.34, 0.5);
    const start = pts[pts.length - 1];
    const lp = [], lr = [], lf = [];
    const lsegs = 4;
    for (let i = 0; i <= lsegs; i++) {
      const t = i / lsegs;
      // Limbs sweep out then curl back up — that reads as "gnarled".
      const horiz = Math.sin(t * 1.15) * len;
      const vert = (rise * t + 0.35 * t * t) * len;
      lp.push(new THREE.Vector3(
        start.x + Math.cos(a) * horiz + rng.sym(0.09) * len,
        start.y + vert,
        start.z + Math.sin(a) * horiz + rng.sym(0.09) * len));
      lr.push(trunkR * 0.55 * (1 - 0.8 * t) + 0.018);
      lf.push(Math.pow(Math.min(1, (start.y + vert) / H), 1.6) * 0.5 + t * 0.22);
    }
    addTube(trunkG, lp, lr, lf, Math.max(mid ? 3 : 4, radial - 2), 0.6);
    maxSpread = Math.max(maxSpread, Math.hypot(lp[lsegs].x, lp[lsegs].z) + clusterSize);

    const tipFlex = lf[lsegs];
    addCluster(lp[lsegs].x, lp[lsegs].y, lp[lsegs].z, 1.2, tipFlex);
    addCluster(lp[3].x, lp[3].y, lp[3].z, 1.0, lf[3]);
    if (detail >= 0.8) {
      addCluster(lp[2].x, lp[2].y, lp[2].z, 0.85, lf[2]);
      // Fill the shoulder of the crown so the canopy is a mass, not a ring.
      addCluster(
        (lp[3].x + lp[lsegs].x) * 0.5 + rng.sym(0.5),
        (lp[3].y + lp[lsegs].y) * 0.5 + rng.range(0.1, 0.8),
        (lp[3].z + lp[lsegs].z) * 0.5 + rng.sym(0.5), 1.05, tipFlex * 0.9);
    }

    // A short fork off each limb, so the silhouette is not a starfish.
    if (rng.bool(0.8)) {
      const fa = a + rng.sym(0.9);
      const fp = [], fr = [], ff = [];
      const from = lp[3];
      for (let i = 0; i <= 3; i++) {
        const t = i / 3;
        fp.push(new THREE.Vector3(
          from.x + Math.cos(fa) * t * len * 0.5,
          from.y + t * len * 0.5,
          from.z + Math.sin(fa) * t * len * 0.5));
        fr.push(trunkR * 0.3 * (1 - 0.85 * t) + 0.012);
        ff.push(lf[3] + t * 0.28);
      }
      // The fork's own tube is a 1 px thread past 60 m — the mid tier keeps the
      // leaf cluster that hangs off it and drops the twig.
      if (!mid) addTube(trunkG, fp, fr, ff, 4, 0.7);
      addCluster(fp[3].x, fp[3].y, fp[3].z, 1.0, ff[3]);
    }
  }

  return {
    trunk: trunkG.build(mid ? 'broadleaf-trunk-mid' : 'broadleaf-trunk'),
    foliage: foliG.build(mid ? 'broadleaf-foliage-mid' : 'broadleaf-foliage'),
    height: H,
    radius: Math.max(maxSpread, H * 0.35),
    trunkRadius: trunkR,
  };
}

/**
 * Crossed-card LOD blob. No impostor bake is possible without a renderer, so
 * this is a stack of alpha-tested cards sized to the tree. It now only appears
 * beyond `TREE_MID_M` (180 m), which is what lets it be built for coverage.
 *
 * Card COUNT and OVERLAP are the whole design problem. Measured off the live
 * bakery, an atlas cell of the needle sheet is only ~10 % opaque above the 0.42
 * alpha test, so cards crossed is what decides whether the tree reads as a mass
 * or as a spike, and for a set of planes that all pass through the trunk axis
 * that number is (total cards) x (bandHeight / totalSpan) — the plane/band split
 * does not change it, only the smoothness of the outline does. The old build was
 * 16 cards at 0.40/1.08 ≈ 5.9 crossed ≈ 46 % coverage, which at 80 m still read
 * as a bottle brush. This one is 18 cards at 0.50/1.07 ≈ 8.4 crossed ≈ 59 %,
 * spread over six bands instead of four so the conical outline is stepped in
 * 11 % increments rather than 24 % ones, for +4 triangles.
 *
 * Vertical proportion is now matched to the geometry it replaces: the full
 * conifer's spire cards top out at 1.07 H, so the billboard spans 0..1.07 too
 * (it used to be an accident that they agreed) and the far matrix widens the
 * conifer to 0.66 H across, which is the real crown (maxR ≈ 0.34 H) rather than
 * the 0.58 H the old cards drew.
 */
function buildTreeBillboard(kind, grid, rng, detail = 1) {
  const g = new Geo();
  const hi = detail >= 0.75;
  const planes = kind === 'conifer' ? 3 : (hi ? 4 : 3);
  // y, width, taper, flexBase, flexTip — bottom band first, overlapping upward.
  const bands = kind === 'conifer'
    ? (hi
      ? [[0.000, 0.80, 0.12, 0.03, 0.18], [0.114, 0.76, 0.18, 0.07, 0.28],
        [0.228, 0.66, 0.26, 0.12, 0.40], [0.342, 0.55, 0.38, 0.20, 0.52],
        [0.456, 0.41, 0.55, 0.28, 0.66], [0.570, 0.27, 0.85, 0.36, 0.80]]
      : [[0.00, 0.78, 0.16, 0.05, 0.22], [0.20, 0.68, 0.26, 0.10, 0.40],
        [0.40, 0.50, 0.45, 0.20, 0.55], [0.57, 0.28, 0.85, 0.32, 0.75]])
    : (hi
      ? [[0.14, 0.98, 0.04, 0.16, 0.42], [0.30, 0.96, 0.14, 0.24, 0.54],
        [0.46, 0.84, 0.30, 0.32, 0.66], [0.58, 0.60, 0.52, 0.40, 0.78]]
      : [[0.16, 0.94, 0.06, 0.18, 0.48], [0.38, 0.86, 0.24, 0.28, 0.62],
        [0.58, 0.58, 0.50, 0.38, 0.75]]);
  const bandH = kind === 'conifer' ? (hi ? 0.50 : 0.48) : 0.52;

  for (let i = 0; i < planes; i++) {
    const a = (i / planes) * Math.PI;
    _q.setFromAxisAngle(_UP, a);
    for (let b = 0; b < bands.length; b++) {
      const [y, w, taper, f0, f1] = bands[b];
      // Alternate planes step sideways a little: cards that all share the trunk
      // axis pile their transparent margins on top of each other, which is what
      // leaves a hole down the middle of the silhouette.
      const off = ((i + b) % 2 === 0 ? 1 : -1) * w * 0.12;
      _m4.compose(_v.set(0, 0, 0), _q, _scl.set(1, 1, 1));
      _m4b.copy(_m4).setPosition(Math.cos(a) * off, y, Math.sin(a) * off);
      addCard(g, _m4b, {
        width: w, height: bandH, segs: 1, bend: 0, taper,
        uv: atlasCell(rng, grid.cols, grid.rows),
        flexBase: f0, flexTip: f1, upBlend: 0.5,
      });
    }
  }
  return g.build(`${kind}-billboard`);
}

/** Rounded shrub: leaf cards shelled over a hemisphere, normals pushed radial. */
function buildBush(rng, grid, cards = 12) {
  const g = new Geo();
  const centre = new THREE.Vector3(0, 0.45, 0);
  for (let i = 0; i < cards; i++) {
    const a = i * GOLDEN;
    const t = (i + 0.5) / cards;
    const el = Math.acos(1 - t * 1.35);          // hemisphere-biased
    _vb.set(Math.cos(a) * Math.sin(el), Math.cos(el) * 0.9 + 0.28, Math.sin(a) * Math.sin(el));
    orientY(_vb, rng.range(0, TAU), _q);
    _v.copy(_vb).multiplyScalar(0.22).add(centre);
    _m4.compose(_v, _q, _scl.set(1, 1, 1));
    addCard(g, _m4, {
      width: rng.range(0.62, 0.9),
      height: rng.range(0.55, 0.82),
      segs: 1,
      bend: -0.1,
      taper: 0.2,
      uv: atlasCell(rng, grid.cols, grid.rows),
      flexBase: 0.15,
      flexTip: 0.85,
      upBlend: 0,
      radial: centre,
    });
  }
  return g.build('bush');
}

/** A metre of rough hedgerow. Instanced end to end along field boundaries. */
function buildHedgeSegment(rng, grid) {
  const g = new Geo();
  for (let i = 0; i < 9; i++) {
    const a = rng.range(0, TAU);
    const side = (i / 9) * 2 - 1;
    _vb.set(Math.cos(a) * 0.8, rng.range(0.5, 1.0), Math.sin(a) * 0.8);
    orientY(_vb, rng.range(0, TAU), _q);
    _v.set(side * 0.85, rng.range(0.1, 0.5), rng.sym(0.18));
    _m4.compose(_v, _q, _scl.set(1, 1, 1));
    addCard(g, _m4, {
      width: rng.range(0.85, 1.25), height: rng.range(0.8, 1.15), segs: 1,
      bend: -0.12, taper: 0.15,
      uv: atlasCell(rng, grid.cols, grid.rows),
      flexBase: 0.1, flexTip: 0.7, upBlend: 0.25,
    });
  }
  return g.build('hedge');
}

/**
 * A grass tuft: a few tapered blades in a fan. Meadow scale — a tuft is
 * 15-25 cm tall before the per-instance scale, so it reads as turf rather than
 * as a spiky ornamental. Blades splay hard, which is what makes it look scruffy
 * instead of like a paintbrush.
 */
function buildGrassTuft(rng, grid, blades = 5, o = {}) {
  const g = new Geo();
  const minH = o.minH === undefined ? 0.14 : o.minH;
  const maxH = o.maxH === undefined ? 0.24 : o.maxH;
  for (let i = 0; i < blades; i++) {
    const a = i * GOLDEN + rng.sym(0.5);
    const tilt = rng.range(0.10, 0.62);
    _vb.set(Math.cos(a) * Math.sin(tilt), Math.cos(tilt), Math.sin(a) * Math.sin(tilt));
    orientY(_vb, rng.range(0, TAU), _q);
    _v.set(Math.cos(a) * rng.range(0, 0.055), 0, Math.sin(a) * rng.range(0, 0.055));
    _m4.compose(_v, _q, _scl.set(1, 1, 1));
    const h = rng.range(minH, maxH);
    addCard(g, _m4, {
      width: rng.range(0.05, 0.085),
      height: h,
      segs: o.segs || 2,
      bend: -h * rng.range(0.16, 0.42),
      taper: 0.86,
      uv: atlasCell(rng, grid.cols, grid.rows),
      flexBase: 0.05,
      flexTip: 1.0,
      flexPow: 1.9,
      upBlend: 0.72,
    });
  }
  // Instance-tinted: the meadow's tone variation lives on instanceColor.
  return withInstanceColor(g.build('grass-tuft'));
}

/**
 * A wildflower: two crossed blossom cards cut out of the `flower` atlas.
 *
 * This used to be a shaded petal fan on a two-card stem cross, with the atlas
 * deliberately thrown away — and that is the bug the review caught. The reasoning
 * behind it ("the flower atlas layout is owned by another stream") was simply
 * wrong: the bakery publishes the layout (4x2, row 0 a single blossom, row 1 a
 * three-head cluster, petals near-white so `material.color` paints them) and the
 * library hands back `flowerRed` already set up as a cut-out. Nulling its `map`
 * and forcing `alphaTest` to 0 left the ONE foliage mesh in this file that
 * ignored its alpha, and it also left the flower atlas's normal map driving a
 * petal fan whose UVs are a radial disc — i.e. wrong normals on top of a hard
 * silhouette.
 *
 * The stems are gone with the fan, and that is a fix rather than a loss: with
 * `material.color` forced to white the stems were being painted with the
 * per-instance BLOSSOM tint, so the meadow had red and pink stalks in it. The
 * atlas has no stem pixels to replace them with, and a 12 mm stalk is sub-pixel
 * past about 4 m, so the blossom heads now sit in the grass on their own.
 *
 * 4 triangles per flower instead of 10.
 */
function buildFlower(rng, grid) {
  const g = new Geo();
  // Cell heads are ~55 % of the cell, so a 0.13 m card draws a ~7 cm blossom.
  const size = rng.range(0.105, 0.155);
  const lift = rng.range(0.075, 0.185);
  for (let c = 0; c < 2; c++) {
    const uv = atlasCell(rng, grid.cols, grid.rows);
    _q.setFromAxisAngle(_UP, c * Math.PI * 0.5 + rng.sym(0.3));
    _m4.compose(_v.set(rng.sym(0.012), lift, rng.sym(0.012)), _q, _scl.set(1, 1, 1));
    addCard(g, _m4, {
      // taper 0: the blossom is a disc in the middle of a square cell, so the
      // card has to stay square or the head comes out as an egg.
      width: size, height: size * rng.range(0.94, 1.08), segs: 1, bend: 0, taper: 0,
      uv,
      // Nothing anchors a floating head, so the whole card sways as one.
      flexBase: 0.6, flexTip: 1.0, flexPow: 1.2, upBlend: 0.55,
    });
  }
  return withInstanceColor(g.build('flower'));
}

/** A fallen leaf lying flat under a canopy. */
function buildLeafLitter(rng, grid) {
  const g = new Geo();
  _q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2 + rng.sym(0.25));
  _m4.compose(_v.set(0, 0.012, 0), _q, _scl.set(1, 1, 1));
  addCard(g, _m4, {
    width: rng.range(0.1, 0.2), height: rng.range(0.1, 0.2), segs: 1, bend: 0.02, taper: 0.1,
    uv: atlasCell(rng, grid.cols, grid.rows), flexBase: 0, flexTip: 0.12, upBlend: 0.9,
  });
  return g.build('litter');
}

/* -------------------------------------------------------------------------- */
/* Scatter helpers                                                             */
/* -------------------------------------------------------------------------- */

/**
 * The well's dressed-stone apron: the two circular base steps, radii 4.15 and
 * 3.35 centred on the well. Nothing grows on a swept step.
 *
 * `isClearGround` already rejects `WELL.base[0].radius + margin`, and every
 * scatter here also rejects the whole plaza (the well stands well inside it), so
 * this is a third, explicit gate rather than the only one — it is 2 subtractions
 * and a compare, it does not depend on a caller passing a margin, and it does not
 * depend on the terrain classifying the apron as anything in particular. The
 * apron reads as 'cobble' from `terrain.surfaceAt` (it is inside the plaza
 * polygon) and as 'stone' to the eye, and neither is soil.
 */
const WELL_APRON_R = 4.15 + 0.35;
function onWellApron(x, z) {
  const dx = x - WELL.position[0];
  const dz = z - WELL.position[2];
  return dx * dx + dz * dz < WELL_APRON_R * WELL_APRON_R;
}

/**
 * Everything a scatter must avoid: buildings, plaza, roads, the well.
 * `roadClear` is the margin OUTSIDE the road's own carriageway, so a 7.5 m
 * road keeps plants off the setts without the caller re-deriving its width.
 */
function plantable(terra, x, z, margin, roadClear) {
  if (insidePlaza(x, z)) return false;
  if (onWellApron(x, z)) return false;
  if (!isClearGround(x, z, margin)) return false;
  if (nearRoad(x, z, roadClear)) return false;
  // Surface last: it is the most expensive test and the others reject most
  // candidates first.
  return isSoil(terra, x, z);
}

/**
 * The hard rule from the review: things only grow out of soil. Paving setts and
 * bare rock stay clear, whatever the geometric tests say.
 */
function isSoil(terra, x, z) {
  const s = terra.surfaceAt(x, z);
  return s === 'grass' || s === 'dirt';
}

/* -------------------------------------------------------------------------- */
/* Grass carpet — a lattice that recycles instances as the player walks         */
/* -------------------------------------------------------------------------- */

class GrassCarpet {
  /**
   * @param {THREE.InstancedMesh} mesh
   * @param {number} n     lattice side, in cells
   * @param {number} cell  cell size in metres
   */
  constructor(mesh, n, cell, terra, seed, opts = {}) {
    this.mesh = mesh;
    this.n = n;
    this.cell = cell;
    this.terra = terra;
    this.seed = seed >>> 0;
    this.minScale = opts.minScale === undefined ? 0.7 : opts.minScale;
    this.maxScale = opts.maxScale === undefined ? 1.3 : opts.maxScale;
    this.margin = opts.margin === undefined ? 1.6 : opts.margin;
    this.roadClear = opts.roadClear === undefined ? 0.4 : opts.roadClear;
    this.maxSlope = opts.maxSlope === undefined ? 0.72 : opts.maxSlope;
    /**
     * `tint(x, z, jitter, outColor)` fills `outColor` with the per-instance
     * multiplier and returns the meadow field at that point (see
     * `meadowFieldAt`) so `_place` can also use it to thin the carpet. Optional:
     * without it the carpet behaves exactly as before.
     */
    this.tint = opts.tint || null;
    /** Added to the local density before the draw — the carpet underfoot keeps
     *  more of its tufts than the one 40 m out, where thinning is free. */
    this.densityBias = opts.densityBias || 0;
    if (this.tint) {
      mesh.instanceColor = new THREE.InstancedBufferAttribute(
        new Float32Array(mesh.instanceMatrix.count * 3).fill(1), 3);
      mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    }
    // Cached cell coordinate per slot, so a step only rewrites what changed.
    this.cx = new Int32Array(n * n).fill(0x7fffffff);
    this.cz = new Int32Array(n * n).fill(0x7fffffff);
    this.s = 0x7fffffff;
    this.t = 0x7fffffff;
    this.live = 0;
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    mesh.boundingSphere = new THREE.Sphere(new THREE.Vector3(), n * cell * 0.75);
    mesh.frustumCulled = true;
  }

  /** Deterministic per-cell hash — same cell always grows the same tuft. */
  _hash(cx, cz) {
    let h = (cx * 374761393 + cz * 668265263 + this.seed) | 0;
    h = Math.imul(h ^ (h >>> 13), 1274126177);
    return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
  }

  /** @returns {boolean} true if any instance moved. */
  refresh(px, pz, force) {
    const half = this.n >> 1;
    const s = Math.round(px / this.cell) - half;
    const t = Math.round(pz / this.cell) - half;
    if (!force && s === this.s && t === this.t) return false;
    this.s = s; this.t = t;
    const n = this.n;
    let dirty = false;
    for (let iz = 0; iz < n; iz++) {
      const cz = t + (((iz - t) % n) + n) % n;
      for (let ix = 0; ix < n; ix++) {
        const slot = iz * n + ix;
        const cx = s + (((ix - s) % n) + n) % n;
        if (this.cx[slot] === cx && this.cz[slot] === cz) continue;
        this.cx[slot] = cx;
        this.cz[slot] = cz;
        dirty = true;
        this._place(slot, cx, cz);
      }
    }
    if (dirty) {
      this.mesh.instanceMatrix.needsUpdate = true;
      if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
      this.mesh.boundingSphere.center.set(
        (s + half) * this.cell, 0, (t + half) * this.cell);
    }
    return dirty;
  }

  _place(slot, cx, cz) {
    const r1 = this._hash(cx, cz);
    const r2 = this._hash(cx + 7919, cz - 104729);
    const r3 = this._hash(cx - 31, cz + 5077);
    const r4 = this._hash(cx + 1013, cz + 2027);   // tone jitter
    const r5 = this._hash(cx - 9721, cz - 613);    // density draw
    const x = (cx + (r1 - 0.5) * 0.92) * this.cell;
    const z = (cz + (r2 - 0.5) * 0.92) * this.cell;

    // `plantable` includes the soil test, so the carpet can never creep onto
    // the setts or the well's apron however close the player stands.
    let ok = plantable(this.terra, x, z, this.margin, this.roadClear);
    if (ok && this.terra.slopeAt(x, z) > this.maxSlope) ok = false;

    // Local density. An even lattice is what made the meadow read as one flat
    // mass; this opens scuffed bare patches and leaves the hollows thick.
    let patch = 0.5;
    if (ok && this.tint) {
      const f = this.tint(x, z, r4, _col);
      patch = f.patch;
      const dens = THREE.MathUtils.clamp(
        0.38 + this.densityBias + 0.95 * patch - 0.22 * f.dry + 0.20 * f.shade, 0.05, 1);
      if (r5 > dens) ok = false;
      else this.mesh.setColorAt(slot, _col);
    }

    if (!ok) {
      _m4.makeScale(0, 0, 0);
      this.mesh.setMatrixAt(slot, _m4);
      return;
    }
    const y = this.terra.heightAt(x, z);
    // Cubed lerp: most tufts sit near the low end, a few stand proud. A linear
    // ramp made every tuft the same height, which is what read as "neon spikes".
    // The patch factor then makes the thick hollows visibly fuller than the
    // thin ground rather than just more numerous.
    const sc = THREE.MathUtils.lerp(this.minScale, this.maxScale, r3 * r3 * r3)
      * (0.86 + 0.42 * patch);
    _q.setFromAxisAngle(_UP, r1 * TAU);
    _m4.compose(_v.set(x, y - 0.03, z), _q, _scl.set(sc, sc * (0.72 + r2 * 0.5), sc));
    this.mesh.setMatrixAt(slot, _m4);
  }
}

/* -------------------------------------------------------------------------- */
/* Factory                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Build every plant in the valley.
 *
 * @param {Object} o
 * @param {import('../contracts.js').MaterialLibrary} o.materials
 * @param {Object} o.terrain  height field provider (heightAt / slopeAt)
 * @param {Object} o.quality  active quality preset
 * @returns {import('../contracts.js').WorldChunk}
 */
export function createVegetation({ materials, terrain, quality }) {
  const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const q = quality || {};
  const density = q.vegetationDensity === undefined ? 1 : q.vegetationDensity;
  const detail = q.geometryDetail === undefined ? 1 : q.geometryDetail;
  const terra = makeTerrainAdapter(terrain);

  const group = new THREE.Group();
  group.name = 'vegetation';
  const colliders = [];
  const disposables = [];
  const meshes = [];

  const uniforms = {
    uWindTime: { value: 0 },
    uWindStrength: { value: 1 },
    uPlayerPos: { value: new THREE.Vector3() },
    uFadeRange: { value: new THREE.Vector2(q.grassDistance || 40, (q.grassDistance || 40) * 1.16) },
  };

  /* ---------------------------------------------------------- materials --- */

  const mkFoliage = (name, wind) => {
    const m = asCutout(baseMaterial(materials, name));
    const w = makeWindMaterials(m, uniforms, wind);
    disposables.push(w.material, w.depthMaterial);
    return w;
  };
  const mkSolid = (name, wind, cutout) => {
    const m = baseMaterial(materials, name);
    if (cutout) asCutout(m); else { sanitize(m); m.side = THREE.FrontSide; m.transparent = false; }
    const w = makeWindMaterials(m, uniforms, wind);
    disposables.push(w.material, w.depthMaterial);
    return w;
  };

  const WIND_TREE = { amp: 0.17, freq: 0.62, bow: 0.35 };
  const WIND_BUSH = { amp: 0.055, freq: 0.95, bow: 0.3 };
  const WIND_GRASS = { amp: 0.055, freq: 1.85, bow: 0.5, fade: true };
  const WIND_STILL = { amp: 0.0, freq: 0.4, bow: 0 };

  const needleMat = mkFoliage('needle', WIND_TREE);
  const leafMat = mkFoliage('leaf', WIND_TREE);
  const leafDarkMat = mkFoliage('leafDark', WIND_BUSH);
  const barkMat = mkSolid('bark', WIND_TREE, false);
  const grassMat = mkFoliage('grassBlade', WIND_GRASS);
  const hedgeMat = mkFoliage('leafDark', WIND_BUSH);
  const litterMat = mkFoliage('leafDark', WIND_STILL);

  /*
   * NO tint multipliers on the foliage clones any more, and that is a deliberate
   * reversal. The old `needle` clone was multiplied by 0.62 on the theory that
   * the atlas was authored bright for mid-distance broadleaf. Measured off the
   * live bakery this pass, it is not:
   *
   *     needle    sRGB(34, 53, 32)   0.032 linear
   *     leaf      sRGB(52, 84, 30)   0.075 linear
   *     leafDark  same map, and the LIBRARY already tints it 0x9fbe8c
   *                                  ~0.05 linear effective
   *     bark      sRGB(69, 57, 42)   0.049 linear
   *
   * A real spruce canopy is 0.06-0.09 linear, so 0.032 is already darker than
   * the reference and 0.62 on top of it was making a black cut-out. The same
   * argument applies to the bush / hedge / litter clones of `leafDark`, which the
   * material library has already knocked down for us. Anything a stream needs
   * darker than that belongs in the bakery, not in a private clone here.
   */

  /*
   * Wildflowers. A cut-out card like every other foliage mesh in this file —
   * `asCutout` gives it the same alphaTest 0.42 / transparent false / DoubleSide
   * state, which is the whole of defect V2: this variant used to null the map and
   * set alphaTest 0, so the blossom alpha was ignored and each card rendered
   * solid. The atlas is kept; only `color` is neutralised, because the four
   * blossom tints below ARE the colours and leaving flowerRed's own red in
   * `color` multiplied every instance by it — the "white" blossoms came out pink
   * and the reds were red-squared.
   */
  const flowerMat = (() => {
    const m = asCutout(baseMaterial(materials, 'flowerRed'));
    if (m.color) m.color.setRGB(1, 1, 1);
    const w = makeWindMaterials(m, uniforms, { amp: 0.05, freq: 1.5, bow: 0.4 });
    disposables.push(w.material, w.depthMaterial);
    return w;
  })();
  // `sanitize()` (inside asCutout) clears vertexColors; the per-blossom tint
  // lives on instanceColor and buildFlower() supplies the white `color`
  // attribute the flag needs.
  flowerMat.material.vertexColors = true;

  // The meadow's tone variation is per-instance, so the grass clone has to read
  // instanceColor. `sanitize()` turned vertexColors off; every geometry drawn
  // with this material is built by `buildGrassTuft`, which supplies the white
  // `color` attribute the flag requires.
  grassMat.material.vertexColors = true;

  const needleGrid = atlasGrid(needleMat.material);
  const leafGrid = atlasGrid(leafMat.material);
  const darkGrid = atlasGrid(leafDarkMat.material);
  const grassGrid = atlasGrid(grassMat.material);
  const flowerGrid = atlasGrid(flowerMat.material);

  /** Register an InstancedMesh: shadows, culling, bookkeeping. */
  function addInstanced(geo, wind, capacity, opts = {}) {
    const mesh = new THREE.InstancedMesh(geo, wind.material, Math.max(1, capacity));
    mesh.castShadow = opts.cast !== false;
    mesh.receiveShadow = opts.receive !== false;
    mesh.customDepthMaterial = wind.depthMaterial;
    mesh.count = 0;
    mesh.name = opts.name || geo.name;
    mesh.frustumCulled = opts.frustumCulled === undefined ? false : opts.frustumCulled;
    if (!mesh.frustumCulled) mesh.boundingSphere = _sphere.clone();
    mesh.instanceMatrix.setUsage(opts.dynamic ? THREE.DynamicDrawUsage : THREE.StaticDrawUsage);
    group.add(mesh);
    meshes.push(mesh);
    disposables.push(geo);
    return mesh;
  }

  /* -------------------------------------------------------------- species -- */

  const speciesRng = new Rng('hollowbrook-species');
  // Every variant costs two InstancedMeshes (trunk + foliage) and therefore
  // ~6 draw calls across the colour, shadow and AO passes. The conifers are the
  // skyline so they keep two shapes; the broadleaves get their variety from
  // per-instance scale, yaw and tilt instead.
  const VARIANTS = [2, 1];   // [conifer, broadleaf]
  /** @type {{conifer:Array,broadleaf:Array}} */
  const variants = { conifer: [], broadleaf: [] };
  /** Mid LOD of the SAME variant — `fork(tag)` is pure, so both passes get the
   *  same generator and therefore the same tree. */
  const midVariants = { conifer: [], broadleaf: [] };
  for (let i = 0; i < VARIANTS[0]; i++) {
    variants.conifer.push(buildConifer(speciesRng.fork(`conifer${i}`), detail, needleGrid));
    midVariants.conifer.push(
      buildConifer(speciesRng.fork(`conifer${i}`), detail, needleGrid, { mid: true }));
  }
  for (let i = 0; i < VARIANTS[1]; i++) {
    variants.broadleaf.push(buildBroadleaf(speciesRng.fork(`broadleaf${i}`), detail, leafGrid));
    midVariants.broadleaf.push(
      buildBroadleaf(speciesRng.fork(`broadleaf${i}`), detail, leafGrid, { mid: true }));
  }

  /* -------------------------------------------------------------- scatter -- */

  const rng = new Rng('hollowbrook-vegetation');
  const forestNoise = valueNoise2D(4242);

  /**
   * @typedef {Object} TreeRec
   * @property {number} species 0 conifer, 1 broadleaf
   * @property {number} variant
   * @property {number} x,z,y,scale,yaw
   * @property {boolean} cut  dropped from the billboard tier (deep field only)
   */
  /** @type {Array} */
  const trees = [];

  const pushTree = (kind, x, z, scale, r) => {
    const species = kind === 'conifer' ? 0 : 1;
    const variant = r.int(0, VARIANTS[species] - 1);
    const v = variants[kind][variant];
    trees.push({
      species, variant,
      x, z,
      y: terra.heightAt(x, z) - 0.15,
      scale,
      yaw: r.range(0, TAU),
      tilt: r.sym(0.045),
      height: v.height * scale,
      trunkRadius: v.trunkRadius * scale,
      cut: false,
    });
  };

  // 1. hand-placed hero trees from the plan
  for (const site of TREE_SITES) {
    const kind = site.kind === 'conifer' ? 'conifer' : 'broadleaf';
    pushTree(kind, site.position[0], site.position[2], site.scale || 1,
      rng.fork(`site${site.position[0]}${site.position[2]}`));
  }

  // 2. forest on the hills. The review found them bald: the old pass allowed
  //    560 trees over a 420 m disc, which is one tree per 1000 m² — a sprinkle,
  //    not a wood. This pass reaches further, plants far more, and shapes them
  //    into belts and clumps that thin out below the ridge lines.
  // ~2 300 trees at `high`. The acceptance below is tuned so the candidate set
  // is already close to the target: capping a much larger set would thin the
  // belts back into the uniform sprinkle this is meant to replace.
  //
  // The hills were rebuilt this pass: the nearest centre moved from 233 m out to
  // 398 m, and four summits — including the dominant 88 m mass to the north —
  // now sit 580-670 m from the plaza. The old REACH of 560 m therefore stopped
  // ON the visible flanks and left exactly those four summits bald. Haze does
  // not hide them either: the lighting stream fogs 170..1500 m, so a summit at
  // 670 m is only ~37 % fogged. 720 m clears the farthest summit; nothing needs
  // planting past it, because a hill's own crown occludes its back slope.
  const targetForest = Math.round(3400 * density);
  const densityGain = THREE.MathUtils.clamp(density, 0.35, 1.35);
  const CELL = 6.5;
  const REACH = 720;
  const beltNoise = valueNoise2D(70715);
  const candidates = [];
  const scatterRng = rng.fork('forest2');
  for (let gz = -REACH; gz <= REACH; gz += CELL) {
    for (let gx = -REACH; gx <= REACH; gx += CELL) {
      const x = gx + scatterRng.sym(CELL * 0.48);
      const z = gz + scatterRng.sym(CELL * 0.48);
      const r = Math.hypot(x, z);
      if (r < 38 || r > REACH) continue;
      // Cheap shaping first — the height-field and layout queries below are the
      // expensive part of a 30 000-cell sweep.
      // Belts: sampled anisotropically so woods stretch along the contours
      // instead of forming round blobs. Clumps: a finer grain on top for the
      // ragged inner edge of each belt.
      const belt = fbm2D(beltNoise, x * 0.0035, z * 0.0105, 4) * 0.5 + 0.5;
      const clump = fbm2D(forestNoise, x * 0.019, z * 0.019, 3) * 0.5 + 0.5;
      // Keep the vale around the village open, and stop short of the crowns.
      const nearFalloff = THREE.MathUtils.smoothstep(r, 42, 135);
      const treeLine = 1 - THREE.MathUtils.smoothstep(hillRelief(x, z), 0.58, 0.94);
      const p = Math.pow(belt, 1.7) * (0.35 + 0.85 * clump)
        * (0.14 + 0.86 * nearFalloff) * (0.15 + 0.85 * treeLine);
      if (scatterRng.next() > p * 0.5 * densityGain) continue;
      if (terra.slopeAt(x, z) > 0.60) continue;       // ~34 degrees; scree above
      if (!plantable(terra, x, z, 4.5, 2.5)) continue;
      candidates.push({ x, z, h: terra.heightAt(x, z), r });
    }
  }
  scatterRng.shuffle(candidates);
  for (let i = 0; i < candidates.length && trees.length < targetForest + TREE_SITES.length; i++) {
    const c = candidates[i];
    // Conifers take the high ground and the far distance; broadleaf hugs the vale.
    const coniferBias = THREE.MathUtils.clamp(0.30 + c.h * 0.020 + (c.r - 60) * 0.0018, 0.1, 0.94);
    const kind = scatterRng.next() < coniferBias ? 'conifer' : 'broadleaf';
    pushTree(kind, c.x, c.z, scatterRng.range(0.72, 1.3), scatterRng);
  }

  /* --------------------------------------------------- tree instanced meshes */

  const perVariant = [new Int32Array(VARIANTS[0]), new Int32Array(VARIANTS[1])];
  for (const t of trees) perVariant[t.species][t.variant]++;

  const treeMeshes = [];   // [species][variant] = { trunk, foliage, mid... }
  for (let s = 0; s < 2; s++) {
    const kind = s === 0 ? 'conifer' : 'broadleaf';
    const row = [];
    for (let v = 0; v < VARIANTS[s]; v++) {
      const cap = Math.max(1, perVariant[s][v]);
      const sp = variants[kind][v];
      const mid = midVariants[kind][v];
      row.push({
        trunk: addInstanced(sp.trunk, barkMat, cap, { name: `${kind}${v}-trunk`, dynamic: true }),
        foliage: addInstanced(sp.foliage, s === 0 ? needleMat : leafMat, cap,
          { name: `${kind}${v}-foliage`, dynamic: true }),
        // The mid tier deliberately does NOT cast. Its band starts at 60 m and
        // `shadowDistance` is 95 m at `high`, and three does not cull instances
        // individually — a casting mid mesh would re-submit all ~170 trees into
        // the shadow map to light the handful still inside the volume.
        midTrunk: mid.trunk
          ? addInstanced(mid.trunk, barkMat, cap,
            { name: `${kind}${v}-trunk-mid`, cast: false, dynamic: true })
          : null,
        midFoliage: addInstanced(mid.foliage, s === 0 ? needleMat : leafMat, cap,
          { name: `${kind}${v}-foliage-mid`, cast: false, dynamic: true }),
        height: sp.height,
      });
    }
    treeMeshes.push(row);
  }

  const billboardMeshes = [
    addInstanced(buildTreeBillboard('conifer', needleGrid, rng.fork('bbc'), detail), needleMat,
      Math.max(1, trees.filter((t) => t.species === 0).length),
      { name: 'conifer-billboard', cast: false, dynamic: true }),
    addInstanced(buildTreeBillboard('broadleaf', leafGrid, rng.fork('bbb'), detail), leafMat,
      Math.max(1, trees.filter((t) => t.species === 1).length),
      { name: 'broadleaf-billboard', cast: false, dynamic: true }),
  ];

  // Bake both LOD matrices once. The per-frame refresh is then a memcpy. The mid
  // tier is real geometry at the real transform, so it reuses `nearMat`.
  const nearMat = new Float32Array(trees.length * 16);
  const farMat = new Float32Array(trees.length * 16);
  const decimRng = rng.fork('bbdecim');
  let cutCount = 0;
  for (let i = 0; i < trees.length; i++) {
    const t = trees[i];
    _q.setFromAxisAngle(_UP, t.yaw);
    _qb.setFromAxisAngle(_v.set(1, 0, 0), t.tilt);
    _q.multiply(_qb);
    _m4.compose(_v.set(t.x, t.y, t.z), _q, _scl.set(t.scale, t.scale, t.scale));
    _m4.toArray(nearMat, i * 16);
    /*
     * Billboard size. Measured off the built geometry rather than guessed: a
     * conifer's foliage bounding box is 13.2 m across at H = 15.4, i.e. 0.86 H,
     * and the card sheet's widest band is 0.80 — so matching the tree exactly
     * would want a scale of ~1.07 H. The old 0.72 H drew a card 0.58 H wide, a
     * third narrower than the tree it stood in for, which is a large part of why
     * the far trees read as spikes. 0.90 H splits the difference (0.72 H drawn,
     * 16 % under the real crown, 25 % wider than before): enough to close the
     * silhouette without turning the hills into a solid canopy.
     */
    const far = Math.hypot(t.x, t.z) > TREE_DECIM_R;
    const cut = far && decimRng.next() > TREE_DECIM_KEEP;
    t.cut = cut;
    if (cut) cutCount++;
    const gain = far ? TREE_DECIM_WIDTH : 1;
    const bw = t.height * (t.species === 0 ? 0.90 : 1.0) * gain;
    const bh = t.height * (far ? TREE_DECIM_HEIGHT : 1);
    _m4.compose(_v.set(t.x, t.y, t.z), _q, _scl.set(bw, bh, bw));
    _m4.toArray(farMat, i * 16);

    // Only trunks near the village are solid — nobody walks into the far forest.
    if (Math.hypot(t.x, t.z) < 80) {
      const hh = Math.min(2.4, t.height * 0.35);
      colliders.push(cylinderCollider(
        [t.x, t.y + hh, t.z], Math.max(0.16, t.trunkRadius * 1.5), hh, null,
        { tag: 'tree', friction: 0.9 }));
    }
  }

  /* -------------------------------------------------------------- bushes --- */

  // One shape only: a second bush mesh cost ~3 draw calls for variety that the
  // per-instance scale and yaw already provide.
  const bushRng = rng.fork('bushes');
  const bushGeo = buildBush(bushRng.fork('b0'), darkGrid, Math.max(7, Math.round(13 * detail)));
  const bushSpots = [];

  // Against the gable ends and back walls of every cottage.
  for (const plot of PLOTS) {
    const n = 3 + (plot.width > 12 ? 2 : 0);
    for (let i = 0; i < n; i++) {
      const side = bushRng.bool() ? 1 : -1;
      const lx = side * (plot.width / 2 + bushRng.range(0.5, 1.3));
      const lz = bushRng.range(-plot.depth / 2 + 0.4, plot.depth / 2 - 0.2);
      plotToWorld(plot, lx, 0, lz, _v);
      if (insideAnyBuilding(_v.x, _v.z, 0.2)) continue;
      if (insidePlaza(_v.x, _v.z)) continue;
      if (!isSoil(terra, _v.x, _v.z)) continue;
      bushSpots.push({ x: _v.x, z: _v.z, s: bushRng.range(0.7, 1.15) });
    }
  }
  // Ragged skirt of scrub where the meadow meets the tree line.
  const scrubTarget = Math.round(260 * density);
  for (let i = 0, guard = 0; i < scrubTarget && guard < scrubTarget * 12; guard++) {
    const a = bushRng.range(0, TAU);
    const r = bushRng.range(42, 230);
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    if (!plantable(terra, x, z, 3.5, 1.5)) continue;
    if (terra.slopeAt(x, z) > 0.62) continue;
    bushSpots.push({ x, z, s: bushRng.range(0.65, 1.5) });
    i++;
  }

  const bushMesh = addInstanced(bushGeo, leafDarkMat, Math.max(1, bushSpots.length),
    { name: 'bush', dynamic: true });
  const bushMat = new Float32Array(bushSpots.length * 16);
  for (let i = 0; i < bushSpots.length; i++) {
    const b = bushSpots[i];
    _q.setFromAxisAngle(_UP, bushRng.range(0, TAU));
    _m4.compose(_v.set(b.x, terra.heightAt(b.x, b.z) - 0.12, b.z), _q,
      _scl.set(b.s, b.s * bushRng.range(0.8, 1.15), b.s));
    _m4.toArray(bushMat, i * 16);
  }

  /* ----------------------------------------------------------- hedgerows --- */

  const hedgeRng = rng.fork('hedges');
  const hedgeGeo = buildHedgeSegment(hedgeRng, darkGrid);
  const hedgeXf = [];
  const HEDGE_LINES = Math.round(7 * Math.min(1.2, Math.max(0.4, density)));
  for (let l = 0; l < HEDGE_LINES; l++) {
    const a = hedgeRng.range(0, TAU);
    let x = Math.cos(a) * hedgeRng.range(70, 120);
    let z = Math.sin(a) * hedgeRng.range(70, 120);
    let dir = hedgeRng.range(0, TAU);
    const len = hedgeRng.range(45, 130);
    const step = 1.7;
    for (let d = 0; d < len; d += step) {
      dir += hedgeRng.sym(0.035);
      x += Math.cos(dir) * step;
      z += Math.sin(dir) * step;
      if (Math.hypot(x, z) > 300) break;
      if (!plantable(terra, x, z, 5, 2.0)) continue;
      if (terra.slopeAt(x, z) > 0.55) continue;
      hedgeXf.push(x, z, dir, hedgeRng.range(0.75, 1.15));
    }
  }
  const hedgeMesh = addInstanced(hedgeGeo, hedgeMat, Math.max(1, hedgeXf.length / 4),
    { name: 'hedgerow', cast: false });
  {
    const n = hedgeXf.length / 4;
    for (let i = 0; i < n; i++) {
      const x = hedgeXf[i * 4], z = hedgeXf[i * 4 + 1];
      const dir = hedgeXf[i * 4 + 2], s = hedgeXf[i * 4 + 3];
      _q.setFromAxisAngle(_UP, -dir);
      _m4.compose(_v.set(x, terra.heightAt(x, z) - 0.15, z), _q, _scl.set(s, s * 1.1, s));
      hedgeMesh.setMatrixAt(i, _m4);
    }
    hedgeMesh.count = n;
    hedgeMesh.instanceMatrix.needsUpdate = true;
  }

  /* ---------------------------------------------- meadow variation field --- */

  /*
   * The meadow used to be one flat, uniformly saturated green mass where it met
   * the plaza: every tuft the same tone, every cell occupied. Two fields fix
   * that, and both are sampled per tuft, so both must be O(1).
   *
   *   shade  1 in the lee of a wall or under a canopy, 0 in open sun. A grid,
   *          because a per-tuft query against 2 300 trees is not affordable in
   *          the walk-time carpet refresh.
   *   dry    broad drought bands, suppressed by shade — grass browns off in the
   *          sun and stays green where it is shaded.
   *
   * `patch` (metre-scale) then drives local density and tuft size.
   */
  const SHELTER_CELL = 5;
  const SHELTER_R = 320;
  const SHELTER_N = Math.ceil((SHELTER_R * 2) / SHELTER_CELL) + 1;
  const shelter = new Float32Array(SHELTER_N * SHELTER_N);

  // Walls: the strip of ground within a few metres of a cottage is shaded for
  // most of the day, and mown/trampled besides.
  for (let iz = 0; iz < SHELTER_N; iz++) {
    const gz = -SHELTER_R + iz * SHELTER_CELL;
    if (Math.abs(gz) > 110) continue;                 // buildings are all central
    for (let ix = 0; ix < SHELTER_N; ix++) {
      const gx = -SHELTER_R + ix * SHELTER_CELL;
      if (Math.abs(gx) > 110) continue;
      let s = 0;
      if (insideAnyBuilding(gx, gz, 3.0)) s = 0.95;
      else if (insideAnyBuilding(gx, gz, 7.0)) s = 0.55;
      else if (insideAnyBuilding(gx, gz, 12.0)) s = 0.22;
      if (s > shelter[iz * SHELTER_N + ix]) shelter[iz * SHELTER_N + ix] = s;
    }
  }
  // Canopies: stamp a radial falloff per tree. Linear in the tree count, once.
  for (const t of trees) {
    if (Math.abs(t.x) > SHELTER_R || Math.abs(t.z) > SHELTER_R) continue;
    const R = Math.max(2.5, t.height * 0.42);
    const i0 = Math.max(0, Math.floor((t.x - R + SHELTER_R) / SHELTER_CELL));
    const i1 = Math.min(SHELTER_N - 1, Math.ceil((t.x + R + SHELTER_R) / SHELTER_CELL));
    const j0 = Math.max(0, Math.floor((t.z - R + SHELTER_R) / SHELTER_CELL));
    const j1 = Math.min(SHELTER_N - 1, Math.ceil((t.z + R + SHELTER_R) / SHELTER_CELL));
    for (let j = j0; j <= j1; j++) {
      const gz = -SHELTER_R + j * SHELTER_CELL;
      for (let i = i0; i <= i1; i++) {
        const gx = -SHELTER_R + i * SHELTER_CELL;
        const d = Math.hypot(gx - t.x, gz - t.z);
        const s = 1 - THREE.MathUtils.smoothstep(d, R * 0.35, R);
        const k = j * SHELTER_N + i;
        if (s > shelter[k]) shelter[k] = s;
      }
    }
  }

  /** Bilinear shelter lookup — blocky 5 m steps would read as a checkerboard. */
  function shelterAt(x, z) {
    const fx = (x + SHELTER_R) / SHELTER_CELL;
    const fz = (z + SHELTER_R) / SHELTER_CELL;
    if (fx < 0 || fz < 0 || fx >= SHELTER_N - 1 || fz >= SHELTER_N - 1) return 0;
    const i = fx | 0, j = fz | 0;
    const tx = fx - i, tz = fz - j;
    const a = shelter[j * SHELTER_N + i], b = shelter[j * SHELTER_N + i + 1];
    const c = shelter[(j + 1) * SHELTER_N + i], d = shelter[(j + 1) * SHELTER_N + i + 1];
    return (a + (b - a) * tx) * (1 - tz) + (c + (d - c) * tx) * tz;
  }

  const dryNoise = valueNoise2D(9081);
  const patchNoise = valueNoise2D(51244);

  function meadowFieldAt(x, z) {
    const sh = shelterAt(x, z);
    const dryN = THREE.MathUtils.clamp(fbm2D(dryNoise, x * 0.0085, z * 0.0085, 3) * 1.5 + 0.5, 0, 1);
    _field.shade = sh;
    _field.dry = dryN * (1 - 0.85 * sh);
    _field.patch = THREE.MathUtils.clamp(fbm2D(patchNoise, x * 0.055, z * 0.055, 2) * 1.4 + 0.5, 0, 1);
    return _field;
  }

  /**
   * Per-instance tuft multiplier. Centred a little BELOW 1 on purpose: the
   * `grassBlade` albedo is brighter this pass, so the tufts must not add gain on
   * top of it. Blue is pulled down hardest — a meadow is olive, not emerald, and
   * the flat over-saturated look was mostly blue sitting too high.
   *
   * @returns the meadow field, so the caller can reuse it for density.
   */
  function meadowTint(x, z, jitter, out) {
    const f = meadowFieldAt(x, z);
    const j = 0.93 + 0.16 * jitter;
    // Shelter shifts hue far more than value: the headline defect this pass is a
    // dark ground, so the lee of a wall must read as DEEPER GREEN, not as a dark
    // patch. Hence only ~20 % off red and 9 % off green.
    out.setRGB(
      THREE.MathUtils.lerp(0.90, 1.18, f.dry) * (1 - 0.20 * f.shade) * j,
      THREE.MathUtils.lerp(1.00, 1.10, f.dry) * (1 - 0.09 * f.shade) * j,
      THREE.MathUtils.lerp(0.84, 0.58, f.dry) * (1 - 0.06 * f.shade) * j);
    return f;
  }

  /* --------------------------------------------------------------- grass --- */

  /*
   * Two lattices, and both were oversized for what the pixels can carry. The far
   * carpet was 9 216 multi-blade tufts (147 k triangles — the single biggest item
   * in this stream) spread over a 58 m disc at one tuft per 1.5 m², all of them
   * beyond the 13 m the near carpet already covers. Fewer, LARGER tufts read the
   * same: 68² tufts at one per 2.9 m², scaled up ~1.45x, with straight blades
   * (segs 1 — the bend of a 20 cm blade is invisible past 13 m) at 8 triangles
   * each instead of 16. 147 k -> 37 k.
   *
   * The lattice shrink is compensated in APPARENT density rather than count: the
   * density field zero-scales tufts it rejects, so lifting `densityBias` stands
   * more of the smaller lattice up for free (a zero-scaled instance costs the
   * same as a live one). Underfoot the near carpet keeps its bent 5-blade tuft
   * and only loses lattice: 64² -> 56².
   */
  const grassRng = rng.fork('grass');
  const nearGeo = buildGrassTuft(grassRng.fork('near'), grassGrid, detail >= 0.9 ? 5 : 3);
  // Straight blades, and the SAME blade length as before: the extra ground each
  // tuft has to cover is bought with the instance scale's width, not with height.
  // A 0.5 m tuft would be a hayfield.
  const farGeo = buildGrassTuft(grassRng.fork('far'), grassGrid, detail >= 0.9 ? 4 : 3,
    { segs: 1 });

  const nearN = THREE.MathUtils.clamp(Math.round(56 * Math.sqrt(density)), 26, 72);
  const farN = THREE.MathUtils.clamp(Math.round(68 * Math.sqrt(density)), 34, 88);
  const grassDist = q.grassDistance || 40;

  const nearMesh = addInstanced(nearGeo, grassMat, nearN * nearN,
    { name: 'grass-near', cast: false, dynamic: true, frustumCulled: true });
  const farMesh = addInstanced(farGeo, grassMat, farN * farN,
    { name: 'grass-far', cast: false, dynamic: true, frustumCulled: true });
  nearMesh.count = nearN * nearN;
  farMesh.count = farN * farN;

  // Scale spread is wide on purpose: a uniform tuft height is what made the
  // meadow read as a bed of spikes.
  const carpets = [
    new GrassCarpet(nearMesh, nearN, Math.max(0.34, 13 / nearN * 2), terra, 1234,
      {
        minScale: 0.56, maxScale: 1.36, margin: 1.2, roadClear: 0.2,
        tint: meadowTint, densityBias: 0.14,
      }),
    new GrassCarpet(farMesh, farN, Math.max(0.7, grassDist / farN * 2), terra, 5678,
      {
        minScale: 0.95, maxScale: 1.40, margin: 2.2, roadClear: 0.5,
        tint: meadowTint, densityBias: 0.12,
      }),
  ];

  /*
   * The plaza boundary. Two jobs, one mesh:
   *   - a few short blades pushing up between the edge SETTS (inside the plaza,
   *     but only in the last metre and never more than a hand tall), and
   *   - a scruffier fringe of tufts and weeds on the soil just outside, to break
   *     the hard line where the meadow meets the cobbles.
   * Everything else keeps off the paving — that is what `isSoil` is for.
   */
  const fringeRng = rng.fork('fringe');
  const fringeGeo = buildGrassTuft(fringeRng, grassGrid, 4, { minH: 0.13, maxH: 0.23 });
  const fringeXf = [];
  const fringeTarget = Math.round(820 * density);
  for (let i = 0, guard = 0; i < fringeTarget && guard < 40000; guard++) {
    const a = fringeRng.range(0, TAU);
    const edge = plazaRadiusAt(a);
    const between = fringeRng.next() < 0.22;
    // Inside: hugging the rim. Outside: a tail that fades over ~4 m.
    const r = between
      ? edge - Math.pow(fringeRng.next(), 2.4) * 1.05
      : edge + Math.pow(fringeRng.next(), 0.75) * 4.2;
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    if (!isClearGround(x, z, 0.7)) continue;
    // This is the ONE scatter in the file that does not run through
    // `plantable` — the `between` blades are inside the plaza by design — so it
    // needs the apron gate of its own.
    if (onWellApron(x, z)) continue;
    if (nearRoad(x, z, 0.25)) continue;
    // Outside the rim the surface must be soil; between the setts it is cobble
    // by definition, so only the tiny scale keeps it honest.
    if (!between && !isSoil(terra, x, z)) continue;
    const s = between ? fringeRng.range(0.26, 0.5) : fringeRng.range(0.55, 1.15);
    fringeXf.push(x, z, s, fringeRng.range(0, TAU));
    i++;
  }
  const fringeMesh = addInstanced(fringeGeo, grassMat, Math.max(1, fringeXf.length / 4),
    { name: 'grass-fringe', cast: false });
  {
    const n = fringeXf.length / 4;
    for (let i = 0; i < n; i++) {
      const x = fringeXf[i * 4], z = fringeXf[i * 4 + 1];
      _q.setFromAxisAngle(_UP, fringeXf[i * 4 + 3]);
      const s = fringeXf[i * 4 + 2];
      _m4.compose(_v.set(x, terra.heightAt(x, z) - 0.02, z), _q,
        _scl.set(s, s * fringeRng.range(0.75, 1.25), s));
      fringeMesh.setMatrixAt(i, _m4);
      // Same tone field as the carpets, or the fringe would be a uniform green
      // band around exactly the boundary the variation is meant to break up.
      meadowTint(x, z, fringeRng.next(), _col);
      fringeMesh.setColorAt(i, _col);
    }
    fringeMesh.count = n;
    fringeMesh.instanceMatrix.needsUpdate = true;
    if (fringeMesh.instanceColor) fringeMesh.instanceColor.needsUpdate = true;
  }

  /* ------------------------------------------------------------ flowers --- */

  const flowerRng = rng.fork('flowers');
  const flowerGeo = buildFlower(flowerRng, flowerGrid);
  // Meadow blossoms, not garden bedding: pulled a little dustier and, for the
  // white, off the top of the range so a sunlit petal cannot clip. The atlas
  // petals are near-white by design (~0.91 linear) and these tints multiply it,
  // so the brightest blossom lands at 0.65 linear — under the 0.551 plaster plus
  // its own specular, which is where it belongs.
  const FLOWER_TINTS = [0xc4564e, 0xd489a8, 0xe3d091, 0xdcd5c4];
  const flowerXf = [];
  for (let i = 0, guard = 0; i < Math.round(1500 * density) && guard < 60000; guard++) {
    const a = flowerRng.range(0, TAU);
    // From the plaza rim outward: weeds in the fringe are part of breaking the
    // cobble/meadow line, and `plantable` still keeps them off the setts.
    const r = flowerRng.range(26, 175);
    const x = Math.cos(a) * r, z = Math.sin(a) * r;
    if (!plantable(terra, x, z, 2.5, 0.6)) continue;
    if (terra.slopeAt(x, z) > 0.6) continue;
    // The blossom head is a cut-out card now, so the instance scale drives its
    // real size — 1.5 made a 15 cm daisy.
    flowerXf.push(x, z, flowerRng.range(0.75, 1.25), flowerRng.int(0, 3));
    i++;
  }
  const flowerMesh = addInstanced(flowerGeo, flowerMat, Math.max(1, flowerXf.length / 4),
    { name: 'wildflowers', cast: false });

  /* ------------------------------------------------------ ground clutter --- */

  // Fallen leaves only. Twigs and pebbles were a mesh each — six draw calls for
  // detail nobody can see from standing height.
  const clutterRng = rng.fork('clutter');
  const litterGeo = buildLeafLitter(clutterRng, darkGrid);

  // Only trees near the village get litter; the far forest is beyond the
  // distance where a 15 cm leaf survives a pixel.
  const litterHosts = trees.filter((t) => Math.hypot(t.x, t.z) < 150);
  const clutterSpots = [];
  const clutterTarget = Math.round(1500 * density);
  for (let i = 0, guard = 0; i < clutterTarget && guard < clutterTarget * 10; guard++) {
    // Cluster around trees — litter belongs under a canopy, not in open field.
    const t = litterHosts[clutterRng.int(0, Math.max(0, litterHosts.length - 1))];
    if (!t) break;
    const a = clutterRng.range(0, TAU);
    const rr = Math.pow(clutterRng.next(), 0.55) * 5.5;
    const x = t.x + Math.cos(a) * rr;
    const z = t.z + Math.sin(a) * rr;
    if (!isClearGround(x, z, 0.8) || insidePlaza(x, z)) continue;
    if (!isSoil(terra, x, z)) continue;
    clutterSpots.push({ x, z, s: clutterRng.range(0.7, 1.5) });
    i++;
  }
  const litterMesh = addInstanced(litterGeo, litterMat, Math.max(1, clutterSpots.length),
    { name: 'leaf-litter', cast: false });
  for (let i = 0; i < clutterSpots.length; i++) {
    const c = clutterSpots[i];
    _q.setFromAxisAngle(_UP, clutterRng.range(0, TAU));
    _m4.compose(_v.set(c.x, terra.heightAt(c.x, c.z), c.z), _q, _scl.set(c.s, c.s, c.s));
    litterMesh.setMatrixAt(i, _m4);
  }
  litterMesh.count = clutterSpots.length;
  litterMesh.instanceMatrix.needsUpdate = true;

  /* ------------------------------------------------------------- runtime --- */

  const state = {
    lastRefreshAt: -1e9,
    lastX: 1e9,
    lastZ: 1e9,
  };

  const nearCounts = [new Int32Array(VARIANTS[0]), new Int32Array(VARIANTS[1])];
  const midCounts = [new Int32Array(VARIANTS[0]), new Int32Array(VARIANTS[1])];
  const farCounts = [0, 0];

  /**
   * Publish a bucket: set the draw count and upload ONLY the matrices actually
   * written. Every tree mesh is capacity-sized for the whole forest but draws a
   * couple of hundred instances at most (measured peak over every position the
   * player can reach: 32 near, 149 mid at `ultra`), and a bare `needsUpdate`
   * re-uploads the entire buffer — 90 kB per mesh, ~760 kB per walk-refresh
   * across the eight tree meshes. Ranges bring that to ~170 kB even with the two
   * extra tiers added this pass. `start`/`count` are in array elements, and three
   * clears the ranges once it has uploaded them.
   */
  function publish(mesh, count) {
    mesh.count = count;
    if (count === 0) return;          // nothing drawn; an empty range list would
    const a = mesh.instanceMatrix;    // mean "upload everything" instead
    a.clearUpdateRanges();
    a.addUpdateRange(0, count * 16);
    a.needsUpdate = true;
  }

  /**
   * Re-bucket every tree into full / mid / billboard / culled. Zero allocation.
   *
   * @param {number} nearDist  full geometry inside this
   * @param {number} midDist   reduced-card geometry inside this
   */
  function refreshTreeLod(px, pz, cull, nearDist, midDist) {
    // Billboards cost ~36 triangles each, so the far forest keeps drawing well
    // past `cullDistance` — bare hills would be a far worse defect. 760 m clears
    // the whole rebuilt hill ring (farthest summit 669 m) and still sits well
    // inside the lighting stream's 1500 m fog end.
    const farCull = Math.max(cull, TREE_FAR_CULL_M);
    for (let s = 0; s < 2; s++) {
      nearCounts[s].fill(0);
      midCounts[s].fill(0);
      farCounts[s] = 0;
    }
    const cull2 = farCull * farCull;
    const near2 = nearDist * nearDist;
    const mid2 = midDist * midDist;
    for (let i = 0; i < trees.length; i++) {
      const t = trees[i];
      const dx = t.x - px, dz = t.z - pz;
      const d2 = dx * dx + dz * dz;
      if (d2 > cull2) continue;
      // Copied by hand: `subarray` allocates a view, and with a few thousand
      // trees that is a few thousand garbage objects per re-bucket.
      if (d2 <= near2) {
        const cell = treeMeshes[t.species][t.variant];
        const o = nearCounts[t.species][t.variant]++ * 16;
        const a = cell.trunk.instanceMatrix.array;
        const b = cell.foliage.instanceMatrix.array;
        for (let k = 0; k < 16; k++) { const val = nearMat[i * 16 + k]; a[o + k] = val; b[o + k] = val; }
      } else if (d2 <= mid2) {
        // Same transform as the full tier — the mid LOD is the real tree.
        const cell = treeMeshes[t.species][t.variant];
        const o = midCounts[t.species][t.variant]++ * 16;
        const a = cell.midTrunk ? cell.midTrunk.instanceMatrix.array : null;
        const b = cell.midFoliage.instanceMatrix.array;
        for (let k = 0; k < 16; k++) {
          const val = nearMat[i * 16 + k];
          b[o + k] = val;
          if (a) a[o + k] = val;
        }
      } else if (!t.cut) {
        // `cut` is the permanently decimated deep field. It is only skipped in
        // the billboard branch: if the player walks out far enough for one to
        // land inside the mid band it draws as a real tree rather than a hole.
        const dst = billboardMeshes[t.species].instanceMatrix.array;
        const o = farCounts[t.species]++ * 16;
        for (let k = 0; k < 16; k++) dst[o + k] = farMat[i * 16 + k];
      }
    }
    for (let s = 0; s < 2; s++) {
      for (let v = 0; v < VARIANTS[s]; v++) {
        const cell = treeMeshes[s][v];
        publish(cell.trunk, nearCounts[s][v]);
        publish(cell.foliage, nearCounts[s][v]);
        publish(cell.midFoliage, midCounts[s][v]);
        if (cell.midTrunk) publish(cell.midTrunk, midCounts[s][v]);
      }
      publish(billboardMeshes[s], farCounts[s]);
    }
  }

  /**
   * Tier radii for a given cull distance. Both shrink with `cullDistance` so the
   * adaptive-resolution manager still has a lever on the expensive tiers, but at
   * `high` (200 m) they sit exactly on the authored 60 / 180 m.
   */
  const nearDistFor = (cull) => Math.min(TREE_NEAR_M, cull * 0.42);
  const midDistFor = (cull) => Math.min(TREE_MID_M, cull * 0.95);

  /** Bushes fade out with the same cull distance as the trees. */
  function refreshBushLod(px, pz, cull) {
    const cull2 = cull * cull;
    const dst = bushMesh.instanceMatrix.array;
    let w = 0;
    for (let i = 0; i < bushSpots.length; i++) {
      const dx = bushSpots[i].x - px, dz = bushSpots[i].z - pz;
      if (dx * dx + dz * dz > cull2) continue;
      const o = w++ * 16;
      for (let k = 0; k < 16; k++) dst[o + k] = bushMat[i * 16 + k];
    }
    publish(bushMesh, w);
  }

  // First fill, from the spawn point.
  {
    const cull0 = q.cullDistance || 200;
    refreshTreeLod(WORLD.playerStart[0], WORLD.playerStart[2],
      cull0, nearDistFor(cull0), midDistFor(cull0));
  }
  refreshBushLod(WORLD.playerStart[0], WORLD.playerStart[2], q.cullDistance || 200);
  for (const c of carpets) c.refresh(WORLD.playerStart[0], WORLD.playerStart[2], true);

  // Flowers, with a tint per instance.
  {
    const n = flowerXf.length / 4;
    for (let i = 0; i < n; i++) {
      const x = flowerXf[i * 4], z = flowerXf[i * 4 + 1];
      const s = flowerXf[i * 4 + 2];
      _q.setFromAxisAngle(_UP, i * GOLDEN);
      _m4.compose(_v.set(x, terra.heightAt(x, z) - 0.02, z), _q, _scl.set(s, s, s));
      flowerMesh.setMatrixAt(i, _m4);
      flowerMesh.setColorAt(i, _col.setHex(FLOWER_TINTS[flowerXf[i * 4 + 3]]));
    }
    flowerMesh.count = n;
    flowerMesh.instanceMatrix.needsUpdate = true;
    if (flowerMesh.instanceColor) flowerMesh.instanceColor.needsUpdate = true;
  }

  const buildMs = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;

  // Count what is actually submitted (instance `count`), not the capacity —
  // near-LOD trees are capacity-sized for the whole forest but only ever draw
  // the handful inside the LOD radius.
  let triangles = 0;
  for (const m of meshes) {
    const idx = m.geometry.getIndex();
    const tri = (idx ? idx.count : m.geometry.getAttribute('position').count) / 3;
    triangles += tri * m.count;
  }

  /** Tufts actually standing (the density field zero-scales the rest). */
  function liveTufts(mesh) {
    const a = mesh.instanceMatrix.array;
    let live = 0;
    for (let i = 0; i < mesh.count; i++) if (a[i * 16] !== 0) live++;
    return live;
  }

  /**
   * Triangles a tier is submitting right now, for the perf HUD. TREES are counted
   * off the foliage meshes only — a tier is two meshes per tree and counting both
   * would double every population.
   */
  function tierTris(pick) {
    let sum = 0, n = 0;
    for (const m of meshes) {
      if (!pick(m.name)) continue;
      const idx = m.geometry.getIndex();
      const tri = (idx ? idx.count : m.geometry.getAttribute('position').count) / 3;
      sum += tri * m.count;
      if (!/-trunk(-mid)?$/.test(m.name)) n += m.count;
    }
    return { instances: n, triangles: Math.round(sum) };
  }
  const tierFull = tierTris((n) => /^(conifer|broadleaf)\d-(trunk|foliage)$/.test(n));
  const tierMid = tierTris((n) => /-mid$/.test(n));
  const tierFar = tierTris((n) => /-billboard$/.test(n));

  const stats = {
    trees: trees.length,
    // Tier populations from the spawn view, and what each one costs.
    treeLodFull: `${tierFull.instances} / ${tierFull.triangles} tri`,
    treeLodMid: `${tierMid.instances} / ${tierMid.triangles} tri`,
    treeLodBillboard: `${tierFar.instances} / ${tierFar.triangles} tri`,
    treeLodRadii: `${Math.round(nearDistFor(q.cullDistance || 200))} / `
      + `${Math.round(midDistFor(q.cullDistance || 200))} / ${TREE_FAR_CULL_M} m`,
    treesDecimated: cutCount,
    treeColliders: colliders.length,
    bushes: bushSpots.length,
    hedgeSegments: hedgeXf.length / 4,
    grassInstances: nearN * nearN + farN * farN + fringeXf.length / 4,
    // Capacity vs. what the patchy density field actually stands up at spawn.
    grassLiveAtSpawn: liveTufts(nearMesh) + liveTufts(farMesh) + fringeXf.length / 4,
    flowers: flowerXf.length / 4,
    clutter: clutterSpots.length,
    objects: meshes.length,
    trianglesAtSpawn: Math.round(triangles),
    syntheticTerrain: terra.synthetic,
    syntheticSurface: terra.syntheticSurface,
    buildMs: Math.round(buildMs),
  };

  /* -------------------------------------------------------------- update --- */

  function update(dt, ctx) {
    // Wind: a slow gust envelope over the base sway, driven entirely by uniforms.
    const el = ctx ? ctx.elapsed : 0;
    uniforms.uWindTime.value += dt;
    uniforms.uWindStrength.value =
      0.8 + 0.35 * Math.sin(el * 0.19) + 0.18 * Math.sin(el * 0.53 + 1.9);

    const p = ctx && ctx.playerPosition ? ctx.playerPosition : null;
    if (!p) return;
    uniforms.uPlayerPos.value.copy(p);

    const qq = (ctx && ctx.quality) || q;
    const gd = qq.grassDistance || grassDist;
    uniforms.uFadeRange.value.set(gd * 0.82, gd);

    for (let i = 0; i < carpets.length; i++) carpets[i].refresh(p.x, p.z, false);

    // Tree LOD only needs re-bucketing when the player has actually travelled.
    const moved = (p.x - state.lastX) ** 2 + (p.z - state.lastZ) ** 2;
    if (moved > 36 || el - state.lastRefreshAt > 2.0) {
      state.lastX = p.x; state.lastZ = p.z; state.lastRefreshAt = el;
      const cull = qq.cullDistance || 200;
      refreshTreeLod(p.x, p.z, cull, nearDistFor(cull), midDistFor(cull));
      refreshBushLod(p.x, p.z, cull);
    }
  }

  function dispose() {
    for (const d of disposables) d.dispose?.();
    group.clear();
  }

  return { group, colliders, interactables: [], lightAnchors: [], update, dispose, stats };
}
