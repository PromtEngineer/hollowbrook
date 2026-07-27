/**
 * ============================================================================
 *  BUILDINGS — the eleven plots of Hollowbrook
 * ============================================================================
 * Everything here is driven by `PLOTS` in layout.js. Nothing is hard-coded that
 * layout.js already knows.
 *
 * Construction strategy
 * ---------------------
 * A building is authored entirely in PLOT-LOCAL space (+X = the long facade
 * axis, +Z = front / toward the plaza, origin on the ground at the plot
 * centre). Every piece is pushed into a per-material bucket; at the end each
 * bucket is merged with `mergeGeometries` into exactly one Mesh. The resulting
 * group is then translated + rotated into the world, which is *identical* to
 * what `plotToWorld` does, so colliders computed from local coordinates line up
 * with the art to the millimetre.
 *
 * Walls are authored in a wall-local frame (u along the wall, v up, n outward)
 * via a single orthonormal Matrix4 per wall. That is what makes the half-timber
 * frame tractable: studs, rails, braces, jambs and lintels are all 2D layout in
 * (u,v) with a fixed n range.
 *
 * The plaster is a real ExtrudeGeometry slab with real holes, so window and
 * door reveals are the genuine thickness of the wall rather than a decal, and
 * the timber frame stands 55 mm proud of it. That inset is the whole style.
 *
 * The thatch is a swept parametric slab: a cross-section profile (top surface,
 * rounded bulging eave lip, underside) swept along the eave line, with
 * low-frequency waviness, a sagging ridge and a bell-cast eave. It is thick,
 * rounded and never reads as an extruded plane.
 *
 * UV convention: **UVs are in metres**. Every merged mesh gets UVs from a
 * dominant-axis planar projection of its local position, so adjacent pieces
 * share a continuous texture and nothing stretches. If the material library
 * bakes its tiling into `texture.repeat` (as `TextureSet.worldScale` implies),
 * this is exactly right.
 * ============================================================================
 */

import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { boxCollider, quatY } from '../contracts.js';
import { PLOTS, plotToWorld } from './layout.js';
// Second, namespace import of the same module on purpose: INTERIOR_USES is new
// and layout.js is being edited in parallel. A named import of an export that is
// not there yet is a LINK error that takes this whole module down; a namespace
// lookup that misses is just `undefined` and falls back below.
import * as LAYOUT from './layout.js';
import { Rng, valueNoise2D } from '../util/rng.js';
// Namespace import on purpose: a named import of a missing export is a LINK
// error that takes the whole module down, whereas a namespace lookup that
// misses is just `undefined` and falls back below.
import * as TEXTURES from '../materials/textures.js';

/* -------------------------------------------------------------------------- */
/* Constants                                                                   */
/* -------------------------------------------------------------------------- */

const WALL_T = 0.34;          // plaster / daub slab thickness
const TIMBER_INSET = 0.055;   // how far the plaster sits behind the timber face
const TIMBER_T = 0.16;        // timber member depth (n)
const PLINTH_H = 0.40;
const PLINTH_OUT = 0.13;
const THATCH_T = 0.46;        // thatch slab thickness measured on the normal
const THATCH_SAG = 0.085;     // ridge droop at mid-span
const BELL_FRAC = 0.42;       // bell-cast lift at the eave lip, as a fraction of the overhang
const EAVE_CLEAR = 2.25;      // minimum walkable clearance under the eave lip
const JETTY = 0.45;

/** The ivy atlas grid, taken from the texture library rather than copied.
 *  Every cell of the ivy atlas is already a 5-9 leaf cluster, so a card must
 *  sample exactly ONE cell; cropping it with the wrong grid (this used to
 *  hard-code 4x2 against a 2x2 atlas) samples half a cluster per card, which is
 *  what made the ivy read as pale confetti. The fallback keeps a stale or
 *  missing export from being able to break the module. */
const IVY_ATLAS = (() => {
  const l = TEXTURES?.ATLAS_LAYOUT?.ivy;
  const cols = Math.round(l?.cols);
  const rows = Math.round(l?.rows);
  return {
    cols: Number.isFinite(cols) && cols > 0 ? cols : 2,
    rows: Number.isFinite(rows) && rows > 0 ? rows : 2,
  };
})();

/* Doorways. These are CLEAR dimensions of the hole cut through the wall, and
 * they exist to fit a human and a 0.32 m capsule, not to look pretty: the plank
 * door went 1.06 -> 1.20 m so that `interiors.js` can return its 60 mm lining
 * into both reveals and still leave 1.08 m in the clear. Head heights are
 * nominal; every one is clamped against `storeyHeight` below. */
const DOOR_W = { plank: 1.20, double: 2.00, barn: 2.55 };
const DOOR_H = { plank: 2.10, double: 2.30, barn: 2.62 };
/** Plaster left between the door head and the wall head. Was 0.36; at 0.30 the
 *  shallowest plot (the granary, 2.70 m storey on a 0.31 m plinth) still clears
 *  2.09 m under its arch instead of 2.03. */
const DOOR_HEADROOM = 0.30;
const DOOR_LEAF_T = 0.062;    // leaf thickness — boards + a little
/** How far the hinge axis stands inboard of the reveal face.
 *
 *  This was 0.045 and it is the reason four of the eleven buildings could not be
 *  entered. A leaf hung 45 mm inside the reveal, swung past about 100 degrees,
 *  puts its heel round the corner of the opening and INTO the pier collider —
 *  which reaches WALL_COL_IN (0.43 m) inboard of the storey rect while the leaf
 *  hangs on a plane only 0.17 m in. Rapier resolved that spawn overlap the only
 *  way a body on a vertical revolute joint can: it spun the leaf. Measured on the
 *  first step, twelve of fifteen leaves picked up 2.0-6.2 rad/s of yaw and ended
 *  up 110-173 degrees from where they were authored — some flat against the
 *  inside of the wall, some swung right back out across the approach. The three
 *  that stayed put were the three whose random angle happened to fall below the
 *  fouling threshold, and they are exactly the three 1.20 m doors the reviewer
 *  found passable.
 *
 *  0.105 moves the axis far enough in that the fouling threshold rises to about
 *  105 degrees, and `buildDoorLeaves` derives that number rather than trusting
 *  this comment — see `openMax` there. */
const HINGE_INSET = 0.105;
const LEAF_GAP = 0.012;       // clearance at the latch edge
const THRESH_LIP = 0.012;     // threshold stone proud of the finished floor
const WEATHER_BAR = 0.060;    // the bar the leaf closes against — see buildDoor
const JOIST_ZONE = 0.20;      // joist underside below the storey line
const UPPER_FLOOR_UP = 0.04;  // upper finished floor above the storey line
const ROOF_BITE = 0.22;       // how far the thatch underside bites into the wall head
/** How far inside a storey rect a wall collider reaches. The plaster's inner
 *  face is at WALL_T + TIMBER_INSET = 0.395; 0.43 puts the collision surface
 *  35 mm proud of it, which is where interiors.js lines the wall anyway. */
const WALL_COL_IN = WALL_T + 0.09;

const NOISE = valueNoise2D(9173);

/* module-scope scratch — nothing here allocates per frame, but build-time
 * churn matters too when eleven buildings each make a few thousand pieces. */
const _v = new THREE.Vector3();
const _v2 = new THREE.Vector3();
const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _col = new THREE.Color();

/* -------------------------------------------------------------------------- */
/* Material bucket keys -> canonical material names (+ variant options)        */
/* -------------------------------------------------------------------------- */

const VC = { vertexColors: true };

const MAT_SPECS = {
  stone: { name: 'stone', opts: VC },
  plaster: { name: 'plaster', opts: VC },
  plasterWorn: { name: 'plasterWorn', opts: VC },
  timber: { name: 'timber', opts: VC },
  woodBeam: { name: 'woodBeam', opts: VC },
  brick: { name: 'brick', opts: VC },
  terracotta: { name: 'terracotta', opts: VC },
  thatch: { name: 'thatch', opts: VC },
  thatchRidge: { name: 'thatchRidge', opts: VC },
  roofTile: { name: 'roofTile', opts: VC },
  woodPlank: { name: 'woodPlank', opts: VC },
  woodDoor: { name: 'woodDoor', opts: VC },
  woodDoorTeal: { name: 'woodDoorTeal', opts: VC },
  woodDoorGreen: { name: 'woodDoor', opts: { vertexColors: true, color: 0x4c5f3a } },
  iron: { name: 'iron', opts: VC },
  soil: { name: 'soil', opts: VC },
  glass: { name: 'glassWindow', opts: null },
};

/** Resolve a bucket key to a shared material, degrading gracefully. */
function makeMatResolver(materials) {
  const cache = new Map();
  let warned = false;
  return function resolve(key) {
    if (cache.has(key)) return cache.get(key);
    const spec = MAT_SPECS[key] || { name: key, opts: null };
    let m = null;
    if (spec.opts && materials && typeof materials.variant === 'function') {
      try { m = materials.variant(spec.name, spec.opts); } catch { m = null; }
    }
    if (!m && materials && typeof materials.get === 'function') {
      try { m = materials.get(spec.name); } catch { m = null; }
    }
    if (!m) {
      if (!warned) { console.warn('[buildings] material library unavailable — using debug material'); warned = true; }
      m = new THREE.MeshStandardMaterial({ color: 0xff00ff, roughness: 0.9 });
    }
    cache.set(key, m);
    return m;
  };
}

/* -------------------------------------------------------------------------- */
/* Geometry plumbing                                                           */
/* -------------------------------------------------------------------------- */

/** Dominant-axis planar projection. UVs come out in metres. */
function applyLocalUV(geo, offU = 0, offV = 0) {
  const pos = geo.getAttribute('position');
  const nor = geo.getAttribute('normal');
  const n = pos.count;
  const uv = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const nx = Math.abs(nor ? nor.getX(i) : 0);
    const ny = Math.abs(nor ? nor.getY(i) : 1);
    const nz = Math.abs(nor ? nor.getZ(i) : 0);
    let u, vv;
    if (ny >= nx && ny >= nz) { u = x; vv = z; }
    else if (nx >= nz) { u = z; vv = y; }
    else { u = x; vv = y; }
    uv[i * 2] = u + offU;
    uv[i * 2 + 1] = vv + offV;
  }
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  return geo;
}

/** Make every geometry mergeable: indexed, exactly {position,normal,uv,color}. */
function normalizeGeo(geo, tint, keepUV, offU, offV) {
  if (!geo.getAttribute('normal')) geo.computeVertexNormals();
  if (!keepUV || !geo.getAttribute('uv')) applyLocalUV(geo, offU || 0, offV || 0);

  const count = geo.getAttribute('position').count;
  if (!geo.getAttribute('color')) {
    const c = new Float32Array(count * 3);
    const r = tint ? tint.r : 1, g = tint ? tint.g : 1, b = tint ? tint.b : 1;
    for (let i = 0; i < count; i++) { c[i * 3] = r; c[i * 3 + 1] = g; c[i * 3 + 2] = b; }
    geo.setAttribute('color', new THREE.BufferAttribute(c, 3));
  }
  for (const k of Object.keys(geo.attributes)) {
    if (k !== 'position' && k !== 'normal' && k !== 'uv' && k !== 'color') geo.deleteAttribute(k);
  }
  if (!geo.index) {
    const idx = count > 65535 ? new Uint32Array(count) : new Uint16Array(count);
    for (let i = 0; i < count; i++) idx[i] = i;
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
  }
  geo.clearGroups();
  return geo;
}

/** Per-material geometry accumulator for one building. */
class Builder {
  constructor() { this.map = new Map(); }
  add(key, geo, tint, keepUV, offU, offV) {
    if (!geo) return;
    const pos = geo.getAttribute('position');
    if (!pos || pos.count === 0) { geo.dispose(); return; }
    normalizeGeo(geo, tint, keepUV, offU, offV);
    let arr = this.map.get(key);
    if (!arr) this.map.set(key, arr = []);
    arr.push(geo);
  }
}

/** Slightly-varied multiplicative vertex tint (values are linear, ~1.0). */
function tintOf(rng, amount = 0.06, warm = 0) {
  const k = 1 + rng.sym(amount);
  return { r: k * (1 + warm), g: k, b: k * (1 - warm * 0.8) };
}

/* -------------------------------------------------------------------------- */
/* Wall frames                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * An oriented wall of a storey box. `m` maps (u,v,n) -> plot-local, where u
 * runs along the wall, v up from the storey base and n outward from the face.
 */
function makeWall(rect, side) {
  const cx = (rect.x0 + rect.x1) / 2, cz = (rect.z0 + rect.z1) / 2;
  let ox, oz, ux, uz, nx, nz, L;
  if (side === 'front') { ox = cx; oz = rect.z1; ux = 1; uz = 0; nx = 0; nz = 1; L = rect.x1 - rect.x0; }
  else if (side === 'back') { ox = cx; oz = rect.z0; ux = -1; uz = 0; nx = 0; nz = -1; L = rect.x1 - rect.x0; }
  else if (side === 'right') { ox = rect.x1; oz = cz; ux = 0; uz = -1; nx = 1; nz = 0; L = rect.z1 - rect.z0; }
  else { ox = rect.x0; oz = cz; ux = 0; uz = 1; nx = -1; nz = 0; L = rect.z1 - rect.z0; }

  const m = new THREE.Matrix4();
  m.set(
    ux, 0, nx, ox,
    0, 1, 0, rect.y0,
    uz, 0, nz, oz,
    0, 0, 0, 1,
  );
  return { side, L, H: rect.y1 - rect.y0, y0: rect.y0, y1: rect.y1, m, ux, uz, nx, nz, ox, oz, openings: [] };
}

/**
 * Distance from one end wall of `plot` to the nearest other plot footprint.
 *
 * layout.js puts a couple of plots close enough that their footprints actually
 * overlap (weaver/stables, saddlery/inn). We must not move them, but we can
 * stop the gable verge from driving a metre of thatch through the neighbour.
 */
function endClearance(plot, sign) {
  const hw = plot.width / 2, hd = plot.depth / 2;
  let best = Infinity;
  const p = new THREE.Vector3();
  for (let i = 0; i <= 6; i++) {
    plotToWorld(plot, sign * hw, 0, -hd + (2 * hd * i) / 6, p);
    for (const other of PLOTS) {
      if (other === plot) continue;
      const dx = p.x - other.position[0], dz = p.z - other.position[2];
      const c = Math.cos(other.rotation), s = Math.sin(other.rotation);
      const lx = dx * c - dz * s;      // true inverse of plotToWorld's rotation
      const lz = dx * s + dz * c;
      const ox = Math.max(0, Math.abs(lx) - other.width / 2);
      const oz = Math.max(0, Math.abs(lz) - other.depth / 2);
      best = Math.min(best, Math.hypot(ox, oz));
    }
  }
  return best;
}

/** Is (x,z) inside another plot's footprint, plus a margin? */
function insideOtherPlot(plot, x, z, margin) {
  for (const other of PLOTS) {
    if (other === plot) continue;
    const dx = x - other.position[0], dz = z - other.position[2];
    const c = Math.cos(other.rotation), s = Math.sin(other.rotation);
    const lx = dx * c - dz * s;      // true inverse of plotToWorld's rotation
    const lz = dx * s + dz * c;
    if (Math.abs(lx) < other.width / 2 + margin && Math.abs(lz) < other.depth / 2 + margin) {
      return other.id;
    }
  }
  return null;
}

/**
 * Slide the primary door along its front wall until you can actually walk up to
 * it. Two pairs of plots in layout.js have footprints that genuinely overlap
 * (weaver/stables, saddlery/inn) — `endClearance` already works round that for
 * the thatch — and on the saddlery the neighbour's wall crosses in front of the
 * authored door position, so the way in was blocked by the inn. Only the saddlery
 * moves, and only by 0.60 m; every other plot's door stays exactly where
 * layout.js put it. Fails soft: if no position on the wall is clear, keep the
 * authored one rather than throwing away the door.
 *
 * Margin 0.62 = the 0.26 m the neighbour's wall collider stands off its footprint
 * plus the 0.32 m capsule radius, plus a little.
 */
function reachableDoorU(plot, dw, uLo, uHi) {
  const hd = plot.depth / 2;
  const p = new THREE.Vector3();
  const blocked = (u) => {
    for (const du of [-dw / 2 + 0.34, 0, dw / 2 - 0.34]) {
      // out to 2.7 m: the whole approach corridor has to be walkable, not just
      // the step in front of the threshold
      for (const out of [0.45, 1.1, 1.9, 2.7]) {
        plotToWorld(plot, u + du, 0, hd + out, p);
        const who = insideOtherPlot(plot, p.x, p.z, 0.62);
        if (who) return who;
      }
    }
    return null;
  };
  const u0 = THREE.MathUtils.clamp(plot.door.x, uLo, uHi);
  const who = blocked(u0);
  if (!who) return u0;
  for (let step = 1; step <= 90; step++) {
    for (const s of [-1, 1]) {
      const u = THREE.MathUtils.clamp(u0 + s * step * 0.12, uLo, uHi);
      if (!blocked(u)) {
        console.info(`[buildings] ${plot.id}: door moved ${(u - u0).toFixed(2)} m along the wall — `
          + `the authored position is behind ${who}'s footprint`);
        return u;
      }
    }
  }
  console.warn(`[buildings] ${plot.id}: no reachable door position on the front wall (${who} overlaps it)`);
  return u0;
}

function wallPoint(wall, u, v, n, out = new THREE.Vector3()) {
  return out.set(
    wall.ox + wall.ux * u + wall.nx * n,
    wall.y0 + v,
    wall.oz + wall.uz * u + wall.nz * n,
  );
}

/** A box authored in wall space, optionally rotated in the u-v plane. */
function wallBox(b, wall, key, u, v, n, du, dv, dn, tint, angle) {
  const g = new THREE.BoxGeometry(Math.abs(du), Math.abs(dv), Math.abs(dn));
  if (angle) g.rotateZ(angle);
  g.translate(u, v, n);
  g.applyMatrix4(wall.m);
  b.add(key, g, tint);
}

/** A box authored directly in plot-local space. */
function localBox(b, key, x, y, z, dx, dy, dz, tint, rotY) {
  const g = new THREE.BoxGeometry(Math.abs(dx), Math.abs(dy), Math.abs(dz));
  if (rotY) g.rotateY(rotY);
  g.translate(x, y, z);
  b.add(key, g, tint);
}

/* interval helpers -------------------------------------------------------- */

/** Complement of `blocked` inside [min,max], with a minimum useful length. */
function freeIntervals(min, max, blocked, minLen = 0.06) {
  const bs = blocked
    .filter((o) => o[1] > min && o[0] < max)
    .map((o) => [Math.max(min, o[0]), Math.min(max, o[1])])
    .sort((a, c) => a[0] - c[0]);
  const out = [];
  let cur = min;
  for (const [a, c] of bs) {
    if (a > cur + minLen) out.push([cur, a]);
    cur = Math.max(cur, c);
  }
  if (max > cur + minLen) out.push([cur, max]);
  return out;
}

/** Openings on a wall that overlap the vertical band [v0,v1] -> u intervals. */
function blockedU(openings, v0, v1, pad = 0) {
  const r = [];
  for (const o of openings) {
    if (o.v1 > v0 && o.v0 < v1) r.push([o.u0 - pad, o.u1 + pad]);
  }
  return r;
}

/** Openings that overlap the vertical strip [u0,u1] -> v intervals. */
function blockedV(openings, u0, u1, pad = 0) {
  const r = [];
  for (const o of openings) {
    if (o.u1 > u0 && o.u0 < u1) r.push([o.v0 - pad, o.v1 + pad]);
  }
  return r;
}

/* -------------------------------------------------------------------------- */
/* Arches                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * Points of a segmental (or semicircular) arch of half-width `a` and rise `f`,
 * measured from the springing line. Returns [[u,v], ...] left to right.
 */
function archPoints(a, f, steps) {
  if (f <= 1e-4) return [[-a, 0], [a, 0]];
  const R = (a * a + f * f) / (2 * f);
  const cy = f - R;
  const th = Math.asin(Math.min(1, a / R));
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const ang = -th + (2 * th * i) / steps;
    pts.push([R * Math.sin(ang), cy + R * Math.cos(ang)]);
  }
  return pts;
}

/** Voussoir ring around an arched head, in `key` material. */
function archRing(b, wall, key, cu, springV, a, f, depthN, ringW, rng, tintFn) {
  if (f <= 1e-4) return;
  const R = (a * a + f * f) / (2 * f);
  const cy = springV + f - R;
  const th = Math.asin(Math.min(1, a / R));
  const steps = Math.max(5, Math.round((2 * th) / 0.20));
  const rm = R - 0.02 + ringW / 2;
  const arcW = 2 * rm * Math.sin(th / steps) * 1.22;
  for (let i = 0; i < steps; i++) {
    const ang = -th + (2 * th * (i + 0.5)) / steps;
    const u = cu + rm * Math.sin(ang);
    const v = cy + rm * Math.cos(ang);
    wallBox(b, wall, key, u, v, depthN, ringW, arcW, TIMBER_T + 0.10, tintFn(rng), Math.PI / 2 - ang);
  }
}

/* -------------------------------------------------------------------------- */
/* Thatch: swept parametric slab                                               */
/* -------------------------------------------------------------------------- */

/**
 * Sweep a deep thatch cross-section along a roof face.
 *
 * The section is authored in (d = distance up the slope from the lip, n = offset
 * along the face normal) and is SYMMETRIC about its medial surface at n = -T/2:
 * profile index j on the top surface pairs with P-1-j underneath at the same d,
 * and the two halves of the eave roll pair with each other. `vergeRoll` depends
 * on that — the midpoint of a pair is exactly the medial point, which is what
 * lets a gable edge roll over onto itself without a seam.
 *
 * Shaping, all of which the roof needs to read as straw rather than card:
 *   `swell`  the last half metre before the lip fattens ABOVE the slope plane,
 *            so the eave terminates in a bulging half-round roll;
 *   `bell`   the sprocketed bell-cast — the lip curls up off the roof plane;
 *   `wave`   two octaves of low-frequency noise over the WHOLE surface, along
 *            the eave *and* up the slope, faded out at the ridge so both faces
 *            still share the ridge line exactly;
 *   `sag`    a pure -Y droop, largest mid-span, shared by both faces.
 *
 * @param {(s:number)=>THREE.Vector3} eaveFn  outermost eave-lip line, s in [0,1]
 * @param {(s:number)=>THREE.Vector3} topFn   ridge/hip line the face runs up to
 * @param {Object} o  { sSegs, detail, thickness, sag, wave, swell, bell, bellD,
 *                      seed, color, weather }
 * @returns {{geometry:THREE.BufferGeometry, rings:Array, P:number, sSegs:number}}
 */
function thatchPatch(eaveFn, topFn, o) {
  const T = o.thickness ?? THATCH_T;
  const detail = Math.max(0.3, o.detail ?? 1);
  const rc = T * 0.60;                     // eave lip roll radius, measured up-slope
  const sSegs = Math.max(4, Math.round((o.sSegs ?? 24) * Math.max(0.55, detail)));
  const nTop = Math.max(4, Math.round(12 * detail));
  const nCap = Math.max(4, Math.round(8 * detail));
  const seed = o.seed ?? 0;
  const wave = o.wave ?? 0.045;
  const sag = o.sag ?? THATCH_SAG;
  const swell = o.swell ?? T * 0.26;
  const bell = o.bell ?? 0;
  const bellD = o.bellD ?? 1.15;
  const base = o.color || { r: 1, g: 1, b: 1 };
  const weather = o.weather ?? 0;

  const eaveP = [], topP = [], dirS = [], nrm = [], lenS = [];
  const tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3();

  for (let i = 0; i <= sSegs; i++) {
    const s = i / sSegs;
    const E = eaveFn(s), Tp = topFn(s);
    eaveP.push(E); topP.push(Tp);
    const S = tmpA.copy(Tp).sub(E);
    const L = Math.max(0.05, S.length());
    lenS.push(L);
    dirS.push(S.clone().multiplyScalar(1 / L));
  }
  // Handedness of the (sweep, slope, normal) frame. `N` is forced to point up,
  // which on some faces (any hip end, and a back slope swept the other way)
  // flips the frame — and with it the winding that puts the outside outside.
  let hand = 0;
  for (let i = 0; i <= sSegs; i++) {
    const a = eaveP[Math.max(0, i - 1)], c = eaveP[Math.min(sSegs, i + 1)];
    const U = tmpB.copy(c).sub(a);
    if (U.lengthSq() < 1e-8) U.set(1, 0, 0);
    U.normalize();
    const N = new THREE.Vector3().crossVectors(U, dirS[i]);
    if (N.lengthSq() < 1e-8) N.set(0, 1, 0);
    N.normalize();
    hand += N.y < 0 ? -1 : 1;
    if (N.y < 0) N.negate();
    nrm.push(N);
  }
  const flip = hand < 0;

  /* Cross-section, ridge -> lip -> ridge. Slope entries carry `f`, the fraction
   * of the slope above the roll; roll entries carry an absolute `d`. Entry j and
   * entry P-1-j are mirror images about n = -T/2 — see `vergeRoll`. */
  const prof = [];
  for (let j = 0; j <= nTop; j++) prof.push({ f: 1 - j / nTop, n: 0 });
  for (let i = 0; i < nCap; i++) {
    const ph = ((i + 1) / (nCap + 1)) * Math.PI;
    prof.push({ d: rc - rc * Math.sin(ph), n: -T / 2 + (T / 2) * Math.cos(ph) });
  }
  for (let j = 0; j <= nTop; j++) prof.push({ f: j / nTop, n: -T });
  const P = prof.length;

  const verts = new Float32Array((sSegs + 1) * P * 3);
  const uvs = new Float32Array((sSegs + 1) * P * 2);
  const cols = new Float32Array((sSegs + 1) * P * 3);
  const rings = [];

  let uAcc = 0;
  const prev = new THREE.Vector3();
  for (let i = 0; i <= sSegs; i++) {
    const s = i / sSegs;
    const E = eaveP[i], D = dirS[i], N = nrm[i], L = lenS[i];
    if (i > 0) uAcc += prev.distanceTo(E);
    prev.copy(E);

    const ring = [];
    let vAcc = 0;
    for (let j = 0; j < P; j++) {
      const pr = prof[j];
      const d = pr.d !== undefined ? pr.d : rc + (L - rc) * pr.f;
      const t = Math.min(1, Math.max(0, d / L));
      // every kind of shaping dies out at the ridge, so the two faces meet on one
      // shared line however hard the surface is pushed around lower down
      const ridgeFade = THREE.MathUtils.smoothstep(1 - t, 0.0, 0.26);
      // the swell belongs to the top half of the section only: that is what
      // fattens the lip into a roll instead of just moving the whole slab
      const topF = Math.pow(THREE.MathUtils.clamp((pr.n + T) / T, 0, 1), 2);
      // value noise peaks well below 1, so `wave` is roughly 2x the amplitude
      // you actually see; two octaves at 3 m and 1.5 m across the straw
      const w = wave * (
        NOISE(uAcc * 0.30 + seed, d * 0.26 + seed * 0.7)
        + 0.5 * NOISE(uAcc * 0.68 + 13.1, d * 0.55 + seed)
      );
      const nOff = pr.n
        + ridgeFade * w
        + bell * Math.pow(1 - t, 1.2) * Math.exp(-Math.pow(d / bellD, 1.7))
        + swell * topF * (1 - t) * Math.exp(-Math.pow((d - rc * 1.45) / (rc * 2.5), 2));
      const p = new THREE.Vector3(
        E.x + D.x * d + N.x * nOff,
        E.y + D.y * d + N.y * nOff,
        E.z + D.z * d + N.z * nOff,
      );
      // ridge sag: a pure -Y drop shared by both faces, so the ridge never cracks
      p.y -= sag * Math.sin(Math.PI * s) * t * t;
      const k = i * P + j;
      verts[k * 3] = p.x; verts[k * 3 + 1] = p.y; verts[k * 3 + 2] = p.z;
      if (j > 0) vAcc += p.distanceTo(ring[j - 1]);
      uvs[k * 2] = uAcc;
      uvs[k * 2 + 1] = vAcc;
      // Weathering: streaks run DOWN the slope (they vary across it and barely
      // along it) and gather toward the eave; the underside keeps its own shade.
      const streak = Math.max(0, NOISE(uAcc * 1.25 + seed * 2.3, d * 0.05));
      const shade = (1 - weather * 0.36 * streak * (0.35 + 0.65 * (1 - t)))
        * (0.80 + 0.20 * Math.pow(THREE.MathUtils.clamp((pr.n + T) / T, 0, 1), 0.55));
      cols[k * 3] = base.r * shade;
      cols[k * 3 + 1] = base.g * shade;
      cols[k * 3 + 2] = base.b * shade;
      ring.push(p);
    }
    rings.push(ring);
  }

  // Index the grid. WINDING MATTERS AND IS EASY TO GET BACKWARDS: +j runs from
  // the ridge DOWN the top surface, so on a right-handed frame (a, a+1, c) is
  // the order that puts the face normal on the outside. Wound the other way the
  // whole roof is culled and you see straight through the top surface onto the
  // underside — which is exactly what made these roofs read as sheets of card.
  const quads = sSegs * (P - 1);
  const index = new Uint32Array(quads * 6);
  let w = 0;
  for (let i = 0; i < sSegs; i++) {
    for (let j = 0; j < P - 1; j++) {
      const a = i * P + j, c = (i + 1) * P + j;
      if (flip) {
        index[w++] = a; index[w++] = c; index[w++] = a + 1;
        index[w++] = a + 1; index[w++] = c; index[w++] = c + 1;
      } else {
        index[w++] = a; index[w++] = a + 1; index[w++] = c;
        index[w++] = a + 1; index[w++] = c + 1; index[w++] = c;
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  geo.computeVertexNormals();

  return { geometry: geo, rings, P, sSegs };
}

/**
 * Orient a CLOSED triangle soup outward, in place. The signed volume of a closed
 * mesh is positive only when its faces are wound outward, so this fixes a whole
 * hand-built shell in one go rather than asking every push site to get the vertex
 * order right. Only valid for closed shells — use `pushTri` for open strips.
 */
function orientClosed(pos) {
  let vol = 0;
  for (let i = 0; i < pos.length; i += 9) {
    const ax = pos[i], ay = pos[i + 1], az = pos[i + 2];
    const bx = pos[i + 3], by = pos[i + 4], bz = pos[i + 5];
    const cx = pos[i + 6], cy = pos[i + 7], cz = pos[i + 8];
    vol += ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
  }
  if (vol >= 0) return pos;
  for (let i = 0; i < pos.length; i += 9) {
    for (let k = 0; k < 3; k++) {
      const t = pos[i + 3 + k]; pos[i + 3 + k] = pos[i + 6 + k]; pos[i + 6 + k] = t;
    }
  }
  return pos;
}

/** Append one triangle, wound so its face normal points along `ref`. */
function pushTri(arr, a, b, c, ref) {
  const nx = (b.y - a.y) * (c.z - a.z) - (b.z - a.z) * (c.y - a.y);
  const ny = (b.z - a.z) * (c.x - a.x) - (b.x - a.x) * (c.z - a.z);
  const nz = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  if (nx * ref.x + ny * ref.y + nz * ref.z >= 0) {
    arr.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
  } else {
    arr.push(a.x, a.y, a.z, c.x, c.y, c.z, b.x, b.y, b.z);
  }
}

/**
 * Roll a thatch edge over on itself: sweep the end cross-section outward while
 * collapsing it onto its own medial line. Because `thatchPatch` makes the
 * section symmetric, the medial point of profile index j is simply the midpoint
 * of j and P-1-j — so the result is a true half-round bull-nose running the
 * whole length of the gable verge, closed at the far side, with no thin edge
 * anywhere on it. Used for gable verges and for dormer hoods.
 *
 * @param {THREE.Vector3[]} ring   one end ring of a thatch patch
 * @param {THREE.Vector3} outward  direction the verge overhangs
 * @param {number} r               roll radius
 */
function vergeRoll(ring, outward, r = 0.2, layers = 4) {
  const P = ring.length;
  if (P < 6) return null;
  const M = [];
  for (let j = 0; j < P; j++) {
    M.push(new THREE.Vector3().addVectors(ring[j], ring[P - 1 - j]).multiplyScalar(0.5));
  }
  const rows = [];
  for (let k = 0; k <= layers; k++) {
    const th = (k / layers) * (Math.PI / 2);
    const shrink = 1 - Math.cos(th);
    const out = r * Math.sin(th);
    const row = [];
    for (let j = 0; j < P; j++) {
      row.push(new THREE.Vector3().lerpVectors(ring[j], M[j], shrink).addScaledVector(outward, out));
    }
    rows.push(row);
  }
  const pos = [];
  const ref = new THREE.Vector3();
  for (let k = 0; k < layers; k++) {
    for (let j = 0; j < P - 1; j++) {
      const A = rows[k][j], B = rows[k][j + 1], C = rows[k + 1][j], D = rows[k + 1][j + 1];
      ref.copy(ring[j]).sub(M[j]);
      if (ref.lengthSq() < 1e-8) ref.copy(outward); else ref.normalize().addScaledVector(outward, 0.7);
      pushTri(pos, A, C, B, ref);
      pushTri(pos, B, C, D, ref);
    }
  }
  if (!pos.length) return null;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geo.computeVertexNormals();
  return geo;
}

/** Swept circular tube along a polyline — hip rolls, ridge liggers, bracket arms. */
function tubeAlong(pts, radius, radialSegs = 6) {
  if (pts.length < 2) return null;
  const N = pts.length;
  const pos = new Float32Array(N * radialSegs * 3);
  const up = new THREE.Vector3(0, 1, 0);
  const tan = new THREE.Vector3(), nx = new THREE.Vector3(), by = new THREE.Vector3();
  for (let i = 0; i < N; i++) {
    const a = pts[Math.max(0, i - 1)], c = pts[Math.min(N - 1, i + 1)];
    tan.copy(c).sub(a);
    if (tan.lengthSq() < 1e-9) tan.set(1, 0, 0);
    tan.normalize();
    nx.crossVectors(up, tan);
    if (nx.lengthSq() < 1e-6) nx.set(1, 0, 0); else nx.normalize();
    by.crossVectors(tan, nx).normalize();
    for (let j = 0; j < radialSegs; j++) {
      const a2 = (j / radialSegs) * Math.PI * 2;
      const cx = Math.cos(a2) * radius, cy = Math.sin(a2) * radius;
      const k = (i * radialSegs + j) * 3;
      pos[k] = pts[i].x + nx.x * cx + by.x * cy;
      pos[k + 1] = pts[i].y + nx.y * cx + by.y * cy;
      pos[k + 2] = pts[i].z + nx.z * cx + by.z * cy;
    }
  }
  const idx = new Uint32Array((N - 1) * radialSegs * 6);
  let w = 0;
  for (let i = 0; i < N - 1; i++) {
    for (let j = 0; j < radialSegs; j++) {
      const j2 = (j + 1) % radialSegs;
      const a = i * radialSegs + j, b = i * radialSegs + j2;
      const c = (i + 1) * radialSegs + j, d = (i + 1) * radialSegs + j2;
      // (a,b,c) not (a,c,b): the (nx, by, tan) frame is right-handed, so this is
      // the order whose normals face out of the tube rather than into it
      idx[w++] = a; idx[w++] = b; idx[w++] = c;
      idx[w++] = b; idx[w++] = d; idx[w++] = c;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.computeVertexNormals();
  return geo;
}

/* -------------------------------------------------------------------------- */
/* Windows                                                                     */
/* -------------------------------------------------------------------------- */

/** Clip an infinite line (through `o`, direction `d`) to a centred rect. */
function clipToRect(ox, oy, dx, dy, hw, hh) {
  let tmin = -1e6, tmax = 1e6;
  const slab = (o, d, h) => {
    if (Math.abs(d) < 1e-6) return Math.abs(o) <= h;
    let t1 = (-h - o) / d, t2 = (h - o) / d;
    if (t1 > t2) { const s = t1; t1 = t2; t2 = s; }
    tmin = Math.max(tmin, t1); tmax = Math.min(tmax, t2);
    return true;
  };
  if (!slab(ox, dx, hw) || !slab(oy, dy, hh)) return null;
  if (tmax <= tmin + 0.04) return null;
  return { mid: (tmin + tmax) / 2, len: tmax - tmin };
}

/** Diamond leaded lights: two families of clipped diagonal cames. */
function leadedLights(b, wall, cu, cv, w, h, n, spacing, rng) {
  const tint = { r: 0.95, g: 0.95, b: 0.98 };
  const hw = w / 2, hh = h / 2;
  const reach = (hw + hh);
  for (const sign of [1, -1]) {
    const a = sign * Math.PI / 4;
    const dx = Math.cos(a), dy = Math.sin(a);
    const px = -dy, py = dx;
    const K = Math.ceil(reach / spacing);
    for (let k = -K; k <= K; k++) {
      const ox = px * k * spacing, oy = py * k * spacing;
      const seg = clipToRect(ox, oy, dx, dy, hw, hh);
      if (!seg) continue;
      wallBox(b, wall, 'iron',
        cu + ox + dx * seg.mid, cv + oy + dy * seg.mid, n,
        seg.len, 0.018, 0.026, tint, a);
    }
  }
}

/**
 * One window: recessed reveal (from the real wall thickness), stone sill,
 * timber frame, mullion/transom, leaded glazing set behind the cames, and an
 * opaque dark backing so the building stays sealed.
 */
function buildWindow(b, wall, win, rng, quality, plot) {
  const detail = quality.geometryDetail ?? 1;
  const cu = (win.u0 + win.u1) / 2;
  const w = win.u1 - win.u0, h = win.v1 - win.v0;
  const cv = (win.v0 + win.v1) / 2;
  const face = win.faceN;                    // outer plaster face
  const recess = face - 0.19;                // glazing plane
  const frameN = recess + 0.045;

  // stone sill, proud of the wall and a little wider than the opening
  wallBox(b, wall, 'stone', cu, win.v0 - 0.035, face - 0.16 + 0.14,
    w + 0.34, 0.11, 0.44, tintOf(rng, 0.05));
  // timber lintel over the head
  wallBox(b, wall, 'timber', cu, win.v1 + 0.075, face - 0.09,
    w + 0.30, 0.15, 0.22, tintOf(rng, 0.07));

  // frame
  const fw = 0.075;
  wallBox(b, wall, 'timber', cu, win.v0 + fw / 2, frameN, w, fw, 0.085, tintOf(rng, 0.05));
  wallBox(b, wall, 'timber', cu, win.v1 - fw / 2, frameN, w, fw, 0.085, tintOf(rng, 0.05));
  wallBox(b, wall, 'timber', win.u0 + fw / 2, cv, frameN, fw, h, 0.085, tintOf(rng, 0.05));
  wallBox(b, wall, 'timber', win.u1 - fw / 2, cv, frameN, fw, h, 0.085, tintOf(rng, 0.05));

  const iw = w - 2 * fw, ih = h - 2 * fw;
  let lights = [{ u: cu, v: cv, w: iw, h: ih }];
  if (w > 0.78) {
    // stone or timber mullion
    wallBox(b, wall, 'stone', cu, cv, frameN, 0.075, h, 0.10, tintOf(rng, 0.04));
    const hwl = (iw - 0.075) / 2;
    lights = [{ u: cu - (hwl + 0.075) / 2 - 0.02, v: cv, w: hwl, h: ih },
      { u: cu + (hwl + 0.075) / 2 + 0.02, v: cv, w: hwl, h: ih }];
  }
  if (h > 1.15) {
    wallBox(b, wall, 'timber', cu, cv + ih * 0.16, frameN, iw, 0.065, 0.09, tintOf(rng, 0.05));
    const nl = [];
    for (const l of lights) {
      const split = l.v + l.h * 0.16;
      nl.push({ u: l.u, v: (l.v - l.h / 2 + split) / 2, w: l.w, h: split - (l.v - l.h / 2) - 0.03 });
      nl.push({ u: l.u, v: (split + l.v + l.h / 2) / 2, w: l.w, h: (l.v + l.h / 2) - split - 0.03 });
    }
    lights = nl;
  }

  const spacing = detail >= 0.75 ? 0.15 : 0.24;
  for (const l of lights) {
    if (l.w < 0.1 || l.h < 0.1) continue;
    // The cames STRADDLE the glass plane (came 26 mm deep at frameN+0.004, glass
    // 18 mm at frameN-0.008) so the leading reads as leading from inside the room
    // as well as from the plaza. Set wholly outboard of the glass, as they were,
    // they vanished indoors and the window became a floating sheet.
    leadedLights(b, wall, l.u, l.v, l.w, l.h, frameN + 0.004, spacing, rng);
    // The pane is 90 mm bigger than its light in both axes, so it laps 45 mm
    // BEHIND the frame, the mullion and the transom. The lights are laid out with
    // 20-30 mm of slack around those bars, and with the opaque backing panel gone
    // that slack was a set of hairline slits: 94 grazing rays out of 25 000 cast
    // from inside these rooms found their way out through them. The lap is
    // invisible — the glass sits inside the frame's own 85 mm depth.
    wallBox(b, wall, 'glass', l.u, l.v, frameN - 0.008, l.w + 0.09, l.h + 0.09, 0.018, null);
  }
  // There used to be an opaque dark panel behind the glazing here, to seal a
  // shell that had nothing inside it. The buildings are enterable now, so the
  // panel is gone: the reveal is the real 340 mm of daub (the plaster slab is an
  // extrusion with a real hole, so the hole's four faces are genuine surfaces),
  // and daylight comes through the leaded lights into the room.

  // shutters — hung outside the timber frame, not sunk into it
  if (win.shutters) {
    const outN = Math.max(face, 0);
    for (const sgn of [-1, 1]) {
      const sw = w * 0.52;
      const su = cu + sgn * (w / 2 + sw / 2 + 0.03);
      const boards = 3;
      for (let i = 0; i < boards; i++) {
        const bw = sw / boards;
        wallBox(b, wall, 'woodPlank', su - sw / 2 + bw * (i + 0.5), cv, outN + 0.045,
          bw - 0.012, h * 0.98, 0.05, tintOf(rng, 0.09));
      }
      for (const vv of [cv - h * 0.34, cv + h * 0.34]) {
        wallBox(b, wall, 'woodPlank', su, vv, outN + 0.078, sw, 0.10, 0.03, tintOf(rng, 0.07));
      }
      wallBox(b, wall, 'iron', su + sgn * (sw / 2 - 0.03), cv, outN + 0.042, 0.05, h * 0.9, 0.05,
        { r: 0.8, g: 0.8, b: 0.82 });
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Doors                                                                       */
/* -------------------------------------------------------------------------- */

function doorMaterialKey(colour) {
  if (colour === 'teal') return 'woodDoorTeal';
  if (colour === 'green') return 'woodDoorGreen';
  return 'woodDoor';
}

/** The plane of the door leaf, measured on the wall normal. */
function leafPlane(faceN) { return faceN - 0.17; }

/**
 * The door SURROUND — everything about a doorway that belongs to the shell:
 * threshold, arch ring or lintel, jambs, the reveal's own lining and the bar the
 * leaf closes against. The leaf itself is a separate rigid body; see
 * `buildDoorLeaves`.
 *
 * The hole through the wall is cut by the plaster extrusion (`wall.openings`),
 * so the reveal here is the genuine thickness of the slab. Nothing in this
 * function may reach inboard of the plaster's inner face: that side of the wall
 * belongs to interiors.js, which lines it.
 */
function buildDoor(b, wall, d, rng, quality, irng) {
  const face = d.faceN;
  const leafN = leafPlane(face);
  const a = d.w / 2;
  const rise = d.rise;
  const B = d.base || 0;
  const head = B + d.jambH + rise;

  // Stone threshold, running the WHOLE depth of the reveal and 60 mm past its
  // inner face, with its top 12 mm proud of the finished floor. It used to be a
  // 110 mm kerb reaching only a third of the way in, which is fine to look at
  // from the plaza and a trip hazard to walk over.
  wallBox(b, wall, 'stone', d.u, B + THRESH_LIP - 0.075, face - 0.03 - 0.02,
    d.w + 0.42, 0.15, WALL_T + 0.06 + 0.34, tintOf(rng, 0.05));

  // Weather bar: a low stone strip immediately outboard of the leaf, which is
  // what stops you seeing daylight (and, from the plaza, the taproom floor)
  // through the 72 mm gap under a closed door. It is deliberately 10 mm SHORTER
  // than that gap, because the leaf has to clear it in both directions — see the
  // note about trapping in buildDoorLeaves.
  wallBox(b, wall, 'stone', d.u, B + WEATHER_BAR / 2, leafN + 0.09,
    d.w + 0.08, WEATHER_BAR, 0.09, tintOf(irng, 0.06));

  if (rise > 0.01) {
    archRing(b, wall, 'stone', d.u, B + d.jambH, a, rise, face + 0.02, 0.27, rng, (r) => tintOf(r, 0.07));
    // jamb blocks
    for (const sgn of [-1, 1]) {
      let v = 0.0;
      let i = 0;
      while (v < d.jambH - 0.05) {
        const bh = Math.min(0.44 + rng.sym(0.08), d.jambH - v);
        wallBox(b, wall, 'stone', d.u + sgn * (a + 0.125), B + v + bh / 2, face + 0.02,
          0.27, bh - 0.02, TIMBER_T + 0.10, tintOf(rng, 0.08));
        v += bh; i++;
        if (i > 12) break;
      }
    }
  } else {
    wallBox(b, wall, 'timber', d.u, B + d.jambH + 0.11, face + 0.01, d.w + 0.44, 0.22, TIMBER_T + 0.08, tintOf(rng, 0.07));
    for (const sgn of [-1, 1]) {
      wallBox(b, wall, 'timber', d.u + sgn * (a + 0.10), B + d.jambH / 2, face + 0.01,
        0.20, d.jambH, TIMBER_T + 0.08, tintOf(rng, 0.07));
    }
  }

  /* -- the reveal ---------------------------------------------------------- */
  // Nothing here may stand proud of either the leaf's swept volume or the
  // plaster's INNER face. The first rules out lining the reveal: the leaf's free
  // edge passes within 3 mm of the latch jamb between roughly 2 and 17 degrees of
  // opening, and its heel sweeps a 31 mm circle around the hinge, so a jamb board
  // would jam the door. The second is the module boundary — interiors.js returns
  // its own 60 mm lining into both reveals and over the head, and two linings in
  // the same 30 mm of air is a z-fight.
  //
  // What the reveal therefore IS, from inside: the genuine 395 mm of the plaster
  // slab. It reads as a framed opening rather than a hole in a plane because the
  // slab is an ExtrudeGeometry with a real hole, so all four faces of the reveal
  // are surfaces, the threshold runs through it, and the pintles are on it.
  const hingeSides = (d.style === 'double' || d.style === 'barn') ? [-1, 1] : [-1];
  for (const hSign of hingeSides) {
    for (const hv of [B + 0.44, head - 0.50]) {
      // The pin the leaf's strap hangs on, ON the hinge axis — which is
      // HINGE_INSET inboard of the reveal, not 50 mm. Geometry only, no collider:
      // the leaf sweeps this exact volume, and a strap wrapping a pintle is what
      // that intersection looks like.
      wallBox(b, wall, 'iron', d.u + hSign * (a - HINGE_INSET), hv, leafN + 0.012,
        0.05, 0.07, 0.075, { r: 0.8, g: 0.8, b: 0.84 });
    }
  }
}

/**
 * The primary door as one or two hinged leaves, each its own rigid body on a
 * vertical revolute joint. Pushes a mesh + `InteractableSpec` per leaf into ctx.
 *
 * Geometry is authored in a leaf-local frame that is a pure Y-rotation away from
 * plot-local: local +X runs along the wall (so the leaf occupies x between 0 and
 * `sign * leafW`), +Y up, +Z along the wall's OUTWARD normal, and the origin is
 * on the hinge axis at mid-height of the leaf. The group's yaw is therefore
 * `plot.rotation + sign * open`, and a positive `sign * open` swings it inward.
 *
 * THE DOOR SWINGS BOTH WAYS, and that is not laziness. The player is a kinematic
 * capsule: it can push a dynamic body, it cannot pull one. A door with a stop on
 * its outboard face — which is what a real cottage door has — opens inward from
 * the plaza and then cannot be opened again from inside, so the first person to
 * shut one behind them is walled up in a taproom for good. Everything that would
 * limit the swing is therefore geometry only; the leaf is stopped at roughly
 * +/-170 degrees by the wall piers either side of the opening, which it cannot
 * rotate through.
 *
 * Every leaf spawns AJAR. Three reasons now, and the third is the whole reason
 * this function was rewritten: an open door is the affordance that tells the
 * player a building can be entered; physics.js caps dynamic bodies at
 * `maxDynamicBodies` — a door that lost the draw comes back as a `fixed` body
 * with no joint, and a shut fixed leaf makes its building unenterable; and the
 * rest angle has to sit inside a WINDOW in which the leaf's collider touches
 * nothing, because a leaf that spawns overlapping the pier is a leaf physics
 * throws across the room (see HINGE_INSET).
 *
 * THE REST ANGLE IS DERIVED, NOT CHOSEN. It is bounded below by the straight walk
 * in and above by the pier:
 *
 *   Lower bound, ~85 deg. Not the width of the throat — the STRAIGHT path. At 60
 *   degrees (which is what a reviewer will ask for, because an ajar door looks
 *   like 60 degrees) a 1.08 m leaf in a 1.20 m opening puts its tip within 20 mm
 *   of the centre line 0.94 m inside the room: the corridor straight in measures
 *   0.64 m and a 0.64 m capsule cannot use it. Only past about 85 degrees has the
 *   leaf swung clear enough that the whole opening is a straight tube.
 *
 *   Upper bound, `openMax`. Past it the heel goes round the corner of the reveal
 *   into the pier collider. Derived below from the actual collider geometry, so
 *   it stays true if WALL_COL_IN, HINGE_INSET or the leaf thickness move.
 *
 * The leaf still swings BOTH WAYS and is still a real hinged body: everything
 * here fixes where it RESTS, not what it can do when it is pushed.
 */
function buildDoorLeaves(plot, d, wall, ctx, rng, rot, toWorld) {
  const key = doorMaterialKey(d.colour);
  const a = d.w / 2;
  const B = d.base || 0;
  const rise = d.rise;
  const leafN = leafPlane(d.faceN);
  // 72 mm off the floor, so the leaf clears the WEATHER_BAR (60 mm) on the way
  // past in either direction. The bar sits right behind that gap and closes it.
  const bottom = B + 0.072;
  // ...and 18 mm under the head, so a square-headed leaf does not rest against
  // the lintel collider. It did, and the door could not move at all.
  const rectH = Math.max(1.2, (B + d.jambH) - bottom - 0.018);
  const R = rise > 0.01 ? (a * a + rise * rise) / (2 * rise) : 0;
  const cyA = rise - R;
  const archAt = (du) => (rise > 0.01
    ? Math.max(0, cyA + Math.sqrt(Math.max(0, R * R - du * du))) : 0);

  const pair = d.style === 'double' || d.style === 'barn';
  const leaves = pair ? 2 : 1;

  /* -- the penetration-free window ---------------------------------------- */
  const hT = DOOR_LEAF_T / 2 + 0.004;              // the collider's half thickness
  // How far inboard the leaf must travel before it is clear of the wall collider
  // and free to swing on round. The pier reaches WALL_COL_IN inside the storey
  // rect; the leaf hangs on leafPlane(faceN), which is `leafN` (negative).
  const clearIn = Math.max(0.04, WALL_COL_IN + leafN);
  // A point (x, z) of the leaf, x from the hinge along the leaf and z across its
  // thickness, sits at Δu = x cos t + z sin t along the wall and Δn = -x sin t +
  // z cos t inboard. It fouls the pier only if some point is BOTH past the reveal
  // (Δu < -HINGE_INSET) and still inside the collider slab (Δn > -clearIn).
  // Bounding those two independently — pessimistic, it lets the worst z apply to
  // both at once — collapses to
  //        |cos t|  <=  (HINGE_INSET * sin t - hT) / clearIn
  // and with sin t >= 0.95 over the range that matters:
  const cLim = THREE.MathUtils.clamp((HINGE_INSET * 0.95 - hT) / clearIn, 0, 1);
  const openMax = Math.PI / 2 + Math.asin(cLim);   // ~105 deg as the numbers stand
  // 85.9-92.8 deg, then held 7 deg clear of openMax. The jitter is cosmetic: no
  // two doors in the village hang at quite the same angle.
  const open = Math.min(openMax - 0.12,
    pair ? rng.range(1.53, 1.62) : rng.range(1.50, 1.60));
  if (!(open > 0.9)) {
    // Can't happen with the constants above; if someone moves them far enough to
    // squeeze the window shut, say so rather than quietly hanging a shut door.
    console.warn(`[buildings] ${plot.id}: door leaf has no penetration-free rest `
      + `angle (openMax ${(openMax * 180 / Math.PI).toFixed(1)} deg)`);
  }

  for (let li = 0; li < leaves; li++) {
    const sign = leaves > 1 ? (li === 0 ? 1 : -1) : 1;
    const share = leaves > 1 ? a : d.w;
    const uh = leaves > 1
      ? (li === 0 ? d.u - a + HINGE_INSET : d.u + a - HINGE_INSET)
      : d.u - a + HINGE_INSET;
    const leafW = share - HINGE_INSET - (leaves > 1 ? 0.008 : LEAF_GAP);
    if (leafW < 0.4) continue;

    const geos = [];
    const push = (g, tint) => { geos.push(normalizeGeo(g, tint)); };
    const yMid = bottom + rectH / 2;

    /* boards, each cut to the arc if there is one */
    const boards = Math.max(3, Math.round(leafW / 0.19));
    const bw = leafW / boards;
    for (let i = 0; i < boards; i++) {
      const x0 = sign * bw * i, x1 = sign * bw * (i + 1);
      const du = Math.max(Math.abs(uh + x0 - d.u), Math.abs(uh + x1 - d.u));
      const top = rectH + archAt(du);
      const g = new THREE.BoxGeometry(bw - 0.012, top, DOOR_LEAF_T);
      g.translate((x0 + x1) / 2, -rectH / 2 + top / 2, 0);
      push(g, tintOf(rng, 0.075));
    }

    /* ledges on the inside face */
    const ledges = d.style === 'barn' ? 3 : 2;
    for (let i = 0; i < ledges; i++) {
      const v = 0.34 + (i * Math.max(0.3, rectH - 0.72)) / Math.max(1, ledges - 1);
      const g = new THREE.BoxGeometry(leafW - 0.03, 0.13, 0.045);
      g.translate(sign * leafW / 2, -rectH / 2 + v, -(DOOR_LEAF_T / 2 + 0.022));
      push(g, tintOf(rng, 0.06));
    }
    if (d.style === 'barn') {
      const run = leafW - 0.1, riseB = rectH - 0.8;
      const g = new THREE.BoxGeometry(Math.hypot(run, riseB), 0.12, 0.04);
      g.rotateZ(sign * Math.atan2(riseB, run));
      g.translate(sign * leafW / 2, 0, -(DOOR_LEAF_T / 2 + 0.022));
      push(g, tintOf(rng, 0.06));
    }

    /* strap hinges and the ring handle, on the outside face. These are iron, but
     * they ride on the door's own material with a near-black vertex tint: a
     * second material would double this chunk's draw calls (13 leaves, not one)
     * for two 80 mm straps nobody can pick out at 0.12 luminance. */
    const irontint = { r: 0.135, g: 0.13, b: 0.125 };
    for (const hv of [0.44, rectH - 0.50]) {
      const g = new THREE.BoxGeometry(leafW * 0.72, 0.075, 0.026);
      g.translate(sign * (0.04 + leafW * 0.36), -rectH / 2 + hv, DOOR_LEAF_T / 2 + 0.014);
      push(g, irontint);
      const p = new THREE.BoxGeometry(0.095, 0.15, 0.032);
      p.translate(sign * 0.05, -rectH / 2 + hv, DOOR_LEAF_T / 2 + 0.016);
      push(p, irontint);
    }
    const ring = new THREE.TorusGeometry(0.075, 0.017, 5, 12);
    ring.translate(sign * (leafW - 0.14), -rectH / 2 + 1.02, DOOR_LEAF_T / 2 + 0.022);
    push(ring, irontint);
    const boss = new THREE.BoxGeometry(0.055, 0.09, 0.05);
    boss.translate(sign * (leafW - 0.14), -rectH / 2 + 1.10, DOOR_LEAF_T / 2 + 0.02);
    push(boss, irontint);
    if (leaves > 1 && li === 0) {
      // meeting stile, lapping the joint between the pair
      const g = new THREE.BoxGeometry(0.075, rectH - 0.04, 0.045);
      g.translate(sign * (leafW + 0.012), 0, DOOR_LEAF_T / 2 + 0.014);
      push(g, tintOf(rng, 0.05));
    }

    let merged = null;
    try { merged = geos.length === 1 ? geos[0] : mergeGeometries(geos, false); } catch { merged = null; }
    if (!merged) { geos.forEach((g) => g.dispose()); continue; }
    if (merged !== geos[0]) geos.forEach((g) => g.dispose());
    else geos.slice(1).forEach((g) => g.dispose());
    merged.computeBoundingSphere();

    const grp = new THREE.Group();
    grp.name = `door-${plot.id}-${li}`;
    const mesh = new THREE.Mesh(merged, ctx.resolve(key));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    grp.add(mesh);

    wallPoint(wall, uh, yMid, leafN, _v);
    const anchorWorld = toWorld(_v.x, _v.y, _v.z);
    grp.position.set(anchorWorld[0], anchorWorld[1], anchorWorld[2]);
    grp.rotation.y = rot + sign * open;
    grp.userData.interactive = true;
    grp.userData.prompt = '[E] open the door';
    grp.userData.kind = 'door';
    /* Everything a controls/physics stream needs to drive this leaf, in the
     * leaf's OWN angle convention: `rest` is where it hangs now, and a revolute
     * motor target of 0 is that same pose because Rapier builds the joint frames
     * from the spawn transform. See the `joint.motor` note below. */
    grp.userData.door = {
      plotId: plot.id,
      leaf: li,
      sign,
      /** rest pose, radians of swing, positive = inward */
      rest: open,
      /** motor targets, RELATIVE to the rest pose (what the joint calls zero) */
      openTarget: 0,
      shutTarget: -open,
      hingeAxis: [0, 1, 0],
    };

    ctx.interactables.push({
      object3D: grp,
      body: 'dynamic',
      // local space: the leaf hangs off the origin, so the body's centre of mass
      // lands at the middle of the leaf and it swings like a door, not a plank
      collider: boxCollider([sign * leafW / 2, 0, 0],
        [leafW / 2, rectH / 2, hT], null, { tag: 'door' }),
      // Lighter than the 30-40 kg this used to be. A real 40 mm oak leaf that
      // size IS about 35 kg, but the player is a KINEMATIC capsule that pushes
      // dynamic bodies through the character controller's impulse path, and at
      // 35 kg on 2.4 of angular damping a walking shove barely moves it — which
      // is what made the reviewer call the doors unshoveable. Half that weight
      // still feels like a slab and still stops dead where you leave it.
      mass: d.style === 'barn' ? 24 : (leaves > 1 ? 17 : 19),
      linearDamping: 0.8,
      // oak on wrought iron does not oscillate; it moves and it stops
      angularDamping: 2.4,
      grabbable: false,
      prompt: '[E] open the door',
      tag: `door-${plot.id}-${li}`,
      joint: {
        type: 'revolute',
        anchorWorld,
        axis: [0, 1, 0],
        /* ADVISORY — physics.js `makeJoint` does not read this today, and the fix
         * above deliberately does not depend on it: the leaf rests where it rests
         * because nothing touches it, not because a limit holds it.
         *
         * Angles are relative to the SPAWN pose (Rapier's revolute frames are
         * built from the spawn transform, so the joint's angle 0 is `open`).
         * min lets it swing shut and on out past the wall face; max stops it
         * `openMax - open` further in, which is where the heel would foul the
         * pier. If limits are ever switched on, these are the right numbers.
         *
         * If you want E to work it (contracts.js has no motor field yet, so this
         * is a suggestion, not part of the contract): the leaf's joint handle is
         * `physics.handleFor(object3D).joint`, and
         *     joint.configureMotorPosition(target, stiffness, damping)
         * with target = object3D.userData.door.openTarget (0) or .shutTarget
         * (-rest), stiffness ~90, damping ~14 drives it while keeping it a real
         * physics body the player can still shove. Toggle on E when
         * `lookRoot.userData.kind === 'door'`; controls.js currently returns from
         * tryGrab() on `grabbable === false`, so it never reaches the door. */
        limits: [-(open + 1.48), openMax - open],
        motor: { stiffness: 90, damping: 14, targets: [-open, 0] },
      },
    });
    ctx.doorGroups.push(grp);
  }
}

/* -------------------------------------------------------------------------- */
/* One building                                                                */
/* -------------------------------------------------------------------------- */

function buildPlot(plot, ctx) {
  const { quality, colliders, lightAnchors, flowerBoxAnchors, interactables } = ctx;
  const detail = quality.geometryDetail ?? 1;
  const thatchDetail = quality.thatchDetail ?? 1;
  const rng = new Rng(plot.seed);
  // Everything the interiors pass added draws from a FORK, not from `rng`. fork()
  // is derived from the seed rather than the current state, so the exterior's
  // shutter choices, plinth rubble, gable tints and ivy jitter come out of this
  // function byte-identical to before the buildings became enterable.
  const irng = rng.fork('interior');
  const b = new Builder();

  const baseY = ctx.groundY(plot.position[0], plot.position[2]);
  const rot = plot.rotation;
  const cosR = Math.cos(rot), sinR = Math.sin(rot);
  const worldQ = quatY(rot);

  /** plot-local -> world */
  const toWorld = (lx, ly, lz) => [
    plot.position[0] + lx * cosR + lz * sinR,
    baseY + ly,
    plot.position[2] - lx * sinR + lz * cosR,
  ];

  /** A world collider from a local box (+ optional local rotation). */
  const pushCollider = (lx, ly, lz, hx, hy, hz, tag, localQuat) => {
    let q = worldQ;
    if (localQuat) {
      _q.set(worldQ[0], worldQ[1], worldQ[2], worldQ[3]).multiply(localQuat);
      q = [_q.x, _q.y, _q.z, _q.w];
    }
    colliders.push(boxCollider(toWorld(lx, ly, lz), [hx, hy, hz], q, { tag }));
  };

  const hw = plot.width / 2, hd = plot.depth / 2;
  const storeys = Math.max(1, plot.storeys | 0);
  const sh = plot.storeyHeight;
  const storeyTop = storeys * sh;
  const jet = plot.jetty && storeys > 1 ? JETTY : 0;

  const plasterKey = plot.plasterTone === 'plasterWorn' ? 'plasterWorn' : 'plaster';
  const stoneBand = plot.style === 'stoneBase' ? Math.min(1.75, sh * 0.55) : 0;

  /* -- roof derivation --------------------------------------------------- */
  const pitch = (plot.roofPitch || 50) * Math.PI / 180;
  const tanP = Math.tan(pitch), cosP = Math.cos(pitch), sinP = Math.sin(pitch);
  const eave = plot.eave || 1.0;
  const thatchVert = THATCH_T / cosP;          // vertical thickness of the slab
  const bellN = BELL_FRAC * eave;              // lift at the lip, along the face normal
  const bellD = 0.9 * eave / cosP;             // how far up the slope the curl reaches

  // The single-storey plots carry a 1.1-1.35 m eave at 48-56 degrees, which puts
  // the lip well below head height. Rather than contradict layout.js, the wall
  // grows an EAVES COURSE above the top storey so the thatch springs high enough
  // to walk (and build a porch) under. Two-storey plots need none.
  const rawTipUnder = (storeyTop + thatchVert - 0.22) - eave * tanP + bellN * cosP - thatchVert;
  // a porch or arcade under the eave needs headroom of its own
  const shelterHead = Math.max(plot.porch ? plot.porch.height : 0,
    plot.arcade ? Math.min(sh - 0.42, 2.62) : 0);
  const needClear = Math.max(EAVE_CLEAR, shelterHead > 0 ? shelterHead + 0.55 : 0);
  const eavesCourse = Math.max(0, Math.min(1.45, needClear - rawTipUnder));
  const wallTop = storeyTop + eavesCourse;

  /* -- storey rectangles ------------------------------------------------- */
  const rects = [];
  for (let i = 0; i < storeys; i++) {
    const j = i > 0 ? jet : 0;
    rects.push({
      x0: -hw - j, x1: hw + j,
      z0: -hd, z1: hd + j,
      y0: i * sh, y1: i === storeys - 1 ? wallTop : (i + 1) * sh,
    });
  }
  const top = rects[rects.length - 1];

  const zRidge = (top.z0 + top.z1) / 2;
  const halfSpan = top.z1 - zRidge;
  // roof top surface at the wall face — 0.22 m inside the wall head, so the
  // underside always bites into the masonry and never opens a slot
  const yWL = wallTop + thatchVert - 0.22;
  const yRidge = yWL + halfSpan * tanP;
  const yEaveTip = yWL - eave * tanP;   // on the plane; the bell lifts it in the mesh
  // the verge (gable end) overhangs less than the eave — partly because that is
  // how thatch is finished, partly because these plots sit shoulder to shoulder
  const vergeMax = eave * 0.62;
  const vergeL = THREE.MathUtils.clamp(endClearance(plot, -1) * 0.45, 0.07, vergeMax);
  const vergeR = THREE.MathUtils.clamp(endClearance(plot, 1) * 0.45, 0.07, vergeMax);
  const xe0 = top.x0 - vergeL, xe1 = top.x1 + vergeR;
  const ze0 = top.z0 - eave, ze1 = top.z1 + eave;
  const planRun = ze1 - zRidge;
  const roofKind = plot.roof || 'thatch';
  const hipped = roofKind === 'thatchHip';
  const halfHip = roofKind === 'thatchHalfHip';
  const tiled = roofKind === 'tile';
  const roofKey = tiled ? 'roofTile' : 'thatch';
  const tGablet = 0.56;
  const hipRun = hipped ? planRun : (halfHip ? (1 - tGablet) * planRun : 0);
  const xh0 = xe0 + hipRun, xh1 = xe1 - hipRun;
  const gabletY = yEaveTip + tGablet * (yRidge - yEaveTip);

  /** Top-surface height of the roof at a local (x,z), bell-cast included. */
  const roofYAt = (lx, lz) => {
    const dz = Math.abs(lz - zRidge);
    let y = yRidge + (dz / Math.max(0.01, planRun)) * (yEaveTip - yRidge);
    let dPlan = Math.max(0, planRun - dz);
    if (hipRun > 0.01) {
      const dx = lx < 0 ? xh0 - lx : lx - xh1;
      if (dx > 0) {
        // a half hip only exists above the gablet line; a full hip runs to the eave
        const yEnd = halfHip ? gabletY : yEaveTip;
        const yh = yRidge + (dx / hipRun) * (yEnd - yRidge);
        if (yh < y) { y = yh; dPlan = Math.max(0, hipRun - dx); }
      }
    }
    const t = dPlan / Math.max(0.01, planRun);
    const d = dPlan / cosP;
    const bell = bellN * Math.pow(Math.max(0, 1 - t), 1.25) * Math.exp(-Math.pow(d / bellD, 1.7));
    return y + bell * cosP;
  };

  /* ---------------------------------------------------------------------- */
  /* Floor level                                                             */
  /* ---------------------------------------------------------------------- */
  // The threshold sits at the top of the steps, so the flight arrives somewhere
  // instead of climbing past a door that starts at ground level — and the
  // FINISHED GROUND FLOOR is that same level. Anything else puts a step at the
  // doorway or, worse, disagrees with the exterior that layout.js already fixed:
  // 0.42 m on the weaver and the hall, 0.31 m on the seven plots with two steps,
  // 0 on the stable and the cooper's, which have none and want an earth floor.
  const stepCount = plot.steps ? Math.max(0, plot.steps.count | 0) : 0;
  const stepRise = stepCount ? Math.min(0.155, 0.42 / stepCount) : 0;
  const doorBase = stepCount * stepRise;
  const floorLocal = doorBase;
  // Effective wall thickness of the ground storey, measured in from the storey
  // rect: the plaster slab runs from faceN back to faceN - WALL_T.
  const faceN0 = plot.style === 'halfTimber' ? -TIMBER_INSET : 0;
  const tEff0 = WALL_T - faceN0;

  /* ---------------------------------------------------------------------- */
  /* Plinth                                                                  */
  /* ---------------------------------------------------------------------- */
  {
    const r = rects[0];
    const o = PLINTH_OUT;
    const w0 = r.x1 - r.x0, d0 = r.z1 - r.z0;
    const zc0 = (r.z0 + r.z1) / 2;
    const stoneT = tintOf(rng, 0.05);
    // A RING, not the solid block this used to be. The finished floor sits at the
    // door threshold, which on nine of the eleven plots is below the plinth cap,
    // so a solid plinth would put a 0.15 m stone bench round the inside of every
    // room. The ring stops 20 mm short of the plaster's inner face, so it is
    // buried in the wall — the base of the shell is still absolutely sealed, and
    // nothing of it shows indoors.
    const tRing = Math.max(0.24, o + tEff0 - 0.02);
    const tCap = tRing + 0.03;
    const yCore = PLINTH_H / 2 - 0.05, hCore = PLINTH_H + 0.10;
    const capY = PLINTH_H + 0.03;
    const capTint = { r: 1.12, g: 1.11, b: 1.08 };
    for (const sgn of [1, -1]) {
      const zR = sgn > 0 ? r.z1 + o - tRing / 2 : r.z0 - o + tRing / 2;
      localBox(b, 'stone', 0, yCore, zR, w0 + 2 * o, hCore, tRing, stoneT);
      const zC = sgn > 0 ? r.z1 + o + 0.03 - tCap / 2 : r.z0 - o - 0.03 + tCap / 2;
      localBox(b, 'stone', 0, capY, zC, w0 + 2 * o + 0.06, 0.07, tCap, capTint);
    }
    for (const sgn of [1, -1]) {
      const xR = sgn > 0 ? r.x1 + o - tRing / 2 : r.x0 - o + tRing / 2;
      localBox(b, 'stone', xR, yCore, zc0, tRing, hCore, d0 + 2 * o - 2 * tRing, stoneT);
      const xC = sgn > 0 ? r.x1 + o + 0.03 - tCap / 2 : r.x0 - o - 0.03 + tCap / 2;
      localBox(b, 'stone', xC, capY, zc0, tCap, 0.07, d0 + 2 * o + 0.06 - 2 * tCap, capTint);
    }
    // Stone underfloor. interiors.js lays the finished floor with its top AT
    // floorY; this is the 40 mm below that, so there is something to stand on and
    // something to look at even on a plot the interior stream never reaches.
    localBox(b, 'stone', 0, floorLocal - 0.13, zc0,
      w0 - 2 * (tEff0 - 0.06), 0.18, d0 - 2 * (tEff0 - 0.06), tintOf(irng, 0.04));
    // rubble faces: protruding stones around the perimeter
    const per = 2 * ((r.x1 - r.x0) + (r.z1 - r.z0));
    const nStones = Math.round(per / 0.62 * detail);
    for (let i = 0; i < nStones; i++) {
      const t = rng.next() * per;
      let x, z, nx = 0, nz = 0;
      const w0 = r.x1 - r.x0, d0 = r.z1 - r.z0;
      if (t < w0) { x = r.x0 + t; z = r.z1 + o; nz = 1; }
      else if (t < w0 + d0) { x = r.x1 + o; z = r.z1 - (t - w0); nx = 1; }
      else if (t < 2 * w0 + d0) { x = r.x1 - (t - w0 - d0); z = r.z0 - o; nz = -1; }
      else { x = r.x0 - o; z = r.z0 + (t - 2 * w0 - d0); nx = -1; }
      const sw = rng.range(0.22, 0.42), sh2 = rng.range(0.12, 0.22);
      const y = rng.range(0.08, PLINTH_H - 0.06);
      localBox(b, 'stone', x + nx * 0.03, y, z + nz * 0.03,
        nx ? 0.09 : sw, sh2, nz ? 0.09 : sw, tintOf(rng, 0.12));
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Openings plan                                                           */
  /* ---------------------------------------------------------------------- */

  const walls = [];
  for (let i = 0; i < storeys; i++) {
    for (const side of ['front', 'right', 'back', 'left']) {
      const w = makeWall(rects[i], side);
      w.storey = i;
      walls.push(w);
    }
  }
  const wallOf = (i, side) => walls.find((w) => w.storey === i && w.side === side);

  const dStyle = DOOR_W[plot.door.style] ? plot.door.style : 'plank';
  const dw = DOOR_W[dStyle];
  const dhTotal = Math.min(DOOR_H[dStyle] + (plot.door.arched ? dw * 0.30 : 0),
    sh - DOOR_HEADROOM - doorBase);
  const dRise = plot.door.arched ? Math.min(dw / 2, dw * 0.36) : 0;
  const door = {
    u: reachableDoorU(plot, dw, -hw + dw / 2 + 0.9, hw - dw / 2 - 0.9),
    w: dw,
    rise: dRise,
    base: doorBase,
    jambH: dhTotal - dRise,
    style: dStyle,
    colour: plot.door.colour,
    // must match the wall it sits in: longRow only frames its upper storey
    faceN: plot.style === 'halfTimber' ? -TIMBER_INSET : 0,
  };
  const frontGround = wallOf(0, 'front');
  frontGround.openings.push({
    u0: door.u - dw / 2, u1: door.u + dw / 2, v0: doorBase, v1: doorBase + dhTotal,
    kind: 'door', rise: dRise, jambH: doorBase + door.jambH,
  });

  // windows
  const winList = [];
  const addWindows = (wall, count, sillV, wW, wH, opts) => {
    if (count <= 0) return;
    const L = wall.L;
    const usable = L - 2 * 1.15;
    if (usable <= wW) return;
    for (let i = 0; i < count; i++) {
      const t = count === 1 ? 0.5 : i / (count - 1);
      const u = -usable / 2 + usable * t;
      const u0 = u - wW / 2, u1 = u + wW / 2;
      let blockedFlag = false;
      for (const o of wall.openings) if (u1 > o.u0 - 0.42 && u0 < o.u1 + 0.42) blockedFlag = true;
      if (blockedFlag) continue;
      const win = {
        u0, u1, v0: sillV, v1: sillV + wH, kind: 'window',
        faceN: wall.faceN, shutters: !!opts.shutters && rng.bool(0.45),
        box: !!opts.box, glow: !!opts.glow,
      };
      wall.openings.push(win);
      winList.push({ wall, win });
    }
  };

  for (let i = 0; i < storeys; i++) {
    const ground = i === 0;
    const fw = wallOf(i, 'front');
    const bw = wallOf(i, 'back');
    const lw = wallOf(i, 'left');
    const rw = wallOf(i, 'right');
    for (const w of [fw, bw, lw, rw]) {
      w.faceN = (plot.style === 'halfTimber' || (plot.style === 'longRow' && i > 0)) ? -TIMBER_INSET : 0;
    }
    const winW = ground ? 0.98 : 0.86;
    const winH = ground ? 1.24 : 1.02;
    const sill = ground ? Math.max(0.92, stoneBand + 0.06) : 0.74;
    const nFront = THREE.MathUtils.clamp(Math.round(fw.L / (ground ? 3.4 : 2.9)), 1, 5);
    addWindows(fw, nFront, sill, winW, winH,
      { shutters: true, box: ground && plot.flowerBoxes, glow: ground });
    addWindows(bw, THREE.MathUtils.clamp(Math.round(bw.L / 4.2), 1, 3), sill, winW * 0.9, winH * 0.92, {});
    addWindows(lw, THREE.MathUtils.clamp(Math.round(lw.L / 3.8), 1, 2), sill, winW * 0.92, winH * 0.94, { shutters: true });
    addWindows(rw, THREE.MathUtils.clamp(Math.round(rw.L / 3.8), 1, 2), sill, winW * 0.92, winH * 0.94, { shutters: true });
  }

  /* ---------------------------------------------------------------------- */
  /* Walls                                                                   */
  /* ---------------------------------------------------------------------- */

  const timberTint = () => tintOf(rng, 0.085, 0.01);
  const plasterTint = () => tintOf(rng, 0.045, 0.008);

  for (const wall of walls) {
    const framed = plot.style === 'halfTimber' || (plot.style === 'longRow' && wall.storey > 0);
    const faceN = wall.faceN ?? 0;

    /* plaster slab with real holes */
    const shape = new THREE.Shape();
    const hl = wall.L / 2;
    shape.moveTo(-hl, 0);
    shape.lineTo(hl, 0);
    shape.lineTo(hl, wall.H);
    shape.lineTo(-hl, wall.H);
    shape.lineTo(-hl, 0);

    for (const o of wall.openings) {
      const p = new THREE.Path();
      p.moveTo(o.u0, o.v0);
      p.lineTo(o.u1, o.v0);
      if (o.kind === 'door' && o.rise > 0.01) {
        p.lineTo(o.u1, o.jambH);
        const a = (o.u1 - o.u0) / 2, cu = (o.u0 + o.u1) / 2;
        const pts = archPoints(a, o.rise, Math.max(6, Math.round(9 * detail)));
        for (let i = pts.length - 1; i >= 0; i--) p.lineTo(cu + pts[i][0], o.jambH + pts[i][1]);
        p.lineTo(o.u0, o.jambH);
      } else {
        p.lineTo(o.u1, o.v1);
        p.lineTo(o.u0, o.v1);
      }
      p.lineTo(o.u0, o.v0);
      shape.holes.push(p);
    }

    const stoneHere = stoneBand > 0 && wall.storey === 0;
    const plasterBase = stoneHere ? stoneBand : 0;
    if (plasterBase > 0) {
      // stone lower band: its own extrusion so the two materials meet cleanly
      const sShape = new THREE.Shape();
      sShape.moveTo(-hl, 0); sShape.lineTo(hl, 0);
      sShape.lineTo(hl, plasterBase); sShape.lineTo(-hl, plasterBase); sShape.lineTo(-hl, 0);
      for (const o of wall.openings) {
        if (o.v0 >= plasterBase) continue;
        const p = new THREE.Path();
        const vt = Math.min(o.v1, plasterBase - 0.001);
        p.moveTo(o.u0, o.v0); p.lineTo(o.u1, o.v0); p.lineTo(o.u1, vt); p.lineTo(o.u0, vt); p.lineTo(o.u0, o.v0);
        sShape.holes.push(p);
      }
      const sg = new THREE.ExtrudeGeometry(sShape, { depth: WALL_T + 0.05, bevelEnabled: false, steps: 1, curveSegments: 4 });
      sg.translate(0, 0, faceN + 0.03 - (WALL_T + 0.05));
      sg.applyMatrix4(wall.m);
      b.add('stone', sg, tintOf(rng, 0.05));
    }

    const g = new THREE.ExtrudeGeometry(shape, { depth: WALL_T, bevelEnabled: false, steps: 1, curveSegments: 4 });
    g.translate(0, 0, faceN - WALL_T);
    g.applyMatrix4(wall.m);
    b.add(plasterKey, g, plasterTint());

    /* the timber frame */
    if (framed) {
      const H = wall.H, L = wall.L, half = L / 2;
      const plateH = 0.20, postW = 0.22, studW = 0.135;
      const midV = H * 0.52;
      const nMid = -TIMBER_T / 2;

      // sill + head plates, broken by openings
      for (const band of [[0, plateH], [H - plateH, H]]) {
        for (const [a, c] of freeIntervals(-half, half, blockedU(wall.openings, band[0], band[1]))) {
          wallBox(b, wall, 'timber', (a + c) / 2, (band[0] + band[1]) / 2, nMid, c - a, plateH, TIMBER_T, timberTint());
        }
      }
      // corner posts
      for (const sgn of [-1, 1]) {
        wallBox(b, wall, 'timber', sgn * (half - postW / 2), H / 2, nMid, postW, H, TIMBER_T, timberTint());
      }
      // mid rail
      for (const [a, c] of freeIntervals(-half + postW, half - postW,
        blockedU(wall.openings, midV - 0.09, midV + 0.09))) {
        wallBox(b, wall, 'timber', (a + c) / 2, midV, nMid, c - a, 0.17, TIMBER_T, timberTint());
      }
      // studs
      const spacing = wall.side === 'front' ? 0.96 : 1.22;
      const span = L - 2 * postW;
      const bays = Math.max(1, Math.round(span / spacing));
      for (let i = 1; i < bays; i++) {
        const u = -half + postW + (span * i) / bays;
        for (const [a, c] of freeIntervals(0, H, blockedV(wall.openings, u - studW / 2, u + studW / 2, 0.04))) {
          if (c - a < 0.14) continue;
          wallBox(b, wall, 'timber', u, (a + c) / 2, nMid, studW, c - a, TIMBER_T, timberTint());
        }
      }
      // jamb studs + lintels around every opening
      for (const o of wall.openings) {
        for (const u of [o.u0 - studW / 2 - 0.01, o.u1 + studW / 2 + 0.01]) {
          for (const [a, c] of freeIntervals(0, H, blockedV(wall.openings, u - studW / 2, u + studW / 2, 0.02))) {
            if (c - a < 0.14) continue;
            wallBox(b, wall, 'timber', u, (a + c) / 2, nMid, studW, c - a, TIMBER_T, timberTint());
          }
        }
        const headV = o.kind === 'door' ? o.v1 : o.v1;
        if (headV + 0.2 < H) {
          wallBox(b, wall, 'timber', (o.u0 + o.u1) / 2, headV + 0.09, nMid,
            (o.u1 - o.u0) + 2 * studW + 0.04, 0.18, TIMBER_T, timberTint());
        }
        if (o.v0 > 0.3) {
          wallBox(b, wall, 'timber', (o.u0 + o.u1) / 2, o.v0 - 0.075, nMid,
            (o.u1 - o.u0) + 2 * studW + 0.04, 0.15, TIMBER_T, timberTint());
        }
      }
      // corner braces
      const braceRun = Math.min(1.5, L * 0.22), braceRise = Math.min(H - plateH - 0.3, 1.6);
      for (const sgn of [-1, 1]) {
        const u0 = sgn * (half - postW), v0 = plateH + 0.05;
        const u1 = u0 - sgn * braceRun, v1 = v0 + braceRise;
        const mu = (u0 + u1) / 2, mv = (v0 + v1) / 2;
        let clash = false;
        for (const o of wall.openings) {
          if (mu > o.u0 - 0.2 && mu < o.u1 + 0.2 && mv > o.v0 - 0.2 && mv < o.v1 + 0.2) clash = true;
        }
        if (clash) continue;
        const len = Math.hypot(u1 - u0, v1 - v0);
        wallBox(b, wall, 'timber', mu, mv, nMid, len + 0.1, 0.16, TIMBER_T * 0.92,
          timberTint(), Math.atan2(v1 - v0, u1 - u0));
      }
    } else {
      // unframed: still give the eye a head plate + corner boards
      wallBox(b, wall, 'timber', 0, wall.H - 0.11, -0.07, wall.L, 0.20, TIMBER_T * 0.9, timberTint());
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Jetty                                                                   */
  /* ---------------------------------------------------------------------- */
  if (jet > 0) {
    const upper = rects[1];
    const sides = [
      { side: 'front', L: upper.x1 - upper.x0, m: makeWall(upper, 'front').m, inner: jet },
      { side: 'left', L: upper.z1 - upper.z0, m: makeWall(upper, 'left').m, inner: jet },
      { side: 'right', L: upper.z1 - upper.z0, m: makeWall(upper, 'right').m, inner: jet },
    ];
    for (const s of sides) {
      const w = { m: s.m, L: s.L, ox: 0, oz: 0, ux: 0, uz: 0, nx: 0, nz: 0, y0: 0 };
      // bressumer: moulded beam under the overhang
      wallBox(b, w, 'timber', 0, 0.02, -0.16, s.L, 0.34, 0.32, timberTint());
      wallBox(b, w, 'timber', 0, -0.17, -0.14, s.L, 0.10, 0.36, timberTint());
      // soffit board seals the underside of the overhang
      wallBox(b, w, 'timber', 0, -0.21, -jet / 2 - 0.14, s.L, 0.06, jet + 0.02, { r: 0.86, g: 0.85, b: 0.83 });
      // exposed joist ends
      const n = Math.max(2, Math.round(s.L / 0.62 * detail));
      for (let i = 0; i < n; i++) {
        const u = -s.L / 2 + (s.L * (i + 0.5)) / n;
        wallBox(b, w, 'timber', u, -0.30, -jet / 2 - 0.14, 0.14, 0.17, jet + 0.06, timberTint());
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Windows + doors (after the walls, so they sit in the holes)             */
  /* ---------------------------------------------------------------------- */

  for (const { wall, win } of winList) buildWindow(b, wall, win, rng, quality, plot);
  buildDoor(b, frontGround, door, rng, quality, irng);
  // The leaf is a rigid body, not part of the merged shell. Its collider and its
  // threshold live in the collider section below.
  buildDoorLeaves(plot, door, frontGround, ctx, irng, rot, toWorld);

  /* flower boxes + light anchors ----------------------------------------- */
  let glowsEmitted = 0;
  for (const { wall, win } of winList) {
    if (wall.side !== 'front' || wall.storey !== 0) continue;
    const cu = (win.u0 + win.u1) / 2;
    if (win.box) {
      const bw2 = (win.u1 - win.u0) + 0.30;
      const bh = 0.26, bd = 0.30;
      const v = win.v0 - 0.20;
      const faceN = (win.faceN ?? 0) + bd / 2 + 0.01;
      // timber box: four sides + base, so it is a real trough
      wallBox(b, wall, 'woodPlank', cu, v, faceN + bd / 2 - 0.025, bw2, bh, 0.05, tintOf(rng, 0.09));
      wallBox(b, wall, 'woodPlank', cu, v - bh / 2 + 0.025, faceN, bw2, 0.05, bd, tintOf(rng, 0.09));
      for (const sgn of [-1, 1]) {
        wallBox(b, wall, 'woodPlank', cu + sgn * (bw2 / 2 - 0.025), v, faceN, 0.05, bh, bd, tintOf(rng, 0.09));
      }
      wallBox(b, wall, 'woodPlank', cu, v, faceN - bd / 2 + 0.025, bw2, bh, 0.05, tintOf(rng, 0.09));
      // soil, sunk a little below the rim
      wallBox(b, wall, 'soil', cu, v + bh / 2 - 0.075, faceN, bw2 - 0.10, 0.10, bd - 0.10,
        { r: 0.9, g: 0.88, b: 0.86 });
      // iron brackets under it
      for (const sgn of [-1, 1]) {
        wallBox(b, wall, 'iron', cu + sgn * (bw2 / 2 - 0.10), v - bh / 2 - 0.10, faceN - bd / 4,
          0.04, 0.22, 0.04, { r: 0.8, g: 0.8, b: 0.82 });
      }
      // the box sticks out further than the wall collider, so it gets its own
      wallPoint(wall, cu, v, faceN, _v);
      const isEnd = wall.side === 'left' || wall.side === 'right';
      pushCollider(_v.x, _v.y, _v.z,
        isEnd ? bd / 2 + 0.03 : bw2 / 2, bh / 2 + 0.02,
        isEnd ? bw2 / 2 : bd / 2 + 0.03, 'building-flowerbox');

      wallPoint(wall, cu, v + bh / 2 - 0.03, faceN, _v);
      flowerBoxAnchors.push({
        position: toWorld(_v.x, _v.y, _v.z),
        width: bw2 - 0.12,
        depth: bd - 0.12,
        rotation: rot,
        normal: [wall.nx * cosR + wall.nz * sinR, 0, -wall.nx * sinR + wall.nz * cosR],
        plot: plot.id,
      });
    }
    if (win.glow && glowsEmitted < 2) {
      glowsEmitted++;
      wallPoint(wall, cu, (win.v0 + win.v1) / 2, (win.faceN ?? 0) - 0.06, _v);
      const p = toWorld(_v.x, _v.y, _v.z);
      lightAnchors.push({
        position: p, kind: 'window', color: 0xffb257, colour: 0xffb257, intensity: 1.5, radius: 6.5, plot: plot.id,
      });
    }
  }

  /* wall lantern beside the door ----------------------------------------- */
  {
    const side = door.u > 0 ? -1 : 1;
    const lu = THREE.MathUtils.clamp(door.u + side * (dw / 2 + 0.62), -hw + 0.5, hw - 0.5);
    wallPoint(frontGround, lu, 2.42, (frontGround.faceN ?? 0) + 0.30, _v);
    lightAnchors.push({
      position: toWorld(_v.x, _v.y, _v.z), kind: 'lantern',
      color: 0xffc07a, colour: 0xffc07a, intensity: 2.2, radius: 8, plot: plot.id,
      rotation: rot,
    });
  }

  /* ---------------------------------------------------------------------- */
  /* Steps                                                                   */
  /* ---------------------------------------------------------------------- */
  if (stepCount > 0) {
    const n = stepCount;
    const sw = plot.steps.width;
    const riseH = stepRise, run = 0.36;
    const z0 = hd + (frontGround.faceN ?? 0);
    for (let i = 0; i < n; i++) {
      const y = (n - i) * riseH;
      const depth = run * (i + 1) + 0.22;
      const zc = z0 + depth / 2 - 0.10;
      const wI = sw + i * 0.10;
      localBox(b, 'stone', door.u, y - riseH / 2 - 0.05, zc, wI, riseH + 0.16, depth, tintOf(rng, 0.09));
      pushCollider(door.u, Math.max(0.02, y - riseH / 2 - 0.02), zc, wI / 2, (riseH + 0.10) / 2, depth / 2, 'building-step');
      // worn nosing
      localBox(b, 'stone', door.u, y - 0.012, zc + depth / 2 - 0.05, wI - 0.16, 0.035, 0.10,
        { r: 1.1, g: 1.09, b: 1.07 });
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Porch / arcade                                                          */
  /* ---------------------------------------------------------------------- */
  const canopies = [];
  /**
   * A lean-to canopy must stay under the main thatch wherever they overlap,
   * otherwise the big eave saws straight through it. Returns null if it cannot.
   */
  const addCanopy = (cx, cw, zIn, zOut, headY) => {
    const under = (lz) => roofYAt(cx, THREE.MathUtils.clamp(lz, ze0, ze1)) - thatchVert - 0.14;
    let yIn = headY + 0.42, yOut = headY + 0.14;
    yIn = Math.min(yIn, under(zIn));
    const zTest = Math.min(zOut, ze1);
    const span = Math.max(0.01, zOut - zIn);
    const yTest = yIn + (yOut - yIn) * ((zTest - zIn) / span);
    const over = yTest - under(zTest);
    if (over > 0) yOut -= over * (span / Math.max(0.01, zTest - zIn));
    if (yOut < headY + 0.04 || yIn < headY + 0.10) return;   // no room: the main thatch IS the roof
    canopies.push({ x: cx, w: cw, zIn, zOut, yIn, yOut });
  };

  if (plot.porch) {
    const p = plot.porch;
    const zFace = hd + (frontGround.faceN ?? 0);
    const zOut = zFace + p.depth;
    for (const sgn of [-1, 1]) {
      const x = door.u + sgn * (p.width / 2 - 0.09);
      localBox(b, 'timber', x, p.height / 2, zOut - 0.10, 0.17, p.height, 0.17, timberTint());
      pushCollider(x, p.height / 2, zOut - 0.10, 0.13, p.height / 2, 0.13, 'building-post');
      // brace back to the wall
      const len = Math.hypot(p.depth - 0.25, 0.55);
      const g = new THREE.BoxGeometry(0.11, 0.11, len);
      g.rotateX(-Math.atan2(0.55, p.depth - 0.25));
      g.translate(x, p.height - 0.32, (zFace + zOut) / 2 - 0.05);
      b.add('timber', g, timberTint());
    }
    localBox(b, 'timber', door.u, p.height + 0.10, zOut - 0.10, p.width + 0.22, 0.20, 0.20, timberTint());
    localBox(b, 'timber', door.u, p.height + 0.10, zFace + 0.06, p.width + 0.22, 0.20, 0.16, timberTint());
    addCanopy(door.u, p.width + 0.55, hd + jet - 0.06, zOut + 0.30, p.height);
  }
  if (plot.arcade) {
    const a = plot.arcade;
    const zFace = hd + (frontGround.faceN ?? 0);
    const zOut = zFace + a.depth;
    const bays = Math.max(2, a.bays | 0);
    const hCol = Math.min(sh - 0.42, 2.62);
    const span = plot.width - 1.2;
    for (let i = 0; i <= bays; i++) {
      const x = -span / 2 + (span * i) / bays;
      localBox(b, 'timber', x, hCol / 2, zOut - 0.12, 0.19, hCol, 0.19, timberTint());
      pushCollider(x, hCol / 2, zOut - 0.12, 0.15, hCol / 2, 0.15, 'building-post');
      // arched braces to the head beam
      for (const sgn of [-1, 1]) {
        if ((i === 0 && sgn < 0) || (i === bays && sgn > 0)) continue;
        const len = 0.86;
        const g = new THREE.BoxGeometry(len, 0.11, 0.13);
        g.rotateZ(sgn * Math.PI / 4);
        g.translate(x + sgn * 0.31, hCol - 0.31, zOut - 0.12);
        b.add('timber', g, timberTint());
      }
    }
    localBox(b, 'timber', 0, hCol + 0.12, zOut - 0.12, span + 0.5, 0.24, 0.22, timberTint());
    addCanopy(0, plot.width + 0.3, hd + jet - 0.06, zOut + 0.35, hCol);
  }

  /* ---------------------------------------------------------------------- */
  /* Roof                                                                    */
  /* ---------------------------------------------------------------------- */

  // Per-plot straw colour: every roof in the village sits somewhere between
  // fresh straw-gold and grey-brown weathered reed, and the older it is the more
  // it carries dark streaking down the slope. Forked off `plot.seed` so adding a
  // draw anywhere else in this function cannot shift the palette.
  const thRng = rng.fork('thatch');
  // A plain rng.next() per plot clumps — eight of the eleven landed in the same
  // half last time, which is how you get a village of identically drab roofs.
  // Golden-ratio hashing of the seed spreads them evenly, jittered by the Rng.
  const age = THREE.MathUtils.clamp(
    (plot.seed * 0.6180339887498949) % 1 + thRng.sym(0.07), 0, 1);
  const thatchCol = {
    r: THREE.MathUtils.lerp(1.20, 0.80, age),   // straw gold -> weathered grey-brown
    g: THREE.MathUtils.lerp(1.04, 0.79, age),
    b: THREE.MathUtils.lerp(0.70, 0.80, age),
  };
  const weather = Math.pow(age, 0.75);
  /** Flat tint for the pieces that are not swept patches (rolls, skirts). */
  const roofTint = (k = 1) => {
    if (tiled) return tintOf(rng, 0.05, 0.012);
    const j = k * (1 + thRng.sym(0.035));
    return { r: thatchCol.r * j, g: thatchCol.g * j, b: thatchCol.b * j };
  };
  // the thatch carries the silhouette of this whole village — spend triangles here
  const thatchQ = detail * (0.72 + 0.14 * THREE.MathUtils.clamp(thatchDetail, 0, 2));
  const faces = [];

  const mkPatch = (eaveFn, topFn, sSegs, seed) => thatchPatch(eaveFn, topFn, {
    sSegs, detail: thatchQ, thickness: THATCH_T, sag: THATCH_SAG,
    wave: tiled ? 0.010 : 0.090, swell: tiled ? 0.02 : THATCH_T * 0.26,
    bell: tiled ? bellN * 0.15 : bellN, bellD, seed,
    color: tiled ? { r: 1, g: 1, b: 1 } : thatchCol, weather: tiled ? 0 : weather,
  });
  /** Roll a swept patch's end over into a rounded verge. */
  const rollEnd = (patch, endIdx, outward, r) => {
    const g = vergeRoll(patch.rings[endIdx], outward, r, Math.max(3, Math.round(4 * thatchQ)));
    if (g) b.add(roofKey, g, roofTint(0.98));
  };

  const eaveLen = xe1 - xe0;
  const segsMain = Math.max(10, Math.round(eaveLen / 0.29));
  // the verge roll adds its own overhang, so give the raked edge back a little
  const vergeRollR = THREE.MathUtils.clamp(THATCH_T * 0.5, 0.11, 0.23);

  // front + back slopes
  const topFnFor = (zr) => (s) => {
    // full ridge point at this s
    const x = xe0 + (xe1 - xe0) * s;
    const full = new THREE.Vector3(x, yRidge, zRidge);
    if (hipRun <= 0.01) return full;
    const eaveP = new THREE.Vector3(x, yEaveTip, zr);
    let tt = 1;
    if (hipped) {
      // trapezoid: ridge shortened, slanted verges are the hip lines
      const xr = xh0 + (xh1 - xh0) * s;
      return new THREE.Vector3(xr, yRidge, zRidge);
    }
    const dEnd = Math.min(x - xe0, xe1 - x);
    if (dEnd < hipRun) tt = tGablet + (1 - tGablet) * (dEnd / hipRun);
    return eaveP.clone().lerp(full, tt);
  };

  for (const dir of [1, -1]) {
    const zEdge = dir > 0 ? ze1 : ze0;
    const eaveFn = (s) => new THREE.Vector3(
      dir > 0 ? xe0 + (xe1 - xe0) * s : xe1 - (xe1 - xe0) * s, yEaveTip, zEdge);
    const inner = topFnFor(zEdge);
    const topFn = dir > 0 ? inner : (s) => inner(1 - s);
    const patch = mkPatch(eaveFn, topFn, segsMain, plot.seed * 0.001 + (dir > 0 ? 0 : 7.3));
    b.add(roofKey, patch.geometry, null, true);   // the patch paints its own straw
    faces.push({ dir, patch, zEdge });
  }

  // hip / gable ends
  if (hipped || halfHip) {
    for (const sgn of [-1, 1]) {
      const xEdge = sgn < 0 ? xe0 : xe1;
      const xTip = sgn < 0 ? xh0 : xh1;
      const yBase = hipped ? yEaveTip : gabletY;
      // the hip face springs from the eave (full hip) or the gablet line (half hip)
      const zSpan = hipped ? [ze1, ze0] : (() => {
        const f = (gabletY - yEaveTip) / Math.max(0.01, yRidge - yEaveTip);
        return [ze1 + (zRidge - ze1) * f, ze0 + (zRidge - ze0) * f];
      })();
      const eaveFn = (s) => new THREE.Vector3(xEdge, yBase,
        sgn < 0 ? zSpan[0] + (zSpan[1] - zSpan[0]) * s : zSpan[1] + (zSpan[0] - zSpan[1]) * s);
      const tipP = new THREE.Vector3(xTip, yRidge, zRidge);
      const topFn = () => tipP;
      const patch = mkPatch(eaveFn, topFn, Math.max(6, Math.round((ze1 - ze0) / 0.40)), plot.seed * 0.001 + sgn * 3.1);
      b.add(roofKey, patch.geometry, null, true);
      // hip rolls hide the seam between the hip face and the main slopes
      for (const zz of zSpan) {
        const pts = [];
        const nSteps = 8;
        for (let i = 0; i <= nSteps; i++) {
          const t = i / nSteps;
          pts.push(new THREE.Vector3(
            xEdge + (xTip - xEdge) * t,
            yBase + (yRidge - yBase) * t + 0.03,
            zz + (zRidge - zz) * t,
          ));
        }
        const roll = tubeAlong(pts, 0.145, 6);
        if (roll) b.add('thatchRidge', roll, roofTint());
        // Where the hip lip roll meets the eave lip roll of the main slope the
        // two curl away from each other and leave a notch at the corner. Real
        // thatch is bunched into a rounded lump there; so is this.
        if (hipped) {
          const lift = bellN * cosP;
          const knob = new THREE.SphereGeometry(0.42, 9, 7);
          knob.translate(
            xEdge - sgn * 0.06,
            yEaveTip + lift - THATCH_T * 0.46,
            zz + (zRidge > zz ? 0.06 : -0.06),
          );
          b.add(roofKey, knob, roofTint(0.99));
        }
      }
    }
  }

  // Gable verges. Each slope's raked edge rolls over onto itself, so from below
  // the verge is a fat half-round of straw wrapping over the gable wall — never
  // the flat card edge a capped end-section gives you.
  if (!hipped) {
    const [fFace, bFace] = faces;
    const LEFT = new THREE.Vector3(-1, 0, 0), RIGHT = new THREE.Vector3(1, 0, 0);
    rollEnd(fFace.patch, 0, LEFT, vergeRollR);
    rollEnd(fFace.patch, fFace.patch.sSegs, RIGHT, vergeRollR);
    rollEnd(bFace.patch, 0, RIGHT, vergeRollR);
    rollEnd(bFace.patch, bFace.patch.sSegs, LEFT, vergeRollR);
  }

  /* ridge roll ------------------------------------------------------------ */
  if (!tiled) {
    // on a gable the ridge cap runs out over the rolled verge, so it never stops
    // short of the silhouette; on a hip it stops where the hip rolls take over
    const rx0 = hipRun > 0.01 ? xh0 : xe0 - vergeRollR * 0.75;
    const rx1 = hipRun > 0.01 ? xh1 : xe1 + vergeRollR * 0.75;
    const segs = Math.max(4, Math.round((rx1 - rx0) / 0.42));
    const K = 11;
    const pos = [];
    const skirt = 0.62;
    const slopeF = new THREE.Vector3(0, yEaveTip - yRidge, ze1 - zRidge).normalize();
    const slopeB = new THREE.Vector3(0, yEaveTip - yRidge, ze0 - zRidge).normalize();
    // face normals of the two slopes (the real slope, after the eave kick)
    const nF = new THREE.Vector3(0, slopeF.z, -slopeF.y).normalize();
    const nB = new THREE.Vector3(0, slopeB.z, -slopeB.y).normalize();
    if (nF.y < 0) nF.negate();
    if (nB.y < 0) nB.negate();
    const ringPts = (x, sag, off) => {
      const c = new THREE.Vector3(x, yRidge - sag, zRidge);
      const sk = skirt * (0.78 + 0.28 * Math.abs(Math.sin((x - rx0) * 2.4)));   // scalloped edge
      const out = [];
      for (let j = 0; j < K; j++) {
        const t = j / (K - 1);
        if (t < 0.34) {
          const f = t / 0.34;
          out.push(c.clone().addScaledVector(slopeB, sk * (1 - f)).addScaledVector(nB, off));
        } else if (t > 0.66) {
          const f = (t - 0.66) / 0.34;
          out.push(c.clone().addScaledVector(slopeF, sk * f).addScaledVector(nF, off));
        } else {
          const a = ((t - 0.34) / 0.32) * Math.PI;
          out.push(new THREE.Vector3(
            x,
            c.y + Math.sin(a) * (0.30 + off) - 0.04,
            c.z - Math.cos(a) * (0.26 + off),
          ));
        }
      }
      return out;
    };
    const rows = [];
    for (let i = 0; i <= segs; i++) {
      const x = rx0 + ((rx1 - rx0) * i) / segs;
      // same droop law as the two slopes, so the cap sits on the sagging ridge
      const sag = THATCH_SAG * Math.sin(Math.PI * THREE.MathUtils.clamp(
        (x - xe0) / Math.max(0.01, xe1 - xe0), 0, 1));
      rows.push([ringPts(x, sag, 0.125), ringPts(x, sag, 0.012)]);
    }
    // outer shell + inner shell -> a closed roll
    for (let i = 0; i < segs; i++) {
      for (const layer of [0, 1]) {
        const A = rows[i][layer], B = rows[i + 1][layer];
        for (let j = 0; j < K - 1; j++) {
          if (layer === 0) {
            pos.push(A[j].x, A[j].y, A[j].z, B[j].x, B[j].y, B[j].z, A[j + 1].x, A[j + 1].y, A[j + 1].z);
            pos.push(A[j + 1].x, A[j + 1].y, A[j + 1].z, B[j].x, B[j].y, B[j].z, B[j + 1].x, B[j + 1].y, B[j + 1].z);
          } else {
            pos.push(A[j].x, A[j].y, A[j].z, A[j + 1].x, A[j + 1].y, A[j + 1].z, B[j].x, B[j].y, B[j].z);
            pos.push(A[j + 1].x, A[j + 1].y, A[j + 1].z, B[j + 1].x, B[j + 1].y, B[j + 1].z, B[j].x, B[j].y, B[j].z);
          }
        }
      }
      // close the two skirt edges
      for (const j of [0, K - 1]) {
        const a = rows[i][0][j], c = rows[i][1][j], d = rows[i + 1][0][j], e = rows[i + 1][1][j];
        if (j === 0) { pos.push(a.x, a.y, a.z, c.x, c.y, c.z, d.x, d.y, d.z, d.x, d.y, d.z, c.x, c.y, c.z, e.x, e.y, e.z); }
        else { pos.push(a.x, a.y, a.z, d.x, d.y, d.z, c.x, c.y, c.z, d.x, d.y, d.z, e.x, e.y, e.z, c.x, c.y, c.z); }
      }
    }
    // end caps
    for (const end of [0, segs]) {
      const A = rows[end][0], B = rows[end][1];
      for (let j = 0; j < K - 1; j++) {
        if (end === 0) { pos.push(A[j].x, A[j].y, A[j].z, A[j + 1].x, A[j + 1].y, A[j + 1].z, B[j].x, B[j].y, B[j].z, A[j + 1].x, A[j + 1].y, A[j + 1].z, B[j + 1].x, B[j + 1].y, B[j + 1].z, B[j].x, B[j].y, B[j].z); }
        else { pos.push(A[j].x, A[j].y, A[j].z, B[j].x, B[j].y, B[j].z, A[j + 1].x, A[j + 1].y, A[j + 1].z, A[j + 1].x, A[j + 1].y, A[j + 1].z, B[j].x, B[j].y, B[j].z, B[j + 1].x, B[j + 1].y, B[j + 1].z); }
      }
    }
    const rg = new THREE.BufferGeometry();
    rg.setAttribute('position', new THREE.Float32BufferAttribute(orientClosed(pos), 3));
    rg.computeVertexNormals();
    b.add('thatchRidge', rg, roofTint(0.94));

    // liggers: hazel rods + cross spars pinning the ridge
    if (thatchDetail >= 1) {
      for (const sgn of [-1, 1]) {
        const nrm2 = sgn > 0 ? nF : nB;
        const slope = sgn > 0 ? slopeF : slopeB;
        for (const f of [0.42, 0.86]) {
          const pts = [];
          for (let i = 0; i <= segs; i++) {
            const x = rx0 + ((rx1 - rx0) * i) / segs;
            const sag = THATCH_SAG * Math.sin(Math.PI * THREE.MathUtils.clamp(
              (x - xe0) / Math.max(0.01, xe1 - xe0), 0, 1));
            pts.push(new THREE.Vector3(x, yRidge - sag, zRidge)
              .addScaledVector(slope, skirt * f).addScaledVector(nrm2, 0.155));
          }
          const rod = tubeAlong(pts, 0.028, 5);
          if (rod) b.add('timber', rod, { r: 0.92, g: 0.88, b: 0.78 });
        }
        // zig-zag cross spars
        const nSp = Math.max(3, Math.round((rx1 - rx0) / 0.52));
        for (let i = 0; i < nSp; i++) {
          const x0 = rx0 + ((rx1 - rx0) * i) / nSp;
          const x1 = rx0 + ((rx1 - rx0) * (i + 1)) / nSp;
          const a = new THREE.Vector3(x0, yRidge, zRidge).addScaledVector(slope, skirt * 0.42).addScaledVector(nrm2, 0.16);
          const c = new THREE.Vector3(x1, yRidge, zRidge).addScaledVector(slope, skirt * 0.86).addScaledVector(nrm2, 0.16);
          const sp = tubeAlong([a, c], 0.022, 4);
          if (sp) b.add('timber', sp, { r: 0.9, g: 0.86, b: 0.76 });
        }
      }
    }
  }

  /* gable end walls ------------------------------------------------------- */
  if (!hipped) {
    for (const sgn of [-1, 1]) {
      const side = sgn < 0 ? 'left' : 'right';
      const rect = { x0: top.x0, x1: top.x1, z0: top.z0, z1: top.z1, y0: wallTop, y1: wallTop + 1 };
      const gw = makeWall(rect, side);
      // match the storey wall below so the plaster does not step at the eaves plate
      gw.faceN = (plot.style === 'halfTimber' || plot.style === 'longRow') ? -TIMBER_INSET : 0;
      const capV = halfHip ? gabletY - wallTop : Infinity;
      // outline: base line, then follow the underside of the thatch to the apex
      const pts = [];
      const zc = (top.z0 + top.z1) / 2;
      const uOf = (z) => (side === 'right' ? -(z - zc) : (z - zc));
      const under = (z) => {
        // +0.30 so the gable pokes into the thatch: the waviness and the
        // bell-cast bulge both lift the real underside above the ideal plane
        const y = roofYAt(sgn < 0 ? top.x0 + 0.05 : top.x1 - 0.05, z) - THATCH_T / cosP + 0.30;
        return y - wallTop;
      };
      const zA = top.z1, zB = top.z0;
      const N = Math.max(6, Math.round(10 * detail));
      const raw = [];
      for (let i = 0; i <= N; i++) {
        const z = zA + (zB - zA) * (i / N);
        raw.push([uOf(z), Math.max(0.05, Math.min(under(z), capV))]);
      }
      const shape = new THREE.Shape();
      shape.moveTo(uOf(zA), 0);
      shape.lineTo(uOf(zB), 0);
      for (let i = raw.length - 1; i >= 0; i--) shape.lineTo(raw[i][0], raw[i][1]);
      shape.lineTo(uOf(zA), 0);
      // ensure CCW so ExtrudeGeometry normalises the winding for us
      const g = new THREE.ExtrudeGeometry(shape, { depth: WALL_T, bevelEnabled: false, steps: 1, curveSegments: 3 });
      g.translate(0, 0, (gw.faceN ?? 0) - WALL_T);
      g.applyMatrix4(gw.m);
      b.add(plasterKey, g, plasterTint());

      // decorative gable framing
      if (plot.style === 'halfTimber' || plot.style === 'longRow') {
        const apex = Math.min(under(zc), capV);
        wallBox(b, gw, 'timber', 0, apex / 2, -TIMBER_T / 2, 0.17, apex, TIMBER_T, timberTint());
        const collar = apex * 0.46;
        const halfAt = (v) => {
          // horizontal half-width of the gable at height v
          let lo = 0, hiU = Math.abs(uOf(zA));
          for (let i = 0; i < 14; i++) {
            const mid = (lo + hiU) / 2;
            const z = side === 'right' ? zc - mid : zc + mid;
            if (Math.min(under(z), capV) > v) lo = mid; else hiU = mid;
          }
          return lo;
        };
        const cw = halfAt(collar);
        wallBox(b, gw, 'timber', 0, collar, -TIMBER_T / 2, cw * 2, 0.16, TIMBER_T, timberTint());
        for (const s2 of [-1, 1]) {
          const len = Math.hypot(cw * 0.9, collar * 0.85);
          wallBox(b, gw, 'timber', s2 * cw * 0.48, collar * 0.52, -TIMBER_T / 2, len, 0.13, TIMBER_T * 0.9,
            timberTint(), s2 * Math.atan2(collar * 0.85, cw * 0.9));
        }
        wallBox(b, gw, 'timber', 0, 0.10, -TIMBER_T / 2, Math.abs(uOf(zA)) * 2, 0.20, TIMBER_T, timberTint());
      }
    }
  }

  /* dormers --------------------------------------------------------------- */
  for (const dm of (plot.dormers || [])) {
    const dx = THREE.MathUtils.clamp(dm.x, top.x0 + 1.2, top.x1 - 1.2);
    const w2 = (dm.width || 2.2) / 2;
    // sit the dormer on the front slope
    const t0 = 0.28, t1 = 0.28 + 0.34;      // fraction of the slope it occupies
    const zAt = (t) => ze1 + (zRidge - ze1) * t;
    const yAt = (t) => yEaveTip + (yRidge - yEaveTip) * t;
    // the little vertical cheek + window, tucked under the hood: its top and
    // sides sit inside the hood's thickness, so only the glazing shows through
    const tC = t0 + 0.09;
    const zf = zAt(tC), yf = yAt(tC) - 0.45;
    const cheekH = 1.05;
    const wRect = { x0: dx - w2 * 0.62, x1: dx + w2 * 0.62, z0: zf - 0.4, z1: zf, y0: yf, y1: yf + cheekH };
    const dw2 = makeWall(wRect, 'front');
    dw2.faceN = 0;
    const wWin = Math.min(w2 * 1.05, 1.02);
    const shape = new THREE.Shape();
    const hl = dw2.L / 2, Hh = dw2.H;
    shape.moveTo(-hl, 0); shape.lineTo(hl, 0); shape.lineTo(hl, Hh); shape.lineTo(-hl, Hh); shape.lineTo(-hl, 0);
    const hp = new THREE.Path();
    const wv0 = 0.47, wv1 = Math.min(Hh - 0.05, 1.01);
    hp.moveTo(-wWin / 2, wv0); hp.lineTo(wWin / 2, wv0); hp.lineTo(wWin / 2, wv1); hp.lineTo(-wWin / 2, wv1); hp.lineTo(-wWin / 2, wv0);
    shape.holes.push(hp);
    const cg = new THREE.ExtrudeGeometry(shape, { depth: 0.30, bevelEnabled: false, steps: 1, curveSegments: 3 });
    cg.translate(0, 0, -0.30);
    cg.applyMatrix4(dw2.m);
    b.add(plasterKey, cg, plasterTint());
    buildWindow(b, dw2, { u0: -wWin / 2, u1: wWin / 2, v0: wv0, v1: wv1, faceN: 0, shutters: false }, rng, quality, plot);
    // cheek returns, so the sides of the dormer are walled rather than open
    for (const sgn of [-1, 1]) {
      localBox(b, plasterKey, dx + sgn * (w2 * 0.62 - 0.07), yf + cheekH / 2, zf - 0.36,
        0.14, cheekH, 0.72, plasterTint());
    }

    /* The hood: its own swept thatch, with the same bulging lip as the main
     * eaves, springing forward over the cheek and dying into the main slope
     * upstream — the underside of its top edge sits inside the main roof, which
     * is what blends the two surfaces instead of pasting a panel on top. */
    const hoodT = Math.min(0.32, THATCH_T * 0.70);
    const halfW = w2 * 0.62 + 0.34;
    const zLip = zf + 0.34;
    const yLip = yf + cheekH + hoodT * 1.6;
    const zTop = zAt(t1);
    const hoodEave = (s) => new THREE.Vector3(dx - halfW + 2 * halfW * s, yLip, zLip);
    const hoodTop = (s) => {
      const x = dx - halfW + 2 * halfW * s;
      return new THREE.Vector3(x, roofYAt(x, zTop) + 0.04, zTop);
    };
    const hood = thatchPatch(hoodEave, hoodTop, {
      sSegs: Math.max(5, Math.round((2 * halfW) / 0.26)), detail: thatchQ,
      thickness: hoodT, sag: 0.02, wave: tiled ? 0.008 : 0.045,
      swell: tiled ? 0.01 : hoodT * 0.30, bell: 0.09, bellD: 0.42,
      seed: plot.seed * 0.001 + dx * 0.37 + 5.9,
      color: tiled ? { r: 1, g: 1, b: 1 } : thatchCol, weather: tiled ? 0 : weather,
    });
    b.add(roofKey, hood.geometry, null, true);
    const hoodR = Math.max(0.09, hoodT * 0.5);
    rollEnd(hood, 0, new THREE.Vector3(-1, 0, 0), hoodR);
    rollEnd(hood, hood.sSegs, new THREE.Vector3(1, 0, 0), hoodR);
  }

  /* chimneys -------------------------------------------------------------- */
  for (const ch of (plot.chimneys || [])) {
    const cx = THREE.MathUtils.clamp(ch.x, top.x0 + 0.9, top.x1 - 0.9);
    const cz = THREE.MathUtils.clamp(ch.z ?? 0, top.z0 + 0.7, top.z1 - 0.7);
    const cw = ch.width || 1.2;
    const cd = cw * 0.74;
    const rY = roofYAt(cx, cz);
    const desired = wallTop + (ch.height || 4);
    const topY = rY + THREE.MathUtils.clamp(desired - rY, 1.0, 2.9);
    const bot = Math.min(wallTop - 1.2, rY - 1.6);

    // shaft, built from a few courses so the silhouette is not a single prism
    const courses = 4;
    for (let i = 0; i < courses; i++) {
      const y0 = bot + ((topY - 0.42 - bot) * i) / courses;
      const y1 = bot + ((topY - 0.42 - bot) * (i + 1)) / courses;
      const inset = i === 0 ? 0 : rng.range(-0.012, 0.012);
      localBox(b, 'brick', cx, (y0 + y1) / 2, cz, cw + inset, y1 - y0 + 0.01, cd + inset, tintOf(rng, 0.07, 0.02));
    }
    // corbelled cap
    for (let i = 0; i < 3; i++) {
      const e = 0.055 * (i + 1);
      localBox(b, 'brick', cx, topY - 0.40 + 0.075 + i * 0.10, cz, cw + 2 * e, 0.105, cd + 2 * e, tintOf(rng, 0.06, 0.02));
    }
    localBox(b, 'brick', cx, topY - 0.045, cz, cw + 0.36, 0.09, cd + 0.36, { r: 1.06, g: 1.02, b: 0.99 });

    // clay pots
    const pots = Math.max(1, ch.pots | 0);
    for (let i = 0; i < pots; i++) {
      const t = pots === 1 ? 0 : (i / (pots - 1)) * 2 - 1;
      const px = cx + t * (cw * 0.5 - 0.20);
      const ph = rng.range(0.46, 0.62);
      const pot = new THREE.CylinderGeometry(0.125, 0.155, ph, Math.max(6, Math.round(10 * detail)), 1, true);
      pot.translate(px, topY + ph / 2, cz);
      b.add('terracotta', pot, tintOf(rng, 0.08, 0.02));
      const rim = new THREE.TorusGeometry(0.132, 0.028, 4, Math.max(6, Math.round(10 * detail)));
      rim.rotateX(Math.PI / 2);
      rim.translate(px, topY + ph, cz);
      b.add('terracotta', rim, tintOf(rng, 0.08, 0.02));
    }

    // thatch swept up against the stack: a flared collar from the roof surface
    // to a lip part-way up the brickwork, so the stack never just intersects
    if (!tiled) {
      const ringN = Math.max(10, Math.round(18 * detail));
      const skirtPos = [];
      const halfW = cw / 2, halfD = cd / 2;
      const outer = 0.48, rise = 0.55;
      /** point on the rounded-rectangle ring at `ang`, pushed out `ex`, raised `ey` */
      const pt = (ang, ex, ey) => {
        const sx = Math.cos(ang), sz = Math.sin(ang);
        const k = Math.max(Math.abs(sx) / halfW, Math.abs(sz) / halfD);
        const rx = sx / k, rz = sz / k;
        const len = Math.hypot(rx, rz) || 1;
        return new THREE.Vector3(
          cx + rx + (rx / len) * ex,
          roofYAt(cx + rx + (rx / len) * ex, cz + rz + (rz / len) * ex) + ey,
          cz + rz + (rz / len) * ex,
        );
      };
      const skirtRef = new THREE.Vector3();
      for (let i = 0; i < ringN; i++) {
        const a0 = (i / ringN) * Math.PI * 2, a1 = ((i + 1) / ringN) * Math.PI * 2;
        const A1 = pt(a0, -0.02, rise), B1 = pt(a1, -0.02, rise);
        const A2 = pt(a0, outer, -0.10), B2 = pt(a1, outer, -0.10);
        // the collar is an open band: its visible face is the top, tipping
        // outward away from the stack
        skirtRef.set(Math.cos(a0) * 0.35, 1, Math.sin(a0) * 0.35).normalize();
        pushTri(skirtPos, A1, A2, B1, skirtRef);
        pushTri(skirtPos, B1, A2, B2, skirtRef);
      }
      const sg = new THREE.BufferGeometry();
      sg.setAttribute('position', new THREE.Float32BufferAttribute(skirtPos, 3));
      sg.computeVertexNormals();
      b.add('thatch', sg, roofTint());
    }

    pushCollider(cx, (bot + topY) / 2, cz, cw / 2 + 0.05, (topY - bot) / 2, cd / 2 + 0.05, 'building-chimney');
  }

  /* porch / arcade canopies ---------------------------------------------- */
  for (const c of canopies) {
    const eaveFn = (s) => new THREE.Vector3(c.x - c.w / 2 + c.w * s, c.yOut, c.zOut);
    const topFn = (s) => new THREE.Vector3(c.x - c.w / 2 + c.w * s, c.yIn, c.zIn - 0.14);
    const patch = thatchPatch(eaveFn, topFn, {
      sSegs: Math.max(5, Math.round(c.w / 0.42)), detail: thatchQ, thickness: 0.28,
      sag: 0.03, wave: 0.045, swell: 0.075, bell: 0.06, bellD: 0.5,
      seed: plot.seed * 0.003, color: tiled ? { r: 1, g: 1, b: 1 } : thatchCol,
      weather: tiled ? 0 : weather,
    });
    b.add(roofKey, patch.geometry, null, true);
    rollEnd(patch, 0, new THREE.Vector3(-1, 0, 0), 0.12);
    rollEnd(patch, patch.sSegs, new THREE.Vector3(1, 0, 0), 0.12);
  }

  /* shop sign ------------------------------------------------------------- */
  if (plot.sign) {
    const off = plot.sign.offset || 2.2;
    let su = door.u + off;
    if (su > hw - 1.0) su = door.u - off;
    su = THREE.MathUtils.clamp(su, -hw + 0.9, hw - 0.9);
    const bracketY = Math.min(sh + 0.18, 3.35);
    const armLen = 1.05;
    const faceN = frontGround.faceN ?? 0;

    // wrought-iron bracket, merged into the building
    wallBox(b, frontGround, 'iron', su, bracketY, faceN + armLen / 2, 0.06, 0.075, armLen, { r: 0.85, g: 0.85, b: 0.88 });
    wallBox(b, frontGround, 'iron', su, bracketY - 0.24, faceN + 0.10, 0.06, 0.5, 0.09, { r: 0.85, g: 0.85, b: 0.88 });
    const brace = new THREE.BoxGeometry(0.055, 0.055, 0.72);
    brace.rotateX(Math.PI / 4);
    brace.translate(su, bracketY - 0.24, faceN + 0.30);
    brace.applyMatrix4(frontGround.m);
    b.add('iron', brace, { r: 0.85, g: 0.85, b: 0.88 });
    // scrollwork
    const scroll = new THREE.TorusGeometry(0.13, 0.022, 4, 10, Math.PI * 1.5);
    scroll.rotateY(Math.PI / 2);
    scroll.translate(su, bracketY - 0.20, faceN + 0.44);
    scroll.applyMatrix4(frontGround.m);
    b.add('iron', scroll, { r: 0.85, g: 0.85, b: 0.88 });

    // the swinging board is its own body — physics writes to it every frame
    const hangW = 0.86, hangH = 0.62, hangT = 0.055;
    wallPoint(frontGround, su, bracketY - 0.03, faceN + armLen - 0.14, _v);
    const anchorLocal = _v.clone();
    const anchorWorld = toWorld(anchorLocal.x, anchorLocal.y, anchorLocal.z);
    const drop = 0.30 + hangH / 2;

    const signGroup = new THREE.Group();
    signGroup.name = `sign-${plot.id}`;
    const boardGeo = new THREE.BoxGeometry(hangW, hangH, hangT);
    normalizeGeo(boardGeo, { r: 1, g: 1, b: 1 });
    const frameGeos = [];
    for (const sgn of [-1, 1]) {
      const eye = new THREE.TorusGeometry(0.055, 0.014, 4, 9);
      eye.translate(sgn * hangW * 0.3, hangH / 2 + 0.05, 0);
      frameGeos.push(normalizeGeo(eye, { r: 0.85, g: 0.85, b: 0.88 }));
      const link = new THREE.BoxGeometry(0.022, drop - hangH / 2, 0.022);
      link.translate(sgn * hangW * 0.3, hangH / 2 + (drop - hangH / 2) / 2 + 0.04, 0);
      frameGeos.push(normalizeGeo(link, { r: 0.85, g: 0.85, b: 0.88 }));
      const edge = new THREE.BoxGeometry(0.05, hangH + 0.06, hangT + 0.03);
      edge.translate(sgn * (hangW / 2 - 0.02), 0, 0);
      frameGeos.push(normalizeGeo(edge, { r: 0.85, g: 0.85, b: 0.88 }));
    }
    const ironGeo = frameGeos.length ? mergeGeometries(frameGeos, false) : null;
    const boardMesh = new THREE.Mesh(boardGeo, ctx.resolve('woodPlank'));
    boardMesh.castShadow = true; boardMesh.receiveShadow = true;
    signGroup.add(boardMesh);
    if (ironGeo) {
      const im = new THREE.Mesh(ironGeo, ctx.resolve('iron'));
      im.castShadow = true; im.receiveShadow = true;
      signGroup.add(im);
    }
    frameGeos.forEach((g2) => g2.dispose());
    signGroup.position.set(anchorWorld[0], anchorWorld[1] - drop, anchorWorld[2]);
    signGroup.rotation.y = rot;
    signGroup.userData.interactive = true;
    signGroup.userData.prompt = plot.sign.text;
    signGroup.userData.kind = 'sign';

    interactables.push({
      object3D: signGroup,
      body: 'dynamic',
      // local space, centred on the board's own origin
      collider: boxCollider([0, 0, 0], [hangW / 2, hangH / 2 + 0.02, hangT / 2 + 0.02], null, { tag: 'sign' }),
      mass: 7,
      linearDamping: 0.6,
      angularDamping: 0.75,
      grabbable: false,
      prompt: plot.sign.text,
      tag: `sign-${plot.id}`,
      joint: {
        type: 'revolute',
        anchorWorld,
        // the sign swings about the horizontal bracket arm, i.e. the plot forward axis
        axis: [Math.sin(rot), 0, Math.cos(rot)],
      },
    });
    ctx.signGroups.push(signGroup);
  }

  /* ---------------------------------------------------------------------- */
  /* Colliders                                                               */
  /* ---------------------------------------------------------------------- */
  {
    const r = rects[0];
    // 0.26 clears everything mounted OUTSIDE the wall the player could otherwise
    // walk through: plinth 0.13, window sills 0.20, shutters, the arch ring.
    const o = 0.26;
    // ...and the slab reaches WALL_COL_IN inside the rect line, i.e. 35 mm past
    // the plaster's inner face. That matters now that the rooms are enterable:
    // the old 0.42 m slab stopped 0.16 m outside the rect line, which left 0.235 m
    // of interior plaster with nothing behind it to stand on.
    const t = o + WALL_COL_IN;
    const hx = (r.x1 - r.x0) / 2 + o, hz = (r.z1 - r.z0) / 2 + o;
    const zF = r.z1 + o - t / 2;
    const gu0 = door.u - dw / 2, gu1 = door.u + dw / 2;
    const doorHead = doorBase + dhTotal;

    // ONE SET OF SLABS PER STOREY, not one set for the whole shell. A jettied
    // upper storey is 0.45 m wider and 0.45 m deeper than the ground floor, so a
    // single slab raised off rects[0] put an invisible wall 0.45 m inside the
    // upper room's real plaster — 30 of 121 floor samples up there were inside it.
    for (let i = 0; i < storeys; i++) {
      const s = rects[i];
      const sx = (s.x1 - s.x0) / 2 + o, sz = (s.z1 - s.z0) / 2 + o;
      const cz = (s.z0 + s.z1) / 2;
      const yc = (s.y0 + s.y1) / 2, hy = (s.y1 - s.y0) / 2;
      // back and both ends: sealed, no openings big enough for a capsule
      pushCollider(0, yc, s.z0 - o + t / 2, sx, hy, t / 2, 'building-wall');
      pushCollider(s.x1 + o - t / 2, yc, cz, t / 2, hy, sz, 'building-wall');
      pushCollider(s.x0 - o + t / 2, yc, cz, t / 2, hy, sz, 'building-wall');
      if (i > 0) {
        pushCollider(0, yc, s.z1 + o - t / 2, sx, hy, t / 2, 'building-wall');
        continue;
      }
      // THE GROUND FRONT WALL IS SEGMENTED AROUND THE DOORWAY. This is the whole
      // feature: a pier either side of the opening, a lintel over it and the
      // threshold under it, so the capsule passes through the hole and nowhere
      // else. The gap is exactly the hole cut in the plaster.
      for (const [a0, a1] of [[-sx, gu0], [gu1, sx]]) {
        if (a1 - a0 < 0.02) continue;
        pushCollider((a0 + a1) / 2, yc, zF, (a1 - a0) / 2, hy, t / 2, 'building-wall');
      }
      if (s.y1 - doorHead > 0.02) {
        pushCollider(door.u, (doorHead + s.y1) / 2, zF, dw / 2, (s.y1 - doorHead) / 2, t / 2,
          'building-lintel');
      }
    }
    // Threshold: fills the wall below the opening and bridges 0.12 m further in
    // than the piers, so there is no seam between it and the floor.
    pushCollider(door.u, floorLocal + THRESH_LIP - 0.30, zF - 0.06,
      dw / 2 + 0.06, 0.30, t / 2 + 0.06, 'building-threshold');
    // The weather bar is solid, so you step over it instead of through it. It is
    // NOT a doorstop: the leaf clears it by 10 mm in both directions, on purpose.
    pushCollider(door.u, floorLocal + WEATHER_BAR / 2,
      r.z1 + leafPlane(frontGround.faceN ?? 0) + 0.09,
      dw / 2 + 0.04, WEATHER_BAR / 2, 0.045, 'building-weatherbar');

    // The old 'building-core' block — a solid box filling the whole shell — is
    // gone. It was what made these buildings sealed. What replaces it is this
    // underfloor slab, whose top is exactly the finished floor: interiors.js lays
    // its own floor and collider at the same level, and if it ever fails to, the
    // player still stands on stone instead of falling out of the world.
    pushCollider(0, floorLocal - 0.30, (r.z0 + r.z1) / 2,
      Math.max(0.2, hx - o - 0.02), 0.30, Math.max(0.2, hz - o - 0.02), 'building-subfloor');

    // sloped roof slabs
    for (const dir of [1, -1]) {
      const zEdge = dir > 0 ? ze1 : ze0;
      const eaveMid = new THREE.Vector3((xe0 + xe1) / 2, yEaveTip, zEdge);
      const ridgeMid = new THREE.Vector3((xe0 + xe1) / 2, yRidge, zRidge);
      const S = new THREE.Vector3().subVectors(ridgeMid, eaveMid);
      const slopeLen = S.length();
      S.normalize();
      const U = new THREE.Vector3(dir > 0 ? 1 : -1, 0, 0);
      const N = new THREE.Vector3().crossVectors(U, S).normalize();
      if (N.y < 0) N.negate();
      const S2 = new THREE.Vector3().crossVectors(U, N).normalize();
      _m.makeBasis(U, N, S2);
      _q.setFromRotationMatrix(_m);
      const c = new THREE.Vector3().addVectors(eaveMid, ridgeMid).multiplyScalar(0.5)
        .addScaledVector(N, -THATCH_T / 2);
      pushCollider(c.x, c.y, c.z, (xe1 - xe0) / 2, THATCH_T / 2 + 0.12, slopeLen / 2, 'building-roof', _q.clone());
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Ivy                                                                     */
  /* ---------------------------------------------------------------------- */
  if (plot.ivy > 0 && ctx.ivy) {
    const cover = plot.ivy;
    const wm = new THREE.Matrix4().makeRotationY(rot);
    wm.setPosition(plot.position[0], baseY, plot.position[2]);
    const sideWall = rng.bool() ? 'left' : 'right';
    const targets = [];
    for (const w of walls) {
      if (w.side === 'front' || w.side === sideWall) targets.push(w);
    }
    // One card samples one atlas cell, and every cell is already a 5-9 leaf
    // cluster — so a card is a sprig ~20-34 cm across, giving leaves of 5-9 cm.
    // Sized any smaller (an earlier pass used 8.5-17.5 cm cards) the leaves
    // shrink to 3 cm flecks and the wall reads as mould, not ivy. Pitch stays
    // well under the card size so neighbouring sprigs overlap.
    //
    // DISTRIBUTION is what makes it read as a plant rather than a texture. An
    // even, sparse dusting shows plaster between every sprig; real ivy grows in
    // connected MATS that thin at the edges. So: a low-frequency mat field (two
    // octaves), biased to grow up from the ground, out of the wall corners and
    // around the window reveals, then thresholded HARD — inside a mat nearly
    // every grid cell is filled and leaf occludes leaf, outside it the plaster
    // is genuinely bare. A narrow fringe band below the threshold puts a few
    // small runners ahead of the mass so the edge is not a cut-out. The pitch
    // is tighter than before (0.160 -> 0.128) and the bare area much larger, so
    // the mats are ~2x denser at a similar total instance count.
    const step = 0.128 / Math.max(0.45, detail);
    const seedU = plot.seed * 0.31;
    const q = new THREE.Quaternion();
    const basis = new THREE.Matrix4();
    const inst = new THREE.Matrix4();
    const AX_Z = new THREE.Vector3(0, 0, 1);
    const AX_X = new THREE.Vector3(1, 0, 0);
    const AX_Y = new THREE.Vector3(0, 1, 0);
    const U = new THREE.Vector3(), N = new THREE.Vector3(), V = new THREE.Vector3(0, 1, 0);

    for (const w of targets) {
      U.set(w.ux, 0, w.uz); N.set(w.nx, 0, w.nz);
      basis.makeBasis(U, V, N);
      const cols = Math.max(2, Math.floor(w.L / step));
      const rows = Math.max(2, Math.floor(w.H / step));
      const faceN = (w.faceN ?? 0);
      // the mat field lives in wall-local metres, offset per wall so the two
      // walls of a plot do not share a pattern
      const fu = seedU + (w.side === 'front' ? 0 : 37.3);
      const fv = (w.side === 'front' ? 0 : 12.7);
      // How much of the wall the plant has taken: sets the threshold the mat
      // field has to clear. 0.15 cover -> a couple of patches; 0.72 -> most of
      // the lower wall under one connected sheet.
      const thresh = 1.50 - 1.10 * THREE.MathUtils.clamp(cover, 0, 1);
      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const u = -w.L / 2 + step * (i + 0.5) + rng.sym(step * 0.4);
          const v = step * (j + 0.5) + rng.sym(step * 0.4);
          if (Math.abs(u) > w.L / 2 - 0.12 || v > w.H - 0.1) continue;
          const vy = v + w.y0;
          // two octaves: 6 m mats with 2 m lobes on their edges
          const mat = 0.5
            + 0.54 * NOISE(u * 0.17 + fu, vy * 0.13 + fv)
            + 0.20 * NOISE(u * 0.52 + fu * 1.7, vy * 0.44 + fv);
          // climbs from the ground up
          const fromBase = 1 - THREE.MathUtils.smoothstep(vy / wallTop, 0.02, 0.95);
          // and out of the corners, where a real plant finds purchase
          const dEdge = w.L / 2 - Math.abs(u);
          const corner = 1 - THREE.MathUtils.smoothstep(dEdge, 0.10, 1.40);
          // and around the window reveals, hugging the frames
          let near = 0;
          let blocked = false;
          for (const o of w.openings) {
            if (u > o.u0 - 0.16 && u < o.u1 + 0.16 && v > o.v0 - 0.16 && v < o.v1 + 0.16) blocked = true;
            const du = Math.max(o.u0 - u, u - o.u1, 0);
            const dv = Math.max(o.v0 - v, v - o.v1, 0);
            const d = Math.sqrt(du * du + dv * dv);
            const g = 1 - THREE.MathUtils.smoothstep(d, 0.10, 0.70);
            if (g > near) near = g;
          }
          if (blocked) continue;
          const field = mat + 0.44 * fromBase + 0.30 * corner + 0.22 * near;
          // near-binary: dense inside the mat, bare outside, with a thin fringe
          const dens = THREE.MathUtils.smoothstep(field, thresh, thresh + 0.09);
          const fringe = 0.13 * THREE.MathUtils.smoothstep(field, thresh - 0.34, thresh);
          if (rng.next() > Math.min(1, 0.97 * dens + fringe)) continue;
          const isFringe = dens < 0.45;
          // pressed against the plaster: 1-3 cm of standoff, jittered
          const outN = faceN + rng.range(0.010, 0.030);
          wallPoint(w, u, v, outN, _v);
          // inside the mat the sprigs are full size and overlap; the runners
          // that reach out ahead of it are the small end of the same range
          const sc = isFringe ? rng.range(0.20, 0.26) : rng.range(0.23, 0.34);
          q.setFromRotationMatrix(basis);
          // spin in the wall plane, then splay left/right and lean the tip out —
          // biased outward so a tilted card cannot bury itself in the plaster
          _q.setFromAxisAngle(AX_Z, rng.range(0, Math.PI * 2));
          q.multiply(_q);
          _q.setFromAxisAngle(AX_Y, rng.sym(0.55));
          q.multiply(_q);
          _q.setFromAxisAngle(AX_X, rng.range(-0.12, 0.85));
          q.multiply(_q);
          const sy = isFringe ? rng.range(1.15, 1.55) : rng.range(0.85, 1.2);
          inst.compose(_v, q, _v2.set(sc * rng.range(0.85, 1.15), sc * sy, sc));
          inst.premultiply(wm);
          ctx.ivy.matrices.push(inst.clone());
          const shade = rng.range(0.66, 1.10);
          ctx.ivy.colors.push(shade * (1 - rng.range(0, 0.15)), shade, shade * (1 - rng.range(0, 0.25)));
        }
      }
      // trailing runners off the top, hanging over the wall head
      const runners = Math.round(22 * cover * detail);
      for (let i = 0; i < runners; i++) {
        const u = rng.range(-w.L / 2 + 0.4, w.L / 2 - 0.4);
        const v = w.H * rng.range(0.55, 0.98);
        wallPoint(w, u, v, (w.faceN ?? 0) + rng.range(0.015, 0.035), _v);
        const sc = rng.range(0.075, 0.13);
        q.setFromRotationMatrix(basis);
        _q.setFromAxisAngle(AX_Z, rng.range(0, Math.PI * 2));
        q.multiply(_q);
        _q.setFromAxisAngle(AX_X, rng.range(0.1, 0.7));
        q.multiply(_q);
        inst.compose(_v, q, _v2.set(sc, sc * rng.range(1.4, 2.6), sc));
        inst.premultiply(wm);
        ctx.ivy.matrices.push(inst.clone());
        ctx.ivy.colors.push(0.72, 0.92, 0.64);
      }
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Publish the interior  (RoomSpec[], see contracts.js)                    */
  /* ---------------------------------------------------------------------- */
  // Every field here is read off the geometry that was actually built above.
  // interiors.js and furnishings.js build against these numbers and cannot check
  // them, so any lie shows up as furniture in a wall.
  {
    const uses = (LAYOUT.INTERIOR_USES || {})[plot.id] || [];
    // Where the flues are. NOT part of the contract — furnishings.js is free to
    // ignore it — but a hearth on the wrong wall has no chimney over it, and this
    // module is the only one that knows where the stacks came down. Positions are
    // room-local (aligned with width/depth, origin at the room centre); the stack
    // is only SOLID above `solidY`, because a chimney is corbelled out of the roof
    // and does not reach the ground floor.
    const flues = (plot.chimneys || []).map((ch) => {
      const cx = THREE.MathUtils.clamp(ch.x, top.x0 + 0.9, top.x1 - 0.9);
      const cz = THREE.MathUtils.clamp(ch.z ?? 0, top.z0 + 0.7, top.z1 - 0.7);
      const cw = ch.width || 1.2;
      return {
        lx: cx, lz: cz, width: cw, depth: cw * 0.74,
        solidY: baseY + Math.min(wallTop - 1.2, roofYAt(cx, cz) - 1.6),
        wall: Math.abs(cz) > Math.abs(cx) * 0.6 ? (cz < 0 ? 'back' : 'front')
          : (cx < 0 ? 'left' : 'right'),
      };
    });
    for (let i = 0; i < storeys; i++) {
      const r = rects[i];
      const mine = walls.filter((w) => w.storey === i);
      // the four walls of a storey share a faceN, so one of them settles it
      const tEff = WALL_T - (mine[0].faceN ?? 0);
      const isTop = i === storeys - 1;
      const fY = i === 0 ? floorLocal : i * sh + UPPER_FLOOR_UP;
      // Below the top storey the ceiling is the underside of the joists. On the
      // top storey it is where the thatch underside meets the wall: the roof's
      // top surface passes ROOF_BITE below the wall head at the wall face, so
      // that is the eaves-level ceiling and `ridgeY` is the apex of the same
      // surface, thatchVert higher than the ridge line's top face.
      const cY = isTop ? wallTop - ROOF_BITE : (i + 1) * sh - JOIST_ZONE;

      const openings = [];
      for (const w of mine) {
        for (const op of w.openings) {
          const cu = (op.u0 + op.u1) / 2;
          // centreWorld is the centre of the hole, mid-slab: the one point both a
          // threshold and a shaft of light can be measured from
          wallPoint(w, cu, (op.v0 + op.v1) / 2, (w.faceN ?? 0) - WALL_T / 2, _v);
          openings.push({
            kind: op.kind === 'door' ? 'door' : 'window',
            wall: w.side,
            lx: cu,
            sillY: baseY + w.y0 + op.v0,
            width: op.u1 - op.u0,
            height: op.v1 - op.v0,
            centreWorld: toWorld(_v.x, _v.y, _v.z),
            // outward normal, rotated into world, negated
            inward: [
              -(w.nx * cosR + w.nz * sinR), 0,
              -(-w.nx * sinR + w.nz * cosR),
            ],
            primary: op.kind === 'door',
          });
        }
      }

      ctx.rooms.push({
        plotId: plot.id,
        storey: i,
        floorY: baseY + fY,
        ceilingY: baseY + cY,
        width: (r.x1 - r.x0) - 2 * tEff,
        depth: (r.z1 - r.z0) - 2 * tEff,
        centre: toWorld((r.x0 + r.x1) / 2, fY, (r.z0 + r.z1) / 2),
        rotation: rot,
        wallThickness: tEff,
        openings,
        openToRoof: isTop,
        ridgeY: baseY + yRidge - thatchVert,
        use: uses[i] || (i === 0 ? 'hall' : 'store'),
        // extras, beyond the contract — see above
        chimneys: flues.map((f) => ({
          ...f,
          lx: f.lx - (r.x0 + r.x1) / 2,
          lz: f.lz - (r.z0 + r.z1) / 2,
        })),
      });
    }
  }

  /* ---------------------------------------------------------------------- */
  /* Merge, per material, into WORLD space                                   */
  /* ---------------------------------------------------------------------- */
  // The plot transform is baked in rather than carried on a Group, because the
  // factory merges these again across a whole spatial cluster of plots: eleven
  // buildings times a dozen materials is 130-odd draw calls we cannot afford.
  // UVs were already projected in plot-local space, so baking is free.
  const world = new THREE.Matrix4().makeRotationY(rot);
  world.setPosition(plot.position[0], baseY, plot.position[2]);

  const parts = new Map();
  let tris = 0;
  for (const [key, geos] of b.map) {
    if (!geos.length) continue;
    let merged = null;
    try {
      merged = geos.length === 1 ? geos[0] : mergeGeometries(geos, false);
    } catch (e) {
      merged = null;
    }
    if (!merged) {
      console.warn(`[buildings] merge failed for ${plot.id}/${key}`);
      geos.forEach((g) => g.dispose());
      continue;
    }
    if (merged !== geos[0]) geos.forEach((g) => g.dispose());
    else geos.slice(1).forEach((g) => g.dispose());
    merged.applyMatrix4(world);
    parts.set(key, merged);
    tris += (merged.index ? merged.index.count : merged.getAttribute('position').count) / 3;
  }

  return { parts, tris };
}

/* -------------------------------------------------------------------------- */
/* Ivy leaf card                                                               */
/* -------------------------------------------------------------------------- */

/**
 * Two quads, each cropped to ONE cell of the ivy atlas, tilted apart and set
 * either side of the stem: 4 triangles that read as a pair of leaf sprigs.
 * Cropping matters — a card mapped 0..1 samples all four clusters at once, which
 * turns the wall into torn paper. The pivot is at the stem so the per-instance
 * scale is the sprig size and grows upward.
 */
function ivyLeafGeometry() {
  const crop = (g, col, row) => {
    const uv = g.getAttribute('uv');
    const iu = 1 / IVY_ATLAS.cols, iv = 1 / IVY_ATLAS.rows;
    const pad = 0.03;                        // keep the filter off the cell seam
    for (let i = 0; i < uv.count; i++) {
      uv.setXY(i,
        (col + pad + uv.getX(i) * (1 - 2 * pad)) * iu,
        (row + pad + uv.getY(i) * (1 - 2 * pad)) * iv);
    }
    uv.needsUpdate = true;
    return g;
  };
  const parts = [];
  const a = new THREE.PlaneGeometry(1.12, 1.0, 1, 1);
  a.rotateX(-0.26); a.rotateZ(-0.30); a.translate(-0.20, 0.46, 0.015);
  parts.push(crop(a, 0, 1));
  const c = new THREE.PlaneGeometry(0.94, 0.86, 1, 1);
  c.rotateX(0.30); c.rotateZ(0.44); c.translate(0.24, 0.32, -0.012);
  parts.push(crop(c, 1, 0));
  const g = parts.length > 1 ? (mergeGeometries(parts, false) || parts[0]) : parts[0];
  for (const p of parts) if (p !== g) p.dispose();
  return g;
}

/* -------------------------------------------------------------------------- */
/* Factory                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Build every plot in `layout.PLOTS`.
 *
 * @param {{materials:Object, terrain:Object, quality:Object}} deps
 * @returns {Object} WorldChunk + `flowerBoxAnchors`
 */
export function createBuildings({ materials, terrain, quality }) {
  const q = quality || {};
  const group = new THREE.Group();
  group.name = 'buildings';

  const colliders = [];
  const interactables = [];
  const lightAnchors = [];
  const flowerBoxAnchors = [];
  const signGroups = [];
  const doorGroups = [];
  /** @type {import('../contracts.js').RoomSpec[]} */
  const rooms = [];
  const resolve = makeMatResolver(materials);

  const heightFn = terrain && (terrain.heightAt || terrain.sampleHeight || terrain.getHeight);
  const groundY = (x, z) => {
    if (typeof heightFn === 'function') {
      try {
        const y = heightFn.call(terrain, x, z);
        if (Number.isFinite(y)) return y;
      } catch { /* terrain not ready — the village floor is flat anyway */ }
    }
    return 0;
  };

  /* Spatial clusters. Merging every plot into one mesh per material would be the
   * fewest draw calls but would also defeat frustum culling — a single mesh
   * spanning the whole plaza is never off-screen. PLOTS is authored going round
   * the ring, so three contiguous chunks of it are three contiguous arcs. */
  const CLUSTERS = 3;
  const clusters = [];
  for (let i = 0; i < CLUSTERS; i++) clusters.push({ parts: new Map(), matrices: [], colors: [] });
  const clusterOf = (i) => Math.min(CLUSTERS - 1, Math.floor((i * CLUSTERS) / PLOTS.length));

  let drawCalls = 0, triangles = 0, built = 0, ivyCount = 0;
  const ctx = {
    quality: q, colliders, interactables, lightAnchors, flowerBoxAnchors,
    resolve, groundY, ivy: clusters[0], signGroups, doorGroups, rooms,
  };

  PLOTS.forEach((plot, i) => {
    const cl = clusters[clusterOf(i)];
    ctx.ivy = cl;                    // this plot's ivy lands in its own cluster
    try {
      const res = buildPlot(plot, ctx);
      for (const [key, geo] of res.parts) {
        let arr = cl.parts.get(key);
        if (!arr) cl.parts.set(key, arr = []);
        arr.push(geo);
      }
      triangles += res.tris;
      built++;
    } catch (err) {
      console.error(`[buildings] plot "${plot.id}" failed to build:`, err);
    }
  });

  /* One mesh per (cluster, material) — around 40 instead of 140. */
  clusters.forEach((cl, ci) => {
    for (const [key, geos] of cl.parts) {
      if (!geos.length) continue;
      let merged = null;
      try {
        merged = geos.length === 1 ? geos[0] : mergeGeometries(geos, false);
      } catch (e) {
        merged = null;
      }
      if (!merged) {
        console.warn(`[buildings] cluster merge failed for ${ci}/${key}`);
        geos.forEach((g) => g.dispose());
        continue;
      }
      if (merged !== geos[0]) geos.forEach((g) => g.dispose());
      else geos.slice(1).forEach((g) => g.dispose());
      // aoMap wants a second uv set; share the first so it is never sampled at 0,0
      if (merged.getAttribute('uv') && !merged.getAttribute('uv1')) {
        merged.setAttribute('uv1', merged.getAttribute('uv'));
      }
      merged.computeBoundingSphere();
      const mesh = new THREE.Mesh(merged, resolve(key));
      mesh.name = `village-${ci}-${key}`;
      mesh.castShadow = key !== 'glass';
      mesh.receiveShadow = true;
      group.add(mesh);
      drawCalls++;
    }
    cl.parts.clear();

    /* ivy: one InstancedMesh per cluster, so it culls with its buildings */
    if (!cl.matrices.length) return;
    try {
      const geo = ivyLeafGeometry();
      let mat = null;
      if (typeof materials?.variant === 'function') {
        try { mat = materials.variant('ivy', { side: THREE.DoubleSide }); } catch { mat = null; }
      }
      if (!mat) mat = resolve('ivy');
      const im = new THREE.InstancedMesh(geo, mat, cl.matrices.length);
      for (let i = 0; i < cl.matrices.length; i++) {
        im.setMatrixAt(i, cl.matrices[i]);
        _col.setRGB(cl.colors[i * 3], cl.colors[i * 3 + 1], cl.colors[i * 3 + 2]);
        im.setColorAt(i, _col);
      }
      im.instanceMatrix.needsUpdate = true;
      if (im.instanceColor) im.instanceColor.needsUpdate = true;
      im.castShadow = true;
      im.receiveShadow = true;
      im.name = `village-${ci}-ivy`;
      im.computeBoundingSphere();     // InstancedMesh culls on its own sphere
      group.add(im);
      drawCalls++;
      ivyCount += cl.matrices.length;
      triangles += (geo.index ? geo.index.count : geo.getAttribute('position').count)
        / 3 * cl.matrices.length;
    } catch (err) {
      console.warn('[buildings] ivy failed:', err);
    }
    cl.matrices.length = 0;
    cl.colors.length = 0;
  });

  // The sign boards and the door leaves are dynamic bodies: physics owns their
  // transform, so they live directly on the chunk group rather than inside a
  // building group (whose matrix would be applied twice).
  for (const s of signGroups) group.add(s);
  for (const d of doorGroups) {
    group.add(d);
    d.traverse((o) => {
      if (!o.isMesh || !o.geometry) return;
      const g = o.geometry;
      triangles += (g.index ? g.index.count : g.getAttribute('position').count) / 3;
    });
  }

  /* -------------------------------------------------------------- interiors */
  // The handoff described in contracts.js. `roomAt` is not part of that contract;
  // it is here because "is the player inside this building" is a question both
  // the interior and the lighting streams need answered, and the shell is the
  // only module that knows the answer without re-deriving a wall.
  const roomsByPlot = new Map();
  for (const r of rooms) {
    let arr = roomsByPlot.get(r.plotId);
    if (!arr) roomsByPlot.set(r.plotId, arr = []);
    arr.push(r);
  }
  const interiors = {
    rooms,
    roomsFor(plotId) { return roomsByPlot.get(plotId) || []; },
    /** The room containing a world point, or null. No allocation. */
    roomAt(x, y, z) {
      for (let i = 0; i < rooms.length; i++) {
        const r = rooms[i];
        const head = r.openToRoof ? Math.max(r.ridgeY, r.ceilingY) : r.ceilingY;
        if (y < r.floorY - 0.5 || y > head + 0.3) continue;
        const dx = x - r.centre[0], dz = z - r.centre[2];
        const c = Math.cos(r.rotation), s = Math.sin(r.rotation);
        // true inverse of plotToWorld's rotation
        const lx = dx * c - dz * s;
        const lz = dx * s + dz * c;
        if (Math.abs(lx) <= r.width / 2 + 0.06 && Math.abs(lz) <= r.depth / 2 + 0.06) return r;
      }
      return null;
    },
  };

  const doorMeshes = doorGroups.length;
  const stats = {
    buildings: built,
    meshes: drawCalls + signGroups.length * 2 + doorMeshes,
    drawCalls: drawCalls + signGroups.length * 2 + doorMeshes,
    triangles: Math.round(triangles),
    colliders: colliders.length,
    interactables: interactables.length,
    lightAnchors: lightAnchors.length,
    flowerBoxes: flowerBoxAnchors.length,
    ivyInstances: ivyCount,
    rooms: rooms.length,
    doorLeaves: doorMeshes,
  };
  console.info('[buildings]', stats);

  return {
    group,
    colliders,
    interactables,
    lightAnchors,
    flowerBoxAnchors,
    interiors,
    update() { /* buildings are static; the signs and doors are driven by physics */ },
    dispose() {
      group.traverse((o) => {
        if (o.isMesh || o.isInstancedMesh) o.geometry?.dispose?.();
      });
      group.clear();
    },
    stats,
  };
}
