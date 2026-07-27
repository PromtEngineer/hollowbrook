/**
 * ============================================================================
 *  PROPS — the well, every dressing prop, the planting, and all physics bodies
 * ============================================================================
 * There are only three kinds of object in this file:
 *
 *   1. STATIC DRESSING — merged into one mesh per material by `Batch`. A cart,
 *      a lamp post and a fence picket 40 m apart share a draw call if they
 *      share a material. Their colliders are emitted in WORLD space.
 *   2. INSTANCED VEGETATION — blossom, leaves, ivy, grass tufts, lantern
 *      flames. One `InstancedMesh` per material for the whole village.
 *   3. DYNAMIC BODIES — barrels, crates, buckets, pots, apples, sacks. These
 *      cannot be merged (they move independently), so each is one mesh with
 *      one material wherever possible and the count is capped by
 *      `quality.maxDynamicBodies`.
 *
 * Positions come from `layout.js`; nothing is hard-coded that layout already
 * knows. Nothing uses Math.random.
 * ============================================================================
 */

import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { Rng } from '../util/rng.js';
import {
  WELL, PROP_SITES, LAMP_POSTS, PLOTS,
  plotToWorld, frontAnchors, roadDistance,
} from './layout.js';
import {
  boxCollider, cylinderCollider, sphereCollider, capsuleCollider,
  quatY, tagInteractive, disposeGroup, variantKey,
} from '../contracts.js';

/* -------------------------------------------------------------------------- */
/* Scratch — hoisted so nothing allocates in a hot path                        */
/* -------------------------------------------------------------------------- */

const _o = new THREE.Object3D();
const _v = new THREE.Vector3();
const _m = new THREE.Matrix4();
const _m2 = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const _e = new THREE.Euler();
const _ONE = new THREE.Vector3(1, 1, 1);
const _YAXIS = new THREE.Vector3(0, 1, 0);
const TAU = Math.PI * 2;

/** Compose a local matrix. Returns a SHARED matrix — consume it immediately. */
function trs(px, py, pz, rx = 0, ry = 0, rz = 0, sx = 1, sy = 1, sz = 1) {
  _o.position.set(px, py, pz);
  _o.rotation.set(rx, ry, rz);
  _o.scale.set(sx, sy, sz);
  _o.updateMatrix();
  return _o.matrix;
}

/** Quaternion array from Euler XYZ — for colliders whose axis is not +Y. */
function quatEuler(x, y, z) {
  _q.setFromEuler(_e.set(x, y, z, 'XYZ'));
  return [_q.x, _q.y, _q.z, _q.w];
}

/**
 * World -> plot-local. NOTE: `layout.insideAnyBuilding` computes this with the
 * wrong inverse rotation (it mirrors local X), so it cannot be trusted for
 * plots whose rotation is not ~0. We do the inverse of `plotToWorld` properly.
 */
function toLocal(plot, x, z, out) {
  const c = Math.cos(plot.rotation), s = Math.sin(plot.rotation);
  const dx = x - plot.position[0], dz = z - plot.position[2];
  out[0] = c * dx - s * dz;
  out[1] = s * dx + c * dz;
  return out;
}

const _loc = [0, 0];

/**
 * Radial keep-outs registered by the hero props, so planting and loose clutter
 * never spawn inside a cart, a trough or a market stall. Cheap and it means the
 * placement rules live in one place instead of in every builder.
 */
function pushOutOfProps(keepOut, x, z, myR) {
  for (let iter = 0; iter < 3; iter++) {
    let moved = false;
    for (const k of keepOut) {
      const dx = x - k.x, dz = z - k.z;
      const d = Math.hypot(dx, dz);
      const need = k.r + myR;
      if (d >= need) continue;
      if (d < 1e-3) { x = k.x + need; z = k.z; }
      else { const f = need / d; x = k.x + dx * f; z = k.z + dz * f; }
      moved = true;
    }
    if (!moved) break;
  }
  return [x, z];
}

/** True if (x,z) falls inside any plot footprint (+margin). Correct inverse. */
function insidePlots(x, z, margin = 0) {
  for (const p of PLOTS) {
    toLocal(p, x, z, _loc);
    if (Math.abs(_loc[0]) < p.width / 2 + margin && Math.abs(_loc[1]) < p.depth / 2 + margin) return true;
  }
  return false;
}

/**
 * Nudge a point out of any building footprint and off the well steps, pushing
 * along whichever axis is cheapest. Two of layout's authored PROP_SITES land
 * inside a footprint (`grindstone` is 0.5 m inside the Weaver's House, `trough`
 * is 0.65 m inside the Long Stable); rather than edit the plan — another stream
 * owns it — we move the prop clear. If clearing would displace the prop more
 * than `maxMove`, we leave it where the author put it: a small clip is better
 * than a prop teleported across the village.
 */
function clearPoint(x0, z0, margin = 0.55, maxMove = 3.0) {
  let x = x0, z = z0;
  for (let iter = 0; iter < 6; iter++) {
    let moved = false;
    for (const p of PLOTS) {
      toLocal(p, x, z, _loc);
      const lx = _loc[0], lz = _loc[1];
      const hw = p.width / 2 + margin, hd = p.depth / 2 + margin;
      if (Math.abs(lx) >= hw || Math.abs(lz) >= hd) continue;
      let nlx = lx, nlz = lz;
      if (hw - Math.abs(lx) < hd - Math.abs(lz)) nlx = (lx < 0 ? -1 : 1) * hw;
      else nlz = (lz < 0 ? -1 : 1) * hd;
      const w = plotToWorld(p, nlx, 0, nlz, _v);
      x = w.x; z = w.z;
      moved = true;
    }
    const wx = x - WELL.position[0], wz = z - WELL.position[2];
    const wr = Math.hypot(wx, wz);
    const keep = WELL.base[0].radius + margin;
    if (wr < keep) {
      const k = wr > 1e-3 ? keep / wr : 1;
      x = WELL.position[0] + (wr > 1e-3 ? wx * k : keep);
      z = WELL.position[2] + (wr > 1e-3 ? wz * k : 0);
      moved = true;
    }
    if (!moved) break;
    if (Math.hypot(x - x0, z - z0) > maxMove) return [x0, z0];
  }
  return [x, z];
}

/**
 * Satisfy both constraints at once: out of every building footprint AND out of
 * every prop keep-out. Each push can undo the other, so alternate. Returns null
 * when no nearby spot works, and the caller drops the object.
 */
function settle(keepOut, x, z, myR, margin = 0.4, maxMove = 2.2) {
  const x0 = x, z0 = z;
  for (let i = 0; i < 5; i++) {
    const [ax, az] = pushOutOfProps(keepOut, x, z, myR);
    const [bx, bz] = clearPoint(ax, az, margin, maxMove * 2);
    const settled = Math.abs(bx - x) < 1e-4 && Math.abs(bz - z) < 1e-4;
    x = bx; z = bz;
    if (settled) break;
  }
  if (Math.hypot(x - x0, z - z0) > maxMove) return null;
  const [cx, cz] = pushOutOfProps(keepOut, x, z, myR * 0.95);
  if (Math.hypot(cx - x, cz - z) > 0.02 || insidePlots(x, z, 0.12)) return null;
  return [x, z];
}

/** World matrix from a position + yaw, allocating a new Matrix4. */
function placeAt(x, y, z, yaw) {
  return new THREE.Matrix4().compose(
    new THREE.Vector3(x, y, z),
    _q.setFromAxisAngle(_YAXIS, yaw).clone(),
    _ONE,
  );
}

/* -------------------------------------------------------------------------- */
/* Geometry cache + primitives (build-time only)                               */
/* -------------------------------------------------------------------------- */

let GC = new Map();

function cached(key, make) {
  let g = GC.get(key);
  if (!g) { g = make(); GC.set(key, g); }
  return g;
}

const gBox = (w, h, d) => cached(`b|${w}|${h}|${d}`, () => new THREE.BoxGeometry(w, h, d));
const gCyl = (rt, rb, h, s = 12, open = false) =>
  cached(`c|${rt}|${rb}|${h}|${s}|${open}`, () => new THREE.CylinderGeometry(rt, rb, h, s, 1, open));
const gSph = (r, ws = 10, hs = 7) =>
  cached(`s|${r}|${ws}|${hs}`, () => new THREE.SphereGeometry(r, ws, hs));
const gPlane = (w, h, ws = 1, hs = 1) =>
  cached(`p|${w}|${h}|${ws}|${hs}`, () => new THREE.PlaneGeometry(w, h, ws, hs));
const gCircle = (r, s = 24) => cached(`o|${r}|${s}`, () => new THREE.CircleGeometry(r, s));
const gRing = (ri, ro, s = 32) => cached(`r|${ri}|${ro}|${s}`, () => new THREE.RingGeometry(ri, ro, s));
const gCone = (r, h, s = 8) => cached(`n|${r}|${h}|${s}`, () => new THREE.ConeGeometry(r, h, s));

/**
 * Strip a geometry to position/normal/uv, guarantee an index, transform, clone.
 * Merging fails if attribute sets disagree, so everything goes through here.
 */
function prep(src, matrix, uvMode = 'world', uvScale = 1) {
  const g = new THREE.BufferGeometry();
  const pos = src.getAttribute('position');
  g.setAttribute('position', pos.clone());
  const nrm = src.getAttribute('normal');
  g.setAttribute('normal', nrm ? nrm.clone() : new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3));
  const uv = src.getAttribute('uv');
  g.setAttribute('uv', uv ? uv.clone() : new THREE.BufferAttribute(new Float32Array(pos.count * 2), 2));
  if (src.index) {
    g.setIndex(src.index.clone());
  } else {
    const idx = new Uint32Array(pos.count);
    for (let i = 0; i < idx.length; i++) idx[i] = i;
    g.setIndex(new THREE.BufferAttribute(idx, 1));
  }
  if (!nrm) g.computeVertexNormals();
  if (matrix) g.applyMatrix4(matrix);
  if (uvMode === 'world') projectUV(g, uvScale);
  else if (uvScale !== 1) scaleUV(g, uvScale, uvScale);
  return g;
}

/**
 * Box-projected UVs from position. Merged static geometry has no other sane
 * shared parameterisation, and this is what stops a 3 m cart plank showing the
 * same texel density as a 0.06 m batten.
 */
function projectUV(g, scale) {
  const p = g.getAttribute('position');
  const n = g.getAttribute('normal');
  const uv = g.getAttribute('uv');
  for (let i = 0; i < p.count; i++) {
    const nx = Math.abs(n.getX(i)), ny = Math.abs(n.getY(i)), nz = Math.abs(n.getZ(i));
    const x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    let u, v;
    if (ny >= nx && ny >= nz) { u = x; v = z; }
    else if (nx >= nz) { u = z; v = y; }
    else { u = x; v = y; }
    uv.setXY(i, u * scale, v * scale);
  }
  uv.needsUpdate = true;
}

function scaleUV(g, su, sv) {
  const uv = g.getAttribute('uv');
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * su, uv.getY(i) * sv);
  uv.needsUpdate = true;
}

/** Displace the 8 corners of a unit box coherently — a hewn block, not a cube. */
function jitterBox(g, amp, rng) {
  const p = g.getAttribute('position');
  const off = new Map();
  for (let i = 0; i < p.count; i++) {
    const k = (p.getX(i) > 0 ? 4 : 0) | (p.getY(i) > 0 ? 2 : 0) | (p.getZ(i) > 0 ? 1 : 0);
    let o = off.get(k);
    if (!o) { o = [rng.sym(amp), rng.sym(amp * 0.7), rng.sym(amp)]; off.set(k, o); }
    p.setXYZ(i, p.getX(i) + o[0], p.getY(i) + o[1], p.getZ(i) + o[2]);
  }
  g.computeVertexNormals();
  return g;
}

/**
 * Alternate the radius by angular sector so a lathed barrel reads as separate
 * staves instead of a smooth drum, then flat-shade it.
 */
function staveify(g, sectors, amp) {
  const p = g.getAttribute('position');
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), z = p.getZ(i);
    const r = Math.hypot(x, z);
    if (r < 1e-4) continue;
    const a = Math.atan2(z, x) + Math.PI;
    const s = Math.floor((a / TAU) * sectors);
    const f = 1 + ((s & 1) ? amp : -amp);
    p.setXYZ(i, x * f, p.getY(i), z * f);
  }
  const flat = g.toNonIndexed();
  flat.computeVertexNormals();
  g.dispose();
  return flat;
}

function finish(geo) {
  const uv = geo.getAttribute('uv');
  if (uv && !geo.getAttribute('uv1')) geo.setAttribute('uv1', uv);
  geo.computeBoundingSphere();
  return geo;
}

/** Merge [{geo, matrix, uv?, uvScale?}] into one owned BufferGeometry. */
function mergeParts(parts) {
  const list = parts.map((p) => prep(p.geo, p.matrix, p.uv || 'world', p.uvScale ?? 1));
  let out = null;
  try {
    out = list.length === 1 ? list[0] : mergeGeometries(list, false);
  } catch (e) { out = null; }
  if (!out) {
    console.warn('[props] mergeParts failed; using first part only');
    out = list[0];
  }
  for (const g of list) if (g !== out) g.dispose();
  return finish(out);
}

/* -------------------------------------------------------------------------- */
/* Batch — one merged mesh per material                                        */
/* -------------------------------------------------------------------------- */

class Batch {
  constructor() {
    this.buckets = new Map();
    this.stack = [new THREE.Matrix4()];
  }
  get top() { return this.stack[this.stack.length - 1]; }
  push(local) {
    this.stack.push(new THREE.Matrix4().multiplyMatrices(this.top, local));
    return this;
  }
  pop() { if (this.stack.length > 1) this.stack.pop(); return this; }

  add(key, material, geo, local, opts) {
    // Shadow flags are per-mesh, so a piece that must not cast gets its own
    // bucket. Otherwise one flat decal would silently disable shadow casting
    // for every other prop sharing its material.
    const cast = !(opts && opts.cast === false);
    const receive = !(opts && opts.receive === false);
    const bkey = cast && receive ? key : `${key}#${cast ? '' : 'c'}${receive ? '' : 'r'}`;
    let b = this.buckets.get(bkey);
    if (!b) { b = { material, geos: [], cast, receive }; this.buckets.set(bkey, b); }
    _m2.copy(this.top);
    if (local) _m2.multiply(local);
    b.geos.push(prep(geo, _m2, (opts && opts.uv) || 'world', (opts && opts.uvScale) ?? 1));
  }

  build(group) {
    let meshes = 0, tris = 0;
    for (const [key, b] of this.buckets) {
      if (!b.geos.length) continue;
      let merged = null;
      try {
        merged = b.geos.length === 1 ? b.geos[0] : mergeGeometries(b.geos, false);
      } catch (e) { merged = null; }
      if (!merged) {
        console.warn(`[props] merge failed for "${key}" — ${b.geos.length} separate meshes`);
        for (const g of b.geos) {
          const mesh = new THREE.Mesh(finish(g), b.material);
          mesh.castShadow = b.cast; mesh.receiveShadow = b.receive;
          group.add(mesh); meshes++;
          tris += g.index.count / 3;
        }
        continue;
      }
      for (const g of b.geos) if (g !== merged) g.dispose();
      finish(merged);
      const mesh = new THREE.Mesh(merged, b.material);
      mesh.name = `props:${key}`;
      mesh.castShadow = b.cast;
      mesh.receiveShadow = b.receive;
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      group.add(mesh);
      meshes++;
      tris += merged.index.count / 3;
    }
    this.buckets.clear();
    return { meshes, tris };
  }
}

/* -------------------------------------------------------------------------- */
/* InstancePool — one InstancedMesh per material                               */
/* -------------------------------------------------------------------------- */

class InstancePool {
  constructor(key, material, geometry, opts = {}) {
    this.key = key;
    this.material = material;
    this.geometry = geometry;
    this.matrices = [];
    this.cast = opts.cast !== false;
    this.receive = opts.receive !== false;
    this.mesh = null;
  }
  add(matrix) { this.matrices.push(matrix.clone()); }
  build(group) {
    const n = this.matrices.length;
    if (!n) { this.geometry.dispose(); return null; }
    const im = new THREE.InstancedMesh(this.geometry, this.material, n);
    for (let i = 0; i < n; i++) im.setMatrixAt(i, this.matrices[i]);
    im.instanceMatrix.needsUpdate = true;
    im.castShadow = this.cast;
    im.receiveShadow = this.receive;
    im.name = `props:${this.key}`;
    // Village-wide extent: culling would be all-or-nothing, so skip the test.
    im.frustumCulled = false;
    group.add(im);
    this.mesh = im;
    return im;
  }
}

/* -------------------------------------------------------------------------- */
/* Frame — a prop site's world transform (yaw only)                            */
/* -------------------------------------------------------------------------- */

class Frame {
  constructor(x, y, z, yaw) {
    this.yaw = yaw;
    this.matrix = placeAt(x, y, z, yaw);
  }
  /** local point -> world, as a fresh array */
  p(x, y, z) { return _v.set(x, y, z).applyMatrix4(this.matrix).toArray(); }
}

/* -------------------------------------------------------------------------- */
/* Material resolution — fail soft if the library is missing or a key is typo'd */
/* -------------------------------------------------------------------------- */

/** Radius around a hero prop that planting and clutter must stay out of. */
const KEEP_OUT_R = {
  cart: 2.5, trough: 1.75, marketStall: 2.5, bench: 1.25, logPile: 1.9,
  anvil: 0.95, grindstone: 1.2, wheelbarrow: 1.4, signpost: 0.65,
  barrelStack: 1.7, crateStack: 1.4, sackPile: 1.15, wellBucketSpare: 0.7,
};

const FALLBACK = {
  stone: [0x8e8b83, 0.94, 0], stoneTrim: [0x9b978d, 0.9, 0],
  woodDark: [0x4a3826, 0.82, 0], woodBeam: [0x6b5133, 0.8, 0],
  woodPlank: [0x8a6b45, 0.78, 0], bark: [0x51402e, 0.92, 0],
  iron: [0x3a3c40, 0.55, 0.85], ironDark: [0x24262a, 0.6, 0.85],
  brass: [0xa8823c, 0.38, 0.9], copper: [0x4e8a78, 0.55, 0.4],
  terracotta: [0xa9613c, 0.85, 0], terracottaDark: [0x83442a, 0.88, 0],
  fabricAwning: [0xd9cfc0, 0.95, 0], rope: [0xa89469, 0.95, 0],
  candleWax: [0xf0e6cc, 0.6, 0], glassLantern: [0xcfe0e6, 0.1, 0],
  lanternEmissive: [0xffbb66, 0.9, 0], soil: [0x3d2f22, 0.98, 0],
  grassBlade: [0x5c7a35, 0.95, 0], leaf: [0x4f7a33, 0.9, 0],
  leafDark: [0x37582a, 0.9, 0], ivy: [0x3f6a2e, 0.9, 0],
  flowerRed: [0xb02a22, 0.85, 0], flowerPink: [0xd77aa0, 0.85, 0],
  flowerYellow: [0xe0bc3c, 0.85, 0], flowerWhite: [0xf0ece2, 0.85, 0],
  cobbleWorn: [0x86837c, 0.9, 0], plasterWorn: [0xb9ac92, 0.95, 0],
};

function createResolver(materials) {
  const owned = [];
  const cache = new Map();
  let warned = false;

  function base(name) {
    try {
      const m = materials && materials.get && materials.get(name);
      if (m) return m;
    } catch (e) { /* fall through to a local stand-in */ }
    const k = `base:${name}`;
    let m = cache.get(k);
    if (!m) {
      if (!warned) { console.warn('[props] material library unavailable — using fallbacks'); warned = true; }
      const f = FALLBACK[name] || [0xb0a48c, 0.85, 0];
      m = new THREE.MeshStandardMaterial({ color: new THREE.Color(f[0]), roughness: f[1], metalness: f[2] });
      m.name = `props-fallback:${name}`;
      cache.set(k, m); owned.push(m);
    }
    return m;
  }

  function variant(name, opts) {
    try {
      const m = materials && materials.variant && materials.variant(name, opts);
      if (m) return m;
    } catch (e) { /* fall through */ }
    const k = variantKey(name, opts);
    let m = cache.get(k);
    if (!m) {
      m = base(name).clone();
      Object.assign(m, opts);
      m.name = `props-variant:${k}`;
      m.needsUpdate = true;
      cache.set(k, m); owned.push(m);
    }
    return m;
  }

  return {
    base, variant,
    dispose() { for (const m of owned) m.dispose(); owned.length = 0; cache.clear(); },
  };
}

/* -------------------------------------------------------------------------- */
/* Reusable part geometries owned by dynamic meshes (never from GC)            */
/* -------------------------------------------------------------------------- */

function latheProfile(pts, segments) {
  return new THREE.LatheGeometry(pts.map((p) => new THREE.Vector2(p[0], p[1])), segments);
}

/** Coopered barrel: staves that bulge, chamfered ends. Origin at the centre. */
function barrelWoodGeo(rMid, rEnd, h, seg) {
  let g = latheProfile([
    [0.02, 0], [rEnd * 0.88, 0], [rEnd, h * 0.045],
    [rMid * 0.94, h * 0.26], [rMid, h * 0.5], [rMid * 0.94, h * 0.74],
    [rEnd, h * 0.955], [rEnd * 0.88, h], [0.02, h],
  ], seg);
  g = staveify(g, seg, 0.013);
  g.translate(0, -h / 2, 0);
  return finish(prep(g, null, 'world', 1.6));
}

/** Five iron hoops. */
function barrelIronGeo(rMid, rEnd, h, seg) {
  const bands = [
    [h * 0.09, rEnd * 1.03], [h * 0.31, rMid * 1.01], [h * 0.5, rMid * 1.025],
    [h * 0.69, rMid * 1.01], [h * 0.91, rEnd * 1.03],
  ];
  return mergeParts(bands.map(([y, r]) => ({
    geo: gCyl(r, r, h * 0.055, seg, true),
    matrix: trs(0, y - h / 2, 0).clone(),
    uvScale: 3,
  })));
}

/** Boarded crate with corner battens. */
function crateGeo(s) {
  const t = s * 0.055;
  const parts = [{ geo: gBox(s, s, s), matrix: trs(0, 0, 0).clone(), uvScale: 1.8 }];
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    parts.push({ geo: gBox(t * 2, s * 1.02, t * 2), matrix: trs(sx * s / 2, 0, sz * s / 2).clone(), uvScale: 3 });
  }
  for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    parts.push({ geo: gBox(s * 1.02, t * 2, t * 2), matrix: trs(0, sy * s * 0.38, sz * s / 2).clone(), uvScale: 3 });
    parts.push({ geo: gBox(t * 2, t * 2, s * 1.02), matrix: trs(sz * s / 2, sy * s * 0.38, 0).clone(), uvScale: 3 });
  }
  return mergeParts(parts);
}

/** Tapered stave bucket with a rim and two iron bands baked in. */
function bucketGeo(rTop, rBot, h, seg) {
  let g = latheProfile([
    [0.02, 0], [rBot * 0.95, 0], [rBot, h * 0.04],
    [(rTop + rBot) * 0.5, h * 0.5],
    [rTop * 0.99, h * 0.92], [rTop * 1.06, h * 0.95], [rTop * 1.06, h],
    [rTop * 0.94, h], [rTop * 0.9, h * 0.9], [rBot * 0.86, h * 0.1], [0.02, h * 0.1],
  ], seg);
  g = staveify(g, seg, 0.015);
  g.translate(0, -h / 2, 0);
  const out = mergeParts([
    { geo: g, matrix: null, uvScale: 2.2 },
    { geo: gCyl(rTop * 1.005, rBot * 1.03, h * 0.07, seg, true), matrix: trs(0, -h * 0.18, 0).clone(), uvScale: 3 },
    { geo: gCyl(rTop * 1.02, rTop * 1.02, h * 0.06, seg, true), matrix: trs(0, h * 0.3, 0).clone(), uvScale: 3 },
  ]);
  g.dispose();
  return out;
}

/** Terracotta pot, hollow with a flared rim, open at the top for soil. */
function potGeo(r, h, seg) {
  const g = latheProfile([
    [0.02, 0], [r * 0.58, 0], [r * 0.62, h * 0.05],
    [r * 0.84, h * 0.5], [r * 0.97, h * 0.86], [r * 0.99, h * 0.9],
    [r * 1.07, h * 0.93], [r * 1.07, h], [r * 0.93, h],
    [r * 0.88, h * 0.86], [r * 0.72, h * 0.4], [r * 0.48, h * 0.12], [0.02, h * 0.12],
  ], seg);
  g.translate(0, -h / 2, 0);
  const out = finish(prep(g, null, 'world', 2.2));
  g.dispose();
  return out;
}

/** A lumpy tied sack. */
function sackGeo() {
  const g = new THREE.SphereGeometry(0.26, 12, 9);
  g.scale(1.0, 1.28, 0.82);
  const p = g.getAttribute('position');
  for (let i = 0; i < p.count; i++) {
    const y = p.getY(i);
    const f = 1 + 0.15 * Math.sin(p.getX(i) * 7.3 + y * 4.1) + 0.1 * Math.sin(p.getZ(i) * 9.7);
    const taper = y > 0.14 ? 0.5 : 1;
    p.setXYZ(i, p.getX(i) * f * taper, y, p.getZ(i) * f * taper);
  }
  g.computeVertexNormals();
  const out = mergeParts([
    { geo: g, matrix: null, uvScale: 3 },
    { geo: gCyl(0.055, 0.075, 0.1, 8), matrix: trs(0, 0.35, 0).clone(), uvScale: 4 },
  ]);
  g.dispose();
  return out;
}

/** Crossed vertical quads, base at the origin. Used for all foliage. */
function crossQuadGeo(size, blades, aspect = 1) {
  const parts = [];
  for (let i = 0; i < blades; i++) {
    const a = (i / blades) * Math.PI;
    parts.push({
      geo: gPlane(size * aspect, size),
      matrix: trs(0, size * 0.5, 0, 0, a, 0).clone(),
      uv: 'keep',
    });
  }
  return mergeParts(parts);
}

/* -------------------------------------------------------------------------- */
/* THE WELL                                                                    */
/* -------------------------------------------------------------------------- */

function buildWell(S) {
  const { B, R, colliders } = S;
  const rng = S.rng.fork('well');
  const w = WELL;
  const y0 = S.groundAt(w.position[0], w.position[2]);
  const f = new Frame(w.position[0], y0, w.position[2], w.rotation);

  const mStone = R.base('stone');
  const mTrim = R.base('stoneTrim');
  const mDark = S.mat.wellDark;
  const mWater = S.mat.water;
  const mBeam = R.base('woodBeam');
  // Light sawn plank for the drum, the crank grip and the moss-free machined
  // parts of the winch: the mechanism only reads as a mechanism if the wood is
  // pale and the fittings are dark. `woodDark` (woodPlank x 0.42) was doing both
  // jobs and the whole winch merged into the shadowed buildings behind it.
  const mLight = R.base('woodPlank');
  const mIron = R.base('ironDark');
  const mRope = R.base('rope');

  const rOut = w.outerRadius, rIn = w.innerRadius;   // 2.35 / 1.62
  const rMid = (rOut + rIn) * 0.5;
  const thick = rOut - rIn;
  const copingH = 0.085;
  const baseH = w.base.reduce((s, b) => s + b.height, 0);   // 0.33
  const wallH = w.wallHeight - copingH;
  const courses = 5;
  const courseH = wallH / courses;                          // ~0.187
  const topY = baseH + w.wallHeight;
  const beamY = w.winch.height;
  const drumY = beamY - 0.30;
  const ringSeg = Math.max(36, Math.round(64 * S.detail));

  // Two block libraries: hewn rubble for the wall, near-square dressed stone
  // for the coping. Jitter stays small — a parapet is lumpy, not spiky.
  const blocks = [];
  for (let i = 0; i < 12; i++) blocks.push(jitterBox(new THREE.BoxGeometry(1, 1, 1), 0.05, rng));
  const dressed = [];
  for (let i = 0; i < 6; i++) dressed.push(jitterBox(new THREE.BoxGeometry(1, 1, 1), 0.022, rng));

  B.push(f.matrix);
  try {
    /* --- two CIRCULAR worn steps ---------------------------------------- */
    // Wear only ever eats *into* the circle (a few mm), so the silhouette
    // stays a clean disc and nothing spreads past the authored radius.
    let stepY = 0;
    for (let i = 0; i < w.base.length; i++) {
      const b = w.base[i];
      const g = new THREE.CylinderGeometry(b.radius - 0.03, b.radius, b.height, ringSeg, 1);
      const p = g.getAttribute('position');
      for (let k = 0; k < p.count; k++) {
        const x = p.getX(k), z = p.getZ(k);
        const r = Math.hypot(x, z);
        if (r < 1e-3) continue;
        const a = Math.atan2(z, x);
        const wear = -0.006 - 0.006 * (1 + Math.sin(a * 13 + i * 1.7))
                            - 0.004 * (1 + Math.sin(a * 29 + 0.6));
        const s = 1 + wear / r;
        p.setXYZ(k, x * s, p.getY(k), z * s);
      }
      g.computeVertexNormals();
      B.add('stoneTrim', mTrim, g, trs(0, stepY + b.height / 2, 0), { uvScale: 0.8 });
      g.dispose();
      colliders.push(cylinderCollider(
        f.p(0, stepY + b.height / 2, 0), b.radius, b.height / 2, null,
        { friction: 0.95, tag: 'well-step' },
      ));
      stepY += b.height;
    }

    /* --- cylindrical parapet: 5 staggered courses of small blocks -------- */
    // Each block spans the full radial band, so the ring is watertight from
    // both faces: no hollow shell, no gaps to see the sky through.
    for (let c = 0; c < courses; c++) {
      const yb = baseH + c * courseH;
      // Random-but-exact partition of the circle: joints never line up
      // between courses and no sliver block is ever left over.
      const n = Math.round(TAU / rng.range(0.155, 0.175));   // ~36-40 blocks
      const wts = [];
      let sum = 0;
      for (let k = 0; k < n; k++) { const v = rng.range(0.72, 1.28); wts.push(v); sum += v; }
      let a = c * 0.41 + rng.sym(0.09);
      for (let k = 0; k < n; k++) {
        const span = (TAU * wts[k]) / sum;
        const mid = a + span * 0.5;
        a += span;
        const rr = rMid + rng.sym(0.015);    // ±1.5 cm proud or recessed
        B.add('stone', mStone, blocks[(k + c * 5) % blocks.length],
          trs(Math.cos(mid) * rr,
            yb + courseH * 0.5, Math.sin(mid) * rr,
            rng.sym(0.018), Math.PI / 2 - mid, rng.sym(0.018),
            span * rMid * 1.06,              // 0.22-0.40 m of arc, slight overlap
            courseH * rng.range(1.0, 1.09),  // courses overlap: no open joints
            thick * rng.range(0.94, 1.0)),
          { uvScale: 1.15 });
      }
    }

    /* --- flat coping course: level, slightly overhanging ----------------- */
    const copN = Math.max(18, Math.round((TAU * rMid) / 0.5));
    const copY = baseH + wallH;
    for (let k = 0; k < copN; k++) {
      const mid = ((k + 0.31) / copN) * TAU;
      B.add('stoneTrim', mTrim, dressed[k % dressed.length],
        trs(Math.cos(mid) * (rMid + 0.005), copY + copingH / 2, Math.sin(mid) * (rMid + 0.005),
          0, Math.PI / 2 - mid, 0,                    // level — this is what reads as "finished"
          (TAU / copN) * rMid * 1.05, copingH, thick + 0.12),
        { uvScale: 1.0 });
    }

    /* --- the shaft: darkness, then water far below ----------------------- */
    const shaftTop = baseH + 0.05;
    const shaftBottom = -5.4;
    B.add('wellDark', mDark,
      gCyl(rIn - 0.005, rIn * 0.8, shaftTop - shaftBottom, Math.max(20, Math.round(32 * S.detail)), true),
      trs(0, (shaftTop + shaftBottom) / 2, 0), { uvScale: 0.7, cast: false, receive: false });
    B.add('water', mWater, gCircle(rIn * 0.8, 24),
      trs(0, -4.72, 0, -Math.PI / 2, 0, 0), { uvScale: 0.5, cast: false, receive: false });
    B.add('wellDark', mDark, gCircle(rIn * 0.79, 20),
      trs(0, shaftBottom + 0.06, 0, -Math.PI / 2, 0, 0), { cast: false, receive: false });
    // Invisible floor well below the mouth so a knocked-in bucket is not lost
    // forever if the rope joint is not honoured.
    colliders.push(cylinderCollider(f.p(0, -0.62, 0), rIn * 0.95, 0.05, null, { tag: 'well-floor', friction: 0.9 }));

    /* --- parapet collision: a ring of boxes ------------------------------ */
    const colN = 16;
    for (let k = 0; k < colN; k++) {
      const a = (k / colN) * TAU;
      colliders.push(boxCollider(
        f.p(Math.cos(a) * rMid, baseH + (topY - baseH) / 2, Math.sin(a) * rMid),
        [(TAU * rMid / colN) * 0.62, (topY - baseH) / 2, (thick + 0.16) / 2],
        quatY(f.yaw + (Math.PI / 2 - a)),
        { tag: 'well-parapet', friction: 0.9 },
      ));
    }

    /* --- the winch: two posts, a beam, a windlass and a crank ------------ */
    const po = w.winch.postOffset;
    const postH = beamY + 0.06;
    // Heavier members: 0.17 x 0.16 m posts vanished at 6 m. A real windlass post
    // is a quarter-metre baulk of oak, and at this distance that is the
    // difference between a winch and two dark lines.
    const postW = 0.25, postD = 0.24;
    for (const sx of [-1, 1]) {
      // The posts are set through the masonry, so only the metre above the
      // coping shows — that plus the 4 m beam is the winch's silhouette.
      B.add('woodBeam', mBeam, gBox(postW, postH, postD),
        trs(sx * po, postH / 2, 0, 0, 0, sx * 0.008), { uvScale: 1.2 });
      // iron shoe where the post passes through the coping — dark band against
      // pale timber, so the eye reads "post socketed into stone".
      B.add('ironDark', mIron, gBox(postW + 0.09, 0.075, postD + 0.09),
        trs(sx * po, topY + 0.04, 0), { uvScale: 4 });
      // knee brace up to the beam
      B.add('woodBeam', mBeam, gBox(0.125, 0.86, 0.13),
        trs(sx * (po - 0.30), beamY - 0.30, 0, 0, 0, sx * 0.785), { uvScale: 1.6 });
      // iron-capped post head instead of a dark wooden pyramid
      B.add('ironDark', mIron, gCone(0.2, 0.14, 4),
        trs(sx * po, postH + 0.06, 0, 0, Math.PI / 4, 0), { uvScale: 3 });
    }
    // Beam: bottom face stays exactly at beamY (the collider and the drum hang
    // off it) — it only gets thicker upward, 0.18 -> 0.24 m.
    B.add('woodBeam', mBeam, gBox(po * 2 + 0.56, 0.24, 0.235), trs(0, beamY + 0.12, 0), { uvScale: 1.2 });
    B.add('ironDark', mIron, gBox(po * 2 + 0.66, 0.055, 0.3), trs(0, beamY + 0.265, 0), { uvScale: 1.6 });

    // windlass drum, axis along local X — pale sawn barrel, dark iron collars
    const drumL = po * 2 - 1.0;
    B.add('woodPlank', mLight, gCyl(0.15, 0.15, drumL, 14), trs(0, drumY, 0, 0, 0, Math.PI / 2), { uvScale: 2 });
    B.add('rope', mRope, gCyl(0.195, 0.195, 0.52, 14, true), trs(0.18, drumY, 0, 0, 0, Math.PI / 2), { uvScale: 8 });
    for (const sx of [-1, 1]) {
      B.add('ironDark', mIron, gCyl(0.195, 0.195, 0.075, 14),
        trs(sx * drumL / 2, drumY, 0, 0, 0, Math.PI / 2), { uvScale: 4 });
      const pinIn = drumL / 2, pinOut = po + (sx > 0 ? 0.22 : 0.04);
      B.add('ironDark', mIron, gCyl(0.042, 0.042, pinOut - pinIn, 8),
        trs(sx * (pinIn + (pinOut - pinIn) / 2), drumY, 0, 0, 0, Math.PI / 2), { uvScale: 5 });
    }
    // crank: offset arm + pale wooden grip, on the +X side
    const crankX = po + 0.2;
    B.add('ironDark', mIron, gBox(0.07, 0.44, 0.07), trs(crankX, drumY + 0.19, 0), { uvScale: 5 });
    B.add('ironDark', mIron, gCyl(0.038, 0.038, 0.14, 8), trs(crankX + 0.08, drumY + 0.38, 0, 0, 0, Math.PI / 2), { uvScale: 6 });
    B.add('woodPlank', mLight, gCyl(0.058, 0.058, 0.26, 10), trs(crankX + 0.26, drumY + 0.38, 0, 0, 0, Math.PI / 2), { uvScale: 4 });

    colliders.push(boxCollider(f.p(0, beamY + 0.09, 0), [po + 0.28, 0.15, 0.12], quatY(f.yaw), { tag: 'well-beam' }));
    for (const sx of [-1, 1]) {
      colliders.push(boxCollider(f.p(sx * po, (topY + postH) / 2, 0),
        [postW / 2 + 0.01, (postH - topY) / 2, postD / 2 + 0.01], quatY(f.yaw), { tag: 'well-post' }));
    }

    /* --- moss in the mortar joints, grass only on the earth -------------- */
    // Upright blades sprouting from the flat tops of two courses of DRESSED
    // stone read as neglect, not charm — the steps are swept every day. So:
    // blades live on the earth outside the bottom step and in the vertical
    // joints of the parapet, and the step tops get flat moss patches instead.
    const tuftN = Math.round(52 * S.vd);
    for (let i = 0; i < tuftN; i++) {
      const a = rng.range(0, TAU);
      const inJoint = rng.bool(0.34);
      if (inJoint) {
        // Hugging the outer parapet face at a course line: the blades stick
        // out sideways from the joint rather than standing on a floor.
        const course = 1 + Math.floor(rng.next() * (courses - 1));
        const yy = baseH + course * courseH - courseH * rng.range(0.0, 0.22);
        _o.position.set(Math.cos(a) * (rOut - 0.035), yy, Math.sin(a) * (rOut - 0.035));
        _o.rotation.set(rng.sym(0.35), -a + rng.sym(0.5), rng.sym(0.3));
        const s = rng.range(0.3, 0.52);
        _o.scale.set(s, s * rng.range(0.7, 1.0), s);
      } else {
        // Earth ring outside the bottom step — real grass at the well's foot.
        const r = rng.range(w.base[0].radius + 0.04, w.base[0].radius + 0.72);
        _o.position.set(Math.cos(a) * r, 0, Math.sin(a) * r);
        _o.rotation.set(0, rng.range(0, TAU), 0);
        const s = rng.range(0.55, 1.0);
        _o.scale.set(s, s * rng.range(0.8, 1.3), s);
      }
      _o.updateMatrix();
      _m.multiplyMatrices(f.matrix, _o.matrix);
      S.pools.grass.add(_m);
    }

    // Flat moss patches on the two step treads: irregular squashed discs lying
    // 12 mm above the stone, hard against the riser where damp collects.
    const mossN = Math.round(18 * S.vd);
    const mMoss = R.base('grass');
    for (let i = 0; i < mossN; i++) {
      const a = rng.range(0, TAU);
      const outer = rng.bool(0.55);
      const rIn0 = outer ? w.base[1].radius + 0.06 : rOut + 0.05;
      const rOut0 = outer ? w.base[0].radius - 0.1 : w.base[1].radius - 0.12;
      if (rOut0 <= rIn0) continue;
      const r = rng.range(rIn0, rOut0);
      const yy = (outer ? w.base[0].height : baseH) + 0.012;
      const mw = rng.range(0.17, 0.3), md = mw * rng.range(0.5, 0.85);
      B.add('mossPatch', mMoss, gCircle(1, 9),
        trs(Math.cos(a) * r, yy, Math.sin(a) * r, -Math.PI / 2, 0, rng.range(0, TAU), mw, md, 1),
        { uvScale: 2.6, cast: false });
    }
  } finally {
    // A failure anywhere above must not leave the batch stack pushed — every
    // prop built afterwards would inherit the well's transform.
    B.pop();
    for (const g of blocks) g.dispose();
    for (const g of dressed) g.dispose();
  }
  return { frame: f, drumY, beamY, topY };
}

/* -------------------------------------------------------------------------- */
/* Static prop builders                                                        */
/* -------------------------------------------------------------------------- */

/** A spoked wheel lying in the frame's XZ plane after the caller's push. */
function spokedWheel(S, r, thickness, spokes, rng) {
  const { B, R } = S;
  const rs = Math.max(14, Math.round(28 * S.detail));
  B.add('woodBeam', R.base('woodBeam'), gCyl(r, r, thickness, rs, true), null, { uvScale: 2 });
  B.add('iron', R.base('iron'), gCyl(r * 1.035, r * 1.035, thickness * 0.72, rs, true), null, { uvScale: 5 });
  B.add('woodDark', R.base('woodDark'), gCyl(r * 0.2, r * 0.2, thickness * 2.6, 10), null, { uvScale: 3 });
  B.add('iron', R.base('iron'), gCyl(r * 0.08, r * 0.08, thickness * 3.4, 8), null, { uvScale: 6 });
  for (let i = 0; i < spokes; i++) {
    const a = (i / spokes) * TAU + rng.sym(0.02);
    // box long axis along local +X, rotated to point radially outward
    B.add('woodBeam', R.base('woodBeam'), gBox(r * 0.82, thickness * 0.62, thickness * 0.5),
      trs(Math.cos(a) * r * 0.44, 0, Math.sin(a) * r * 0.44, 0, -a, 0), { uvScale: 3 });
  }
}

/** Farm cart: four spoked wheels, a tilting bed, shafts down to the ground. */
function buildCart(S, f) {
  const { B, R, colliders } = S;
  const rng = S.rng.fork('cart');
  B.push(f.matrix);
  const mPlank = R.base('woodPlank');
  const mBeam = R.base('woodBeam');
  const bedW = 1.72, bedL = 2.9, bedY = 0.92, tilt = -0.09;
  const spokes = Math.max(6, Math.round(9 * S.detail));

  for (const sx of [-1, 1]) {
    B.add('woodBeam', mBeam, gBox(0.12, 0.16, bedL), trs(sx * (bedW / 2 - 0.09), bedY - 0.1, 0, tilt, 0, 0), { uvScale: 1.2 });
  }
  for (const t of [-0.38, 0, 0.38]) {
    B.add('woodBeam', mBeam, gBox(bedW, 0.11, 0.13), trs(0, bedY - 0.1 - t * bedL * tilt, t * bedL), { uvScale: 1.2 });
  }
  const planks = 9;
  for (let i = 0; i < planks; i++) {
    const x = (i / (planks - 1) - 0.5) * (bedW - 0.06);
    B.add('woodPlank', mPlank, gBox(bedW / planks - 0.012, 0.045, bedL), trs(x, bedY, 0, tilt, 0, 0), { uvScale: 1.6 });
  }
  for (const sx of [-1, 1]) {
    B.add('woodPlank', mPlank, gBox(0.05, 0.42, bedL), trs(sx * bedW / 2, bedY + 0.2, 0, tilt, 0, sx * 0.06), { uvScale: 1.6 });
    for (let i = 0; i < 4; i++) {
      const z = (i / 3 - 0.5) * (bedL - 0.3);
      B.add('woodBeam', mBeam, gBox(0.06, 0.5, 0.07), trs(sx * (bedW / 2 + 0.02), bedY + 0.24 - z * tilt, z), { uvScale: 2 });
    }
  }
  B.add('woodPlank', mPlank, gBox(bedW, 0.44, 0.05), trs(0, bedY + 0.2 + bedL * 0.5 * tilt, bedL / 2), { uvScale: 1.6 });

  const wheels = [
    [-(bedW / 2 + 0.12), -bedL * 0.28, 0.48],
    [bedW / 2 + 0.12, -bedL * 0.28, 0.48],
    [-(bedW / 2 + 0.12), bedL * 0.3, 0.63],
    [bedW / 2 + 0.12, bedL * 0.3, 0.63],
  ];
  for (const [wx, wz, wr] of wheels) {
    B.push(trs(wx, wr, wz, 0, 0, Math.PI / 2));
    spokedWheel(S, wr, 0.075, spokes, rng);
    B.pop();
    colliders.push(cylinderCollider(f.p(wx, wr, wz), wr, 0.06,
      quatEuler(0, f.yaw, Math.PI / 2), { tag: 'cart-wheel' }));
  }

  B.add('woodDark', R.base('woodDark'), gCyl(0.06, 0.06, bedW + 0.34, 8), trs(0, 0.48, -bedL * 0.28, 0, 0, Math.PI / 2), { uvScale: 3 });
  B.add('woodDark', R.base('woodDark'), gCyl(0.07, 0.07, bedW + 0.34, 8), trs(0, 0.63, bedL * 0.3, 0, 0, Math.PI / 2), { uvScale: 3 });

  /* Shafts of an unhitched cart: they slope DOWN to the cobbles and the yoke
     bar sits across their tips. The previous sign on the X rotation tilted
     them upward instead, which left the yoke bar lying loose on the setts
     three metres in front of the cart looking like a dropped fence rail. */
  const shaftTopY = 0.66, shaftTipY = 0.075;
  const shaftNear = -bedL / 2 + 0.13, shaftFar = -bedL / 2 - 2.23;
  const shaftLen = Math.hypot(shaftFar - shaftNear, shaftTopY - shaftTipY);
  const shaftTilt = -Math.atan2(shaftTopY - shaftTipY, Math.abs(shaftFar - shaftNear));
  for (const sx of [-1, 1]) {
    B.add('woodBeam', mBeam, gBox(0.09, 0.11, shaftLen),
      trs(sx * 0.52, (shaftTopY + shaftTipY) / 2, (shaftNear + shaftFar) / 2, shaftTilt, sx * 0.04, 0),
      { uvScale: 1.6 });
    B.add('iron', R.base('iron'), gBox(0.1, 0.07, 0.17), trs(sx * 0.55, shaftTipY, shaftFar + 0.06), { uvScale: 5 });
  }
  B.add('woodBeam', mBeam, gBox(1.26, 0.085, 0.085), trs(0, shaftTipY + 0.05, shaftFar + 0.1), { uvScale: 2 });

  colliders.push(boxCollider(f.p(0, bedY + 0.12, 0), [bedW / 2 + 0.06, 0.36, bedL / 2], quatY(f.yaw), { tag: 'cart' }));
  colliders.push(boxCollider(f.p(0, 0.33, (shaftNear + shaftFar) / 2), [0.68, 0.34, 1.22], quatY(f.yaw), { tag: 'cart-shafts' }));
  B.pop();
}

function buildLogPile(S, f) {
  const { B, R, colliders } = S;
  const rng = S.rng.fork('logs');
  B.push(f.matrix);
  const mBark = R.base('bark');
  const mCut = R.base('woodPlank');   // pale sawn end, and it merges with the fences
  const rows = [5, 4, 3, 2];
  let y = 0;
  for (let r = 0; r < rows.length; r++) {
    const n = rows[r];
    const rad = 0.17 + rng.sym(0.01);
    y += rad * (r === 0 ? 1 : 1.72);
    for (let i = 0; i < n; i++) {
      const x = (i - (n - 1) / 2) * rad * 2.1 + (r % 2 ? rad * 0.4 : 0);
      const len = rng.range(1.5, 2.2);
      B.add('bark', mBark, gCyl(rad, rad * 0.94, len, 9, true),
        trs(x, y, rng.sym(0.06), 0, rng.sym(0.05), Math.PI / 2), { uvScale: 2.2 });
      for (const sz of [-1, 1]) {
        B.add('woodPlank', mCut, gCircle(rad * 0.95, 9),
          trs(x - sz * len / 2, y, rng.sym(0.06), 0, sz > 0 ? -Math.PI / 2 : Math.PI / 2, 0), { uvScale: 3 });
      }
    }
  }
  for (let i = 0; i < 5; i++) {
    B.add('bark', mBark, gCyl(0.11, 0.1, rng.range(0.4, 0.66), 7, true),
      trs(rng.range(-1.6, 1.6), 0.11, rng.range(1.0, 1.9), 0, rng.range(0, TAU), Math.PI / 2), { uvScale: 3 });
  }
  colliders.push(boxCollider(f.p(0, y / 2 + 0.1, 0), [1.35, y / 2 + 0.2, 0.55], quatY(f.yaw), { tag: 'logpile' }));
  B.pop();
}

function buildTrough(S, f) {
  const { B, R, colliders } = S;
  B.push(f.matrix);
  const mStone = R.base('stone');
  const L = 2.5, W = 0.86, H = 0.62, t = 0.13;
  B.add('stone', mStone, gBox(L, t, W), trs(0, t / 2, 0), { uvScale: 0.9 });
  for (const sz of [-1, 1]) B.add('stone', mStone, gBox(L, H, t), trs(0, H / 2, sz * (W / 2 - t / 2)), { uvScale: 0.9 });
  for (const sx of [-1, 1]) B.add('stone', mStone, gBox(t, H, W - t * 2), trs(sx * (L / 2 - t / 2), H / 2, 0), { uvScale: 0.9 });
  // Same flags as the well's water so both share one merged mesh.
  B.add('water', S.mat.water, gPlane(L - t * 2.2, W - t * 2.2),
    trs(0, H - 0.13, 0, -Math.PI / 2, 0, 0), { cast: false, receive: false });
  for (const sx of [-1, 1]) B.add('stone', mStone, gBox(0.4, 0.16, W * 0.8), trs(sx * L * 0.3, 0.05, 0), { uvScale: 1.2 });
  colliders.push(boxCollider(f.p(0, H / 2, 0), [L / 2, H / 2, W / 2], quatY(f.yaw), { tag: 'trough' }));
  B.pop();
}

function buildBench(S, f) {
  const { B, R, colliders } = S;
  B.push(f.matrix);
  const mPlank = R.base('woodPlank');
  const mBeam = R.base('woodBeam');
  const L = 1.9, seatY = 0.46;
  for (const sx of [-1, 1]) {
    B.add('woodBeam', mBeam, gBox(0.1, seatY, 0.46), trs(sx * (L / 2 - 0.16), seatY / 2, 0), { uvScale: 1.6 });
    B.add('woodBeam', mBeam, gBox(0.08, 0.62, 0.09), trs(sx * (L / 2 - 0.16), seatY + 0.3, -0.19), { uvScale: 2 });
  }
  for (let i = 0; i < 3; i++) {
    B.add('woodPlank', mPlank, gBox(L, 0.045, 0.14), trs(0, seatY, (i - 1) * 0.16), { uvScale: 1.8 });
  }
  B.add('woodPlank', mPlank, gBox(L, 0.16, 0.04), trs(0, seatY + 0.44, -0.21), { uvScale: 1.8 });
  B.add('woodPlank', mPlank, gBox(L, 0.14, 0.04), trs(0, seatY + 0.24, -0.2), { uvScale: 1.8 });
  B.add('woodBeam', mBeam, gBox(L - 0.2, 0.06, 0.06), trs(0, 0.14, 0), { uvScale: 2 });
  colliders.push(boxCollider(f.p(0, seatY / 2 + 0.02, 0), [L / 2, seatY / 2 + 0.04, 0.24], quatY(f.yaw), { tag: 'bench' }));
  colliders.push(boxCollider(f.p(0, seatY + 0.34, -0.2), [L / 2, 0.34, 0.06], quatY(f.yaw), { tag: 'bench-back' }));
  B.pop();
}

function buildMarketStall(S, f, idx) {
  const { B, R, colliders } = S;
  const rng = S.rng.fork(`stall${idx}`);
  B.push(f.matrix);
  const mBeam = R.base('woodBeam');
  const mPlank = R.base('woodPlank');
  const W = 3.2, D = 2.0, postH = 2.24;

  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const h = postH + (sz < 0 ? 0.18 : 0);
    B.add('woodBeam', mBeam, gBox(0.1, h, 0.1), trs(sx * W / 2, h / 2, sz * D / 2), { uvScale: 1.6 });
    colliders.push(boxCollider(f.p(sx * W / 2, h / 2, sz * D / 2), [0.06, h / 2, 0.06], quatY(f.yaw), { tag: 'stall-post' }));
  }
  for (const sz of [-1, 1]) {
    B.add('woodBeam', mBeam, gBox(W + 0.16, 0.08, 0.08), trs(0, postH + (sz < 0 ? 0.18 : 0), sz * D / 2), { uvScale: 2 });
  }

  /* canopy — a real sagging cloth, not a flat plane */
  const segU = Math.max(7, Math.round(18 * S.detail));
  const segV = Math.max(5, Math.round(12 * S.detail));
  const cloth = new THREE.PlaneGeometry(W + 0.7, D + 0.9, segU, segV);
  const cp = cloth.getAttribute('position');
  for (let i = 0; i < cp.count; i++) {
    const x = cp.getX(i), y = cp.getY(i);
    const u = x / (W + 0.7), v = y / (D + 0.9);
    const slope = -(0.5 - v) * 0.46;                       // low at the front
    const sag = -0.17 * (1 - 4 * u * u) * (0.45 + 0.55 * (v + 0.5));
    const ripple = 0.028 * Math.sin(u * 13.0 + idx) * (v + 0.5);
    cp.setZ(i, slope + sag + ripple);
  }
  cloth.computeVertexNormals();
  const uvc = cloth.getAttribute('uv');
  for (let i = 0; i < uvc.count; i++) uvc.setXY(i, uvc.getX(i) * 5.5, uvc.getY(i) * 2.0);
  B.add('awning', S.mat.awning, cloth, trs(0, postH + 0.26, 0.1, -Math.PI / 2, 0, 0), { uv: 'keep' });
  cloth.dispose();
  B.add('awning', S.mat.awning, gPlane(W + 0.7, 0.24), trs(0, postH - 0.12, D / 2 + 0.44, 0.24, 0, 0), { uv: 'keep' });

  /* trestle table */
  const tY = 0.82;
  for (const sx of [-1, 1]) {
    for (const d of [-1, 1]) {
      B.add('woodBeam', mBeam, gBox(0.07, tY, 0.07), trs(sx * (W / 2 - 0.35) + d * 0.16, tY / 2, -0.1 + d * 0.2, 0, 0, -d * 0.14), { uvScale: 2 });
    }
    B.add('woodBeam', mBeam, gBox(0.06, 0.06, 0.62), trs(sx * (W / 2 - 0.35), 0.22, -0.1), { uvScale: 2 });
  }
  for (let i = 0; i < 5; i++) {
    B.add('woodPlank', mPlank, gBox(W - 0.3, 0.04, 0.19), trs(0, tY, -0.1 + (i - 2) * 0.2), { uvScale: 1.8 });
  }
  colliders.push(boxCollider(f.p(0, tY, -0.1), [(W - 0.3) / 2, 0.07, 0.5], quatY(f.yaw), { tag: 'stall-table' }));
  // trestle legs only — the space under the table stays open for crates
  for (const sx of [-1, 1]) {
    colliders.push(boxCollider(f.p(sx * (W / 2 - 0.35), 0.4, -0.1), [0.17, 0.4, 0.3], quatY(f.yaw), { tag: 'stall-trestle' }));
  }

  /* produce */
  for (let i = 0; i < 3; i++) {
    const x = (i - 1) * 0.78 + rng.sym(0.05);
    B.add('woodPlank', mPlank, S.geo.crateSmall, trs(x, tY + 0.22, -0.06, 0, rng.sym(0.2), 0), { uvScale: 1.8 });
    const heap = Math.round(7 * S.detail) + 4;
    for (let k = 0; k < heap; k++) {
      const a = rng.range(0, TAU), r = rng.range(0, 0.15);
      B.add('apple', S.mat.apple, gSph(0.05, 7, 5),
        trs(x + Math.cos(a) * r, tY + 0.45 + rng.range(0, 0.05), -0.06 + Math.sin(a) * r), { uvScale: 4, cast: false });
    }
  }
  for (let i = 0; i < 2; i++) {
    B.add('woodPlank', mPlank, S.geo.crate, trs(-W / 2 - 0.44, 0.29 + i * 0.59, D * 0.18, 0, rng.sym(0.35), 0), { uvScale: 1.8 });
  }
  colliders.push(boxCollider(f.p(-W / 2 - 0.44, 0.59, D * 0.18), [0.3, 0.59, 0.3], quatY(f.yaw), { tag: 'crates' }));
  B.pop();
}

function buildWheelbarrow(S, f) {
  const { B, R, colliders } = S;
  const rng = S.rng.fork('barrow');
  B.push(f.matrix);
  const mPlank = R.base('woodPlank');
  const mBeam = R.base('woodBeam');
  const trayY = 0.52;
  B.add('woodPlank', mPlank, gBox(0.62, 0.04, 0.94), trs(0, trayY, 0), { uvScale: 2 });
  for (const sx of [-1, 1]) B.add('woodPlank', mPlank, gBox(0.05, 0.3, 0.94), trs(sx * 0.33, trayY + 0.14, 0, 0, 0, sx * 0.22), { uvScale: 2 });
  B.add('woodPlank', mPlank, gBox(0.72, 0.3, 0.05), trs(0, trayY + 0.14, -0.47, -0.2, 0, 0), { uvScale: 2 });
  B.add('woodPlank', mPlank, gBox(0.5, 0.24, 0.05), trs(0, trayY + 0.11, 0.47, 0.24, 0, 0), { uvScale: 2 });
  for (const sx of [-1, 1]) {
    // Handles rise toward the back; the two rear legs actually reach the
    // ground, so the barrow rests on wheel + legs instead of levitating.
    B.add('woodBeam', mBeam, gBox(0.06, 0.07, 1.95), trs(sx * 0.3, trayY - 0.05, 0.42, -0.1, 0, 0), { uvScale: 2 });
    B.add('woodBeam', mBeam, gBox(0.055, trayY - 0.02, 0.055), trs(sx * 0.3, (trayY - 0.02) / 2, 0.34, 0.05, 0, 0), { uvScale: 2.5 });
  }
  B.push(trs(0, 0.29, -0.98, 0, 0, Math.PI / 2));
  spokedWheel(S, 0.29, 0.07, 6, rng);
  B.pop();
  colliders.push(boxCollider(f.p(0, trayY, -0.1), [0.4, 0.28, 0.7], quatY(f.yaw), { tag: 'wheelbarrow' }));
  colliders.push(cylinderCollider(f.p(0, 0.29, -0.98), 0.3, 0.06, quatEuler(0, f.yaw, Math.PI / 2), { tag: 'barrow-wheel' }));
  B.pop();
}

function buildAnvil(S, f) {
  const { B, R, colliders } = S;
  B.push(f.matrix);
  const mIron = R.base('iron');
  const mDark = R.base('ironDark');
  const stumpH = 0.56;
  B.add('bark', R.base('bark'), gCyl(0.31, 0.34, stumpH, 12), trs(0, stumpH / 2, 0), { uvScale: 1.6 });
  B.add('woodPlank', R.base('woodPlank'), gCircle(0.31, 12), trs(0, stumpH + 0.003, 0, -Math.PI / 2, 0, 0), { uvScale: 3 });
  const aY = stumpH;
  B.add('ironDark', mDark, gBox(0.42, 0.09, 0.24), trs(0, aY + 0.045, 0), { uvScale: 3 });
  B.add('ironDark', mDark, gBox(0.2, 0.17, 0.14), trs(0, aY + 0.16, 0), { uvScale: 3 });
  B.add('iron', mIron, gBox(0.62, 0.1, 0.2), trs(0, aY + 0.3, 0), { uvScale: 3 });
  B.add('iron', mIron, gCone(0.09, 0.34, 8), trs(0.44, aY + 0.3, 0, 0, 0, -Math.PI / 2), { uvScale: 4 });
  B.add('iron', mIron, gBox(0.1, 0.1, 0.16), trs(-0.36, aY + 0.3, 0), { uvScale: 4 });
  B.add('woodBeam', R.base('woodBeam'), gCyl(0.02, 0.024, 0.32, 7), trs(-0.02, aY + 0.38, 0.02, 0, 0.6, Math.PI / 2), { uvScale: 4 });
  B.add('ironDark', mDark, gBox(0.12, 0.05, 0.05), trs(0.13, aY + 0.38, -0.09, 0, 0.6, 0), { uvScale: 5 });
  colliders.push(cylinderCollider(f.p(0, stumpH / 2, 0), 0.33, stumpH / 2, null, { tag: 'anvil-stump' }));
  colliders.push(boxCollider(f.p(0, aY + 0.2, 0), [0.31, 0.2, 0.13], quatY(f.yaw), { tag: 'anvil' }));
  B.pop();
}

function buildGrindstone(S, f) {
  const { B, R, colliders } = S;
  B.push(f.matrix);
  const mBeam = R.base('woodBeam');
  const mIron = R.base('iron');
  const axleY = 0.86;
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      B.add('woodBeam', mBeam, gBox(0.1, axleY, 0.1), trs(sx * 0.42, axleY / 2, sz * 0.3, 0, 0, sx * 0.09), { uvScale: 1.8 });
    }
    B.add('woodBeam', mBeam, gBox(0.09, 0.09, 0.72), trs(sx * 0.42, axleY - 0.03, 0), { uvScale: 2 });
  }
  for (const sz of [-1, 1]) B.add('woodBeam', mBeam, gBox(0.95, 0.08, 0.08), trs(0, 0.14, sz * 0.3), { uvScale: 2 });
  B.add('stone', R.base('stone'), gCyl(0.44, 0.44, 0.11, Math.max(16, Math.round(30 * S.detail))), trs(0, axleY, 0, 0, 0, Math.PI / 2), { uvScale: 1.6 });
  B.add('iron', mIron, gCyl(0.028, 0.028, 1.0, 8), trs(0, axleY, 0, 0, 0, Math.PI / 2), { uvScale: 6 });
  B.add('iron', mIron, gBox(0.04, 0.22, 0.04), trs(0.52, axleY + 0.09, 0), { uvScale: 6 });
  B.add('woodDark', R.base('woodDark'), gCyl(0.035, 0.035, 0.14, 8), trs(0.6, axleY + 0.18, 0, 0, 0, Math.PI / 2), { uvScale: 4 });
  B.add('stone', R.base('stone'), gBox(0.5, 0.14, 0.34), trs(0, 0.28, 0), { uvScale: 2 });
  colliders.push(boxCollider(f.p(0, axleY / 2 + 0.1, 0), [0.52, axleY / 2 + 0.28, 0.42], quatY(f.yaw), { tag: 'grindstone' }));
  B.pop();
}

/**
 * A painted finger board: a plank whose far end comes to a point. Extruded from
 * a Shape so the arrow is a real silhouette rather than a cone stuck on the end
 * (the old version pointed its cone back at the post).
 */
function fingerBoardGeo(len, h, thick) {
  const tip = h * 0.85;
  const s = new THREE.Shape();
  s.moveTo(0, -h / 2);
  s.lineTo(len - tip, -h / 2);
  s.lineTo(len, 0);
  s.lineTo(len - tip, h / 2);
  s.lineTo(0, h / 2);
  s.closePath();
  const g = new THREE.ExtrudeGeometry(s, { depth: thick, bevelEnabled: false, curveSegments: 1, steps: 1 });
  g.translate(0, 0, -thick / 2);
  return g;
}

function buildSignpost(S, f) {
  const { B, R, colliders } = S;
  B.push(f.matrix);
  const H = 2.6;
  const mCream = S.mat.paintCream;
  const mIron = R.base('ironDark');
  B.add('woodBeam', R.base('woodBeam'), gCyl(0.085, 0.11, H, 8), trs(0, H / 2, 0), { uvScale: 1.6 });
  B.add('stone', R.base('stone'), gCyl(0.26, 0.33, 0.2, 10), trs(0, 0.1, 0), { uvScale: 2 });
  B.add('ironDark', mIron, gCyl(0.1, 0.1, 0.045, 10), trs(0, 0.23, 0), { uvScale: 6 });
  B.add('woodDark', R.base('woodDark'), gCone(0.14, 0.2, 8), trs(0, H + 0.09, 0), { uvScale: 3 });

  const bh = 0.26, bt = 0.05;
  const fingers = [
    { y: H - 0.28, a: 0.0, len: 1.15 },
    { y: H - 0.66, a: 2.32, len: 1.0 },
    { y: H - 1.04, a: 4.12, len: 1.08 },
  ];
  for (const fi of fingers) {
    const g = fingerBoardGeo(fi.len, bh, bt);
    B.push(trs(0, fi.y, 0, 0, fi.a, -0.035));
    B.add('paintCream', mCream, g, trs(0.09, 0, 0), { uvScale: 2.0 });
    // iron strap and bolt fixing the board to the post
    B.add('ironDark', mIron, gBox(0.24, bh + 0.03, 0.012), trs(0.16, 0, bt / 2 + 0.007), { uvScale: 6 });
    B.add('ironDark', mIron, gCyl(0.018, 0.018, bt + 0.05, 6), trs(0.19, 0, 0, Math.PI / 2, 0, 0), { uvScale: 8 });
    // Painted lettering: three word-blocks a side, readable as text at a glance.
    for (const sz of [-1, 1]) {
      const zz = sz * (bt / 2 + 0.004);
      for (let k = 0; k < 3; k++) {
        const wlen = [0.30, 0.22, 0.17][k];
        const x0 = 0.30 + [0, 0.34, 0.60][k];
        if (x0 + wlen > fi.len - bh * 0.5) continue;
        B.add('ironDark', mIron, gBox(wlen, 0.055, 0.008), trs(x0 + wlen / 2, 0.012, zz), { uvScale: 8 });
      }
    }
    B.pop();
    g.dispose();
  }
  colliders.push(cylinderCollider(f.p(0, H / 2, 0), 0.13, H / 2, null, { tag: 'signpost' }));
  B.pop();
}

/** Wrought-iron lamp post, glazed lantern head, lit candle. */
function buildLampPost(S, pos, height, idx) {
  const { B, R, colliders, lightAnchors } = S;
  const [lx, lz] = clearPoint(pos[0], pos[2], 0.5);
  const f = new Frame(lx, S.groundAt(lx, lz), lz, (idx * 1.37) % TAU);
  S.keepOut.push({ x: lx, z: lz, r: 0.75 });
  B.push(f.matrix);
  const mIron = R.base('ironDark');
  const mBrass = R.base('brass');
  const H = height;

  /* --- plinth and moulded iron base ------------------------------------- */
  B.add('stoneTrim', R.base('stoneTrim'), gCyl(0.28, 0.35, 0.19, 10), trs(0, 0.085, 0), { uvScale: 2 });
  B.add('ironDark', mIron, gCyl(0.15, 0.23, 0.3, 10), trs(0, 0.31, 0), { uvScale: 3 });
  B.add('ironDark', mIron, gCyl(0.105, 0.15, 0.14, 10), trs(0, 0.53, 0), { uvScale: 3 });

  /* --- tapered shaft with collars (not a smooth modern pole) ------------- */
  const shaftTop = H - 0.66;
  B.add('ironDark', mIron, gCyl(0.052, 0.098, shaftTop - 0.56, 8), trs(0, (shaftTop + 0.56) / 2, 0), { uvScale: 2.5 });
  for (const cy of [0.95, 1.9]) {
    B.add('ironDark', mIron, gCyl(0.078, 0.078, 0.06, 10), trs(0, cy, 0), { uvScale: 6 });
  }
  B.add('ironDark', mIron, gCyl(0.088, 0.062, 0.1, 10), trs(0, shaftTop + 0.04, 0), { uvScale: 6 });

  /* --- the lantern head: solid pan, thick frame, glowing panes, peaked cap */
  const hw = 0.15, hh = 0.36;
  const headY = shaftTop + 0.29 + hh / 2;
  const panY = headY - hh / 2;
  // Spigot: the shaft has to actually reach the lantern, or the head floats.
  B.add('ironDark', mIron, gCyl(0.05, 0.062, panY - 0.06 - (shaftTop + 0.05), 8),
    trs(0, (panY - 0.06 + shaftTop + 0.05) / 2, 0), { uvScale: 4 });

  /* --- scrolled brackets under the pan ----------------------------------- */
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * TAU;
    B.push(trs(0, panY - 0.07, 0, 0, -a, 0));
    B.add('ironDark', mIron, gBox(0.24, 0.032, 0.03), trs(0.115, 0.035, 0, 0, 0, 0.42), { uvScale: 8 });
    B.add('ironDark', mIron, gBox(0.15, 0.03, 0.028), trs(0.2, -0.11, 0, 0, 0, 1.15), { uvScale: 8 });
    B.pop();
  }

  B.add('ironDark', mIron, gCyl(0.26, 0.2, 0.075, 4), trs(0, panY - 0.03, 0, 0, Math.PI / 4, 0), { uvScale: 4 });
  B.add('ironDark', mIron, gCyl(0.235, 0.235, 0.035, 4), trs(0, panY + 0.02, 0, 0, Math.PI / 4, 0), { uvScale: 5 });
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * TAU + Math.PI / 4;
    B.add('ironDark', mIron, gBox(0.036, hh, 0.036),
      trs(Math.cos(a) * hw * 1.41, headY, Math.sin(a) * hw * 1.41, 0, -a, 0), { uvScale: 10 });
  }
  // Glowing glazed panes — warm, near-opaque, so the lantern still reads as a
  // lit lantern from across the plaza instead of vanishing into the ironwork.
  for (let i = 0; i < 4; i++) {
    const a = (i / 4) * TAU;
    B.add('lanternGlass', S.mat.lanternGlass, gPlane(hw * 1.94, hh * 0.9),
      trs(Math.cos(a) * hw, headY, Math.sin(a) * hw, 0, -a + Math.PI / 2, 0),
      { uv: 'keep', cast: false });
  }
  B.add('ironDark', mIron, gCyl(0.225, 0.225, 0.04, 4), trs(0, headY + hh / 2 + 0.015, 0, 0, Math.PI / 4, 0), { uvScale: 5 });
  B.add('ironDark', mIron, gCone(0.3, 0.21, 4), trs(0, headY + hh / 2 + 0.14, 0, 0, Math.PI / 4, 0), { uvScale: 5 });
  B.add('brass', mBrass, gSph(0.048, 8, 6), trs(0, headY + hh / 2 + 0.28, 0), { uvScale: 8 });
  B.add('brass', mBrass, gCyl(0.014, 0.014, 0.1, 6), trs(0, headY + hh / 2 + 0.35, 0), { uvScale: 8 });

  /* --- candle and its brass drip pan ------------------------------------ */
  const candleY = panY + 0.13;
  B.add('brass', mBrass, gCyl(0.055, 0.068, 0.022, 8), trs(0, candleY - 0.09, 0), { uvScale: 10 });
  B.add('brass', mBrass, gCyl(0.026, 0.03, 0.15, 8), trs(0, candleY, 0), { uvScale: 6 });

  const flameY = candleY + 0.09;
  _m.multiplyMatrices(f.matrix, trs(0, flameY, 0));
  S.pools.flame.add(_m);

  colliders.push(cylinderCollider(f.p(0, (H - 0.4) / 2 + 0.2, 0), 0.11, (H - 0.4) / 2 + 0.2, null, { tag: 'lamp-post' }));
  lightAnchors.push({
    position: f.p(0, flameY + 0.05, 0),
    kind: 'lantern',
    color: 0xffb066, colour: 0xffb066,
    intensity: 2.2, distance: 11, radius: 11, decay: 2,
  });
  B.pop();
}

/* -------------------------------------------------------------------------- */
/* Planting                                                                    */
/* -------------------------------------------------------------------------- */

/** Blossom + leaves in a dome over a circular opening. Caller has pushed. */
function plantCluster(S, worldMatrix, radius, height, density, rng, palette) {
  const { B, R } = S;
  B.add('soil', R.base('soil'), gCircle(radius * 0.96, 12), trs(0, 0, 0, -Math.PI / 2, 0, 0), { uvScale: 3, cast: false });
  const n = Math.max(3, Math.round(density * S.vd));
  for (let i = 0; i < n; i++) {
    const a = rng.range(0, TAU);
    const rr = radius * Math.sqrt(rng.next()) * 0.95;
    const px = Math.cos(a) * rr, pz = Math.sin(a) * rr;
    const droop = rng.bool(0.25) ? rng.range(0.2, 0.5) : 0;
    const py = height * rng.range(0.3, 1.0) - droop * height * 0.5;
    if (rng.bool(0.85)) {
      _o.position.set(px * 1.05, py * 0.5, pz * 1.05);
      _o.rotation.set(rng.sym(0.45), rng.range(0, TAU), rng.sym(0.45));
      const ls = rng.range(0.9, 1.6) * radius * 1.4;
      _o.scale.set(ls, ls, ls);
      _o.updateMatrix();
      _m.multiplyMatrices(worldMatrix, _o.matrix);
      S.pools.leaf.add(_m);
    }
    _o.position.set(px, py, pz);
    _o.rotation.set(rng.sym(0.5), rng.range(0, TAU), rng.sym(0.5));
    const s = rng.range(0.7, 1.25) * radius * 1.3;
    _o.scale.set(s, s, s);
    _o.updateMatrix();
    _m.multiplyMatrices(worldMatrix, _o.matrix);
    S.pools[rng.pick(palette)].add(_m);
  }
}

/** Blossom across a rectangular opening (window boxes, hanging baskets). */
function spreadBlossom(S, worldMatrix, hx, hz, height, count, rng, palette) {
  const n = Math.max(3, Math.round(count * S.vd));
  for (let i = 0; i < n; i++) {
    const px = rng.sym(hx), pz = rng.sym(hz);
    const drop = rng.bool(0.32) ? -rng.range(0.06, 0.3) : 0;
    if (rng.bool(0.8)) {
      _o.position.set(px, height * 0.3 + drop * 0.6, pz);
      _o.rotation.set(rng.sym(0.6), rng.range(0, TAU), rng.sym(0.6));
      const ls = rng.range(0.17, 0.3);
      _o.scale.set(ls, ls, ls);
      _o.updateMatrix();
      _m.multiplyMatrices(worldMatrix, _o.matrix);
      S.pools.leaf.add(_m);
    }
    _o.position.set(px, height * rng.range(0.4, 1.0) + drop, pz);
    _o.rotation.set(rng.sym(0.5), rng.range(0, TAU), rng.sym(0.5));
    const s = rng.range(0.13, 0.24);
    _o.scale.set(s, s, s);
    _o.updateMatrix();
    _m.multiplyMatrices(worldMatrix, _o.matrix);
    S.pools[rng.pick(palette)].add(_m);
  }
}

/** A terracotta pot standing on the ground, planted. */
function addPot(S, worldMatrix, size, rng, palette) {
  const { B, R, colliders } = S;
  const r = size, h = size * 1.12;
  const g = potGeo(r, h, Math.max(10, Math.round(20 * S.detail)));
  const dark = rng.bool(0.3);
  B.push(worldMatrix);
  B.add(dark ? 'terracottaDark' : 'terracotta', R.base(dark ? 'terracottaDark' : 'terracotta'),
    g, trs(0, h / 2, 0, 0, rng.range(0, TAU), 0), { uvScale: 2 });
  const soilLocal = trs(0, h * 0.92, 0).clone();
  B.push(soilLocal);
  plantCluster(S, new THREE.Matrix4().multiplyMatrices(worldMatrix, soilLocal),
    r * 0.8, size * 1.6, 17, rng, palette);
  B.pop();
  B.pop();
  g.dispose();
  _v.setFromMatrixPosition(worldMatrix);
  colliders.push(cylinderCollider([_v.x, _v.y + h / 2, _v.z], r * 1.05, h / 2, null, { tag: 'pot' }));
}

/** Wooden window box — only built when the buildings stream supplied no anchors. */
function addWindowBox(S, worldMatrix, width, rng, palette) {
  const { B, R } = S;
  const w = width, d = 0.24, h = 0.2;
  B.push(worldMatrix);
  const mDark = R.base('woodDark');
  B.add('woodDark', mDark, gBox(w, h, d), trs(0, h / 2, 0), { uvScale: 2.6 });
  B.add('woodDark', mDark, gBox(w + 0.06, 0.03, d + 0.06), trs(0, h, 0), { uvScale: 2.6 });
  for (const sx of [-1, 1]) {
    B.add('ironDark', R.base('ironDark'), gBox(0.02, 0.14, d + 0.06), trs(sx * w * 0.34, h - 0.07, 0), { uvScale: 10 });
  }
  const soilLocal = trs(0, h * 0.94, 0).clone();
  B.push(soilLocal);
  B.add('soil', R.base('soil'), gPlane(w * 0.9, d * 0.8), trs(0, 0, 0, -Math.PI / 2, 0, 0), { uvScale: 4, cast: false });
  B.pop();
  B.pop();
  spreadBlossom(S, new THREE.Matrix4().multiplyMatrices(worldMatrix, soilLocal),
    w * 0.44, d * 0.34, 0.3, 23, rng, palette);
}

/** Climbing rose: stems up a wall plus a heavy load of foliage and blossom. */
function addClimber(S, plot, lx, rng, palette) {
  const { B, R } = S;
  const wallZ = plot.depth / 2 + 0.07;
  const top = Math.min(plot.storeys * plot.storeyHeight - 0.5, 5.2);
  const base = plotToWorld(plot, lx, 0, wallZ).toArray();
  const fr = new Frame(base[0], S.groundAt(base[0], base[2]), base[2], plot.rotation);
  B.push(fr.matrix);
  const steps = Math.max(5, Math.round(9 * S.detail));
  for (let s = 0; s < 3; s++) {
    let x = rng.sym(0.22), y = 0;
    const sway = rng.range(0.1, 0.3), ph = rng.range(0, TAU);
    for (let i = 0; i < steps; i++) {
      const h = top / steps;
      const nx = x + Math.sin(ph + y * 1.6) * sway * h;
      B.add('woodDark', R.base('woodDark'), gCyl(0.014, 0.018, h * 1.06, 5),
        trs((x + nx) / 2, y + h / 2, 0, 0, 0, Math.atan2(x - nx, h)), { uvScale: 8 });
      x = nx; y += h;
    }
  }
  B.pop();
  const n = Math.round(82 * S.vd);
  for (let i = 0; i < n; i++) {
    const px = rng.sym(0.62);
    const py = rng.range(0.25, top);
    _o.position.set(px, py, rng.range(0.0, 0.12));
    _o.rotation.set(rng.sym(0.6), rng.range(0, TAU), rng.sym(0.6));
    const ls = rng.range(0.2, 0.4);
    _o.scale.set(ls, ls, ls);
    _o.updateMatrix();
    _m.multiplyMatrices(fr.matrix, _o.matrix);
    S.pools.ivy.add(_m);
    if (rng.bool(0.72)) {
      _o.position.set(px + rng.sym(0.16), py + rng.sym(0.12), rng.range(0.04, 0.18));
      _o.rotation.set(rng.sym(0.5), rng.range(0, TAU), rng.sym(0.5));
      const s = rng.range(0.14, 0.26);
      _o.scale.set(s, s, s);
      _o.updateMatrix();
      _m.multiplyMatrices(fr.matrix, _o.matrix);
      S.pools[rng.pick(palette)].add(_m);
    }
  }
}

/** Hanging baskets under the inn's arcade. */
function addHangingBaskets(S, plot, rng) {
  const { B, R } = S;
  const arc = plot.arcade;
  if (!arc) return 0;
  let made = 0;
  const y = 2.36;
  const lz = plot.depth / 2 + arc.depth * 0.82;
  for (let i = 0; i < arc.bays; i++) {
    if (i % 2 === 1) continue;
    const t = arc.bays === 1 ? 0 : (i / (arc.bays - 1)) * 2 - 1;
    const lx = t * (plot.width / 2 - 1.5);
    const wp = plotToWorld(plot, lx, 0, lz).toArray();
    const fr = new Frame(wp[0], S.groundAt(wp[0], wp[2]) + y, wp[2], plot.rotation);
    B.push(fr.matrix);
    for (let c = 0; c < 3; c++) {
      const a = (c / 3) * TAU;
      B.add('ironDark', R.base('ironDark'), gCyl(0.006, 0.006, 0.34, 4),
        trs(Math.cos(a) * 0.09, -0.17, Math.sin(a) * 0.09, Math.cos(a) * 0.16, 0, -Math.sin(a) * 0.16), { uvScale: 12 });
    }
    B.add('ironDark', R.base('ironDark'), gCyl(0.008, 0.008, 0.1, 4), trs(0, 0.05, 0), { uvScale: 12 });
    const bg = potGeo(0.2, 0.17, Math.max(9, Math.round(18 * S.detail)));
    B.add('rope', R.base('rope'), bg, trs(0, -0.42, 0), { uvScale: 5 });
    bg.dispose();
    B.pop();
    spreadBlossom(S, new THREE.Matrix4().multiplyMatrices(fr.matrix, trs(0, -0.36, 0)),
      0.17, 0.17, 0.16, 27, rng, ['flowerPink', 'flowerWhite', 'flowerRed']);
    made++;
  }
  return made;
}

/** Lattice trellis flat against a wall, with ivy on it. */
function addTrellis(S, plot, lx, rng) {
  const { B, R } = S;
  const wp = plotToWorld(plot, lx, 0, plot.depth / 2 + 0.08).toArray();
  const fr = new Frame(wp[0], S.groundAt(wp[0], wp[2]), wp[2], plot.rotation);
  B.push(fr.matrix);
  const m = R.base('woodDark');
  const W = 1.35, H = 2.1;
  for (const sx of [-1, 1]) B.add('woodDark', m, gBox(0.045, H, 0.035), trs(sx * W / 2, H / 2, 0), { uvScale: 4 });
  B.add('woodDark', m, gBox(W + 0.05, 0.045, 0.035), trs(0, H, 0), { uvScale: 4 });
  const bars = 5;
  for (let i = 0; i < bars; i++) {
    const t = (i / (bars - 1) - 0.5) * W;
    B.add('woodDark', m, gBox(0.025, H * 1.02, 0.02), trs(t, H / 2, 0.012, 0, 0, 0.22), { uvScale: 6 });
    B.add('woodDark', m, gBox(0.025, H * 1.02, 0.02), trs(t, H / 2, -0.012, 0, 0, -0.22), { uvScale: 6 });
  }
  B.pop();
  const n = Math.round(48 * S.vd);
  for (let i = 0; i < n; i++) {
    _o.position.set(rng.sym(W * 0.5), rng.range(0.2, H), rng.range(0.02, 0.1));
    _o.rotation.set(rng.sym(0.5), rng.range(0, TAU), rng.sym(0.5));
    const s = rng.range(0.2, 0.36);
    _o.scale.set(s, s, s);
    _o.updateMatrix();
    _m.multiplyMatrices(fr.matrix, _o.matrix);
    S.pools.ivy.add(_m);
  }
}

/**
 * Picket fences between neighbouring plots. Runs are derived from the PLOTS
 * ordering — consecutive plots around the ring are neighbours — and join their
 * nearest front corners. Wide gaps (the road mouths) are left open.
 */
function addFences(S) {
  const { B, R, colliders } = S;
  const rng = S.rng.fork('fences');
  const mPlank = R.base('woodPlank');
  const mBeam = R.base('woodBeam');
  const mDark = R.base('woodDark');
  let runs = 0;

  for (let i = 0; i < PLOTS.length; i++) {
    const a = PLOTS[i];
    const b = PLOTS[(i + 1) % PLOTS.length];
    const ca = [plotToWorld(a, -a.width / 2, 0, a.depth / 2), plotToWorld(a, a.width / 2, 0, a.depth / 2)];
    const cb = [plotToWorld(b, -b.width / 2, 0, b.depth / 2), plotToWorld(b, b.width / 2, 0, b.depth / 2)];
    let best = null;
    for (const pa of ca) for (const pb of cb) {
      const d = pa.distanceTo(pb);
      if (!best || d < best.d) best = { d, pa, pb };
    }
    if (!best || best.d < 1.8 || best.d > 13.5) continue;

    // A fence across a road mouth would wall the player into the plaza.
    let crossesRoad = false;
    for (let s = 0; s <= 8 && !crossesRoad; s++) {
      const t = s / 8;
      const sx = best.pa.x + (best.pb.x - best.pa.x) * t;
      const sz = best.pa.z + (best.pb.z - best.pa.z) * t;
      const rd = roadDistance(sx, sz);
      if (rd.road && rd.distance < rd.road.width * 0.5 + 1.4) crossesRoad = true;
    }
    if (crossesRoad) continue;
    runs++;

    const dx = best.pb.x - best.pa.x, dz = best.pb.z - best.pa.z;
    const len = Math.hypot(dx, dz);
    const theta = Math.atan2(-dz, dx); // local +X runs along the fence
    const midX = (best.pa.x + best.pb.x) / 2;
    const midZ = (best.pa.z + best.pb.z) / 2;
    const fr = new Frame(midX, S.groundAt(midX, midZ), midZ, theta);
    B.push(fr.matrix);

    const H = 0.94;
    const gateW = runs % 3 === 1 ? 1.15 : 0;
    const posts = Math.max(2, Math.round(len / 1.9));
    for (let p = 0; p <= posts; p++) {
      const x = (p / posts - 0.5) * len;
      if (gateW && Math.abs(x) < gateW * 0.5) continue;
      B.add('woodBeam', mBeam, gBox(0.1, H + 0.28, 0.1), trs(x, (H + 0.28) / 2 - 0.14, 0, 0, 0, rng.sym(0.02)), { uvScale: 2 });
      B.add('woodDark', mDark, gCone(0.09, 0.1, 4), trs(x, H + 0.19, 0, 0, Math.PI / 4, 0), { uvScale: 4 });
    }
    for (const ry of [H * 0.28, H * 0.74]) {
      if (gateW) {
        const segLen = (len - gateW) / 2;
        for (const sx of [-1, 1]) {
          B.add('woodPlank', mPlank, gBox(segLen, 0.07, 0.035), trs(sx * (gateW / 2 + segLen / 2), ry, 0), { uvScale: 2.6 });
        }
      } else {
        B.add('woodPlank', mPlank, gBox(len, 0.07, 0.035), trs(0, ry, 0), { uvScale: 2.6 });
      }
    }
    const step = 0.185;
    const n = Math.max(2, Math.floor(len / step));
    for (let k = 0; k <= n; k++) {
      const x = (k / n - 0.5) * len;
      if (gateW && Math.abs(x) < gateW * 0.5) continue;
      const ph = H * rng.range(0.94, 1.03);
      B.add('woodPlank', mPlank, gBox(0.062, ph, 0.022), trs(x, ph / 2 - 0.02, 0.024, 0, 0, rng.sym(0.025)), { uvScale: 3 });
      B.add('woodPlank', mPlank, gCone(0.05, 0.07, 3), trs(x, ph, 0.024, 0, 0, 0, 1, 1, 0.44), { uvScale: 4 });
    }
    if (gateW) {
      B.push(trs(-gateW / 2, 0, 0, 0, 0.7, 0));
      for (const ry of [H * 0.28, H * 0.74]) {
        B.add('woodPlank', mPlank, gBox(gateW * 0.94, 0.06, 0.03), trs(gateW * 0.47, ry, 0), { uvScale: 3 });
      }
      B.add('woodPlank', mPlank, gBox(gateW * 1.02, 0.05, 0.03), trs(gateW * 0.47, H * 0.51, 0, 0, 0, 0.42), { uvScale: 3 });
      for (let k = 0; k < 6; k++) {
        B.add('woodPlank', mPlank, gBox(0.055, H * 0.92, 0.02), trs(gateW * (0.08 + k * 0.155), H * 0.46, 0.022), { uvScale: 3 });
      }
      B.pop();
    }
    B.pop();

    const segs = Math.max(1, Math.round(len / 2.6));
    for (let s = 0; s < segs; s++) {
      const x = ((s + 0.5) / segs - 0.5) * len;
      if (gateW && Math.abs(x) < gateW * 0.6) continue;
      colliders.push(boxCollider(fr.p(x, H / 2, 0), [len / segs / 2, H / 2, 0.07], quatY(theta), { tag: 'fence' }));
    }
  }
  return runs;
}

/* -------------------------------------------------------------------------- */
/* Factory                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Build every prop, planter and physics object in Hollowbrook.
 *
 * @param {Object} opts
 * @param {import('../contracts.js').MaterialLibrary} [opts.materials]
 * @param {Object} [opts.terrain]    optional; only `heightAt(x,z)` is used
 * @param {Object} [opts.buildings]  optional; `flowerBoxAnchors` used if present
 * @param {Object} [opts.quality]
 * @returns {import('../contracts.js').WorldChunk}
 */
export function createProps({ materials, terrain, buildings, quality } = {}) {
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const t0 = now();
  const q = quality || {};
  const group = new THREE.Group();
  group.name = 'props';
  const colliders = [];
  const interactables = [];
  const lightAnchors = [];

  if (!GC) GC = new Map();

  const R = createResolver(materials);
  const detail = Math.max(0.4, q.geometryDetail ?? 1);
  const vd = Math.max(0.2, q.vegetationDensity ?? 1);
  const maxDyn = Math.max(8, Math.min(46, Math.round((q.maxDynamicBodies ?? 70) * 0.55)));

  const groundAt = (x, z) => {
    try {
      const h = terrain && terrain.heightAt && terrain.heightAt(x, z);
      return Number.isFinite(h) ? h : 0;
    } catch (e) { return 0; }
  };

  /* --------------------------------------------------------- shared mats */
  // One material per merged bucket, and no more: at ~3.3 draw calls per object
  // (colour + shadow + AO) every extra material costs more than it is worth.
  // Merged away since the review: wetCobble (the pale "puddle" round the well),
  // logCut (cut ends now use woodPlank), appleGreen, paintRed and candleWax.
  const DS = THREE.DoubleSide;
  const mat = {
    wellDark: R.variant('stone', { color: new THREE.Color(0x101315), roughness: 0.99, metalness: 0, side: DS }),
    water: R.variant('stone', {
      color: new THREE.Color(0x0c1c20), roughness: 0.05, metalness: 0,
      emissive: new THREE.Color(0x0a1a20), emissiveIntensity: 0.3,
    }),
    paintCream: R.variant('woodPlank', { color: new THREE.Color(0xdccfae), roughness: 0.66 }),
    apple: R.variant('terracotta', { color: new THREE.Color(0xa5251d), roughness: 0.34 }),
    // Sacking is a TINT ON plasterWorn, i.e. a multiplier on that map. The
    // plaster albedo is being pulled down this loop (it measured 0.551 linear,
    // ~25% too bright), so the old 0xa9987a would have taken the grain sacks
    // down with it. Pre-multiplied by ~1.28 to hold them where they were.
    sackCloth: R.variant('plasterWorn', { color: new THREE.Color(0xd8c29c), roughness: 0.96 }),
    awning: R.variant('fabricAwning', { side: DS }),
    // Lit horn/glass: warm and nearly opaque so a lantern head reads as lit
    // from 25 m rather than dissolving into its own frame.
    lanternGlass: R.variant('glassLantern', {
      color: new THREE.Color(0xffcf95), transparent: true, opacity: 0.72, side: DS, depthWrite: false,
      emissive: new THREE.Color(0xff9b45), emissiveIntensity: 1.15,
    }),
  };

  /* ------------------------------------------------------------ instancing */
  const pools = {
    flowerRed: new InstancePool('flowerRed', R.variant('flowerRed', { side: DS }), crossQuadGeo(1, 3, 1), { cast: false }),
    flowerPink: new InstancePool('flowerPink', R.variant('flowerPink', { side: DS }), crossQuadGeo(1, 3, 1), { cast: false }),
    flowerYellow: new InstancePool('flowerYellow', R.variant('flowerYellow', { side: DS }), crossQuadGeo(1, 3, 1), { cast: false }),
    flowerWhite: new InstancePool('flowerWhite', R.variant('flowerWhite', { side: DS }), crossQuadGeo(1, 3, 1), { cast: false }),
    leaf: new InstancePool('leafSprig', R.variant('leaf', { side: DS }), crossQuadGeo(1, 2, 1.25), { cast: false }),
    ivy: new InstancePool('ivySprig', R.variant('ivy', { side: DS }), crossQuadGeo(1, 2, 1.15), { cast: false }),
    grass: new InstancePool('grassTuft', R.variant('grassBlade', { side: DS }), crossQuadGeo(0.34, 3, 1.1), { cast: false }),
    flame: new InstancePool('lanternFlame', R.variant('lanternEmissive', { side: DS }), crossQuadGeo(0.1, 2, 0.55), { cast: false, receive: false }),
  };

  /* ---------------------------------------------------- shared prop meshes */
  const seg = Math.max(10, Math.round(20 * detail));
  const geo = {
    barrelWood: barrelWoodGeo(0.34, 0.28, 0.86, seg),
    barrelIron: barrelIronGeo(0.34, 0.28, 0.86, seg),
    crate: crateGeo(0.58),
    crateSmall: crateGeo(0.42),
    bucket: bucketGeo(0.19, 0.15, 0.32, Math.max(8, seg - 4)),
    pot: potGeo(0.17, 0.2, Math.max(8, seg - 4)),
    apple: new THREE.SphereGeometry(0.043, 7, 5),
    sack: sackGeo(),
    // The well rope hangs down the shaft, which is deliberately near-black
    // (`wellDark` is stone x 0.06). At 13 mm it was one pixel of tan on black
    // and the bucket looked unattached; 24 mm with 7 sides reads as a rope.
    rope: (() => {
      const g = new THREE.CylinderGeometry(0.024, 0.024, 1.05, 7, 1, true);
      g.translate(0, 0.62, 0);
      return finish(prep(g, null, 'world', 6));
    })(),
  };

  const S = {
    B: new Batch(), R, mat, geo, group, colliders, interactables, lightAnchors, pools, keepOut: [],
    detail, vd, groundAt, dynBudget: maxDyn,
    rng: new Rng('hollowbrook-props'),
  };

  /* ------------------------------------------------------- dynamic spawner */
  const dyn = { barrel: 0, crate: 0, bucket: 0, pot: 0, apple: 0, sack: 0 };
  const mWoodDark = R.base('woodDark');
  const mIron = R.base('iron');
  const mPlank = R.base('woodPlank');

  function spawn(object3D, spec) {
    if (S.dynBudget <= 0) return false;
    S.dynBudget--;
    group.add(object3D);
    tagInteractive(object3D, spec.prompt, spec.tag || 'prop');
    interactables.push({
      object3D, body: 'dynamic', grabbable: true,
      linearDamping: 0.14, angularDamping: 0.4,
      ...spec,
    });
    return true;
  }

  const makeBarrel = (x, y, z, yaw, tipped) => {
    const g = new THREE.Group();
    const a = new THREE.Mesh(geo.barrelWood, mWoodDark);
    const b = new THREE.Mesh(geo.barrelIron, mIron);
    a.castShadow = a.receiveShadow = b.castShadow = b.receiveShadow = true;
    g.add(a, b);
    g.position.set(x, y, z);
    g.rotation.set(tipped ? Math.PI / 2 : 0, yaw, 0);
    if (spawn(g, {
      collider: cylinderCollider([0, 0, 0], 0.345, 0.43, null, { friction: 0.7, restitution: 0.05 }),
      mass: 24, prompt: 'Barrel', tag: 'barrel', linearDamping: 0.12, angularDamping: 0.5,
    })) { dyn.barrel++; return true; }
    return false;
  };
  const makeCrate = (x, y, z, yaw, small) => {
    const mesh = new THREE.Mesh(small ? geo.crateSmall : geo.crate, mPlank);
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.position.set(x, y, z);
    mesh.rotation.y = yaw;
    const h = small ? 0.22 : 0.3;
    if (spawn(mesh, {
      collider: boxCollider([0, 0, 0], [h, h, h], null, { friction: 0.8 }),
      mass: small ? 7 : 12, prompt: small ? 'Small crate' : 'Crate', tag: 'crate',
    })) { dyn.crate++; return true; }
    return false;
  };
  const makeBucket = (x, y, z, yaw, joint) => {
    const g = new THREE.Group();
    const mesh = new THREE.Mesh(geo.bucket, mWoodDark);
    mesh.castShadow = mesh.receiveShadow = true;
    g.add(mesh);
    if (joint) {
      const rope = new THREE.Mesh(geo.rope, R.base('rope'));
      rope.castShadow = false; rope.receiveShadow = false;
      g.add(rope);
    }
    g.position.set(x, y, z);
    g.rotation.y = yaw;
    if (spawn(g, {
      collider: cylinderCollider([0, 0, 0], 0.185, 0.16, null, { friction: 0.7 }),
      mass: 4, prompt: joint ? 'Well bucket' : 'Bucket', tag: 'bucket',
      linearDamping: joint ? 0.35 : 0.16, angularDamping: joint ? 0.6 : 0.4,
      joint: joint || undefined,
    })) { dyn.bucket++; return true; }
    return false;
  };
  const makePot = (x, y, z, yaw) => {
    const mesh = new THREE.Mesh(geo.pot, R.base('terracotta'));
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.position.set(x, y, z);
    mesh.rotation.y = yaw;
    if (spawn(mesh, {
      collider: cylinderCollider([0, 0, 0], 0.175, 0.1, null, { friction: 0.75, restitution: 0.08 }),
      mass: 6, prompt: 'Flower pot', tag: 'pot',
    })) { dyn.pot++; return true; }
    return false;
  };
  const makeApple = (x, y, z) => {
    const mesh = new THREE.Mesh(geo.apple, mat.apple);
    mesh.castShadow = false; mesh.receiveShadow = true;
    if (mesh.position.set(x, y, z), spawn(mesh, {
      collider: sphereCollider([0, 0, 0], 0.043, { friction: 0.6, restitution: 0.22 }),
      mass: 0.2, prompt: 'Apple', tag: 'apple',
      linearDamping: 0.3, angularDamping: 0.7, ccd: true,
    })) { dyn.apple++; return true; }
    return false;
  };
  const makeSack = (x, y, z, yaw) => {
    const mesh = new THREE.Mesh(geo.sack, mat.sackCloth);
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.position.set(x, y, z);
    mesh.rotation.y = yaw;
    if (spawn(mesh, {
      collider: capsuleCollider([0, 0, 0], 0.19, 0.1, null, { friction: 0.95, restitution: 0.01 }),
      mass: 18, prompt: 'Sack of grain', tag: 'sack',
      linearDamping: 0.3, angularDamping: 0.8,
    })) { dyn.sack++; return true; }
    return false;
  };

  /* ------------------------------------------------------------- the well */
  let well = null;
  try { well = buildWell(S); } catch (e) { console.error('[props] well failed:', e); }

  /* -------------------------------------------------------- authored sites */
  const pending = [];
  try {
    const rng = S.rng.fork('sites');
    let stalls = 0;
    for (const site of PROP_SITES) {
      const [sx, sz] = clearPoint(site.position[0], site.position[2], 0.6);
      const f = new Frame(sx, groundAt(sx, sz), sz, site.rotation || 0);
      S.keepOut.push({ x: sx, z: sz, r: KEEP_OUT_R[site.kind] ?? 1.2 });
      switch (site.kind) {
        case 'cart': buildCart(S, f); break;
        case 'logPile': buildLogPile(S, f); break;
        case 'trough': buildTrough(S, f); break;
        case 'bench': buildBench(S, f); break;
        case 'marketStall': buildMarketStall(S, f, stalls++); break;
        case 'wheelbarrow': buildWheelbarrow(S, f); break;
        case 'anvil': buildAnvil(S, f); break;
        case 'grindstone': buildGrindstone(S, f); break;
        case 'signpost': buildSignpost(S, f); break;

        case 'barrelStack': {
          // The bottom row is merged static geometry: a stack of dynamic
          // barrels resting on each other jitters awake on load.
          const count = site.count || 3;
          const nStatic = Math.max(1, Math.ceil(count / 2));
          for (let i = 0; i < nStatic; i++) {
            const x = (i - (nStatic - 1) / 2) * 0.74;
            S.B.push(f.matrix);
            S.B.add('woodDark', mWoodDark, geo.barrelWood, trs(x, 0.43, 0, 0, rng.range(0, TAU), 0), { uvScale: 1.6 });
            S.B.add('iron', mIron, geo.barrelIron, trs(x, 0.43, 0, 0, rng.range(0, TAU), 0), { uvScale: 3 });
            S.B.pop();
            colliders.push(cylinderCollider(f.p(x, 0.43, 0), 0.35, 0.43, null, { tag: 'barrel-static' }));
          }
          pending.push({ kind: 'barrelStack', f, n: count - nStatic });
          break;
        }
        case 'crateStack': {
          const count = site.count || 3;
          const nStatic = Math.max(1, Math.ceil(count / 2));
          for (let i = 0; i < nStatic; i++) {
            const x = (i - (nStatic - 1) / 2) * 0.62;
            S.B.push(f.matrix);
            S.B.add('woodPlank', mPlank, geo.crate, trs(x, 0.3, 0, 0, rng.sym(0.3), 0), { uvScale: 1.8 });
            S.B.pop();
            colliders.push(boxCollider(f.p(x, 0.3, 0), [0.3, 0.3, 0.3], quatY(f.yaw), { tag: 'crate-static' }));
          }
          pending.push({ kind: 'crateStack', f, n: count - nStatic });
          break;
        }
        case 'sackPile': {
          for (let i = 0; i < 3; i++) {
            const a = (i / 3) * TAU;
            S.B.push(f.matrix);
            S.B.add('sackCloth', mat.sackCloth, geo.sack,
              trs(Math.cos(a) * 0.32, 0.3, Math.sin(a) * 0.32, 0, rng.range(0, TAU), rng.sym(0.25)), { uvScale: 2.5 });
            S.B.pop();
            colliders.push(boxCollider(f.p(Math.cos(a) * 0.32, 0.28, Math.sin(a) * 0.32),
              [0.24, 0.28, 0.2], quatY(f.yaw), { tag: 'sack-static' }));
          }
          pending.push({ kind: 'sackPile', f, n: 2 });
          break;
        }
        case 'wellBucketSpare':
          pending.push({ kind: 'bucket', f, n: 2 });
          break;
        default:
          console.warn(`[props] unknown PROP_SITES kind "${site.kind}" — skipped`);
      }
    }
  } catch (e) { console.error('[props] prop sites failed:', e); }

  /* ------------------------------------------------------------ lamp posts */
  try {
    LAMP_POSTS.forEach((lp, i) => buildLampPost(S, lp.position, lp.height || 3.5, i));
  } catch (e) { console.error('[props] lamp posts failed:', e); }

  /* ---------------------------------------------------------------- fences */
  let fenceRuns = 0;
  try { fenceRuns = addFences(S); } catch (e) { console.error('[props] fences failed:', e); }

  /* -------------------------------------------------------------- planting */
  let potCount = 0, boxCount = 0, basketCount = 0;
  try {
    const rng = S.rng.fork('planting');
    const palettes = [
      ['flowerRed', 'flowerPink', 'flowerRed', 'flowerWhite'],
      ['flowerPink', 'flowerWhite', 'flowerPink'],
      ['flowerRed', 'flowerYellow', 'flowerRed'],
      ['flowerYellow', 'flowerWhite', 'flowerPink'],
    ];

    /** Plant a pot if a clear spot exists near (x,z); returns whether it did. */
    const placePot = (x0, z0, size, prng, pal) => {
      const at = settle(S.keepOut, x0, z0, size * 1.25, 0.22, 1.5);
      if (!at) return false;
      addPot(S, placeAt(at[0], groundAt(at[0], at[1]), at[1], prng.range(0, TAU)), size, prng, pal);
      return true;
    };

    const anchors = buildings && Array.isArray(buildings.flowerBoxAnchors) && buildings.flowerBoxAnchors.length
      ? buildings.flowerBoxAnchors : null;

    if (anchors) {
      for (const a of anchors) {
        const pos = Array.isArray(a.position) ? a.position
          : (a.position && a.position.isVector3 ? a.position.toArray() : null);
        if (!pos) continue;
        const yaw = typeof a.rotation === 'number' ? a.rotation
          : (typeof a.yaw === 'number' ? a.yaw : (a.plot ? a.plot.rotation : 0));
        const width = a.width || 0.9;
        const m = placeAt(pos[0], pos[1], pos[2], yaw);
        // The box itself belongs to the buildings stream — only plant it.
        S.B.push(m);
        S.B.add('soil', R.base('soil'), gPlane(width * 0.86, 0.18), trs(0, 0.01, 0, -Math.PI / 2, 0, 0), { uvScale: 4, cast: false });
        S.B.pop();
        spreadBlossom(S, m, width * 0.42, 0.09, 0.28, 21, rng, rng.pick(palettes));
        boxCount++;
      }
    }

    for (let pi = 0; pi < PLOTS.length; pi++) {
      const plot = PLOTS[pi];
      const pal = palettes[pi % palettes.length];
      const prng = rng.fork(`plot${plot.seed}`);

      if (plot.flowerBoxes && !anchors) {
        const n = plot.width > 13 ? 3 : 2;
        for (const an of frontAnchors(plot, n, 0.2, 2.2)) {
          const y = groundAt(an.position[0], an.position[2]) + (plot.storeys > 1 ? 1.18 : 1.24);
          addWindowBox(S, placeAt(an.position[0], y, an.position[2], plot.rotation), 1.0, prng, pal);
          boxCount++;
        }
      }

      // Jitter in PLOT-LOCAL space: a world-space offset could shove a pot
      // through the wall, or into the doorway the steps occupy.
      const potN = plot.flowerBoxes ? 4 : 2;
      const doorKeepOut = (plot.steps ? plot.steps.width / 2 + 0.55 : 0.95);
      const frontZ = plot.depth / 2;
      for (const an of frontAnchors(plot, potN + 1, 0.62, 1.4)) {
        if (Math.abs(an.lx - (plot.door ? plot.door.x : 0)) < doorKeepOut) continue;
        if (!prng.bool(plot.flowerBoxes ? 0.9 : 0.55)) continue;
        const lx = an.lx + prng.sym(0.42);
        const lz = frontZ + 0.62 + prng.range(0, 0.34);
        const size = prng.pick([0.18, 0.22, 0.26, 0.3]);
        const w0 = plotToWorld(plot, lx, 0, lz);
        if (placePot(w0.x, w0.z, size, prng, pal)) potCount++;
        if (prng.bool(0.6)) {
          const w1 = plotToWorld(plot, lx + prng.range(0.3, 0.52) * prng.pick([-1, 1]), 0, lz + prng.range(0.1, 0.34));
          if (placePot(w1.x, w1.z, size * 0.68, prng, pal)) potCount++;
        }
      }

      const tufts = Math.round(17 * vd);
      for (let k = 0; k < tufts; k++) {
        const lx = prng.sym(plot.width / 2 - 0.4);
        const wp = plotToWorld(plot, lx, 0, plot.depth / 2 + prng.range(0.06, 0.32));
        _o.position.set(wp.x, groundAt(wp.x, wp.z), wp.z);
        _o.rotation.set(0, prng.range(0, TAU), 0);
        const s = prng.range(0.6, 1.2);
        _o.scale.set(s, s * prng.range(0.8, 1.4), s);
        _o.updateMatrix();
        pools.grass.add(_o.matrix);
      }
    }

    const byId = (id) => PLOTS.find((p) => p.id === id) || PLOTS[0];
    addClimber(S, byId('rosecot'), 1.5, rng.fork('rose1'), ['flowerPink', 'flowerRed', 'flowerWhite']);
    addClimber(S, byId('weaver'), -2.6, rng.fork('rose2'), ['flowerRed', 'flowerPink']);
    addClimber(S, byId('apothecary'), 2.4, rng.fork('rose3'), ['flowerWhite', 'flowerYellow']);
    addTrellis(S, byId('chandler'), -3.6, rng.fork('trellis1'));
    addTrellis(S, byId('saddlery'), -3.2, rng.fork('trellis2'));
    basketCount = addHangingBaskets(S, byId('inn'), rng.fork('baskets'));
  } catch (e) { console.error('[props] planting failed:', e); }

  /* ------------------------------------------------------- merge + emit */
  let built = { meshes: 0, tris: 0 };
  try { built = S.B.build(group); } catch (e) { console.error('[props] static build failed:', e); }

  /* -------------------------------------------------------- dynamic bodies */
  try {
    const rng = S.rng.fork('dynamics');

    // the well bucket, hanging from the windlass on a rope joint
    if (well) {
      const drop = WELL.winch.bucketDrop;
      const anchor = well.frame.p(0.18, well.drumY, 0);
      const bp = well.frame.p(0.18, well.drumY - drop, 0);
      makeBucket(bp[0], bp[1], bp[2], well.frame.yaw, { type: 'rope', anchorWorld: anchor, length: drop });
    }

    for (const d of pending) {
      const f = d.f;
      if (d.kind === 'barrelStack') {
        for (let i = 0; i < d.n; i++) {
          const wp = f.p((i - (d.n - 1) / 2) * 0.74, 1.32, 0);
          const [bx, bz] = clearPoint(wp[0], wp[2], 0.4, 1.0);
          makeBarrel(bx, wp[1], bz, rng.range(0, TAU), false);
        }
        const lp = f.p(0, 0.36, 1.15);
        makeBarrel(lp[0], lp[1], lp[2], rng.range(0, TAU), true);
      } else if (d.kind === 'crateStack') {
        for (let i = 0; i < d.n; i++) {
          const wp = f.p((i - (d.n - 1) / 2) * 0.64, 0.92 + i * 0.02, rng.sym(0.08));
          const [cx, cz] = clearPoint(wp[0], wp[2], 0.36, 1.0);
          makeCrate(cx, wp[1], cz, rng.sym(0.5), i > 0);
        }
      } else if (d.kind === 'sackPile') {
        for (let i = 0; i < d.n; i++) {
          const wp = f.p(rng.sym(0.25), 0.94 + i * 0.66, rng.sym(0.25));
          makeSack(wp[0], wp[1], wp[2], rng.range(0, TAU));
        }
      } else if (d.kind === 'bucket') {
        for (const [ox, oz, dy] of [[0, 0, 0], [0.58, 0.62, 1.1]]) {
          const w = f.p(ox, 0.17, oz);
          const [cx, cz] = clearPoint(w[0], w[2], 0.45);
          makeBucket(cx, w[1], cz, f.yaw + dy, null);
        }
      }
    }

    /* Loose clutter around the plaza — the cheapest way to make a world feel
       lived in. Ordered least-important-last so the budget trims the right end.
       A loose apple is 4 cm across and invisible past ~4 m, but as a separate
       dynamic body it still costs a full object (~3.3 draw calls), so only a
       handful are kept: enough for the throw toy, not thirteen. */
    const clutter = [
      ['crate', 11.2, -13.6], ['crate', -10.1, -16.2], ['crate', 15.6, 13.8],
      ['crateSmall', 13.1, -11.4], ['crateSmall', -15.6, 18.4],
      ['pot', -6.2, 5.3], ['pot', 14.4, 3.4], ['pot', 1.4, 20.9],
      ['pot', -16.2, -7.1], ['pot', 7.2, 17.7], ['pot', -23.1, 9.4],
      ['pot', 22.4, -10.2], ['pot', -3.1, -19.4],
      ['bucket', -25.2, 9.6], ['bucket', -17.1, 21.6],
      ['crateSmall', 25.2, -5.6], ['crateSmall', -22.6, -14.4],
      ['apple', 12.3, -13.0], ['apple', 12.9, -13.5], ['apple', 11.6, -12.4],
      ['apple', 4.7, 8.9], ['apple', 5.1, 9.4], ['apple', -13.2, 15.9],
    ];
    for (const [kind, cx, cz] of clutter) {
      const at = settle(S.keepOut, cx, cz, kind === 'apple' ? 0.14 : 0.46, 0.5, 3.0);
      if (!at) { console.warn(`[props] no clear spot for ${kind} near ${cx},${cz}`); continue; }
      const x = at[0], z = at[1];
      const y = groundAt(x, z);
      const yaw = rng.range(0, TAU);
      if (kind === 'crate') makeCrate(x, y + 0.31, z, yaw, false);
      else if (kind === 'crateSmall') makeCrate(x, y + 0.23, z, yaw, true);
      else if (kind === 'pot') makePot(x, y + 0.11, z, yaw);
      else if (kind === 'bucket') makeBucket(x, y + 0.17, z, yaw, null);
      else if (kind === 'apple') makeApple(x, y + 0.05, z);
    }
  } catch (e) { console.error('[props] dynamic bodies failed:', e); }

  // Count dynamic meshes toward the draw budget.
  for (const it of interactables) {
    it.object3D.traverse((o) => {
      if (!o.isMesh) return;
      built.meshes++;
      built.tris += o.geometry.index ? o.geometry.index.count / 3 : o.geometry.getAttribute('position').count / 3;
    });
  }

  /* ------------------------------------------------------------- instances */
  const instanceCounts = {};
  for (const k of Object.keys(pools)) {
    const im = pools[k].build(group);
    instanceCounts[k] = im ? im.count : 0;
    if (im) {
      built.meshes++;
      const t = im.geometry.index ? im.geometry.index.count / 3 : im.geometry.getAttribute('position').count / 3;
      built.tris += t * im.count;
    }
  }

  /* ------------------------------------------------------------ flicker */
  const flame = pools.flame.mesh;
  const flameBase = flame ? pools.flame.matrices.map((m) => m.clone()) : [];
  const flamePhase = flameBase.map((_, i) => i * 2.399);
  const _fm = new THREE.Matrix4();
  const _fs = new THREE.Vector3();
  const _fp = new THREE.Vector3();
  const _fq = new THREE.Quaternion();
  let elapsed = 0;

  for (const k of Object.keys(pools)) pools[k].matrices.length = 0;

  function update(dt) {
    if (!flame) return;
    elapsed += dt;
    for (let i = 0; i < flameBase.length; i++) {
      const ph = flamePhase[i];
      const s = 0.84 + 0.13 * Math.sin(elapsed * 9.3 + ph) + 0.07 * Math.sin(elapsed * 23.1 + ph * 1.7);
      flameBase[i].decompose(_fp, _fq, _fs);
      _fm.compose(_fp, _fq, _fs.set(s, s * (1.05 + 0.12 * Math.sin(elapsed * 13.7 + ph)), s));
      flame.setMatrixAt(i, _fm);
    }
    flame.instanceMatrix.needsUpdate = true;
  }

  /* -------------------------------------------------------------- teardown */
  for (const g of GC.values()) g.dispose();
  GC.clear();

  const blossoms = instanceCounts.flowerRed + instanceCounts.flowerPink +
    instanceCounts.flowerYellow + instanceCounts.flowerWhite;

  const stats = {
    drawCalls: built.meshes,
    triangles: Math.round(built.tris),
    staticColliders: colliders.length,
    dynamicBodies: interactables.length,
    dynamicBudget: maxDyn,
    dynamicKinds: dyn,
    blossoms,
    leafSprigs: instanceCounts.leaf,
    ivySprigs: instanceCounts.ivy,
    grassTufts: instanceCounts.grass,
    lanterns: lightAnchors.length,
    potsPlanted: potCount,
    windowBoxes: boxCount,
    hangingBaskets: basketCount,
    fenceRuns,
    buildMs: Math.round(now() - t0),
  };

  return {
    group,
    colliders,
    interactables,
    lightAnchors,
    update,
    dispose() {
      disposeGroup(group);
      // budget-trimmed prop types may never have been attached to a mesh
      for (const k of Object.keys(geo)) geo[k].dispose();
      R.dispose();
    },
    stats,
  };
}
