/**
 * ============================================================================
 *  FURNISHINGS — the CONTENTS of the rooms
 * ============================================================================
 * `buildings.js` cuts the openings and publishes `chunk.interiors.rooms`
 * (RoomSpec[]). `interiors.js` builds the structure — floors, linings, joists,
 * stairs. This file builds everything you could carry out of the house: tables,
 * beds, counters, casks, the hearth and its fire, and the loose clutter that
 * turns a lit box into a room somebody lives in.
 *
 * Three kinds of object, and the split is a performance decision:
 *
 *   1. STATIC FURNITURE — merged by material, PER ROOM, into a handful of
 *      meshes under one `THREE.Group` per room. Per-room is the whole point:
 *      it is the unit of culling, so 18 rooms of furniture cost the draw calls
 *      of the two or three you can actually see.
 *   2. GRABBABLES — one mesh each, parented to the chunk root (physics writes
 *      world transforms onto them), hidden with their room but kept visible
 *      while the player is near them, so a tankard carried outdoors does not
 *      vanish from your hand.
 *   3. FLAMES — a single InstancedMesh for every fire and candle in the
 *      village, plus one for the ember beds. One draw call, flickered in
 *      `update`, scaled to zero for hidden rooms.
 *
 * Everything is authored in ROOM-LOCAL metres: origin at the centre of the
 * finished floor, +X along the room's width, +Z toward the front (door) wall,
 * y up from the floor. `room.centre` + `room.rotation` place that frame in the
 * world, and it matches `layout.plotToWorld` exactly, so a collider is just the
 * same point pushed through the same rotation.
 *
 * Nothing here re-derives a wall position, a floor level or an opening: every
 * number comes from the RoomSpec. Nothing uses Math.random.
 * ============================================================================
 */

import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { Rng } from '../util/rng.js';
import { PLOTS, INTERIOR_USES, plotToWorld } from './layout.js';
import {
  boxCollider, cylinderCollider, sphereCollider, quatY, tagInteractive, disposeGroup,
} from '../contracts.js';

/* -------------------------------------------------------------------------- */
/* Tunables                                                                    */
/* -------------------------------------------------------------------------- */

/** Distance (m) from a room's centre at which its contents appear / vanish. */
// 11.5 m is measured, not guessed: the deepest room's centre is 5.35 m behind
// its front wall, so this reveals a room while you are still ~6 m outside its
// door and keeps a NEIGHBOUR's interior (plots sit 10-14 m apart) out of the
// visible set most of the time. The gap to HIDE_R is the hysteresis band.
const SHOW_R = 11.5;
const HIDE_R = 15.5;
/**
 * Once the player is INSIDE a room, a room in a DIFFERENT building has to be
 * this close to earn a slot. You cannot see another house's furniture from
 * indoors except through your own doorway and then through theirs, so the wide
 * outdoor radius only ever bought a third room nobody could see. The rooms of
 * the building you are standing in are exempt: the stairwell opening genuinely
 * shows the storey above.
 */
const INDOOR_CROSS_SHOW_R = 5.5;
const INDOOR_CROSS_HIDE_R = 7.5;
/**
 * Hard cap on how many rooms may be drawn at once.
 *
 * Two, not three. Measured: the useful set indoors is the room you are in plus
 * the one room its openings show you (the storey above through the stair well,
 * or the room beyond the door). A third slot was always spent on a room behind
 * a solid wall — ~11 draw calls and ~3.5 k triangles of furniture that could
 * not be seen from anywhere in the building.
 */
const MAX_VISIBLE = { low: 1, medium: 2, high: 2, ultra: 3 };
/**
 * Grabbable bodies this stream may add. The village already spends ~47 dynamic
 * bodies outdoors (props) and buildings spends ~11 on hinged doors and signs,
 * and `physics.addInteractables` gives GRABBABLES priority over JOINTED bodies
 * when it runs out of budget — so being greedy here would freeze the doors shut.
 */
const DYN_BUDGET = { low: 6, medium: 18, high: 34, ultra: 44 };
/** Keep a grabbable visible this far from the player even if its room is not. */
const CARRY_VISIBLE_R2 = 6 * 6;

/**
 * Which rooms keep their loose objects when the dynamic-body budget is tight.
 * A tankard you can throw across the taproom is the single most valuable
 * grabbable in the village; the fifth crock in an attic store is the least.
 */
const USE_PRIORITY = {
  taproom: 0, kitchen: 1, bakery: 1, shop: 2, hall: 2, workshop: 3,
  bedroom: 4, stable: 5, store: 6,
};

/**
 * Interior material keys are new to the contract; `materials/` is being written
 * in the same pass. If a key is not defined yet we fall back to the nearest
 * exterior material rather than filling the room with magenta.
 */
const MAT_FALLBACK = {
  hearthStone: ['stone', 'stoneTrim'],
  flagstone: ['stone', 'cobbleWorn'],
  soot: ['ironDark', 'timberDark'],
  linen: ['fabricAwning', 'plaster'],
  sackcloth: ['fabricAwning', 'plasterWorn'],
  strawLitter: ['thatch', 'grass'],
  pewter: ['iron', 'brass'],
  ceilingBeam: ['woodBeam', 'timber'],
  lathPlaster: ['plaster'],
  floorBoard: ['woodPlank'],
};

/* -------------------------------------------------------------------------- */
/* Scratch — hoisted; update() must never allocate                             */
/* -------------------------------------------------------------------------- */

const _o = new THREE.Object3D();
const _v = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _q2 = new THREE.Quaternion();
const _YUP = new THREE.Vector3(0, 1, 0);
const _e = new THREE.Euler();
const _m = new THREE.Matrix4();
const _s = new THREE.Vector3();
const _pos = new THREE.Vector3();
const _rot = new THREE.Quaternion();
const _one = new THREE.Vector3(1, 1, 1);
const TAU = Math.PI * 2;

/** Compose a local matrix. Returns a SHARED matrix — consume it immediately. */
function trs(px, py, pz, rx = 0, ry = 0, rz = 0, sx = 1, sy = 1, sz = 1) {
  _o.position.set(px, py, pz);
  _o.rotation.set(rx, ry, rz);
  _o.scale.set(sx, sy, sz);
  _o.updateMatrix();
  return _o.matrix;
}

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

/**
 * Offsets along an object's own axes. Ry(yaw) sends local +X to
 * (cos, -sin) and local +Z to (sin, cos) — worth writing down once, because
 * "offset it sideways" is the single easiest thing to get backwards here.
 */
const alongX = (x, yaw, d) => x + Math.cos(yaw) * d;
const alongXz = (z, yaw, d) => z - Math.sin(yaw) * d;
const alongZ = (x, yaw, d) => x + Math.sin(yaw) * d;
const alongZz = (z, yaw, d) => z + Math.cos(yaw) * d;

/* -------------------------------------------------------------------------- */
/* Geometry primitives (build time only)                                       */
/* -------------------------------------------------------------------------- */

let GC = new Map();

function cached(key, make) {
  let g = GC.get(key);
  if (!g) { g = make(); GC.set(key, g); }
  return g;
}

const gBox = (w, h, d) => cached(`b|${w}|${h}|${d}`, () => new THREE.BoxGeometry(w, h, d));
const gCyl = (rt, rb, h, s = 10, open = false) =>
  cached(`c|${rt}|${rb}|${h}|${s}|${open}`, () => new THREE.CylinderGeometry(rt, rb, h, s, 1, open));
const gSph = (r, ws = 8, hs = 6) =>
  cached(`s|${r}|${ws}|${hs}`, () => new THREE.SphereGeometry(r, ws, hs));
const gPlane = (w, h) => cached(`p|${w}|${h}`, () => new THREE.PlaneGeometry(w, h));
const gTorus = (r, t, rs = 5, ts = 8, arc = TAU) =>
  cached(`t|${r}|${t}|${rs}|${ts}|${arc}`, () => new THREE.TorusGeometry(r, t, rs, ts, arc));

/** Cached lathe. `key` must be unique per profile — the profile is not hashed. */
function gLathe(key, pts, seg = 9) {
  return cached(`l|${key}|${seg}`, () =>
    new THREE.LatheGeometry(pts.map((p) => new THREE.Vector2(p[0], p[1])), seg));
}

/** A single quad from four corners, wound a-b-c / a-c-d. */
function quad(a, b, c, d) {
  const g = new THREE.BufferGeometry();
  const p = new Float32Array([
    a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2],
    a[0], a[1], a[2], c[0], c[1], c[2], d[0], d[1], d[2],
  ]);
  g.setAttribute('position', new THREE.BufferAttribute(p, 3));
  g.computeVertexNormals();
  return g;
}

/**
 * Strip to position/normal/uv, guarantee an index, transform, clone. Merging
 * fails if attribute sets disagree, so every part goes through here.
 */
function prep(src, matrix, uvMode = 'world', uvScale = 1) {
  const g = new THREE.BufferGeometry();
  const pos = src.getAttribute('position');
  g.setAttribute('position', pos.clone());
  const nrm = src.getAttribute('normal');
  g.setAttribute('normal', nrm ? nrm.clone()
    : new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3));
  const uv = src.getAttribute('uv');
  g.setAttribute('uv', uv ? uv.clone()
    : new THREE.BufferAttribute(new Float32Array(pos.count * 2), 2));
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
  else if (uvScale !== 1) scaleUV(g, uvScale);
  return g;
}

/** Box-projected UVs from position — the only sane shared parameterisation. */
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

function scaleUV(g, s) {
  const uv = g.getAttribute('uv');
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * s, uv.getY(i) * s);
  uv.needsUpdate = true;
}

function finish(geo) {
  const uv = geo.getAttribute('uv');
  if (uv && !geo.getAttribute('uv1')) geo.setAttribute('uv1', uv);
  geo.computeBoundingSphere();
  return geo;
}

/** Merge [{geo, matrix, uvScale?, uv?}] into one owned BufferGeometry. */
function mergeParts(parts) {
  const list = parts.map((p) => prep(p.geo, p.matrix, p.uv || 'world', p.uvScale ?? 2));
  let out = null;
  try {
    out = list.length === 1 ? list[0] : mergeGeometries(list, false);
  } catch (e) { out = null; }
  if (!out) {
    console.warn('[furnishings] mergeParts failed; using the first part only');
    out = list[0];
  }
  for (const g of list) if (g !== out) g.dispose();
  return finish(out);
}

/** Coherently displace the 8 corners of a box — a hewn block, not a cube. */
function hew(g, amp, rng) {
  const p = g.getAttribute('position');
  const off = new Map();
  for (let i = 0; i < p.count; i++) {
    const k = (p.getX(i) > 0 ? 4 : 0) | (p.getY(i) > 0 ? 2 : 0) | (p.getZ(i) > 0 ? 1 : 0);
    let o = off.get(k);
    if (!o) { o = [rng.sym(amp), rng.sym(amp * 0.6), rng.sym(amp)]; off.set(k, o); }
    p.setXYZ(i, p.getX(i) + o[0], p.getY(i) + o[1], p.getZ(i) + o[2]);
  }
  g.computeVertexNormals();
  return g;
}

/** Three intersecting quads — a flame, a straw wisp, a bunch of herbs. */
/**
 * A bed of embers, lying flat.
 *
 * This used to be a single 1x1 plane on an opaque emissive material, which read
 * as exactly what it was: a hard-edged orange rectangle sticking out past the
 * firedogs. Coals have no straight edges, so the silhouette has to come from the
 * geometry — an irregular fan whose rim alpha falls to zero, so the glow dies out
 * instead of stopping. Alpha is carried in a 4-component colour attribute:
 * three multiplies diffuse by vColor.rgb and takes diffuseColor.a from vColor.a,
 * and the final alpha scales the emissive too, which is the whole point.
 *
 * Deterministic by construction — the jitter is a fixed hash of the vertex index,
 * so every hearth in the village bakes the same bed.
 */
function emberBedGeometry(radius = 0.34, segments = 14) {
  const hash = (i) => {
    const s = Math.sin(i * 12.9898) * 43758.5453;
    return s - Math.floor(s);
  };
  const pos = [], col = [], nrm = [], uv = [], idx = [];
  // centre, then a hot mid ring, then the cool ragged rim
  const rings = [
    { r: 0, a: 1.0, warm: 1.0 },
    { r: 0.52, a: 0.78, warm: 0.92 },
    { r: 1.0, a: 0.0, warm: 0.55 },
  ];
  for (let ri = 0; ri < rings.length; ri++) {
    const ring = rings[ri];
    const count = ri === 0 ? 1 : segments;
    for (let s = 0; s < count; s++) {
      const t = (s / segments) * Math.PI * 2;
      // ragged: each rim vertex pulls in or out, and dips a few millimetres
      const jitter = ri === 0 ? 1 : 0.72 + 0.56 * hash(ri * 31 + s);
      const r = ring.r * radius * jitter;
      pos.push(Math.cos(t) * r, ri === 0 ? 0.004 : -0.002 * hash(s * 7), Math.sin(t) * r);
      nrm.push(0, 1, 0);
      uv.push(0.5 + Math.cos(t) * ring.r * 0.5, 0.5 + Math.sin(t) * ring.r * 0.5);
      col.push(ring.warm, ring.warm * 0.42, ring.warm * 0.12, ring.a);
    }
  }
  const ringStart = [0, 1, 1 + segments];
  for (let s = 0; s < segments; s++) {
    const n = (s + 1) % segments;
    // centre fan
    idx.push(0, ringStart[1] + s, ringStart[1] + n);
    // mid -> rim quad, as two triangles
    idx.push(ringStart[1] + s, ringStart[2] + s, ringStart[2] + n);
    idx.push(ringStart[1] + s, ringStart[2] + n, ringStart[1] + n);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 4));
  g.setIndex(idx);
  g.computeBoundingSphere();
  return g;
}

/** Triangles one ember bed costs, so the stats stay honest. */
const EMBER_TRIS = 14 * 3;

function crossQuad(w, blades, h) {
  const parts = [];
  for (let i = 0; i < blades; i++) {
    const g = gPlane(1, 1).clone();
    g.translate(0, 0.5, 0);
    parts.push({ geo: g, matrix: trs(0, 0, 0, 0, (i / blades) * Math.PI, 0, w, h, 1), uvScale: 1 });
  }
  const out = mergeParts(parts.map((p) => ({ ...p, uv: 'keep' })));
  for (const p of parts) p.geo.dispose();
  return out;
}

/* -------------------------------------------------------------------------- */
/* The fire                                                                    */
/* -------------------------------------------------------------------------- */

const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a || 1e-6), 0, 1);
  return t * t * (3 - 2 * t);
};

/**
 * Emissive gain on the flame texture. MeshBasicMaterial multiplies `color` by
 * the texel, and a THREE.Color happily holds values above 1, so this is the
 * flame's HDR exposure.
 *
 * MEASURED against the bloom threshold in post.js (2.6 at `high`, soft knee
 * 2.4). At 3.1 the texture's hottest pixels land at linear luminance ~2.85 —
 * just over the line, so the core picks up a few percent of bloom and nothing
 * else in the flame does. The orange body sits at ~1.34 and the outer licks at
 * ~0.5, an order of magnitude above the ~0.1 of a hearth-lit plaster wall, so
 * the fire is still by a wide margin the brightest thing in the room while
 * staying a long way inside the tone mapper's shoulder.
 */
const FLAME_GAIN = 3.1;

/**
 * The flame card's texture: a teardrop alpha silhouette, a near-white core at
 * the root, an orange body and deep red-orange licks at the tip.
 *
 * There is a reason this is a texture and not a flat emissive quad. The old
 * flame was an opaque `lanternEmissive` variant, which means (a) its silhouette
 * was a literal square and (b) it was a LIT material sitting 0.05-0.3 m from
 * its own hearth PointLight. three's point-light attenuation clamps at
 * 1/max(d^2, 0.01), so at that range the card's diffuse term was amplified
 * ~100x: measured ~(5.3, 2.6, 1.2) of *diffuse* on top of ~(3.6, 1.6, 0.4) of
 * emissive, luminance ~5.0, which is twice the bloom threshold. That is the
 * white blob. The fix is not a smaller number on a lit material — it is to
 * stop lighting the fire. `MeshBasicMaterial` is unlit, so the flame is exactly
 * the texel times the gain and nothing the light pool does can blow it out.
 */
function makeFlameTexture() {
  if (typeof document === 'undefined' || !document.createElement) return null;
  const W = 48, H = 96;
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const ctx2d = cv.getContext('2d');
  if (!ctx2d) return null;
  const img = ctx2d.createImageData(W, H);
  const d = img.data;

  // Three colour stops, in sRGB bytes because that is what a canvas stores.
  const COLD = [206, 56, 10];      // the tips and the outer edge of a lick
  const MID = [255, 150, 40];      // the body — the colour a fire actually is
  const HOT = [255, 246, 224];     // the core, and only the core

  for (let py = 0; py < H; py++) {
    // Canvas row 0 is the TOP of the image and PlaneGeometry's v = 1 is the
    // top, so t = 0 is the root of the flame and t = 1 is the tip.
    const t = 1 - (py + 0.5) / H;
    // Width profile: narrow at the root, widest around a third of the way up,
    // drawn to a point at the tip. The skew leans the licks over so three
    // crossed blades do not read as a symmetrical cone.
    const bulge = Math.pow(Math.max(0, Math.sin(Math.PI * Math.min(1, 0.10 + t * 0.88))), 0.62);
    const hw = Math.max(1e-3, 0.47 * bulge * (1 - 0.52 * t * t));
    const cx = 0.5 + 0.055 * Math.sin(t * 6.1) * t;

    for (let px = 0; px < W; px++) {
      const u = (px + 0.5) / W;
      const dd = Math.abs(u - cx) / hw;          // 0 at the axis, 1 at the edge

      let a = 1 - smoothstep(0.44, 1.0, dd);
      a *= 1 - smoothstep(0.70, 1.0, t);          // the tip dissolves
      a *= smoothstep(0.0, 0.05, t);              // and the root is not a cut edge

      // Heat: hottest on the axis low down, falling off both outward and up.
      let heat = (1 - smoothstep(0.0, 0.58, dd)) * (1 - smoothstep(0.08, 0.60, t));
      // Internal structure — a darker sheath just inside the silhouette, so the
      // flame has a visible inside rather than one flat wash of colour.
      heat *= 1 - 0.26 * Math.exp(-Math.pow((dd - 0.62) * 4.2, 2));
      heat = clamp(heat, 0, 1);

      let r, g, b;
      if (heat < 0.62) {
        const k = heat / 0.62;
        r = COLD[0] + (MID[0] - COLD[0]) * k;
        g = COLD[1] + (MID[1] - COLD[1]) * k;
        b = COLD[2] + (MID[2] - COLD[2]) * k;
      } else {
        const k = (heat - 0.62) / 0.38;
        r = MID[0] + (HOT[0] - MID[0]) * k;
        g = MID[1] + (HOT[1] - MID[1]) * k;
        b = MID[2] + (HOT[2] - MID[2]) * k;
      }

      const o = (py * W + px) * 4;
      d[o] = r; d[o + 1] = g; d[o + 2] = b;
      d[o + 3] = Math.round(clamp(a, 0, 1) * 255);
    }
  }
  ctx2d.putImageData(img, 0, 0);

  const tex = new THREE.CanvasTexture(cv);
  // The bytes above are sRGB; without this they would be read as linear and the
  // whole flame would come out pale.
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.name = 'furn:flame';
  return tex;
}

/* -------------------------------------------------------------------------- */
/* Bin — one merged mesh per (material, shadow flag) inside one room           */
/* -------------------------------------------------------------------------- */

class Bin {
  constructor() { this.buckets = new Map(); }

  /**
   * The bucket is keyed by the MATERIAL, not by the name the caller passed:
   * two call sites that ask for the same material always share a draw call, and
   * a call site that mislabels one can never silently repaint another's
   * geometry. `key` is only a label for the mesh name.
   *
   * `opts.cast` is a per-material vote, not a per-part flag: splitting the
   * bucket by shadow flag would double the room's draw calls to save shadow
   * work the sun cannot see anyway (these meshes are indoors), so one big
   * caster in a material makes the whole merged mesh cast.
   */
  add(key, material, geo, matrix, opts) {
    const bkey = material.uuid;
    let b = this.buckets.get(bkey);
    if (!b) { b = { key, material, geos: [], cast: false }; this.buckets.set(bkey, b); }
    if (opts && opts.cast) b.cast = true;
    b.geos.push(prep(geo, matrix, (opts && opts.uv) || 'world', (opts && opts.uvScale) ?? 1.6));
  }

  build(group, name) {
    let meshes = 0, tris = 0;
    for (const [bkey, b] of this.buckets) {
      if (!b.geos.length) continue;
      let merged = null;
      try {
        merged = b.geos.length === 1 ? b.geos[0] : mergeGeometries(b.geos, false);
      } catch (e) { merged = null; }
      if (!merged) {
        console.warn(`[furnishings] merge failed for ${name}/${bkey}`);
        for (const g of b.geos) g.dispose();
        continue;
      }
      for (const g of b.geos) if (g !== merged) g.dispose();
      finish(merged);
      const mesh = new THREE.Mesh(merged, b.material);
      mesh.name = `furn:${name}:${b.key}`;
      mesh.castShadow = b.cast;
      mesh.receiveShadow = true;
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
/* Material resolution — fail soft if a key or the whole library is missing     */
/* -------------------------------------------------------------------------- */

const LOCAL_FALLBACK = {
  woodDark: [0x4a3826, 0.82, 0], woodPlank: [0x8a6b45, 0.78, 0],
  woodBeam: [0x6b5133, 0.8, 0], timberDark: [0x33261a, 0.85, 0],
  stone: [0x8e8b83, 0.94, 0], brick: [0x9a5a45, 0.9, 0],
  iron: [0x3a3c40, 0.55, 0.85], ironDark: [0x24262a, 0.6, 0.85],
  brass: [0xa8823c, 0.38, 0.9], copper: [0x4e8a78, 0.55, 0.4],
  terracotta: [0xa9613c, 0.85, 0], terracottaDark: [0x83442a, 0.88, 0],
  candleWax: [0xf0e6cc, 0.6, 0], rope: [0xa89469, 0.95, 0],
  fabricAwning: [0xd9cfc0, 0.95, 0], plaster: [0xd8cbb0, 0.95, 0],
  thatch: [0xb9974f, 0.95, 0], leaf: [0x4f7a33, 0.9, 0],
  lanternEmissive: [0xffbb66, 0.9, 0], glassWindow: [0xb6c6cc, 0.2, 0],
  hearthStone: [0x6f6a62, 0.95, 0], soot: [0x1c1a18, 0.9, 0],
  linen: [0xe2d9c4, 0.92, 0], sackcloth: [0xb2a07a, 0.95, 0],
  strawLitter: [0xc4a862, 0.95, 0], pewter: [0x8d9096, 0.42, 0.8],
};

function createResolver(materials) {
  const owned = [];
  const local = new Map();
  const substituted = [];
  let warned = false;

  /** Canonical name, or the nearest defined stand-in. */
  function key(name) {
    if (materials && typeof materials.has === 'function') {
      if (materials.has(name)) return name;
      for (const alt of MAT_FALLBACK[name] || []) {
        if (materials.has(alt)) {
          if (!substituted.includes(name)) substituted.push(`${name}->${alt}`);
          return alt;
        }
      }
    }
    return name;
  }

  function localMat(name) {
    let m = local.get(name);
    if (!m) {
      if (!warned) {
        console.warn('[furnishings] material library unavailable — using flat fallbacks');
        warned = true;
      }
      const f = LOCAL_FALLBACK[name] || [0xb0a48c, 0.85, 0];
      m = new THREE.MeshStandardMaterial({
        color: new THREE.Color(typeof f[0] === 'number' ? f[0] : 0xb0a48c),
        roughness: f[1], metalness: f[2],
      });
      m.name = `furn-fallback:${name}`;
      local.set(name, m); owned.push(m);
    }
    return m;
  }

  function base(name) {
    const k = key(name);
    try {
      const m = materials && materials.get && materials.get(k);
      if (m) return m;
    } catch (e) { /* fall through */ }
    return localMat(k);
  }

  function variant(name, opts) {
    const k = key(name);
    try {
      const m = materials && materials.variant && materials.variant(k, opts);
      if (m) return m;
    } catch (e) { /* fall through */ }
    const ck = `${k}|${JSON.stringify(opts)}`;
    let m = local.get(ck);
    if (!m) {
      m = localMat(k).clone();
      if (opts && opts.color !== undefined) m.color = new THREE.Color(opts.color);
      if (opts && opts.roughness !== undefined) m.roughness = opts.roughness;
      if (opts && opts.side === 'double') m.side = THREE.DoubleSide;
      if (opts && opts.emissive !== undefined) m.emissive = new THREE.Color(opts.emissive);
      if (opts && opts.emissiveIntensity !== undefined) m.emissiveIntensity = opts.emissiveIntensity;
      m.needsUpdate = true;
      local.set(ck, m); owned.push(m);
    }
    return m;
  }

  return {
    base, variant, substituted,
    dispose() { for (const m of owned) m.dispose(); owned.length = 0; local.clear(); },
  };
}

/* -------------------------------------------------------------------------- */
/* Room frame — local <-> world, and the four walls                            */
/* -------------------------------------------------------------------------- */

/** Room-local -> world. Matches layout.plotToWorld's rotation exactly. */
function toWorld(room, lx, ly, lz, out = _v) {
  const c = Math.cos(room.rotation), s = Math.sin(room.rotation);
  return out.set(
    room.centre[0] + lx * c + lz * s,
    room.floorY + ly,
    room.centre[2] - lx * s + lz * c,
  );
}

/** World -> room-local (x,z). Correct inverse of the above. */
function toLocal(room, x, z, out) {
  const c = Math.cos(room.rotation), s = Math.sin(room.rotation);
  const dx = x - room.centre[0], dz = z - room.centre[2];
  out[0] = c * dx - s * dz;
  out[1] = s * dx + c * dz;
  return out;
}

const _loc = [0, 0];

/**
 * A wall as an origin + an "along" axis + an "inward" axis, all in room-local
 * space, plus the yaw that turns an object's local +Z into `inward`.
 */
function wallInfo(C, wall) {
  switch (wall) {
    case 'front': return { ax: -1, az: 0, ix: 0, iz: -1, ox: 0, oz: C.hd, span: C.hw, yaw: Math.PI };
    case 'left': return { ax: 0, az: -1, ix: 1, iz: 0, ox: -C.hw, oz: 0, span: C.hd, yaw: Math.PI / 2 };
    case 'right': return { ax: 0, az: 1, ix: -1, iz: 0, ox: C.hw, oz: 0, span: C.hd, yaw: -Math.PI / 2 };
    default: return { ax: 1, az: 0, ix: 0, iz: 1, ox: 0, oz: -C.hd, span: C.hw, yaw: 0 };
  }
}

const WALLS = ['back', 'front', 'left', 'right'];

function wallPoint(wi, s, t, out = [0, 0]) {
  out[0] = wi.ox + wi.ax * s + wi.ix * t;
  out[1] = wi.oz + wi.az * s + wi.iz * t;
  return out;
}

/** AABB (room-local) of a slab sitting on `wall`, `len` along it, t0..t1 deep. */
function wallRect(C, wall, s, len, t0, t1) {
  const wi = wallInfo(C, wall);
  const a = wallPoint(wi, s - len / 2, t0, [0, 0]);
  const b = wallPoint(wi, s + len / 2, t1, [0, 0]);
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
}

/** AABB of a free-standing item. Yaw is only ever axis-aligned in a room. */
function rectAt(x, z, dx, dz, yaw = 0) {
  const swap = Math.abs(Math.cos(yaw)) < 0.5;
  const ex = (swap ? dz : dx) * 0.5;
  const ez = (swap ? dx : dz) * 0.5;
  return [x - ex, z - ez, x + ex, z + ez];
}

function overlaps(a, b, pad = 0.02) {
  return a[0] < b[2] - pad && a[2] > b[0] + pad && a[1] < b[3] - pad && a[3] > b[1] + pad;
}

/** Occupancy source tags, kept in `C.occTag` alongside `C.occ`. */
const OCC_OURS = 0;
/** A volume the interiors stream published — a stair flight, a chimney breast. */
const OCC_KEEPOUT = 1;

function claim(C, rect, tag = OCC_OURS) {
  C.occ.push(rect);
  C.occTag.push(tag);
  return rect;
}

function freeRect(C, rect) {
  for (let i = 0; i < C.occ.length; i++) {
    if (overlaps(rect, C.occ[i])) {
      // Count only rejections caused by REAL published geometry: that is the
      // honest measure of "furniture that had to move because of a keep-out",
      // as opposed to furniture stepping around other furniture.
      if (C.occTag[i] === OCC_KEEPOUT) C.koBlocks++;
      return false;
    }
  }
  return true;
}

/**
 * `pad` is the clearance demanded from the wall FACE. It defaults to zero
 * because most furniture is supposed to touch a wall — an earlier default of
 * 0.04 silently rejected every wall-lined item (a shelf run's rectangle starts
 * exactly on the wall plane), which emptied the rooms of dressers, chests,
 * settles and pegs while the tables and beds, placed by `findFree`, stayed.
 */
function insideRoom(C, rect, pad = 0) {
  const e = 1e-3;
  return rect[0] > -C.hw + pad - e && rect[2] < C.hw - pad + e &&
    rect[1] > -C.hd + pad - e && rect[3] < C.hd - pad + e;
}

/** True when nothing tall would stand in front of a window. */
function clearOfWindows(C, rect, pad = 0.3) {
  for (const w of C.windows) {
    if (w.lx > rect[0] - pad && w.lx < rect[2] + pad &&
        w.lz > rect[1] - pad && w.lz < rect[3] + pad) return false;
  }
  return true;
}

/**
 * Find a free run along a wall, starting from `prefer` and walking outward in
 * 0.32 m steps. This is what keeps 18 differently-shaped rooms honest without
 * hand-placing every dresser.
 */
function findOnWall(C, wall, len, t0, t1, prefer = 0, tall = false) {
  const wi = wallInfo(C, wall);
  const lim = wi.span - len / 2 - 0.1;
  if (lim < 0) return null;
  const p = clamp(prefer, -lim, lim);
  for (let k = 0; k < 26; k++) {
    const step = Math.ceil(k / 2) * 0.32 * (k % 2 ? 1 : -1);
    const s = p + step;
    if (s < -lim || s > lim) continue;
    const rect = wallRect(C, wall, s, len, t0, t1);
    if (!freeRect(C, rect) || !insideRoom(C, rect)) continue;
    if (tall && !clearOfWindows(C, rect)) continue;
    return { s, rect, wi, x: wi.ox + wi.ax * s + wi.ix * (t0 + t1) / 2,
      z: wi.oz + wi.az * s + wi.iz * (t0 + t1) / 2, yaw: wi.yaw };
  }
  return null;
}

/** Find a free spot for a free-standing item near (x,z), spiralling outward. */
function findFree(C, x, z, dx, dz, yaw = 0, reach = 2.4) {
  for (let k = 0; k < 40; k++) {
    const a = k * 2.399963;
    const r = k === 0 ? 0 : reach * Math.sqrt(k / 40);
    const px = x + Math.cos(a) * r, pz = z + Math.sin(a) * r;
    const rect = rectAt(px, pz, dx, dz, yaw);
    if (freeRect(C, rect) && insideRoom(C, rect, 0.16)) return { x: px, z: pz, rect, yaw };
  }
  return null;
}

/* -------------------------------------------------------------------------- */
/* Emit helpers                                                               */
/* -------------------------------------------------------------------------- */

/** A world-space static box collider from a room-local box. */
function addBox(C, lx, ly, lz, hx, hy, hz, yaw = 0, tag = 'furniture') {
  const p = toWorld(C.room, lx, ly, lz, _v);
  const swap = Math.abs(Math.cos(yaw)) < 0.5;
  C.out.colliders.push(boxCollider(
    [p.x, p.y, p.z],
    [swap ? hz : hx, hy, swap ? hx : hz],
    quatY(C.room.rotation),
    { tag, friction: 0.9, restitution: 0.02 },
  ));
}

/** Box collider straight from a room-local AABB and a height range. */
function addRectBox(C, rect, y0, y1, tag = 'furniture', shrink = 0.02) {
  const hx = Math.max(0.04, (rect[2] - rect[0]) / 2 - shrink);
  const hz = Math.max(0.04, (rect[3] - rect[1]) / 2 - shrink);
  addBox(C, (rect[0] + rect[2]) / 2, (y0 + y1) / 2, (rect[1] + rect[3]) / 2,
    hx, Math.max(0.03, (y1 - y0) / 2), hz, 0, tag);
}

function addCyl(C, lx, ly, lz, r, hh, tag = 'furniture') {
  const p = toWorld(C.room, lx, ly, lz, _v);
  C.out.colliders.push(cylinderCollider([p.x, p.y, p.z], r, hh, null,
    { tag, friction: 0.9, restitution: 0.02 }));
}

/**
 * A cylinder whose axis is not vertical — a cask on its side. The Euler is the
 * same one the mesh's local matrix used, composed under the room's rotation, so
 * the collider cannot drift out of agreement with the geometry.
 */
function addCylRot(C, lx, ly, lz, r, hh, rx, ry, rz, tag = 'furniture') {
  const p = toWorld(C.room, lx, ly, lz, _v);
  const pos = [p.x, p.y, p.z];
  _q.setFromEuler(_e.set(rx, ry, rz, 'XYZ'));
  _q2.setFromAxisAngle(_YUP, C.room.rotation).multiply(_q);
  C.out.colliders.push(cylinderCollider(pos, r, hh, [_q2.x, _q2.y, _q2.z, _q2.w],
    { tag, friction: 0.9, restitution: 0.02 }));
}

/** A flame instance. `s` is its height in metres. */
function addFlame(C, lx, ly, lz, s, kind) {
  const p = toWorld(C.room, lx, ly, lz, _v);
  C.out.flames.push({
    x: p.x, y: p.y, z: p.z, s, room: C.idx,
    phase: C.rng.range(0, TAU), rate: C.rng.range(0.85, 1.25), kind,
  });
}

function addEmber(C, lx, ly, lz, s) {
  const p = toWorld(C.room, lx, ly, lz, _v);
  C.out.embers.push({ x: p.x, y: p.y, z: p.z, s, room: C.idx, phase: C.rng.range(0, TAU) });
}

/**
 * A light anchor for the lighting stream's pool. `indoor` and `plotId` are
 * extra, ignored fields — they let lighting gate interior pools by occupancy
 * later without another contract change.
 */
function addAnchor(C, lx, ly, lz, kind, colour, intensity) {
  const p = toWorld(C.room, lx, ly, lz, _v);
  C.out.anchors.push({
    position: [p.x, p.y, p.z], kind,
    colour, color: colour, intensity,
    indoor: true, plotId: C.room.plotId, storey: C.room.storey,
  });
}

/**
 * Propose a grabbable. Nothing is spawned yet: candidates are sorted by `rank`
 * across every room afterwards, so a tight dynamic-body budget takes the fifth
 * tankard out of every room before it takes the first one out of any.
 */
function addGrab(C, kind, lx, ly, lz, rank, yaw = 0) {
  // Anything standing on the floor has to be nudged clear of the furniture:
  // a dynamic body that spawns inside a static box gets fired across the room
  // on the first physics step. Things resting on a surface are trusted — the
  // recipe put them there on purpose.
  if (ly < 0.35) {
    const spot = findFree(C, lx, lz, 0.42, 0.42, 0, 2.4);
    if (!spot) return;
    lx = spot.x; lz = spot.z;
    claim(C, spot.rect);
  }
  const p = toWorld(C.room, lx, ly, lz, _v);
  C.out.grabs.push({
    kind, rank, room: C.idx,
    pos: [p.x, p.y, p.z], yaw: C.room.rotation + yaw,
  });
}

/* -------------------------------------------------------------------------- */
/* THE KIT — parts every room is composed from                                 */
/* -------------------------------------------------------------------------- */

/* --- surfaces ------------------------------------------------------------- */

/** Trestle table: plank top, two splayed trestles, a stretcher. */
function kitTable(C, x, z, yaw, len, wide, h = 0.74) {
  const M = C.mat, B = C.bin;
  const t = 0.055;
  const boards = Math.max(2, Math.round(wide / 0.34));
  const bw = wide / boards;
  for (let i = 0; i < boards; i++) {
    const off = (i - (boards - 1) / 2) * bw;
    B.add('plank', M.plank, gBox(len, t, bw * 0.96),
      trs(alongZ(x, yaw, off), h - t / 2, alongZz(z, yaw, off), 0, yaw, 0), { uvScale: 1.4, cast: true });
  }
  for (const sx of [-1, 1]) {
    const ox = sx * (len / 2 - 0.5);
    const px = alongX(x, yaw, ox), pz = alongXz(z, yaw, ox);
    // trestle: a post between two splayed bearers
    B.add('wood', M.wood, gBox(0.1, h - t, 0.1), trs(px, (h - t) / 2, pz, 0, yaw, 0), { uvScale: 3, cast: true });
    B.add('wood', M.wood, gBox(0.13, 0.07, 0.72), trs(px, 0.045, pz, 0, yaw, 0), { uvScale: 3 });
    B.add('wood', M.wood, gBox(0.11, 0.06, 0.62), trs(px, h - t - 0.03, pz, 0, yaw, 0), { uvScale: 3 });
  }
  B.add('wood', M.wood, gBox(len - 1.1, 0.07, 0.07),
    trs(x, h * 0.42, z, 0, yaw, 0), { uvScale: 3 });
  const rect = rectAt(x, z, len, wide, yaw);
  addRectBox(C, rect, 0, h + 0.02, 'table', 0.06);
  return { rect, top: h };
}

/** Backless bench. Low enough that the player can step onto it. */
function kitBench(C, x, z, yaw, len, h = 0.45) {
  const M = C.mat, B = C.bin;
  B.add('plank', M.plank, gBox(len, 0.055, 0.3),
    trs(x, h - 0.03, z, 0, yaw, 0), { uvScale: 1.8, cast: true });
  for (const sx of [-1, 1]) {
    const ox = sx * (len / 2 - 0.28);
    B.add('wood', M.wood, gBox(0.055, h - 0.06, 0.26),
      trs(alongX(x, yaw, ox), (h - 0.06) / 2, alongXz(z, yaw, ox), 0, yaw, 0.09 * sx), { uvScale: 3 });
  }
  const rect = rectAt(x, z, len, 0.32, yaw);
  addRectBox(C, rect, 0, h, 'bench', 0.03);
  return rect;
}

/** High-backed settle — the seat you put beside a fire. */
function kitSettle(C, x, z, yaw, len) {
  const M = C.mat, B = C.bin;
  const h = 0.45, back = 1.18;
  B.add('plank', M.plank, gBox(len, 0.055, 0.42), trs(x, h - 0.03, z, 0, yaw, 0), { uvScale: 1.6, cast: true });
  // back panel, pushed to the rear of the seat (its local -Z)
  B.add('plank', M.plank, gBox(len, back - 0.1, 0.05),
    trs(alongZ(x, yaw, -0.2), back / 2 + 0.05, alongZz(z, yaw, -0.2), 0, yaw, 0), { uvScale: 1.6, cast: true });
  for (const sx of [-1, 1]) {
    const ox = sx * (len / 2 - 0.03);
    B.add('wood', M.wood, gBox(0.07, back, 0.42),
      trs(alongX(x, yaw, ox), back / 2, alongXz(z, yaw, ox), 0, yaw, 0), { uvScale: 3 });
  }
  const rect = rectAt(x, z, len, 0.5, yaw);
  addRectBox(C, rect, 0, back, 'settle', 0.04);
  return rect;
}

/** Counter / bar: panelled front, worn plank top with a bullnose. */
function kitCounter(C, x, z, yaw, len, h = 0.9, d = 0.62) {
  const M = C.mat, B = C.bin;
  B.add('plank', M.plank, gBox(len, 0.07, d + 0.06), trs(x, h - 0.035, z, 0, yaw, 0), { uvScale: 1.3, cast: true });
  B.add('wood', M.wood, gBox(len, h - 0.07, 0.06),
    trs(x + Math.sin(yaw) * (d / 2), (h - 0.07) / 2, z + Math.cos(yaw) * (d / 2), 0, yaw, 0), { uvScale: 1.6 });
  const panels = Math.max(2, Math.round(len / 0.7));
  for (let i = 0; i < panels; i++) {
    const off = (i - (panels - 1) / 2) * (len / panels);
    B.add('wood', M.wood, gBox(0.07, h - 0.14, 0.1),
      trs(x + Math.cos(yaw) * off + Math.sin(yaw) * (d / 2 - 0.03),
        (h - 0.1) / 2, z - Math.sin(yaw) * off + Math.cos(yaw) * (d / 2 - 0.03), 0, yaw, 0), { uvScale: 3 });
  }
  for (const sx of [-1, 1]) {
    const ox = sx * (len / 2 - 0.04);
    B.add('wood', M.wood, gBox(0.08, h - 0.07, d),
      trs(x + Math.cos(yaw) * ox, (h - 0.07) / 2, z - Math.sin(yaw) * ox, 0, yaw, 0), { uvScale: 2 });
  }
  const rect = rectAt(x, z, len, d + 0.06, yaw);
  addRectBox(C, rect, 0, h + 0.02, 'counter', 0.03);
  return { rect, top: h };
}

/** Open shelving on brackets. `stock` fills it. */
function kitShelves(C, wall, s, len, tiers, y0, dy, stock) {
  const M = C.mat, B = C.bin;
  const wi = wallInfo(C, wall);
  const d = 0.26;
  const rect = wallRect(C, wall, s, len, 0.0, d);
  for (let i = 0; i < tiers; i++) {
    const y = y0 + i * dy;
    const p = wallPoint(wi, s, d / 2, [0, 0]);
    B.add('plank', M.plank, gBox(len, 0.035, d), trs(p[0], y, p[1], 0, wi.yaw, 0), { uvScale: 1.6 });
    // brackets
    for (const sx of [-1, 1]) {
      const q = wallPoint(wi, s + sx * (len / 2 - 0.1), d / 2, [0, 0]);
      B.add('wood', M.wood, gBox(0.05, 0.14, d - 0.04), trs(q[0], y - 0.09, q[1], 0, wi.yaw, 0), { uvScale: 4 });
    }
    if (stock) stock(C, wall, s, len, y + 0.02, i);
  }
  addRectBox(C, rect, y0 - 0.16, y0 + (tiers - 1) * dy + 0.06, 'shelves', 0.01);
  claim(C, rect);
  return rect;
}

/** Dresser: cupboard base, open shelves above, crockery on them. */
function kitDresser(C, wall, prefer) {
  const M = C.mat, B = C.bin;
  const len = clamp(C.W * 0.28, 1.15, 1.7);
  const spot = findOnWall(C, wall, len, 0, 0.5, prefer, true);
  if (!spot) return null;
  const wi = spot.wi, s = spot.s;
  const top = Math.min(2.0, C.H - 0.3);
  const base = 0.88;
  const c = wallPoint(wi, s, 0.24, [0, 0]);
  B.add('wood', M.wood, gBox(len, base, 0.46), trs(c[0], base / 2, c[1], 0, wi.yaw, 0), { uvScale: 1.4, cast: true });
  // two drawers + a pair of doors, faked with applied mouldings
  const f = wallPoint(wi, s, 0.47, [0, 0]);
  for (const sx of [-1, 1]) {
    B.add('plank', M.plank, gBox(len / 2 - 0.08, base * 0.52, 0.02),
      trs(f[0] + wi.ax * sx * len * 0.24, base * 0.3, f[1] + wi.az * sx * len * 0.24, 0, wi.yaw, 0), { uvScale: 2 });
    B.add('plank', M.plank, gBox(len / 2 - 0.08, 0.16, 0.02),
      trs(f[0] + wi.ax * sx * len * 0.24, base * 0.78, f[1] + wi.az * sx * len * 0.24, 0, wi.yaw, 0), { uvScale: 2 });
    B.add('iron', M.iron, gSph(0.022, 6, 4),
      trs(f[0] + wi.ax * sx * len * 0.24, base * 0.78, f[1] + wi.az * sx * len * 0.24), { uvScale: 8 });
  }
  // upper carcass
  const u = wallPoint(wi, s, 0.15, [0, 0]);
  B.add('wood', M.wood, gBox(len, 0.04, 0.3), trs(u[0], top, u[1], 0, wi.yaw, 0), { uvScale: 1.6, cast: true });
  for (const sx of [-1, 1]) {
    const q = wallPoint(wi, s + sx * (len / 2 - 0.02), 0.15, [0, 0]);
    B.add('wood', M.wood, gBox(0.04, top - base, 0.3), trs(q[0], (top + base) / 2, q[1], 0, wi.yaw, 0), { uvScale: 3 });
  }
  const tiers = 3;
  for (let i = 0; i < tiers; i++) {
    const y = base + 0.14 + i * ((top - base - 0.2) / (tiers - 1));
    B.add('plank', M.plank, gBox(len - 0.06, 0.03, 0.28), trs(u[0], y, u[1], 0, wi.yaw, 0), { uvScale: 1.8 });
    stockCrockery(C, wall, s, len - 0.24, y + 0.015, i);
  }
  const rect = spot.rect;
  addRectBox(C, rect, 0, base + 0.02, 'dresser', 0.02);
  addRectBox(C, wallRect(C, wall, s, len, 0, 0.32), base, top, 'dresser', 0.02);
  claim(C, rect);
  return { s, wall, len, base, top, rect };
}

/* --- storage -------------------------------------------------------------- */

/** Boarded chest with iron straps. */
function kitChest(C, x, z, yaw, len = 0.95) {
  const M = C.mat, B = C.bin;
  const h = 0.52, d = 0.48;
  B.add('wood', M.wood, gBox(len, h - 0.07, d), trs(x, (h - 0.07) / 2, z, 0, yaw, 0), { uvScale: 1.8, cast: true });
  B.add('plank', M.plank, gBox(len + 0.03, 0.08, d + 0.03), trs(x, h - 0.035, z, 0, yaw, 0), { uvScale: 2 });
  for (const sx of [-1, 1]) {
    B.add('iron', M.iron, gBox(0.05, h + 0.01, d + 0.05),
      trs(x + Math.cos(yaw) * sx * len * 0.32, h / 2 - 0.02, z - Math.sin(yaw) * sx * len * 0.32, 0, yaw, 0), { uvScale: 6 });
  }
  B.add('iron', M.iron, gBox(0.09, 0.13, 0.03),
    trs(x + Math.sin(yaw) * (d / 2 + 0.01), h - 0.13, z + Math.cos(yaw) * (d / 2 + 0.01), 0, yaw, 0), { uvScale: 8 });
  const rect = rectAt(x, z, len, d, yaw);
  addRectBox(C, rect, 0, h, 'chest', 0.02);
  return rect;
}

/** Coopered barrel, upright or on its side. */
function barrelGeo(seg) {
  return gLathe('barrel', [
    [0.02, 0], [0.26, 0], [0.29, 0.05], [0.335, 0.28], [0.345, 0.43],
    [0.335, 0.58], [0.29, 0.81], [0.26, 0.86], [0.02, 0.86],
  ], seg);
}

function kitBarrel(C, x, y, z, yaw, lying = false, scale = 1) {
  const M = C.mat, B = C.bin;
  const g = barrelGeo(C.seg);
  const rx = lying ? Math.PI / 2 : 0;
  B.add('wood', M.wood, g, trs(x, y + (lying ? 0.345 * scale : 0), z, rx, yaw, 0, scale, scale, scale),
    { uvScale: 1.8, cast: true });
  for (const t of [0.08, 0.43, 0.78]) {
    B.add('iron', M.iron, gCyl(0.35 * scale, 0.35 * scale, 0.05 * scale, C.seg, true),
      lying
        ? trs(x + Math.cos(yaw) * (t - 0.43) * 0.86 * scale, y + 0.345 * scale,
          z - Math.sin(yaw) * (t - 0.43) * 0.86 * scale, Math.PI / 2, yaw, 0)
        : trs(x, y + t * 0.86 * scale, z, 0, yaw, 0),
      { uvScale: 4 });
  }
  if (lying) {
    addCylRot(C, x, y + 0.345 * scale, z, 0.35 * scale, 0.43 * scale, rx, yaw, 0, 'barrel');
  } else {
    addRectBox(C, rectAt(x, z, 0.7 * scale, 0.7 * scale), y, y + 0.86 * scale, 'barrel', 0.02);
  }
}

/** A lumpy tied sack. */
function sackGeo() {
  return cached('sack', () => {
    const g = new THREE.SphereGeometry(0.26, 10, 8);
    g.scale(1.0, 1.24, 0.84);
    const p = g.getAttribute('position');
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i);
      const f = 1 + 0.14 * Math.sin(p.getX(i) * 7.3 + y * 4.1) + 0.09 * Math.sin(p.getZ(i) * 9.7);
      const taper = y > 0.13 ? 0.52 : 1;
      p.setXYZ(i, p.getX(i) * f * taper, y, p.getZ(i) * f * taper);
    }
    g.computeVertexNormals();
    g.translate(0, 0.32, 0);
    return g;
  });
}

function kitSack(C, x, y, z, yaw, scale = 1) {
  const M = C.mat, B = C.bin;
  B.add('sack', M.sack, sackGeo(), trs(x, y, z, 0, yaw, 0, scale, scale, scale), { uvScale: 3, cast: true });
  B.add('wood', M.wood, gCyl(0.05 * scale, 0.06 * scale, 0.07 * scale, 7),
    trs(x, y + 0.63 * scale, z, 0, yaw, 0), { uvScale: 6 });
}

/** Boarded crate with corner battens. */
function crateGeo(s) {
  return cached(`crate|${s}`, () => {
    const t = s * 0.05;
    const parts = [{ geo: gBox(s, s, s), matrix: trs(0, 0, 0).clone(), uvScale: 1.8 }];
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      parts.push({ geo: gBox(t * 2, s * 1.02, t * 2), matrix: trs(sx * s / 2, 0, sz * s / 2).clone(), uvScale: 3 });
    }
    for (const sz of [-1, 1]) {
      parts.push({ geo: gBox(s * 1.02, t * 2, t * 2), matrix: trs(0, s * 0.34, sz * s / 2).clone(), uvScale: 3 });
      parts.push({ geo: gBox(t * 2, t * 2, s * 1.02), matrix: trs(sz * s / 2, s * 0.34, 0).clone(), uvScale: 3 });
    }
    return mergeParts(parts);
  });
}

function kitCrate(C, x, y, z, yaw, s = 0.56) {
  C.bin.add('plank', C.mat.plank, crateGeo(s), trs(x, y + s / 2, z, 0, yaw, 0), { uv: 'keep', cast: true });
  addRectBox(C, rectAt(x, z, s, s, yaw), y, y + s, 'crate', 0.02);
}

/* --- vessels and clutter (static) ---------------------------------------- */

const crockGeo = (seg) => gLathe('crock', [
  [0.0, 0], [0.055, 0], [0.082, 0.05], [0.088, 0.13], [0.072, 0.19],
  [0.078, 0.205], [0.064, 0.205], [0.058, 0.185], [0.05, 0.06], [0.0, 0.055],
], seg);

const jugGeo = (seg) => gLathe('jug', [
  [0.0, 0], [0.05, 0], [0.072, 0.04], [0.078, 0.12], [0.05, 0.19],
  [0.035, 0.24], [0.04, 0.255], [0.028, 0.255], [0.026, 0.2], [0.045, 0.13], [0.0, 0.06],
], seg);

const tankardGeo = (seg) => cached(`tankard|${seg}`, () => mergeParts([
  {
    geo: gLathe('tankardBody', [
      [0.0, 0], [0.046, 0], [0.05, 0.012], [0.053, 0.115], [0.056, 0.128],
      [0.05, 0.132], [0.046, 0.02], [0.0, 0.018],
    ], seg),
    matrix: trs(0, 0, 0).clone(), uvScale: 5,
  },
  { geo: gTorus(0.032, 0.007, 4, 6, Math.PI), matrix: trs(0.052, 0.068, 0, 0, 0, -Math.PI / 2).clone(), uvScale: 8 },
]));

const bowlGeo = (seg) => gLathe('bowl', [
  [0.0, 0], [0.04, 0], [0.09, 0.045], [0.1, 0.07], [0.092, 0.072],
  [0.082, 0.05], [0.038, 0.014], [0.0, 0.012],
], seg);

const plateGeo = (seg) => gLathe('plate', [
  [0.0, 0], [0.09, 0.004], [0.11, 0.024], [0.112, 0.03], [0.1, 0.028], [0.085, 0.012], [0.0, 0.008],
], seg);

const bottleGeo = (seg) => gLathe('bottle', [
  [0.0, 0], [0.044, 0], [0.048, 0.02], [0.046, 0.135], [0.02, 0.185],
  [0.017, 0.26], [0.021, 0.272], [0.014, 0.276], [0.0, 0.276],
], seg);

const pailGeo = (seg) => cached(`pail|${seg}`, () => mergeParts([
  {
    geo: gLathe('pailBody', [
      [0.0, 0], [0.145, 0], [0.15, 0.02], [0.185, 0.3], [0.196, 0.315],
      [0.176, 0.315], [0.17, 0.29], [0.135, 0.03], [0.0, 0.026],
    ], seg),
    matrix: trs(0, 0, 0).clone(), uvScale: 2.6,
  },
  { geo: gCyl(0.172, 0.16, 0.03, seg, true), matrix: trs(0, 0.14, 0).clone(), uvScale: 5 },
]));

const loafGeo = () => cached('loaf', () => {
  const g = gSph(0.115, 9, 7).clone();
  g.scale(1.45, 0.68, 0.95);
  g.translate(0, 0.078, 0);
  return finish(prep(g, null, 'world', 3));
});

/** Pewter and stoneware on a shelf run. */
function stockCrockery(C, wall, s, len, y, tier) {
  const M = C.mat, B = C.bin, wi = wallInfo(C, wall), rng = C.rng;
  const n = Math.max(2, Math.floor(len / 0.3));
  for (let i = 0; i < n; i++) {
    if (rng.bool(0.22)) continue;
    const off = (i - (n - 1) / 2) * (len / n) + rng.sym(0.03);
    const p = wallPoint(wi, s + off, 0.13 + rng.sym(0.02), [0, 0]);
    const r = rng.next();
    if (tier === 0 || r < 0.35) {
      B.add('pewter', M.pewter, plateGeo(C.seg),
        trs(p[0], y + 0.055, p[1], Math.PI / 2 - 0.16, wi.yaw, 0), { uvScale: 5 });
    } else if (r < 0.7) {
      B.add('crock', M.crock, crockGeo(C.seg), trs(p[0], y, p[1], 0, rng.range(0, TAU), 0), { uvScale: 4 });
    } else {
      B.add('pewter', M.pewter, tankardGeo(C.seg), trs(p[0], y, p[1], 0, rng.range(0, TAU), 0), { uv: 'keep' });
    }
  }
}

/** A lit candle: brass stick, wax, flame instance, light anchor. */
function kitCandle(C, x, y, z, lit = true) {
  const M = C.mat, B = C.bin;
  B.add('iron', M.iron, gCyl(0.052, 0.062, 0.018, 8), trs(x, y + 0.009, z), { uvScale: 8 });
  B.add('iron', M.iron, gCyl(0.014, 0.017, 0.05, 7), trs(x, y + 0.043, z), { uvScale: 8 });
  const wax = 0.115;
  B.add('wax', M.wax, gCyl(0.011, 0.013, wax, 7), trs(x, y + 0.068 + wax / 2, z), { uvScale: 8 });
  if (lit) {
    addFlame(C, x, y + 0.068 + wax, z, 0.075, 'candle');
    addAnchor(C, x, y + 0.11 + wax, z, 'candle', 0xffc47a, 0.9);
  }
}

/** Bunches of herbs hung from a joist. */
function kitHerbs(C, x, z, y, n) {
  const M = C.mat, B = C.bin, rng = C.rng;
  const g = cached('herb', () => crossQuad(0.16, 3, 0.34));
  B.add('wood', M.wood, gBox(0.04, 0.04, 1.1), trs(x, y + 0.02, z), { uvScale: 4 });
  for (let i = 0; i < n; i++) {
    const off = (i - (n - 1) / 2) * (1.0 / Math.max(1, n));
    B.add('leafCard', M.leaf, g, trs(x, y - 0.34, z + off, Math.PI, rng.range(0, TAU), 0),
      { uv: 'keep' });
  }
}

/** A rug — one quad, no collider, slightly proud of the floor. */
function kitRug(C, x, z, w, d, yaw = 0) {
  const rect = rectAt(x, z, w, d, yaw);
  C.bin.add('rug', C.mat.rug, gPlane(w, d), trs(x, 0.012, z, -Math.PI / 2, 0, yaw), { uvScale: 1.1 });
  return rect;
}

/** Peg rail with a couple of garments on it. */
function kitPegs(C, wall, prefer) {
  const M = C.mat, B = C.bin;
  const len = 1.1;
  const spot = findOnWall(C, wall, len, 0, 0.14, prefer, true);
  if (!spot) return null;
  const wi = spot.wi;
  const y = 1.62;
  const p = wallPoint(wi, spot.s, 0.05, [0, 0]);
  B.add('wood', M.wood, gBox(len, 0.09, 0.05), trs(p[0], y, p[1], 0, wi.yaw, 0), { uvScale: 4 });
  for (let i = 0; i < 4; i++) {
    const off = (i - 1.5) * 0.28;
    const q = wallPoint(wi, spot.s + off, 0.11, [0, 0]);
    B.add('wood', M.wood, gCyl(0.014, 0.018, 0.13, 6), trs(q[0], y, q[1], Math.PI / 2, wi.yaw, 0), { uvScale: 8 });
    if (i === 1 || i === 3) {
      const c = wallPoint(wi, spot.s + off, 0.13, [0, 0]);
      B.add('linen', M.linen, gPlane(0.42, 0.72), trs(c[0], y - 0.36, c[1], 0, wi.yaw, 0), { uvScale: 1.4 });
    }
  }
  claim(C, spot.rect);
  return spot;
}

/* --- the hearth ----------------------------------------------------------- */

/**
 * The centrepiece. Stone jambs and fireback in the chimney breast, an oak
 * bressumer over it, a raised hearthstone, firedogs, logs, and a fire that the
 * lighting stream turns into a pool of light.
 */
function kitHearth(C, opts = {}) {
  const M = C.mat, B = C.bin, rng = C.rng;
  const wall = opts.wall || 'back';
  const wi = wallInfo(C, wall);
  const fw = clamp(opts.width ?? 1.5, 1.0, Math.min(2.6, C.W * 0.34));
  const half = fw / 2;
  let s = clamp(opts.s ?? 0, -(wi.span - half - 0.55), wi.span - half - 0.55);

  // Never bury a window behind the chimney breast, and never block a door.
  for (let k = 0; k < 14; k++) {
    const r = wallRect(C, wall, s, fw + 0.9, 0, 0.85);
    if (freeRect(C, r) && clearOfWindows(C, r, 0.2) && insideRoom(C, r)) break;
    const step = Math.ceil((k + 1) / 2) * 0.4 * (k % 2 ? 1 : -1);
    s = clamp((opts.s ?? 0) + step, -(wi.span - half - 0.55), wi.span - half - 0.55);
  }

  const fh = clamp(C.H * 0.6, 1.15, 1.5);       // firebox opening height
  const jh = fh + 0.16;                          // top of the jambs
  const stoneD = 0.46;                           // firebox depth
  const at = (ss, t) => wallPoint(wi, s + ss, t, [0, 0]);

  // raised hearthstone, projecting into the room
  const hsW = fw + 0.66, hsD = stoneD + 0.42;
  const hc = at(0, hsD / 2 - 0.02);
  B.add('hearth', M.hearth, gBox(hsW, 0.1, hsD), trs(hc[0], 0.05, hc[1], 0, wi.yaw, 0), { uvScale: 1.1 });
  addRectBox(C, wallRect(C, wall, s, hsW, 0, hsD - 0.04), 0, 0.1, 'hearthstone', 0.0);

  // fireback and firebox cheeks, sooty brick
  const bc = at(0, 0.05);
  B.add('soot', M.soot, gBox(fw, fh + 0.1, 0.09), trs(bc[0], (fh + 0.1) / 2, bc[1], 0, wi.yaw, 0), { uvScale: 1.6 });
  for (const sx of [-1, 1]) {
    const q = at(sx * (half - 0.06), stoneD / 2 + 0.05);
    B.add('soot', M.soot, gBox(0.1, fh, stoneD), trs(q[0], fh / 2, q[1], 0, wi.yaw, 0), { uvScale: 2 });
  }

  // stone jambs
  for (const sx of [-1, 1]) {
    const q = at(sx * (half + 0.18), stoneD * 0.55);
    const g = hew(gBox(0.36, jh, stoneD * 1.1).clone(), 0.014, rng);
    B.add('hearth', M.hearth, g, trs(q[0], jh / 2, q[1], 0, wi.yaw, 0), { uvScale: 1.2, cast: true });
    g.dispose();
    addRectBox(C, wallRect(C, wall, s + sx * (half + 0.18), 0.36, 0, stoneD * 1.1), 0, jh, 'hearth-jamb', 0.01);
  }

  // oak bressumer
  const brC = at(0, stoneD * 0.5);
  const brW = fw + 0.84;
  B.add('wood', M.wood, gBox(brW, 0.27, stoneD * 1.15), trs(brC[0], jh + 0.135, brC[1], 0, wi.yaw, 0),
    { uvScale: 1.3, cast: true });

  // smoke hood: three sooty quads closing the gap up to the ceiling
  const hoodTop = Math.max(jh + 0.55, C.H - 0.02);
  const y0 = jh + 0.27;
  const bw = brW / 2, tw = Math.min(0.55, brW / 2 - 0.1);
  const bd = stoneD * 1.15, td = 0.24;
  const P = (ss, t, y) => { const q = at(ss, t); return [q[0], y, q[1]]; };
  B.add('soot', M.soot, quad(P(-bw, bd, y0), P(bw, bd, y0), P(tw, td, hoodTop), P(-tw, td, hoodTop)),
    null, { uvScale: 1.4 });
  for (const sx of [-1, 1]) {
    B.add('soot', M.soot, quad(P(sx * bw, bd, y0), P(sx * bw, 0, y0), P(sx * tw, 0, hoodTop), P(sx * tw, td, hoodTop)),
      null, { uvScale: 1.4 });
  }
  B.add('soot', M.soot, gBox(tw * 2, 0.04, td), trs(at(0, td / 2)[0], hoodTop, at(0, td / 2)[1], 0, wi.yaw, 0), { uvScale: 2 });

  if (opts.fire === false) {
    claim(C, wallRect(C, wall, s, hsW, 0, hsD + 0.15));
    return { wall, s, width: fw, front: hsD, hearthTop: 0.1, jh, wi };
  }

  // firedogs
  for (const sx of [-1, 1]) {
    const q = at(sx * (half - 0.3), stoneD * 0.55);
    B.add('iron', M.iron, gBox(0.05, 0.05, stoneD * 0.9), trs(q[0], 0.19, q[1], 0, wi.yaw, 0), { uvScale: 6 });
    B.add('iron', M.iron, gCyl(0.02, 0.026, 0.3, 6), trs(q[0], 0.25, q[1], 0, wi.yaw, 0), { uvScale: 8 });
  }
  // logs
  const logs = 4;
  for (let i = 0; i < logs; i++) {
    const q = at(rng.sym(half - 0.28), stoneD * 0.5 + rng.sym(0.07));
    const r = rng.range(0.045, 0.07);
    B.add('bark', M.bark, gCyl(r, r * 0.92, fw * 0.62, 7),
      trs(q[0], 0.16 + i * 0.055, q[1], 0, wi.yaw + rng.sym(0.3), Math.PI / 2 + rng.sym(0.08)), { uvScale: 3 });
  }
  addRectBox(C, wallRect(C, wall, s, fw * 0.9, 0.05, stoneD * 0.85), 0.1, 0.62, 'fire', 0.01);

  // fire + embers + the light the room is lit by
  const fc = at(0, stoneD * 0.5);
  addEmber(C, fc[0], 0.13, fc[1], fw * 0.72);
  const flames = 4;
  for (let i = 0; i < flames; i++) {
    const q = at(rng.sym(half * 0.5), stoneD * 0.5 + rng.sym(0.08));
    addFlame(C, q[0], 0.2 + rng.range(0, 0.05), q[1], rng.range(0.3, 0.46), 'fire');
  }
  addAnchor(C, fc[0], 0.42, fc[1], 'fire', 0xff9440, opts.intensity ?? 1.0);

  claim(C, wallRect(C, wall, s, hsW, 0, hsD + 0.2));
  return { wall, s, width: fw, front: hsD, hearthTop: 0.1, jh, wi };
}

/** Cooking gear in the hearth: crane, cauldron, trivet, kettle. */
function kitRange(C, h) {
  const M = C.mat, B = C.bin, wi = h.wi;
  const at = (ss, t) => wallPoint(wi, h.s + ss, t, [0, 0]);
  // crane: an upright in the jamb with a swinging arm
  const u = at(-(h.width / 2 - 0.02), 0.2);
  B.add('iron', M.iron, gCyl(0.022, 0.022, h.jh * 0.82, 7), trs(u[0], h.jh * 0.41, u[1]), { uvScale: 6 });
  const a = at(-h.width * 0.12, 0.34);
  B.add('iron', M.iron, gBox(h.width * 0.7, 0.03, 0.03), trs(a[0], h.jh * 0.78, a[1], 0, wi.yaw, 0), { uvScale: 6 });
  // pot hook + cauldron
  const c = at(0.06, 0.36);
  B.add('iron', M.iron, gCyl(0.008, 0.008, 0.2, 5), trs(c[0], h.jh * 0.66, c[1]), { uvScale: 8 });
  const cauldron = gLathe('cauldron', [
    [0.0, 0], [0.09, 0], [0.17, 0.06], [0.19, 0.17], [0.165, 0.23],
    [0.172, 0.245], [0.155, 0.245], [0.15, 0.2], [0.155, 0.07], [0.0, 0.05],
  ], C.seg);
  B.add('iron', M.iron, cauldron, trs(c[0], h.jh * 0.36, c[1]), { uvScale: 2.6 });
  B.add('iron', M.iron, gTorus(0.16, 0.008, 4, 8, Math.PI), trs(c[0], h.jh * 0.6, c[1], 0, 0, 0), { uvScale: 6 });
  // trivet + kettle on the hearthstone
  const t = at(h.width * 0.42, h.front * 0.72);
  B.add('iron', M.iron, gCyl(0.13, 0.13, 0.02, 8, true), trs(t[0], 0.28, t[1]), { uvScale: 5 });
  for (let i = 0; i < 3; i++) {
    B.add('iron', M.iron, gCyl(0.012, 0.012, 0.18, 5),
      trs(t[0] + Math.cos(i * 2.094) * 0.1, 0.19, t[1] + Math.sin(i * 2.094) * 0.1), { uvScale: 8 });
  }
  B.add('iron', M.iron, gSph(0.1, 8, 6), trs(t[0], 0.37, t[1], 0, 0, 0, 1, 0.85, 1), { uvScale: 3 });
}

/* --- beds and bedroom furniture ------------------------------------------ */

/** Rope bed: turned posts, side rails, straw mattress, linen, bolster. */
function kitBed(C, x, z, yaw) {
  const M = C.mat, B = C.bin;
  const L = 1.95, Wd = 1.06;
  const rail = 0.4, mat = 0.15;
  const co = Math.cos(yaw), si = Math.sin(yaw);
  const corner = (sx, sz) => [
    x + co * sx * (L / 2 - 0.05) + si * sz * (Wd / 2 - 0.05),
    z - si * sx * (L / 2 - 0.05) + co * sz * (Wd / 2 - 0.05),
  ];
  for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
    const p = corner(sx, sz);
    const ph = sx < 0 ? 0.82 : 0.6;   // taller at the head
    B.add('wood', M.wood, gBox(0.09, ph, 0.09), trs(p[0], ph / 2, p[1], 0, yaw, 0), { uvScale: 4, cast: true });
    B.add('wood', M.wood, gSph(0.05, 7, 5), trs(p[0], ph + 0.03, p[1]), { uvScale: 6 });
  }
  // rails
  B.add('wood', M.wood, gBox(L, 0.11, 0.06), trs(x + si * (Wd / 2 - 0.05), rail, z + co * (Wd / 2 - 0.05), 0, yaw, 0), { uvScale: 2 });
  B.add('wood', M.wood, gBox(L, 0.11, 0.06), trs(x - si * (Wd / 2 - 0.05), rail, z - co * (Wd / 2 - 0.05), 0, yaw, 0), { uvScale: 2 });
  B.add('wood', M.wood, gBox(0.06, 0.11, Wd), trs(x + co * (L / 2 - 0.05), rail, z - si * (L / 2 - 0.05), 0, yaw, 0), { uvScale: 2 });
  B.add('wood', M.wood, gBox(0.06, 0.11, Wd), trs(x - co * (L / 2 - 0.05), rail, z + si * (L / 2 - 0.05), 0, yaw, 0), { uvScale: 2 });
  // headboard
  B.add('plank', M.plank, gBox(0.05, 0.42, Wd - 0.06),
    trs(x - co * (L / 2 - 0.05), 0.58, z + si * (L / 2 - 0.05), 0, yaw, 0), { uvScale: 1.8, cast: true });
  // straw mattress, then the sheet over it
  B.add('sack', M.sack, gBox(L - 0.14, mat, Wd - 0.14), trs(x, rail + mat / 2, z, 0, yaw, 0), { uvScale: 2 });
  B.add('linen', M.linen, gBox(L - 0.2, 0.05, Wd - 0.1), trs(x, rail + mat + 0.02, z, 0, yaw, 0), { uvScale: 1.4 });
  // bolster at the head
  B.add('linen', M.linen, gCyl(0.115, 0.115, Wd - 0.18, 8),
    trs(x - co * (L / 2 - 0.28), rail + mat + 0.1, z + si * (L / 2 - 0.28), 0, yaw, Math.PI / 2), { uvScale: 2 });
  // a blanket folded over the foot
  B.add('rug', M.blanket, gBox(0.5, 0.07, Wd - 0.06), trs(x + co * (L / 2 - 0.36), rail + mat + 0.06, z - si * (L / 2 - 0.36), 0, yaw, 0), { uvScale: 1.6 });
  const rect = rectAt(x, z, L, Wd, yaw);
  addRectBox(C, rect, 0, rail + mat + 0.04, 'bed', 0.03);
  return rect;
}

/** Washstand with a basin and a jug. */
function kitWashstand(C, wall, prefer) {
  const M = C.mat, B = C.bin;
  const len = 0.72;
  const spot = findOnWall(C, wall, len, 0, 0.44, prefer, false);
  if (!spot) return null;
  const wi = spot.wi, h = 0.78;
  const c = wallPoint(wi, spot.s, 0.22, [0, 0]);
  B.add('plank', M.plank, gBox(len, 0.05, 0.44), trs(c[0], h, c[1], 0, wi.yaw, 0), { uvScale: 2, cast: true });
  for (const sx of [-1, 1]) for (const sz of [0.18, 0.72]) {
    const q = wallPoint(wi, spot.s + sx * (len / 2 - 0.07), 0.44 * sz, [0, 0]);
    B.add('wood', M.wood, gBox(0.055, h, 0.055), trs(q[0], h / 2, q[1], 0, wi.yaw, 0), { uvScale: 4 });
  }
  B.add('wood', M.wood, gBox(len - 0.14, 0.04, 0.36), trs(c[0], 0.22, c[1], 0, wi.yaw, 0), { uvScale: 2 });
  B.add('pewter', M.pewter, bowlGeo(C.seg), trs(c[0], h + 0.025, c[1]), { uvScale: 3 });
  const j = wallPoint(wi, spot.s + len * 0.32, 0.14, [0, 0]);
  B.add('crock', M.crock, jugGeo(C.seg), trs(j[0], 0.22 + 0.04, j[1], 0, C.rng.range(0, TAU), 0), { uvScale: 3 });
  addRectBox(C, spot.rect, 0, h + 0.02, 'washstand', 0.02);
  claim(C, spot.rect);
  return spot;
}

/* --- trade fittings ------------------------------------------------------- */

/** Timber stalls along a wall, with a hay rack in each. */
function kitStalls(C, wall, count) {
  const M = C.mat, B = C.bin;
  const wi = wallInfo(C, wall);
  const usable = wi.span * 2 - 0.6;
  const w = Math.min(2.7, usable / count);
  const depth = Math.min(2.9, (wall === 'back' || wall === 'front' ? C.D : C.W) * 0.42);
  const postH = 1.9, panelH = 1.15;
  for (let i = 0; i <= count; i++) {
    const s = (i - count / 2) * w;
    // partition: two posts and a boarded panel between them
    for (const t of [0.12, depth - 0.1]) {
      const p = wallPoint(wi, s, t, [0, 0]);
      B.add('wood', M.wood, gBox(0.13, postH, 0.13), trs(p[0], postH / 2, p[1], 0, wi.yaw, 0), { uvScale: 3, cast: true });
    }
    const c = wallPoint(wi, s, depth / 2, [0, 0]);
    B.add('plank', M.plank, gBox(0.07, panelH - 0.1, depth - 0.24), trs(c[0], (panelH - 0.1) / 2, c[1], 0, wi.yaw, 0), { uvScale: 1.5, cast: true });
    B.add('wood', M.wood, gBox(0.09, 0.1, depth - 0.2), trs(c[0], panelH, c[1], 0, wi.yaw, 0), { uvScale: 2 });
    addRectBox(C, wallRect(C, wall, s, 0.14, 0.1, depth), 0, postH, 'stall-post', 0.0);
  }
  // hay racks and litter, one per stall
  for (let i = 0; i < count; i++) {
    const s = (i - count / 2 + 0.5) * w;
    const p = wallPoint(wi, s, 0.24, [0, 0]);
    for (let k = 0; k < 6; k++) {
      const q = wallPoint(wi, s + (k - 2.5) * 0.16, 0.3, [0, 0]);
      B.add('wood', M.wood, gCyl(0.018, 0.018, 0.55, 5), trs(q[0], 1.12, q[1], 0.5, wi.yaw, 0), { uvScale: 6 });
    }
    B.add('wood', M.wood, gBox(w * 0.72, 0.07, 0.1), trs(p[0], 0.86, p[1], 0, wi.yaw, 0), { uvScale: 3 });
    const hay = wallPoint(wi, s, 0.42, [0, 0]);
    B.add('straw', M.straw, gSph(0.3, 8, 6), trs(hay[0], 1.16, hay[1], 0, 0, 0, 1.5, 0.5, 0.7), { uvScale: 2.4 });
    const lit = wallPoint(wi, s, depth * 0.62, [0, 0]);
    B.add('straw', M.straw, gPlane(w * 0.9, depth * 0.6), trs(lit[0], 0.016, lit[1], -Math.PI / 2, 0, wi.yaw), { uvScale: 1.6 });
    claim(C, wallRect(C, wall, s, w * 0.9, 0.1, depth - 0.2));
  }
  return { depth, w };
}

/** A domed brick bread oven in the chimney breast. */
function kitOven(C, wall, s) {
  const M = C.mat, B = C.bin, wi = wallInfo(C, wall);
  const at = (ss, t) => wallPoint(wi, s + ss, t, [0, 0]);
  const plinth = 0.72, r = 1.02;
  const c = at(0, r * 0.72);
  B.add('hearth', M.hearth, gBox(r * 2.1, plinth, r * 1.45), trs(c[0], plinth / 2, c[1], 0, wi.yaw, 0), { uvScale: 1.2, cast: true });
  // dome
  const dome = cached('ovenDome', () => {
    const g = new THREE.SphereGeometry(1, 14, 8, 0, TAU, 0, Math.PI / 2);
    return g;
  });
  B.add('brick', M.brick, dome, trs(c[0], plinth, c[1], 0, wi.yaw, 0, r, r * 0.78, r * 0.72), { uvScale: 1.3, cast: true });
  // arched mouth: a sooty recess with a brick surround
  const m = at(0, r * 0.05);
  B.add('soot', M.soot, gBox(0.62, 0.44, 0.5), trs(m[0] + wi.ix * 0.28, plinth + 0.24, m[1] + wi.iz * 0.28, 0, wi.yaw, 0), { uvScale: 2 });
  const f = at(0, 0.02);
  for (const sx of [-1, 1]) {
    B.add('brick', M.brick, gBox(0.2, 0.5, 0.12), trs(f[0] + wi.ax * sx * 0.41, plinth + 0.25, f[1] + wi.az * sx * 0.41, 0, wi.yaw, 0), { uvScale: 3 });
  }
  B.add('brick', M.brick, gBox(1.02, 0.14, 0.14), trs(f[0], plinth + 0.55, f[1], 0, wi.yaw, 0), { uvScale: 3 });
  // fire inside the mouth
  addEmber(C, m[0] + wi.ix * 0.3, plinth + 0.04, m[1] + wi.iz * 0.3, 0.5);
  for (let i = 0; i < 3; i++) {
    addFlame(C, m[0] + wi.ix * 0.32 + wi.ax * C.rng.sym(0.16), plinth + 0.07,
      m[1] + wi.iz * 0.32 + wi.az * C.rng.sym(0.16), C.rng.range(0.16, 0.26), 'fire');
  }
  addAnchor(C, m[0] + wi.ix * 0.2, plinth + 0.3, m[1] + wi.iz * 0.2, 'fire', 0xff8c34, 1.1);
  // iron door leaning against the plinth
  const d = at(r * 0.9, 0.16);
  B.add('iron', M.iron, gBox(0.58, 0.5, 0.04), trs(d[0], 0.3, d[1], 0.18, wi.yaw, 0), { uvScale: 3 });
  const rect = wallRect(C, wall, s, r * 2.1, 0, r * 1.5);
  addRectBox(C, rect, 0, plinth + r * 0.78, 'oven', 0.02);
  claim(C, wallRect(C, wall, s, r * 2.1 + 0.3, 0, r * 1.5 + 0.35));
  return { s, wall, plinth, r };
}

/** Heavy workbench with a vice and a tool board over it. */
function kitWorkbench(C, wall, prefer) {
  const M = C.mat, B = C.bin;
  const len = clamp(C.W * 0.34, 1.5, 2.2), h = 0.86, d = 0.68;
  const spot = findOnWall(C, wall, len, 0.05, 0.05 + d, prefer, false);
  if (!spot) return null;
  const wi = spot.wi;
  const c = wallPoint(wi, spot.s, 0.05 + d / 2, [0, 0]);
  B.add('beam', M.beam, gBox(len, 0.1, d), trs(c[0], h - 0.05, c[1], 0, wi.yaw, 0), { uvScale: 1.3, cast: true });
  for (const sx of [-1, 1]) for (const t of [0.18, 0.78]) {
    const q = wallPoint(wi, spot.s + sx * (len / 2 - 0.12), 0.05 + d * t, [0, 0]);
    B.add('wood', M.wood, gBox(0.12, h - 0.1, 0.12), trs(q[0], (h - 0.1) / 2, q[1], 0, wi.yaw, 0), { uvScale: 3, cast: true });
  }
  B.add('wood', M.wood, gBox(len - 0.3, 0.09, 0.06), trs(c[0], 0.24, c[1], 0, wi.yaw, 0), { uvScale: 3 });
  // vice at one end
  const v = wallPoint(wi, spot.s - (len / 2 - 0.26), 0.05 + d * 0.9, [0, 0]);
  B.add('wood', M.wood, gBox(0.3, 0.22, 0.08), trs(v[0], h - 0.2, v[1], 0, wi.yaw, 0), { uvScale: 3 });
  B.add('iron', M.iron, gCyl(0.022, 0.022, 0.34, 7), trs(v[0], h - 0.2, v[1], Math.PI / 2, wi.yaw, Math.PI / 2), { uvScale: 6 });
  B.add('wood', M.wood, gCyl(0.028, 0.028, 0.24, 6), trs(v[0], h - 0.2, v[1] + wi.iz * 0.2 + 0.0, 0, wi.yaw, Math.PI / 2), { uvScale: 6 });
  // tool board
  const tb = wallPoint(wi, spot.s, 0.03, [0, 0]);
  const boardY = Math.min(1.75, C.H - 0.5);
  B.add('plank', M.plank, gBox(len * 0.86, 0.82, 0.03), trs(tb[0], boardY - 0.2, tb[1], 0, wi.yaw, 0), { uvScale: 1.4 });
  const rng = C.rng;
  for (let i = 0; i < 7; i++) {
    const off = (i - 3) * (len * 0.86 / 8);
    const q = wallPoint(wi, spot.s + off, 0.08, [0, 0]);
    const kind = i % 3;
    if (kind === 0) {   // saw
      B.add('iron', M.iron, gBox(0.09, 0.34, 0.008), trs(q[0], boardY - 0.32, q[1], 0, wi.yaw, 0.1), { uvScale: 5 });
      B.add('wood', M.wood, gBox(0.07, 0.11, 0.03), trs(q[0], boardY - 0.1, q[1], 0, wi.yaw, 0), { uvScale: 6 });
    } else if (kind === 1) {  // chisels
      B.add('iron', M.iron, gCyl(0.009, 0.009, 0.16, 5), trs(q[0], boardY - 0.4, q[1], 0, wi.yaw, 0), { uvScale: 8 });
      B.add('wood', M.wood, gCyl(0.017, 0.013, 0.12, 6), trs(q[0], boardY - 0.26, q[1], 0, wi.yaw, 0), { uvScale: 8 });
    } else {            // mallet on a peg
      B.add('wood', M.wood, gCyl(0.045, 0.045, 0.14, 7), trs(q[0], boardY - 0.28, q[1], 0, wi.yaw, Math.PI / 2), { uvScale: 5 });
      B.add('wood', M.wood, gCyl(0.016, 0.016, 0.24, 5), trs(q[0], boardY - 0.44, q[1], 0, wi.yaw, 0), { uvScale: 8 });
    }
    if (rng.bool(0.2)) B.add('iron', M.iron, gCyl(0.012, 0.012, 0.05, 5), trs(q[0], boardY - 0.16, q[1], Math.PI / 2, wi.yaw, 0), { uvScale: 8 });
  }
  addRectBox(C, spot.rect, 0, h + 0.02, 'workbench', 0.02);
  claim(C, spot.rect);
  return { spot, top: h };
}

/** Wood shavings and offcuts scattered on the floor. */
function kitShavings(C, x, z, r, n) {
  const M = C.mat, B = C.bin, rng = C.rng;
  for (let i = 0; i < n; i++) {
    const a = rng.range(0, TAU), d = r * Math.sqrt(rng.next());
    const px = x + Math.cos(a) * d, pz = z + Math.sin(a) * d;
    const rect = rectAt(px, pz, 0.2, 0.2);
    if (!insideRoom(C, rect, 0.1)) continue;
    B.add('plank', M.plank, gBox(rng.range(0.06, 0.16), 0.012, rng.range(0.02, 0.05)),
      trs(px, 0.008, pz, 0, rng.range(0, TAU), 0), { uvScale: 5 });
  }
}

/** Beam-hung balance: a yoke, two chains, two pans. */
function kitBalance(C, x, z, y) {
  const M = C.mat, B = C.bin;
  B.add('iron', M.iron, gBox(0.5, 0.02, 0.02), trs(x, y - 0.3, z), { uvScale: 6 });
  B.add('iron', M.iron, gCyl(0.008, 0.008, 0.3, 5), trs(x, y - 0.15, z), { uvScale: 8 });
  for (const sx of [-1, 1]) {
    B.add('iron', M.iron, gCyl(0.006, 0.006, 0.22, 4), trs(x + sx * 0.24, y - 0.41, z), { uvScale: 8 });
    B.add('pewter', M.pewter, plateGeo(C.seg), trs(x + sx * 0.24, y - 0.52, z, 0, 0, 0, 1.3, 1, 1.3), { uvScale: 4 });
  }
}

/** Shop scales on a counter top. */
function kitScales(C, x, y, z, yaw) {
  const M = C.mat, B = C.bin;
  B.add('wood', M.wood, gBox(0.3, 0.03, 0.18), trs(x, y + 0.015, z, 0, yaw, 0), { uvScale: 4 });
  B.add('iron', M.iron, gCyl(0.014, 0.02, 0.34, 6), trs(x, y + 0.18, z), { uvScale: 6 });
  B.add('iron', M.iron, gBox(0.34, 0.014, 0.014), trs(x, y + 0.35, z, 0, yaw, 0), { uvScale: 6 });
  for (const sx of [-1, 1]) {
    B.add('iron', M.iron, gCyl(0.004, 0.004, 0.1, 4), trs(x + Math.cos(yaw) * sx * 0.16, y + 0.3, z - Math.sin(yaw) * sx * 0.16), { uvScale: 8 });
    B.add('pewter', M.pewter, plateGeo(C.seg), trs(x + Math.cos(yaw) * sx * 0.16, y + 0.25, z - Math.sin(yaw) * sx * 0.16, 0, 0, 0, 0.8, 1, 0.8), { uvScale: 5 });
  }
}

/**
 * Density pass. The plots in `layout.js` are big — the Moot Hall's ground floor
 * is 15.3 x 10.3 m of clear floor — so a recipe's authored pieces leave a barn
 * of empty boards behind them. This tops the room up with stock parts drawn
 * ONLY from materials the room already uses, so extra clutter costs triangles
 * (cheap: the worst room is 6 k) and not draw calls (the constraint).
 */
function kitFiller(C) {
  const area = C.W * C.D;
  let n = clamp(Math.round((area - 38) / 11), 0, 9);
  if (C.room.openToRoof) n = Math.min(n, 3);   // low eaves, less usable floor
  const rng = C.rng, M = C.mat, B = C.bin;
  const kinds = rng.shuffle([
    'barrel', 'crate', 'sacks', 'chest', 'shelf', 'firewood', 'basket', 'stool',
  ]);
  for (let i = 0; i < n; i++) {
    const kind = kinds[i % kinds.length];
    const wall = WALLS[rng.int(0, 3)];
    switch (kind) {
      case 'barrel': {
        const sp = findOnWall(C, wall, 0.78, 0.06, 0.84, rng.sym(C.hw * 0.6), false);
        if (!sp) break;
        kitBarrel(C, sp.x, 0, sp.z, rng.range(0, TAU), false, rng.range(0.86, 1.04));
        claim(C, sp.rect);
        break;
      }
      case 'crate': {
        const sp = findOnWall(C, wall, 0.62, 0.05, 0.67, rng.sym(C.hw * 0.6), false);
        if (!sp) break;
        kitCrate(C, sp.x, 0, sp.z, rng.sym(0.35), 0.56);
        if (rng.bool(0.55) && !C.room.openToRoof) kitCrate(C, sp.x + rng.sym(0.05), 0.56, sp.z + rng.sym(0.05), rng.sym(0.5), 0.42);
        claim(C, sp.rect);
        break;
      }
      case 'sacks': {
        const sp = findOnWall(C, wall, 1.3, 0.05, 0.66, rng.sym(C.hw * 0.6), false);
        if (!sp) break;
        for (const sx of [-0.32, 0.32]) {
          kitSack(C, alongX(sp.x, sp.yaw, sx), 0, alongXz(sp.z, sp.yaw, sx), rng.range(0, TAU), rng.range(0.88, 1.02));
        }
        addRectBox(C, sp.rect, 0, 0.6, 'sack', 0.05);
        claim(C, sp.rect);
        break;
      }
      case 'chest': {
        const sp = findOnWall(C, wall, 0.9, 0, 0.48, rng.sym(C.hw * 0.6), true);
        if (!sp) break;
        claim(C, kitChest(C, sp.x, sp.z, sp.yaw, 0.9));
        break;
      }
      case 'shelf': {
        const len = clamp(C.W * 0.24, 0.9, 1.6);
        const sp = findOnWall(C, wall, len, 0, 0.28, rng.sym(C.hw * 0.55), true);
        if (!sp) break;
        kitShelves(C, wall, sp.s, len, rng.int(1, 2), 1.16 + rng.sym(0.1), 0.44, stockCrockery);
        break;
      }
      case 'firewood': {
        const sp = findOnWall(C, wall, 1.0, 0.05, 0.5, rng.sym(C.hw * 0.6), false);
        if (!sp) break;
        for (let r = 0; r < 3; r++) {
          for (let c2 = 0; c2 < 4 - r; c2++) {
            const off = (c2 - (3 - r) / 2) * 0.17;
            const rr = rng.range(0.055, 0.078);
            B.add('wood', M.wood, gCyl(rr, rr * 0.94, 0.46, 6),
              trs(alongX(sp.x, sp.yaw, off), 0.07 + r * 0.15, alongXz(sp.z, sp.yaw, off),
                0, sp.yaw + rng.sym(0.12), Math.PI / 2), { uvScale: 3 });
          }
        }
        addRectBox(C, sp.rect, 0, 0.52, 'firewood', 0.04);
        claim(C, sp.rect);
        break;
      }
      case 'basket': {
        const sp = findOnWall(C, wall, 0.6, 0.05, 0.62, rng.sym(C.hw * 0.6), false);
        if (!sp) break;
        B.add('plank', M.plank, gLathe('basket', [
          [0.0, 0], [0.2, 0], [0.23, 0.05], [0.27, 0.3], [0.29, 0.34],
          [0.265, 0.345], [0.25, 0.31], [0.205, 0.06], [0.0, 0.05],
        ], C.seg), trs(sp.x, 0, sp.z, 0, rng.range(0, TAU), 0), { uvScale: 3, cast: true });
        addCyl(C, sp.x, 0.17, sp.z, 0.28, 0.17, 'basket');
        claim(C, sp.rect);
        break;
      }
      default: {
        const sp = findFree(C, rng.sym(C.hw * 0.5), rng.sym(C.hd * 0.5), 0.42, 0.42, 0, 2.2);
        if (!sp) break;
        // a joint stool: three splayed legs, a round top
        B.add('plank', M.plank, gCyl(0.17, 0.175, 0.05, 10), trs(sp.x, 0.42, sp.z), { uvScale: 3, cast: true });
        for (let k = 0; k < 3; k++) {
          const a = k * 2.094 + rng.next();
          B.add('wood', M.wood, gCyl(0.023, 0.029, 0.42, 6),
            trs(sp.x + Math.cos(a) * 0.1, 0.21, sp.z + Math.sin(a) * 0.1,
              Math.sin(a) * 0.18, 0, -Math.cos(a) * 0.18), { uvScale: 5 });
        }
        addCyl(C, sp.x, 0.22, sp.z, 0.19, 0.22, 'stool');
        claim(C, sp.rect);
        break;
      }
    }
  }
  return n;
}

/* -------------------------------------------------------------------------- */
/* ROOM RECIPES                                                               */
/* -------------------------------------------------------------------------- */

/** Where does this room's hearth go? Follow the plot's chimney if there is one. */
function hearthSite(C) {
  const plot = C.plot;
  let cx = 0, cz = -C.hd;
  if (plot && plot.chimneys && plot.chimneys.length) {
    // the biggest stack is the main fireplace
    let best = plot.chimneys[0];
    for (const ch of plot.chimneys) if ((ch.width || 1) * ch.height > (best.width || 1) * best.height) best = ch;
    plotToWorld(plot, best.x, 0, best.z, _v);
    toLocal(C.room, _v.x, _v.z, _loc);
    cx = _loc[0]; cz = _loc[1];
  }
  // nearest wall to the stack
  let wall = 'back', d = cz + C.hd;
  if (C.hd - cz < d) { wall = 'front'; d = C.hd - cz; }
  if (cx + C.hw < d) { wall = 'left'; d = cx + C.hw; }
  if (C.hw - cx < d) { wall = 'right'; d = C.hw - cx; }
  // a hearth on the front wall would fight the door — fall back to the back wall
  if (wall === 'front' && C.doors.some((dr) => dr.lz > 0)) wall = 'back';
  const wi = wallInfo(C, wall);
  const s = (cx - wi.ox) * wi.ax + (cz - wi.oz) * wi.az;
  return { wall, s };
}

/** The side wall furthest from the hearth — where a staircase most likely is. */
function farSideWall(C, hearth) {
  if (hearth && (hearth.wall === 'left' || hearth.wall === 'right')) {
    return hearth.wall === 'left' ? 'right' : 'left';
  }
  const s = hearth ? hearth.s : 0;
  return s >= 0 ? 'left' : 'right';
}

function recipeHall(C, h) {
  const far = farSideWall(C, h);
  const near = far === 'left' ? 'right' : 'left';
  const long = C.W >= C.D;
  // table in the middle of the floor, running the long way
  const len = clamp((long ? C.W : C.D) * 0.42, 1.6, 3.2);
  const tSpot = findFree(C, 0, C.hd * 0.12, len, 0.92, long ? 0 : Math.PI / 2, 2.0);
  if (tSpot) {
    const t = kitTable(C, tSpot.x, tSpot.z, tSpot.yaw, len, 0.92);
    claim(C, t.rect);
    // benches either side
    for (const sx of [-1, 1]) {
      const bx = tSpot.x + (long ? 0 : sx * 0.78), bz = tSpot.z + (long ? sx * 0.78 : 0);
      const r = rectAt(bx, bz, len * 0.85, 0.32, tSpot.yaw);
      if (freeRect(C, r) && insideRoom(C, r, 0.1)) claim(C, kitBench(C, bx, bz, tSpot.yaw, len * 0.85));
    }
    // dressing on the table
    kitCandle(C, tSpot.x, t.top, tSpot.z, true);
    C.bin.add('crock', C.mat.crock, bowlGeo(C.seg), trs(tSpot.x + 0.4, t.top, tSpot.z - 0.2, 0, 0.6, 0), { uvScale: 3 });
    addGrab(C, 'crock', tSpot.x - 0.45, t.top + 0.11, tSpot.z + 0.12, 0);
    addGrab(C, 'book', tSpot.x + 0.7, t.top + 0.03, tSpot.z + 0.2, 3);
    addGrab(C, 'apple', tSpot.x + 0.15, t.top + 0.05, tSpot.z - 0.3, 4);
  }
  // settle drawn up to the fire
  if (h) {
    const sw = h.wall === 'left' || h.wall === 'right' ? 'back' : near;
    const spot = findOnWall(C, sw, 1.5, 0, 0.5, 0, false);
    if (spot) claim(C, kitSettle(C, spot.x, spot.z, spot.yaw, 1.5));
    kitRug(C, ...(() => {
      const wi = h.wi;
      const p = wallPoint(wi, h.s, h.front + 0.75, [0, 0]);
      return [p[0], p[1], 2.1, 1.5, wi.yaw];
    })());
  }
  const chest = findOnWall(C, far, 0.95, 0, 0.5, -C.hd * 0.3, true);
  if (chest) claim(C, kitChest(C, chest.x, chest.z, chest.yaw, 0.95));
  const sh = findOnWall(C, near, clamp(C.W * 0.3, 1.0, 1.6), 0, 0.28, 0, true);
  if (sh) kitShelves(C, near, sh.s, clamp(C.W * 0.3, 1.0, 1.6), 2, 1.24, 0.42, stockCrockery);
  kitPegs(C, 'front', C.doors.length ? C.doors[0].lx + 1.2 : 0);
  if (C.H > 2.2) kitHerbs(C, 0, -C.hd * 0.45, C.H - 0.18, 3);
  addGrab(C, 'stool', C.hw * 0.4, 0, -C.hd * 0.2, 2);
  addGrab(C, 'tankard', 0, 0.78, C.hd * 0.2, 1);
}

function recipeKitchen(C, h) {
  if (h) kitRange(C, h);
  const far = farSideWall(C, h);
  const near = far === 'left' ? 'right' : 'left';
  const dr = kitDresser(C, far, 0);
  const len = clamp(C.W * 0.3, 1.2, 1.8);
  const tSpot = findFree(C, 0, C.hd * 0.18, len, 0.8, C.W >= C.D ? 0 : Math.PI / 2, 2.0);
  if (tSpot) {
    const t = kitTable(C, tSpot.x, tSpot.z, tSpot.yaw, len, 0.8);
    claim(C, t.rect);
    C.bin.add('crock', C.mat.crock, jugGeo(C.seg), trs(tSpot.x - 0.3, t.top, tSpot.z + 0.1, 0, 1.1, 0), { uvScale: 3 });
    C.bin.add('plank', C.mat.plank, gBox(0.34, 0.03, 0.22), trs(tSpot.x + 0.25, t.top + 0.015, tSpot.z - 0.08, 0, 0.2, 0), { uvScale: 3 });
    addGrab(C, 'loaf', tSpot.x + 0.25, t.top + 0.05, tSpot.z - 0.08, 0);
    addGrab(C, 'crock', tSpot.x + 0.05, t.top + 0.11, tSpot.z + 0.24, 1);
    addGrab(C, 'apple', tSpot.x - 0.1, t.top + 0.05, tSpot.z - 0.22, 4);
  }
  // churn and pail by the wall
  const ch = findOnWall(C, near, 0.44, 0, 0.44, -C.hd * 0.25, false);
  if (ch) {
    C.bin.add('wood', C.mat.wood, gCyl(0.17, 0.21, 0.86, 9), trs(ch.x, 0.43, ch.z, 0, 0, 0), { uvScale: 2.4, cast: true });
    C.bin.add('wood', C.mat.wood, gCyl(0.02, 0.02, 0.5, 6), trs(ch.x, 1.02, ch.z), { uvScale: 6 });
    C.bin.add('iron', C.mat.iron, gCyl(0.215, 0.215, 0.04, 9, true), trs(ch.x, 0.62, ch.z), { uvScale: 5 });
    addCyl(C, ch.x, 0.43, ch.z, 0.22, 0.43, 'churn');
    claim(C, ch.rect);
  }
  if (C.H > 2.2) kitHerbs(C, C.hw * 0.2, -C.hd * 0.3, C.H - 0.18, 4);
  const sh = findOnWall(C, near, clamp(C.W * 0.26, 0.9, 1.4), 0, 0.28, C.hd * 0.3, true);
  if (sh) kitShelves(C, near, sh.s, clamp(C.W * 0.26, 0.9, 1.4), 2, 1.3, 0.4, stockCrockery);
  addGrab(C, 'pail', -C.hw * 0.5, 0, C.hd * 0.35, 2);
  addGrab(C, 'bottle', dr ? 0 : 0.3, dr ? dr.base + 0.02 : 0.78, dr ? -C.hd + 0.24 : 0, 3);
}

function recipeShop(C, h) {
  const trade = C.plot ? C.plot.id : 'shop';
  const far = farSideWall(C, h);
  const near = far === 'left' ? 'right' : 'left';
  // counter across the room, leaving at least 1.1 m to get past it
  const front = C.doors.length ? C.doors[0] : null;
  const across = C.W >= C.D;
  const span = across ? C.W : C.D;
  const len = clamp(span - 1.6, 1.4, span * 0.72);
  const side = front && front.lx > 0 ? -1 : 1;
  const cx = across ? side * (span - len) * 0.42 : 0;
  const spot = findFree(C, across ? cx : 0, across ? -C.hd * 0.05 : side * (span - len) * 0.42,
    across ? len : 0.68, across ? 0.68 : len, 0, 1.6);
  let top = 0.9;
  if (spot) {
    const c = kitCounter(C, spot.x, spot.z, across ? 0 : Math.PI / 2, len);
    claim(C, c.rect);
    top = c.top;
    kitScales(C, spot.x + (across ? len * 0.3 : 0), top, spot.z + (across ? 0 : len * 0.3), across ? 0 : Math.PI / 2);
    addGrab(C, 'book', spot.x - (across ? len * 0.3 : 0), top + 0.03, spot.z - (across ? 0 : len * 0.3), 3);
  }
  // stock on the wall behind
  const sw = across ? 'back' : near;
  const shelfLen = clamp(span * 0.5, 1.2, 2.4);
  const sh = findOnWall(C, sw, shelfLen, 0, 0.28, h ? -h.s * 0.8 : 0, true);
  if (sh) kitShelves(C, sw, sh.s, shelfLen, 3, 1.02, 0.42, stockFor(trade));
  const dr = kitDresser(C, far, C.hd * 0.2);
  // crates of goods near the door
  const cr = findFree(C, front ? front.lx * 0.6 : 0, C.hd * 0.55, 0.6, 0.6, 0, 1.4);
  if (cr) {
    kitCrate(C, cr.x, 0, cr.z, C.rng.sym(0.4), 0.56);
    claim(C, cr.rect);
    if (C.rng.bool(0.7)) kitCrate(C, cr.x + 0.06, 0.56, cr.z - 0.04, C.rng.sym(0.5), 0.44);
  }
  addGrab(C, 'stool', spot ? spot.x - 0.3 : 0, 0, spot ? spot.z - 0.85 : -C.hd * 0.4, 2);
  addGrab(C, 'bottle', spot ? spot.x : 0, top + 0.02, spot ? spot.z + 0.14 : 0, 0);
  addGrab(C, 'crock', spot ? spot.x + 0.5 : 0.5, top + 0.11, spot ? spot.z - 0.1 : 0, 1);
  addGrab(C, 'candlestick', dr ? 0 : -0.5, dr ? dr.base + 0.02 : top + 0.02, dr ? -C.hd + 0.24 : 0, 4);
}

/** Shelf stock varies by trade — candles, physick jars, or leather. */
function stockFor(trade) {
  if (trade === 'chandler') {
    return (C, wall, s, len, y) => {
      const wi = wallInfo(C, wall), B = C.bin, M = C.mat, rng = C.rng;
      const n = Math.max(3, Math.floor(len / 0.18));
      for (let i = 0; i < n; i++) {
        if (rng.bool(0.18)) continue;
        const p = wallPoint(wi, s + (i - (n - 1) / 2) * (len / n), 0.12 + rng.sym(0.03), [0, 0]);
        const hh = rng.range(0.14, 0.26);
        B.add('wax', M.wax, gCyl(0.017, 0.019, hh, 6), trs(p[0], y + hh / 2, p[1]), { uvScale: 6 });
      }
    };
  }
  if (trade === 'apothecary') {
    return (C, wall, s, len, y, tier) => {
      const wi = wallInfo(C, wall), B = C.bin, M = C.mat, rng = C.rng;
      const n = Math.max(3, Math.floor(len / 0.22));
      for (let i = 0; i < n; i++) {
        if (rng.bool(0.15)) continue;
        const p = wallPoint(wi, s + (i - (n - 1) / 2) * (len / n), 0.12 + rng.sym(0.02), [0, 0]);
        if (tier === 1 && rng.bool(0.5)) {
          B.add('glass', M.glass, bottleGeo(C.seg), trs(p[0], y, p[1], 0, rng.range(0, TAU), 0), { uvScale: 4 });
        } else {
          B.add('crock', M.crock, crockGeo(C.seg), trs(p[0], y, p[1], 0, rng.range(0, TAU), 0, 0.9, 0.9, 0.9), { uvScale: 4 });
        }
      }
    };
  }
  if (trade === 'saddlery') {
    return (C, wall, s, len, y) => {
      const wi = wallInfo(C, wall), B = C.bin, M = C.mat, rng = C.rng;
      const n = Math.max(2, Math.floor(len / 0.42));
      for (let i = 0; i < n; i++) {
        const p = wallPoint(wi, s + (i - (n - 1) / 2) * (len / n), 0.13, [0, 0]);
        B.add('wood', M.wood, gCyl(0.09, 0.09, 0.34, 8), trs(p[0], y + 0.09, p[1], 0, wi.yaw, Math.PI / 2), { uvScale: 3 });
        if (rng.bool(0.5)) {
          B.add('rope', M.rope, gTorus(0.09, 0.012, 4, 8), trs(p[0], y + 0.1, p[1] + 0.01, Math.PI / 2, wi.yaw, 0), { uvScale: 5 });
        }
      }
    };
  }
  return stockCrockery;
}

function recipeTaproom(C, h) {
  const far = farSideWall(C, h);
  const across = C.W >= C.D;
  const span = across ? C.W : C.D;
  // bar along the wall away from the fire
  const barLen = clamp(span * 0.34, 2.0, 5.0);
  const bar = findOnWall(C, 'back', barLen, 0.35, 0.35 + 0.66, h ? -h.s * 1.4 : span * 0.25, true);
  let barTop = 0.95;
  if (bar) {
    const c = kitCounter(C, bar.x, bar.z, bar.yaw, barLen, 0.95, 0.66);
    claim(C, c.rect);
    barTop = c.top;
    // casks on stillage behind the bar
    const wi = bar.wi;
    const st = wallPoint(wi, bar.s, 0.14, [0, 0]);
    C.bin.add('wood', C.mat.wood, gBox(barLen * 0.82, 0.12, 0.34), trs(st[0], 0.32, st[1], 0, wi.yaw, 0), { uvScale: 2 });
    const casks = Math.max(2, Math.floor(barLen / 1.15));
    for (let i = 0; i < casks; i++) {
      const p = wallPoint(wi, bar.s + (i - (casks - 1) / 2) * (barLen / casks), 0.32, [0, 0]);
      kitBarrel(C, p[0], 0.38, p[1], wi.yaw + Math.PI / 2, true, 0.82);
    }
    // tankards on the bar
    for (let i = 0; i < 4; i++) {
      const p = wallPoint(wi, bar.s + (i - 1.5) * 0.32, 0.78, [0, 0]);
      C.bin.add('pewter', C.mat.pewter, tankardGeo(C.seg), trs(p[0], barTop, p[1], 0, C.rng.range(0, TAU), 0), { uv: 'keep' });
    }
    kitCandle(C, wallPoint(wi, bar.s - barLen * 0.42, 0.5, [0, 0])[0], barTop,
      wallPoint(wi, bar.s - barLen * 0.42, 0.5, [0, 0])[1], true);
    addGrab(C, 'tankard', bar.x, barTop + 0.07, bar.z + (across ? 0.2 : 0), 0);
    addGrab(C, 'bottle', bar.x + (across ? 0.5 : 0), barTop + 0.14, bar.z + (across ? 0.12 : 0.5), 2);
  }
  // two long tables with benches, in the body of the room
  const tLen = clamp(span * 0.3, 1.8, 3.0);
  for (let i = 0; i < 2; i++) {
    const bx = across ? (i ? 1 : -1) * span * 0.2 : 0;
    const bz = across ? C.hd * 0.3 : (i ? 1 : -1) * span * 0.2;
    const spot = findFree(C, bx, bz, tLen, 0.9, across ? 0 : Math.PI / 2, 2.2);
    if (!spot) continue;
    const t = kitTable(C, spot.x, spot.z, spot.yaw, tLen, 0.9);
    claim(C, t.rect);
    for (const sx of [-1, 1]) {
      const px = spot.x + (across ? 0 : sx * 0.76), pz = spot.z + (across ? sx * 0.76 : 0);
      const r = rectAt(px, pz, tLen * 0.82, 0.32, spot.yaw);
      if (freeRect(C, r) && insideRoom(C, r, 0.1)) claim(C, kitBench(C, px, pz, spot.yaw, tLen * 0.82));
    }
    for (let k = 0; k < 2; k++) {
      C.bin.add('pewter', C.mat.pewter, tankardGeo(C.seg),
        trs(spot.x + C.rng.sym(tLen * 0.3), t.top, spot.z + C.rng.sym(0.25), 0, C.rng.range(0, TAU), 0), { uv: 'keep' });
    }
    if (i === 0) kitCandle(C, spot.x, t.top, spot.z, true);
    addGrab(C, 'tankard', spot.x + tLen * 0.28, t.top + 0.07, spot.z + 0.18, i === 0 ? 1 : 4);
    addGrab(C, i === 0 ? 'loaf' : 'stool', spot.x - tLen * 0.2, i === 0 ? t.top + 0.05 : 0,
      spot.z - (i === 0 ? 0.15 : 0.95), 3);
  }
  // settle by the fire
  if (h) {
    const sw = h.wall === 'left' || h.wall === 'right' ? 'back' : far;
    const spot = findOnWall(C, sw, 1.7, 0, 0.5, h.s, false);
    if (spot) claim(C, kitSettle(C, spot.x, spot.z, spot.yaw, 1.7));
  }
  // wall clutter: a scoring board and a row of horse brasses
  const cl = findOnWall(C, far, 0.6, 0, 0.1, C.hd * 0.1, true);
  if (cl) {
    const wi = cl.wi, p = wallPoint(wi, cl.s, 0.05, [0, 0]);
    C.bin.add('wood', C.mat.wood, gCyl(0.24, 0.24, 0.05, 12), trs(p[0], 1.5, p[1], Math.PI / 2, wi.yaw, 0), { uvScale: 3 });
    C.bin.add('soot', C.mat.soot, gCyl(0.2, 0.2, 0.06, 12), trs(p[0] + wi.ix * 0.01, 1.5, p[1] + wi.iz * 0.01, Math.PI / 2, wi.yaw, 0), { uvScale: 3 });
    for (let i = 0; i < 3; i++) {
      const q = wallPoint(wi, cl.s + (i - 1) * 0.16, 0.06, [0, 0]);
      C.bin.add('iron', C.mat.iron, gCyl(0.008, 0.008, 0.22, 4), trs(q[0], 1.62, q[1], 0, wi.yaw, 0.2 * (i - 1)), { uvScale: 8 });
    }
    for (let i = 0; i < 4; i++) {
      const q = wallPoint(wi, cl.s + (i - 1.5) * 0.14, 0.04, [0, 0]);
      C.bin.add('brass', C.mat.brass, gCyl(0.05, 0.05, 0.012, 10), trs(q[0], 1.05, q[1], Math.PI / 2, wi.yaw, 0), { uvScale: 6 });
    }
  }
  // a spare cask by the door
  const sp = findFree(C, 0, C.hd * 0.55, 0.75, 0.75, 0, 1.8);
  if (sp) { kitBarrel(C, sp.x, 0, sp.z, C.rng.range(0, TAU), false, 1.0); claim(C, sp.rect); }
}

function recipeBedroom(C) {
  // Under an open roof the headroom collapses toward the eaves: keep the bed
  // running along the ridge and everything tall near the middle.
  const tight = !!C.room.openToRoof;
  const along = C.W >= C.D;
  const bedYaw = along ? 0 : Math.PI / 2;
  const bx = along ? -C.hw * 0.28 : 0;
  const bz = along ? (tight ? 0 : -C.hd * 0.22) : -C.hd * 0.28;
  const spot = findFree(C, bx, bz, 1.95, 1.06, bedYaw, 1.8);
  if (spot) claim(C, kitBed(C, spot.x, spot.z, spot.yaw));
  const far = along ? 'right' : 'front';
  const chest = findOnWall(C, tight ? (along ? 'right' : 'front') : 'back', 0.95, 0, 0.5, C.hw * 0.3, true);
  if (chest) claim(C, kitChest(C, chest.x, chest.z, chest.yaw, 0.95));
  const ws = kitWashstand(C, tight ? 'left' : (along ? 'left' : 'back'), tight ? 0 : -C.hd * 0.2);
  if (ws) {
    kitCandle(C, ws.x + 0.24, 0.78 + 0.05, ws.z, true);
    addGrab(C, 'crock', ws.x - 0.2, 0.83, ws.z, 1);
  } else {
    kitCandle(C, 0, 0.78, -C.hd + 0.4, true);
  }
  if (!tight) kitPegs(C, far === 'right' ? 'right' : 'front', 0);
  const rug = findFree(C, 0, C.hd * 0.25, 1.5, 1.1, 0, 1.2);
  if (rug) { kitRug(C, rug.x, rug.z, 1.5, 1.1); }
  // the small stuff that sells a bedroom: a light to go to bed by, a book on
  // the chest, a stool wherever there is room for one
  if (chest) {
    const cx = (chest.rect[0] + chest.rect[2]) / 2, cz = (chest.rect[1] + chest.rect[3]) / 2;
    addGrab(C, 'candlestick', cx - 0.22, 0.55, cz, 0);
    addGrab(C, 'book', cx + 0.2, 0.55, cz, 3);
  } else {
    addGrab(C, 'candlestick', 0, 0, -C.hd * 0.5, 0);
    addGrab(C, 'book', 0.4, 0, -C.hd * 0.5, 3);
  }
  addGrab(C, 'stool', C.hw * 0.35, 0, C.hd * 0.3, 2);
}

function recipeStore(C) {
  const rng = C.rng;
  // sacks along the long walls, barrels in a corner, crates stacked
  const wall = C.W >= C.D ? 'back' : 'left';
  const wi = wallInfo(C, wall);
  const n = Math.min(7, Math.max(3, Math.floor(wi.span * 2 / 0.75)));
  let placed = 0;
  for (let i = 0; i < n; i++) {
    const s = (i - (n - 1) / 2) * 0.72;
    const t = 0.42 + rng.sym(0.06);
    const p = wallPoint(wi, s, t, [0, 0]);
    const rect = rectAt(p[0], p[1], 0.62, 0.56);
    if (!freeRect(C, rect) || !insideRoom(C, rect, 0.08)) continue;
    kitSack(C, p[0], 0, p[1], rng.range(0, TAU), rng.range(0.9, 1.1));
    addRectBox(C, rect, 0, 0.6, 'sack', 0.04);
    claim(C, rect);
    placed++;
    if (rng.bool(0.35) && !C.room.openToRoof) kitSack(C, p[0] + rng.sym(0.06), 0.56, p[1] + rng.sym(0.06), rng.range(0, TAU), 0.9);
  }
  for (let i = 0; i < 3; i++) {
    const sp = findFree(C, C.hw * 0.55, -C.hd * 0.4 + i * 0.8, 0.75, 0.75, 0, 1.6);
    if (!sp) continue;
    kitBarrel(C, sp.x, 0, sp.z, rng.range(0, TAU), false, 1.0);
    claim(C, sp.rect);
  }
  for (let i = 0; i < 3; i++) {
    const sp = findFree(C, -C.hw * 0.5, C.hd * 0.35, 0.6, 0.6, 0, 2.0);
    if (!sp) continue;
    kitCrate(C, sp.x, 0, sp.z, rng.sym(0.5), 0.56);
    if (i === 0 && !C.room.openToRoof) kitCrate(C, sp.x + 0.04, 0.56, sp.z - 0.03, rng.sym(0.6), 0.44);
    claim(C, sp.rect);
  }
  // a hanging balance and a scoop
  if (C.H > 2.1) kitBalance(C, 0, C.hd * 0.1, C.H - 0.05);
  addGrab(C, 'pail', C.hw * 0.2, 0, C.hd * 0.5, 0);
  addGrab(C, 'crock', -C.hw * 0.3, 0, -C.hd * 0.55, 1);
  addGrab(C, 'apple', 0.3, 0.05, C.hd * 0.3, 3);
  return placed;
}

function recipeStable(C) {
  const wall = C.W >= C.D ? 'back' : 'left';
  const span = wall === 'back' ? C.W : C.D;
  const count = clamp(Math.floor(span / 2.7), 2, 5);
  kitStalls(C, wall, count);
  // trough near the door, tack on the front wall
  const tr = findFree(C, C.hw * 0.45, C.hd * 0.42, 1.5, 0.6, 0, 1.6);
  if (tr) {
    const M = C.mat, B = C.bin;
    B.add('wood', M.wood, gBox(1.5, 0.42, 0.6), trs(tr.x, 0.21, tr.z, 0, 0, 0), { uvScale: 1.6, cast: true });
    B.add('wood', M.wood, gBox(1.36, 0.3, 0.46), trs(tr.x, 0.3, tr.z, 0, 0, 0), { uvScale: 2 });
    B.add('water', M.water, gBox(1.34, 0.02, 0.44), trs(tr.x, 0.33, tr.z), { uvScale: 1.5 });
    addRectBox(C, tr.rect, 0, 0.42, 'trough', 0.02);
    claim(C, tr.rect);
  }
  const tack = findOnWall(C, 'front', 1.4, 0, 0.16, -C.hw * 0.3, true);
  if (tack) {
    const M = C.mat, B = C.bin, wi = tack.wi;
    for (let i = 0; i < 3; i++) {
      const p = wallPoint(wi, tack.s + (i - 1) * 0.46, 0.08, [0, 0]);
      B.add('wood', M.wood, gCyl(0.02, 0.026, 0.16, 6), trs(p[0], 1.6, p[1], Math.PI / 2, wi.yaw, 0), { uvScale: 8 });
      if (i === 1) {
        // a saddle on a bracket
        B.add('wood', M.wood, gSph(0.24, 8, 6), trs(p[0], 1.34, p[1] + wi.iz * 0.12, 0, wi.yaw, 0, 1.0, 0.55, 0.8), { uvScale: 2 });
      } else {
        // a bridle: two hanging straps and a loop
        B.add('rope', M.rope, gTorus(0.13, 0.014, 4, 8), trs(p[0], 1.4, p[1] + wi.iz * 0.04, 0, wi.yaw, 0), { uvScale: 5 });
        B.add('wood', M.wood, gBox(0.03, 0.5, 0.012), trs(p[0], 1.14, p[1] + wi.iz * 0.03, 0, wi.yaw, 0), { uvScale: 5 });
      }
    }
    claim(C, tack.rect);
  }
  // litter and loose straw in the aisle
  const M = C.mat, B = C.bin, rng = C.rng;
  const wisp = cached('wisp', () => crossQuad(0.22, 2, 0.1));
  for (let i = 0; i < 14; i++) {
    const px = rng.sym(C.hw - 0.5), pz = rng.sym(C.hd - 0.5);
    B.add('straw', M.straw, wisp, trs(px, 0.01, pz, 0, rng.range(0, TAU), 0), { uv: 'keep' });
  }
  const lantern = findOnWall(C, 'front', 0.3, 0, 0.2, C.hw * 0.4, true);
  if (lantern) kitCandle(C, lantern.x, 1.35, lantern.z, true);
  addGrab(C, 'pail', -C.hw * 0.1, 0, C.hd * 0.5, 0);
  addGrab(C, 'stool', C.hw * 0.15, 0, C.hd * 0.3, 2);
  addGrab(C, 'apple', 0.4, 0.05, C.hd * 0.35, 1);
  addGrab(C, 'crock', -C.hw * 0.4, 0, C.hd * 0.45, 3);
}

function recipeBakery(C, hSite) {
  const oven = kitOven(C, hSite.wall, hSite.s);
  const far = farSideWall(C, oven);
  const M = C.mat, B = C.bin;
  // dough trough on legs
  const dt = findOnWall(C, far, 1.7, 0.1, 0.9, -C.hd * 0.1, false);
  if (dt) {
    const wi = dt.wi, c = wallPoint(wi, dt.s, 0.5, [0, 0]);
    B.add('wood', M.wood, gBox(1.7, 0.34, 0.72), trs(c[0], 0.62, c[1], 0, wi.yaw, 0), { uvScale: 1.5, cast: true });
    B.add('wood', M.wood, gBox(1.56, 0.26, 0.6), trs(c[0], 0.7, c[1], 0, wi.yaw, 0), { uvScale: 2 });
    B.add('linen', M.linen, gBox(1.4, 0.06, 0.5), trs(c[0], 0.79, c[1], 0, wi.yaw, 0), { uvScale: 1.6 });
    for (const sx of [-1, 1]) for (const t of [0.2, 0.8]) {
      const q = wallPoint(wi, dt.s + sx * 0.72, 0.1 + 0.8 * t, [0, 0]);
      B.add('wood', M.wood, gBox(0.09, 0.45, 0.09), trs(q[0], 0.225, q[1], 0, wi.yaw, 0), { uvScale: 3 });
    }
    addRectBox(C, dt.rect, 0, 0.8, 'dough-trough', 0.03);
    claim(C, dt.rect);
  }
  // peels leaning by the oven
  const wi = wallInfo(C, oven.wall);
  for (let i = 0; i < 2; i++) {
    const p = wallPoint(wi, oven.s + (i ? 1.35 : -1.35), 0.3, [0, 0]);
    B.add('wood', M.wood, gCyl(0.022, 0.022, 2.0, 6), trs(p[0], 1.0, p[1], 0.12 * (i ? 1 : -1), wi.yaw, 0.1), { uvScale: 4 });
    B.add('plank', M.plank, gBox(0.3, 0.02, 0.36), trs(p[0] + wi.ix * 0.1, 0.16, p[1] + wi.iz * 0.1, 0, wi.yaw, 0), { uvScale: 3 });
  }
  // cooling rack of loaves
  const rk = findOnWall(C, oven.wall === 'back' ? 'front' : 'back', 1.5, 0.1, 0.6, C.hw * 0.25, true);
  if (rk) {
    const w2 = rk.wi, c = wallPoint(w2, rk.s, 0.4, [0, 0]);
    for (let i = 0; i < 7; i++) {
      const q = wallPoint(w2, rk.s + (i - 3) * 0.22, 0.4, [0, 0]);
      B.add('wood', M.wood, gCyl(0.015, 0.015, 0.5, 5), trs(q[0], 0.92, q[1], Math.PI / 2, w2.yaw, Math.PI / 2), { uvScale: 6 });
    }
    for (const sx of [-1, 1]) {
      const q = wallPoint(w2, rk.s + sx * 0.74, 0.4, [0, 0]);
      B.add('wood', M.wood, gBox(0.07, 0.92, 0.5), trs(q[0], 0.46, q[1], 0, w2.yaw, 0), { uvScale: 3, cast: true });
    }
    for (let i = 0; i < 5; i++) {
      const q = wallPoint(w2, rk.s + (i - 2) * 0.3, 0.4 + C.rng.sym(0.06), [0, 0]);
      B.add('crock', M.crock, loafGeo(), trs(q[0], 0.94, q[1], 0, C.rng.range(0, TAU), 0), { uv: 'keep' });
    }
    addRectBox(C, rk.rect, 0, 0.95, 'rack', 0.03);
    claim(C, rk.rect);
    addGrab(C, 'loaf', rk.x, 1.0, rk.z, 0);
    addGrab(C, 'loaf', rk.x + 0.4, 1.0, rk.z, 1);
  }
  // flour sacks
  for (let i = 0; i < 3; i++) {
    const sp = findFree(C, -C.hw * 0.55, C.hd * 0.3, 0.62, 0.56, 0, 2.0);
    if (!sp) continue;
    kitSack(C, sp.x, 0, sp.z, C.rng.range(0, TAU), 1.0);
    addRectBox(C, sp.rect, 0, 0.6, 'sack', 0.04);
    claim(C, sp.rect);
  }
  addGrab(C, 'pail', C.hw * 0.3, 0, C.hd * 0.45, 2);
  addGrab(C, 'crock', -C.hw * 0.2, 0, -C.hd * 0.3, 3);
  return oven;
}

function recipeWorkshop(C, h) {
  const trade = C.plot ? C.plot.id : 'workshop';
  const far = farSideWall(C, h);
  const near = far === 'left' ? 'right' : 'left';
  const wb = kitWorkbench(C, far, 0);
  const M = C.mat, B = C.bin, rng = C.rng;
  if (wb) {
    kitShavings(C, wb.spot.x, wb.spot.z + 0.4, 0.9, 12);
    addGrab(C, 'mallet', wb.spot.x + 0.4, wb.top + 0.05, wb.spot.z, 0);
    addGrab(C, 'crock', wb.spot.x - 0.5, wb.top + 0.11, wb.spot.z + 0.1, 3);
  }
  // whetstone on a stand
  const ws = findFree(C, C.hw * 0.4, C.hd * 0.3, 0.5, 0.4, 0, 1.6);
  if (ws) {
    B.add('wood', M.wood, gBox(0.44, 0.62, 0.34), trs(ws.x, 0.31, ws.z, 0, 0.2, 0), { uvScale: 2, cast: true });
    B.add('hearth', M.hearth, gCyl(0.19, 0.19, 0.06, 12), trs(ws.x, 0.68, ws.z, 0, 0, Math.PI / 2), { uvScale: 3 });
    B.add('iron', M.iron, gCyl(0.014, 0.014, 0.3, 6), trs(ws.x, 0.68, ws.z, 0, 0, Math.PI / 2), { uvScale: 6 });
    addRectBox(C, ws.rect, 0, 0.7, 'whetstone', 0.02);
    claim(C, ws.rect);
  }
  if (trade === 'weaver') {
    // an upright loom: two side frames, a top beam, warp threads, cloth on the roll
    const spot = findOnWall(C, near, 1.7, 0.1, 0.85, 0, true);
    if (spot) {
      const wi = spot.wi, top = Math.min(2.05, C.H - 0.25);
      for (const sx of [-1, 1]) {
        const q = wallPoint(wi, spot.s + sx * 0.8, 0.45, [0, 0]);
        B.add('beam', M.beam, gBox(0.1, top, 0.1), trs(q[0], top / 2, q[1], 0, wi.yaw, 0), { uvScale: 3, cast: true });
      }
      const c = wallPoint(wi, spot.s, 0.45, [0, 0]);
      B.add('beam', M.beam, gBox(1.72, 0.12, 0.12), trs(c[0], top - 0.06, c[1], 0, wi.yaw, 0), { uvScale: 2.4 });
      B.add('beam', M.beam, gCyl(0.07, 0.07, 1.6, 8), trs(c[0], 0.72, c[1], 0, wi.yaw, Math.PI / 2), { uvScale: 3 });
      for (let i = 0; i < 14; i++) {
        const q = wallPoint(wi, spot.s + (i - 6.5) * 0.11, 0.45, [0, 0]);
        B.add('linen', M.linen, gBox(0.012, top - 0.86, 0.006), trs(q[0], 0.8 + (top - 0.86) / 2, q[1], 0, wi.yaw, 0), { uvScale: 4 });
      }
      B.add('linen', M.linen, gBox(1.5, 0.5, 0.09), trs(c[0], 0.5, c[1], 0, wi.yaw, 0), { uvScale: 1.4 });
      addRectBox(C, spot.rect, 0, top, 'loom', 0.04);
      claim(C, spot.rect);
    }
    // bolts of cloth stacked
    for (let i = 0; i < 3; i++) {
      const sp = findFree(C, -C.hw * 0.5, C.hd * 0.4, 0.5, 0.5, 0, 1.8);
      if (!sp) continue;
      for (let k = 0; k < 3; k++) {
        B.add('linen', M.linen, gCyl(0.11, 0.11, 0.7, 8), trs(sp.x, 0.12 + k * 0.23, sp.z + rng.sym(0.04), 0, rng.sym(0.3), Math.PI / 2), { uvScale: 2 });
      }
      addRectBox(C, sp.rect, 0, 0.6, 'cloth-bolts', 0.03);
      claim(C, sp.rect);
    }
    addGrab(C, 'stool', 0, 0, C.hd * 0.2, 2);
  } else {
    // a cooper's yard: a half-made barrel, loose staves, hoops
    const sp = findFree(C, -C.hw * 0.35, 0, 0.85, 0.85, 0, 2.0);
    if (sp) {
      kitBarrel(C, sp.x, 0, sp.z, rng.range(0, TAU), false, 1.05);
      claim(C, sp.rect);
      for (let i = 0; i < 3; i++) {
        B.add('iron', M.iron, gTorus(0.36, 0.02, 4, 12), trs(sp.x + 0.7, 0.03 + i * 0.05, sp.z + 0.5, Math.PI / 2, rng.range(0, TAU), 0), { uvScale: 4 });
      }
    }
    const st = findOnWall(C, near, 1.1, 0, 0.3, C.hd * 0.2, true);
    if (st) {
      const wi = st.wi;
      for (let i = 0; i < 9; i++) {
        const q = wallPoint(wi, st.s + (i - 4) * 0.1, 0.12, [0, 0]);
        B.add('plank', M.plank, gBox(0.09, 0.95, 0.022), trs(q[0], 0.48, q[1], rng.sym(0.05), wi.yaw, rng.sym(0.06)), { uvScale: 2.5 });
      }
      claim(C, st.rect);
    }
    kitShavings(C, 0, C.hd * 0.15, 1.3, 16);
    addGrab(C, 'mallet', -C.hw * 0.2, 0.05, C.hd * 0.3, 2);
  }
  const sh = findOnWall(C, 'back', clamp(C.W * 0.26, 0.9, 1.5), 0, 0.28, C.hw * 0.3, true);
  if (sh) kitShelves(C, 'back', sh.s, clamp(C.W * 0.26, 0.9, 1.5), 2, 1.3, 0.4, stockCrockery);
  addGrab(C, 'bottle', C.hw * 0.2, 0, -C.hd * 0.4, 4);
  if (C.H > 2.2) kitHerbs(C, -C.hw * 0.3, -C.hd * 0.3, C.H - 0.18, 2);
}

/* -------------------------------------------------------------------------- */
/* Grabbables                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Per-kind geometry, material, collider and mass. Built once and shared; each
 * spawned instance is its own mesh because physics writes its transform.
 */
function makeGrabKinds(R, seg) {
  const wood = R.base('woodDark');
  const plank = R.base('woodPlank');
  // Every geometry here goes through mergeParts, which CLONES: the shared
  // build-time cache is disposed at the end of the build, and a grabbable's
  // geometry has to outlive it. The clone also lets us re-centre the part on
  // its collider — physics rotates a body about its centre of mass, and a mug
  // that pivots about its base looks broken the first time you throw it.
  const kinds = {
    tankard: {
      geo: mergeParts([{ geo: tankardGeo(seg), matrix: trs(0, -0.068, 0).clone(), uv: 'keep' }]),
      mat: R.base('pewter'), mass: 0.4, prompt: 'Tankard',
      collider: (p) => cylinderCollider(p, 0.058, 0.068, null, { friction: 0.7 }),
      offset: 0.068,
    },
    bottle: {
      geo: mergeParts([{ geo: bottleGeo(seg), matrix: trs(0, -0.138, 0).clone(), uvScale: 4 }]),
      mat: R.variant('terracotta', { color: 0x7d6a4e }), mass: 0.8, prompt: 'Bottle',
      collider: (p) => cylinderCollider(p, 0.05, 0.138, null, { friction: 0.6 }),
      offset: 0.138,
    },
    loaf: {
      geo: mergeParts([{ geo: loafGeo(), matrix: trs(0, -0.078, 0).clone(), uv: 'keep' }]),
      mat: R.variant('terracotta', { color: 0xd2a25e, roughness: 0.95 }),
      mass: 0.5, prompt: 'Loaf', collider: (p) => boxCollider(p, [0.16, 0.075, 0.11], null, { friction: 0.8 }),
      offset: 0.078,
    },
    apple: {
      geo: mergeParts([{ geo: gSph(0.043, 8, 6), matrix: null, uvScale: 6 }]),
      mat: R.variant('candleWax', { color: 0xa8331f, roughness: 0.45 }),
      mass: 0.2, prompt: 'Apple',
      collider: (p) => sphereCollider(p, 0.043, { friction: 0.6, restitution: 0.22 }),
      offset: 0.05, ccd: true,
    },
    stool: {
      geo: mergeParts([
        { geo: gCyl(0.165, 0.17, 0.05, 12), matrix: trs(0, 0.2, 0).clone(), uvScale: 2.5 },
        ...[0, 1, 2].map((i) => ({
          geo: gCyl(0.022, 0.028, 0.42, 6),
          matrix: trs(Math.cos(i * 2.094) * 0.1, -0.03, Math.sin(i * 2.094) * 0.1,
            Math.sin(i * 2.094) * 0.18, 0, -Math.cos(i * 2.094) * 0.18).clone(),
          uvScale: 5,
        })),
      ]),
      mat: wood, mass: 4, prompt: 'Stool',
      collider: (p) => cylinderCollider(p, 0.17, 0.225, null, { friction: 0.85 }),
      offset: 0.225,
    },
    crock: {
      geo: mergeParts([{ geo: crockGeo(seg), matrix: trs(0, -0.1, 0).clone(), uvScale: 4 }]),
      mat: R.variant('terracotta', { color: 0xa08a68 }), mass: 1.5, prompt: 'Crock',
      collider: (p) => cylinderCollider(p, 0.09, 0.103, null, { friction: 0.7 }),
      offset: 0.103,
    },
    candlestick: {
      geo: mergeParts([
        { geo: gCyl(0.055, 0.065, 0.02, 8), matrix: trs(0, -0.08, 0).clone(), uvScale: 8 },
        { geo: gCyl(0.015, 0.018, 0.06, 7), matrix: trs(0, -0.04, 0).clone(), uvScale: 8 },
        { geo: gCyl(0.012, 0.014, 0.11, 7), matrix: trs(0, 0.045, 0).clone(), uvScale: 8 },
      ]),
      mat: R.base('brass'), mass: 0.7, prompt: 'Candlestick',
      collider: (p) => cylinderCollider(p, 0.06, 0.09, null, { friction: 0.7 }),
      offset: 0.09,
    },
    book: {
      geo: mergeParts([
        { geo: gBox(0.16, 0.05, 0.22), matrix: trs(0, 0, 0).clone(), uvScale: 3 },
        { geo: gBox(0.145, 0.036, 0.2), matrix: trs(0.008, 0.008, 0).clone(), uvScale: 4 },
      ]),
      mat: R.variant('woodDark', { color: 0x6a3a2c }), mass: 1.2, prompt: 'Book',
      collider: (p) => boxCollider(p, [0.085, 0.028, 0.11], null, { friction: 0.8 }),
      offset: 0.028,
    },
    pail: {
      geo: mergeParts([{ geo: pailGeo(seg), matrix: trs(0, -0.16, 0).clone(), uv: 'keep' }]),
      mat: wood, mass: 3, prompt: 'Pail',
      collider: (p) => cylinderCollider(p, 0.19, 0.16, null, { friction: 0.75 }),
      offset: 0.16,
    },
    mallet: {
      geo: mergeParts([
        { geo: gCyl(0.048, 0.048, 0.17, 8), matrix: trs(0, 0.13, 0, 0, 0, Math.PI / 2).clone(), uvScale: 4 },
        { geo: gCyl(0.019, 0.022, 0.3, 6), matrix: trs(0, -0.02, 0).clone(), uvScale: 6 },
      ]),
      mat: plank, mass: 1.8, prompt: 'Mallet',
      collider: (p) => boxCollider(p, [0.085, 0.17, 0.05], null, { friction: 0.8 }),
      offset: 0.17,
    },
  };
  return kinds;
}

/* -------------------------------------------------------------------------- */
/* Factory                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * @param {Object} opts
 * @param {import('../contracts.js').MaterialLibrary} opts.materials
 * @param {Object} opts.buildings   the buildings chunk; `.interiors.rooms` is read
 * @param {Object} [opts.interiors] the interiors chunk; `.roomAt` is used if present
 * @param {Object} [opts.terrain]   unused — floor levels come from the RoomSpec
 * @param {Object} opts.quality
 * @returns {import('../contracts.js').WorldChunk}
 */
export function createFurnishings({ materials, buildings, interiors, terrain, quality } = {}) {
  const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const q = quality || {};
  const group = new THREE.Group();
  group.name = 'furnishings';

  const colliders = [];
  const interactables = [];
  const lightAnchors = [];

  const rooms = buildings && buildings.interiors && Array.isArray(buildings.interiors.rooms)
    ? buildings.interiors.rooms : null;

  if (!rooms || !rooms.length) {
    console.warn('[furnishings] buildings.interiors.rooms is missing or empty — ' +
      'no interior contents built. (buildings.js must publish RoomSpec[].)');
    return {
      group, colliders, interactables, lightAnchors,
      update: null, dispose() { disposeGroup(group); },
      stats: { rooms: 0, reason: 'no RoomSpec[] published by buildings.js' },
    };
  }

  const R = createResolver(materials);
  const detail = clamp(q.geometryDetail ?? 1, 0.5, 1.3);
  const seg = Math.max(7, Math.round(9 * detail));
  const maxVisible = MAX_VISIBLE[q.name] ?? 3;
  const dynBudget = DYN_BUDGET[q.name] ?? 18;

  /**
   * The shared palette. It is deliberately SHORT: with one merged mesh per
   * material per room, every extra material is a draw call in every room that
   * uses it. Several entries are aliases on purpose — the blanket shares the
   * rug's cloth, the burning logs share the dark oak, the bread shares the
   * crockery clay — because a second material there would cost more than the
   * half-shade of colour it buys.
   */
  const mat = {
    wood: R.base('woodDark'),          // carcass, posts, rails, logs
    plank: R.base('woodPlank'),        // boards, tops, seats — the light wood
    beam: R.base('woodBeam'),          // bench and loom timbers
    iron: R.base('ironDark'),          // ironmongery, candle prickets, chains
    brass: R.base('brass'),            // the inn's horse brasses
    pewter: R.base('pewter'),          // plates, tankards, basins
    hearth: R.base('hearthStone'),     // hearthstone, jambs, whetstone
    brick: R.base('brick'),            // the bread oven's dome
    soot: R.base('soot'),              // firebacks and the smoke hood
    linen: R.base('linen'),            // sheets, cloths, warp threads
    sack: R.base('sackcloth'),         // sacks and straw mattresses
    straw: R.variant('strawLitter', { side: 'double' }),
    rope: R.base('rope'),
    wax: R.base('candleWax'),
    crock: R.base('terracotta'),       // crocks, jugs, bowls, loaves
    glass: R.base('glassWindow'),      // physick bottles, trough water
    leaf: R.variant('leaf', { side: 'double' }),
    rug: R.variant('sackcloth', { color: 0x8a5a44, side: 'double' }),
  };
  mat.bread = mat.crock;
  mat.blanket = mat.rug;
  mat.water = mat.glass;
  mat.bark = mat.wood;

  /* ------------------------------------------------------------ flame pools */
  const flames = [];
  const embers = [];
  const grabs = [];
  const out = { colliders, flames, embers, anchors: lightAnchors, grabs };

  /* -------------------------------------------------------- build each room */
  const built = [];          // per-room record used for culling
  let meshes = 0, tris = 0, skipped = 0;
  let keepOutRooms = 0, stairGuessRooms = 0, fillerItems = 0;
  let keepOutVolumes = 0, keepOutBlocks = 0, keepOutMovedRooms = 0;
  let stairGuessAgreed = 0, stairGuessWrong = 0;
  const keepOutReasons = {};
  const stats = {};

  const validUse = (u) => typeof u === 'string' && u.length > 0;

  rooms.forEach((room, idx) => {
    if (!room || !Array.isArray(room.centre) || !(room.width > 0.5) || !(room.depth > 0.5) ||
        typeof room.floorY !== 'number' || typeof room.rotation !== 'number') {
      skipped++;
      return;
    }
    const plot = PLOTS.find((p) => p.id === room.plotId) || null;
    let use = validUse(room.use) ? room.use : null;
    if (!use && plot && INTERIOR_USES[plot.id]) {
      use = INTERIOR_USES[plot.id][Math.min(room.storey | 0, INTERIOR_USES[plot.id].length - 1)];
    }
    if (!use) use = room.storey > 0 ? 'bedroom' : 'hall';

    const ceilY = typeof room.ceilingY === 'number' ? room.ceilingY : room.floorY + 2.4;
    const H = clamp(ceilY - room.floorY, 1.9, 6.0);

    const rg = new THREE.Group();
    rg.name = `furn:${room.plotId}:${room.storey}`;
    rg.position.set(room.centre[0], room.floorY, room.centre[2]);
    rg.rotation.y = room.rotation;
    rg.updateMatrix();
    rg.matrixAutoUpdate = false;
    rg.visible = false;

    const C = {
      room, plot, idx: built.length, use, seg, detail, quality: q,
      W: room.width, D: room.depth, H,
      hw: room.width / 2, hd: room.depth / 2,
      bin: new Bin(), mat, out, occ: [], occTag: [], koBlocks: 0,
      doors: [], windows: [],
      rng: new Rng(`hollowbrook-furn-${room.plotId}-${room.storey}-${plot ? plot.seed : 0}`),
    };

    /* --- parse the openings into room-local space ------------------------- */
    for (const o of room.openings || []) {
      if (!o || !Array.isArray(o.centreWorld)) continue;
      toLocal(room, o.centreWorld[0], o.centreWorld[2], _loc);
      const lx = _loc[0], lz = _loc[1];
      let ix = 0, iz = 0;
      if (Array.isArray(o.inward)) {
        // rotate the world inward vector into room space
        const c = Math.cos(room.rotation), s = Math.sin(room.rotation);
        ix = c * o.inward[0] - s * o.inward[2];
        iz = s * o.inward[0] + c * o.inward[2];
        const l = Math.hypot(ix, iz) || 1;
        ix /= l; iz /= l;
      } else {
        // no inward published: point at the room centre
        const l = Math.hypot(lx, lz) || 1;
        ix = -lx / l; iz = -lz / l;
      }
      const rec = { lx, lz, ix, iz, w: o.width || 0.9, h: o.height || 1.2, wall: o.wall, kind: o.kind };
      if (o.kind === 'door') C.doors.push(rec); else C.windows.push(rec);
    }
    // primary door first — recipes use doors[0] as "the way in"
    C.doors.sort((a, b) => (b.w - a.w));

    /* --- reserve the approach to every door ------------------------------- */
    for (const d of C.doors) {
      const cx = d.lx + d.ix * 1.0, cz = d.lz + d.iz * 1.0;
      const alongX = Math.abs(d.ix) > 0.5;
      claim(C, [
        cx - (alongX ? 1.3 : 0.85), cz - (alongX ? 0.85 : 1.3),
        cx + (alongX ? 1.3 : 0.85), cz + (alongX ? 0.85 : 1.3),
      ]);
    }

    /* --- reserve whatever the interiors stream says is already there ------ */
    const ko = gatherKeepOuts(interiors, room);
    for (const r of ko) {
      claim(C, r, OCC_KEEPOUT);
      if (r.reason) keepOutReasons[r.reason] = (keepOutReasons[r.reason] || 0) + 1;
    }
    if (ko.length) { keepOutRooms++; keepOutVolumes += ko.length; }

    /* --- the hearth, then the stair reservation, then the contents -------- */
    let hearth = null;
    const site = hearthSite(C);
    try {
      if (use === 'bakery') {
        hearth = recipeBakery(C, site);
      } else if (room.storey === 0 && plot && plot.chimneys && plot.chimneys.length) {
        const cold = use === 'store' || use === 'stable';
        hearth = kitHearth(C, {
          wall: site.wall, s: site.s,
          width: use === 'taproom' ? 2.4 : use === 'hall' ? 1.8 : 1.5,
          fire: !cold,
          intensity: use === 'taproom' ? 1.25 : 1.0,
        });
      }
    } catch (err) {
      console.error(`[furnishings] hearth failed in ${room.plotId}/${room.storey}:`, err);
    }

    if (plot && plot.storeys > 1) {
      // The back of the side wall furthest from the fire — the usual place for a
      // flight. This is the FALLBACK: it is only reserved when the interiors
      // stream published nothing for this room.
      const sw = farSideWall(C, hearth);
      const wi = wallInfo(C, sw);
      const s = -(wi.span - 1.7);
      const guess = wallRect(C, sw, s, 3.2, 0, 1.25);
      if (!ko.length) {
        claim(C, guess);
        stairGuessRooms++;
      } else {
        // Real keep-outs win outright. Scoring the guess against them anyway is
        // the only way to know whether the old behaviour was harmless or was
        // standing a dresser on the stairs — so score it, and do not claim it.
        let hit = false;
        for (const r of ko) if (overlaps(guess, r, 0)) { hit = true; break; }
        if (hit) stairGuessAgreed++; else stairGuessWrong++;
      }
    }

    try {
      switch (use) {
        case 'kitchen': recipeKitchen(C, hearth); break;
        case 'shop': recipeShop(C, hearth); break;
        case 'taproom': recipeTaproom(C, hearth); break;
        case 'bedroom': recipeBedroom(C); break;
        case 'store': recipeStore(C); break;
        case 'stable': recipeStable(C); break;
        case 'bakery': break;                 // done above, with the oven
        case 'workshop': recipeWorkshop(C, hearth); break;
        default: recipeHall(C, hearth); break;
      }
    } catch (err) {
      console.error(`[furnishings] room ${room.plotId}/${room.storey} (${use}) failed:`, err);
    }

    try {
      fillerItems += kitFiller(C);
    } catch (err) {
      console.error(`[furnishings] filler failed in ${room.plotId}/${room.storey}:`, err);
    }

    // How much the published volumes actually changed: each block is one
    // candidate placement a keep-out rejected, forcing the search to step on.
    keepOutBlocks += C.koBlocks;
    if (C.koBlocks > 0) keepOutMovedRooms++;

    const b = C.bin.build(rg, `${room.plotId}${room.storey}`);
    meshes += b.meshes;
    tris += b.tris;
    group.add(rg);
    built.push({
      room, group: rg, use, label: `${room.plotId}/${room.storey}`,
      cx: room.centre[0], cy: room.floorY + H * 0.5, cz: room.centre[2],
      plotId: room.plotId, storey: room.storey,
      visible: false, dist: Infinity,
      grabbables: [],
      meshes: b.meshes, tris: b.tris,
    });
  });

  /* --------------------------------------------------------- grabbables */
  const grabKinds = makeGrabKinds(R, seg);
  // rank first (so every room keeps its best object before any room keeps its
  // second), then by how interesting the room is, then by build order
  for (const g of grabs) g.prio = USE_PRIORITY[built[g.room] ? built[g.room].use : ''] ?? 5;
  grabs.sort((a, b) => (a.rank - b.rank) || (a.prio - b.prio) || (a.room - b.room));
  let spawned = 0;
  const dynCounts = {};
  for (const g of grabs) {
    if (spawned >= dynBudget) break;
    const k = grabKinds[g.kind];
    if (!k) continue;
    const mesh = new THREE.Mesh(k.geo, k.mat);
    mesh.castShadow = false;
    mesh.receiveShadow = true;
    mesh.position.set(g.pos[0], g.pos[1] + k.offset, g.pos[2]);
    mesh.quaternion.setFromAxisAngle(_v.set(0, 1, 0), g.yaw);
    mesh.visible = false;
    mesh.name = `furn:grab:${g.kind}`;
    group.add(mesh);
    tagInteractive(mesh, k.prompt, g.kind);
    interactables.push({
      object3D: mesh, body: 'dynamic', grabbable: true,
      collider: k.collider([0, 0, 0]),
      mass: k.mass, prompt: k.prompt, tag: g.kind,
      linearDamping: 0.16, angularDamping: 0.45, ccd: !!k.ccd,
    });
    const rec = built[g.room];
    if (rec) rec.grabbables.push(mesh);
    dynCounts[g.kind] = (dynCounts[g.kind] || 0) + 1;
    spawned++;
    meshes++;
    tris += k.geo.index ? k.geo.index.count / 3 : k.geo.getAttribute('position').count / 3;
  }

  /* -------------------------------------------------------------- flames */
  // NOT via `cached()`: these two are used directly as InstancedMesh geometry
  // and must survive the teardown of the build-time geometry cache.
  const flameTex = makeFlameTexture();
  const flameMat = new THREE.MeshBasicMaterial({
    map: flameTex || null,
    color: flameTex
      // HDR gain on the texel. Above 1 on purpose — see FLAME_GAIN.
      ? new THREE.Color().setRGB(FLAME_GAIN, FLAME_GAIN, FLAME_GAIN, THREE.LinearSRGBColorSpace)
      // No canvas (headless): a mid-flame orange at the same exposure, so the
      // fire is still a fire rather than a black card.
      : new THREE.Color().setRGB(FLAME_GAIN, FLAME_GAIN * 0.305, FLAME_GAIN * 0.019,
        THREE.LinearSRGBColorSpace),
    side: THREE.DoubleSide,
    transparent: true,
    // The three crossed blades overlap; writing depth would make them cut holes
    // in each other in whatever order they happened to be drawn.
    depthWrite: false,
    fog: true,
  });
  flameMat.name = 'furn:flame';
  const flameMesh = buildInstances(group, 'flame', flames, crossQuad(1, 3, 1), flameMat);
  const emberMesh = buildInstances(group, 'ember', embers,
    finish(emberBedGeometry()),
    // transparent + no depth write, exactly like the flame blades above: the rim
    // alpha is what removes the hard edge, and it only works if the material
    // actually blends. vertexColors carries that alpha.
    R.variant('lanternEmissive', {
      side: 'double', color: 0x1a0a04, emissive: 0xff5a18, emissiveIntensity: 1.15,
      transparent: true, depthWrite: false, vertexColors: true,
    }));
  if (flameMesh) meshes++;
  if (emberMesh) meshes++;
  tris += flames.length * 6 + embers.length * EMBER_TRIS;

  /* ------------------------------------------------------------- culling */
  const n = built.length;
  const bestIdx = new Int32Array(Math.max(1, maxVisible)).fill(-1);
  const bestD = new Float32Array(Math.max(1, maxVisible)).fill(Infinity);
  const wantVis = new Uint8Array(n);
  let occupied = -1;
  let roomAtOk = typeof (interiors && interiors.roomAt) === 'function';
  let elapsed = 0;
  let visibleCount = 0;
  let updateCalls = 0;
  let lastOccupied = -2;
  let disposed = false;

  /**
   * Index of the room the interiors stream says the player is in, or -1.
   *
   * The signature matters and this is where it went wrong: `interiors.roomAt`
   * takes THREE NUMBERS (x, y, z), not a Vector3. Handing it the vector made
   * `x` an object and `y`/`z` undefined, so inside the probe loop every
   * comparison was against NaN — `NaN < min` and `NaN > max` are both false —
   * and the very first room accepted every point in the world. The result was
   * `occupiedRoom` frozen on the first RoomSpec (weaver/0) whether the player
   * was in the inn or 55 m away at the spawn point, which pinned the visible
   * set to the weaver's two rooms and left the room you were actually standing
   * in unfurnished.
   *
   * The resolved RoomSpec is NOT the same object `buildings.js` published
   * (interiors normalises it), so identity alone cannot map it back. The
   * WeakMap caches the plotId/storey scan per resolved object: one linear pass
   * the first time a room is entered, a pointer compare every frame after, and
   * no allocation in either path.
   */
  const roomIndexOf = new WeakMap();

  function askRoomAt(p) {
    if (!roomAtOk) return -1;
    let r = null;
    try {
      r = interiors.roomAt(p.x, p.y, p.z);
      // Tolerate an implementation that wants the vector instead.
      if (r === undefined) r = interiors.roomAt(p);
    } catch (err) {
      console.warn('[furnishings] interiors.roomAt threw; falling back to distance culling', err);
      roomAtOk = false;
      return -1;
    }
    if (!r || typeof r !== 'object') return -1;
    const hit = roomIndexOf.get(r);
    if (hit !== undefined) return hit;
    let found = -1;
    for (let i = 0; i < n; i++) {
      if (built[i].room === r ||
          (r.plotId === built[i].plotId && r.storey === built[i].storey)) { found = i; break; }
    }
    roomIndexOf.set(r, found);
    return found;
  }

  /** Is any instance in this pool inside a room that is currently resident? */
  function anyLit(pool) {
    for (let i = 0; i < pool.length; i++) {
      const r = built[pool[i].room];
      if (r && r.visible) return true;
    }
    return false;
  }

  function update(dt, ctx) {
    elapsed += dt || 0;
    const p = ctx && ctx.playerPosition ? ctx.playerPosition : null;

    if (p) {
      occupied = askRoomAt(p);
      const occPlot = occupied >= 0 ? built[occupied].plotId : null;
      for (let i = 0; i < maxVisible; i++) { bestIdx[i] = -1; bestD[i] = Infinity; }
      for (let i = 0; i < n; i++) {
        const r = built[i];
        const dx = r.cx - p.x, dy = r.cy - p.y, dz = r.cz - p.z;
        // vertical distance counts for less: the storey above is close by
        let d = Math.sqrt(dx * dx + dz * dz) + Math.abs(dy) * 0.45;
        const sameBuilding = !!occPlot && r.plotId === occPlot;
        if (i === occupied) d = -1;
        else if (sameBuilding) d *= 0.35;
        r.dist = d;
        // Indoors, another building's rooms are behind two walls and a street.
        const indoorCross = occupied >= 0 && !sameBuilding;
        const limit = indoorCross
          ? (r.visible ? INDOOR_CROSS_HIDE_R : INDOOR_CROSS_SHOW_R)
          : (r.visible ? HIDE_R : SHOW_R);
        if (d > limit) continue;
        // insertion into the top-K, no sort, no allocation
        for (let k = 0; k < maxVisible; k++) {
          if (d < bestD[k]) {
            for (let j = maxVisible - 1; j > k; j--) { bestD[j] = bestD[j - 1]; bestIdx[j] = bestIdx[j - 1]; }
            bestD[k] = d; bestIdx[k] = i;
            break;
          }
        }
      }
      wantVis.fill(0);
      visibleCount = 0;
      for (let k = 0; k < maxVisible; k++) {
        if (bestIdx[k] < 0) continue;
        wantVis[bestIdx[k]] = 1;
        visibleCount++;
      }
      for (let i = 0; i < n; i++) {
        const want = !!wantVis[i];
        const r = built[i];
        if (r.visible !== want) {
          r.visible = want;
          r.group.visible = want;
        }
        // A carried tankard must not vanish because its room is behind you.
        for (let g = 0; g < r.grabbables.length; g++) {
          const o = r.grabbables[g];
          const vis = want || o.position.distanceToSquared(p) < CARRY_VISIBLE_R2;
          if (o.visible !== vis) o.visible = vis;
        }
      }
    }

    /* flicker: one instanceMatrix upload for the whole village */
    // These two InstancedMeshes have frustumCulled off (village-wide extent), so
    // with no room resident they were still two draw calls and two instanceMatrix
    // uploads a frame for 45 + 9 instances scaled to nothing. Outdoors that is
    // the entire cost of this stream, so switch them off wholesale instead.
    if (flameMesh && anyLit(flames)) {
      flameMesh.visible = true;
      for (let i = 0; i < flames.length; i++) {
        const f = flames[i];
        const on = built[f.room] ? built[f.room].visible : false;
        if (!on) {
          _o.matrix.makeScale(0, 0, 0);
          flameMesh.setMatrixAt(i, _o.matrix);
          continue;
        }
        const t = elapsed * f.rate;
        const w = 0.86 + 0.14 * Math.sin(t * 9.1 + f.phase) + 0.07 * Math.sin(t * 21.3 + f.phase * 1.7);
        const hgt = 1.0 + 0.17 * Math.sin(t * 13.3 + f.phase * 2.1);
        _pos.set(f.x, f.y, f.z);
        _rot.setFromAxisAngle(_v.set(0, 1, 0), f.phase);
        _s.set(f.s * w, f.s * hgt, f.s * w);
        _m.compose(_pos, _rot, _s);
        flameMesh.setMatrixAt(i, _m);
      }
      flameMesh.instanceMatrix.needsUpdate = true;
    } else if (flameMesh && flameMesh.visible) {
      flameMesh.visible = false;
    }

    if (emberMesh && anyLit(embers)) {
      emberMesh.visible = true;
      for (let i = 0; i < embers.length; i++) {
        const e = embers[i];
        const on = built[e.room] ? built[e.room].visible : false;
        if (!on) {
          _o.matrix.makeScale(0, 0, 0);
          emberMesh.setMatrixAt(i, _o.matrix);
          continue;
        }
        const s = e.s * (0.94 + 0.06 * Math.sin(elapsed * 3.1 + e.phase));
        _pos.set(e.x, e.y, e.z);
        _rot.identity();
        _s.set(s, 1, s * 0.62);
        _m.compose(_pos, _rot, _s);
        emberMesh.setMatrixAt(i, _m);
      }
      emberMesh.instanceMatrix.needsUpdate = true;
    } else if (emberMesh && emberMesh.visible) {
      emberMesh.visible = false;
    }

    stats.visibleRooms = visibleCount;
    stats.updateCalls = ++updateCalls;
    // Only on change: building this string every frame would be the one
    // allocation in the whole update path.
    if (occupied !== lastOccupied) {
      lastOccupied = occupied;
      stats.occupiedRoom = occupied >= 0 ? built[occupied].label : '-';
    }
  }

  /* -------------------------------------------------------------- teardown */
  for (const g of GC.values()) g.dispose();
  GC = new Map();

  // Rooms start hidden and are revealed by update(). If the integrator forgets
  // to tick this chunk, every interior is invisible — which looks exactly like
  // "furnishings failed" — so say so out loud rather than leaving it to be
  // discovered in a screenshot.
  if (typeof setTimeout === 'function') {
    setTimeout(() => {
      if (!updateCalls && !disposed) {
        console.warn('[furnishings] update() has never been called — interior contents ' +
          'stay hidden until it is. Add the chunk to main.js\'s `updatables`.');
      }
    }, 4000);
  }

  // Worst case is the K heaviest rooms drawn together, not K average rooms.
  const heaviest = built.map((r) => r.meshes).sort((a, b) => b - a);
  const fattest = built.map((r) => r.tris).sort((a, b) => b - a);
  const sumTop = (arr) => arr.slice(0, maxVisible).reduce((a, b) => a + b, 0);
  const grabPerRoom = built.length ? spawned / built.length : 0;

  Object.assign(stats, {
    rooms: built.length,
    skippedRooms: skipped,
    uses: built.reduce((acc, r) => { acc[r.use] = (acc[r.use] || 0) + 1; return acc; }, {}),
    meshes,
    triangles: Math.round(tris),
    /** What is actually drawn at the worst moment, not what was built. */
    drawCallsWorst: sumTop(heaviest) + Math.ceil(grabPerRoom * maxVisible) + 2,
    trianglesWorst: Math.round(sumTop(fattest)),
    worstRoomDrawCalls: heaviest[0] || 0,
    staticColliders: colliders.length,
    grabbables: spawned,
    grabbablesProposed: grabs.length,
    grabbableKinds: dynCounts,
    dynamicBudget: dynBudget,
    fires: lightAnchors.filter((a) => a.kind === 'fire').length,
    candles: lightAnchors.filter((a) => a.kind === 'candle').length,
    flameCards: flames.length,
    maxVisibleRooms: maxVisible,
    /** Rooms where interiors.js told us what to avoid, vs where we guessed. */
    roomsWithPublishedKeepOuts: keepOutRooms,
    roomsWithGuessedStair: stairGuessRooms,
    keepOutVolumes,
    keepOutReasons,
    /** Candidate placements the published volumes rejected, and in how many rooms. */
    keepOutBlockedPlacements: keepOutBlocks,
    roomsWhereFurnitureMoved: keepOutMovedRooms,
    /** Was the old guess right? Only scored where a real keep-out exists. */
    stairGuessAgreed,
    stairGuessWrong,
    fillerItems,
    visibleRooms: 0,
    occupiedRoom: '-',
    materialSubstitutions: R.substituted.slice(),
    buildMs: Math.round((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0),
  });
  console.info('[furnishings]', stats);

  return {
    group,
    colliders,
    interactables,
    lightAnchors,
    update,
    dispose() {
      disposed = true;
      disposeGroup(group);
      for (const k of Object.keys(grabKinds)) grabKinds[k].geo?.dispose?.();
      // Owned here rather than by the resolver, so released here too.
      flameMat.dispose();
      flameTex?.dispose?.();
      R.dispose();
    },
    stats,
    /** For the review pass: jump the camera into a room from the console. */
    rooms: built.map((r) => ({
      plotId: r.plotId, storey: r.storey, use: r.use,
      centre: [r.cx, r.room.floorY, r.cz],
      get visible() { return r.visible; },
    })),
  };
}

/** One InstancedMesh from a list of {x,y,z,s}. */
function buildInstances(group, name, list, geo, material) {
  if (!list.length) return null;
  const im = new THREE.InstancedMesh(geo, material, list.length);
  for (let i = 0; i < list.length; i++) {
    _pos.set(list[i].x, list[i].y, list[i].z);
    _rot.identity();
    _s.set(list[i].s, list[i].s, list[i].s);
    _m.compose(_pos, _rot, _s);
    im.setMatrixAt(i, _m);
  }
  im.instanceMatrix.needsUpdate = true;
  im.castShadow = false;
  im.receiveShadow = false;
  im.frustumCulled = false;      // village-wide extent; culling is all-or-nothing
  im.name = `furn:${name}`;
  group.add(im);
  return im;
}

/**
 * Ask the interiors stream what is already occupying this room (a staircase,
 * a chimney breast).
 *
 * The contract form is `interiors.keepOuts`: an array of WORLD-space oriented
 * boxes, `{ plotId, storey, centre, halfExtents, rotation, reason }`. Those are
 * rotated into this room's local frame and reduced to an axis-aligned rectangle
 * — conservative, i.e. an obliquely-rotated flight reserves slightly more floor
 * than it strictly needs, which is the right way to be wrong here.
 *
 * The older room-local forms are still accepted so that a stream publishing
 * `keepOutsFor(room)` keeps working. Anything we cannot understand is ignored
 * rather than guessed at, and an absent `keepOuts` simply yields [] — the
 * caller then falls back to its own stair guess.
 */
function gatherKeepOuts(interiors, room) {
  if (!interiors) return [];
  const raw = [];
  const tryFn = (fn) => {
    if (typeof fn !== 'function') return;
    try {
      const r = fn.call(interiors, room);
      if (Array.isArray(r)) raw.push(...r);
    } catch (e) { /* ignore — the contract does not require this */ }
  };
  tryFn(interiors.keepOutsFor);
  tryFn(interiors.obstaclesFor);
  tryFn(interiors.stairFootprintFor);
  if (Array.isArray(interiors.keepOuts)) {
    for (const k of interiors.keepOuts) {
      if (!k) continue;
      if (k.plotId && k.plotId !== room.plotId) continue;
      if (k.storey !== undefined && k.storey !== room.storey) continue;
      raw.push(k);
    }
  }
  const ceil = typeof room.ceilingY === 'number' ? room.ceilingY : room.floorY + 2.4;
  const out = [];
  for (const k of raw) {
    if (!k) continue;
    if (Array.isArray(k) && k.length >= 4) {
      out.push([Math.min(k[0], k[2]), Math.min(k[1], k[3]), Math.max(k[0], k[2]), Math.max(k[1], k[3])]);
    } else if (Array.isArray(k.centre) && Array.isArray(k.halfExtents)) {
      /* --- the contract form: a world-space oriented box ------------------- */
      const h = k.halfExtents;
      const hx = Math.abs(h[0] ?? 0), hy = Math.abs(h[1] ?? 0), hz = Math.abs(h[2] ?? 0);
      if (!(hx > 0) || !(hz > 0)) continue;
      // A box entirely above the ceiling or below the floor is somebody else's
      // storey however it is labelled — a stair well cut in the floor above,
      // for instance. Only applied when a y extent was actually published.
      if (typeof k.centre[1] === 'number' && hy > 0) {
        const y0 = k.centre[1] - hy, y1 = k.centre[1] + hy;
        if (y1 < room.floorY - 0.05 || y0 > ceil + 0.05) continue;
      }
      toLocal(room, k.centre[0], k.centre[2], _loc);
      // The box's yaw relative to the room's, then the AABB of the rotated box.
      const rel = (typeof k.rotation === 'number' ? k.rotation : room.rotation) - room.rotation;
      const ca = Math.abs(Math.cos(rel)), sa = Math.abs(Math.sin(rel));
      const ax = ca * hx + sa * hz;
      const az = sa * hx + ca * hz;
      const rect = [_loc[0] - ax, _loc[1] - az, _loc[0] + ax, _loc[1] + az];
      // Carried for the report only; nothing places furniture by reason.
      if (typeof k.reason === 'string') rect.reason = k.reason;
      out.push(rect);
    } else if (k.x0 !== undefined && k.z0 !== undefined) {
      out.push([Math.min(k.x0, k.x1), Math.min(k.z0, k.z1), Math.max(k.x0, k.x1), Math.max(k.z0, k.z1)]);
    } else if (k.lx !== undefined && k.halfW !== undefined) {
      out.push([k.lx - k.halfW, k.lz - k.halfD, k.lx + k.halfW, k.lz + k.halfD]);
    } else if (Array.isArray(k.min) && Array.isArray(k.max)) {
      out.push([Math.min(k.min[0], k.max[0]), Math.min(k.min[1], k.max[1]),
        Math.max(k.min[0], k.max[0]), Math.max(k.min[1], k.max[1])]);
    }
  }
  return out;
}

export default createFurnishings;
