/**
 * ============================================================================
 *  INTERIORS — the structure inside the shell
 * ============================================================================
 * `world/buildings.js` owns the shell: it cuts the openings, hangs the doors and
 * publishes each building's interior as `RoomSpec[]` on `chunk.interiors`. This
 * module owns everything structural INSIDE that shell — floors, the lath-and-
 * plaster lining over the shell's inner face, joists and ceilings, the roof
 * underside where a room is open to it, staircases, upper floors with their
 * stairwell, balustrades and the thresholds you walk in over.
 *
 * It never re-derives a wall position. Every number below comes from a RoomSpec
 * (`centre`, `rotation`, `width`, `depth`, `floorY`, `ceilingY`, `openings`), so
 * if the shell moves, this moves with it.
 *
 * Construction strategy (deliberately the same as buildings.js)
 * ------------------------------------------------------------
 * A room is authored in ROOM-LOCAL space: origin at the centre of the finished
 * floor, +X along `width`, +Z along `depth` (so +Z is the building's front), and
 * Y measured up from `floorY`. Every piece is pushed into a per-material bucket
 * and each bucket is merged into exactly one Mesh, so a whole room costs 3-5
 * draw calls. The room's Group is then placed at `centre` with `rotation.y`,
 * which is bit-for-bit what `plotToWorld` does — colliders computed from the
 * same local coordinates therefore line up with the art.
 *
 * UVs are in metres (dominant-axis planar projection), matching buildings.js, so
 * a board reads the same size indoors as a plank does outdoors.
 *
 * Visibility
 * ----------
 * Eleven interiors cannot all be resident in the frame. Everything is built once
 * at load, then hidden; `update()` shows only the building the player is inside
 * or standing in the doorway of (plus hysteresis), capped at three. A building
 * toggles as a unit rather than room-by-room, because you can see the upper
 * storey up the stairwell from the ground floor — toggling only the room you are
 * standing in would open a hole in the ceiling.
 * ============================================================================
 */

import * as THREE from 'three';
import { boxCollider, quatY } from '../contracts.js';
import { PLOTS, INTERIOR_USES } from './layout.js';
import { Rng, hashSeed } from '../util/rng.js';

/* -------------------------------------------------------------------------- */
/* Dimensions — every one of these is checked against a human                   */
/* -------------------------------------------------------------------------- */

/* The shell reports `width`/`depth` to the inner face of its own plaster and puts
 * its wall COLLIDER 0.09 m inside the storey rect — 35 mm proud of that plaster,
 * "which is where interiors.js lines the wall anyway" (buildings.js). So: 20 mm of
 * air, then a 35 mm lath-and-plaster skin. The air gap is the whole defence
 * against z-fighting the exterior wall, and 55 mm of return is what the shell
 * widened its doorways to allow for. */
const GAP = 0.02;          // air gap between our lining and the shell's inner face
const LIN_T = 0.035;       // lath + plaster lining thickness
const STUD_P = 0.045;      // how far a stud stands proud of the plaster
const STUD_W = 0.135;      // stud width
const STUD_SPACING = 1.15; // stud centres — interior framing, not a modern wall
const SKIRT_H = 0.14;      // skirting height
const SKIRT_P = 0.03;      // skirting projection past the stud face

const BOARD_W = 0.205;     // floor board width
const BOARD_GAP = 0.006;   // joint
const BOARD_T = 0.048;     // board thickness
const BOARD_CUP = 0.005;   // how much a board dishes across its width
const BOARD_SEG = 3.1;     // butt-joint spacing along a board run

const CEIL_BOARD_W = 0.24;
const CEIL_T = 0.032;

const JOIST_W = 0.095;
const JOIST_SPACING = 0.40;
const JOIST_MIN_D = 0.11;
const JOIST_MAX_D = 0.23;

const FLAG_CELL = 0.56;    // flagstone module
const FLAG_T = 0.05;

const RISER_TARGET = 0.178;
const RISER_MIN = 0.170;
const RISER_MAX = 0.185;
const GOING = 0.265;       // tread going
const GOING_MIN = 0.250;
const STAIR_W = 0.98;
const RAIL_H = 0.94;
const NEWEL_H = 1.02;

const THRESH_OUT = 0.30;   // how far a threshold reaches into the room
const RAMP_RUN = 0.62;     // run of the levelling ramp when floor != sill

const DOOR_CLEAR_MIN = 1.00;
const HEAD_CLEAR_MIN = 2.00;

/* Visibility. The plots sit ~16 m apart around the ring, so a 7.5 m show radius
 * measured from the doorway can only ever catch one or two of them, and it is
 * wide enough that an interior is already resident before a doorway is close
 * enough to see into — no pop-in. */
const SHOW_R = 7.5;        // show an interior inside this radius of its door
const HIDE_R = 10.5;       // hide it again outside this one — hysteresis
const MAX_VISIBLE = 3;
const CHECK_INTERVAL = 0.11;
const CHECK_MOVE = 0.4;

/* Fallbacks that mirror buildings.js, used only if the handoff is missing */
const FB_WALL_T = 0.34;
const FB_JETTY = 0.45;
const FB_DOOR_W = { plank: 1.06, double: 1.94, barn: 2.55 };
const FB_DOOR_H = { plank: 2.06, double: 2.26, barn: 2.62 };

/* module-scope scratch — update() and roomAt() must not allocate */
const _q = new THREE.Quaternion();
const _qa = new THREE.Quaternion();
const _ax = new THREE.Vector3();
const _AX_X = new THREE.Vector3(1, 0, 0);
const _AX_Z = new THREE.Vector3(0, 0, 1);

/* -------------------------------------------------------------------------- */
/* Materials                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Interior material keys are new (contracts.js MATERIAL_KEYS), and the material
 * stream is being extended in parallel. Each bucket therefore names a chain of
 * candidates; the first one the library actually knows wins. A missing key comes
 * back as the magenta debug material, which is detectable by name, so a late
 * material library degrades to plausible wood/stone instead of magenta.
 */
const MAT_CHAIN = {
  floor: ['floorBoard', 'woodPlank', 'woodBeam'],
  flag: ['flagstone', 'stone'],
  lining: ['lathPlaster', 'plasterWarm', 'plaster'],
  beam: ['ceilingBeam', 'timberDark', 'timber', 'woodBeam'],
  joinery: ['woodDark', 'woodPlank'],
  stone: ['hearthStone', 'stoneTrim', 'stone'],
  roofUnder: ['thatch', 'strawLitter'],
};

const VC = { vertexColors: true };

function makeMatResolver(materials) {
  const cache = new Map();
  let warned = false;

  const known = (name) => {
    if (!materials || typeof materials.get !== 'function') return null;
    let m = null;
    try { m = materials.get(name); } catch { return null; }
    if (!m) return null;
    // The library hands back a shared magenta material for an unknown key.
    if (m.name === 'DEBUG_MISSING') return null;
    return m;
  };

  return function resolve(key) {
    if (cache.has(key)) return cache.get(key);
    const chain = MAT_CHAIN[key] || [key];
    let out = null;
    for (const name of chain) {
      if (!known(name)) continue;
      if (typeof materials.variant === 'function') {
        try { out = materials.variant(name, VC); } catch { out = null; }
      }
      if (!out) { try { out = materials.get(name); } catch { out = null; } }
      if (out) break;
    }
    if (!out) {
      if (!warned) {
        console.warn('[interiors] no usable interior materials — using a debug material');
        warned = true;
      }
      out = new THREE.MeshStandardMaterial({ color: 0xff00ff, roughness: 0.9 });
    }
    cache.set(key, out);
    return out;
  };
}

/* -------------------------------------------------------------------------- */
/* Geometry plumbing (same conventions as buildings.js)                        */
/* -------------------------------------------------------------------------- */

/**
 * One growable triangle soup per material key.
 *
 * The first version of this file built every piece as its own BoxGeometry, ran a
 * UV/colour/index normalisation pass over it and then called `mergeGeometries`
 * once per bucket. That is ~6 000 BufferGeometries and ~40 000 typed arrays per
 * load: 905 ms in the browser, most of it allocation and the GC that follows it.
 * Writing vertices straight into a doubling Float32Array instead costs one
 * allocation per bucket per room, and there is nothing left to merge.
 *
 * The output is identical geometry: box faces carry their exact axis normals,
 * and a triangle soup gets the face normal that `computeVertexNormals` used to
 * hand it (non-indexed geometry has no shared vertices, so it was already flat).
 */
class Bucket {
  constructor() {
    this.cap = 1536;                       // vertices
    this.pos = new Float32Array(this.cap * 3);
    this.nor = new Float32Array(this.cap * 3);
    this.col = new Float32Array(this.cap * 3);
    this.n = 0;
  }
  ensure(extra) {
    if (this.n + extra <= this.cap) return;
    let cap = this.cap;
    while (cap < this.n + extra) cap *= 2;
    const used = this.n * 3;
    const grow = (a) => { const o = new Float32Array(cap * 3); o.set(a.subarray(0, used)); return o; };
    this.pos = grow(this.pos);
    this.nor = grow(this.nor);
    this.col = grow(this.col);
    this.cap = cap;
  }
}

/** Per-material geometry accumulator for one room. */
class Builder {
  constructor() { this.map = new Map(); }

  bucket(key) {
    let bk = this.map.get(key);
    if (!bk) this.map.set(key, bk = new Bucket());
    return bk;
  }

  /** One vertex with an explicit normal. */
  vert(bk, x, y, z, nx, ny, nz, tint) {
    const o = bk.n * 3;
    const p = bk.pos, nn = bk.nor, c = bk.col;
    p[o] = x; p[o + 1] = y; p[o + 2] = z;
    nn[o] = nx; nn[o + 1] = ny; nn[o + 2] = nz;
    if (tint) { c[o] = tint.r; c[o + 1] = tint.g; c[o + 2] = tint.b; }
    else { c[o] = 1; c[o + 1] = 1; c[o + 2] = 1; }
    bk.n++;
  }

  /** One triangle; the normal follows (b-a) x (c-a), as `quad` below assumes. */
  tri(key, ax, ay, az, bx, by, bz, cx, cy, cz, tint) {
    let nx = (by - ay) * (cz - az) - (bz - az) * (cy - ay);
    let ny = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
    let nz = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    const l = Math.sqrt(nx * nx + ny * ny + nz * nz);
    if (l < 1e-12) return;                 // degenerate: it would draw nothing
    nx /= l; ny /= l; nz /= l;
    const bk = this.bucket(key);
    bk.ensure(3);
    this.vert(bk, ax, ay, az, nx, ny, nz, tint);
    this.vert(bk, bx, by, bz, nx, ny, nz, tint);
    this.vert(bk, cx, cy, cz, nx, ny, nz, tint);
  }

  /** A flat [x,y,z, x,y,z, ...] triangle soup. */
  soup(key, arr, tint) {
    if (!arr || arr.length < 9) return;
    const bk = this.bucket(key);
    bk.ensure(arr.length / 3);
    for (let i = 0; i + 8 < arr.length; i += 9) {
      this.tri(key, arr[i], arr[i + 1], arr[i + 2], arr[i + 3], arr[i + 4],
        arr[i + 5], arr[i + 6], arr[i + 7], arr[i + 8], tint);
    }
  }

  /** One mesh per bucket. Returns {meshes, tris}. */
  finish(resolve, namePrefix) {
    const meshes = [];
    let tris = 0;
    for (const [key, bk] of this.map) {
      const n = bk.n;
      if (n < 3) continue;
      const len = n * 3;
      const pos = new Float32Array(len); pos.set(bk.pos.subarray(0, len));
      const nor = new Float32Array(len); nor.set(bk.nor.subarray(0, len));
      const col = new Float32Array(len); col.set(bk.col.subarray(0, len));
      // Dominant-axis planar projection, UVs in metres — same rule as
      // buildings.js, so a board reads the same size indoors as out.
      const uv = new Float32Array(n * 2);
      for (let i = 0; i < n; i++) {
        const o = i * 3;
        const ax = nor[o] < 0 ? -nor[o] : nor[o];
        const ay = nor[o + 1] < 0 ? -nor[o + 1] : nor[o + 1];
        const az = nor[o + 2] < 0 ? -nor[o + 2] : nor[o + 2];
        let u, v;
        if (ay >= ax && ay >= az) { u = pos[o]; v = pos[o + 2]; }
        else if (ax >= az) { u = pos[o + 2]; v = pos[o + 1]; }
        else { u = pos[o]; v = pos[o + 1]; }
        uv[i * 2] = u;
        uv[i * 2 + 1] = v;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
      // Bounds by hand: BufferGeometry.computeBoundingSphere() walks the same
      // vertices through Vector3/Box3 and costs about three times as much.
      let x0 = Infinity, y0 = Infinity, z0 = Infinity;
      let x1 = -Infinity, y1 = -Infinity, z1 = -Infinity;
      for (let i = 0; i < len; i += 3) {
        const px = pos[i], py = pos[i + 1], pz = pos[i + 2];
        if (px < x0) x0 = px; if (px > x1) x1 = px;
        if (py < y0) y0 = py; if (py > y1) y1 = py;
        if (pz < z0) z0 = pz; if (pz > z1) z1 = pz;
      }
      const mx = (x0 + x1) / 2, my = (y0 + y1) / 2, mz = (z0 + z1) / 2;
      let r2 = 0;
      for (let i = 0; i < len; i += 3) {
        const dx = pos[i] - mx, dy = pos[i + 1] - my, dz = pos[i + 2] - mz;
        const d = dx * dx + dy * dy + dz * dz;
        if (d > r2) r2 = d;
      }
      geo.boundingBox = new THREE.Box3(new THREE.Vector3(x0, y0, z0), new THREE.Vector3(x1, y1, z1));
      geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(mx, my, mz), Math.sqrt(r2));
      const mesh = new THREE.Mesh(geo, resolve(key));
      mesh.name = `${namePrefix}.${key}`;
      // Interiors are lit from outside, so casting from most of this geometry is
      // pure shadow-map cost. Floors and ceiling boards are the exception: they
      // are the horizontal barrier between storeys, and without them the sun
      // coming through an upper window also lands on the ground floor.
      mesh.castShadow = key === 'floor' || key === 'flag';
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      mesh.updateMatrix();
      meshes.push(mesh);
      tris += n / 3;
    }
    this.map.clear();
    return { meshes, tris };
  }
}

/** Slightly-varied multiplicative vertex tint (linear, ~1.0). */
function tintOf(rng, amount = 0.05, warm = 0) {
  const k = 1 + rng.sym(amount);
  return { r: k * (1 + warm), g: k, b: k * (1 - warm * 0.8) };
}

/* The six faces of a box as [nAxis, nSign, uAxis, uSign, vAxis, vSign], with the
 * in-plane axes chosen so u x v = n — which makes the corner order below wind
 * counter-clockwise seen from outside, exactly like BoxGeometry did. */
const BOX_FACES = [
  [0, 1, 1, 1, 2, 1], [0, -1, 2, 1, 1, 1],
  [1, 1, 2, 1, 0, 1], [1, -1, 0, 1, 2, 1],
  [2, 1, 0, 1, 1, 1], [2, -1, 1, 1, 0, 1],
];
const CORNERS = [-1, -1, 1, -1, 1, 1, -1, 1];
const _h = [0, 0, 0];
const _cnr = new Float64Array(12);

/**
 * A box authored in room-local space (x,y,z = centre; d* = full sizes), written
 * straight into the bucket. `rotAxis` ('x' or 'z') tilts it about its own centre
 * first — the same order as BoxGeometry.rotateX().translate() used to.
 */
function emitBox(b, key, x, y, z, dx, dy, dz, tint, rotAxis, angle) {
  if (dx <= 1e-4 || dy <= 1e-4 || dz <= 1e-4) return;
  _h[0] = dx / 2; _h[1] = dy / 2; _h[2] = dz / 2;
  const rot = rotAxis ? 1 : 0;
  const rc = rot ? Math.cos(angle) : 1;
  const rs = rot ? Math.sin(angle) : 0;
  const rotX = rotAxis === 'x';
  const bk = b.bucket(key);
  bk.ensure(36);
  for (let fi = 0; fi < 6; fi++) {
    const f = BOX_FACES[fi];
    const na = f[0], ns = f[1], ua = f[2], us = f[3], va = f[4], vs = f[5];
    let nx = 0, ny = 0, nz = 0;
    if (na === 0) nx = ns; else if (na === 1) ny = ns; else nz = ns;
    if (rot) {
      if (rotX) { const a = ny, c = nz; ny = rc * a - rs * c; nz = rs * a + rc * c; }
      else { const a = nx, c = ny; nx = rc * a - rs * c; ny = rs * a + rc * c; }
    }
    for (let ci = 0; ci < 4; ci++) {
      const su = CORNERS[ci * 2], sv = CORNERS[ci * 2 + 1];
      let px = 0, py = 0, pz = 0;
      const vn = ns * _h[na];
      if (na === 0) px = vn; else if (na === 1) py = vn; else pz = vn;
      const vu = su * us * _h[ua];
      if (ua === 0) px = vu; else if (ua === 1) py = vu; else pz = vu;
      const vv = sv * vs * _h[va];
      if (va === 0) px = vv; else if (va === 1) py = vv; else pz = vv;
      if (rot) {
        if (rotX) { const a = py, c = pz; py = rc * a - rs * c; pz = rs * a + rc * c; }
        else { const a = px, c = py; px = rc * a - rs * c; py = rs * a + rc * c; }
      }
      _cnr[ci * 3] = x + px; _cnr[ci * 3 + 1] = y + py; _cnr[ci * 3 + 2] = z + pz;
    }
    // two triangles: 0-1-2 and 0-2-3
    b.vert(bk, _cnr[0], _cnr[1], _cnr[2], nx, ny, nz, tint);
    b.vert(bk, _cnr[3], _cnr[4], _cnr[5], nx, ny, nz, tint);
    b.vert(bk, _cnr[6], _cnr[7], _cnr[8], nx, ny, nz, tint);
    b.vert(bk, _cnr[0], _cnr[1], _cnr[2], nx, ny, nz, tint);
    b.vert(bk, _cnr[6], _cnr[7], _cnr[8], nx, ny, nz, tint);
    b.vert(bk, _cnr[9], _cnr[10], _cnr[11], nx, ny, nz, tint);
  }
}

/** A box authored in room-local space (x,y,z = centre; d* = full sizes). */
function localBox(b, key, x, y, z, dx, dy, dz, tint) {
  emitBox(b, key, x, y, z, dx, dy, dz, tint, null, 0);
}

/** A box rotated about local X (used for stair strings, rafters, handrails). */
function tiltedBoxX(b, key, x, y, z, dx, dy, dz, angle, tint) {
  emitBox(b, key, x, y, z, dx, dy, dz, tint, 'x', angle);
}

/** A box rotated about local Z. */
function tiltedBoxZ(b, key, x, y, z, dx, dy, dz, angle, tint) {
  emitBox(b, key, x, y, z, dx, dy, dz, tint, 'z', angle);
}

function tri(o, ax, ay, az, bx, by, bz, cx, cy, cz) {
  o.push(ax, ay, az, bx, by, bz, cx, cy, cz);
}

/** Quad a-b-c-d; the normal follows (b-a) x (c-a). */
function quad(o, a, b, c, d) {
  tri(o, a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  tri(o, a[0], a[1], a[2], c[0], c[1], c[2], d[0], d[1], d[2]);
}

/* -------------------------------------------------------------------------- */
/* Board fields                                                               */
/* -------------------------------------------------------------------------- */

/**
 * A field of boards. `runAxis` is the axis a board's length lies along; rows step
 * across the other axis on a lattice shared by every region of the same floor, so
 * the joints line up across a stairwell opening.
 *
 * Faces up (a floor) or down (a ceiling). Boards dish slightly across their width
 * — a flat quad reads as lino, a cupped one reads as oak.
 */
function boardField(b, key, o) {
  const runX = o.runAxis === 'x';
  const a0 = runX ? o.x0 : o.z0, a1 = runX ? o.x1 : o.z1;
  const c0 = runX ? o.z0 : o.x0, c1 = runX ? o.z1 : o.x1;
  if (a1 - a0 < 0.05 || c1 - c0 < 0.05) return 0;

  const down = !!o.faceDown;
  const bw = o.boardW ?? BOARD_W;
  const gap = down ? 0.0 : BOARD_GAP;         // ceilings butt tight: no peep holes
  const th = o.thickness ?? BOARD_T;
  const cup = down ? -(o.cup ?? 0.004) : (o.cup ?? BOARD_CUP);
  const y = o.y;
  const seg = o.segLen ?? BOARD_SEG;
  const rng = o.rng;
  const out = [];

  // Lattice, anchored on `origin` so neighbouring regions stay in register.
  const org = o.origin ?? 0;
  const pitch = bw + gap;
  const i0 = Math.floor((c0 - org) / pitch) - 1;
  const i1 = Math.ceil((c1 - org) / pitch) + 1;

  const P = (a, c, yy) => (runX ? [a, yy, c] : [c, yy, a]);
  // Mapping (a,c) -> (x,z) is a mirror when boards run along Z, so every winding
  // flips with it. Getting this wrong turns a floor into a one-way surface.
  const flip = !runX;
  const Q = (arr, a, b, c, d) => (flip ? quad(arr, d, c, b, a) : quad(arr, a, b, c, d));

  for (let i = i0; i <= i1; i++) {
    const lo = Math.max(c0, org + i * pitch);
    const hi = Math.min(c1, org + i * pitch + bw);
    if (hi - lo < 0.03) continue;
    const mid = (lo + hi) / 2;
    // butt joints: phase varies per row but is deterministic
    const phase = seg * (0.17 + 0.66 * fract(Math.abs(i) * 0.6180339887 + 0.31));
    let s = a0;
    let guard = 0;
    while (s < a1 - 0.02 && guard++ < 400) {
      let e = Math.min(a1, nextJoint(s, a0, seg, phase));
      if (e - s < 0.25) e = Math.min(a1, s + 0.25);
      const yTop = y + (rng ? rng.sym(0.0015) : 0);
      const yMid = yTop - cup;
      // top (or bottom) surface, two quads across the width
      if (!down) {
        Q(out, P(s, lo, yTop), P(s, mid, yMid), P(e, mid, yMid), P(e, lo, yTop));
        Q(out, P(s, mid, yMid), P(s, hi, yTop), P(e, hi, yTop), P(e, mid, yMid));
        // the two long edges, so a joint has a real shadow line
        Q(out, P(s, lo, yTop), P(e, lo, yTop), P(e, lo, yTop - th), P(s, lo, yTop - th));
        Q(out, P(s, hi, yTop - th), P(e, hi, yTop - th), P(e, hi, yTop), P(s, hi, yTop));
      } else {
        Q(out, P(s, lo, yTop), P(e, lo, yTop), P(e, mid, yMid), P(s, mid, yMid));
        Q(out, P(s, mid, yMid), P(e, mid, yMid), P(e, hi, yTop), P(s, hi, yTop));
      }
      s = e;
    }
  }
  b.soup(key, out, o.tint);

  // A solid backing plane so a joint can never become a hole into the void.
  const back = [];
  const yb = down ? y + th : y - th - 0.004;
  if (down) {
    Q(back, P(a0, c0, yb), P(a1, c0, yb), P(a1, c1, yb), P(a0, c1, yb));
  } else {
    Q(back, P(a0, c0, yb), P(a0, c1, yb), P(a1, c1, yb), P(a1, c0, yb));
  }
  b.soup(key, back, o.backTint || { r: 0.62, g: 0.58, b: 0.54 });
  return out.length / 9;
}

function fract(v) { return v - Math.floor(v); }

function nextJoint(s, a0, seg, phase) {
  const k = Math.floor((s - a0 - phase) / seg) + 1;
  return a0 + phase + k * seg;
}

/**
 * Flagstones: a jittered lattice of slabs with mortar gaps and a little height
 * variation, over a dark base plane that shows through the joints.
 */
function flagField(b, key, o) {
  const { x0, x1, z0, z1, y, rng } = o;
  const cell = o.cell ?? FLAG_CELL;
  const nx = Math.max(1, Math.round((x1 - x0) / cell));
  const nz = Math.max(1, Math.round((z1 - z0) / cell));
  const cx = (x1 - x0) / nx, cz = (z1 - z0) / nz;
  // jittered lattice, shared corners so the joints are continuous
  const jx = new Float32Array((nx + 1) * (nz + 1));
  const jz = new Float32Array((nx + 1) * (nz + 1));
  for (let i = 0; i <= nx; i++) {
    for (let k = 0; k <= nz; k++) {
      const idx = k * (nx + 1) + i;
      const edgeX = i === 0 || i === nx, edgeZ = k === 0 || k === nz;
      jx[idx] = x0 + i * cx + (edgeX ? 0 : rng.sym(cx * 0.16));
      jz[idx] = z0 + k * cz + (edgeZ ? 0 : rng.sym(cz * 0.16));
    }
  }
  const out = [];
  const inset = 0.017;
  for (let i = 0; i < nx; i++) {
    for (let k = 0; k < nz; k++) {
      const a = k * (nx + 1) + i, c = k * (nx + 1) + i + 1;
      const d = (k + 1) * (nx + 1) + i + 1, e = (k + 1) * (nx + 1) + i;
      const px = [jx[a], jx[c], jx[d], jx[e]];
      const pz = [jz[a], jz[c], jz[d], jz[e]];
      const mx = (px[0] + px[1] + px[2] + px[3]) / 4;
      const mz = (pz[0] + pz[1] + pz[2] + pz[3]) / 4;
      const yy = y - rng.range(0, 0.012);
      const pt = [];
      for (let n = 0; n < 4; n++) {
        const dx = px[n] - mx, dz = pz[n] - mz;
        const len = Math.hypot(dx, dz) || 1;
        pt.push([px[n] - (dx / len) * inset, yy, pz[n] - (dz / len) * inset]);
      }
      // wound counter-clockwise seen from above -> +Y
      quad(out, pt[0], pt[3], pt[2], pt[1]);
    }
  }
  b.soup(key, out, o.tint);
  const base = [];
  quad(base, [x0, y - FLAG_T, z0], [x0, y - FLAG_T, z1], [x1, y - FLAG_T, z1], [x1, y - FLAG_T, z0]);
  b.soup(key, base, { r: 0.44, g: 0.42, b: 0.40 });
  return out.length / 9;
}

/* -------------------------------------------------------------------------- */
/* Rooms: sanitation and the local frame                                       */
/* -------------------------------------------------------------------------- */

const num = (v) => typeof v === 'number' && Number.isFinite(v);

function sanitizeRooms(list) {
  const out = [];
  let rejected = 0;
  for (const r of list || []) {
    if (!r || typeof r !== 'object') { rejected++; continue; }
    if (!num(r.floorY) || !num(r.ceilingY) || !num(r.width) || !num(r.depth)) { rejected++; continue; }
    if (!num(r.rotation)) { rejected++; continue; }
    const c = r.centre;
    if (!Array.isArray(c) || c.length < 3 || !num(c[0]) || !num(c[1]) || !num(c[2])) { rejected++; continue; }
    if (r.width < 1.8 || r.depth < 1.8 || r.width > 60 || r.depth > 60) { rejected++; continue; }
    const head = r.openToRoof && num(r.ridgeY) ? Math.max(r.ridgeY, r.ceilingY) : r.ceilingY;
    if (head - r.floorY < 2.0) { rejected++; continue; }
    if (r.ceilingY - r.floorY < 0.7) { rejected++; continue; }
    out.push(r);
  }
  return { rooms: out, rejected };
}

/** Room-local frame helpers. All of them are pure arithmetic on the RoomSpec. */
function frameOf(room) {
  const rot = room.rotation;
  const cos = Math.cos(rot), sin = Math.sin(rot);
  const cx = room.centre[0], cz = room.centre[2];
  return {
    rot, cos, sin, cx, cz,
    /** local (x,z) + height above floorY -> world triple */
    toWorld(lx, ly, lz) {
      return [cx + lx * cos + lz * sin, room.floorY + ly, cz - lx * sin + lz * cos];
    },
    /** world (x,z) -> local x */
    localX(wx, wz) { return (wx - cx) * cos - (wz - cz) * sin; },
    /** world (x,z) -> local z */
    localZ(wx, wz) { return (wx - cx) * sin + (wz - cz) * cos; },
  };
}

/**
 * The four walls of a room in local space. `u` runs along the wall, `n` runs
 * INTO the room from the shell's inner face (so n = 0 is the shell face).
 */
function wallsOf(room) {
  const hw = room.width / 2, hd = room.depth / 2;
  return [
    { side: 'front', ux: 1, uz: 0, nx: 0, nz: -1, ox: 0, oz: hd, L: room.width, openings: [] },
    { side: 'back', ux: 1, uz: 0, nx: 0, nz: 1, ox: 0, oz: -hd, L: room.width, openings: [] },
    { side: 'right', ux: 0, uz: 1, nx: -1, nz: 0, ox: hw, oz: 0, L: room.depth, openings: [] },
    { side: 'left', ux: 0, uz: 1, nx: 1, nz: 0, ox: -hw, oz: 0, L: room.depth, openings: [] },
  ];
}

/** A box authored in wall space -> a local-space box. */
function wallBox(b, key, w, u0, u1, v0, v1, n0, n1, tint) {
  const xA = w.ox + w.ux * u0 + w.nx * n0;
  const xB = w.ox + w.ux * u1 + w.nx * n1;
  const zA = w.oz + w.uz * u0 + w.nz * n0;
  const zB = w.oz + w.uz * u1 + w.nz * n1;
  const x0 = Math.min(xA, xB), x1 = Math.max(xA, xB);
  const z0 = Math.min(zA, zB), z1 = Math.max(zA, zB);
  localBox(b, key, (x0 + x1) / 2, (v0 + v1) / 2, (z0 + z1) / 2,
    Math.max(x1 - x0, 1e-3), Math.max(v1 - v0, 1e-3), Math.max(z1 - z0, 1e-3), tint);
}

/**
 * Attach each opening to its wall, in wall coordinates. `centreWorld` is the
 * authority (it removes any doubt about which way `lx` runs on a back or side
 * wall); `lx` is the fallback.
 */
function placeOpenings(room, walls, frame, warn) {
  const bySide = new Map(walls.map((w) => [w.side, w]));
  let doors = 0;
  for (const op of room.openings || []) {
    if (!op || !num(op.width) || !num(op.height) || op.width <= 0.05) continue;
    const w = bySide.get(op.wall) || bySide.get('front');
    let u = num(op.lx) ? op.lx : 0;
    const cw = op.centreWorld;
    if (Array.isArray(cw) && num(cw[0]) && num(cw[2])) {
      u = (w.side === 'front' || w.side === 'back')
        ? frame.localX(cw[0], cw[2])
        : frame.localZ(cw[0], cw[2]);
    }
    const half = op.width / 2;
    const lim = Math.max(0, w.L / 2 - half);
    if (u > lim + 0.02 || u < -lim - 0.02) warn(`opening off the end of the ${w.side} wall`);
    u = THREE.MathUtils.clamp(u, -lim, lim);
    const sill = num(op.sillY) ? op.sillY : room.floorY;
    const v0 = sill - room.floorY;
    const isDoor = op.kind === 'door';
    const rec = {
      kind: isDoor ? 'door' : 'window',
      u0: u - half, u1: u + half, v0, v1: v0 + op.height,
      u, width: op.width, height: op.height,
      primary: !!op.primary, spec: op, wall: w,
    };
    if (isDoor) {
      doors++;
      if (op.width < DOOR_CLEAR_MIN - 1e-3) warn(`door only ${op.width.toFixed(2)} m clear`);
      if (op.height < HEAD_CLEAR_MIN - 1e-3) warn(`door only ${op.height.toFixed(2)} m tall`);
    }
    w.openings.push(rec);
  }
  for (const w of walls) w.openings.sort((a, c) => a.u0 - c.u0);
  return doors;
}

/* -------------------------------------------------------------------------- */
/* Wall lining                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Lath-and-plaster panels between exposed studs, over the shell's inner face,
 * inset far enough that it can never z-fight the exterior plaster. Panels are
 * broken around openings and the reveal is returned at every one, so the wall
 * reads as a real 340 mm of daub rather than a painted line.
 */
function buildLining(b, room, walls, ctx) {
  const { rng, colliders, frame } = ctx;
  const H = Math.max(0.4, room.ceilingY - room.floorY);
  const nIn = GAP + LIN_T;             // plaster face, measured from the shell
  const nStud = nIn + STUD_P;          // stud face
  const linTint = () => tintOf(rng, 0.035, 0.01);
  const oakTint = () => tintOf(rng, 0.07);

  const jw = 0.06;                     // depth of the returned reveal

  for (const w of walls) {
    const half = w.L / 2;
    const ops = w.openings;

    /* -- plaster panels -------------------------------------------------------
     * Openings are widened by `jw` for the panel breaks, and that jw-wide strip
     * around each opening is the RETURN: it runs all the way back to the shell's
     * inner face (n = 0) instead of stopping at the plaster face, so the wall
     * reads as its real thickness. Tiling it this way means no two coplanar
     * faces ever point the same direction — which is what z-fights.
     */
    let cur = -half;
    for (const op of ops) {
      const a = op.u0 - jw, c = op.u1 + jw;
      if (a > cur + 0.04) wallBox(b, 'lining', w, cur, a, 0, H, GAP, nIn, linTint());
      const vTop = Math.min(H, op.v1);
      const vBot = Math.max(0, op.v0);
      // jamb returns
      wallBox(b, 'lining', w, a, op.u0, vBot, vTop, 0, nIn, linTint());
      wallBox(b, 'lining', w, op.u1, c, vBot, vTop, 0, nIn, linTint());
      // head return, then the panel above it
      if (vTop < H) {
        const hv = Math.min(H, vTop + jw);
        wallBox(b, 'lining', w, a, c, vTop, hv, 0, nIn, linTint());
        if (hv < H - 0.03) wallBox(b, 'lining', w, a, c, hv, H, GAP, nIn, linTint());
      }
      // sill return + the panel under it
      if (vBot > 0.05) {
        const sv = Math.max(0, vBot - jw);
        wallBox(b, 'lining', w, a, c, sv, vBot, 0, nIn, linTint());
        if (sv > 0.03) wallBox(b, 'lining', w, a, c, 0, sv, GAP, nIn, linTint());
        // an oak sill board over the return
        wallBox(b, 'joinery', w, a - 0.02, c + 0.02, vBot - 0.045, vBot, 0, nIn + 0.035, oakTint());
      }
      cur = Math.max(cur, c);
    }
    if (half > cur + 0.04) wallBox(b, 'lining', w, cur, half, 0, H, GAP, nIn, linTint());

    // -- exposed frame: wall plate and studs ---------------------------------
    const plateT = 0.11;
    const headBlocked = ops.filter((o) => o.v1 > H - plateT - 0.02)
      .map((o) => [o.u0 - jw, o.u1 + jw]);
    for (const [u0, u1] of freeSpans(-half, half, headBlocked)) {
      wallBox(b, 'beam', w, u0, u1, H - plateT, H, GAP, nStud, oakTint());
    }
    const nStuds = Math.max(1, Math.round(w.L / STUD_SPACING));
    for (let i = 1; i < nStuds; i++) {
      const u = -half + (w.L * i) / nStuds;
      const u0 = u - STUD_W / 2, u1 = u + STUD_W / 2;
      // studs stop at an opening rather than crossing it
      const segs = freeSpans(0, H - plateT, blockedV(ops, u0, u1, jw));
      for (const [v0, v1] of segs) {
        if (v1 - v0 < 0.12) continue;
        wallBox(b, 'beam', w, u0, u1, v0, v1, nIn - 0.005, nStud, oakTint());
      }
    }

    // -- skirting, broken at every door so a doorway has no lip --------------
    const doorSpans = ops.filter((o) => o.kind === 'door')
      .map((o) => [o.u0 - jw, o.u1 + jw]);
    for (const [u0, u1] of freeSpans(-half, half, doorSpans)) {
      wallBox(b, 'joinery', w, u0, u1, 0, SKIRT_H, nIn - 0.01, nStud + SKIRT_P, oakTint());
    }

    // -- collider for the lining face, broken at doors ------------------------
    for (const [u0, u1] of freeSpans(-half, half, doorSpans)) {
      const len = u1 - u0;
      if (len < 0.12) continue;
      const uc = (u0 + u1) / 2;
      const nc = (GAP + nStud) / 2, nt = (nStud - GAP) / 2;
      const x = w.ox + w.ux * uc + w.nx * nc;
      const z = w.oz + w.uz * uc + w.nz * nc;
      const hx = w.ux ? len / 2 : nt;
      const hz = w.uz ? len / 2 : nt;
      colliders.push(boxCollider(frame.toWorld(x, H / 2, z), [hx, H / 2, hz],
        ctx.worldQ, { tag: 'interior-wall' }));
    }
  }

  // Corner posts. Each is 0.13 square and stands 0.05 proud of the studs, which
  // keeps its face 0.145 m clear of where the capsule can put the camera.
  const px = room.width / 2 - 0.045 - 0.065;
  const pz = room.depth / 2 - 0.045 - 0.065;
  for (const sx of [-1, 1]) {
    for (const sz of [-1, 1]) {
      localBox(b, 'beam', sx * px, H / 2, sz * pz, 0.13, H, 0.13, oakTint());
    }
  }
}

/** Complement of `blocked` inside [min,max]. */
function freeSpans(min, max, blocked, minLen = 0.05) {
  const bs = (blocked || [])
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

/** Openings overlapping the vertical strip [u0,u1] -> blocked v intervals. */
function blockedV(ops, u0, u1, pad = 0) {
  const r = [];
  for (const o of ops) if (o.u1 > u0 && o.u0 < u1) r.push([o.v0 - pad, o.v1 + pad]);
  return r;
}

/* -------------------------------------------------------------------------- */
/* Ceilings, joists and the roof underside                                     */
/* -------------------------------------------------------------------------- */

/**
 * Exposed joists at 400 mm centres running the short way, with boards above them.
 * `nextFloorY` (the floor of the room above, if any) caps the joist depth so the
 * assembly can never push up through the boards of the storey above.
 */
function buildFlatCeiling(b, room, ctx, nextFloorY, hole) {
  const { rng } = ctx;
  const hw = room.width / 2 - GAP - LIN_T;
  const hd = room.depth / 2 - GAP - LIN_T;
  const yc = room.ceilingY - room.floorY;      // joist underside, room-local
  let jd = JOIST_MAX_D;
  if (num(nextFloorY)) {
    jd = (nextFloorY - room.floorY) - yc - BOARD_T - CEIL_T - 0.01;
  }
  jd = THREE.MathUtils.clamp(jd, JOIST_MIN_D, JOIST_MAX_D);

  const shortIsX = room.width <= room.depth;   // joists span the short way
  const oak = () => tintOf(rng, 0.075);
  const runAxis = shortIsX ? 'z' : 'x';

  // Boards above the joists, seen from below — in four regions if the flight
  // that starts in this room needs a well through the ceiling. A ceiling that
  // sealed over the stairwell would put a lid on the staircase.
  const regions = hole
    ? [
      { x0: -hw, x1: Math.min(hw, hole.x0), z0: -hd, z1: hd },
      { x0: Math.max(-hw, hole.x1), x1: hw, z0: -hd, z1: hd },
      { x0: Math.max(-hw, hole.x0), x1: Math.min(hw, hole.x1), z0: -hd, z1: Math.min(hd, hole.z0) },
      { x0: Math.max(-hw, hole.x0), x1: Math.min(hw, hole.x1), z0: Math.max(-hd, hole.z1), z1: hd },
    ]
    : [{ x0: -hw, x1: hw, z0: -hd, z1: hd }];

  for (const r of regions) {
    if (r.x1 - r.x0 < 0.06 || r.z1 - r.z0 < 0.06) continue;
    boardField(b, 'floor', {
      runAxis, ...r,
      y: yc + jd, faceDown: true, boardW: CEIL_BOARD_W, thickness: CEIL_T,
      rng, origin: 0, segLen: 3.6, tint: tintOf(rng, 0.05, 0.01),
      backTint: { r: 0.55, g: 0.52, b: 0.48 },
    });
  }

  // joists, trimmed round the well
  const span = shortIsX ? room.width : room.depth;
  const along = shortIsX ? hd : hw;            // half-length of the run they sit on
  const cross = shortIsX ? hw : hd;
  const n = Math.max(2, Math.round((2 * along) / JOIST_SPACING));
  for (let i = 0; i <= n; i++) {
    const p = -along + (2 * along * i) / n;
    const sag = 0.006 * Math.cos((p / Math.max(0.1, along)) * Math.PI * 0.5);
    // `p` is the joist's position across the room, `spans` its extent along it
    const inWell = hole && (shortIsX
      ? (p > hole.z0 - JOIST_W && p < hole.z1 + JOIST_W)
      : (p > hole.x0 - JOIST_W && p < hole.x1 + JOIST_W));
    const blocked = inWell
      ? [shortIsX ? [hole.x0 - 0.06, hole.x1 + 0.06] : [hole.z0 - 0.06, hole.z1 + 0.06]]
      : [];
    for (const [a, c] of freeSpans(-cross, cross, blocked, 0.12)) {
      const mid = (a + c) / 2, len = c - a;
      if (shortIsX) localBox(b, 'beam', mid, yc + jd / 2 - sag, p, len, jd, JOIST_W, oak());
      else localBox(b, 'beam', p, yc + jd / 2 - sag, mid, JOIST_W, jd, len, oak());
    }
  }
  // a binder down the middle of a wide span — 5 m of unsupported oak sags
  if (span > 5.2 && !hole) {
    if (shortIsX) localBox(b, 'beam', 0, yc + jd / 2 + 0.02, 0, 2 * hw, jd + 0.05, 0.19, oak());
    else localBox(b, 'beam', 0, yc + jd / 2 + 0.02, 0, 0.19, jd + 0.05, 2 * hd, oak());
  }
  return yc + jd;
}

/**
 * Coordinates along one axis: every breakpoint, plus enough intermediate steps
 * that no span is longer than `target`. Breakpoints are where a surface creases.
 */
function axisSamples(breaks, target) {
  const out = [breaks[0]];
  for (let i = 1; i < breaks.length; i++) {
    const a = breaks[i - 1], c = breaks[i];
    if (c - a < 1e-3) continue;
    const n = Math.max(1, Math.ceil((c - a) / Math.max(0.2, target)));
    for (let k = 1; k <= n; k++) out.push(a + ((c - a) * k) / n);
  }
  return out;
}

/**
 * The roof underside for a room open to it: rafters, purlins, a ridge beam and
 * the thatch underside above them. A hipped or half-hipped plot gets end slopes
 * so nothing pokes out through the shell.
 */
function buildOpenRoof(b, room, ctx, plot) {
  const { rng } = ctx;
  const hw = room.width / 2 - GAP - LIN_T;
  const hd = room.depth / 2 - GAP - LIN_T;
  const pitch = ((plot && plot.roofPitch) || 52) * Math.PI / 180;

  /* buildings.js publishes ridgeY as the apex of the thatch UNDERSIDE (it takes
   * thatchVert off the ridge line) and ceilingY as where that same surface meets
   * the wall head. Ours is that surface, dropped by DROP and run 0.12 m past the
   * wall so its edge dies inside the masonry rather than landing exactly on the
   * corner. DROP has to beat the shell's ridge sag (THATCH_SAG = 0.085) or the
   * two soffits cross somewhere along the ridge and z-fight there. */
  const DROP = 0.16;
  const spring = room.ceilingY - room.floorY - DROP;
  let apex = num(room.ridgeY) ? room.ridgeY - room.floorY - DROP
    : spring + hd * Math.tan(pitch);
  apex = Math.max(apex, spring + 0.85);

  const roof = (plot && plot.roof) || 'thatch';
  const hipRun = roof === 'thatchHip' ? hd : (roof === 'thatchHalfHip' ? hd * 0.44 : 0);
  const ridgeHalf = Math.max(0.2, hw - hipRun);

  const under = (x, z) => {
    let y = apex - (Math.abs(z) / Math.max(0.1, hd)) * (apex - spring);
    if (hipRun > 0.05) {
      const dx = Math.abs(x) - ridgeHalf;
      if (dx > 0) {
        const yh = apex - (dx / hipRun) * (apex - spring);
        if (yh < y) y = yh;
      }
    }
    return y;
  };

  /* -- thatch underside ------------------------------------------------------
   * Sampled on a grid, but the grid LINES are pinned to the creases (the ridge
   * at z = 0 and the hips at x = +/-ridgeHalf). Without that, a cell straddling
   * the ridge interpolates straight across it and the roof loses up to 0.9 m of
   * its apex — a blunt ridge, and geometry that no longer matches its collider.
   */
  const detail = ctx.detail;
  const shw = hw + 0.12, shd = hd + 0.12;   // run the sheet into the wall
  // A hip's crease runs diagonally in plan, so no rectangular grid line can lie
  // on it and every cell that straddles it interpolates BELOW the crease. The
  // error goes as the cell size squared, so hipped plots get a finer grid: at
  // 1.3 m the corner of a hip dished by 0.29 m, at 0.6 m by about 0.07 m.
  const cell = (hipRun > 0.05 ? 0.6 : 1.15) / Math.max(0.4, detail);
  const xs = axisSamples(hipRun > 0.05 ? [-shw, -ridgeHalf, ridgeHalf, shw] : [-shw, shw], cell);
  const zs = axisSamples([-shd, 0, shd], cell * 0.85);
  const out = [];
  for (let i = 0; i < xs.length - 1; i++) {
    const xa = xs[i], xb = xs[i + 1];
    for (let k = 0; k < zs.length - 1; k++) {
      const za = zs[k], zb = zs[k + 1];
      const A = [xa, under(xa, za), za], B = [xb, under(xb, za), za];
      const C = [xb, under(xb, zb), zb], D = [xa, under(xa, zb), zb];
      quad(out, A, B, C, D);   // wound to face down
    }
  }
  b.soup('roofUnder', out, tintOf(rng, 0.07, 0.015));

  // -- gable lining above the wall head (only where the end is a real gable) --
  if (hipRun < 0.06) {
    for (const sx of [-1, 1]) {
      const x = sx * hw;
      const g = [];
      const apexP = [x, apex - 0.02, 0];
      const a = [x, spring, -hd], c = [x, spring, hd];
      // face inward: the +X gable faces -X, the -X gable faces +X
      if (sx > 0) tri(g, a[0], a[1], a[2], c[0], c[1], c[2], apexP[0], apexP[1], apexP[2]);
      else tri(g, a[0], a[1], a[2], apexP[0], apexP[1], apexP[2], c[0], c[1], c[2]);
      b.soup('lining', g, tintOf(rng, 0.03, 0.01));
    }
  }

  // -- rafters, purlins, ridge, collars --------------------------------------
  const oak = () => tintOf(rng, 0.075);
  const slope = Math.atan2(apex - spring, hd);
  const rl = Math.hypot(hd, apex - spring);
  const step = 0.5 / Math.max(0.5, detail);
  const nRaft = Math.max(2, Math.round((2 * ridgeHalf) / step));
  for (let i = 0; i <= nRaft; i++) {
    const x = -ridgeHalf + (2 * ridgeHalf * i) / nRaft;
    for (const sz of [-1, 1]) {
      // three's rotateX(+t) sends +Z to (0,-sin t, cos t) — it tips +Z DOWN — so
      // the rafter that falls toward +Z is +slope, not -slope.
      tiltedBoxX(b, 'beam', x, (spring + apex) / 2 - 0.06, sz * hd / 2,
        0.085, 0.165, rl, sz * slope, oak());
    }
    // A collar is a beam across the room at head height: only fit one where it
    // is actually above a head. In a low attic it would be a face-height plank.
    const yC = spring + (apex - spring) * 0.62;
    if (i % 3 === 1 && yC > 2.05) {
      const zc = hd * (apex - yC) / Math.max(0.1, apex - spring);
      localBox(b, 'beam', x, yC, 0, 0.075, 0.14, 2 * zc, oak());
    }
  }
  localBox(b, 'beam', 0, apex - 0.14, 0,
    2 * ridgeHalf + (hipRun > 0.05 ? 0 : 0.3), 0.24, 0.15, oak());
  for (const t of [0.34, 0.72]) {
    for (const sz of [-1, 1]) {
      const z = sz * hd * (1 - t);
      const y = under(0, z) - 0.20;
      if (y < 2.05) continue;
      localBox(b, 'beam', 0, y, z, 2 * ridgeHalf, 0.14, 0.12, oak());
    }
  }

  /* -- collision for the slopes ----------------------------------------------
   * The underside is planar, so one rotated box per slope is exact. Without
   * these, an attic with a low wall head lets the player push their head — and
   * the camera — straight out through the thatch.
   */
  const { colliders, frame, worldQ } = ctx;
  const T = 0.35;
  for (const sz of [-1, 1]) {
    const ny = hd / rl, nz = (apex - spring) / rl;   // upward normal of the slope
    const cy = (spring + apex) / 2 + (T / 2) * ny;
    const cz = sz * (hd / 2 + (T / 2) * nz);
    // note: the collider follows the same plane as the sheet, so the player is
    // stopped by the roof they can see rather than by the shell's thatch
    _qa.setFromAxisAngle(_AX_X, sz * slope);
    _q.set(worldQ[0], worldQ[1], worldQ[2], worldQ[3]).multiply(_qa);
    // +0.15 on the length so the two slopes overlap at the ridge instead of
    // meeting exactly, which leaves a hairline crack the solver can find.
    colliders.push(boxCollider(frame.toWorld(0, cy, cz), [hw + 0.1, T / 2, rl / 2 + 0.15],
      [_q.x, _q.y, _q.z, _q.w], { tag: 'interior-roof' }));
  }
  if (hipRun > 0.3) {
    const hl = Math.hypot(hipRun, apex - spring);
    const slopeE = Math.atan2(apex - spring, hipRun);
    for (const sx of [-1, 1]) {
      const ny = hipRun / hl, nx = (apex - spring) / hl;
      const cy = (spring + apex) / 2 + (T / 2) * ny;
      const cx = sx * (ridgeHalf + hipRun / 2 + (T / 2) * nx);
      _qa.setFromAxisAngle(_AX_Z, -sx * slopeE);
      _q.set(worldQ[0], worldQ[1], worldQ[2], worldQ[3]).multiply(_qa);
      colliders.push(boxCollider(frame.toWorld(cx, cy, 0), [hl / 2 + 0.15, T / 2, hd + 0.1],
        [_q.x, _q.y, _q.z, _q.w], { tag: 'interior-roof' }));
    }
  }
  return apex;
}

/* -------------------------------------------------------------------------- */
/* Floors                                                                      */
/* -------------------------------------------------------------------------- */

const FLAG_USES = new Set(['bakery', 'stable', 'workshop', 'store']);

/**
 * A floor over the region rect, optionally with a stairwell hole. The collider is
 * four boxes around the hole (a Rapier box cannot have one) or a single box when
 * there is none.
 */
function buildFloor(b, room, ctx, opts) {
  const { rng, colliders, frame } = ctx;
  const hw = room.width / 2 - GAP * 0.5;      // tuck the floor under the lining
  const hd = room.depth / 2 - GAP * 0.5;
  const hole = opts.hole;
  const flags = opts.flags;
  const key = flags ? 'flag' : 'floor';
  const runAxis = room.width >= room.depth ? 'x' : 'z';
  const tint = tintOf(rng, flags ? 0.06 : 0.055, flags ? 0 : 0.012);

  const regions = hole
    ? [
      { x0: -hw, x1: hole.x0, z0: -hd, z1: hd },
      { x0: hole.x1, x1: hw, z0: -hd, z1: hd },
      { x0: hole.x0, x1: hole.x1, z0: -hd, z1: hole.z0 },
      { x0: hole.x0, x1: hole.x1, z0: hole.z1, z1: hd },
    ]
    : [{ x0: -hw, x1: hw, z0: -hd, z1: hd }];

  for (const r of regions) {
    if (r.x1 - r.x0 < 0.06 || r.z1 - r.z0 < 0.06) continue;
    if (flags) {
      flagField(b, key, { ...r, y: 0, rng, tint });
    } else {
      boardField(b, key, {
        runAxis, ...r, y: 0, rng, origin: 0, tint,
        segLen: BOARD_SEG * (0.85 + 0.3 * rng.next()),
      });
    }
    // collider: 0.25 m of substance so nothing can tunnel through a floor
    const cx = (r.x0 + r.x1) / 2, cz = (r.z0 + r.z1) / 2;
    colliders.push(boxCollider(frame.toWorld(cx, -0.125, cz),
      [(r.x1 - r.x0) / 2, 0.125, (r.z1 - r.z0) / 2], ctx.worldQ,
      { tag: `interior-floor:${room.plotId || '?'}:${room.storey | 0}` }));
  }
  return regions;
}

/* -------------------------------------------------------------------------- */
/* Thresholds                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * A threshold at every external door: a slab through the full thickness of the
 * wall with its top at the door's sill, and — if the shell's sill and the
 * finished floor disagree by more than a few millimetres — a shallow ramp inside
 * so there is no step to trip the character controller.
 */
function buildThresholds(b, room, walls, ctx) {
  const { rng, colliders, frame } = ctx;
  const wt = num(room.wallThickness) ? room.wallThickness : FB_WALL_T;
  const stoneTint = () => tintOf(rng, 0.05);
  let count = 0;

  for (const w of walls) {
    for (const op of w.openings) {
      if (op.kind !== 'door') continue;
      count++;
      // 6 mm proud of the finished floor: a real threshold stands slightly up,
      // and coincident surfaces z-fight. Autostep is 0.45 m, so it is nothing.
      const top = op.v0 + 0.006;
      const halfW = op.width / 2 + 0.09;
      // slab: from the outer face of the wall to THRESH_OUT inside the room
      const n0 = -(wt - 0.01), n1 = THRESH_OUT;
      wallBox(b, 'stone', w, op.u - halfW, op.u + halfW, top - 0.16, top, n0, n1, stoneTint());

      const xA = w.ox + w.ux * (op.u - halfW) + w.nx * n0;
      const xB = w.ox + w.ux * (op.u + halfW) + w.nx * n1;
      const zA = w.oz + w.uz * (op.u - halfW) + w.nz * n0;
      const zB = w.oz + w.uz * (op.u + halfW) + w.nz * n1;
      colliders.push(boxCollider(
        frame.toWorld((xA + xB) / 2, top - 0.14, (zA + zB) / 2),
        [Math.max(0.05, Math.abs(xB - xA) / 2), 0.14, Math.max(0.05, Math.abs(zB - zA) / 2)],
        ctx.worldQ, { tag: 'interior-threshold' }));

      // Level the sill into the floor if the shell disagrees with us. A rotated
      // box, not a step: 0.1 m over 0.62 m is 9 degrees, well inside the
      // controller's 48-degree limit, and the player never feels it.
      const drop = op.v0;
      if (Math.abs(drop) > 0.012) {
        const run = RAMP_RUN;
        const ang = Math.atan2(drop, run);
        const len = Math.hypot(run, drop);
        const nm = n1 + run / 2;
        const x = w.ox + w.ux * op.u + w.nx * nm;
        const z = w.oz + w.uz * op.u + w.nz * nm;
        const alongZ = w.uz === 0;   // front/back wall: the run lies along Z
        // rotateX(+t) tips +Z DOWN; rotateZ(+t) tips +X UP. The high end is the
        // one toward the doorway, i.e. against the wall's inward normal.
        const tilt = alongZ ? ang * w.nz : -ang * w.nx;
        const rampTint = tintOf(rng, 0.06);
        if (alongZ) {
          tiltedBoxX(b, 'joinery', x, drop / 2 - 0.02, z,
            op.width + 0.14, 0.09, len, tilt, rampTint);
        } else {
          tiltedBoxZ(b, 'joinery', x, drop / 2 - 0.02, z,
            len, 0.09, op.width + 0.14, tilt, rampTint);
        }

        _ax.set(alongZ ? 1 : 0, 0, alongZ ? 0 : 1);
        _qa.setFromAxisAngle(_ax, tilt);
        _q.set(ctx.worldQ[0], ctx.worldQ[1], ctx.worldQ[2], ctx.worldQ[3]).multiply(_qa);
        colliders.push(boxCollider(frame.toWorld(x, drop / 2 - 0.02, z),
          alongZ ? [(op.width + 0.14) / 2, 0.045, len / 2] : [len / 2, 0.045, (op.width + 0.14) / 2],
          [_q.x, _q.y, _q.z, _q.w], { tag: 'interior-ramp' }));
        if (Math.abs(drop) > 0.28) {
          console.warn(`[interiors] ${room.plotId}: door sill is ${drop.toFixed(2)} m off the finished floor — ramped, but the shell should agree`);
        }
      }
    }
  }
  return count;
}

/* -------------------------------------------------------------------------- */
/* The chimney breast                                                          */
/* -------------------------------------------------------------------------- */

/* furnishings.js builds what is IN the fireplace — hearthstone, fireback, jambs,
 * logs, fire. Nobody built the masonry the flue actually runs up, so from inside
 * the taproom the fire sat against a flat sooty rectangle: a hole in the wall
 * rather than a fireplace. This is that masonry: two piers, a bressumer across
 * the opening, a mantel shelf and a corbelled gathering up to the ceiling. */
const BREAST_P = 0.42;      // how far the breast face stands off the shell's face
const BREAST_PIER = 0.52;   // masonry each side of the opening
const BREAST_REV = 0.62;    // reveal each side of the firebox, before the pier
const BREAST_LINT = 0.32;   // bressumer depth
const BREAST_GATHER = 4;    // corbel courses in the gathering

/** A wall-space rect (u along the wall, n into the room) as a room-local AABB. */
function wallRectLocal(w, u0, u1, n0, n1) {
  const xA = w.ox + w.ux * u0 + w.nx * n0;
  const xB = w.ox + w.ux * u1 + w.nx * n1;
  const zA = w.oz + w.uz * u0 + w.nz * n0;
  const zB = w.oz + w.uz * u1 + w.nz * n1;
  return {
    x0: Math.min(xA, xB), x1: Math.max(xA, xB),
    z0: Math.min(zA, zB), z1: Math.max(zA, zB),
  };
}

/** Same 20 mm slack as furnishings' `overlaps()`, so the two agree on a graze. */
function rectsOverlap(a, c, pad = 0.02) {
  return a.x0 < c.x1 - pad && a.x1 > c.x0 + pad && a.z0 < c.z1 - pad && a.z1 > c.z0 + pad;
}

/**
 * Where the fireplace goes. This deliberately reproduces `hearthSite()` and the
 * nudge loop in furnishings.js step for step — the biggest stack, the nearest
 * wall, the front-wall fallback, the same 0.4 m search — because the breast and
 * the fire in it MUST end up in the same place. `blocked` is the keep-out list
 * this room has already published, which is what furnishings will be testing
 * against when it picks its own site.
 */
function hearthPlacement(room, walls, frame, plot, use, blocked) {
  if (!plot || !Array.isArray(plot.chimneys) || !plot.chimneys.length) return null;
  let best = plot.chimneys[0];
  for (const ch of plot.chimneys) {
    if ((ch.width || 1) * ch.height > (best.width || 1) * best.height) best = ch;
  }
  const cos = Math.cos(plot.rotation), sin = Math.sin(plot.rotation);
  const wx = plot.position[0] + best.x * cos + best.z * sin;
  const wz = plot.position[2] - best.x * sin + best.z * cos;
  const lx = frame.localX(wx, wz), lz = frame.localZ(wx, wz);
  const hw = room.width / 2, hd = room.depth / 2;

  let side = 'back', d = lz + hd;
  if (hd - lz < d) { side = 'front'; d = hd - lz; }
  if (lx + hw < d) { side = 'left'; d = lx + hw; }
  if (hw - lx < d) { side = 'right'; d = hw - lx; }
  const bySide = new Map(walls.map((w) => [w.side, w]));
  if (side === 'front') {
    const front = bySide.get('front');
    if (front && front.openings.some((o) => o.kind === 'door')) side = 'back';
  }
  const w = bySide.get(side) || walls[0];
  const uRaw = (side === 'front' || side === 'back') ? lx : lz;

  const fw = THREE.MathUtils.clamp(
    use === 'taproom' ? 2.4 : use === 'hall' ? 1.8 : 1.5,
    1.0, Math.min(2.6, room.width * 0.34));
  const span = w.L / 2;
  const lim = Math.max(0, span - fw / 2 - 0.55);
  const u0 = THREE.MathUtils.clamp(uRaw, -lim, lim);
  const testHalf = (fw + 0.9) / 2;

  /* The three tests furnishings applies to a candidate site, in its terms: the
   * slab must be inside the room, no window centre may be within 0.2 m of it,
   * and it must miss everything already claimed — which at that moment is the
   * approach it reserves in front of each door, plus our own keep-outs. */
  const winPts = [];
  const doorRects = [];
  for (const ww of walls) {
    for (const op of ww.openings) {
      const px = ww.ox + ww.ux * op.u, pz = ww.oz + ww.uz * op.u;
      if (op.kind !== 'door') { winPts.push(px, pz); continue; }
      const cx = px + ww.nx * 1.0, cz = pz + ww.nz * 1.0;
      const alongX = Math.abs(ww.nx) > 0.5;
      doorRects.push({
        x0: cx - (alongX ? 1.3 : 0.85), x1: cx + (alongX ? 1.3 : 0.85),
        z0: cz - (alongX ? 0.85 : 1.3), z1: cz + (alongX ? 0.85 : 1.3),
      });
    }
  }

  const okAt = (s) => {
    const r = wallRectLocal(w, s - testHalf, s + testHalf, 0, 0.85);
    const e = 1e-3;
    if (!(r.x0 > -hw - e && r.x1 < hw + e && r.z0 > -hd - e && r.z1 < hd + e)) return false;
    for (let i = 0; i < winPts.length; i += 2) {
      if (winPts[i] > r.x0 - 0.2 && winPts[i] < r.x1 + 0.2 &&
          winPts[i + 1] > r.z0 - 0.2 && winPts[i + 1] < r.z1 + 0.2) return false;
    }
    for (const q of doorRects) if (rectsOverlap(r, q)) return false;
    for (const q of blocked) if (rectsOverlap(r, q)) return false;
    return true;
  };

  /* The search walks outward in 0.4 m steps from the RAW stack position, exactly
   * as furnishings does, and — again exactly as furnishings does — keeps the last
   * step even if nothing came free. Bailing out here instead would put the fire
   * somewhere with no masonry round it, which is the defect we are fixing; the
   * breast itself shrinks per side rather than covering a window. */
  /* furnishings measures its front and left walls in the opposite direction to
   * `wallsOf()` (its `ax` is -1 where our `ux` is +1), so on those two walls its
   * search steps the other way along the wall. Without this flip the fire and the
   * breast set off in opposite directions the moment either has to move. */
  const flip = (w.side === 'front' || w.side === 'left') ? -1 : 1;
  let u = u0;
  for (let k = 0; k < 14 && !okAt(u); k++) {
    const step = Math.ceil((k + 1) / 2) * 0.4 * (k % 2 ? 1 : -1);
    u = THREE.MathUtils.clamp(uRaw + flip * step, -lim, lim);
  }
  return { w, u, fw, clear: okAt(u) };
}

/**
 * The breast itself. Everything is authored in wall space so it sits on the
 * shell's inner face however the room is rotated. The firebox opening is left
 * 0.2 m wider each side than the widest thing furnishings puts in it, so the two
 * assemblies read as one fireplace and never intersect.
 */
function buildChimneyBreast(b, room, ctx, place, blocked) {
  const { rng, colliders, frame } = ctx;
  const { w, u, fw } = place;
  const H = Math.max(0.4, room.ceilingY - room.floorY);
  const stone = () => tintOf(rng, 0.055);
  const oak = () => tintOf(rng, 0.07);

  const hO = THREE.MathUtils.clamp(H - 0.62, 1.98, 2.12);   // head of the firebox
  if (hO > H - 0.16) return null;                   // ceiling too low for a breast
  const P = BREAST_P;

  /* How far the masonry may run each way before it would cover a window, a door
   * or the end of the wall. The reveal never drops below fw/2 + 0.46 — that is
   * the widest thing furnishings puts in the opening (its bressumer, fw/2 + 0.42)
   * plus clearance — and beyond the reveal comes as much pier as will fit. */
  const revMin = fw / 2 + 0.46;
  const side = [0, 0];      // [-side, +side] half widths: [reveal, reveal+pier]
  const rev = [0, 0], pier = [0, 0];
  for (let si = 0; si < 2; si++) {
    const sgn = si === 0 ? -1 : 1;
    let reach = w.L / 2 - sgn * u - 0.06;
    for (const op of w.openings) {
      if (sgn > 0 && op.u0 > u) reach = Math.min(reach, op.u0 - u - 0.07);
      else if (sgn < 0 && op.u1 < u) reach = Math.min(reach, u - op.u1 - 0.07);
    }
    rev[si] = THREE.MathUtils.clamp(reach, revMin, fw / 2 + BREAST_REV);
    pier[si] = THREE.MathUtils.clamp(reach - rev[si], 0, BREAST_PIER);
    if (pier[si] < 0.16) pier[si] = 0;              // a 60 mm pier reads as junk
    side[si] = rev[si] + pier[si];
  }
  const hL = rev[0], hR = rev[1];                   // the opening
  const bL = side[0], bR = side[1];                 // the whole breast

  // piers, floor to the underside of the bressumer
  if (pier[0] > 0) wallBox(b, 'stone', w, u - bL, u - hL, 0, hO, 0, P, stone());
  if (pier[1] > 0) wallBox(b, 'stone', w, u + hR, u + bR, 0, hO, 0, P, stone());
  // a chamfer on each pier's inner edge, so the reveal is not a plain arris
  if (pier[0] > 0) wallBox(b, 'stone', w, u - hL - 0.075, u - hL, 0, hO, P - 0.075, P, stone());
  if (pier[1] > 0) wallBox(b, 'stone', w, u + hR, u + hR + 0.075, 0, hO, P - 0.075, P, stone());
  // the bressumer: one oak beam across the whole opening, proud of the face
  wallBox(b, 'beam', w, u - hL - 0.09, u + hR + 0.09, hO, hO + BREAST_LINT,
    -0.01, P + 0.07, oak());
  // mantel shelf over it
  const yM = hO + BREAST_LINT;
  wallBox(b, 'stone', w, u - bL - 0.05, u + bR + 0.05, yM, yM + 0.085,
    0, P + 0.13, stone());

  /* The gathering: courses that corbel in as they rise, so the flue visibly
   * narrows into the ceiling instead of stopping as a slab. */
  const yTop = room.openToRoof ? Math.min(H - 0.26, yM + 1.75) : H;
  const gh = yTop - (yM + 0.085);
  if (gh > 0.12) {
    const n = Math.max(2, Math.min(BREAST_GATHER, Math.round(gh / 0.24)));
    const wTop = Math.max(0.42, fw * 0.34);
    const dTop = 0.16;
    for (let i = 0; i < n; i++) {
      const t = (i + 1) / n;                        // 1 = the topmost course
      const v0 = yM + 0.085 + (gh * i) / n;
      const v1 = yM + 0.085 + (gh * (i + 1)) / n;
      const kL = bL + (Math.min(wTop, bL) - bL) * t;
      const kR = bR + (Math.min(wTop, bR) - bR) * t;
      const dK = P + (dTop - P) * t;
      wallBox(b, 'stone', w, u - kL, u + kR, v0, v1, 0, dK, stone());
    }
  }
  // A soot-dark throat in the head of the opening: looking up from the fire you
  // see the flue gather, not the flat underside of a beam.
  {
    const tw = Math.min(Math.min(hL, hR) - 0.08, fw * 0.5);
    if (tw > 0.2) {
      wallBox(b, 'stone', w, u - tw, u + tw, hO - 0.11, hO, 0.05, P - 0.05,
        { r: 0.17, g: 0.155, b: 0.145 });
    }
  }

  // Colliders: the piers only. The firebox stays open so the player can walk up
  // to the fire, and the mass above it is out of reach.
  const pierCol = (uA, uB) => {
    const r = wallRectLocal(w, uA, uB, 0, P);
    const cx = (r.x0 + r.x1) / 2, cz = (r.z0 + r.z1) / 2;
    colliders.push(boxCollider(frame.toWorld(cx, hO / 2, cz),
      [Math.max(0.03, (r.x1 - r.x0) / 2), hO / 2, Math.max(0.03, (r.z1 - r.z0) / 2)],
      ctx.worldQ, { tag: 'interior-hearth' }));
  };
  const piers = [];
  if (pier[0] > 0) { pierCol(u - bL, u - hL); piers.push(wallRectLocal(w, u - bL, u - hL, 0, P + 0.13)); }
  if (pier[1] > 0) { pierCol(u + hR, u + bR); piers.push(wallRectLocal(w, u + hR, u + bR, 0, P + 0.13)); }

  /* The keep-out is the PIERS, not the whole breast. furnishings.js claims every
   * keep-out before it picks its hearth site and then nudges the fire off
   * anything it hits — a box across the opening would walk the fire out of the
   * fireplace we just built round it. The firebox stays unclaimed, and their own
   * hearth claims it a moment later. */
  return { top: yTop, u, fw, wall: w.side, piers };
}

/* -------------------------------------------------------------------------- */
/* Stairs                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * Plan a straight flight from `lower` up to `upper`, hugging whichever wall
 * leaves the room usable. Returns null when it cannot be made to fit, so a bad
 * RoomSpec loses its stair rather than the whole interior.
 */
function planStair(lower, upper) {
  const rise = upper.floorY - lower.floorY;
  if (!(rise > 1.9) || rise > 5.0) return null;

  let n = Math.max(2, Math.round(rise / RISER_TARGET));
  let riser = rise / n;
  for (let guard = 0; guard < 6 && (riser < RISER_MIN || riser > RISER_MAX); guard++) {
    n += riser > RISER_MAX ? 1 : -1;
    n = Math.max(2, n);
    riser = rise / n;
  }
  const treads = n - 1;                        // the upper floor is the last tread

  const hw = lower.width / 2 - GAP - LIN_T - STUD_P;
  const hd = lower.depth / 2 - GAP - LIN_T - STUD_P;

  // Which way does the flight run? Across the depth (against a side wall) is the
  // first choice — it also lands the stair head near the ridge, where an open
  // roof is highest. Along the width against the back wall is the fallback.
  const cands = [
    { axis: 'z', avail: 2 * hd - 1.0, cross: hw },
    { axis: 'x', avail: 2 * hw - 1.0, cross: hd },
  ];
  let plan = null;
  for (const c of cands) {
    let going = GOING;
    let run = treads * going;
    if (run > c.avail) {
      going = Math.max(GOING_MIN, c.avail / treads);
      run = treads * going;
    }
    if (run > c.avail || c.cross < STAIR_W + 0.5) continue;
    plan = { axis: c.axis, going, run, cross: c.cross };
    break;
  }
  if (!plan) return null;

  // Put the flight on the side of the room the front door is not on.
  let doorU = 0;
  for (const op of lower.openings || []) {
    if (op && op.kind === 'door') { doorU = num(op.lx) ? op.lx : 0; break; }
  }
  const sgn = plan.axis === 'z' ? (doorU <= 0 ? 1 : -1) : -1;   // 'x' hugs the back wall

  // s runs up the flight, t across it. Origin: the bottom riser.
  const startInset = plan.axis === 'z' ? Math.min(1.25, 2 * hd - plan.run - 0.35) : Math.min(1.0, 2 * hw - plan.run - 0.3);
  const sStart = (plan.axis === 'z' ? hd : hw) - Math.max(0.3, startInset);
  const tOuter = sgn * (plan.axis === 'z' ? hw : hd);
  const tInner = tOuter - sgn * STAIR_W;

  return {
    axis: plan.axis, sgn, n, riser, going: plan.going, run: plan.run,
    sStart, tOuter, tInner, rise,
    /** flight-space (s,t) -> room-local (x,z) */
    toLocal(s, t) {
      return plan.axis === 'z' ? [t, sStart - s] : [sStart - s, t];
    },
    /** the stairwell footprint in the LOWER room's local space */
    holeLower: (() => {
      const t0 = Math.min(tOuter, tInner) - 0.06, t1 = Math.max(tOuter, tInner) + 0.06;
      const s0 = -0.12, s1 = plan.run + 0.02;
      const a0 = sStart - s1, a1 = sStart - s0;
      return plan.axis === 'z'
        ? { x0: t0, x1: t1, z0: a0, z1: a1 }
        : { x0: a0, x1: a1, z0: t0, z1: t1 };
    })(),
  };
}

/** Build the flight into the lower room's bucket. */
function buildStair(b, room, ctx, st) {
  const { rng, colliders, frame } = ctx;
  const oak = () => tintOf(rng, 0.07);
  const isZ = st.axis === 'z';
  const openSide = st.tInner;                       // the side away from the wall

  const boxAt = (s0, s1, y0, y1, t0, t1, key, tint) => {
    const [ax0, az0] = st.toLocal(s0, t0);
    const [ax1, az1] = st.toLocal(s1, t1);
    localBox(b, key, (ax0 + ax1) / 2, (y0 + y1) / 2, (az0 + az1) / 2,
      Math.max(1e-3, Math.abs(ax1 - ax0)), Math.max(1e-3, y1 - y0),
      Math.max(1e-3, Math.abs(az1 - az0)), tint);
  };

  for (let i = 1; i <= st.n; i++) {
    const top = i * st.riser;
    const s0 = (i - 1) * st.going;
    const s1 = i === st.n ? st.run : s0 + st.going;
    // tread, with a nosing over the riser below
    boxAt(s0 - 0.035, s1 + 0.02, top - 0.05, top, st.tInner, st.tOuter, 'joinery', oak());
    // riser board
    boxAt(s0 - 0.015, s0 + 0.015, top - st.riser, top - 0.05, st.tInner, st.tOuter, 'joinery', oak());
    // boxed string on the open side: closes the flank of every step
    boxAt(s0 - 0.035, s1 + 0.02, 0, top - 0.05, openSide, openSide + (st.tOuter > st.tInner ? -0.045 : 0.045), 'joinery', oak());

    // one solid collider per step, from the floor to the step's top: autostep
    // (0.45 m) will walk these, but only because each one is a real box.
    const [cx0, cz0] = st.toLocal(s0 - 0.035, st.tInner);
    const [cx1, cz1] = st.toLocal(s1 + 0.02, st.tOuter);
    colliders.push(boxCollider(
      frame.toWorld((cx0 + cx1) / 2, top / 2, (cz0 + cz1) / 2),
      [Math.max(0.02, Math.abs(cx1 - cx0) / 2), top / 2, Math.max(0.02, Math.abs(cz1 - cz0) / 2)],
      ctx.worldQ, { tag: 'interior-stair' }));
  }

  // newel post at the foot, on the open side
  {
    const [nx, nz] = st.toLocal(-0.05, openSide);
    localBox(b, 'beam', nx, NEWEL_H / 2, nz, 0.115, NEWEL_H, 0.115, oak());
  }
  // handrail following the pitch, plus balusters
  {
    const ang = Math.atan2(st.rise, st.run);
    const len = Math.hypot(st.run, st.rise);
    const [mx, mz] = st.toLocal(st.run / 2, openSide);
    const y = RAIL_H + st.rise / 2 - 0.02;
    // The flight climbs toward -Z (or -X), so the -axis end is the high one.
    // rotateX(+a) tips +Z down; rotateZ(+a) tips +X up — hence the sign split.
    if (isZ) tiltedBoxX(b, 'joinery', mx, y, mz, 0.075, 0.06, len, ang, oak());
    else tiltedBoxZ(b, 'joinery', mx, y, mz, len, 0.06, 0.075, -ang, oak());
    const nBal = Math.max(2, Math.round(st.run / 0.42));
    for (let i = 1; i < nBal; i++) {
      const s = (st.run * i) / nBal;
      const yTop = RAIL_H + (st.rise * i) / nBal;
      const [bx, bz] = st.toLocal(s, openSide);
      localBox(b, 'joinery', bx, yTop / 2, bz, 0.042, yTop, 0.042, oak());
    }
  }
  // a wall string against the wall side, so the treads die into something
  {
    const ang = Math.atan2(st.rise, st.run);
    const len = Math.hypot(st.run, st.rise);
    const [mx, mz] = st.toLocal(st.run / 2, st.tOuter);
    if (isZ) tiltedBoxX(b, 'beam', mx, st.rise / 2 + 0.12, mz, 0.05, 0.3, len, ang, oak());
    else tiltedBoxZ(b, 'beam', mx, st.rise / 2 + 0.12, mz, len, 0.3, 0.05, -ang, oak());
  }
  return { steps: st.n, riser: st.riser, going: st.going };
}

/**
 * Trim the stairwell in the upper floor: a trimmer beam round the well and a
 * balustrade on the two open sides. The stair head is deliberately left clear.
 */
function buildWellTrim(b, room, ctx, hole, st) {
  const { rng, colliders, frame } = ctx;
  const oak = () => tintOf(rng, 0.07);
  const jd = 0.19;
  // trimmers, hanging below the boards like the joists they replace
  localBox(b, 'beam', (hole.x0 + hole.x1) / 2, -jd / 2, hole.z0 - 0.05,
    hole.x1 - hole.x0, jd, 0.1, oak());
  localBox(b, 'beam', (hole.x0 + hole.x1) / 2, -jd / 2, hole.z1 + 0.05,
    hole.x1 - hole.x0, jd, 0.1, oak());
  localBox(b, 'beam', hole.x0 - 0.05, -jd / 2, (hole.z0 + hole.z1) / 2,
    0.1, jd, hole.z1 - hole.z0, oak());
  localBox(b, 'beam', hole.x1 + 0.05, -jd / 2, (hole.z0 + hole.z1) / 2,
    0.1, jd, hole.z1 - hole.z0, oak());

  // Balustrade. The stair head is the edge you arrive at, so it stays open; the
  // other exposed edges get a rail the player cannot walk through.
  const isZ = st.axis === 'z';
  const hwl = room.width / 2 - GAP - LIN_T;
  const hdl = room.depth / 2 - GAP - LIN_T;
  const edges = [];
  if (isZ) {
    // The flight climbs toward -Z, so the head — the edge you step off onto the
    // landing — is at hole.z0 and must stay clear. Rail the other three.
    const xIn = Math.abs(hole.x0) < Math.abs(hole.x1) ? hole.x0 : hole.x1;
    const xOut = xIn === hole.x0 ? hole.x1 : hole.x0;
    edges.push({ x0: xIn, x1: xIn, z0: hole.z0, z1: hole.z1 });
    edges.push({ x0: hole.x0, x1: hole.x1, z0: hole.z1, z1: hole.z1 });
    // A jettied upper storey is 0.45 m wider than the room the stair sits in, so
    // there can be a strip of floor on the far side of the well. Rail that too.
    if (hwl - Math.abs(xOut) > 0.3) edges.push({ x0: xOut, x1: xOut, z0: hole.z0, z1: hole.z1 });
  } else {
    const zIn = Math.abs(hole.z0) < Math.abs(hole.z1) ? hole.z0 : hole.z1;
    const zOut = zIn === hole.z0 ? hole.z1 : hole.z0;
    edges.push({ x0: hole.x0, x1: hole.x1, z0: zIn, z1: zIn });
    edges.push({ x0: hole.x1, x1: hole.x1, z0: hole.z0, z1: hole.z1 });
    if (hdl - Math.abs(zOut) > 0.3) edges.push({ x0: hole.x0, x1: hole.x1, z0: zOut, z1: zOut });
  }

  for (const e of edges) {
    const len = Math.hypot(e.x1 - e.x0, e.z1 - e.z0);
    if (len < 0.35) continue;
    const ux = (e.x1 - e.x0) / len, uz = (e.z1 - e.z0) / len;
    const cx = (e.x0 + e.x1) / 2, cz = (e.z0 + e.z1) / 2;
    const thick = 0.075;
    // top rail
    localBox(b, 'joinery', cx, RAIL_H, cz,
      ux ? len : thick, 0.07, uz ? len : thick, oak());
    // bottom rail + balusters
    localBox(b, 'joinery', cx, 0.14, cz, ux ? len : thick, 0.08, uz ? len : thick, oak());
    const nB = Math.max(2, Math.round(len / 0.16));
    for (let i = 0; i <= nB; i++) {
      const t = i / nB;
      const x = e.x0 + (e.x1 - e.x0) * t, z = e.z0 + (e.z1 - e.z0) * t;
      localBox(b, 'joinery', x, RAIL_H / 2 + 0.09, z, 0.036, RAIL_H - 0.18, 0.036, oak());
    }
    // newels at the ends
    for (const t of [0, 1]) {
      const x = e.x0 + (e.x1 - e.x0) * t, z = e.z0 + (e.z1 - e.z0) * t;
      localBox(b, 'beam', x, (RAIL_H + 0.08) / 2, z, 0.1, RAIL_H + 0.08, 0.1, oak());
    }
    // one collider per run: you cannot stroll off the landing
    colliders.push(boxCollider(frame.toWorld(cx, (RAIL_H + 0.08) / 2, cz),
      [(ux ? len / 2 : 0.05), (RAIL_H + 0.08) / 2, (uz ? len / 2 : 0.05)],
      ctx.worldQ, { tag: 'interior-balustrade' }));
  }
}

/* -------------------------------------------------------------------------- */
/* Fallback rooms                                                              */
/* -------------------------------------------------------------------------- */

/**
 * If the shell has not published `chunk.interiors` yet, derive a conservative
 * set of rooms from PLOTS so the buildings are still walk-in-able.
 *
 * These rooms are marked `_derived`, and a derived room gets STRUCTURE ONLY:
 * floors, joists, ceilings, stairs, upper floors and thresholds. No wall lining
 * and no roof underside, because without the shell's real opening list a lining
 * would plaster over a window from the inside — a visible regression on the
 * outside of the building, which is worse than a bare interior.
 */
function deriveRooms(terrain) {
  const rooms = [];
  const groundY = (x, z) => {
    const f = terrain && (terrain.heightAt || terrain.groundY);
    if (typeof f === 'function') {
      try { const y = f.call(terrain, x, z); if (num(y)) return y; } catch { /* fall through */ }
    }
    return 0;
  };
  for (const plot of PLOTS) {
    const baseY = groundY(plot.position[0], plot.position[2]);
    const storeys = Math.max(1, plot.storeys | 0);
    const sh = plot.storeyHeight;
    const uses = INTERIOR_USES[plot.id] || [];
    const stepCount = plot.steps ? Math.max(0, plot.steps.count | 0) : 0;
    const stepRise = stepCount ? Math.min(0.155, 0.42 / stepCount) : 0;
    const doorBase = stepCount * stepRise;
    const style = FB_DOOR_W[plot.door.style] ? plot.door.style : 'plank';
    const dw = FB_DOOR_W[style];
    const dh = Math.min(FB_DOOR_H[style] + (plot.door.arched ? dw * 0.30 : 0), sh - 0.36 - doorBase);
    const jet = plot.jetty && storeys > 1 ? FB_JETTY : 0;
    const pitch = (plot.roofPitch || 52) * Math.PI / 180;

    for (let s = 0; s < storeys; s++) {
      const grew = s > 0 ? jet : 0;
      const width = plot.width + 2 * grew - 2 * FB_WALL_T;
      const depth = plot.depth + grew - 2 * FB_WALL_T;
      const zOff = s > 0 ? grew / 2 : 0;         // the jetty only overhangs the front
      const floorY = baseY + (s === 0 ? doorBase : s * sh + 0.25);
      const ceilingY = baseY + (s + 1) * sh;
      const cos = Math.cos(plot.rotation), sin = Math.sin(plot.rotation);
      const centre = [
        plot.position[0] + zOff * sin,
        floorY,
        plot.position[2] + zOff * cos,
      ];
      const top = s === storeys - 1;
      const room = {
        plotId: plot.id, storey: s, floorY,
        ceilingY: top ? Math.max(ceilingY, floorY + 2.1) : ceilingY,
        width, depth, centre, rotation: plot.rotation,
        wallThickness: FB_WALL_T,
        openings: [],
        openToRoof: false,
        use: uses[s] || (s === 0 ? 'hall' : 'store'),
        _derived: true,
      };
      if (top) room.ridgeY = room.ceilingY + (depth / 2) * Math.tan(pitch);
      if (s === 0) {
        const lx = THREE.MathUtils.clamp(plot.door.x, -width / 2 + dw / 2, width / 2 - dw / 2);
        room.openings.push({
          kind: 'door', wall: 'front', lx, sillY: baseY + doorBase,
          width: dw, height: Math.max(2.0, dh), primary: true,
          centreWorld: [
            plot.position[0] + lx * cos + (depth / 2) * sin,
            baseY + doorBase + dh / 2,
            plot.position[2] - lx * sin + (depth / 2) * cos,
          ],
          inward: [-sin, 0, -cos],
        });
      }
      rooms.push(room);
    }
  }
  return rooms;
}

/* -------------------------------------------------------------------------- */
/* Factory                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * @param {Object} o
 * @param {import('../contracts.js').MaterialLibrary} o.materials
 * @param {Object} o.buildings   the buildings WorldChunk (reads `.interiors`)
 * @param {Object} [o.terrain]
 * @param {Object} o.quality
 * @returns {import('../contracts.js').WorldChunk & {
 *   rooms: Array, roomAt: Function, roomsFor: Function, keepOuts: Array }}
 */
export function createInteriors({ materials, buildings, terrain, quality } = {}) {
  const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const q = quality || {};
  const detail = q.geometryDetail ?? 1;

  const group = new THREE.Group();
  group.name = 'interiors';
  const colliders = [];
  const interactables = [];
  const lightAnchors = [];
  /** Volumes furnishings.js must not put furniture in. See the contract below. */
  const keepOuts = [];
  const resolve = makeMatResolver(materials);

  const stats = {
    rooms: 0, buildings: 0, meshes: 0, triangles: 0, colliders: 0,
    stairs: 0, thresholds: 0, breasts: 0, keepOuts: 0, rejected: 0,
    derived: false, visible: 0, visibleMeshes: 0, visibleTriangles: 0,
    buildMs: 0, warnings: 0,
  };

  /* -- the handoff ---------------------------------------------------------- */
  let raw = buildings && buildings.interiors && buildings.interiors.rooms;
  let derived = false;
  if (!Array.isArray(raw) || raw.length === 0) {
    console.warn('[interiors] buildings.interiors.rooms is missing or empty — deriving structure-only rooms from PLOTS');
    raw = deriveRooms(terrain);
    derived = true;
    stats.derived = true;
  }

  const san = sanitizeRooms(raw);
  stats.rejected = san.rejected;
  if (san.rejected) console.warn(`[interiors] ${san.rejected} RoomSpec(s) rejected as unusable`);
  const rooms = san.rooms;
  if (!rooms.length) {
    console.warn('[interiors] no usable rooms — the interiors chunk is empty');
    return emptyChunk(group, stats);
  }

  const plotById = new Map(PLOTS.map((p) => [p.id, p]));

  /* -- group rooms by plot -------------------------------------------------- */
  const byPlot = new Map();
  for (const r of rooms) {
    const id = r.plotId || 'unknown';
    let arr = byPlot.get(id);
    if (!arr) byPlot.set(id, arr = []);
    arr.push(r);
  }
  for (const arr of byPlot.values()) arr.sort((a, b) => (a.storey | 0) - (b.storey | 0));

  /* -- build ---------------------------------------------------------------- */
  /** @type {Array<{id:string, group:THREE.Group, rooms:Array, door:number[], visible:boolean, dist:number}>} */
  const units = [];

  for (const [plotId, list] of byPlot) {
    const plot = plotById.get(plotId) || null;
    const unitGroup = new THREE.Group();
    unitGroup.name = `interior:${plotId}`;
    unitGroup.visible = false;
    unitGroup.matrixAutoUpdate = false;

    // plan the stairs first: an upper floor needs to know where its well is
    const stairPlans = [];
    for (let i = 0; i < list.length - 1; i++) {
      const st = planStair(list[i], list[i + 1]);
      if (!st && list.length > 1) {
        console.warn(`[interiors] ${plotId}: no stair would fit between storey ${i} and ${i + 1}`);
        stats.warnings++;
      }
      stairPlans.push(st);
    }

    let doorWorld = null;
    let unitMeshes = 0, unitTris = 0;

    for (let i = 0; i < list.length; i++) {
      const room = list[i];
      const rng = new Rng((hashSeed(`${plotId}:${room.storey | 0}`) ^ ((plot && plot.seed) || 1337)) >>> 0);
      const frame = frameOf(room);
      const worldQ = quatY(room.rotation);
      const b = new Builder();
      const ctx = { rng, colliders, frame, worldQ, detail, quality: q };
      let warned = 0;
      const warn = (msg) => {
        if (warned++ < 2) console.warn(`[interiors] ${plotId} storey ${room.storey | 0}: ${msg}`);
        stats.warnings++;
      };

      const walls = wallsOf(room);
      placeOpenings(room, walls, frame, warn);

      // hole in THIS room's floor, converted from the flight below
      let hole = null;
      const below = stairPlans[i - 1];
      if (below && below.holeLower) {
        const lower = list[i - 1];
        const dx = room.centre[0] - lower.centre[0], dz = room.centre[2] - lower.centre[2];
        const cos = Math.cos(room.rotation), sin = Math.sin(room.rotation);
        const offX = dx * cos - dz * sin;        // lower-local -> this-local shift
        const offZ = dx * sin + dz * cos;
        const h = below.holeLower;
        hole = { x0: h.x0 - offX, x1: h.x1 - offX, z0: h.z0 - offZ, z1: h.z1 - offZ };
        const hwc = room.width / 2 - 0.12, hdc = room.depth / 2 - 0.12;
        hole.x0 = THREE.MathUtils.clamp(hole.x0, -hwc, hwc);
        hole.x1 = THREE.MathUtils.clamp(hole.x1, -hwc, hwc);
        hole.z0 = THREE.MathUtils.clamp(hole.z0, -hdc, hdc);
        hole.z1 = THREE.MathUtils.clamp(hole.z1, -hdc, hdc);
        if (hole.x1 - hole.x0 < 0.6 || hole.z1 - hole.z0 < 0.6) hole = null;
      }

      const use = room.use || (INTERIOR_USES[plotId] || [])[room.storey | 0] || 'hall';
      const flags = (room.storey | 0) === 0 && FLAG_USES.has(use);

      buildFloor(b, room, ctx, { hole, flags });

      if (!room._derived) buildLining(b, room, walls, ctx);

      // the flight that starts in this room needs a well through this ceiling
      const up = stairPlans[i];
      if (room.openToRoof && !room._derived) {
        buildOpenRoof(b, room, ctx, plot);
      } else {
        const next = list[i + 1];
        buildFlatCeiling(b, room, ctx, next ? next.floorY : null, up ? up.holeLower : null);
      }

      stats.thresholds += buildThresholds(b, room, walls, ctx);

      if (up) { buildStair(b, room, ctx, up); stats.stairs++; }
      if (below && hole) buildWellTrim(b, room, ctx, hole, below);

      /* -- keep-out volumes -------------------------------------------------
       * furnishings.js was guessing where the staircases and the chimney breast
       * are (roomsWithGuessedStair 14, roomsWithPublishedKeepOuts 0) and putting
       * furniture through a flight of stairs. We know exactly where they are.
       * Every entry carries BOTH the documented world box (centre / halfExtents
       * / rotation) and the room-local rect the consumer reads, so it does not
       * matter which of the two a reader was written against. */
      const roomRects = [];
      const keep = (reason, rect, y0, y1) => {
        const rx = THREE.MathUtils.clamp(rect.x0, -room.width / 2, room.width / 2);
        const rX = THREE.MathUtils.clamp(rect.x1, -room.width / 2, room.width / 2);
        const rz = THREE.MathUtils.clamp(rect.z0, -room.depth / 2, room.depth / 2);
        const rZ = THREE.MathUtils.clamp(rect.z1, -room.depth / 2, room.depth / 2);
        if (rX - rx < 0.05 || rZ - rz < 0.05) return null;
        const cx = (rx + rX) / 2, cz = (rz + rZ) / 2;
        const out = {
          plotId, storey: room.storey | 0, reason,
          centre: frame.toWorld(cx, (y0 + y1) / 2, cz),
          halfExtents: [(rX - rx) / 2, Math.max(0.05, (y1 - y0) / 2), (rZ - rz) / 2],
          rotation: room.rotation,
          // the same box in room-local metres, in all three shapes a reader
          // might expect (rect / centre+half / min-max)
          x0: rx, z0: rz, x1: rX, z1: rZ,
          lx: cx, lz: cz, halfW: (rX - rx) / 2, halfD: (rZ - rz) / 2,
          min: [rx, rz], max: [rX, rZ],
        };
        keepOuts.push(out);
        roomRects.push({ x0: rx, x1: rX, z0: rz, z1: rZ });
        return out;
      };

      // the swept volume of the flight that starts here, plus its foot landing
      if (up && up.holeLower) {
        const h = up.holeLower;
        const r = { x0: h.x0 - 0.06, x1: h.x1 + 0.06, z0: h.z0 - 0.06, z1: h.z1 + 0.06 };
        // the flight climbs toward -Z (axis 'z') or -X, so the foot is the +end
        if (up.axis === 'z') r.z1 += 0.78; else r.x1 += 0.78;
        keep('stair', r, 0, up.rise);
      }
      // the well in this floor, plus the landing you step off onto
      if (below && hole) {
        const r = { x0: hole.x0 - 0.1, x1: hole.x1 + 0.1, z0: hole.z0 - 0.1, z1: hole.z1 + 0.1 };
        if (below.axis === 'z') r.z0 -= 0.72; else r.x0 -= 0.72;
        keep('stairwell', r, 0, Math.max(1.2, RAIL_H + 0.1));
      }
      // a clear box in front of every external door
      for (const w of walls) {
        for (const op of w.openings) {
          if (op.kind !== 'door') continue;
          keep('door-swing', wallRectLocal(w, op.u - op.width / 2 - 0.16,
            op.u + op.width / 2 + 0.16, 0, 1.15), 0, Math.max(1.2, op.height));
        }
      }
      // the chimney breast — built here, so nobody has to guess at it either
      if ((room.storey | 0) === 0 && !room._derived && use !== 'bakery') {
        const place = hearthPlacement(room, walls, frame, plot, use, roomRects);
        const breast = place ? buildChimneyBreast(b, room, ctx, place) : null;
        if (breast) {
          for (const r of breast.piers) keep('hearth', r, 0, breast.top);
          stats.breasts++;
        }
      }

      const roomGroup = new THREE.Group();
      roomGroup.name = `room:${plotId}:${room.storey | 0}`;
      roomGroup.position.set(room.centre[0], room.floorY, room.centre[2]);
      roomGroup.rotation.y = room.rotation;
      roomGroup.updateMatrix();
      roomGroup.matrixAutoUpdate = false;

      const fin = b.finish(resolve, roomGroup.name);
      for (const m of fin.meshes) roomGroup.add(m);
      stats.meshes += fin.meshes.length;
      stats.triangles += fin.tris;
      unitMeshes += fin.meshes.length;
      unitTris += fin.tris;
      unitGroup.add(roomGroup);

      room._group = roomGroup;
      room._use = use;
      stats.rooms++;

      /* Deliberately NO light anchor per room. lighting.js maps an unknown kind
       * onto `lantern` and ranks every anchor for a three-slot pool, so eighteen
       * zero-intensity "fill" anchors would do nothing but crowd out the hearths
       * furnishings.js emits. The interior's contribution to lighting is
       * `roomAt` (main.js hands it to lighting.setRoomResolver) and `rooms`. */

      // remember the primary door for the visibility test
      for (const w of walls) {
        for (const op of w.openings) {
          if (op.kind !== 'door') continue;
          const nIn = 0.35;
          const x = w.ox + w.ux * op.u + w.nx * nIn;
          const z = w.oz + w.uz * op.u + w.nz * nIn;
          const p = frame.toWorld(x, 1.0, z);
          if (!doorWorld || op.primary) doorWorld = p;
        }
      }
    }

    if (!doorWorld) {
      const r0 = list[0];
      doorWorld = [r0.centre[0], r0.floorY + 1.0, r0.centre[2]];
    }

    unitGroup.updateMatrix();
    group.add(unitGroup);
    units.push({
      id: plotId, group: unitGroup, rooms: list,
      dx: doorWorld[0], dy: doorWorld[1], dz: doorWorld[2],
      visible: false, dist: Infinity,
      meshes: unitMeshes, tris: unitTris,
    });
    stats.buildings++;
  }

  stats.colliders = colliders.length;
  stats.keepOuts = keepOuts.length;

  /* -- roomAt probes: flat numbers, so the query allocates nothing ---------- */
  const P_STRIDE = 8;
  const probe = new Float64Array(rooms.length * P_STRIDE);
  for (let i = 0; i < rooms.length; i++) {
    const r = rooms[i];
    const o = i * P_STRIDE;
    probe[o] = r.centre[0];
    probe[o + 1] = r.centre[2];
    probe[o + 2] = Math.cos(r.rotation);
    probe[o + 3] = Math.sin(r.rotation);
    probe[o + 4] = r.width / 2;
    probe[o + 5] = r.depth / 2;
    // The bottom margin lets a crouched player still be "in" the room; the top
    // one is deliberately tight so the ground floor does not claim a point that
    // belongs to the storey above (the two overlap inside the floor structure).
    probe[o + 6] = r.floorY - 0.45;
    probe[o + 7] = (r.openToRoof && num(r.ridgeY) ? Math.max(r.ridgeY, r.ceilingY) : r.ceilingY)
      + (r.openToRoof ? 0.15 : 0.06);
  }

  /**
   * Which room contains a point, or null. Allocation-free and cheap enough to
   * call every frame from the lighting and furnishing streams.
   */
  function roomAt(x, y, z) {
    for (let i = 0; i < rooms.length; i++) {
      const o = i * P_STRIDE;
      if (y < probe[o + 6] || y > probe[o + 7]) continue;
      const dx = x - probe[o], dz = z - probe[o + 1];
      const lx = dx * probe[o + 2] - dz * probe[o + 3];
      if (lx < -probe[o + 4] || lx > probe[o + 4]) continue;
      const lz = dx * probe[o + 3] + dz * probe[o + 2];
      if (lz < -probe[o + 5] || lz > probe[o + 5]) continue;
      return rooms[i];
    }
    return null;
  }

  function roomsFor(plotId) { return byPlot.get(plotId) || []; }

  /* -- visibility ----------------------------------------------------------- */
  let sinceCheck = 1e3;
  let lastX = 1e9, lastY = 1e9, lastZ = 1e9;
  const SHOW2 = SHOW_R * SHOW_R, HIDE2 = HIDE_R * HIDE_R;

  function refresh(px, py, pz) {
    const inside = roomAt(px, py, pz);
    const insideId = inside ? inside.plotId : null;
    let shown = 0;
    for (const u of units) {
      const dx = u.dx - px, dy = (u.dy - py) * 0.55, dz = u.dz - pz;
      const d2 = dx * dx + dy * dy + dz * dz;
      u.dist = d2;
      const isIn = u.id === insideId;
      if (isIn || d2 < SHOW2) u.visible = true;
      else if (d2 > HIDE2) u.visible = false;
      if (u.visible) shown++;
    }
    // Hard cap. The plots sit 16 m apart on the ring so this should never bite,
    // but a budget you only hope you are inside is not a budget.
    if (shown > MAX_VISIBLE) {
      const sorted = units.filter((u) => u.visible).sort((a, b) => a.dist - b.dist);
      for (let i = MAX_VISIBLE; i < sorted.length; i++) {
        if (sorted[i].id !== insideId) sorted[i].visible = false;
      }
      shown = 0;
      for (const u of units) if (u.visible) shown++;
    }
    let vm = 0, vt = 0;
    for (const u of units) {
      if (u.group.visible !== u.visible) u.group.visible = u.visible;
      if (u.visible) { vm += u.meshes; vt += u.tris; }
    }
    /* What this stream is actually costing the frame. A Group with visible=false
     * is skipped whole by projectObject, so a hidden interior submits nothing —
     * these two numbers are the honest resident cost, and outdoors they are 0. */
    stats.visible = shown;
    stats.visibleMeshes = vm;
    stats.visibleTriangles = vt;
  }

  function update(dt, ctx) {
    const p = ctx && ctx.playerPosition;
    if (!p) return;
    sinceCheck += dt || 0;
    const moved = Math.abs(p.x - lastX) + Math.abs(p.y - lastY) + Math.abs(p.z - lastZ);
    if (sinceCheck < CHECK_INTERVAL && moved < CHECK_MOVE) return;
    sinceCheck = 0;
    lastX = p.x; lastY = p.y; lastZ = p.z;
    refresh(p.x, p.y, p.z);
  }

  function dispose() {
    group.traverse((o) => { if (o.isMesh) o.geometry?.dispose?.(); });
    group.clear();
    units.length = 0;
  }

  stats.buildMs = Math.round(((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0) * 10) / 10;
  console.info('[interiors]', stats);

  /**
   * Volumes nothing else may put anything in.
   *
   * Each entry is the same box twice over: `centre`/`halfExtents`/`rotation` is
   * the world-space oriented box (rotation is the room's, so the box is axis
   * aligned in room space), and `x0/z0/x1/z1` — plus `lx/lz/halfW/halfD` and
   * `min/max` — is the same footprint in room-local metres. `reason` is one of
   * 'stair' | 'stairwell' | 'hearth' | 'threshold' | 'door-swing'.
   *
   * @type {Array<{plotId:string, storey:number, reason:string,
   *   centre:number[], halfExtents:number[], rotation:number,
   *   x0:number, z0:number, x1:number, z1:number}>}
   */
  const keepOutList = keepOuts;

  /* Deliberately only the flat `keepOuts` array, and no `keepOutsFor()` helper:
   * furnishings.js reads BOTH if both exist, which would claim every box twice
   * and double its own keep-out diagnostics. One list, one reading. */

  return {
    group, colliders, interactables, lightAnchors,
    rooms, roomAt, roomsFor,
    keepOuts: keepOutList,
    update, dispose, stats,
  };
}

function emptyChunk(group, stats) {
  return {
    group,
    colliders: [],
    interactables: [],
    lightAnchors: [],
    rooms: [],
    keepOuts: [],
    roomAt() { return null; },
    roomsFor() { return []; },
    update() { },
    dispose() { group.clear(); },
    stats,
  };
}
