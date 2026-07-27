/**
 * ============================================================================
 *  THE VILLAGE PLAN  —  single source of truth for *where things are*
 * ============================================================================
 * Terrain, buildings, props, vegetation and lighting all read this file. It is
 * authored (not random) so the composition matches the reference image: a
 * cobbled plaza with a round stone well slightly right of centre, a big
 * half-timbered cottage on the near left, a row of thatched cottages on the
 * right, more cottages across the back, green hills and conifers beyond.
 *
 * Convention
 *   +X east, +Z south, +Y up. The player enters from the south (+Z) looking
 *   north. A building's LOCAL +Z is its front (facing the plaza), local +X is
 *   its long axis. `width` runs along local X, `depth` along local Z.
 * ============================================================================
 */

import * as THREE from 'three';

const D = Math.PI / 180;

/** Position on a ring around the plaza. Angles in degrees, 0 = east, 90 = south. */
function ring(angleDeg, radius) {
  const a = angleDeg * D;
  return [Math.cos(a) * radius, 0, Math.sin(a) * radius];
}

/** Y rotation that makes local +Z point at the plaza centre, plus a skew. */
function faceCentre(pos, skewDeg = 0) {
  return Math.atan2(-pos[0], -pos[2]) + skewDeg * D;
}

/* -------------------------------------------------------------------------- */
/* Plaza                                                                       */
/* -------------------------------------------------------------------------- */

/** Organic radius of the cobbled area at a given bearing. */
export function plazaRadiusAt(angleRad) {
  return (
    26.0 +
    2.4 * Math.sin(3 * angleRad + 0.6) +
    1.5 * Math.sin(5 * angleRad + 2.1) -
    1.1 * Math.cos(2 * angleRad - 0.4) +
    0.8 * Math.sin(7 * angleRad + 1.2)
  );
}

/** Closed polygon of the cobbled plaza, 72 points. */
export const PLAZA_POLY = (() => {
  const pts = [];
  for (let i = 0; i < 72; i++) {
    const a = (i / 72) * Math.PI * 2;
    const r = plazaRadiusAt(a);
    pts.push(new THREE.Vector2(Math.cos(a) * r, Math.sin(a) * r));
  }
  return pts;
})();

/** Roads leaving the plaza. Cobbles fade into packed dirt along these. */
export const ROADS = [
  { name: 'south-gate', points: [[1, 24], [2.5, 42], [6, 62], [10, 92]], width: 7.5, surface: 'cobble' },
  { name: 'east-lane', points: [[24, 8], [40, 2], [58, -6], [86, -14]], width: 5.5, surface: 'dirt' },
  { name: 'west-lane', points: [[-25, 1], [-42, -3], [-64, -2], [-92, 4]], width: 5.0, surface: 'dirt' },
  { name: 'north-track', points: [[-2, -26], [-4, -44], [-2, -66], [3, -96]], width: 4.2, surface: 'dirt' },
];

/**
 * Where the player spawns and what they look at.
 *
 * Yaw convention: a camera at yaw θ faces (-sinθ, 0, -cosθ). So yaw 0 looks
 * along -Z — from the southern approach, straight up the road into the plaza.
 */
export const SPAWN = { position: [0, 0, 34], yaw: 0 };

/** Yaw that makes a camera at `from` look at `to`. */
export function yawTowards(from, to) {
  return Math.atan2(-(to[0] - from[0]), -(to[2] - from[2]));
}

/* -------------------------------------------------------------------------- */
/* The well                                                                    */
/* -------------------------------------------------------------------------- */

export const WELL = {
  position: [2.2, 0, 5.6],
  rotation: -0.32,
  /** Outer radius of the stone parapet. */
  outerRadius: 2.35,
  innerRadius: 1.62,
  wallHeight: 1.02,
  /** Two circular steps the parapet sits on. */
  base: [
    { radius: 4.15, height: 0.16 },
    { radius: 3.35, height: 0.17 },
  ],
  /** No roof frame in the reference image — an open well with a bucket winch. */
  winch: { height: 2.35, postOffset: 1.85, bucketDrop: 1.15 },
};

/* -------------------------------------------------------------------------- */
/* Buildings                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * @typedef {Object} Plot
 * @property {string} id
 * @property {string} name
 * @property {[number,number,number]} position   world, ground level
 * @property {number} rotation                   radians, local +Z faces plaza
 * @property {number} width   along local X
 * @property {number} depth   along local Z
 * @property {number} storeys
 * @property {number} storeyHeight
 * @property {'halfTimber'|'plaster'|'stoneBase'|'longRow'} style
 * @property {'thatch'|'thatchHip'|'thatchHalfHip'|'tile'} roof
 * @property {number} roofPitch    degrees
 * @property {number} eave         overhang in metres
 * @property {boolean} jetty       upper floor overhangs the lower
 * @property {Array} chimneys      {x,z,height,pots,style}
 * @property {Array} dormers       {x,width}
 * @property {Object|null} porch   {width,depth,height}
 * @property {Object|null} arcade  {bays,depth}  covered walkway along the front
 * @property {Object} door         {x,style,colour,arched}
 * @property {Object|null} steps   {width,count}
 * @property {number} ivy          0..1 coverage on the front
 * @property {boolean} flowerBoxes
 * @property {Object|null} sign    {text,offset}  hanging shop sign
 * @property {string} plasterTone  material key
 * @property {number} seed
 */

/** @type {Plot[]} */
export const PLOTS = [
  {
    id: 'weaver',
    name: "The Weaver's House",
    position: ring(140, 30.5),
    rotation: 0, // filled below
    skew: -6,
    width: 13.5, depth: 9.5,
    storeys: 2, storeyHeight: 2.95,
    style: 'halfTimber',
    roof: 'thatch', roofPitch: 53, eave: 1.05,
    jetty: true,
    chimneys: [
      { x: -5.1, z: -0.4, height: 5.4, pots: 2, style: 'brick', width: 1.5 },
      { x: 5.4, z: -2.6, height: 3.6, pots: 1, style: 'brick', width: 1.1 },
    ],
    dormers: [{ x: 2.6, width: 2.3 }],
    porch: null,
    arcade: null,
    door: { x: -4.4, style: 'plank', colour: 'teal', arched: false },
    steps: { width: 3.4, count: 3 },
    ivy: 0.72,
    flowerBoxes: true,
    sign: null,
    plasterTone: 'plaster',
    seed: 1101,
  },
  {
    id: 'stables',
    name: 'The Long Stable',
    position: ring(166, 33),
    rotation: 0, skew: 4,
    width: 16.5, depth: 8.0,
    storeys: 1, storeyHeight: 3.3,
    style: 'stoneBase',
    roof: 'thatchHalfHip', roofPitch: 48, eave: 1.15,
    jetty: false,
    chimneys: [{ x: -6.0, z: -1.5, height: 3.1, pots: 1, style: 'brick', width: 1.15 }],
    dormers: [],
    porch: null,
    arcade: null,
    door: { x: 3.2, style: 'barn', colour: 'wood', arched: true },
    steps: null,
    ivy: 0.3,
    flowerBoxes: false,
    sign: null,
    plasterTone: 'plasterWorn',
    seed: 2202,
  },
  {
    id: 'granary',
    name: 'The Granary',
    position: ring(196, 34),
    rotation: 0, skew: -3,
    width: 10.0, depth: 8.5,
    storeys: 2, storeyHeight: 2.7,
    style: 'halfTimber',
    roof: 'thatch', roofPitch: 55, eave: 0.95,
    jetty: true,
    chimneys: [{ x: -3.6, z: -1.0, height: 4.4, pots: 2, style: 'brick', width: 1.25 }],
    dormers: [],
    porch: null,
    arcade: null,
    door: { x: 0.6, style: 'plank', colour: 'wood', arched: true },
    steps: { width: 2.4, count: 2 },
    ivy: 0.15,
    flowerBoxes: true,
    sign: null,
    plasterTone: 'plaster',
    seed: 3303,
  },
  {
    id: 'chandler',
    name: "The Chandler's",
    position: ring(224, 33),
    rotation: 0, skew: 5,
    width: 11.5, depth: 9.0,
    storeys: 2, storeyHeight: 2.85,
    style: 'halfTimber',
    roof: 'thatch', roofPitch: 51, eave: 1.0,
    jetty: false,
    chimneys: [{ x: 4.4, z: -1.8, height: 4.6, pots: 2, style: 'brick', width: 1.3 }],
    dormers: [{ x: -1.8, width: 2.1 }],
    porch: { width: 2.9, depth: 1.5, height: 2.4 },
    arcade: null,
    door: { x: 0.0, style: 'plank', colour: 'wood', arched: true },
    steps: { width: 3.0, count: 2 },
    ivy: 0.45,
    flowerBoxes: true,
    sign: { text: 'candles', offset: 2.2 },
    plasterTone: 'plaster',
    seed: 4404,
  },
  {
    id: 'bakery',
    name: 'The Bakehouse',
    position: ring(252, 34),
    rotation: 0, skew: -4,
    width: 12.0, depth: 9.0,
    storeys: 1, storeyHeight: 3.4,
    style: 'plaster',
    roof: 'thatch', roofPitch: 56, eave: 1.25,
    jetty: false,
    chimneys: [{ x: -4.2, z: -2.2, height: 5.0, pots: 2, style: 'brick', width: 1.6 }],
    dormers: [],
    porch: { width: 3.6, depth: 1.9, height: 2.5 },
    arcade: null,
    door: { x: 0.0, style: 'plank', colour: 'wood', arched: true },
    steps: { width: 3.8, count: 2 },
    ivy: 0.35,
    flowerBoxes: true,
    sign: { text: 'bread', offset: 2.4 },
    plasterTone: 'plaster',
    seed: 5505,
  },
  {
    id: 'hall',
    name: 'The Moot Hall',
    position: ring(280, 37),
    rotation: 0, skew: 0,
    width: 16.0, depth: 11.0,
    storeys: 2, storeyHeight: 3.1,
    style: 'halfTimber',
    roof: 'thatch', roofPitch: 52, eave: 1.15,
    jetty: true,
    chimneys: [
      { x: -6.4, z: -2.4, height: 5.6, pots: 2, style: 'brick', width: 1.5 },
      { x: 6.4, z: -2.4, height: 5.2, pots: 2, style: 'brick', width: 1.4 },
    ],
    dormers: [{ x: -3.4, width: 2.4 }, { x: 3.4, width: 2.4 }],
    porch: null,
    arcade: null,
    door: { x: 0.0, style: 'double', colour: 'wood', arched: true },
    steps: { width: 4.6, count: 4 },
    ivy: 0.2,
    flowerBoxes: false,
    sign: null,
    plasterTone: 'plaster',
    seed: 6606,
  },
  {
    id: 'apothecary',
    name: 'The Apothecary',
    position: ring(308, 34),
    rotation: 0, skew: 6,
    width: 10.5, depth: 8.5,
    storeys: 2, storeyHeight: 2.8,
    style: 'halfTimber',
    roof: 'thatch', roofPitch: 54, eave: 1.0,
    jetty: true,
    chimneys: [{ x: 3.8, z: -1.6, height: 4.5, pots: 1, style: 'brick', width: 1.2 }],
    dormers: [{ x: 0.2, width: 2.2 }],
    porch: null,
    arcade: null,
    door: { x: -2.6, style: 'plank', colour: 'green', arched: true },
    steps: { width: 2.6, count: 2 },
    ivy: 0.5,
    flowerBoxes: true,
    sign: { text: 'physick', offset: 2.1 },
    plasterTone: 'plaster',
    seed: 7707,
  },
  {
    id: 'cooper',
    name: "The Cooper's Yard",
    position: ring(336, 33),
    rotation: 0, skew: -5,
    width: 12.5, depth: 8.0,
    storeys: 1, storeyHeight: 3.2,
    style: 'stoneBase',
    roof: 'thatchHalfHip', roofPitch: 49, eave: 1.1,
    jetty: false,
    chimneys: [{ x: -4.8, z: -1.2, height: 3.4, pots: 1, style: 'brick', width: 1.2 }],
    dormers: [],
    porch: null,
    arcade: null,
    door: { x: 2.0, style: 'barn', colour: 'wood', arched: false },
    steps: null,
    ivy: 0.25,
    flowerBoxes: false,
    sign: { text: 'barrels', offset: 2.0 },
    plasterTone: 'plasterWorn',
    seed: 8808,
  },
  {
    id: 'saddlery',
    name: 'The Saddlery',
    position: ring(8, 34),
    rotation: 0, skew: 4,
    width: 11.0, depth: 8.5,
    storeys: 2, storeyHeight: 2.75,
    style: 'halfTimber',
    roof: 'thatch', roofPitch: 53, eave: 1.0,
    jetty: false,
    chimneys: [{ x: -4.0, z: -2.0, height: 4.3, pots: 2, style: 'brick', width: 1.25 }],
    dormers: [],
    porch: null,
    arcade: null,
    door: { x: 1.8, style: 'plank', colour: 'wood', arched: true },
    steps: { width: 2.8, count: 2 },
    ivy: 0.4,
    flowerBoxes: true,
    sign: null,
    plasterTone: 'plaster',
    seed: 9909,
  },
  {
    id: 'inn',
    name: 'The Ploughman — an inn',
    position: ring(32, 32.5),
    rotation: 0, skew: -2,
    width: 19.0, depth: 9.5,
    storeys: 2, storeyHeight: 2.9,
    style: 'longRow',
    roof: 'thatch', roofPitch: 50, eave: 1.35,
    jetty: true,
    chimneys: [
      { x: -7.6, z: -2.0, height: 5.2, pots: 2, style: 'brick', width: 1.45 },
      { x: 2.2, z: -2.6, height: 4.6, pots: 1, style: 'brick', width: 1.2 },
    ],
    dormers: [{ x: -5.0, width: 2.3 }, { x: 5.6, width: 2.3 }],
    porch: null,
    arcade: { bays: 5, depth: 2.1 },
    door: { x: -1.0, style: 'double', colour: 'wood', arched: true },
    steps: { width: 3.2, count: 2 },
    ivy: 0.55,
    flowerBoxes: true,
    sign: { text: 'the ploughman', offset: 3.1 },
    plasterTone: 'plaster',
    seed: 1010,
  },
  {
    id: 'rosecot',
    name: 'Rose Cottage',
    position: ring(62, 34),
    rotation: 0, skew: 7,
    width: 9.0, depth: 7.5,
    storeys: 1, storeyHeight: 3.0,
    style: 'plaster',
    roof: 'thatchHip', roofPitch: 55, eave: 1.2,
    jetty: false,
    chimneys: [{ x: -3.2, z: -1.0, height: 4.0, pots: 1, style: 'brick', width: 1.15 }],
    dormers: [{ x: 0.0, width: 2.0 }],
    porch: { width: 2.6, depth: 1.4, height: 2.35 },
    arcade: null,
    door: { x: 0.0, style: 'plank', colour: 'teal', arched: true },
    steps: { width: 2.6, count: 2 },
    ivy: 0.65,
    flowerBoxes: true,
    sign: null,
    plasterTone: 'plaster',
    seed: 1111,
  },
];

// Resolve facings once, at module load.
for (const p of PLOTS) p.rotation = faceCentre(p.position, p.skew || 0);

/**
 * What each building is used for, ground floor first. One entry per storey the
 * plot actually has, so the interior stream can furnish a room without guessing
 * what trade the building belongs to. Values come from ROOM_USES in contracts.js.
 */
export const INTERIOR_USES = {
  weaver: ['workshop', 'bedroom'],
  stables: ['stable'],
  granary: ['store', 'store'],
  chandler: ['shop', 'bedroom'],
  bakery: ['bakery'],
  hall: ['hall', 'hall'],
  apothecary: ['shop', 'bedroom'],
  cooper: ['workshop'],
  saddlery: ['shop', 'bedroom'],
  inn: ['taproom', 'bedroom'],
  rosecot: ['kitchen'],
};

/* -------------------------------------------------------------------------- */
/* Derived helpers                                                             */
/* -------------------------------------------------------------------------- */

const _v = new THREE.Vector3();

/** Transform a point from a plot's local space into world space. */
export function plotToWorld(plot, lx, ly, lz, out = new THREE.Vector3()) {
  const c = Math.cos(plot.rotation), s = Math.sin(plot.rotation);
  return out.set(
    plot.position[0] + lx * c + lz * s,
    plot.position[1] + ly,
    plot.position[2] - lx * s + lz * c,
  );
}

/** Unit vector pointing away from the plot's front face. */
export function plotForward(plot, out = new THREE.Vector3()) {
  return out.set(Math.sin(plot.rotation), 0, Math.cos(plot.rotation));
}

/** Unit vector along the plot's local +X. */
export function plotRight(plot, out = new THREE.Vector3()) {
  return out.set(Math.cos(plot.rotation), 0, -Math.sin(plot.rotation));
}

/**
 * Evenly spaced anchors along the front wall, pushed `out` metres forward.
 * Used by props (planters, benches, barrels) and lighting (wall lanterns) so
 * they never disagree about where a wall is.
 */
export function frontAnchors(plot, count, outset = 0.9, inset = 1.2) {
  const res = [];
  const halfW = plot.width / 2 - inset;
  for (let i = 0; i < count; i++) {
    const t = count === 1 ? 0 : (i / (count - 1)) * 2 - 1;
    const lx = t * halfW;
    const lz = plot.depth / 2 + outset;
    res.push({ position: plotToWorld(plot, lx, 0, lz).toArray(), lx, plot });
  }
  return res;
}

/** World-space footprint rectangle (4 corners) with an optional margin. */
export function plotFootprint(plot, margin = 0) {
  const hw = plot.width / 2 + margin;
  const hd = plot.depth / 2 + margin;
  return [
    plotToWorld(plot, -hw, 0, -hd).clone(),
    plotToWorld(plot, hw, 0, -hd).clone(),
    plotToWorld(plot, hw, 0, hd).clone(),
    plotToWorld(plot, -hw, 0, hd).clone(),
  ];
}

/** True if (x,z) falls inside any building footprint (+margin). */
export function insideAnyBuilding(x, z, margin = 0.5) {
  for (const p of PLOTS) {
    const dx = x - p.position[0];
    const dz = z - p.position[2];
    const c = Math.cos(-p.rotation), s = Math.sin(-p.rotation);
    // inverse of plotToWorld's rotation
    const lx = dx * c - dz * s;
    const lz = dx * s + dz * c;
    if (Math.abs(lx) < p.width / 2 + margin && Math.abs(lz) < p.depth / 2 + margin) return true;
  }
  return false;
}

/** True if (x,z) is on the cobbled plaza. */
export function insidePlaza(x, z) {
  const r = Math.hypot(x, z);
  if (r > 34) return false;
  return r <= plazaRadiusAt(Math.atan2(z, x));
}

/** Distance from (x,z) to the nearest road centreline, and that road. */
export function roadDistance(x, z) {
  let best = Infinity, bestRoad = null;
  for (const road of ROADS) {
    for (let i = 0; i < road.points.length - 1; i++) {
      const [ax, az] = road.points[i];
      const [bx, bz] = road.points[i + 1];
      const vx = bx - ax, vz = bz - az;
      const len2 = vx * vx + vz * vz || 1;
      let t = ((x - ax) * vx + (z - az) * vz) / len2;
      t = Math.max(0, Math.min(1, t));
      const d = Math.hypot(x - (ax + vx * t), z - (az + vz * t));
      if (d < best) { best = d; bestRoad = road; }
    }
  }
  return { distance: best, road: bestRoad };
}

/** Keep-out test used by every scatter system. */
export function isClearGround(x, z, margin = 1.0) {
  if (insideAnyBuilding(x, z, margin)) return false;
  const wr = Math.hypot(x - WELL.position[0], z - WELL.position[2]);
  if (wr < WELL.base[0].radius + margin) return false;
  return true;
}

/* -------------------------------------------------------------------------- */
/* Authored dressing                                                           */
/* -------------------------------------------------------------------------- */

/** Hero props with hand-placed positions. Everything else is scattered. */
export const PROP_SITES = [
  { kind: 'cart', position: [-13.5, 0, 15.8], rotation: 1.05 },
  { kind: 'barrelStack', position: [24.0, 0, -8.5], rotation: 0.4, count: 5 },
  { kind: 'barrelStack', position: [-19.0, 0, -16.0], rotation: -0.8, count: 3 },
  { kind: 'logPile', position: [-25.5, 0, -3.0], rotation: 0.25 },
  { kind: 'trough', position: [-27.0, 0, 12.6], rotation: 1.9 },
  { kind: 'marketStall', position: [12.0, 0, -12.5], rotation: -0.55 },
  { kind: 'marketStall', position: [-9.0, 0, -17.0], rotation: 0.35 },
  { kind: 'bench', position: [8.5, 0, 12.0], rotation: -2.2 },
  { kind: 'bench', position: [-8.0, 0, 9.0], rotation: 2.4 },
  { kind: 'crateStack', position: [19.5, 0, 14.0], rotation: -0.3, count: 4 },
  { kind: 'sackPile', position: [-15.0, 0, -25.0], rotation: 0.6 },
  { kind: 'wheelbarrow', position: [6.0, 0, 18.5], rotation: 2.1 },
  { kind: 'anvil', position: [26.5, 0, -4.0], rotation: 0.9 },
  { kind: 'grindstone', position: [-21.0, 0, 24.0], rotation: 0.1 },
  { kind: 'wellBucketSpare', position: [5.4, 0, 8.4], rotation: 0.7 },
  { kind: 'signpost', position: [3.0, 0, 22.0], rotation: 0.15 },
];

/** Free-standing lamp posts. Wall lanterns are derived from PLOTS. */
export const LAMP_POSTS = [
  { position: [-6.5, 0, 4.0], height: 3.5 },
  { position: [13.5, 0, 2.5], height: 3.5 },
  { position: [0.5, 0, 21.5], height: 3.5 },
  { position: [-17.0, 0, -8.0], height: 3.5 },
  { position: [17.5, 0, -18.0], height: 3.5 },
];

/** Hand-placed hero trees. The vegetation stream scatters the rest. */
export const TREE_SITES = [
  { kind: 'conifer', position: [40, 0, -36], scale: 1.5 },
  { kind: 'conifer', position: [46, 0, -30], scale: 1.25 },
  { kind: 'conifer', position: [35, 0, -44], scale: 1.35 },
  { kind: 'conifer', position: [-44, 0, -30], scale: 1.3 },
  { kind: 'broadleaf', position: [-40, 0, 26], scale: 1.4 },
  { kind: 'broadleaf', position: [44, 0, 20], scale: 1.3 },
  { kind: 'broadleaf', position: [-33, 0, -38], scale: 1.15 },
  { kind: 'broadleaf', position: [28, 0, 38], scale: 1.2 },
  { kind: 'broadleaf', position: [-14, 0, 44], scale: 1.1 },
  { kind: 'conifer', position: [58, 0, -10], scale: 1.6 },
  { kind: 'conifer', position: [-56, 0, 8], scale: 1.45 },
  { kind: 'broadleaf', position: [52, 0, 40], scale: 1.5 },
];

/**
 * Rolling hills behind the village. Each is a smooth radial bump the terrain
 * stream sums into the height field. Kept out here so vegetation can query the
 * same shape without importing the terrain module.
 */
/**
 * Rolling hills behind the village.
 *
 * These numbers are set by what they do to the SKYLINE, not by what they look
 * like on a map. Measured from the spawn eye, the worst terrain elevation across
 * the forward arc is 8.5°, against a roofline of about 15.3° — so every roof
 * reads against sky rather than against hillside, which is what the reference
 * image does. An earlier, closer set peaked at 19.4° and buried the rooflines in
 * one bald green dome.
 *
 * The nearest centre is 398 m out, so nothing rises until ~138 m; the two big
 * ranges sit at 600-630 m where the fog turns them to pale haze. Note the
 * terrain's ridging modulator has mean ~0.65, so `height` is a ceiling and real
 * crests land around 55-70% of it.
 */
export const HILLS = [
  { x: -120, z: -380, radius: 260, height: 46, sharpness: 1.8 },
  { x: 220, z: -430, radius: 300, height: 58, sharpness: 1.7 },
  { x: -420, z: -300, radius: 280, height: 52, sharpness: 1.8 },
  { x: 430, z: -180, radius: 260, height: 44, sharpness: 1.9 },
  { x: -450, z: 120, radius: 300, height: 50, sharpness: 1.8 },
  { x: 100, z: -620, radius: 380, height: 88, sharpness: 1.5 },
  { x: -260, z: -600, radius: 340, height: 76, sharpness: 1.6 },
  { x: 520, z: 260, radius: 300, height: 56, sharpness: 1.7 },
  { x: -520, z: 420, radius: 320, height: 52, sharpness: 1.7 },
  { x: 160, z: 520, radius: 340, height: 44, sharpness: 1.8 },
];

/**
 * Far ridge — a hazy silhouette beyond the fog, nothing more. Heights are kept
 * under the hills' apparent scale on purpose: at 900 m these subtend ~8°, so
 * they sit below the rooflines and read as depth instead of as a grey wall.
 */
export const MOUNTAINS = [
  { angle: -35, distance: 940, width: 560, height: 132 },
  { angle: -8, distance: 1020, width: 660, height: 158 },
  { angle: 22, distance: 960, width: 520, height: 120 },
  { angle: -62, distance: 980, width: 480, height: 106 },
  { angle: 48, distance: 1060, width: 600, height: 114 },
];

/**
 * Sun for the reference image's late-afternoon light: warm, from the upper
 * right and slightly behind, so shadows rake across the plaza toward the
 * camera's left. Owned by the lighting stream but declared here so the sky,
 * the env map and the terrain shading all start from the same numbers.
 */
/**
 * Sun for the reference image's late-afternoon light.
 *
 * The bearing matters more than it looks: at the original 118° (south-east) every
 * façade facing the plaza fell into shade and the square read flat. From the
 * north-east the near-left cottage's front goes from N·L 0.40 to 0.88 and shadows
 * rake south-west across the setts, toward the camera's lower left — which is what
 * the reference does.
 */
export const SUN = {
  azimuthDeg: 57,   // compass bearing the light comes FROM
  // 36° rather than a lower, more golden 28°: shadows scale as 1/tan(elevation),
  // so 28° threw 1.88x the caster's height — a 10 m eave laid 19 m of shade across
  // a plaza of radius 26 and put the well, the hero prop, in shadow. At 36° it is
  // 1.38x and the centre of the square stays sunlit, as in the reference.
  elevationDeg: 36,
  colour: 0xffd9a0,
  intensity: 3.6,
};
