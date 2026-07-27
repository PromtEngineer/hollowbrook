/**
 * ============================================================================
 *  TERRAIN & GROUND
 * ============================================================================
 * The village floor is a hard promise: inside `WORLD.villageRadius` (46 m) the
 * ground is EXACTLY y = 0, because every other stream places things by dropping
 * them on y = 0. Outside that the ground blends over 46..70 m into rolling
 * hills built from `HILLS`, fbm undulation and road corridors flattened along
 * `ROADS`.
 *
 * Layer stack, bottom to top (chosen so nothing can z-fight — every pair of
 * co-located surfaces is separated by real geometry, not by depth bias alone):
 *
 *   meadow        y = heightAt(x,z) + gutter    opaque, depth-writes
 *   wear decal    y = meadow + 0.012            transparent, depthWrite:false
 *   roads         y = heightAt + 0.040 (crown)  opaque, edges dive to -0.090
 *   plaza         y = 0.013 ± 0.010             opaque, its own annulus hole
 *
 * The meadow is a polar annulus whose INNER ring is the plaza polygon pulled
 * 0.25 m inward and dropped 30 mm, so the cobbles always overlap it: no gap and
 * no coincident surfaces. Roads bury their outer lanes 90 mm below the terrain
 * so their silhouette is a real intersection curve rather than an alpha edge,
 * and their leading cross-section is buried under the plaza rim.
 *
 * `heightAt` / `slopeAt` / `surfaceAt` are the public sampling API. They are
 * pure, allocation-free and defined everywhere (including 10 km out).
 * ============================================================================
 */

import * as THREE from 'three';
import { WORLD } from '../config.js';
import {
  PLAZA_POLY, plazaRadiusAt, insidePlaza, insideAnyBuilding,
  ROADS, HILLS, MOUNTAINS, PLOTS, WELL, SUN, plotToWorld,
} from './layout.js';
import { valueNoise2D, fbm2D, Rng } from '../util/rng.js';
import { boxCollider, quatY } from '../contracts.js';

/* -------------------------------------------------------------------------- */
/* Tunables                                                                    */
/* -------------------------------------------------------------------------- */

const VILLAGE_R = WORLD.villageRadius;   // 46 — flat, exactly y = 0
const BLEND_R = 70;                      // hills reach full strength here
const MEADOW_R = 262;                    // detailed ground ends
const FAR_R = 1500;                      // coarse skirt ends (fog eats it at 620)
const BOUND_R = 300;                     // invisible wall ring

/** Metres of world covered by one texture tile, per surface. */
const TILE = {
  cobble: 2.35,
  dirt: 3.10,
  grass: 4.60,
  soil: 2.80,
  far: 46.0,
};

/**
 * Heightfield collider: 513 x 513 samples over 640 x 640 m — 1.25 m cells.
 *
 * 640 m rather than 800: the player cannot leave the `worldBound` ring at
 * BOUND_R = 300, whose 48 slabs are 1.6 m thick, so the reachable region is a
 * polygon of inradius 299.2 m and circumradius 299.84 m. A field spanning
 * +-320 m therefore covers every square metre a body can rest on with 20 m to
 * spare, and the 44% of area we stop paying for buys the resolution instead.
 *
 * Cost, measured: 513^2 = 263 169 samples = 1.004 MB per copy (ours, plus one
 * in the wasm heap once Rapier owns it) against 257^2 = 66 049 = 0.252 MB
 * before — about +1.5 MB resident. The bake goes 28 ms -> 90 ms (warm, three
 * runs each), once, at startup. In exchange the worst |collider - heightAt|
 * anywhere inside r <= 300 drops from 0.0660 m to 0.0118 m, and inside the
 * village from 0.0219 m to 0.0035 m — both measured by ray-dropping onto the
 * real Rapier collider, see the convention note on `buildHeightfieldCollider`.
 */
const HF_SAMPLES = 513;
const HF_SIZE = 640;

/** Plaza vertical budget. Keeps |visual - physics| under 22 mm. */
const PLAZA_BASE = 0.013;
const PLAZA_DISH = 0.008;
const PLAZA_N1 = 0.005;
const PLAZA_N2 = 0.004;

/**
 * Floor on the plaza's vertex-colour LUMINANCE, and the knee over which it is
 * approached. Vertex colours multiply the cobble albedo, so a 0.37 multiplier
 * (what this used to reach) is a 2.7x darkening stacked on top of the map: at the
 * material library's new ~0.28 linear cobble that is not wear, it is a stain.
 * 0.72 keeps the wear tonal.
 */
const PLAZA_FLOOR = 0.72;
const PLAZA_KNEE = 0.10;

/** Road cross-section, as a fraction of half-width -> (lift, colour blend). */
const ROAD_LANES = [-1.45, -1.00, -0.62, -0.22, 0.22, 0.62, 1.00, 1.45];
const ROAD_CROWN = 0.040;
const ROAD_SKIRT = -0.090;
/** Lift of the ribbon where it is still buried under the plaza cobbles. */
const ROAD_BURIED = -0.085;

/**
 * Vertex-colour level of the PAVED part of a road ribbon, by surface.
 *
 * These are not aesthetic knobs, they are map compensation. The plaza is drawn
 * on the `cobble` set and the south approach on `cobbleWorn`, which is a
 * lighter bake of the same stone; with both ribbons at level 1.0 the sunlit
 * road measured 0.670 screen luminance against the sunlit plaza's 0.586 — a
 * 14.3% jump, which reads as a pale strip laid ON the square rather than as the
 * same paving worn smooth. The vertex colours were already all but equal at the
 * join (road crown 0.823 against plaza 0.854 inside r < 34), so the residual is
 * the map, and the multiplier is the only thing left that can cancel it:
 * 0.670 x 0.885 = 0.593, within 1.2% of the plaza.
 *
 * `dirt` stays at 1.0. Those lanes are drawn on `dirtPath`, whose tint the
 * materials stream already pulled to a pale warm earth, and they measure 0.667
 * at the crown — already well under the plaza. Darkening them would put the
 * lanes back to being the dimmest ground in the village.
 *
 * The level scales only the paving term. The lip's grass and soil targets are
 * lerped in afterwards at full strength, so the road still dies into the meadow
 * and into the gate aprons on exactly the colours it did before.
 */
const ROAD_LEVEL = { cobble: 0.885, dirt: 1.0 };

const D2R = Math.PI / 180;

/* -------------------------------------------------------------------------- */
/* Noise fields (module scope: deterministic, built once, never reseeded)      */
/* -------------------------------------------------------------------------- */

const noiseHill = valueNoise2D(0x51a2b7);   // broad undulation
const noiseBump = valueNoise2D(0x2c9f41);   // finer undulation
const noisePatch = valueNoise2D(0x7d13ee);  // colour patchiness
const noiseGrain = valueNoise2D(0x1f04c5);  // fine colour / relief grain
const noiseRidge = valueNoise2D(0x3ba7d9);  // hill ridging (spurs and valleys)

/* -------------------------------------------------------------------------- */
/* Small maths helpers                                                         */
/* -------------------------------------------------------------------------- */

function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
/** Vertex colours multiply the albedo map, so keep them <= 1: an albedo
 *  multiplier above 1 is an energy gain the PBR response cannot justify. */
function alb(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
function smooth01(t) { const x = clamp01(t); return x * x * (3 - 2 * x); }
/** Rec.709 luminance of a linear triple. */
function lum(r, g, b) { return 0.2126 * r + 0.7152 * g + 0.0722 * b; }
/**
 * Monotone, C1 soft floor on a multiplier's luminance. Everything at or above
 * `floor + knee` is returned untouched; below that the tail is compressed into
 * [floor, floor+knee] instead of being clipped, so the darkest wear still varies
 * — it just stops being a stain. Guarantees the result is never below `floor`.
 */
function softFloor(L, floor, knee) {
  const t = (L - floor) / knee;
  if (t >= 1) return L;
  if (t <= -1) return floor;
  const s = 0.5 + 0.5 * t;
  return floor + knee * s * s;
}
function smoothstep(e0, e1, x) { return smooth01((x - e0) / (e1 - e0 || 1e-6)); }
function lerp(a, b, t) { return a + (b - a) * t; }

/* -------------------------------------------------------------------------- */
/* Height field                                                                */
/* -------------------------------------------------------------------------- */

/* --- hills ---------------------------------------------------------------- */

/**
 * A hill whose centre is nearer than HILL_NEAR subtends far too much sky: it
 * reads as a green tent pitched behind the rooftops. Suppress it instead of
 * trusting the authored height, ramping back to full strength by HILL_FULL.
 */
const HILL_NEAR = 250;
const HILL_FULL = 430;

/**
 * Second, independent brake: the ground for the first few hundred metres out of
 * the village is held down regardless of which hill covers it, so the rooflines
 * (about 16 degrees up at 34 m) are always seen against SKY.
 */
const HILL_RISE_IN = 90;
const HILL_RISE_OUT = 330;
const HILL_RISE_FLOOR = 0.16;

/** Per-hill amplitude, resolved once — `hillsAt` must never allocate. */
const HILL_AMP = HILLS.map((H) =>
  0.18 + 0.82 * smoothstep(HILL_NEAR, HILL_FULL, Math.hypot(H.x, H.z)));

/** Per-hill noise phase, so no two hills share a ridge pattern. */
const HILL_PHASE = HILLS.map((_, i) => i * 37.13 + 11.7);

/**
 * Sum of the authored HILLS. `sharpness` is the falloff exponent.
 *
 * Each bump is multiplied by a ridged fbm (four octaves, absolute-value creases)
 * so a radial dome becomes a range with spurs and valleys. The modulator is
 * strictly bounded and multiplicative, so the rim still reaches exactly 0 with a
 * zero derivative and the 46..70 m village seam stays C1.
 */
function hillsAt(x, z) {
  let h = 0;
  for (let i = 0; i < HILLS.length; i++) {
    const H = HILLS[i];
    const dx = x - H.x, dz = z - H.z;
    const d2 = dx * dx + dz * dz;
    const r = H.radius;
    if (d2 >= r * r) continue;
    const u = 1 - Math.sqrt(d2) / r;          // 1 at the summit, 0 at the rim
    const s = u * u * (3 - 2 * u);            // C1 at both ends
    const prof = Math.pow(s, H.sharpness);
    if (prof < 1e-5) continue;

    const ph = HILL_PHASE[i];
    // ~600 m spurs, ~250 m shoulders, ~110 m folds, ~50 m grain.
    const n1 = noiseRidge(x * 0.0017 + ph, z * 0.0017 - ph);
    const n2 = noiseRidge(x * 0.0041 - ph, z * 0.0041 + ph);
    const n3 = noiseRidge(x * 0.0092 + ph * 2, z * 0.0092 - ph * 2);
    const n4 = noiseRidge(x * 0.0205 - ph * 3, z * 0.0205 + ph * 3);
    // |sum| creases the field along its zero set: those creases are the valleys
    // between spurs, and they are what stops the bump reading as one bald dome.
    const crease = 1 - Math.min(1, Math.abs(n1 * 0.9 + n2 * 0.55) * 1.35);
    const mod = clamp01(0.30 + 0.85 * crease + 0.26 * n3 + 0.12 * n4);
    h += H.height * HILL_AMP[i] * prof * mod;
  }
  if (h <= 0) return 0;
  const r = Math.sqrt(x * x + z * z);
  return h * (HILL_RISE_FLOOR + (1 - HILL_RISE_FLOOR) *
    smoothstep(HILL_RISE_IN, HILL_RISE_OUT, r));
}

/**
 * Broad + medium undulation on top of the hills. The 36 m octave is tapered
 * out past 170 m: the ground mesh coarsens there, and high-frequency relief a
 * linear mesh cannot follow only buys sag between the visual and the collider.
 */
function undulationAt(x, z) {
  const r = Math.sqrt(x * x + z * z);
  const fine = 1.55 * (1 - 0.66 * smoothstep(170, 340, r));
  return (
    fbm2D(noiseHill, x * 0.0080, z * 0.0080, 3, 2.03, 0.5) * 6.4 +
    noiseBump(x * 0.0275, z * 0.0275) * fine
  );
}

function wildHeight(x, z) {
  return hillsAt(x, z) + undulationAt(x, z);
}

/* --- road corridors ------------------------------------------------------- */

/**
 * Flattened list of road segments with a Gaussian-ish influence radius. Built
 * once at module load so `heightAt` never allocates.
 */
const ROAD_SEGS = (() => {
  const out = [];
  // fadeA / fadeB scale the segment's influence at its two ends (0 = none,
  // 1 = full). They are 1 at every interior join, so the partition stays
  // continuous; only the two virtual end segments taper, which lets the
  // flattening die away ALONG the lane instead of leaving a shelf cut into a
  // hillside where the polyline stops.
  const push = (ax, az, bx, bz, sigma, cut, half, road, fadeA, fadeB) => {
    const vx = bx - ax, vz = bz - az;
    out.push({
      ax, az, vx, vz,
      inv: 1 / (vx * vx + vz * vz || 1),
      sigma, cut, half, road, fadeA, fadeB,
      minX: Math.min(ax, bx) - cut, maxX: Math.max(ax, bx) + cut,
      minZ: Math.min(az, bz) - cut, maxZ: Math.max(az, bz) + cut,
    });
  };
  for (const road of ROADS) {
    const half = road.width * 0.5;
    const sigma = half + 4.5;                 // gentle enough for a 3 m mesh
    const cut = sigma * 3;                    // exp(-(3)^4) ~ 1e-36, safe hard cut
    const pts = road.points;
    const n = pts.length;
    const ex = 18;                            // the ribbon runs ~10 m past the ends
    const d0 = Math.hypot(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1]) || 1;
    push(pts[0][0] + (pts[0][0] - pts[1][0]) / d0 * ex,
      pts[0][1] + (pts[0][1] - pts[1][1]) / d0 * ex,
      pts[0][0], pts[0][1], sigma, cut, half, road, 0, 1);
    for (let i = 0; i < n - 1; i++) {
      push(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], sigma, cut, half, road, 1, 1);
    }
    const dn = Math.hypot(pts[n - 1][0] - pts[n - 2][0], pts[n - 1][1] - pts[n - 2][1]) || 1;
    push(pts[n - 1][0], pts[n - 1][1],
      pts[n - 1][0] + (pts[n - 1][0] - pts[n - 2][0]) / dn * ex,
      pts[n - 1][1] + (pts[n - 1][1] - pts[n - 2][1]) / dn * ex,
      sigma, cut, half, road, 1, 0);
  }
  return out;
})();

/**
 * Terrain height before the village flattening: hills + noise, with the ground
 * pulled toward the road centreline profile inside each corridor.
 *
 * The corridor blend is a smooth partition (1 - prod(1 - a_k)) over segments,
 * so it is continuous even where two segments are equidistant — the naive
 * "height at the nearest point" formulation is not.
 */
function shapedHeight(x, z) {
  const h = wildHeight(x, z);
  let num = 0, den = 0, notA = 1;
  for (let i = 0; i < ROAD_SEGS.length; i++) {
    const s = ROAD_SEGS[i];
    if (x < s.minX || x > s.maxX || z < s.minZ || z > s.maxZ) continue;
    let t = ((x - s.ax) * s.vx + (z - s.az) * s.vz) * s.inv;
    if (t < 0) t = 0; else if (t > 1) t = 1;
    const cx = s.ax + s.vx * t, cz = s.az + s.vz * t;
    const dx = x - cx, dz = z - cz;
    const d = Math.sqrt(dx * dx + dz * dz);
    if (d > s.cut) continue;
    const u = d / s.sigma;
    const u2 = u * u;
    const a = Math.exp(-u2 * u2) * (s.fadeA + (s.fadeB - s.fadeA) * t);
    if (a < 1e-7) continue;
    num += a * wildHeight(cx, cz);
    den += a;
    notA *= (1 - a);
  }
  if (den <= 0) return h;
  const A = (1 - notA) * 0.9;                 // never a perfectly flat trench
  return h + (num / den - h) * A;
}

/**
 * Ground height in metres. EXACTLY 0 inside the village radius; blends to the
 * hills over 46..70 m with a smoothstep (zero derivative at both ends, so the
 * seam is invisible and walkable).
 * @param {number} x @param {number} z @returns {number}
 */
export function heightAt(x, z) {
  const r2 = x * x + z * z;
  if (r2 <= VILLAGE_R * VILLAGE_R) return 0;
  const r = Math.sqrt(r2);
  if (r >= BLEND_R) return shapedHeight(x, z);
  return shapedHeight(x, z) * smooth01((r - VILLAGE_R) / (BLEND_R - VILLAGE_R));
}

const SLOPE_EPS = 0.6;
/** Slope in radians from horizontal. Four `heightAt` samples, no allocation. */
export function slopeAt(x, z) {
  const gx = heightAt(x + SLOPE_EPS, z) - heightAt(x - SLOPE_EPS, z);
  const gz = heightAt(x, z + SLOPE_EPS) - heightAt(x, z - SLOPE_EPS);
  const inv = 1 / (2 * SLOPE_EPS);
  return Math.atan(Math.sqrt(gx * gx + gz * gz) * inv);
}

/**
 * Analytic surface normal from the height field, written straight into a normal
 * buffer. Computing normals from the mesh triangles makes the shading resolution
 * a function of the ring spacing, which is what produced the visible triangular
 * faceting on the hills; sampling `heightAt` instead gives smooth normals however
 * coarse the band is. `eps` should be about the local cell size.
 */
function writeNormalAt(arr, i3, x, z, eps) {
  const e = eps > 0.5 ? eps : 0.5;
  const gx = heightAt(x + e, z) - heightAt(x - e, z);
  const gz = heightAt(x, z + e) - heightAt(x, z - e);
  const nx = -gx, ny = 2 * e, nz = -gz;
  const inv = 1 / (Math.sqrt(nx * nx + ny * ny + nz * nz) || 1);
  arr[i3] = nx * inv; arr[i3 + 1] = ny * inv; arr[i3 + 2] = nz * inv;
}

/** Distance to the nearest road centreline (allocation-free mirror of
 *  layout.roadDistance, which returns a fresh object per call). */
let _roadHalf = 0, _roadSurface = 'dirt';
function roadNearest(x, z) {
  let best = Infinity;
  _roadHalf = 0; _roadSurface = 'dirt';
  for (let i = 0; i < ROAD_SEGS.length; i++) {
    const s = ROAD_SEGS[i];
    if (x < s.minX || x > s.maxX || z < s.minZ || z > s.maxZ) continue;
    let t = ((x - s.ax) * s.vx + (z - s.az) * s.vz) * s.inv;
    if (t < 0) t = 0; else if (t > 1) t = 1;
    const dx = x - (s.ax + s.vx * t), dz = z - (s.az + s.vz * t);
    const d = Math.sqrt(dx * dx + dz * dz);
    if (d < best) { best = d; _roadHalf = s.half; _roadSurface = s.road.surface; }
  }
  return best;
}

/**
 * What the player is standing on.
 * @returns {'cobble'|'dirt'|'grass'|'stone'}
 */
export function surfaceAt(x, z) {
  if (insidePlaza(x, z)) return 'cobble';
  const rd = roadNearest(x, z);
  if (rd < _roadHalf + 0.5) return _roadSurface === 'cobble' ? 'cobble' : 'dirt';
  const r = Math.hypot(x, z);
  if (r < VILLAGE_R) {
    if (insideAnyBuilding(x, z, 1.8)) return 'dirt';
    if (r < plazaRadiusAt(Math.atan2(z, x)) + 2.4) return 'dirt';
    return 'grass';
  }
  if (r > BLEND_R && slopeAt(x, z) > 0.60) return 'stone';   // ~34 degrees
  return 'grass';
}

/* -------------------------------------------------------------------------- */
/* Shared scatter helpers used by the vertex painters                          */
/* -------------------------------------------------------------------------- */

/** Signed-ish distance from (x,z) to the nearest building footprint (0 inside). */
function buildingDistance(x, z) {
  let best = Infinity;
  for (let i = 0; i < PLOTS.length; i++) {
    const p = PLOTS[i];
    const dx = x - p.position[0], dz = z - p.position[2];
    const c = Math.cos(-p.rotation), s = Math.sin(-p.rotation);
    const lx = dx * c - dz * s;
    const lz = dx * s + dz * c;
    const qx = Math.abs(lx) - p.width * 0.5;
    const qz = Math.abs(lz) - p.depth * 0.5;
    const ox = qx > 0 ? qx : 0, oz = qz > 0 ? qz : 0;
    const d = Math.sqrt(ox * ox + oz * oz) + Math.min(Math.max(qx, qz), 0);
    if (d < best) best = d;
  }
  return best;
}

/**
 * 0..1 "there is a road here" field, for painting rather than shaping. The
 * outer edge is both wide and noise-modulated so the packed earth dissolves into
 * the meadow over 3..8 m instead of ending on a clean parallel line.
 */
function roadWear(x, z) {
  const d = roadNearest(x, z);
  if (!isFinite(d)) return 0;
  const n = 0.62 + 0.80 * (0.5 + 0.5 * noisePatch(x * 0.13 - 5, z * 0.13 + 21));
  return 1 - smoothstep(_roadHalf * 0.55, (_roadHalf + 5.0) * n, d);
}

/**
 * Trodden earth at the plaza rim, 0..1. The band starts 1.1 m INSIDE the cobble
 * boundary, so wherever the setts poke out into the turf the turf underneath is
 * already the same colour — that is what turns the old hard rim into an
 * interleaved transition. Its outer edge is noise-widened over 2.0..5.6 m.
 */
function plazaRimWear(x, z) {
  const d = Math.hypot(x, z) - plazaRadiusAt(Math.atan2(z, x));
  if (d > 6.6) return 0;
  const w = 2.0 + 3.6 * (0.5 + 0.5 * noisePatch(x * 0.17 + 31, z * 0.17 - 12));
  return 1 - smoothstep(-1.1, w, d);
}

/**
 * Where each road's centreline crosses the plaza boundary — the gate mouths.
 * Found by walking the polyline (extended 20 m back under the cobbles) until the
 * plaza clearance changes sign, so it tracks `plazaRadiusAt` exactly rather than
 * assuming `points[0]` is on the rim.
 */
const ROAD_MOUTHS = (() => {
  const out = [];
  for (const road of ROADS) {
    const p = road.points;
    const bx = p[0][0] - p[1][0], bz = p[0][1] - p[1][1];
    const bl = Math.hypot(bx, bz) || 1;
    const path = [[p[0][0] + bx / bl * 20, p[0][1] + bz / bl * 20], ...p.map((q) => [q[0], q[1]])];
    let found = null;
    for (let i = 0; i < path.length - 1 && !found; i++) {
      const a = path[i], b = path[i + 1];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      const steps = Math.max(2, Math.ceil(len / 0.25));
      for (let s = 0; s < steps; s++) {
        const t0 = s / steps, t1 = (s + 1) / steps;
        const x0 = a[0] + (b[0] - a[0]) * t0, z0 = a[1] + (b[1] - a[1]) * t0;
        const x1 = a[0] + (b[0] - a[0]) * t1, z1 = a[1] + (b[1] - a[1]) * t1;
        if (plazaClearance(x0, z0) <= 0 && plazaClearance(x1, z1) > 0) {
          found = [(x0 + x1) * 0.5, (z0 + z1) * 0.5];
          break;
        }
      }
    }
    out.push({
      x: found ? found[0] : p[0][0],
      z: found ? found[1] : p[0][1],
      half: road.width * 0.5,
    });
  }
  return out;
})();

/** Radius of the scuffed apron at a gate mouth (metres, before noise). */
const APRON_OUT = 15.0;

/**
 * Broad trodden apron at each gate mouth, 0..1.
 *
 * This is the THREE-surface case. At a gate the plaza rim arc, the road ribbon's
 * outer lane and the open meadow all terminate within a few metres of one
 * another, and a feather authored for two surfaces (cobble->turf, or road->turf)
 * leaves an untouched wedge of bright turf in the gap between the road edge and
 * the cobble arc — the flat green wedge in the left foreground. One earth field,
 * consumed by all three painters (plaza silt, meadow wear, road lip, wear decal),
 * makes the junction one continuous scuff instead of three edges.
 */
function mouthApron(x, z) {
  let best = 0;
  for (let i = 0; i < ROAD_MOUTHS.length; i++) {
    const m = ROAD_MOUTHS[i];
    const dx = x - m.x, dz = z - m.z;
    const d2 = dx * dx + dz * dz;
    if (d2 > (APRON_OUT * 1.6) * (APRON_OUT * 1.6)) continue;
    const d = Math.sqrt(d2);
    const n = 0.72 + 0.56 * (0.5 + 0.5 * noisePatch(x * 0.105 + 7.5, z * 0.105 - 3.3));
    const a = 1 - smoothstep((m.half + 1.0) * 0.6, APRON_OUT * n, d);
    if (a > best) best = a;
  }
  if (best <= 0) return 0;
  // Holed, so the apron is a scuff pattern rather than a disc.
  const hole = 0.58 + 0.42 * (0.5 + 0.5 * noiseGrain(x * 0.24 + 17, z * 0.24 - 8));
  return best * hole;
}

/**
 * Combine wear fields. `Math.max` is wrong where fields overlap: two reasons for
 * the ground to be bare should leave it barer, and taking the max is what let a
 * pale wedge survive between the road edge and the plaza rim. This is a
 * probabilistic union pulled 70% of the way back toward the max, so overlaps
 * darken without the whole village turning to mud.
 */
function unionWear(a, b, c, d) {
  const u = 1 - (1 - a) * (1 - b) * (1 - c) * (1 - d);
  const m = Math.max(a, b, c, d);
  return clamp01(m + (u - m) * 0.70);
}

/** Desire lines: door -> well, door -> south gate mouth, well -> gate. */
const DESIRE = (() => {
  const segs = [];
  const gate = [ROADS[0].points[0][0], ROADS[0].points[0][1]];
  const well = [WELL.position[0], WELL.position[2]];
  const v = new THREE.Vector3();
  for (const p of PLOTS) {
    plotToWorld(p, p.door?.x || 0, 0, p.depth * 0.5 + 1.7, v);
    const dp = [v.x, v.z];
    segs.push([dp[0], dp[1], well[0], well[1]]);
    segs.push([dp[0], dp[1], gate[0], gate[1]]);
  }
  segs.push([well[0], well[1], gate[0], gate[1]]);
  return segs;
})();

/**
 * Foot-polish along the desire lines, 0..1.
 *
 * Deliberately wide, wandering and holed: a tight Gaussian around a straight
 * segment paints a pale strip that reads as a concrete path laid over the setts.
 * `wob` breathes the half-width between ~1.2 m and ~5 m along the line, and
 * `patchy` eats irregular holes out of it, so what survives is a scuff pattern
 * rather than a band.
 */
function desireWear(x, z) {
  let best = Infinity;
  for (let i = 0; i < DESIRE.length; i++) {
    const s = DESIRE[i];
    const vx = s[2] - s[0], vz = s[3] - s[1];
    const len2 = vx * vx + vz * vz || 1;
    let t = ((x - s[0]) * vx + (z - s[1]) * vz) / len2;
    if (t < 0) t = 0; else if (t > 1) t = 1;
    const dx = x - (s[0] + vx * t), dz = z - (s[1] + vz * t);
    const d2 = dx * dx + dz * dz;
    if (d2 < best) best = d2;
  }
  const d = Math.sqrt(best);
  const wob = 0.75 + 1.35 * (0.5 + 0.5 * noisePatch(x * 0.055 + 3.1, z * 0.055 - 7.4));
  const w = 1 - smoothstep(1.5 * wob, 5.4 * wob, d);
  const patchy = 0.22 + 0.78 * (0.5 + 0.5 * noiseGrain(x * 0.17 - 4, z * 0.17 + 9));
  return w * patchy;
}

/* -------------------------------------------------------------------------- */
/* Material access (fail-soft; geometry streams never author materials)        */
/* -------------------------------------------------------------------------- */

const FLAG_KEYS = [
  'vertexColors', 'transparent', 'depthWrite', 'depthTest', 'fog', 'side',
  'polygonOffset', 'polygonOffsetFactor', 'polygonOffsetUnits', 'alphaTest',
  'opacity', 'flatShading', 'toneMapped',
  // `roughness` is in here for one reason: the mountain silhouette asks for a
  // value off the 1.0 ceiling, and a fully-rough untextured material is the
  // exact pair `dev/audit.js` flags as reading like plastic. If a library
  // variant ignored the request we would silently keep the defect, so treat it
  // as structural and clone rather than trust.
  'roughness',
];

/**
 * Ask the library for a material. If the library is missing, or its variant
 * ignored a flag we structurally depend on (vertex colours, decal blending),
 * fall back to a private clone so the ground still renders correctly.
 */
function requestMaterial(materials, name, opts, owned) {
  let m = null;
  try {
    m = (opts && materials?.variant) ? materials.variant(name, opts) : materials?.get?.(name);
    if (!m && materials?.get) m = materials.get(name);
  } catch (err) {
    console.warn(`[terrain] materials.${opts ? 'variant' : 'get'}("${name}") threw:`, err);
  }
  if (!m) {
    m = new THREE.MeshStandardMaterial({ color: 0x8a8577, roughness: 0.95, metalness: 0 });
    owned.push(m);
    return m;
  }
  if (opts) {
    let mismatch = false;
    for (const k of FLAG_KEYS) {
      if (k in opts && m[k] !== opts[k]) { mismatch = true; break; }
    }
    if (mismatch) {
      const clone = m.clone();
      for (const k of FLAG_KEYS) if (k in opts) clone[k] = opts[k];
      clone.needsUpdate = true;
      owned.push(clone);
      m = clone;
    }
  }
  return m;
}

/**
 * uv multiplier that yields exactly one texture tile per `tileMetres`, whatever
 * repeat the material library baked into the map.
 */
function uvScaleFor(material, tileMetres) {
  const rep = material?.map?.repeat?.x;
  const r = (typeof rep === 'number' && rep > 1e-4) ? rep : 1;
  return 1 / (tileMetres * r);
}

/* -------------------------------------------------------------------------- */
/* Colour palette (authored as sRGB hex; THREE.Color converts to linear)       */
/* -------------------------------------------------------------------------- */

const _c = new THREE.Color();
const _v3 = new THREE.Vector3();

/** Linear-space RGB triple from an sRGB hex. */
function lin(hex) {
  _c.setHex(hex, THREE.SRGBColorSpace);
  return [_c.r, _c.g, _c.b];
}

/**
 * These are ALBEDO MULTIPLIERS over the baked maps, not colours in their own
 * right. The material library's cobble sits near 0.28 linear and its grass near
 * 0.13, so every darkening term here is multiplied against a base almost three
 * times brighter than the one this palette was first tuned against: what used to
 * read as subtle wear now reads as dirt. The damp/moss/far tints are therefore
 * kept close to neutral and close to 1 — variation is tonal, not grime — and the
 * green tints are deliberately desaturated, because a 0.13 grass map with a
 * strongly green multiplier renders as vivid turf rather than summer meadow.
 */
const COL = {
  cobbleClean: lin(0xfaf3e6),
  cobbleWorn: lin(0xffffff),
  /**
   * The three sett families the plaza's base tone is drawn from. `cobbleClean`
   * alone (r-b = 0.165 linear, i.e. distinctly warm) was the whole base tone of
   * the square, so once the albedo was recalibrated upward the near field read as
   * one sheet of warm sandstone. These are that same tone split into an ochre, a
   * grey-brown and a cool grey.
   *
   * They are normalised to EXACTLY lum(cobbleClean) = 0.90138 by `equalLum`
   * below, and luminance is a linear functional, so any mix of them has that
   * luminance too: swapping the base tone for a mix of the three cannot move the
   * plaza's measured level, only its chroma. Measured in the browser over the
   * plaza's 17 689 vertices, before -> after:
   *
   *   mean colour luminance   0.81359 -> 0.81363   (unchanged, as predicted)
   *   mean (r - b)            0.1261  -> 0.0641    (-49%: much less golden)
   *   sd   (r - b)            0.0323  -> 0.0469    (+45%: group-to-group variety)
   *   min  colour luminance   0.72    -> 0.72      (PLAZA_FLOOR still binds)
   *
   * and on screen, in the sunlit near field (bottom 30% of the frame, camera on
   * the square): luminance 0.6008 -> 0.6010 looking down the square and 0.6108 ->
   * 0.6101 looking across it, with r-b down 5.2% and 5.0% respectively. Arrival
   * ground-half luminance 0.4744 -> 0.4743.
   */
  settOchre: lin(0xf7edd6),
  settGreyBrown: lin(0xeeeae2),
  settGrey: lin(0xe1e7f0),
  cobbleDamp: lin(0xb0b3aa),
  cobbleMoss: lin(0x9aa678),
  grass: lin(0xe8f0d4),
  grassDry: lin(0xece5c8),
  grassLush: lin(0xc6d6b2),
  grassFar: lin(0xbfcdbd),
  soil: lin(0xd9c7a8),
  /** The trodden band between cobble and turf. It has to sit clearly ABOVE the
   *  meadow's own tone: it is painted on the grass map, so brightness is the only
   *  thing separating "bare earth" from "grass" on that surface. */
  earth: lin(0xe4d3b0),
  /** Dry path dust. Deliberately near the top of the multiplier range: the soil
   *  map is the only ground map that can bridge the gap between the cobble map
   *  (~0.28 linear) and the grass map (~0.135), and it can only do that if its
   *  multiplier is close to 1. */
  dirt: lin(0xf1e6cd),
  rock: lin(0x9c968b),
  haze: lin(0xbcd0e2),
  /**
   * The two ends of the far-ridge rock ramp — the sunlit flank and the shaded
   * one. They replace the old single `ridge` tone: with one tone the only thing
   * a mountain vertex could vary was how much haze sat over it, which is a
   * gradient, not structure.
   *
   * The span is wide (0.090..0.615 luminance) and the haze weights in
   * `buildMountains` were then solved so that the mean crest still lands on
   * 0.484 and the mean foot on 0.555 — exactly where the single-tone version
   * put them. The range's tone against the sky is therefore unchanged; all the
   * new variation lives inside that envelope.
   */
  ridgeLit: lin(0xb9d0e4),
  ridgeShade: lin(0x3f5670),
};

/**
 * Rescale a multiplier so its Rec.709 luminance matches `target`, hue and
 * saturation untouched. Used to pin the three sett families to the tone the
 * plaza was calibrated on.
 */
function equalLum(c, target) {
  const L = lum(c[0], c[1], c[2]);
  if (L <= 1e-4) return c;
  const k = target / L;
  return [alb(c[0] * k), alb(c[1] * k), alb(c[2] * k)];
}
(() => {
  const target = lum(COL.cobbleClean[0], COL.cobbleClean[1], COL.cobbleClean[2]);
  COL.settOchre = equalLum(COL.settOchre, target);
  COL.settGreyBrown = equalLum(COL.settGreyBrown, target);
  COL.settGrey = equalLum(COL.settGrey, target);
})();

/**
 * The far skirt and the mountain ridges fade toward the horizon haze. If the sky
 * stream publishes that colour, use it so the two agree; otherwise keep the
 * authored default. Probed defensively — the sky module is built in parallel and
 * its accessor names are not part of the contract.
 */
function adoptHaze(sky) {
  if (!sky) return false;
  const cands = [
    sky.hazeColour, sky.hazeColor, sky.horizonColour, sky.horizonColor,
    sky.fogColour, sky.fogColor,
    typeof sky.getHorizonColour === 'function' ? safeCall(sky.getHorizonColour, sky) : null,
    typeof sky.getHorizonColor === 'function' ? safeCall(sky.getHorizonColor, sky) : null,
  ];
  for (const c of cands) {
    if (c == null) continue;
    if (c.isColor) { COL.haze = [c.r, c.g, c.b]; return true; }
    if (typeof c === 'number' && isFinite(c)) { COL.haze = lin(c); return true; }
  }
  return false;
}

function safeCall(fn, self) {
  try { return fn.call(self); } catch { return null; }
}

/** Sun direction (pointing toward the sun) from the shared SUN declaration. */
const SUN_DIR = (() => {
  const az = SUN.azimuthDeg * D2R, el = SUN.elevationDeg * D2R;
  const ce = Math.cos(el);
  return new THREE.Vector3(Math.sin(az) * ce, Math.sin(el), -Math.cos(az) * ce).normalize();
})();

/* -------------------------------------------------------------------------- */
/* Plaza                                                                       */
/* -------------------------------------------------------------------------- */

/**
 * The plaza is star-shaped about the origin, so it is meshed by warping a
 * square grid onto the unit disc (elliptical grid mapping — no polar
 * singularity, near-uniform cells) and then stretching each disc point out to
 * `plazaRadiusAt(theta)`. The boundary is therefore exact, not an approximation
 * of PLAZA_POLY.
 */
function buildPlaza(materials, quality, owned) {
  const detail = quality?.geometryDetail || 1;
  const N = Math.max(56, Math.min(200, Math.round(132 * detail)));   // cells / side
  const V = N + 1;
  const count = V * V;

  const pos = new Float32Array(count * 3);
  const uv = new Float32Array(count * 2);
  const col = new Float32Array(count * 3);

  const mtl = requestMaterial(materials, 'cobble', {
    vertexColors: true,
  }, owned);
  const us = uvScaleFor(mtl, TILE.cobble);
  const ca = Math.cos(0.235), sa = Math.sin(0.235);   // break uv/grid alignment

  for (let iy = 0; iy < V; iy++) {
    const v = (iy / N) * 2 - 1;
    for (let ix = 0; ix < V; ix++) {
      const u = (ix / N) * 2 - 1;
      // Square -> unit disc.
      const dx = u * Math.sqrt(1 - v * v * 0.5);
      const dz = v * Math.sqrt(1 - u * u * 0.5);
      const rr = Math.min(1, Math.sqrt(dx * dx + dz * dz));
      const ang = Math.atan2(dz, dx);
      const R = plazaRadiusAt(ang);
      let x = dx * R, z = dz * R;

      // Interleave the cobble edge with the turf: only the outermost ring of
      // setts moves, and only OUTWARD (the meadow's inner ring sits 0.25 m
      // inside the boundary, so retracting would open a gap), in irregular
      // tongues 0.15..0.60 m long. The rim also dips ~20 mm so the resulting lip
      // over the grass is small enough to read as settling, not as a kerb.
      // Two octaves: a ~20 m wander that stops the boundary being an arc at all,
      // and a ~2 m raggedness on top of it. The albedo bridge across this join is
      // weak (cobble map ~0.28 against grass ~0.135, whatever the vertex colours
      // do), so the boundary has to be broken GEOMETRICALLY or it reads as a
      // clean curve wherever the earth fringe happens to be thin.
      const rimW = smooth01((rr - 0.84) / 0.16);
      if (rimW > 0) {
        const wander = 0.5 + 0.5 * noisePatch(x * 0.052 - 8.3, z * 0.052 + 4.9);
        const ragged = 0.5 + 0.5 * noiseGrain(x * 0.55, z * 0.55);
        const tongue = (0.14 + 0.62 * wander) * (0.45 + 0.55 * ragged) * rimW;
        const k = 1 + tongue / (R || 1);
        x *= k; z *= k;
      }

      // Relief: gentle dish toward the middle plus two noise octaves. Stays
      // inside +-22 mm of the y=0 physics plane.
      const n1 = noisePatch(x * 0.62, z * 0.62);
      const n2 = noiseGrain(x * 2.35, z * 2.35);
      const y = PLAZA_BASE - PLAZA_DISH * (1 - rr * rr) + n1 * PLAZA_N1 + n2 * PLAZA_N2
        - 0.020 * rimW;

      const i3 = (iy * V + ix) * 3;
      pos[i3] = x; pos[i3 + 1] = y; pos[i3 + 2] = z;

      const i2 = (iy * V + ix) * 2;
      uv[i2] = (x * ca - z * sa) * us;
      uv[i2 + 1] = (x * sa + z * ca) * us;

      /* --- vertex colour ------------------------------------------------- */
      // Low contrast on purpose: the foot polish is a hint, not a surfaced path.
      // The centre term is broad (sigma ~14 m) so it reads as the whole middle of
      // the square being walked, with no discernible edge anywhere.
      const wear = Math.max(desireWear(x, z), Math.exp(-(x * x + z * z) / 420) * 0.45);
      const damp = smoothstep(0.70, 1.0, rr);
      const bd = buildingDistance(x, z);
      const mossN = 0.5 + 0.5 * noisePatch(x * 0.9 + 40, z * 0.9 - 17);
      const moss = clamp01((1 - smoothstep(0.1, 2.9, bd)) * mossN * 1.15) * 0.26;
      const grime = 0.5 + 0.5 * noiseGrain(x * 0.21 - 9, z * 0.21 + 5);
      // Silt: the outer ~2 m of setts take the same trodden soil as the grass
      // fringe, so cobble -> earth -> turf is one continuous ramp. The gate
      // aprons carry that same soil INTO the square, which is what joins the
      // three surfaces at a road mouth.
      const siltN = 0.55 + 0.45 * (0.5 + 0.5 * noisePatch(x * 0.5 - 21, z * 0.5 + 13));
      const rimSilt = 1 - smoothstep(0, 2.4 * siltN, R * (1 - rr));
      const silt = clamp01(Math.max(rimSilt, mouthApron(x, z) * 0.82)) * 0.46;

      // Most of the tonal variation is a directionless ~12 m mottle (damp and dry
      // patches), so the square still has life without any of it lining up into a
      // path. The floor matters more than the range: with the cobble map near
      // 0.28 linear, a 0.69 multiplier is a visible dirty patch, so the whole
      // swing now lives in 0.81..1.04 and the tints that pull off-white (damp,
      // moss, silt) are weak enough that no vertex drops below ~0.72 luminance.
      const mottle = 0.5 + 0.5 * noisePatch(x * 0.085 + 61, z * 0.085 - 23);

      // Which sett family this patch of paving belongs to. `gA` is a ~2.2 m
      // group field (about one texture tile), `gB` a ~1.2 m raggedness inside
      // the group, `region` a ~31 m bias so whole stretches of the square lean
      // grey or lean ochre rather than every group being an independent draw.
      // Mean 0.5 by construction, and the three families are luminance-matched,
      // so this term is pure chroma: it cannot move the measured level.
      const gA = 0.5 + 0.5 * noisePatch(x * 0.46 - 63, z * 0.46 + 29);
      const gB = 0.5 + 0.5 * noiseGrain(x * 0.83 + 12, z * 0.83 - 51);
      const region = 0.5 + 0.5 * noisePatch(x * 0.032 + 88, z * 0.032 - 44);
      // Centred at 0.38, not 0.50: the reference paving is mostly grey and
      // grey-brown with ochre setts THROUGH it, and because the three families
      // are luminance-matched the bias costs nothing in level — it is the one
      // knob here that can cool the near field without touching the albedo.
      const gsel = clamp01(0.38 + 0.62 * (gA - 0.5) + 0.40 * (gB - 0.5)
        + 0.46 * (region - 0.5));
      // 0 -> cool grey, 0.5 -> grey-brown, 1 -> warm ochre.
      let bR, bG, bB;
      if (gsel < 0.5) {
        const t = gsel * 2;
        bR = lerp(COL.settGrey[0], COL.settGreyBrown[0], t);
        bG = lerp(COL.settGrey[1], COL.settGreyBrown[1], t);
        bB = lerp(COL.settGrey[2], COL.settGreyBrown[2], t);
      } else {
        const t = gsel * 2 - 1;
        bR = lerp(COL.settGreyBrown[0], COL.settOchre[0], t);
        bG = lerp(COL.settGreyBrown[1], COL.settOchre[1], t);
        bB = lerp(COL.settGreyBrown[2], COL.settOchre[2], t);
      }
      // Per-group brightness, on its own field so a group's tone and its level
      // are not locked together. Strictly zero-mean, hence level-preserving.
      const groupBr = 0.5 + 0.5 * noiseGrain(x * 0.46 + 71, z * 0.46 + 33);
      let br = 0.868 + 0.050 * wear + 0.040 * grime + 0.050 * mottle
        + 0.095 * (groupBr - 0.5);
      br *= 1 - 0.085 * damp;

      const cl = COL.cobbleWorn, cd = COL.cobbleDamp, cm = COL.cobbleMoss;
      let r = lerp(bR, cl[0], wear * 0.55) * br;
      let g = lerp(bG, cl[1], wear * 0.55) * br;
      let b = lerp(bB, cl[2], wear * 0.55) * br;
      r = lerp(r, cd[0], damp * 0.20); g = lerp(g, cd[1], damp * 0.20); b = lerp(b, cd[2], damp * 0.20);
      r = lerp(r, cm[0], moss); g = lerp(g, cm[1], moss); b = lerp(b, cm[2], moss);
      r = lerp(r, COL.soil[0], silt); g = lerp(g, COL.soil[1], silt);
      b = lerp(b, COL.soil[2], silt);

      // Lift the darkest tail (rim damp + moss + gate silt all stacking) back to
      // PLAZA_FLOOR, hue preserved. Cheaper and more predictable than trying to
      // keep every one of five overlapping wear fields individually shallow.
      const L = lum(r, g, b);
      if (L > 1e-4) {
        const k = softFloor(L, PLAZA_FLOOR, PLAZA_KNEE) / L;
        r *= k; g *= k; b *= k;
      }

      col[i3] = alb(r); col[i3 + 1] = alb(g); col[i3 + 2] = alb(b);
    }
  }

  const idx = new Uint32Array(N * N * 6);
  let o = 0;
  for (let iy = 0; iy < N; iy++) {
    for (let ix = 0; ix < N; ix++) {
      const a = iy * V + ix, b = a + 1, c = a + V, d = c + 1;
      idx[o++] = a; idx[o++] = c; idx[o++] = b;
      idx[o++] = b; idx[o++] = c; idx[o++] = d;
    }
  }

  const geo = new THREE.BufferGeometry();
  const uvAttr = new THREE.BufferAttribute(uv, 2);
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', uvAttr);
  geo.setAttribute('uv1', uvAttr);
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  geo.computeVertexNormals();
  geo.computeBoundingSphere();

  const mesh = new THREE.Mesh(geo, mtl);
  mesh.name = 'terrain.plaza';
  mesh.castShadow = false;
  mesh.receiveShadow = true;
  mesh.matrixAutoUpdate = false;
  mesh.updateMatrix();
  return { mesh, tris: N * N * 2, verts: count };
}

/* -------------------------------------------------------------------------- */
/* Meadow + far skirt + wear decal                                             */
/* -------------------------------------------------------------------------- */

/**
 * Ring offsets, measured outward from the plaza boundary. Geometric growth to a
 * 3 m cap, held out to 180 m — anywhere the player can actually stand, the
 * linear mesh has to track `heightAt` to within a few centimetres — then free
 * growth out to the meadow edge.
 */
const RING_CAP_R = 180;

function meadowRings(detail) {
  const step0 = 0.72 / Math.max(0.5, detail || 1);
  const target = MEADOW_R - 26;
  const offs = [-0.25];
  let s = 0.45, step = step0;
  while (s < target) {
    offs.push(s);
    s += step;
    const r = 26 + s;
    // 2.4 m out to 135 m — the roads live in there, and a road corridor cut
    // into a hillside is the one feature a coarse mesh cannot follow.
    // Past 180 m the cap used to be lifted entirely, which grew 10 m rings right
    // where the near hills live — long thin triangles that faceted visibly. 4.6 m
    // costs ~13 extra rings (about 7 k triangles) and reads smooth.
    const cap = r < 135 ? 2.4 : (r < RING_CAP_R ? 3.6 : 4.6);
    step = Math.min(cap, step * (r < RING_CAP_R ? 1.058 : 1.05));
  }
  offs.push(target);
  return offs;
}

/** Ring spacing near a given offset — the bound on mesh-vs-heightAt sag. */
function ringStepAt(offs, off) {
  for (let k = 1; k < offs.length; k++) {
    if (offs[k] >= off) return offs[k] - offs[k - 1];
  }
  return offs[offs.length - 1] - offs[offs.length - 2];
}

/** Radius of ring `k` at bearing `ang`: plaza-shaped near in, circular far out. */
function ringRadius(ang, off) {
  const w = smooth01(off / 12);
  return lerp(plazaRadiusAt(ang), 26.0, w) + off;
}

/** Small gutter that hides the plaza rim and gives the roads headroom. */
function ringGutter(off) {
  return -0.030 * (1 - smooth01((off + 0.25) / 3.0));
}

function buildMeadow(materials, quality, offs, owned) {
  const detail = quality?.geometryDetail || 1;
  // Angular divisions. 288 keeps the cells roughly square out at 230 m (5.0 m of
  // arc against a 4.6 m ring step); the old 256 left them stretched.
  const A = Math.max(144, Math.min(416, Math.round(288 * detail)));
  const K = offs.length;
  const count = A * K;

  const mtl = requestMaterial(materials, 'grass', { vertexColors: true }, owned);
  const us = uvScaleFor(mtl, TILE.grass);
  const ca = Math.cos(-0.41), sa = Math.sin(-0.41);

  const pos = new Float32Array(count * 3);
  const uv = new Float32Array(count * 2);
  const col = new Float32Array(count * 3);

  for (let k = 0; k < K; k++) {
    const off = offs[k];
    const gut = ringGutter(off);
    for (let j = 0; j < A; j++) {
      const ang = (j / A) * Math.PI * 2;
      const R = ringRadius(ang, off);
      const x = Math.cos(ang) * R, z = Math.sin(ang) * R;
      const i = k * A + j;
      pos[i * 3] = x;
      pos[i * 3 + 1] = heightAt(x, z) + gut;
      pos[i * 3 + 2] = z;
      uv[i * 2] = (x * ca - z * sa) * us;
      uv[i * 2 + 1] = (x * sa + z * ca) * us;
    }
  }

  const idx = new Uint32Array((K - 1) * A * 6);
  let o = 0;
  for (let k = 0; k < K - 1; k++) {
    for (let j = 0; j < A; j++) {
      const j2 = (j + 1) % A;
      const a = k * A + j, b = k * A + j2, c = (k + 1) * A + j, d = (k + 1) * A + j2;
      idx[o++] = a; idx[o++] = b; idx[o++] = c;
      idx[o++] = b; idx[o++] = d; idx[o++] = c;
    }
  }

  const geo = new THREE.BufferGeometry();
  const uvAttr = new THREE.BufferAttribute(uv, 2);
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', uvAttr);
  geo.setAttribute('uv1', uvAttr);
  geo.setIndex(new THREE.BufferAttribute(idx, 1));

  // Normals come from the height field, not from the triangles: the ring band
  // past 180 m is far coarser than the relief it carries, and mesh normals there
  // shade every quad as a flat facet. Second pass also paints with them.
  const nrm = new Float32Array(count * 3);
  geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
  for (let k = 0; k < K; k++) {
    const eps = Math.max(0.9, 0.85 * ringStepAt(offs, offs[k]));
    for (let j = 0; j < A; j++) {
      const i = k * A + j;
      const x = pos[i * 3], z = pos[i * 3 + 2];
      writeNormalAt(nrm, i * 3, x, z, eps);
      const r = Math.hypot(x, z);

      const ndl = clamp01(nrm[i * 3] * SUN_DIR.x + nrm[i * 3 + 1] * SUN_DIR.y + nrm[i * 3 + 2] * SUN_DIR.z);
      const patch = 0.5 + 0.5 * noisePatch(x * 0.045, z * 0.045);
      const fine = 0.5 + 0.5 * noiseGrain(x * 0.34, z * 0.34);
      const bd = buildingDistance(x, z);
      const worn = unionWear(
        roadWear(x, z),
        1 - smoothstep(0.4, 5.5, bd),
        plazaRimWear(x, z),
        mouthApron(x, z),
      );

      /* --- large-scale tonal field --------------------------------------- */
      // Inside the village the ground is exactly flat, so `ndl` is a constant
      // there and the ONLY thing that varied the turf was `patch` — one 22 m
      // octave. Over a 15 m wedge beside the south approach that is very nearly
      // constant, which is why the meadow read as a painted surface rather than
      // as a field. `broad` is a ~60 m octave (whole-field drying) and `mid` a
      // ~9 m one, ADDED to the original 22 m octave at full weight rather than
      // averaged with it: averaging measurably *reduced* the spread inside the
      // village (near-field vertex sd 0.0257 against 0.0284 before), because the
      // 60 m octave is nearly constant over the 110 m the player can see and so
      // contributes nothing there while diluting the octave that did. Its MEAN
      // is exactly 0.5 — the clamp is symmetric about it — and everything below
      // is linear in it, so the meadow's mean albedo, and with it the measured
      // ground luminance, is unmoved. Measured over the meadow between 26 and
      // 90 m, as the sd of the mean colour luminance per cell (which is what
      // "large-scale" means — total per-vertex sd is dominated by the fine
      // octave and the trodden edges):
      //
      //   12 m cells   0.0221 -> 0.0258      24 m cells   0.0208 -> 0.0247
      //   mean         0.7260 -> 0.7235
      //
      // and the arrival ground-half screen luminance stayed at 0.474.
      const broad = 0.5 + 0.5 * noisePatch(x * 0.0165 + 51, z * 0.0165 - 37);
      const mid = 0.5 + 0.5 * noiseBump(x * 0.115 - 19, z * 0.115 + 66);
      const tonal = clamp01(0.5 + 0.55 * (broad - 0.5) + 1.00 * (patch - 0.5)
        + 0.55 * (mid - 0.5));
      // Sheltered turf: the band just outside the trodden ring around a
      // building never dries out. Masked by `worn` so it cannot fight the bare
      // earth inside that ring, and narrow enough that its net effect on the
      // screen average is a fraction of a percent.
      const shelter = (1 - smoothstep(2.0, 13.0, bd)) * (1 - worn);
      const far = smoothstep(60, 235, r);
      // surfaceAt() calls anything past 34 degrees 'stone'; make the ground
      // agree with the footstep it is about to trigger.
      const rock = smoothstep(0.50, 0.74, Math.acos(clamp01(nrm[i * 3 + 1])));

      // Lush in the hollows and in the lee of the buildings, sun-bleached on the
      // sun-facing slopes and across the dry patches. Same centre as before
      // (0.85 * ndl + 0.175) with a 1.26x wider swing on a field that carries
      // real large-scale structure, so a stretch of meadow now drifts from
      // sun-dried yellow to deep green over tens of metres.
      const dry = clamp01(ndl * 0.85 + 0.175 + 0.44 * (tonal - 0.5) - 0.24 * shelter);
      let rr = lerp(COL.grassLush[0], COL.grassDry[0], dry);
      let gg = lerp(COL.grassLush[1], COL.grassDry[1], dry);
      let bb = lerp(COL.grassLush[2], COL.grassDry[2], dry);

      // Tighter swing than before: the grass map is ~1.8x brighter than it was,
      // so the old 0.84..1.21 range put the bright end into flat vivid turf and
      // the dark end into shadowless black-green. Driven off `tonal` rather than
      // `patch` so the level follows the same large-scale drying the hue does.
      const br = (0.935 + 0.17 * tonal) * (0.94 + 0.10 * fine) * (1 - 0.05 * shelter);
      rr *= br; gg *= br; bb *= br;

      // Trodden earth around buildings, roads, the plaza rim and the gate
      // aprons. 0.78 lands on very nearly the same colour the plaza's own silt
      // reaches, which is what makes the cobble/turf join disappear.
      rr = lerp(rr, COL.earth[0], worn * 0.80);
      gg = lerp(gg, COL.earth[1], worn * 0.80);
      bb = lerp(bb, COL.earth[2], worn * 0.80);

      // Scree and bare rock where the hillside is too steep to hold turf.
      rr = lerp(rr, COL.rock[0], rock * 0.78);
      gg = lerp(gg, COL.rock[1], rock * 0.78);
      bb = lerp(bb, COL.rock[2], rock * 0.78);

      // Aerial perspective baked into albedo: cooler and flatter with distance.
      rr = lerp(rr, COL.grassFar[0], far * 0.40);
      gg = lerp(gg, COL.grassFar[1], far * 0.40);
      bb = lerp(bb, COL.grassFar[2], far * 0.40);

      col[i * 3] = alb(rr); col[i * 3 + 1] = alb(gg); col[i * 3 + 2] = alb(bb);
    }
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.computeBoundingSphere();

  const mesh = new THREE.Mesh(geo, mtl);
  mesh.name = 'terrain.meadow';
  mesh.castShadow = false;
  mesh.receiveShadow = true;
  mesh.matrixAutoUpdate = false;
  mesh.updateMatrix();

  return { mesh, tris: (K - 1) * A * 2, verts: count, A, offs, pos };
}

/**
 * A soil decal over the inner meadow rings. Shares the meadow's exact vertex
 * positions (+12 mm) so the two surfaces can never disagree about where the
 * ground is; blended, depth-tested, no depth write.
 */
function buildWearDecal(meadow, materials, owned) {
  const { A, offs, pos } = meadow;
  let K = 0;
  while (K < offs.length && offs[K] < 20) K++;
  K = Math.min(K, offs.length);
  if (K < 2) return null;

  const count = A * K;
  const p = new Float32Array(count * 3);
  const uv = new Float32Array(count * 2);
  const col = new Float32Array(count * 4);

  const mtl = requestMaterial(materials, 'soil', {
    vertexColors: true, transparent: true, depthWrite: false,
    polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -2,
  }, owned);
  const us = uvScaleFor(mtl, TILE.soil);
  const ca = Math.cos(0.72), sa = Math.sin(0.72);

  for (let i = 0; i < count; i++) {
    const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
    p[i * 3] = x; p[i * 3 + 1] = y + 0.012; p[i * 3 + 2] = z;
    uv[i * 2] = (x * ca - z * sa) * us;
    uv[i * 2 + 1] = (x * sa + z * ca) * us;

    const bd = buildingDistance(x, z);
    const rw = roadWear(x, z);
    const edge = plazaRimWear(x, z);
    const n = 0.5 + 0.5 * noisePatch(x * 0.30 + 11, z * 0.30 - 3);
    const n2 = 0.5 + 0.5 * noiseGrain(x * 1.1, z * 1.1);
    // Same union (and the same gate apron) the meadow underneath uses, so the
    // decal cannot disagree with the vertex colours it sits 12 mm above.
    const ap = mouthApron(x, z);
    let a = unionWear(rw, edge, 1 - smoothstep(0.4, 4.6, bd), ap);
    a = clamp01(a * (0.45 + 0.75 * n)) * 0.92;
    // The plaza rim and the gate aprons get a stronger blend than the general
    // wear does. On this surface the soil MAP is the only thing that can read as
    // bare earth — the meadow underneath is painted on grass, so tinting it
    // browner still renders as dark grass — and a 0.33 alpha at the cobble arc
    // was letting the turf show through as an unbroken bright band. Road and
    // building wear elsewhere is deliberately left where the reviewer liked it.
    a = Math.max(a, clamp01(Math.max(edge, ap * 0.90) * (0.72 + 0.45 * n)) * 0.95);

    // Brighter than the meadow's own trodden tone on purpose: this is the one
    // layer in the join that carries a different MAP (soil, not grass), so it is
    // the only thing that can read as bare earth rather than as dark grass.
    const br = 0.93 + 0.14 * n2;
    col[i * 4] = alb(COL.dirt[0] * br);
    col[i * 4 + 1] = alb(COL.dirt[1] * br);
    col[i * 4 + 2] = alb(COL.dirt[2] * br);
    col[i * 4 + 3] = a;
  }

  const idx = new Uint32Array((K - 1) * A * 6);
  let o = 0;
  for (let k = 0; k < K - 1; k++) {
    for (let j = 0; j < A; j++) {
      const j2 = (j + 1) % A;
      const a = k * A + j, b = k * A + j2, c = (k + 1) * A + j, d = (k + 1) * A + j2;
      idx[o++] = a; idx[o++] = b; idx[o++] = c;
      idx[o++] = b; idx[o++] = d; idx[o++] = c;
    }
  }

  const geo = new THREE.BufferGeometry();
  const uvAttr = new THREE.BufferAttribute(uv, 2);
  geo.setAttribute('position', new THREE.BufferAttribute(p, 3));
  geo.setAttribute('uv', uvAttr);
  geo.setAttribute('uv1', uvAttr);
  geo.setAttribute('color', new THREE.BufferAttribute(col, 4));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  // Same analytic normals as the meadow it sits 12 mm above, so the two layers
  // can never shade differently and outline the decal's triangles.
  const dn = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) writeNormalAt(dn, i * 3, p[i * 3], p[i * 3 + 2], 1.0);
  geo.setAttribute('normal', new THREE.BufferAttribute(dn, 3));
  geo.computeBoundingSphere();

  const mesh = new THREE.Mesh(geo, mtl);
  mesh.name = 'terrain.wear';
  mesh.castShadow = false;
  mesh.receiveShadow = true;
  mesh.renderOrder = -1;   // lowest surface in the world: first in the blend pass
  mesh.matrixAutoUpdate = false;
  mesh.updateMatrix();
  return { mesh, tris: (K - 1) * A * 2, verts: count };
}

/**
 * Coarse skirt from the meadow edge to 900 m. Shares the meadow's angular
 * divisions and its outer ring formula, so the seam is vertex-exact.
 */
function buildFarGround(meadow, materials, owned) {
  const A = meadow.A;
  const radii = [];
  let r = MEADOW_R, step = 16;
  radii.push(r);
  // The visible hill range lives between 300 and 700 m. Rings of 40 m+ there made
  // every crest line a polyline; 24 m costs ~13 extra rings (about 7 k triangles)
  // and only past 720 m does the spacing run free again.
  while (r < FAR_R) {
    r += step;
    step = Math.min(r < 720 ? 24 : Infinity, step * 1.24);
    radii.push(Math.min(r, FAR_R));
  }
  const K = radii.length;
  const count = A * K;

  const mtl = requestMaterial(materials, 'hillFar', { vertexColors: true }, owned);
  const us = uvScaleFor(mtl, TILE.far);

  const pos = new Float32Array(count * 3);
  const uv = new Float32Array(count * 2);
  const col = new Float32Array(count * 3);

  for (let k = 0; k < K; k++) {
    for (let j = 0; j < A; j++) {
      const ang = (j / A) * Math.PI * 2;
      const R = k === 0 ? ringRadius(ang, MEADOW_R - 26) : radii[k];
      const x = Math.cos(ang) * R, z = Math.sin(ang) * R;
      const i = k * A + j;
      const h = heightAt(x, z);
      pos[i * 3] = x; pos[i * 3 + 1] = h; pos[i * 3 + 2] = z;
      uv[i * 2] = x * us; uv[i * 2 + 1] = z * us;

      const far = smoothstep(MEADOW_R, 640, R);
      const patch = 0.5 + 0.5 * noisePatch(x * 0.010, z * 0.010);
      const hi = smoothstep(4, 70, h);
      let rr = lerp(COL.grassFar[0], COL.grassLush[0], patch * 0.55);
      let gg = lerp(COL.grassFar[1], COL.grassLush[1], patch * 0.55);
      let bb = lerp(COL.grassFar[2], COL.grassLush[2], patch * 0.55);
      rr = lerp(rr, COL.haze[0], far * 0.75 + hi * 0.18);
      gg = lerp(gg, COL.haze[1], far * 0.75 + hi * 0.18);
      bb = lerp(bb, COL.haze[2], far * 0.75 + hi * 0.18);
      col[i * 3] = alb(rr); col[i * 3 + 1] = alb(gg); col[i * 3 + 2] = alb(bb);
    }
  }

  const idx = new Uint32Array((K - 1) * A * 6);
  let o = 0;
  for (let k = 0; k < K - 1; k++) {
    for (let j = 0; j < A; j++) {
      const j2 = (j + 1) % A;
      const a = k * A + j, b = k * A + j2, c = (k + 1) * A + j, d = (k + 1) * A + j2;
      idx[o++] = a; idx[o++] = b; idx[o++] = c;
      idx[o++] = b; idx[o++] = d; idx[o++] = c;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.setIndex(new THREE.BufferAttribute(idx, 1));
  // Analytic normals again: the first skirt rings are 16 m across and carry the
  // outer flanks of the hills, which is exactly where facets showed.
  const fn = new Float32Array(count * 3);
  for (let k = 0; k < K; k++) {
    const eps = k === 0 ? 3.0 : Math.max(3.0, (radii[k] - radii[k - 1]) * 0.7);
    for (let j = 0; j < A; j++) {
      const i = k * A + j;
      writeNormalAt(fn, i * 3, pos[i * 3], pos[i * 3 + 2], eps);
    }
  }
  geo.setAttribute('normal', new THREE.BufferAttribute(fn, 3));
  geo.computeBoundingSphere();

  const mesh = new THREE.Mesh(geo, mtl);
  mesh.name = 'terrain.far';
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.matrixAutoUpdate = false;
  mesh.updateMatrix();
  return { mesh, tris: (K - 1) * A * 2, verts: count };
}

/* -------------------------------------------------------------------------- */
/* Roads                                                                       */
/* -------------------------------------------------------------------------- */

function catmull(p0, p1, p2, p3, t, out) {
  const t2 = t * t, t3 = t2 * t;
  out[0] = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t +
    (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
    (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3);
  out[1] = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t +
    (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
    (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3);
  return out;
}

/** Dense, arc-length resampled centreline, extended 10 m back into the plaza. */
function roadStations(road, spacing) {
  const p = road.points.map((q) => [q[0], q[1]]);
  const n = p.length;
  const e0 = [p[0][0] - p[1][0], p[0][1] - p[1][1]];
  const l0 = Math.hypot(e0[0], e0[1]) || 1;
  const en = [p[n - 1][0] - p[n - 2][0], p[n - 1][1] - p[n - 2][1]];
  const ln = Math.hypot(en[0], en[1]) || 1;
  const ctrl = [
    [p[0][0] + e0[0] / l0 * 10, p[0][1] + e0[1] / l0 * 10],
    ...p,
    [p[n - 1][0] + en[0] / ln * 10, p[n - 1][1] + en[1] / ln * 10],
  ];

  const dense = [];
  const tmp = [0, 0];
  for (let i = 0; i < ctrl.length - 1; i++) {
    const a = ctrl[Math.max(0, i - 1)], b = ctrl[i], c = ctrl[i + 1];
    const d = ctrl[Math.min(ctrl.length - 1, i + 2)];
    const steps = 20;
    for (let s = 0; s < steps; s++) {
      catmull(a, b, c, d, s / steps, tmp);
      dense.push([tmp[0], tmp[1]]);
    }
  }
  dense.push(ctrl[ctrl.length - 1].slice());

  // Arc-length resample.
  const out = [];
  let acc = 0;
  out.push(dense[0]);
  for (let i = 1; i < dense.length; i++) {
    const seg = Math.hypot(dense[i][0] - dense[i - 1][0], dense[i][1] - dense[i - 1][1]);
    acc += seg;
    if (acc >= spacing) {
      out.push(dense[i]);
      acc = 0;
    }
  }
  const last = dense[dense.length - 1];
  if (Math.hypot(last[0] - out[out.length - 1][0], last[1] - out[out.length - 1][1]) > 0.5) out.push(last);
  return out;
}

/** Signed clearance outside the plaza boundary, negative inside. */
function plazaClearance(x, z) {
  return Math.hypot(x, z) - plazaRadiusAt(Math.atan2(z, x));
}

/**
 * Trim the centreline so it starts 2.6 m INSIDE the plaza. Everything still
 * under the cobbles is buried 85 mm below grade, so the ribbon surfaces out of
 * the ground underneath the opaque plaza rim — no visible leading edge, and no
 * near-coplanar band over open meadow.
 */
function trimToPlaza(stations) {
  let k = 0;
  while (k < stations.length && plazaClearance(stations[k][0], stations[k][1]) < -2.6) k++;
  if (k === 0) return stations;
  if (k >= stations.length) return stations.slice(-2);
  // Bisect for the exact -2.6 m crossing between k-1 and k.
  let a = stations[k - 1], b = stations[k];
  for (let s = 0; s < 12; s++) {
    const m = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    if (plazaClearance(m[0], m[1]) < -2.6) a = m; else b = m;
  }
  return [a, ...stations.slice(k)];
}

function buildRoadRibbon(roads, materials, tileMetres, matName, offs, owned, seed, level = 1) {
  const rng = new Rng(seed);
  const L = ROAD_LANES.length;
  const verts = [];
  const uvs = [];
  const cols = [];
  const idx = [];

  const mtl = requestMaterial(materials, matName, {
    vertexColors: true,
    polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
  }, owned);
  const us = uvScaleFor(mtl, tileMetres);

  for (const road of roads) {
    const st = trimToPlaza(roadStations(road, 2.0));
    if (st.length < 3) continue;
    const half = road.width * 0.5;
    const base = verts.length / 3;
    let along = 0;
    let total = 0;
    for (let i = 1; i < st.length; i++) {
      total += Math.hypot(st[i][0] - st[i - 1][0], st[i][1] - st[i - 1][1]);
    }

    for (let i = 0; i < st.length; i++) {
      const cur = st[i];
      const prev = st[Math.max(0, i - 1)];
      const next = st[Math.min(st.length - 1, i + 1)];
      let tx = next[0] - prev[0], tz = next[1] - prev[1];
      const tl = Math.hypot(tx, tz) || 1;
      tx /= tl; tz /= tl;
      const nx = -tz, nz = tx;
      if (i > 0) along += Math.hypot(cur[0] - prev[0], cur[1] - prev[1]);

      // Organic width and edge wander.
      const wob = 1 + 0.13 * noisePatch(along * 0.11 + road.width, 0.5) +
        0.06 * noiseGrain(along * 0.44, 3.1);
      const lat = 0.55 * noiseBump(along * 0.07 + 12, 7.7);

      for (let l = 0; l < L; l++) {
        const f = ROAD_LANES[l];
        const jitter = rng.sym(0.03);
        const off = (f + jitter) * half * wob + lat;
        const x = cur[0] + nx * off;
        const z = cur[1] + nz * off;
        const af = Math.abs(f);

        // The meadow is a linear mesh sampling `heightAt`; in a valley its
        // chords sit ABOVE the true surface by roughly curvature*step^2/8, which
        // out at 100 m is more than the crown height. Lift the ribbon by the
        // local ring spacing so it can never be swallowed, but only where the
        // ground actually has relief — inside the village it stays at 40 mm.
        const rr = Math.hypot(x, z);
        const extra = smoothstep(VILLAGE_R, BLEND_R + 20, rr) * 0.075 *
          ringStepAt(offs, rr - 26);

        let lift;
        if (af > 1.2) lift = ROAD_SKIRT - extra * 1.2;
        else {
          lift = (ROAD_CROWN + extra) * (1 - 0.30 * f * f) -
            smoothstep(0.62, 1.0, af) * (0.045 + extra * 1.4);
        }

        // Where the lane is still under the plaza, sink it below the cobbles;
        // only clear of them does it rise to its crown. Driven by the plaza
        // clearance of THIS vertex, not by station index, so a coarse station
        // spacing can never leave a lip of dirt proud of the setts.
        const entry = smoothstep(-2.6, 2.2, plazaClearance(x, z));
        lift = lerp(ROAD_BURIED, lift, entry);

        // Bury the far terminus too, so the lane dies out under the turf
        // instead of ending in a visible transverse lip.
        const tail = smoothstep(total - 9, total - 1.5, along);
        lift = lerp(lift, ROAD_BURIED - extra * 1.2, tail);

        verts.push(x, heightAt(x, z) + lift, z);
        uvs.push(off * us, along * us);

        // Cobble near the square, packed earth further out; grass at the lip.
        // Start greening the lane earlier and over a wider band, so the ribbon
        // dissolves into the meadow instead of ending on a hard parallel line.
        const toGrass = smoothstep(0.42, 1.10, af);
        const dust = 0.5 + 0.5 * noiseGrain(x * 0.5, z * 0.5);
        const rut = 1 - Math.min(1, Math.abs(af - 0.42) * 2.6);
        const fade = smoothstep(8, 34, along);
        // `level` is the per-surface map compensation described at ROAD_LEVEL;
        // it multiplies the paving only, never the lip targets below.
        let br = (0.86 + 0.20 * dust) * (1 - 0.11 * rut) * (1 - 0.11 * fade) * level;
        const cA = road.surface === 'cobble' ? COL.cobbleClean : COL.dirt;
        let r = cA[0] * br, g = cA[1] * br, b = cA[2] * br;
        // The lane's outer lip greens into the meadow — EXCEPT inside a gate
        // apron, where it must go to earth instead. A green lip arriving at the
        // cobble arc is the third edge that made the left-foreground corner read
        // as a wedge; here the lip, the rim silt and the meadow all land on soil.
        const apron = mouthApron(x, z);
        const lipR = lerp(COL.grassLush[0] * 0.86, COL.soil[0] * 0.94, apron);
        const lipG = lerp(COL.grassLush[1] * 0.86, COL.soil[1] * 0.94, apron);
        const lipB = lerp(COL.grassLush[2] * 0.86, COL.soil[2] * 0.94, apron);
        r = lerp(r, lipR, toGrass * 0.85);
        g = lerp(g, lipG, toGrass * 0.85);
        b = lerp(b, lipB, toGrass * 0.85);
        cols.push(alb(r), alb(g), alb(b));
      }
    }

    for (let i = 0; i < st.length - 1; i++) {
      for (let l = 0; l < L - 1; l++) {
        const a = base + i * L + l, b = a + 1;
        const c = base + (i + 1) * L + l, d = c + 1;
        idx.push(a, b, c, b, d, c);
      }
    }
  }

  if (!idx.length) return null;
  const geo = new THREE.BufferGeometry();
  const uvAttr = new THREE.Float32BufferAttribute(uvs, 2);
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  geo.setAttribute('uv', uvAttr);
  geo.setAttribute('uv1', uvAttr);
  geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  geo.setIndex(idx);
  geo.computeVertexNormals();
  geo.computeBoundingSphere();

  const mesh = new THREE.Mesh(geo, mtl);
  mesh.name = `terrain.road.${matName}`;
  mesh.castShadow = false;
  mesh.receiveShadow = true;
  mesh.renderOrder = 1;
  mesh.matrixAutoUpdate = false;
  mesh.updateMatrix();
  return { mesh, tris: idx.length / 3, verts: verts.length / 3 };
}

/* -------------------------------------------------------------------------- */
/* Far mountain silhouette                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Height fractions of the mountain curtain's vertex rows, measured from y = 0 to
 * that column's own crest. Row 0 is the buried base at y = -40. Five rows above
 * ground rather than one is the whole point: with a single quad per column the
 * only tonal variation a vertex colour can express is a linear ramp from the
 * foot of the range to the crest, which is exactly what makes a distant ridge
 * read as a cut-out. The columns are UNCHANGED — same count, same `top` — so the
 * silhouette against the sky is bit-identical to before.
 */
const MTN_ROWS = [0, 0.20, 0.44, 0.66, 0.85, 1.0];

function buildMountains(materials, owned) {
  const rng = new Rng(0x4d4f554e);
  const verts = [];
  const cols = [];
  const nrms = [];
  const idx = [];
  const ridge = valueNoise2D(0x9e3f21);

  // Roughness off the 1.0 ceiling. A perfectly Lambertian, perfectly untextured
  // surface is what the audit reads as plastic; 0.93 still has no visible
  // specular at a kilometre but does let the env map's directionality through,
  // which is what the authored normals below are there to exploit.
  const mtl = requestMaterial(materials, 'mountain', {
    vertexColors: true, fog: false, side: THREE.DoubleSide, roughness: 0.93,
  }, owned);

  const R = MTN_ROWS.length;            // vertex rows per column
  const S = 56;                         // columns per range — do not change:
  const tops = new Float64Array(S + 1); // the crest line is the verified silhouette

  for (let m = 0; m < MOUNTAINS.length; m++) {
    const M = MOUNTAINS[m];
    const halfAng = (M.width * 0.5) / M.distance;
    const a0 = M.angle * D2R;
    const base = verts.length / 3;
    const phase = m * 13.7 + rng.range(0, 40);

    // Pass 1: the crest line. Unchanged formula — the silhouette is verified.
    for (let s = 0; s <= S; s++) {
      const t = s / S;
      const bell = Math.pow(Math.max(0, Math.cos((t * 2 - 1) * Math.PI * 0.5)), 0.62);
      const jag = 0.68 +
        0.34 * ridge(t * 6.2 + phase, 0.5) +
        0.16 * ridge(t * 17.0 + phase, 3.5);
      tops[s] = M.height * bell * Math.max(0.15, jag);
    }

    const ds = M.width / S;             // metres of arc per column

    for (let s = 0; s <= S; s++) {
      const t = s / S;
      const ang = a0 + (t * 2 - 1) * halfAng;
      const ca = Math.cos(ang), sa = Math.sin(ang);
      const x = ca * M.distance;
      const z = sa * M.distance;
      const top = tops[s];

      /* --- which way this stretch of ridge faces ------------------------- */
      // The curtain is a single surface at constant distance, so its true
      // normal is the same radial vector for every vertex of every range — one
      // constant N.L over the whole silhouette, which is the cut-out. Author
      // the normals instead: tilt along the arc by the crest's own slope (a
      // flank on the descending side of a peak faces that way) and tilt up
      // toward the crest, where a real range rolls over. Positions are
      // untouched, so this cannot move the silhouette by a pixel; what it does
      // is make the sky env map and the raking sun land differently along the
      // range.
      const slope = (tops[Math.min(S, s + 1)] - tops[Math.max(0, s - 1)]) /
        (ds * (s === 0 || s === S ? 1 : 2));
      // Two sharpnesses on purpose. The NORMAL wants the soft one, so it sweeps
      // continuously along the range instead of snapping between two attitudes;
      // the baked lit/shade COLOUR wants the hard one, because a real range
      // reads as flanks with an edge between them, and it is that term which
      // carries most of the tonal variation (crest-row luminance sd 0.0341 with
      // it, 0.0201 with the soft curve).
      const tilt = -Math.tanh(slope * 0.9);          // +1 = faces -t, -1 = +t
      const facing = -Math.tanh(slope * 2.2);
      // Inward (toward the village, hence toward the camera) and along the arc.
      const inX = -ca, inZ = -sa;
      const tgX = -sa, tgZ = ca;
      const sunSide = tgX * SUN_DIR.x + tgZ * SUN_DIR.z;
      const lit = clamp01(0.5 + 0.5 * facing * sunSide);

      const hz = clamp01(1 - top / (M.height + 1));

      for (let r = 0; r < R; r++) {
        const g = MTN_ROWS[r];
        verts.push(x, r === 0 ? -40 : top * g, z);

        const ft = tilt * 0.45;
        const fu = 0.10 + 0.30 * g * g;
        const nx = inX + tgX * ft, ny = fu, nz = inZ + tgZ * ft;
        const ninv = 1 / (Math.hypot(nx, ny, nz) || 1);
        nrms.push(nx * ninv, ny * ninv, nz * ninv);

        /* --- vertex colour ---------------------------------------------- */
        // `band` is the height fraction pushed around by a slow noise, so the
        // strata are not level lines drawn across the range.
        const band = clamp01(g + 0.16 * ridge(t * 3.1 + phase, 5.3));
        // Three horizontal frequencies, so no single wavelength reads as a
        // pattern: `flank` is whole-shoulder scale, `strata` is a few per
        // shoulder, `grain` is per-vertex.
        const flank = 0.5 + 0.5 * ridge(t * 2.3 - phase, 1.7);
        const strata = 0.5 + 0.5 * ridge(t * 7.4 + phase, band * 4.6 + 2.2);
        const grain = 0.5 + 0.5 * ridge(t * 19.0 - phase, band * 11.0 + 7.0);
        // Two more octaves purely as a multiplier at the end — cloud-shadow
        // dapple, the one term that can add tone without touching the mean.
        const mottle = 0.62 * (0.5 + 0.5 * ridge(t * 4.6 + phase * 0.5, band * 2.7 - 4.1)) +
          0.38 * (0.5 + 0.5 * ridge(t * 11.3 - phase, band * 6.1 + 12.4));

        // Weights sum with the constant to a mean of 0.64, which with the haze
        // below is what pins the crest at 0.484 luminance.
        const rockMix = clamp01(0.17 + 0.60 * lit + 0.18 * strata + 0.16 * flank
          - 0.16 * (1 - band));
        let rr = lerp(COL.ridgeShade[0], COL.ridgeLit[0], rockMix);
        let gg = lerp(COL.ridgeShade[1], COL.ridgeLit[1], rockMix);
        let bb = lerp(COL.ridgeShade[2], COL.ridgeLit[2], rockMix);

        // Aerial perspective: haze pools at the foot of the range and thickens
        // over the columns that never get high enough to break out of it. The
        // constant and the (1 - band) weight are SOLVED, not eyeballed — Newton
        // on the two row means against the old 0.484 / 0.555 envelope.
        const hazeAmt = clamp01(0.114 + 0.38 * hz + 0.453 * (1 - band) + 0.06 * grain);
        rr = lerp(rr, COL.haze[0], hazeAmt);
        gg = lerp(gg, COL.haze[1], hazeAmt);
        bb = lerp(bb, COL.haze[2], hazeAmt);

        const br = 0.89 + 0.22 * mottle;
        cols.push(alb(rr * br), alb(gg * br), alb(bb * br));
      }
    }

    for (let s = 0; s < S; s++) {
      for (let r = 0; r < R - 1; r++) {
        const a = base + s * R + r, b = a + 1;
        const c = base + (s + 1) * R + r, d = c + 1;
        idx.push(a, b, c, b, d, c);
      }
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
  geo.setAttribute('normal', new THREE.Float32BufferAttribute(nrms, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
  geo.setIndex(idx);
  geo.computeBoundingSphere();

  const mesh = new THREE.Mesh(geo, mtl);
  mesh.name = 'terrain.mountains';
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.renderOrder = -1;
  mesh.matrixAutoUpdate = false;
  mesh.updateMatrix();
  return { mesh, tris: idx.length / 3, verts: verts.length / 3 };
}

/* -------------------------------------------------------------------------- */
/* Colliders                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Rapier / parry heightfield convention. This is the ONE number in this file
 * that cannot be checked by looking at the world, so it was checked by building
 * this exact collider in node against @dimforge/rapier3d-compat 0.19.3 and
 * ray-dropping 97 798 rays onto it — not read off a .d.ts, which is what got it
 * wrong before. Rapier's ROW index runs with **+Z**:
 *
 *   heights[ col * nrows + row ]                          (column-major)
 *   row  i -> z = ( i / (nrows - 1) - 0.5 ) * scale.z     (z INCREASES with i)
 *   col  j -> x = ( j / (ncols - 1) - 0.5 ) * scale.x     (x increases with j)
 *
 * which is exactly what `physics.js` documents at the top of that file. The
 * previous version wrote z = (0.5 - i/(nrows-1)) * scale.z, i.e. the same
 * buffer MIRRORED IN Z. That is invisible inside r <= 46, because the village
 * floor is exactly y = 0 and therefore z-symmetric; the moment the player left
 * the square the collider handed them heightAt(x, -z). Measured error against
 * heightAt with the mirror in place: 1.257 m at (45.3, -40.8), 19.94 m worst
 * anywhere inside r <= 300, 1.617 m mean. With the sign corrected: 0.0006 m at
 * that same probe, 0.0118 m worst, 0.0006 m mean, 0.0035 m inside r <= 46.
 *
 * The field is square and centred on the origin, so a transposed or mirrored
 * reader would still land on the right shape if the terrain were symmetric — it
 * is not. `physics.js` reads `field.order` and `field.rowsAlongX`; the pair
 * emitted below is its no-copy default path.
 */
function buildHeightfieldCollider() {
  const n = HF_SAMPLES;
  const heights = new Float32Array(n * n);
  const inv = 1 / (n - 1);
  for (let j = 0; j < n; j++) {                 // column -> x
    const x = (j * inv - 0.5) * HF_SIZE;
    for (let i = 0; i < n; i++) {               // row -> z, INCREASING with i
      const z = (i * inv - 0.5) * HF_SIZE;
      heights[j * n + i] = heightAt(x, z);
    }
  }
  return {
    shape: 'heightfield',
    field: {
      nrows: n,
      ncols: n,
      heights,
      scale: [HF_SIZE, 1, HF_SIZE],
      /* The flags physics.js actually reads. Both are its defaults, so the
       * buffer is handed to Rapier without a repack. */
      order: 'column-major',
      rowsAlongX: false,
      /* extra, informational — see the comment above */
      layout: 'column-major',
      index: 'heights[col * nrows + row]',
      rowAxis: '+z',
      colAxis: '+x',
    },
    position: [0, 0, 0],
    quaternion: [0, 0, 0, 1],
    friction: 0.92,
    restitution: 0.0,
    tag: 'terrain',
  };
}

/** A ring of tall invisible slabs so the player cannot leave the detailed world. */
function buildBoundColliders() {
  const out = [];
  const N = 48;
  const halfSpan = (Math.PI * BOUND_R) / N * 1.06;   // slight overlap
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2;
    const x = Math.cos(a) * BOUND_R;
    const z = Math.sin(a) * BOUND_R;
    const g = heightAt(x, z);
    // Local +X must run along the tangent (-sin a, cos a) in (x, z).
    const psi = Math.atan2(-Math.cos(a), -Math.sin(a));
    out.push(boxCollider([x, g + 26, z], [halfSpan, 30, 0.8], quatY(psi), {
      friction: 0.2,
      restitution: 0.0,
      tag: 'worldBound',
    }));
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Factory                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Build the ground: plaza, roads, meadow, far skirt, mountain silhouette,
 * one heightfield collider and the world-bound ring.
 *
 * @param {{materials:Object, quality:Object, sky:Object}} deps
 * @returns {Object} WorldChunk + { heightAt, slopeAt, surfaceAt }
 */
export function createTerrain({ materials, quality, sky } = {}) {
  const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const group = new THREE.Group();
  group.name = 'terrain';
  const owned = [];          // materials this module had to clone / invent
  const stats = { tris: 0, verts: 0, meshes: 0 };

  const add = (part, key) => {
    if (!part) return;
    group.add(part.mesh);
    stats[key] = part.tris;
    stats.tris += part.tris;
    stats.verts += part.verts;
    stats.meshes++;
  };

  const hazeFromSky = adoptHaze(sky);

  // One ring table shared by the meadow, the wear decal and the road lift, so
  // the three can never disagree about how coarse the ground is at a radius.
  const offs = meadowRings(quality?.geometryDetail || 1);

  let meadow = null;
  try {
    meadow = buildMeadow(materials, quality, offs, owned);
    add(meadow, 'meadowTris');
  } catch (err) { console.error('[terrain] meadow failed:', err); }

  try { add(buildPlaza(materials, quality, owned), 'plazaTris'); }
  catch (err) { console.error('[terrain] plaza failed:', err); }

  try {
    const cobbleRoads = ROADS.filter((r) => r.surface === 'cobble');
    const dirtRoads = ROADS.filter((r) => r.surface !== 'cobble');
    if (cobbleRoads.length) {
      add(buildRoadRibbon(cobbleRoads, materials, TILE.cobble, 'cobbleWorn', offs, owned,
        0x2051, ROAD_LEVEL.cobble), 'roadCobbleTris');
    }
    if (dirtRoads.length) {
      add(buildRoadRibbon(dirtRoads, materials, TILE.dirt, 'dirtPath', offs, owned,
        0x77a3, ROAD_LEVEL.dirt), 'roadDirtTris');
    }
  } catch (err) { console.error('[terrain] roads failed:', err); }

  if (meadow) {
    try { add(buildWearDecal(meadow, materials, owned), 'wearTris'); }
    catch (err) { console.error('[terrain] wear decal failed:', err); }
    try { add(buildFarGround(meadow, materials, owned), 'farTris'); }
    catch (err) { console.error('[terrain] far ground failed:', err); }
  }

  try { add(buildMountains(materials, owned), 'mountainTris'); }
  catch (err) { console.error('[terrain] mountains failed:', err); }

  // The meadow keeps a reference to its position buffer for the decal; drop it.
  if (meadow) meadow.pos = null;

  /* ------------------------------------------------------------- colliders */
  let colliders = [];
  try {
    colliders = [buildHeightfieldCollider(), ...buildBoundColliders()];
  } catch (err) {
    console.error('[terrain] colliders failed:', err);
  }

  stats.colliders = colliders.length;
  stats.meadowRings = offs.length;
  stats.hfSamples = `${HF_SAMPLES}x${HF_SAMPLES}`;
  stats.hfCellM = +(HF_SIZE / (HF_SAMPLES - 1)).toFixed(3);
  stats.hfExtentM = HF_SIZE;
  stats.hfBufferMB = +((HF_SAMPLES * HF_SAMPLES * 4) / 1048576).toFixed(3);
  stats.ownedMaterials = owned.length;
  stats.buildMs = Math.round((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0);
  stats.plazaPolyPoints = PLAZA_POLY.length;
  stats.haze = hazeFromSky ? 'from sky' : 'authored default';

  return {
    group,
    colliders,
    interactables: [],
    lightAnchors: [],

    heightAt,
    slopeAt,
    surfaceAt,
    /** Convenience for other streams: drop a point onto the ground. */
    placeOnGround(x, z, out = _v3) { return out.set(x, heightAt(x, z), z); },

    update() { /* terrain is static */ },

    dispose() {
      group.traverse((o) => { if (o.isMesh) o.geometry?.dispose?.(); });
      for (const m of owned) m.dispose?.();
      owned.length = 0;
      group.clear();
    },

    stats,
  };
}
