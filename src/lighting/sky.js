/**
 * ============================================================================
 *  SKY + HDRI
 * ============================================================================
 * One inverted sphere carrying an analytic Preetham atmosphere plus a
 * procedural cumulus deck, rendered in HDR (`toneMapped = false` — tone mapping
 * happens once, in OutputPass). That same material is then rendered into a
 * `WebGLCubeRenderTarget` by a `CubeCamera` and pushed through
 * `PMREMGenerator` — the prefiltered cube that falls out IS the project's
 * HDRI. Nothing in Hollowbrook loads an .hdr; every scrap of indirect light in
 * the village comes out of this file.
 *
 * Two things here are easy to get wrong and expensive when you do:
 *
 *  1. The PMREM target is allocated ONCE and handed back to
 *     `pmrem.fromCubemap( cube, target )` on every rebake. That keeps the
 *     `envMap` *texture object identity* stable, so `scene.environment` and any
 *     material that cached the env map stay valid across a time-of-day change —
 *     and it means we never leak a render target.
 *  2. Rebaking costs milliseconds. It is debounced on a timer rather than in
 *     `update()`, because the time-of-day slider lives in the pause menu and
 *     `main.js` does not call `update()` while paused.
 * ============================================================================
 */

import * as THREE from 'three';
import { Rng } from '../util/rng.js';
import { SUN } from '../world/layout.js';

/* -------------------------------------------------------------------------- */
/* Tuning                                                                      */
/* -------------------------------------------------------------------------- */

const DEG = Math.PI / 180;

/** 0 = midnight, 0.5 = noon. 0.66 ≈ 15:50 — the hour in the reference image. */
export const DEFAULT_TIME_OF_DAY = 0.66;

/**
 * Multiplier on layout.js's authored 3.6, so the effective sun is 4.39.
 *
 * This was 1.0 while the baked ground albedos were far below any real material
 * (cobble measured 0.099 linear against 0.24-0.34 for weathered setts, grass
 * 0.074 against 0.10-0.16). With the albedos corrected the whole picture moves
 * up the ACES curve together, and the direct term has to come with it or the
 * sun stops reading as a sun.
 *
 * 1.22 was solved, not guessed: it is the gain that puts sunlit plaza cobble at
 * 0.549 sRGB luminance — the middle of the 0.50-0.62 target — for the expected
 * albedo, and still lands inside the band at both ends of the plausible albedo
 * range (0.511 at cobble 0.24, 0.582 at 0.32). It is NOT a disguised exposure
 * rise: `toneMappingExposure` stays at 1.0, so sunlit plaster peaks at 218 of
 * 255 rather than clipping.
 *
 * Do not push it toward the 1.6 an earlier pass used. At that level lit and
 * shadowed surfaces tone-map to nearly the same value and the square goes flat.
 * The check that matters is the LINEAR lit:shadow irradiance ratio on the
 * ground, which is 2.60 here against 2.72 before — this scales the picture, it
 * does not compress it.
 *
 * Kept as a named constant because lighting.js and the ground-bounce term below
 * must agree on the number.
 */
export const SUN_GAIN = 1.22;

/**
 * The env map is baked from this same dome, so the only knob for "how much
 * indirect light does the sky throw" is the level it is captured at.
 * `main.js` hands `materials.setEnvironment(envMap, 1.0)` and materials.js
 * assigns `envMap` per material, which makes `scene.environmentIntensity` a
 * no-op — so the bake is where that balance has to live.
 *
 * 0.98, up from 0.72. The 0.72 was compensation for the wrong problem: with a
 * near-black ground the indirect term was the only thing keeping shadows from
 * dying, so it had been pulled down to stop it looking ambient-lit. Now that
 * the ground carries a real albedo, the honest value is "the sky as bright as
 * it is drawn", and the shadowed half of the frame is the part that measured
 * too dark (46.6% of ground pixels under 0.20 luminance). This lifts shadowed
 * plaza cobble from 0.233 to 0.341 and a shadowed plaster wall from 0.243 to
 * 0.385, while indirect/direct on the ground only goes 0.580 -> 0.625, so the
 * sun still clearly models the square.
 */
const ENV_BAKE_GAIN = 0.98;

/**
 * Scales raw Preetham radiance into the range ACES likes. Solved offline (see
 * the report): at t = 0.66 the zenith lands near #2b68a9 and the
 * cosine-weighted sky irradiance is ~0.35x the sun's, so shadowed ground sits
 * at roughly a quarter of sunlit ground — what a clear late afternoon does.
 */
const SKY_RADIANCE_SCALE = 0.006;

/** Stand-in for the real solar disc (~6e6) so bloom and half-float survive. */
const SUN_DISC_RADIANCE = 46.0;

const BASE_TURBIDITY = 3.4;
const BASE_RAYLEIGH = 2.6;
const MIE_COEFFICIENT = 0.0045;
const MIE_G = 0.80;

/**
 * Mean albedo of the ground hemisphere — drives the bounce term in the env,
 * which is what fills the lower half of the dome and therefore the only thing
 * lighting shaded wall faces from below.
 *
 * Raised with the ground textures: an area-weighted mix of the corrected setts
 * (~0.28 linear), meadow grass (~0.13) and stone/soil (~0.20) comes out near
 * 0.22 rather than the 0.20 authored against the old dark bakes. Kept slightly
 * warm and slightly green-shy, which is what a cobbled square surrounded by
 * grass actually bounces.
 */
const GROUND_ALBEDO = [0.24, 0.225, 0.155];

/** Ceiling on how often the PMREM may be rebuilt while the slider is dragged. */
const REBAKE_INTERVAL_MS = 260;

/* --- Preetham constants (same numbers as three/addons/objects/Sky.js) ------ */
const TOTAL_RAYLEIGH = [5.804542996261093e-6, 1.3562911419845635e-5, 3.0265902468824876e-5];
const MIE_CONST = [1.8399918514433978e14, 2.7798023919660528e14, 4.0790479543861094e14];
const CUTOFF_ANGLE = 1.6110731556870734;
const STEEPNESS = 1.5;
const SUN_EE = 1000.0;
const RAYLEIGH_ZENITH_LENGTH = 8.4e3;
const MIE_ZENITH_LENGTH = 1.25e3;

/** SUN.colour as linear working-space floats. */
const SUN_LINEAR = new THREE.Color(SUN.colour);

/** GLSL's smoothstep, for the JS side. Works with edge0 > edge1. */
export function smoothstep(edge0, edge1, x) {
  const d = edge1 - edge0;
  const t = Math.min(1, Math.max(0, (x - edge0) / (Math.abs(d) < 1e-9 ? 1e-9 : d)));
  return t * t * (3 - 2 * t);
}

/* -------------------------------------------------------------------------- */
/* Solar path                                                                  */
/* -------------------------------------------------------------------------- */

/*
 * The sun rides a circle of angular radius SUN_RHO about a fixed pole. The pole
 * is SOLVED AT LOAD from exactly one constraint: at t = DEFAULT_TIME_OF_DAY the
 * direction must equal the authored hero sun to nine digits.
 *
 * The hero sun arrives from the NORTH-EAST (bearing 57, elevation 36). The
 * bearing is the whole point of the composition and must not move: the near-left
 * cottage's plaza-facing wall (the weaver, whose front normal measures bearing
 * 56.0) takes the light at N.L = cos(36) = 0.809, and every shadow rakes off to
 * bearing 237 — south-west, across the setts and toward the camera's lower left.
 *
 * The ELEVATION was 28, and 28 was too low. tan(28) puts a shadow at 1.88x the
 * caster's height, so the ~10 m eaves on the east side threw ~19 m across a
 * plaza of radius 26 and swallowed the centre of the square — including the
 * well, which is the hero prop. At 36 the shadow is 1.38x height, ~13 m, which
 * clears the centre by a good 13 m: a brightly sunlit square with shadow only
 * across the near corner, which is what the reference has. 36 is also the
 * ceiling: the weaver's front N.L is cos(elevation), so 37 would drop it to
 * 0.799 and break the "plaza-facing facades stay above 0.8" guarantee.
 *
 * No real latitude puts the sun in the north-east at 15:50, so something has to
 * give. What gives is the direction of travel: in this world the sun rises in
 * the south (05:05, bearing 171), transits at 72.9 deg in the south-east
 * (bearing 117) and sets in the east-north-east (18:47, bearing 63) — a 13.6 h
 * day. The hero hour is exact to 0 deg, the transit is high and on the side of
 * the plaza the player spawns facing, and only sweeping the whole slider reveals
 * which way the sun is going.
 */

/**
 * Angular radius of the sun's daily circle. The one free knob in the solve — and
 * it is the knob that had to move to raise the hero elevation at all.
 *
 * The solve asks for a circle of radius SUN_RHO about some pole that carries the
 * hero direction at phase (0.66 - 0.5) * 2pi = 57.6 deg past the transit. A
 * hero at 57.6 deg of phase past a transit is well down the arc, so the higher
 * the hero sits the higher the transit must be, and past some elevation no pole
 * on the candidate ring satisfies the phase at all. At the old 80 deg the
 * ceiling is a hero elevation of 33: scanned across 50-89 deg of rho, 80 fails
 * to find any pole from 34 up, which would have dropped the whole village onto
 * the baked fallback axis below and silently thrown the composition away.
 *
 * 56 deg is solvable for hero elevations 5-45.5, so 36 sits with 9.5 deg of
 * margin and the widened window below cannot walk off the edge of the solve. It
 * also keeps the transit at a believable 72.9 deg rather than the near-vertical
 * 86-90 the values just under the old ceiling produce.
 */
const SUN_RHO = 56 * DEG;

/** Where the sun has to be at DEFAULT_TIME_OF_DAY for the reference framing. */
const ART_SUN_AZIMUTH_DEG = 57;
const ART_SUN_ELEVATION_DEG = 36;
/**
 * How far layout.js may stray from that before we override it. The elevation
 * window is widened from [22, 36] to [22, 40] so layout.js can legitimately
 * carry the new 36 — and the upper bound stays inside the 45.5 deg the SUN_RHO
 * above can actually solve, so no value the window admits can fail the solve.
 */
const SUN_AZIMUTH_WINDOW = [40, 78];
const SUN_ELEVATION_WINDOW = [22, 40];

function bearingToVector(bearingDeg, elevationDeg, out) {
  const a = bearingDeg * DEG;
  const e = elevationDeg * DEG;
  const c = Math.cos(e);
  return out.set(Math.sin(a) * c, Math.sin(e), -Math.cos(a) * c);
}

/**
 * layout.js is the authority on the sun — but only while it stays inside the
 * north-east window the composition depends on. A value outside it (the stale
 * south-east 118 shipped in the first pass, say) would put every plaza-facing
 * wall in shade, so it is overridden loudly rather than obeyed quietly.
 */
function heroSunAngles() {
  const az = Number(SUN.azimuthDeg);
  const el = Number(SUN.elevationDeg);
  const ok = Number.isFinite(az) && Number.isFinite(el) &&
    az >= SUN_AZIMUTH_WINDOW[0] && az <= SUN_AZIMUTH_WINDOW[1] &&
    el >= SUN_ELEVATION_WINDOW[0] && el <= SUN_ELEVATION_WINDOW[1];
  if (ok) return { az, el };
  console.warn(
    `[sky] layout.js SUN (azimuth ${SUN.azimuthDeg}, elevation ${SUN.elevationDeg}) is ` +
    'outside the art-directed north-east window; using ' +
    `${ART_SUN_AZIMUTH_DEG}/${ART_SUN_ELEVATION_DEG} instead. Set ` +
    `SUN.azimuthDeg = ${ART_SUN_AZIMUTH_DEG} and SUN.elevationDeg = ` +
    `${ART_SUN_ELEVATION_DEG} in layout.js so the terrain shading agrees.`);
  return { az: ART_SUN_AZIMUTH_DEG, el: ART_SUN_ELEVATION_DEG };
}

/** The orthonormal frame the daily circle is swept in. `v` is the direction of
 * travel at the transit; `u` is the transit (highest) point. */
function poleFrame(pole, u, v) {
  u.set(0, 1, 0).addScaledVector(pole, -pole.y);
  // Degenerate only if the pole is exactly vertical, which no solve produces.
  if (u.lengthSq() < 1e-8) u.set(1, 0, 0).addScaledVector(pole, -pole.x);
  u.normalize();
  v.crossVectors(u, pole).normalize();
}

/**
 * Find the pole whose circle carries `hero` at phase `psi0`. Candidate poles
 * form a ring at angle `rho` from `hero`; the phase error along that ring is
 * continuous, so a scan-plus-bisection is exact enough (the residual below is
 * ~1e-9 rad) and costs a fraction of a millisecond, once.
 *
 * Two roots exist. Take the one with the higher transit: a high noon reads as a
 * summer day, and it keeps the midday sun on the side of the plaza the player
 * spawns facing.
 */
function solveSunPole(hero, rho, psi0) {
  const e = new THREE.Vector3(0, 1, 0).addScaledVector(hero, -hero.y).normalize();
  const f = new THREE.Vector3().crossVectors(hero, e).normalize();
  const cr = Math.cos(rho);
  const sr = Math.sin(rho);
  const p = new THREE.Vector3();
  const u = new THREE.Vector3();
  const v = new THREE.Vector3();

  const poleAt = (theta) => p.copy(hero).multiplyScalar(cr)
    .addScaledVector(e, Math.cos(theta) * sr)
    .addScaledVector(f, Math.sin(theta) * sr)
    .normalize();

  const phaseError = (theta) => {
    poleFrame(poleAt(theta), u, v);
    let d = Math.atan2(hero.dot(v), hero.dot(u)) - psi0;
    while (d > Math.PI) d -= Math.PI * 2;
    while (d < -Math.PI) d += Math.PI * 2;
    return d;
  };

  let best = null;
  let prevTheta = -Math.PI;
  let prevErr = phaseError(prevTheta);
  for (let i = 1; i <= 360; i++) {
    const theta = -Math.PI + (i / 360) * Math.PI * 2;
    const err = phaseError(theta);
    // `< 1` rejects the 2*PI wrap, which is a jump rather than a crossing.
    if (prevErr * err < 0 && Math.abs(err - prevErr) < 1) {
      let lo = prevTheta, hi = theta, loErr = prevErr;
      for (let k = 0; k < 60; k++) {
        const mid = (lo + hi) * 0.5;
        const midErr = phaseError(mid);
        if (loErr * midErr <= 0) hi = mid; else { lo = mid; loErr = midErr; }
      }
      const pole = poleAt((lo + hi) * 0.5).clone();
      poleFrame(pole, u, v);
      const transitY = pole.y * cr + u.y * sr;
      if (!best || transitY > best.transitY) best = { pole, transitY };
    }
    prevTheta = theta;
    prevErr = err;
  }

  if (!best) {
    // Never seen; keep the world lit rather than throwing at module scope.
    console.warn('[sky] sun-path solve found no pole; using the baked fallback axis.');
    // The exact pole the scan above produces for SUN_RHO = 56 and a hero of
    // (57, 36), so the fallback lands on the authored composition instead of the
    // previous rho's axis.
    return new THREE.Vector3(0.853101031, 0.291151121, 0.432954565);
  }
  return best.pole;
}

const _heroSun = (() => {
  const a = heroSunAngles();
  return bearingToVector(a.az, a.el, new THREE.Vector3());
})();

const _pole = solveSunPole(_heroSun, SUN_RHO, (DEFAULT_TIME_OF_DAY - 0.5) * Math.PI * 2);
const _basisU = new THREE.Vector3();
const _basisV = new THREE.Vector3();
poleFrame(_pole, _basisU, _basisV);

/**
 * Unit vector pointing AT the sun for a normalised time of day.
 * @param {number} t01 0 = midnight, 0.5 = noon
 * @param {THREE.Vector3} out
 */
export function sunDirectionAt(t01, out) {
  const psi = (t01 - 0.5) * Math.PI * 2;
  const cr = Math.cos(SUN_RHO);
  const c = Math.cos(psi) * Math.sin(SUN_RHO);
  const s = Math.sin(psi) * Math.sin(SUN_RHO);
  out.set(
    _pole.x * cr + _basisU.x * c + _basisV.x * s,
    _pole.y * cr + _basisU.y * c + _basisV.y * s,
    _pole.z * cr + _basisU.z * c + _basisV.z * s,
  );
  return out.normalize();
}

// Cheap insurance: the whole composition hangs off this one direction, so say so
// out loud if the solve ever stops landing on it.
{
  const d = sunDirectionAt(DEFAULT_TIME_OF_DAY, new THREE.Vector3());
  const offDeg = Math.acos(Math.min(1, Math.max(-1, d.dot(_heroSun)))) / DEG;
  if (offDeg > 0.05) {
    console.warn(`[sky] hero sun is ${offDeg.toFixed(3)} deg off the authored direction.`);
  }
}

/* -------------------------------------------------------------------------- */
/* Analytic sky, evaluated on the CPU                                          */
/* -------------------------------------------------------------------------- */

/**
 * The JS twin of the fragment shader's scattering. Fog, the hemisphere light,
 * the ground-bounce colour and the sun's own tint are all read out of this, so
 * they cannot drift out of agreement with what you can see.
 */
class Atmosphere {
  constructor() {
    this.sun = new THREE.Vector3(0, 1, 0);
    this.betaR = [0, 0, 0];
    this.betaM = [0, 0, 0];
    this.sunE = 0;
    /** Twilight/night lift — see NIGHT_SKY / DUSK_SKY. */
    this.nightSky = [0, 0, 0];
  }

  configure(sunDir, turbidity, rayleigh) {
    this.sun.copy(sunDir);
    const sunfade = 1 - Math.min(1, Math.max(0, 1 - Math.exp(sunDir.y)));
    const rc = rayleigh - (1 - sunfade);
    const cM = 0.2 * turbidity * 10e-18;
    for (let i = 0; i < 3; i++) {
      this.betaR[i] = TOTAL_RAYLEIGH[i] * rc;
      this.betaM[i] = 0.434 * cM * MIE_CONST[i] * MIE_COEFFICIENT;
    }
    const cosZ = Math.max(-1, Math.min(1, sunDir.y));
    this.sunE = SUN_EE * Math.max(0, 1 - Math.exp(-((CUTOFF_ANGLE - Math.acos(cosZ)) / STEEPNESS)));
  }

  /**
   * Atmospheric transmittance along the path to the sun. The elevation is
   * floored at ~1.1 deg: past that the path length runs away and the tint
   * degenerates to pure red, which is not a colour any sunset has.
   */
  sunExtinction(out) {
    const za = Math.acos(Math.max(0.02, this.sun.y));
    const inv = 1 / (Math.cos(za) + 0.15 * Math.pow(93.885 - (za * 180) / Math.PI, -1.253));
    for (let i = 0; i < 3; i++) {
      out[i] = Math.exp(-(this.betaR[i] * RAYLEIGH_ZENITH_LENGTH * inv +
        this.betaM[i] * MIE_ZENITH_LENGTH * inv));
    }
    return out;
  }

  /** HDR radiance looking along (dx,dy,dz). Writes 3 floats into `out`. */
  radiance(dx, dy, dz, out) {
    const za = Math.acos(Math.max(0, dy));
    const inv = 1 / (Math.cos(za) + 0.15 * Math.pow(93.885 - (za * 180) / Math.PI, -1.253));
    const sR = RAYLEIGH_ZENITH_LENGTH * inv;
    const sM = MIE_ZENITH_LENGTH * inv;
    const cosTheta = dx * this.sun.x + dy * this.sun.y + dz * this.sun.z;
    const ct = cosTheta * 0.5 + 0.5;
    const rPhase = 0.05968310365946075 * (1 + ct * ct);
    const g2 = MIE_G * MIE_G;
    const mPhase = 0.07957747154594767 *
      ((1 - g2) / Math.pow(Math.max(1e-4, 1 - 2 * MIE_G * cosTheta + g2), 1.5));
    const mixT = Math.min(1, Math.max(0, Math.pow(Math.max(0, 1 - this.sun.y), 5)));
    for (let i = 0; i < 3; i++) {
      const bR = this.betaR[i];
      const bM = this.betaM[i];
      const Fex = Math.exp(-(bR * sR + bM * sM));
      const ratio = (bR * rPhase + bM * mPhase) / (bR + bM);
      let Lin = Math.pow(Math.max(0, this.sunE * ratio * (1 - Fex)), 1.5);
      Lin *= (1 - mixT) + Math.pow(Math.max(0, this.sunE * ratio * Fex), 0.5) * mixT;
      out[i] = (Lin + 0.1 * Fex) * SKY_RADIANCE_SCALE + NIGHT_LIFT[i] +
        this.nightSky[i] * (0.55 + 0.45 * Math.pow(1 - Math.max(0, dy), 2));
    }
    return out;
  }
}

const NIGHT_LIFT = [0, 0.0003, 0.00075];

/*
 * Preetham's earth-shadow hack collapses `sunE` well before the sun reaches the
 * horizon, so the model paints an almost black sky from about 5 deg elevation
 * down — twilight in reality is still a luminous deep blue. These two terms put
 * that back: a broad night floor and a gaussian bump centred on the horizon.
 */
const NIGHT_SKY = [0.0040, 0.0060, 0.0160];
const DUSK_SKY = [0.0260, 0.0240, 0.0550];

/* -------------------------------------------------------------------------- */
/* Tileable gradient noise, baked to a texture                                 */
/* -------------------------------------------------------------------------- */

/**
 * Perlin-style gradient noise whose lattice wraps at `period`, so an fbm built
 * from it tiles exactly. rng.js's `valueNoise2D` wraps at a fixed 256, which
 * would force the base octave to one lattice cell per texel — useless for cloud
 * shapes — hence this local variant. Gradients still come from `Rng`, so the
 * sky is byte-identical between reloads.
 */
function periodicNoise(period, rng) {
  const g = new Float32Array(period * period * 2);
  for (let i = 0; i < period * period; i++) {
    const a = rng.next() * Math.PI * 2;
    g[i * 2] = Math.cos(a);
    g[i * 2 + 1] = Math.sin(a);
  }
  return function noise(x, y) {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const xf = x - xi;
    const yf = y - yi;
    const x0 = ((xi % period) + period) % period;
    const y0 = ((yi % period) + period) % period;
    const x1 = (x0 + 1) % period;
    const y1 = (y0 + 1) % period;
    const i00 = (y0 * period + x0) * 2;
    const i10 = (y0 * period + x1) * 2;
    const i01 = (y1 * period + x0) * 2;
    const i11 = (y1 * period + x1) * 2;
    const n00 = g[i00] * xf + g[i00 + 1] * yf;
    const n10 = g[i10] * (xf - 1) + g[i10 + 1] * yf;
    const n01 = g[i01] * xf + g[i01 + 1] * (yf - 1);
    const n11 = g[i11] * (xf - 1) + g[i11 + 1] * (yf - 1);
    const u = xf * xf * xf * (xf * (xf * 6 - 15) + 10);
    const v = yf * yf * yf * (yf * (yf * 6 - 15) + 10);
    const a = n00 + u * (n10 - n00);
    const b = n01 + u * (n11 - n01);
    return a + v * (b - a);
  };
}

/** fbm over `octaves` periodic noises; tiles over uv in [0,1). */
function periodicFbm(basePeriod, octaves, rng) {
  const layers = [];
  for (let i = 0; i < octaves; i++) {
    const freq = basePeriod << i;
    layers.push({ noise: periodicNoise(freq, rng), freq });
  }
  return function fbm(u, v) {
    let amp = 0.5;
    let sum = 0;
    let norm = 0;
    for (let i = 0; i < layers.length; i++) {
      sum += amp * layers[i].noise(u * layers[i].freq, v * layers[i].freq);
      norm += amp;
      amp *= 0.5;
    }
    return sum / norm; // roughly [-0.7, 0.7]
  };
}

function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v | 0;
}

/**
 * RGBA cloud noise atlas. R = cumulus mass, G = medium detail, B = erosion,
 * A = domain-warp field. Four mutually prime base periods so the channels never
 * line up into a visible grid.
 */
function buildCloudNoise(size, anisotropy) {
  const rng = new Rng('hollowbrook-clouds');
  // Three octaves from a 4-cell lattice, not five from a 5-cell one. This is the
  // single biggest lever on cloud SIZE: every extra octave adds detail at the
  // scale of the coverage threshold, which chops the contour into fragments. On
  // 5 octaves the mean contiguous cloud island measured 0.27 shell units — a
  // field of small puffs, however large you make the base. On 3 it is 0.6-0.8,
  // and the deck reads as cumulus masses. Fine detail comes from G and B, where
  // the amplitude is under our control.
  const fR = periodicFbm(4, 3, rng.fork('mass'));
  const fG = periodicFbm(11, 3, rng.fork('detail'));
  const fB = periodicFbm(23, 2, rng.fork('erode'));
  const fA = periodicFbm(3, 2, rng.fork('warp'));
  const data = new Uint8Array(size * size * 4);
  const inv = 1 / size;
  let p = 0;
  for (let y = 0; y < size; y++) {
    const v = y * inv;
    for (let x = 0; x < size; x++) {
      const u = x * inv;
      data[p++] = clamp255((fR(u, v) * 0.72 + 0.5) * 255);
      data[p++] = clamp255((fG(u, v) * 0.72 + 0.5) * 255);
      data[p++] = clamp255((fB(u, v) * 0.72 + 0.5) * 255);
      data[p++] = clamp255((fA(u, v) * 0.72 + 0.5) * 255);
    }
  }
  const tex = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = anisotropy;
  tex.colorSpace = THREE.NoColorSpace; // data, not colour
  tex.needsUpdate = true;
  return tex;
}

/* -------------------------------------------------------------------------- */
/* Shaders                                                                     */
/* -------------------------------------------------------------------------- */

const SKY_VERT = /* glsl */`
varying vec3 vWorldPosition;

void main() {
  vec4 world = modelMatrix * vec4( position, 1.0 );
  vWorldPosition = world.xyz;
  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
  // Pin to the far plane: the dome can then never be clipped by camera.far and
  // never fights real geometry for depth, whatever radius it happens to have.
  gl_Position.z = gl_Position.w;
}
`;

const SKY_FRAG = /* glsl */`
varying vec3 vWorldPosition;

uniform vec3  uSunDirection;
uniform vec3  uBetaR;
uniform vec3  uBetaM;
uniform float uSunE;
uniform float uSunFadeMix;
uniform float uSunDisc;
uniform vec3  uSunTint;
uniform vec3  uGroundColour;
uniform vec3  uHorizonTint;
uniform float uHorizonTintStrength;
uniform vec3  uNightSky;

uniform sampler2D uNoise;
uniform vec2  uDrift0;
uniform vec2  uDrift1;
uniform float uCoverage;
uniform float uClusterBias;
uniform float uCloudSoft;
uniform float uCloudScale;
uniform float uShellK;
uniform vec3  uCloudLit;
uniform vec3  uCloudShadow;
uniform float uSilverLining;
uniform vec2  uSunPlaneStep;
uniform float uMarchSteps;
uniform float uNight;
uniform float uCloudAmount;
uniform float uEnvGain;

#define SKY_PI 3.141592653589793
const float RAYLEIGH_ZENITH = 8.4e3;
const float MIE_ZENITH = 1.25e3;
const float MIE_G_C = ${MIE_G.toFixed(4)};
const float SKY_SCALE = ${SKY_RADIANCE_SCALE.toFixed(6)};

// Decorrelating rotations between noise lookups. Without these the channels
// line up and the cloud field visibly repeats every few tile widths.
const mat2 ROT1 = mat2( 0.8253, -0.5646,  0.5646,  0.8253 );
const mat2 ROT2 = mat2( 0.3624,  0.9320, -0.9320,  0.3624 );
const mat2 ROT3 = mat2( -0.7071, 0.7071, -0.7071, -0.7071 );

float hgPhase( float cosTheta, float g ) {
  float g2 = g * g;
  return ( 0.25 / SKY_PI ) * ( 1.0 - g2 ) /
    pow( max( 1e-4, 1.0 - 2.0 * g * cosTheta + g2 ), 1.5 );
}

/** Preetham in-scattering for a view direction. Returns HDR radiance. */
vec3 atmosphere( vec3 dir ) {
  float zenithAngle = acos( max( 0.0, dir.y ) );
  float inv = 1.0 / ( cos( zenithAngle ) +
    0.15 * pow( 93.885 - ( zenithAngle * 180.0 ) / SKY_PI, -1.253 ) );
  float sR = RAYLEIGH_ZENITH * inv;
  float sM = MIE_ZENITH * inv;

  vec3 Fex = exp( -( uBetaR * sR + uBetaM * sM ) );

  float cosTheta = dot( dir, uSunDirection );
  float ct = cosTheta * 0.5 + 0.5;
  float rPhase = 0.05968310365946075 * ( 1.0 + ct * ct );
  float mPhase = hgPhase( cosTheta, MIE_G_C );

  vec3 ratio = ( uBetaR * rPhase + uBetaM * mPhase ) / ( uBetaR + uBetaM );
  vec3 Lin = pow( max( vec3( 0.0 ), uSunE * ratio * ( 1.0 - Fex ) ), vec3( 1.5 ) );
  Lin *= mix( vec3( 1.0 ),
    pow( max( vec3( 0.0 ), uSunE * ratio * Fex ), vec3( 0.5 ) ), uSunFadeMix );

  vec3 col = ( Lin + 0.1 * Fex ) * SKY_SCALE + vec3( 0.0, 0.0003, 0.00075 );

  // Twilight floor: Preetham goes black several degrees before sunset.
  col += uNightSky * ( 0.55 + 0.45 * pow( 1.0 - max( 0.0, dir.y ), 2.0 ) );

  // A little authored warmth in the horizon band on the sun's side: Preetham
  // alone reads slightly too neutral for a 36 degree afternoon.
  float band = pow( max( 0.0, 1.0 - abs( dir.y ) ), 7.0 );
  vec2 hv = normalize( dir.xz + vec2( 1e-5 ) );
  vec2 hs = normalize( uSunDirection.xz + vec2( 1e-5 ) );
  float side = max( 0.0, dot( hv, hs ) );
  col += uHorizonTint * ( band * ( 0.25 + 0.75 * side ) * uHorizonTintStrength );

  // Solar disc, deliberately clamped: the physical value (~6e6) would turn
  // bloom into a white sheet and saturate the half-float composer target.
  float disc = smoothstep( 0.99990, 0.99997, cosTheta );
  col += uSunTint * ( disc * uSunDisc );

  return col;
}

/* --- clouds ------------------------------------------------------------- */

vec4 nz( vec2 p ) { return texture2D( uNoise, p ); }

/**
 * Slow, large domain warp — an advection field several cloud diameters wide.
 * This is what turns round blobs into leaning towers and anvils.
 */
vec2 warpAt( vec2 p ) {
  float a = nz( p * 0.055 ).a;
  float b = nz( ROT1 * p * 0.091 + vec2( 0.37, 0.61 ) ).a;
  return ( vec2( a, b ) - 0.5 ) * 1.6;
}

/**
 * Weather. One cell spans several cloud diameters, so the deck breaks up into
 * crowded clusters with wide open blue between them instead of an even field of
 * identical puffs.
 */
float clusterAt( vec2 p ) {
  return nz( p * 0.021 ).r * 0.62 +
         nz( ROT2 * p * 0.044 + vec2( 0.63, 0.19 ) ).g * 0.38;
}

/**
 * Density of one horizontal slice through the deck (2 taps).
 *
 * The detail amplitudes are small on purpose: at 0.26 the medium octave was as
 * tall as the coverage transition itself, so it cut every mass into confetti.
 */
float slabDensity( vec2 q, vec2 w, float cov ) {
  float mass = nz( ( q + w ) * uCloudScale ).r;
  float med  = nz( ROT2 * ( q + w * 0.6 ) * uCloudScale * 2.0 + vec2( 0.11, 0.37 ) ).g;
  return smoothstep( cov, cov + uCloudSoft, mass + 0.12 * ( med - 0.5 ) );
}

/** Same, plus the erosion octave that frays the silhouette (3 taps). */
float slabShape( vec2 q, vec2 w, float cov ) {
  float mass = nz( ( q + w ) * uCloudScale ).r;
  float med  = nz( ROT2 * ( q + w * 0.6 ) * uCloudScale * 2.0 + vec2( 0.11, 0.37 ) ).g;
  float fine = nz( ROT3 * ( q + w * 0.3 ) * uCloudScale * 4.5 ).b;
  return smoothstep( cov, cov + uCloudSoft,
    mass + 0.12 * ( med - 0.5 ) + 0.06 * ( fine - 0.5 ) );
}

void main() {
  vec3 dir = normalize( vWorldPosition - cameraPosition );

  vec3 col = atmosphere( dir );

  // Ground hemisphere. Without it Preetham clamps to the horizon value below
  // y = 0 and the env map ends up lighting the whole village from underneath.
  float above = smoothstep( -0.045, 0.02, dir.y );
  col = mix( uGroundColour, col, above );

  /* ---- stars, only worth evaluating once the sun is well down ---- */
  if ( uNight > 0.002 && dir.y > 0.0 ) {
    vec3 s = dir * 260.0;
    vec3 cell = floor( s );
    float h = fract( sin( dot( cell, vec3( 12.9898, 78.233, 37.719 ) ) ) * 43758.5453 );
    if ( h > 0.986 ) {
      vec3 jitter = vec3( fract( h * 71.0 ), fract( h * 131.0 ), fract( h * 197.0 ) );
      float d = length( fract( s ) - jitter );
      // GLSL smoothstep is undefined when edge0 > edge1, so invert instead.
      float star = ( 1.0 - smoothstep( 0.0, 0.42, d ) ) *
        ( 0.35 + 0.65 * fract( h * 313.0 ) );
      col += vec3( 0.85, 0.90, 1.0 ) * star * uNight * 1.6 *
        smoothstep( 0.0, 0.18, dir.y );
    }
  }

  /* ---- cumulus ---- */
  if ( dir.y > 0.0 && uCloudAmount > 0.001 ) {
    float cosTheta = dot( dir, uSunDirection );
    float silver = hgPhase( cosTheta, 0.76 ) * uSilverLining;

    // A curved shell, not a flat plane: the ray length to the deck stays finite
    // at the horizon (tMax below), so the field converges into a haze band
    // instead of shearing into infinitely fine noise behind a hard cut-off.
    float bk = uShellK * max( dir.y, 0.0 );
    float t0 = sqrt( bk * bk + 2.0 * uShellK + 1.0 ) - bk;
    float tMax = sqrt( 2.0 * uShellK + 1.0 );

    vec2 s = dir.xz * t0;          // the column of air this pixel looks through
    vec2 q0 = s + uDrift0;

    // Crowded here, open blue there.
    float cov = uCoverage + uClusterBias * ( 0.5 - clusterAt( q0 ) );

    vec2 w0 = warpAt( q0 );

    /* -- three slices up through the deck ------------------------------- *
     * A slice at relative altitude h looks at the same field sampled at
     * q * h, which is the correct parallax for a higher layer: the top of a
     * tower shows up nearer the zenith than its base. Stacking three of them
     * with a rising coverage threshold gives real vertical development —
     * billowing, tapering tops instead of one flat cut-out — and the front-to-
     * back composite below is what puts sunlit shoulders behind grey bases. */
    float d0 = slabShape( q0, w0, cov );
    float d1 = slabShape( s * 1.22 + uDrift0, w0 * 1.06, cov + 0.035 );
    float d2 = slabDensity( s * 1.50 + uDrift0, w0 * 1.12, cov + 0.085 );

    // Taps toward the sun across the base slice. This is what carves the bright
    // sunlit shoulder and the heavy grey underside.
    //
    // Deliberately NOT guarded by a density test: a texture fetch inside
    // non-uniform control flow has undefined derivatives, and the wrong mip
    // level shows up as shimmering blocks along every cloud silhouette. A few
    // wasted taps on clear sky is the cheaper mistake.
    float occl = 0.0;
    vec2 sp = q0;
    for ( int i = 0; i < 4; i ++ ) {
      if ( float( i ) >= uMarchSteps ) break;
      sp += uSunPlaneStep;
      occl += slabDensity( sp, w0, cov );
    }

    // The base carries the whole tower standing on it; the top carries nothing.
    float litBase = exp( -( occl * 0.80 + ( d1 + d2 ) * 1.15 ) );
    float litMid  = exp( -( occl * 0.45 + d2 * 0.70 ) );
    float litTop  = exp( -occl * 0.18 );

    // Opacity from optical thickness, not from coverage: a cumulus core is
    // opaque and only its edges are thin. Summing coverages instead left every
    // mass a 40%-alpha veil that the sky showed straight through.
    float aT = 1.0 - exp( -2.6 * ( d0 + 0.85 * d1 + 0.70 * d2 ) );

    // Colour weights are front to back — the base slice is the nearest, so it
    // hides what stands behind it, and the sunlit shoulders show up in the gaps.
    float wBase = d0;
    float wMid = ( 1.0 - d0 ) * d1 * 0.85;
    float wTop = ( 1.0 - d0 ) * ( 1.0 - d1 ) * d2 * 0.70;
    float wSum = wBase + wMid + wTop;

    vec3 body = ( mix( uCloudShadow, uCloudLit, litBase ) * wBase +
                  mix( uCloudShadow, uCloudLit, litMid ) * wMid +
                  mix( uCloudShadow, uCloudLit, litTop ) * wTop ) / max( wSum, 1e-4 );

    // Silver lining: thin edges transmit, so weight by (1 - density).
    body += uCloudLit * ( silver * ( 1.0 - aT ) * ( 0.35 + 0.65 * litTop ) );

    // Aerial perspective on the far rim of the deck. Both the colour and the
    // opacity go, so the deck dissolves into the sky rather than ending on a
    // line.
    float rim = smoothstep( 2.4, max( 3.0, tMax * 0.92 ), t0 );
    body = mix( body, col, rim * 0.90 );

    float a = clamp( aT * uCloudAmount * ( 1.0 - 0.35 * rim ), 0.0, 1.0 ) *
      smoothstep( 0.0, 0.035, dir.y );
    col = mix( col, body, a );

    /* -- high thin deck: no march, just a sun-side gradient ------------- */
    float d3 = slabShape( s * 2.60 + uDrift1, w0 * 0.5, cov + 0.22 );
    float grad = 0.45 + 0.55 * smoothstep( -0.4, 0.7, cosTheta );
    vec3 high = mix( uCloudShadow, uCloudLit, grad ) * 0.92 + uCloudLit * silver * 0.30;
    high = mix( high, col, rim * 0.95 );
    col = mix( col, high, d3 * uCloudAmount * 0.40 * ( 1.0 - a ) *
      smoothstep( 0.0, 0.05, dir.y ) );
  }

  // uEnvGain is 1.0 for the visible dome and < 1 only while the cube camera is
  // capturing the PMREM source, which is the only handle this project has on how
  // hard the sky lights the village.
  gl_FragColor = vec4( max( col, vec3( 0.0 ) ) * uEnvGain, 1.0 );
}
`;

/* -------------------------------------------------------------------------- */
/* Factory                                                                     */
/* -------------------------------------------------------------------------- */

const _tmpVec = new THREE.Vector3();
const _fex = [0, 0, 0];
const _rgb = [0, 0, 0];
const _acc = [0, 0, 0];

/**
 * Build the sky dome and the environment map it bakes into.
 *
 * @param {Object} args
 * @param {THREE.WebGLRenderer} args.renderer
 * @param {THREE.Scene} args.scene   the dome adds itself
 * @param {Object} args.quality      active preset from config.js
 * @returns {Promise<Object>} { group, envMap, sunDirection, sunColour,
 *   skyColour, setTimeOfDay, update, dispose, stats } plus a few extras that
 *   lighting.js reads (horizonColour, groundColour, sunLevel, sampleRadiance).
 */
export async function createSky({ renderer, scene, quality }) {
  const q = quality || {};
  const group = new THREE.Group();
  group.name = 'sky';

  const noiseSize = (q.textureSize | 0) <= 512 ? 128 : 256;
  const noiseTex = buildCloudNoise(noiseSize, Math.min(8, q.anisotropy || 4));

  const atmos = new Atmosphere();

  const uniforms = {
    uSunDirection: { value: new THREE.Vector3(0, 1, 0) },
    uBetaR: { value: new THREE.Vector3() },
    uBetaM: { value: new THREE.Vector3() },
    uSunE: { value: 0 },
    uSunFadeMix: { value: 0 },
    uSunDisc: { value: SUN_DISC_RADIANCE },
    uSunTint: { value: new THREE.Color(1, 1, 1) },
    uGroundColour: { value: new THREE.Color(0.1, 0.1, 0.09) },
    uHorizonTint: { value: new THREE.Color(1.0, 0.56, 0.25) },
    uHorizonTintStrength: { value: 0.55 },
    uNightSky: { value: new THREE.Vector3() },

    uNoise: { value: noiseTex },
    uDrift0: { value: new THREE.Vector2() },
    uDrift1: { value: new THREE.Vector2() },
    /**
     * Mean coverage threshold, measured rather than guessed: 0.51 with the
     * cluster bias below puts 26% of the sky above 5 deg under cloud, in masses
     * averaging 16 deg of arc with clear gaps averaging 15 deg. That is the
     * reference's "big cumulus, plenty of open blue".
     */
    uCoverage: { value: 0.51 },
    /** Swing either side of it. Dense clusters vs near-clear sky. */
    uClusterBias: { value: 0.26 },
    uCloudSoft: { value: 0.14 },
    /**
     * uv per unit of shell distance. The mass octave is 1/4 of a tile, so a
     * cumulus mass comes out ~1/0.16/4 = 1.6 shell heights across — a 2.4 km
     * cloud on a 1.5 km base. This is the single knob for cloud SIZE.
     */
    uCloudScale: { value: 0.16 },
    /** Planet radius in cloud-base heights. Sets where the deck converges. */
    uShellK: { value: 20.0 },
    uCloudLit: { value: new THREE.Color(1, 1, 1) },
    uCloudShadow: { value: new THREE.Color(0.4, 0.46, 0.6) },
    uSilverLining: { value: 0.9 },
    uSunPlaneStep: { value: new THREE.Vector2(0.2, 0) },
    uMarchSteps: { value: q.name === 'low' ? 2 : 3 },
    uNight: { value: 0 },
    uCloudAmount: { value: 1 },
    uEnvGain: { value: 1 },
  };

  const material = new THREE.ShaderMaterial({
    name: 'HollowbrookSky',
    uniforms,
    vertexShader: SKY_VERT,
    fragmentShader: SKY_FRAG,
    side: THREE.BackSide,
    depthWrite: false,
    depthTest: true,
    fog: false,
    toneMapped: false,      // OutputPass owns tone mapping
    transparent: false,
  });

  // Radius is irrelevant (the vertex shader pins z to the far plane); a small
  // dome recentred on the eye every frame keeps the camera inside it always.
  const geometry = new THREE.SphereGeometry(1, 48, 24);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = 'skyDome';
  mesh.frustumCulled = false;
  mesh.renderOrder = -1000;   // drawn first, writes no depth
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  mesh.userData.noCollide = true;
  group.add(mesh);
  scene.add(group);

  /* ---------------------------------------------------------------- HDRI */

  const envSize = Math.max(64, q.envSize | 0) || 256;
  const cubeTarget = new THREE.WebGLCubeRenderTarget(envSize, {
    type: THREE.HalfFloatType,
    format: THREE.RGBAFormat,
    colorSpace: THREE.LinearSRGBColorSpace,
    generateMipmaps: false,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
  });
  const cubeCamera = new THREE.CubeCamera(0.05, 20, cubeTarget);

  // The bake sees only the dome. A private scene keeps the village — and its
  // lights and shadow maps — out of a capture that must be pure sky.
  const bakeScene = new THREE.Scene();
  const bakeMesh = new THREE.Mesh(geometry, material);
  bakeMesh.frustumCulled = false;
  bakeScene.add(bakeMesh);

  const pmrem = new THREE.PMREMGenerator(renderer);
  // Warming the shader up front hides a ~10 ms hitch on the first bake. If the
  // driver refuses, fail soft — the bake below will just be a little slower.
  try { pmrem.compileCubemapShader(); } catch (err) {
    console.warn('[sky] PMREM shader precompile skipped:', err);
  }
  /**
   * Allocated by the FIRST fromCubemap() call and reused by every one after.
   * The first call must pass null: PMREMGenerator only builds its ping-pong
   * target, LOD planes and filter materials inside _allocateTargets(), which it
   * skips entirely when you hand it a target. Pass a target on call one and it
   * throws on an empty _lodMeshes.
   * @type {THREE.WebGLRenderTarget|null}
   */
  let pmremTarget = null;

  const stats = {
    envSize,
    noiseTexture: `${noiseSize}x${noiseSize}`,
    bakes: 0,
    lastBakeMs: 0,
    timeOfDay: DEFAULT_TIME_OF_DAY,
    sunElevationDeg: 0,
    sunAzimuthDeg: 0,
    envBakeGain: ENV_BAKE_GAIN,
    drawCalls: 1,
    triangles: geometry.index ? geometry.index.count / 3 : 0,
  };

  const sunDirection = new THREE.Vector3();
  const sunColour = new THREE.Color(SUN.colour);
  const skyColour = new THREE.Color(0.25, 0.45, 0.85);
  const horizonColour = new THREE.Color(0.7, 0.72, 0.72);
  const groundColour = new THREE.Color(0.12, 0.11, 0.08);

  let sunLevel = 1;
  let timeOfDay = -1;
  let bakeTimer = 0;
  let lastBakeAt = -1e9;
  let envDirty = false;
  let disposed = false;
  let refSunLuma = 1;
  /** SUN.colour divided by the reference hue: makes t=0.66 exact by design. */
  const SUN_CORRECTION = [1, 1, 1];
  const SUN_REF_LUMA =
    0.2126 * SUN_LINEAR.r + 0.7152 * SUN_LINEAR.g + 0.0722 * SUN_LINEAR.b;

  const drift0 = uniforms.uDrift0.value;
  const drift1 = uniforms.uDrift1.value;
  const WIND_X = -0.8290;   // normalised (-0.83, 0.56)
  const WIND_Z = 0.5592;

  /* ------------------------------------------------------------- baking */

  const api = {
    group,
    envMap: null,
    sunDirection,
    sunColour,
    skyColour,
    /* extras: not part of the minimum contract, but lighting.js reads them. */
    horizonColour,
    groundColour,
    material,
    get sunLevel() { return sunLevel; },
    get timeOfDay() { return timeOfDay; },
    setTimeOfDay,
    update,
    dispose,
    stats,

    /** HDR sky radiance in a direction — used for fog and probe colours. */
    sampleRadiance(dir, out) {
      atmos.radiance(dir.x, dir.y, dir.z, _rgb);
      return (out || new THREE.Color())
        .setRGB(_rgb[0], _rgb[1], _rgb[2], THREE.LinearSRGBColorSpace);
    },

    /** Force the environment map to rebuild right now. */
    rebakeEnvironment() { bakeNow(); },
  };

  function bakeNow() {
    if (disposed) return;
    const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const t0 = now();
    const prevDisc = uniforms.uSunDisc.value;
    const prevGain = uniforms.uEnvGain.value;
    // A sub-texel solar disc becomes a blocky hot square after PMREM, and the
    // DirectionalLight already carries the direct term.
    uniforms.uSunDisc.value = 0;
    // Capture the sky darker than it is drawn: this is the indirect-light level.
    uniforms.uEnvGain.value = ENV_BAKE_GAIN;
    try {
      cubeCamera.update(renderer, bakeScene);
      pmremTarget = pmrem.fromCubemap(cubeTarget.texture, pmremTarget);
      api.envMap = pmremTarget.texture;
      // Object identity never changes (we reuse the target), so this is only
      // needed for the very first bake — but it costs nothing to be sure.
      if (scene.environment !== pmremTarget.texture) scene.environment = pmremTarget.texture;
    } catch (err) {
      console.warn('[sky] environment bake failed:', err);
      // PMREM restores the previous target itself; only a throw can leave one
      // bound, and a stuck render target is a black screen.
      try { renderer.setRenderTarget(null); } catch { /* renderer is gone */ }
    } finally {
      uniforms.uSunDisc.value = prevDisc;
      uniforms.uEnvGain.value = prevGain;
    }
    lastBakeAt = now();
    stats.lastBakeMs = Math.round((lastBakeAt - t0) * 100) / 100;
    stats.bakes++;
    envDirty = false;
  }

  function scheduleBake() {
    envDirty = true;
    if (bakeTimer || disposed) return;
    const now = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const wait = Math.max(0, REBAKE_INTERVAL_MS - (now - lastBakeAt));
    bakeTimer = setTimeout(() => {
      bakeTimer = 0;
      if (envDirty) bakeNow();
    }, wait);
  }

  /* ------------------------------------------------------ time of day */

  function turbidityAt(above) { return BASE_TURBIDITY + 2.4 * Math.pow(1 - above, 3.0); }
  function rayleighAt(above) { return BASE_RAYLEIGH + 0.5 * Math.pow(1 - above, 2.0); }

  function applyTime(t01) {
    const t = ((t01 % 1) + 1) % 1;
    timeOfDay = t;
    sunDirectionAt(t, sunDirection);
    const above = Math.max(0, sunDirection.y);

    atmos.configure(sunDirection, turbidityAt(above), rayleighAt(above));

    // Twilight / night floor. Must be set before any radiance() call below,
    // because fog and the probe colours read straight out of the model.
    const twilight = smoothstep(0.22, -0.25, sunDirection.y);
    const duskBump = Math.exp(-Math.pow(sunDirection.y / 0.10, 2));
    for (let i = 0; i < 3; i++) {
      atmos.nightSky[i] = NIGHT_SKY[i] * twilight + DUSK_SKY[i] * duskBump;
    }
    uniforms.uNightSky.value.set(
      atmos.nightSky[0], atmos.nightSky[1], atmos.nightSky[2]);

    uniforms.uSunDirection.value.copy(sunDirection);
    uniforms.uBetaR.value.set(atmos.betaR[0], atmos.betaR[1], atmos.betaR[2]);
    uniforms.uBetaM.value.set(atmos.betaM[0], atmos.betaM[1], atmos.betaM[2]);
    uniforms.uSunE.value = atmos.sunE;
    uniforms.uSunFadeMix.value =
      Math.min(1, Math.max(0, Math.pow(Math.max(0, 1 - sunDirection.y), 5)));

    /* --- sun tint from real extinction, pinned to SUN.colour at 0.66 -------
     * The colour carries hue only (largest channel is always 1) and every bit
     * of brightness lives in `sunLevel`. Fold brightness into the colour as
     * well and noon comes out DIMMER than late afternoon, because normalising
     * a whiter tint throws the energy away. */
    atmos.sunExtinction(_fex);
    const peak = Math.max(_fex[0], _fex[1], _fex[2], 1e-6);
    let r = (_fex[0] / peak) * SUN_CORRECTION[0];
    let g = (_fex[1] / peak) * SUN_CORRECTION[1];
    let b = (_fex[2] / peak) * SUN_CORRECTION[2];
    const m = Math.max(r, g, b, 1e-6);
    r /= m; g /= m; b /= m;
    sunColour.setRGB(r, g, b, THREE.LinearSRGBColorSpace);
    uniforms.uSunTint.value.copy(sunColour);

    const luma = 0.2126 * _fex[0] + 0.7152 * _fex[1] + 0.0722 * _fex[2];
    const colLuma = Math.max(0.05, 0.2126 * r + 0.7152 * g + 0.0722 * b);
    sunLevel = Math.min(1.45, luma / refSunLuma) * (SUN_REF_LUMA / colLuma) *
      smoothstep(-0.035, 0.075, sunDirection.y);

    /* --- representative sky / horizon / ground colours ------------------ */
    atmos.radiance(0, 1, 0, _rgb);
    skyColour.setRGB(_rgb[0], _rgb[1], _rgb[2], THREE.LinearSRGBColorSpace);

    // Horizon average, weighted toward the sun's side and carrying the same
    // authored warm band the fragment shader adds — otherwise the fog reads
    // cooler than the sky it is supposed to blend into.
    // Strongest when the sun is low, and gone once it is under the horizon —
    // otherwise the band keeps glowing orange all night and drags the fog with
    // it.
    // Trimmed from 0.50/0.62: added on top of an analytic horizon that is
    // already above 1.0 radiance, the old strength tone-mapped the whole horizon
    // band to white and the hills went with it.
    const tintStrength = (0.36 + 0.52 * Math.pow(1 - above, 2.5)) *
      smoothstep(-0.10, 0.07, sunDirection.y);
    const tint = uniforms.uHorizonTint.value;
    const bandWeight = Math.pow(1 - 0.06, 7.0);
    const sunHoriz = Math.hypot(sunDirection.x, sunDirection.z) || 1e-3;
    _acc[0] = _acc[1] = _acc[2] = 0;
    let wsum = 0;
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      const dx = Math.cos(a);
      const dz = Math.sin(a);
      const side = Math.max(0,
        (dx * sunDirection.x + dz * sunDirection.z) / sunHoriz);
      const w = 1 + 1.6 * side;
      atmos.radiance(dx * 0.998, 0.06, dz * 0.998, _rgb);
      const k = bandWeight * (0.25 + 0.75 * side) * tintStrength;
      _acc[0] += (_rgb[0] + tint.r * k) * w;
      _acc[1] += (_rgb[1] + tint.g * k) * w;
      _acc[2] += (_rgb[2] + tint.b * k) * w;
      wsum += w;
    }
    horizonColour.setRGB(_acc[0] / wsum, _acc[1] / wsum, _acc[2] / wsum,
      THREE.LinearSRGBColorSpace);

    // Ground bounce: sunlit ground albedo x (direct + sky) irradiance. This
    // fills the lower hemisphere of the env map and is where the warm fill on
    // shaded wall faces comes from — without it plaster shadows go acid blue.
    const sunTerm = (SUN.intensity * SUN_GAIN) * above / Math.PI * sunLevel;
    const skyR = skyColour.r * 0.45 + horizonColour.r * 0.55;
    const skyG = skyColour.g * 0.45 + horizonColour.g * 0.55;
    const skyB = skyColour.b * 0.45 + horizonColour.b * 0.55;
    groundColour.setRGB(
      GROUND_ALBEDO[0] * (sunColour.r * sunTerm + skyR) * 0.78,
      GROUND_ALBEDO[1] * (sunColour.g * sunTerm + skyG) * 0.78,
      GROUND_ALBEDO[2] * (sunColour.b * sunTerm + skyB) * 0.78,
      THREE.LinearSRGBColorSpace,
    );
    uniforms.uGroundColour.value.copy(groundColour);

    /* --- cloud shading -------------------------------------------------- */
    const day = smoothstep(-0.06, 0.22, sunDirection.y);

    // Undersides read as cool grey-blue: sky light only, and dimmed hard. This
    // is the darkest thing in the sky and it has to stay clearly below the tops
    // or the deck reads as a field of flat white puffs.
    const shR = (skyColour.r * 0.55 + horizonColour.r * 0.45) * 0.46 + 0.012;
    const shG = (skyColour.g * 0.55 + horizonColour.g * 0.45) * 0.46 + 0.014;
    const shB = (skyColour.b * 0.55 + horizonColour.b * 0.45) * 0.48 + 0.022;
    uniforms.uCloudShadow.value.setRGB(shR, shG, shB, THREE.LinearSRGBColorSpace);

    // Sunlit tops. Interpolated toward the shadow colour by `day` so that after
    // sunset the deck flattens into dim grey instead of glowing red — at that
    // point the extinction-derived sun hue is meaningless.
    //
    // ~1.6 total, not the 3.8 this used to be: a cloud top is a ~0.9-albedo
    // Lambertian sheet under a 3.6 sun, so 3.6 * 0.9 / PI ~= 1.0, plus sky. Any
    // hotter and every top clips to the same flat white after ACES, which is
    // exactly what a "uniform field of puffs" looks like.
    //
    // Deliberately NOT scaled by SUN_GAIN. The gain exists to bring the village
    // ground up to a real albedo; the cloud deck was already measured correct and
    // is left exactly where it was. The side effect is wanted: the ground used to
    // sit far too dark against the sky, and holding the deck still while the
    // ground rises is what closes that gap.
    const litGain = (1.05 + 0.55 * day) * (0.55 + 0.45 * Math.min(1, sunLevel));
    uniforms.uCloudLit.value.setRGB(
      shR * 1.15 + day * (sunColour.r * litGain + horizonColour.r * 0.25 - shR * 1.15),
      shG * 1.15 + day * (sunColour.g * litGain + horizonColour.g * 0.25 - shG * 1.15),
      shB * 1.15 + day * (sunColour.b * litGain + horizonColour.b * 0.25 - shB * 1.15),
      THREE.LinearSRGBColorSpace,
    );
    // Scaled with uCloudLit: the two multiply, so leaving this at its old level
    // after halving the lit colour would still blow the edges near the sun out.
    uniforms.uSilverLining.value = 0.40 + 0.85 * day;

    // How far a sun ray travels sideways while it climbs through the deck. A low
    // sun gives a long rake and tall, dramatic cloud shading. Sized against
    // uCloudScale: three steps should cross about two cloud radii, so the march
    // reaches the far side of a tower and the base actually goes dark.
    const horiz = Math.hypot(sunDirection.x, sunDirection.z) || 1e-3;
    const stepLen = Math.min(1.25, 0.42 * horiz / Math.max(0.16, sunDirection.y));
    uniforms.uSunPlaneStep.value.set(
      (sunDirection.x / horiz) * stepLen,
      (sunDirection.z / horiz) * stepLen,
    );

    uniforms.uHorizonTintStrength.value = tintStrength;
    uniforms.uNight.value = smoothstep(0.10, -0.14, sunDirection.y);

    stats.timeOfDay = t;
    stats.sunElevationDeg =
      Math.round(Math.asin(Math.max(-1, Math.min(1, sunDirection.y))) / DEG * 100) / 100;
    // Compass bearing the light comes FROM — the number defect F1 was about.
    stats.sunAzimuthDeg = Math.round(
      ((Math.atan2(sunDirection.x, -sunDirection.z) / DEG) + 360) % 360 * 100) / 100;
  }

  function setTimeOfDay(t01) {
    const t = ((t01 % 1) + 1) % 1;
    if (Math.abs(t - timeOfDay) < 1e-5) return;
    applyTime(t);
    scheduleBake();
  }

  function update(dt, ctx) {
    // Keep the dome centred on the eye: the shader works off
    // normalize( worldPos - cameraPosition ), so a static dome would shear.
    if (ctx && ctx.camera) group.position.copy(ctx.camera.position);

    const step = dt * 0.011;
    drift0.x += WIND_X * step;
    drift0.y += WIND_Z * step;
    drift1.x += WIND_X * step * 0.42;
    drift1.y += WIND_Z * step * 0.42;

    // Belt and braces: if a bake was requested but the timer never fired.
    if (envDirty && !bakeTimer) scheduleBake();
  }

  function dispose() {
    disposed = true;
    if (bakeTimer) { clearTimeout(bakeTimer); bakeTimer = 0; }
    const envTex = pmremTarget ? pmremTarget.texture : null;
    if (envTex && scene.environment === envTex) scene.environment = null;
    scene.remove(group);
    bakeScene.remove(bakeMesh);
    geometry.dispose();
    material.dispose();
    noiseTex.dispose();
    cubeTarget.dispose();
    if (pmremTarget) pmremTarget.dispose();
    pmrem.dispose();
  }

  /* --------------------------------------------------------------- boot */

  // Establish the reference extinction BEFORE the first apply, so that
  // t = 0.66 hands the DirectionalLight exactly layout.js's SUN.colour.
  sunDirectionAt(DEFAULT_TIME_OF_DAY, _tmpVec);
  {
    const above = Math.max(0, _tmpVec.y);
    atmos.configure(_tmpVec, turbidityAt(above), rayleighAt(above));
    atmos.sunExtinction(_fex);
    const peak = Math.max(_fex[0], _fex[1], _fex[2], 1e-6);
    SUN_CORRECTION[0] = SUN_LINEAR.r / Math.max(1e-6, _fex[0] / peak);
    SUN_CORRECTION[1] = SUN_LINEAR.g / Math.max(1e-6, _fex[1] / peak);
    SUN_CORRECTION[2] = SUN_LINEAR.b / Math.max(1e-6, _fex[2] / peak);
    refSunLuma = 0.2126 * _fex[0] + 0.7152 * _fex[1] + 0.0722 * _fex[2];
  }

  applyTime(DEFAULT_TIME_OF_DAY);
  bakeNow();

  return api;
}
