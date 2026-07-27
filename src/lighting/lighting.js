/**
 * ============================================================================
 *  LIGHTING
 * ============================================================================
 * Nothing outside this directory creates a light. What lives here:
 *
 *   sun    one DirectionalLight with a single, well-fitted, texel-snapped
 *          orthographic shadow cascade that follows the player.
 *   hemi   a weak HemisphereLight that keeps shadowed faces from going dead.
 *          The real indirect light is the PMREM env map from sky.js.
 *   ambient a near-zero AmbientLight, present only as a black-floor guard.
 *   fog    warm aerial perspective whose colour is read out of the same
 *          analytic atmosphere the sky is drawn with.
 *   pool   6-8 PointLights recycled across arbitrarily many light anchors,
 *          cross-faded so reassignment never pops. Selection is ROOM-AWARE: an
 *          anchor in the room you are standing in outranks a nearer one on the
 *          far side of the wall.
 *   indoor a mode, not a light. Image-based lighting in three has no occlusion,
 *          so the sky env map lights the inside of a sealed room exactly as hard
 *          as the outside of it. The fix lives in the interior materials
 *          (INTERIOR_ENV below) plus the small ambient/hemisphere trim in
 *          applyIndoor(); see the block comment on INTERIOR_ENV.
 *
 * Two rules this file lives by:
 *   - the shadow-camera centre is snapped to whole shadow-map texels. Skip that
 *     and every shadow edge crawls as the player walks. It is the single most
 *     visible shadow defect there is.
 *   - the number of lights in the scene NEVER changes after construction.
 *     Adding or removing one (or toggling `visible`) rewrites NUM_POINT_LIGHTS
 *     and recompiles every material in the village mid-frame.
 * ============================================================================
 */

import * as THREE from 'three';
import { Rng } from '../util/rng.js';
import { SUN } from '../world/layout.js';
import { SUN_GAIN, DEFAULT_TIME_OF_DAY, sunDirectionAt, smoothstep } from './sky.js';

/* -------------------------------------------------------------------------- */
/* Tuning                                                                      */
/* -------------------------------------------------------------------------- */

/** Vertical span the shadow cascade must swallow: ground to tallest chimney. */
const SHADOW_WORLD_HEIGHT = 34;

/**
 * Aerial perspective, and nothing nearer than that. `THREE.Fog` is linear in
 * view distance with no height term, so the only way to keep the village crisp
 * is to start it past the far edge of the village: 170 m clears the whole plaza
 * (radius 26), every building (< 40 m), the hero trees (< 60 m) and the first
 * rise of the hills (layout.js has nothing above ground before ~138 m). It
 * reaches 0.06 at 250 m, 0.17-0.34 across the hill crests (400-620 m) and
 * 0.6 at the far ridge (940-1060 m). config.js's 60/620 and the 90 m this file
 * shipped with both ate the mid-ground instead.
 */
const FOG_NEAR = 170;
const FOG_FAR = 1500;
/**
 * Linear ceiling on the fog colour. The analytic horizon runs above 1.0
 * radiance, which after ACES is 0.94 sRGB — a white band that swallowed the
 * hills. 0.46 tone-maps to about 0.78: unmistakably haze, still clearly
 * brighter than the hills it sits behind, and never brighter than the sky
 * above it.
 */
const FOG_MAX_LUMINANCE = 0.46;
/** Fraction of the sky's own horizon luminance the fog is allowed to track. */
const FOG_LUMINANCE_SCALE = 0.62;
/**
 * Warm haze hue, normalised so the largest channel is 1. The fog carries the
 * horizon's hue pulled this far toward it: the raw analytic horizon reads
 * slightly green, and a near-neutral fog is what made the whole picture milky.
 */
const FOG_HAZE_HUE = new THREE.Color(0xffd2a6);
FOG_HAZE_HUE.multiplyScalar(
  1 / Math.max(FOG_HAZE_HUE.r, FOG_HAZE_HUE.g, FOG_HAZE_HUE.b, 1e-5));
const FOG_HAZE_MIX = 0.32;

/** HemisphereLight tints, per the art direction. Modulated by time of day. */
const HEMI_SKY = new THREE.Color(0x9fc4ff);
const HEMI_GROUND = new THREE.Color(0x6b5a3c);
/**
 * Weak on purpose. The PMREM env map is the real indirect light; this only
 * shapes it — at 0.30 it was adding a flat 8% of the sun's irradiance to every
 * surface regardless of what it could see, which is the classic way to make a
 * sunlit square look ambient-lit.
 *
 * 0.22, up from 0.16, alongside the ground albedo correction and sky.js's
 * ENV_BAKE_GAIN 0.72 -> 0.98. It still contributes only 0.099 of the 3.08 total
 * irradiance reaching sunlit ground (3.2%), so it shapes rather than fills; the
 * measured job of lifting the shadows belongs to the env bake, not to this.
 */
const HEMI_DAY_INTENSITY = 0.22;
const HEMI_NIGHT_INTENSITY = 0.035;

/** Anchors further than this from the player get no real light at all. */
const ANCHOR_RANGE = 35;
const ANCHOR_RANGE_SQ = ANCHOR_RANGE * ANCHOR_RANGE;
/**
 * Indoor anchors get a much shorter leash. A hearth two houses away has no
 * business claiming a pool slot, and with 11 interiors' worth of fires and
 * candles in the anchor list the pool would otherwise spend its whole budget on
 * lights the player cannot see.
 */
const INDOOR_ANCHOR_RANGE = 20;
const INDOOR_ANCHOR_RANGE_SQ = INDOOR_ANCHOR_RANGE * INDOOR_ANCHOR_RANGE;

/* --------------------------------------------------------------------------
 * INTERIOR LIGHTING MODE
 * --------------------------------------------------------------------------
 * `scene.environment` is a PMREM cube of the sky, and three's IBL has no
 * occlusion whatsoever: the interior face of a sealed wall receives exactly the
 * same sky irradiance as the exterior face. Left alone, every room in the
 * village reads as a brightly, flatly lit box and the hearth fire does nothing.
 *
 * There is exactly ONE clean handle on this, and it is per-material:
 * `envMapIntensity`. `scene.environmentIntensity` is a dead knob in this project
 * — materials.js assigns `envMap` explicitly on every material (it has to; the
 * renderer overwrites `envMapIntensity` with `scene.environmentIntensity`
 * whenever `material.envMap === null`), so the scene-level value is never read.
 * Do not try to drive interiors from main.js's `scene.environmentIntensity = 1.0`
 * line: it would change nothing.
 *
 * So: interior surfaces must be built from `materials.variant(key, { ... ,
 * envMapIntensity: INTERIOR_ENV.surface })`. `variant()` stores that as the
 * material's absolute env scale (it REPLACES the definition's own `env`, it does
 * not multiply it), which is why these numbers are absolute and already include
 * the 0.8-0.9 a plaster or timber definition would normally carry.
 *
 * How the numbers were chosen. Exterior calibration: sunlit ground irradiance
 * ~4.3 (direct 2.58, env ~1.61, hemisphere 0.099), shadowed plaster wall
 * measured 0.385 sRGB luminance. A room with two small leaded windows has a real
 * daylight factor of 1-3%, which at `surface = 0.03` would be black. The honest
 * cinematic answer is the one below:
 *
 *   surface 0.16  ->  env term drops to 0.16/0.85 = 19% of what the same plaster
 *                     takes outdoors. With the hemisphere trim in applyIndoor()
 *                     an interior wall lands near 0.20-0.21 sRGB luminance
 *                     (~52 of 255): unmistakably indoors, still readable, and
 *                     comfortably below the 0.55-0.60 a sunlit window shaft on
 *                     the floor keeps (the sun is a real DirectionalLight and IS
 *                     correctly occluded by the shell, so shafts survive intact).
 *   deep    0.09  ->  ceiling boards, joist undersides, stair soffits, cupboard
 *                     interiors, windowless stores and attics. Roughly half the
 *                     wall level, which is what a real room does above head
 *                     height.
 *   reveal  0.45  ->  door and window reveals, thresholds, sills, the inner
 *                     faces of an open door leaf. These genuinely see a big lump
 *                     of sky, and if you starve them the openings read as holes
 *                     cut in cardboard.
 *
 * A hearth's peak irradiance at 2 m is 8.4/4 = 2.1 (see KIND.fire.indoor, which
 * used to be nearly three times this and clipped). An interior wall's env fill at
 * `surface` 0.16 is ~0.17 of irradiance, so at 2 m the fire is still 12x the room
 * and it falls off as 1/d² across a 4 m room, which is what makes a room read as
 * firelit rather than as evenly lit. It no longer exceeds the 2.58 the sun puts
 * on open ground, and that is deliberate: nothing indoors should out-expose the
 * exterior the same frame can see through the doorway.
 * -------------------------------------------------------------------------- */
export const INTERIOR_ENV = {
  surface: 0.16,
  deep: 0.09,
  reveal: 0.45,
};
/** Convenience aliases — some streams would rather import a bare number. */
export const INTERIOR_ENV_SCALE = INTERIOR_ENV.surface;
export const INTERIOR_ENV_SCALE_DEEP = INTERIOR_ENV.deep;
export const INTERIOR_ENV_SCALE_REVEAL = INTERIOR_ENV.reveal;

/** The material keys that exist only inside a building and can be dimmed wholesale. */
export const INTERIOR_ONLY_MATERIALS = ['lathPlaster', 'floorBoard', 'ceilingBeam', 'soot'];

/**
 * FALLBACK, not the main path. If an interior stream builds its walls straight off
 * `materials.get('lathPlaster')` instead of a variant carrying
 * `envMapIntensity: INTERIOR_ENV.surface`, those surfaces will be lit by the full
 * unoccluded sky and the rooms will look like daylit boxes. This retrofits the
 * scale onto the canonical instances of keys that only ever appear indoors.
 *
 * Call it from main.js AFTER the interiors are built and only if needed:
 *
 *     applyInteriorEnv(materials, sky.envMap);
 *
 * Pass your own key list to widen it, but be careful: `flagstone`, `sackcloth`
 * and `strawLitter` can legitimately appear outdoors (a doorstep, a market stall,
 * a stable yard) and dimming them there would be a visible exterior regression.
 * The right long-term home for these numbers is `env:` on the interior keys in
 * materials.js.
 */
export function applyInteriorEnv(materials, envMap, keys = INTERIOR_ONLY_MATERIALS) {
  if (!materials || typeof materials.get !== 'function') return 0;
  let n = 0;
  for (const key of keys) {
    if (materials.has && !materials.has(key)) continue;
    const m = materials.get(key);
    if (!m || !m.isMeshStandardMaterial) continue;
    m.userData.envScale = INTERIOR_ENV.surface;
    n++;
  }
  // setEnvironment() is what re-reads userData.envScale across the library.
  materials.setEnvironment?.(envMap || null, 1.0);
  return n;
}

/**
 * Indoors the sky's hemisphere fill is mostly walled off, so take it down —
 * but not to nothing, because what is left stands in for the skylight a window
 * throws around a room that no amount of point lights will fake.
 *
 * 0.40, was 0.55. The indoor fill that used to arrive as a flat AmbientLight
 * (INDOOR_AMBIENT 0.52, ~0.0159 of luminance-weighted irradiance) is folded in
 * here instead: 0.15 of hemiBase is +0.033 intensity, worth about the same
 * irradiance, except that a hemisphere is SHAPED — up-facing and down-facing
 * surfaces get different colours, and with the tilt below the two halves also
 * point along the room. A flat ambient term cannot do any of that; it is the
 * single most reliable way to make a room look like a lit box.
 */
const INDOOR_HEMI_CUT = 0.40;
/**
 * Interior hemisphere tints. The "sky" half stands in for the cold daylight
 * coming off the leaded glazing; the "ground" half for the warm bounce off
 * boards, straw litter and the hearth. Blended in by indoorMix, so outdoors the
 * hemisphere is bit-for-bit the art-directed HEMI_SKY/HEMI_GROUND pair the
 * exterior was signed off with.
 */
const INDOOR_HEMI_SKY = new THREE.Color(0x93b4e0);
const INDOOR_HEMI_GROUND = new THREE.Color(0x6d4a2a);
const INDOOR_HEMI_TINT_MIX = 0.70;
/**
 * How far the hemisphere's own axis is tilted off vertical toward the sun,
 * indoors only. three takes a HemisphereLight's axis from its world POSITION
 * (WebGLLights: `direction.setFromMatrixPosition(light.matrixWorld).normalize()`),
 * so this is a real, free handle and not a hack: at 0.55 the fill's bright/cool
 * lobe faces the window wall and its warm lobe faces the room's back, which is
 * the only cheap source of horizontal directionality an occlusion-free IBL fill
 * can be given. Outdoors the axis is exactly (0, 1, 0).
 */
const INDOOR_HEMI_TILT = 0.55;
/**
 * The pure-black guard, warmed indoors — and that is ALL it is now. It used to
 * be raised to 0.52 indoors, which at the AmbientLight's ~0.031 linear colour is
 * ~0.0159 of irradiance, ~9% of the interior fill, applied flat to every normal
 * in the world with no shape and no occlusion. That fill now arrives through the
 * hemisphere (INDOOR_HEMI_CUT above). 0.05 leaves ~0.0015 — under 1% of the
 * interior level, i.e. a corner behind a dresser is dark rather than dead, and
 * nothing more.
 */
const INDOOR_AMBIENT = 0.05;
const INDOOR_AMBIENT_TINT = new THREE.Color(0x3a3026);
/** Seconds for the indoor/outdoor crossfade. Long enough to read as adaptation. */
const INDOOR_RAMP = 0.42;

/**
 * The sun's cascade, clamped while indoors. Outdoors the ortho box is fitted to
 * quality.shadowDistance (95 m at high -> 100.9 m across a 4096 map -> 40.6
 * texels/m, 2.46 cm per texel). That resolves a doorway and the outline of a
 * window, but not a 4 cm leaded glazing bar. Indoors nothing beyond the room
 * matters much, so shrink the box: 48 m gives 60.4 m across -> 67.8 texels/m
 * -> 1.5 cm per texel at high, which does resolve the bars. See the report for
 * the low preset, where 1024 texels is the binding constraint and this is not
 * enough.
 */
const INDOOR_SHADOW_DISTANCE = 48;

/** Cross-fade time when a pool slot changes anchor. */
const FADE_TIME = 0.25;

/**
 * Per-kind candela and reach. `intensity` on the anchor scales the candela.
 *
 * `indoor` overrides apply when the anchor resolves to a room. Two things change
 * inside: the reach comes in (a room is 3-5 m, and a 16 m fire spills across the
 * plaza through walls it cannot be occluded by), and the day floor goes almost
 * to 1 — an indoor fire or candle burns just as brightly at 16:00 as at
 * midnight, whereas a street lantern does not.
 */
const KIND = {
  lantern: {
    candela: 9.0, distance: 14, dayFloor: 0.00, flicker: 0.06, weight: 1.0,
    indoor: { candela: 7.0, distance: 8.5, dayFloor: 0.72 },
  },
  window: {
    candela: 6.5, distance: 12, dayFloor: 0.16, flicker: 0.0, weight: 1.3,
    indoor: { candela: 4.0, distance: 7.0, dayFloor: 0.30 },
  },
  fire: {
    candela: 16.0, distance: 16, dayFloor: 0.22, flicker: 0.30, fire: true,
    /**
     * `weight` multiplies the squared distance in the pool's ranking, so 0.25
     * means "count a fire as half as far away as it is". Without it a furnished
     * taproom with three candles on the tables crowds its own hearth out of a
     * 3-slot pool purely on metres, which is precisely backwards: the hearth is
     * the light that tells you what room you are in.
     */
    weight: 0.25,
    /**
     * 6.0, down from 17.0. The 17 was solved for "brightest thing in the room"
     * and overshot into "only thing in the room": with the taproom hearth's
     * authored scale of 1.1, the day floor of 0.88 and a flare peak of 1.45 the
     * PointLight measured 20.6, which put the hearth stone 0.5 m from the anchor
     * at a linear radiance of 5.5 — ACES clips anything past 2.87 — so the fire
     * and its surround were one solid white blob with no flame structure left.
     *
     * The number, worked the same way the exterior was: a hearth is a 1-2 kW
     * radiant source, and in real photometry it loses to daylight by four orders
     * of magnitude, so the scene's own scale is the only thing worth calibrating
     * against. Peak effective intensity is now 6.0 x 1.1 x 0.88 x 1.45 = 8.4,
     * nominal 5.8. That puts
     *   the hearth stone at 0.5 m  ->  1.72 linear, 0.951 sRGB (2.49 / 0.974 on
     *                                  a flare) — hot, structured, not clipped
     *   plaster at 1.0 m           ->  0.81 linear, 0.883 sRGB
     *   an oak settle at 1.5 m     ->  0.35 linear, 0.727 sRGB
     *   the far wall at 3.0 m      ->  0.09 linear, 0.365 sRGB
     * against an env-lit interior wall's 0.20 and a candle anchor's ~0.3
     * intensity. So it stays 25x a candle and 9x the room's own fill at 2 m —
     * unambiguously the brightest thing in the room and unambiguously lighting
     * the walls and the furniture — while peaking just under a sunlit exterior
     * plaster face (0.858 sRGB), which is the right ceiling for an interior.
     *
     * This is the LIGHT only. The flame cards' own emissive belongs to
     * furnishings.js and is being brought down there in the same pass.
     */
    indoor: { candela: 6.0, distance: 11, dayFloor: 0.88 },
  },
  candle: {
    candela: 3.5, distance: 7, dayFloor: 0.00, flicker: 0.13, fire: true,
    weight: 1.5,
    indoor: { candela: 3.4, distance: 5.5, dayFloor: 0.80 },
  },
};
const KIND_DEFAULT = KIND.lantern;

/* --------------------------------------------------------------------------
 * Room-aware pool scoring
 * --------------------------------------------------------------------------
 * Penalties multiply the SQUARED distance, so a penalty of 16 means "count this
 * anchor as four times further away than it is". The hearth 5 m away in the room
 * you are standing in (25) must beat the lantern 4 m away through the wall
 * (16 x 9 = 144), and it does.
 * -------------------------------------------------------------------------- */
/** Player and anchor are in different rooms — the worst case, walls both ways. */
const PENALTY_CROSS_ROOM = 16;
/** Player indoors, anchor outdoors. */
const PENALTY_INSIDE_OUT = 9;
/**
 * Player outdoors, anchor indoors. Only 4: a fire glimpsed through an open door
 * from the square is worth a slot, and it is the cue that tells you the building
 * has an inside.
 */
const PENALTY_OUTSIDE_IN = 4;
/**
 * Hard cull for anchors in a DIFFERENT room from the player. A point light is not
 * occluded by anything (unless it is the one shadow caster), so a candle two rooms
 * away would put a soft patch on your floor straight through the wall. Six metres
 * still admits the hearth in the next room glowing through an open doorway, which
 * is a cue worth keeping.
 */
const CROSS_ROOM_RANGE_SQ = 36;

function poolSizeFor(quality) {
  switch (quality && quality.name) {
    case 'low': return 3;
    case 'medium': return 5;
    case 'ultra': return 8;
    default: return 7;
  }
}

/**
 * How many pool slots cast. Raised from 0 to 1 at `medium`, because the hearth
 * casting shadows of the furniture across the floor is the largest single
 * readability win available indoors and it is worth a fixed cost to have it on
 * the tier most machines land on.
 *
 * What it costs, honestly: a shadow-casting PointLight is a 6-face cube render
 * of everything inside its far plane (clamped to 9.5 m indoors, so one room and
 * its doorway) at up to 5.5 Hz, and it permanently raises NUM_POINT_LIGHT_SHADOWS,
 * which adds a cube-shadow lookup to the fragment shader of every lit material in
 * the village whether or not the light is doing anything. `low` still gets none.
 */
function shadowCastersFor(quality) {
  switch (quality && quality.name) {
    case 'ultra': return 2;
    case 'high': return 1;
    case 'medium': return 1;
    default: return 0;
  }
}

function pointShadowSizeFor(quality) {
  switch (quality && quality.name) {
    case 'ultra': return 768;
    case 'high': return 512;
    default: return 384;
  }
}

/* -------------------------------------------------------------------------- */
/* Scratch — hoisted, update() must never allocate                             */
/* -------------------------------------------------------------------------- */

const _f = new THREE.Vector3();   // light axis, target -> sun
const _r = new THREE.Vector3();   // shadow-map "right"
const _u = new THREE.Vector3();   // shadow-map "up"
const _centre = new THREE.Vector3();
const _snapped = new THREE.Vector3();
const _worldUp = new THREE.Vector3(0, 1, 0);
const _tmpDir = new THREE.Vector3();
const _tmpCol = new THREE.Color();
const _fogCol = new THREE.Color();
const _fireCol = new THREE.Color();   // per-slot fire tint, rebuilt every frame

/** Rec.709 luminance of a linear colour. */
function luminance(c) {
  return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

/* -------------------------------------------------------------------------- */
/* Flame colour                                                                */
/* -------------------------------------------------------------------------- */

/**
 * Tanner Helland's blackbody approximation, valid well below 6600 K, which is
 * all a wood fire needs. Returns an sRGB triple with the red channel pinned at
 * 1, so the candela in KIND stays the only brightness knob.
 *
 * 1900 K is the base of a settled log fire, 2200 K a flaring one. Sanity check:
 * 2000 K comes out (255, 137, 14) — the orange everyone recognises as firelight.
 */
function kelvinToColour(kelvin, out) {
  const t = Math.max(10, Math.min(66, kelvin / 100));
  const g = Math.max(0, Math.min(255, 99.4708025861 * Math.log(t) - 161.1195681661));
  const b = t <= 19 ? 0
    : Math.max(0, Math.min(255, 138.5177312231 * Math.log(t - 10) - 305.0447927307));
  return out.setRGB(1, g / 255, b / 255, THREE.SRGBColorSpace);
}

/**
 * The flame gamut, baked once as linear working-space colours. 1850-2250 K in
 * nine steps: lerping between two of these is exact enough and, unlike calling
 * kelvinToColour() per light per frame, allocates nothing and costs no logs.
 */
const FIRE_KELVIN_MIN = 1850;
const FIRE_KELVIN_MAX = 2250;
const FIRE_LUT = (() => {
  const n = 9;
  const out = [];
  const c = new THREE.Color();
  for (let i = 0; i < n; i++) {
    kelvinToColour(FIRE_KELVIN_MIN + (FIRE_KELVIN_MAX - FIRE_KELVIN_MIN) * (i / (n - 1)), c);
    out.push([c.r, c.g, c.b]);
  }
  return out;
})();
/** Mid-gamut reference (2050 K) — authored fire colours are corrected against it. */
const FIRE_REF = FIRE_LUT[(FIRE_LUT.length - 1) >> 1];

/** Writes the flame colour for a 0..1 position through the gamut into `out`. */
function fireColourAt(t01, out) {
  const n = FIRE_LUT.length - 1;
  const x = Math.max(0, Math.min(1, t01)) * n;
  const i = Math.min(n - 1, x | 0);
  const f = x - i;
  const a = FIRE_LUT[i];
  const b = FIRE_LUT[i + 1];
  return out.setRGB(
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
    THREE.LinearSRGBColorSpace,
  );
}

/* -------------------------------------------------------------------------- */
/* Factory                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * @param {Object} args
 * @param {THREE.Scene} args.scene
 * @param {Object} args.sky        the object createSky() returned
 * @param {Object} args.quality
 * @param {THREE.WebGLRenderer} args.renderer
 * @returns {Object} { sun, ambient, hemi, fog, addLightAnchors, setTimeOfDay,
 *                     setRoomResolver, setPlayerRoom, interior, indoorMix,
 *                     update, dispose, stats }
 *
 * New since interiors, all optional to wire:
 *   setRoomResolver(fn)  fn(x,y,z) -> room identity | null. Makes the light pool
 *                        prefer anchors in the player's room. Without it the pool
 *                        is pure distance, as before.
 *   setPlayerRoom(room)  override the player's room directly; `undefined` gives
 *                        control back to the resolver.
 *   interior             { surface, deep, reveal } — the envMapIntensity values
 *                        interior geometry must be built with. Same numbers as
 *                        the exported INTERIOR_ENV.
 *   indoorMix            0..1, read-only, ramped. Handy for a HUD or for post.
 */
export function createLighting({ scene, sky, quality, renderer }) {
  const q = quality || {};

  /* ------------------------------------------------------------------ sun */

  const sun = new THREE.DirectionalLight(SUN.colour, SUN.intensity * SUN_GAIN);
  sun.name = 'sun';
  sun.castShadow = true;
  sun.shadow.mapSize.set(q.shadowMapSize || 2048, q.shadowMapSize || 2048);
  // A small negative constant bias plus a normal-space offset: the constant
  // alone peter-pans thatch eaves, the normal offset alone leaves acne on the
  // near-tangent setts of the plaza.
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.03;
  sun.shadow.blurSamples = q.softShadows ? 12 : 4;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 200;
  // The target must live in the scene or its matrixWorld is never refreshed
  // and DirectionalLightShadow.updateMatrices() reads a stale position.
  sun.target.name = 'sunTarget';
  scene.add(sun);
  scene.add(sun.target);

  /* -------------------------------------------------------------- ambient */

  const hemi = new THREE.HemisphereLight(HEMI_SKY.getHex(), HEMI_GROUND.getHex(),
    HEMI_DAY_INTENSITY);
  hemi.name = 'hemisphere';
  scene.add(hemi);

  // Deliberately almost nothing: a flat ambient term is the classic non-AAA
  // tell. This exists only so that a fully enclosed interior is not pure black.
  const ambient = new THREE.AmbientLight(0x2a2f38, 0.02);
  ambient.name = 'ambientFloor';
  scene.add(ambient);

  /* ------------------------------------------------------------------ fog */

  const fog = new THREE.Fog(0xffffff, FOG_NEAR, FOG_FAR);
  scene.fog = fog;

  /* ----------------------------------------------------------- light pool */

  const poolSize = poolSizeFor(q);
  const shadowCasters = Math.min(shadowCastersFor(q), poolSize);
  /** @type {Array<{light:THREE.PointLight, anchor:number, fade:number, retiring:boolean, shadowClock:number}>} */
  const slots = [];
  for (let i = 0; i < poolSize; i++) {
    const light = new THREE.PointLight(0xffffff, 0, 12, 2);
    light.name = `poolLight${i}`;
    light.castShadow = i < shadowCasters;   // fixed for the lifetime of the run
    if (light.castShadow) {
      const s = pointShadowSizeFor(q);
      light.shadow.mapSize.set(s, s);
      light.shadow.bias = -0.004;
      light.shadow.normalBias = 0.04;
      light.shadow.camera.near = 0.12;
      light.shadow.camera.far = 18;
      // Refreshed on demand, not every frame: a point-light shadow is six
      // renders of the world and these only matter after dusk.
      light.shadow.autoUpdate = false;
      light.shadow.needsUpdate = true;
    }
    scene.add(light);
    slots.push({
      light, anchor: -1, fade: 0, retiring: false, shadowClock: 0, shadowWarm: 0,
      worth: -1e3,
    });
  }
  /** Countdown to the next shadow-slot reconciliation. */
  let swapClock = 0;

  /** @type {Array<Object>} flattened anchor records */
  const anchors = [];
  const anchorRng = new Rng('hollowbrook-lanterns');

  const bestIdx = new Int32Array(poolSize).fill(-1);
  /** Room-penalised score, NOT metres. See PENALTY_* above. */
  const bestScore = new Float32Array(poolSize).fill(Infinity);
  const claimed = new Uint8Array(poolSize);

  /* ----------------------------------------------------------- room state */

  /**
   * `roomAt(x,y,z)` from the interiors stream, if anyone wired it. Everything
   * below degrades to pure distance when it is absent, which is exactly the
   * behaviour this file shipped with.
   * @type {((x:number,y:number,z:number)=>*)|null}
   */
  let roomResolver = null;
  /** Bumped whenever the resolver or the anchor list changes. */
  let roomEpoch = 0;
  let anchorRoomsAt = -1;
  /** Interned room identities. 0 is reserved for "outdoors / unknown". */
  const roomIds = new Map();
  let nextRoomId = 1;
  let playerRoom = 0;
  let playerRoomForced = false;
  let indoorMix = 0;          // 0 outdoors, 1 indoors, ramped
  let roomProbeClock = 0;
  /** Last raw value the resolver returned, so identity can short-circuit. */
  let lastRoomObject = 0;
  const _lastProbe = new THREE.Vector3(1e9, 1e9, 1e9);

  /** Map an arbitrary room identity onto a small integer for === comparison. */
  function internRoom(v) {
    if (v === null || v === undefined || v === false) return 0;
    const k = (typeof v === 'object')
      ? (v.key !== undefined ? v.key
        : v.id !== undefined ? `${v.plotId || v.id}#${v.storey ?? 0}`
          : v)
      : v;
    let id = roomIds.get(k);
    if (id === undefined) { id = nextRoomId++; roomIds.set(k, id); }
    return id;
  }

  /**
   * Ask the resolver where a point is.
   *
   * Two probe sets, and the difference matters. A hearth's flame anchor sits in
   * the firebox — inside the chimney breast, i.e. legitimately OUTSIDE the room's
   * clear volume — so a fire is allowed to reach 0.6 m sideways to find its room.
   * Nothing else is: a street lantern bracketed 0.3 m off a cottage wall would
   * otherwise probe straight through 0.4 m of plaster, be declared indoors, and
   * start burning at 72% in broad daylight on a signed-off exterior. Anything
   * that wants certainty should stamp `room:` on the anchor and skip all of this.
   */
  const PROBE_NEAR = [[0, 0, 0], [0, 0.35, 0]];
  const PROBE_FIRE = [
    [0, 0, 0], [0, 0.35, 0],
    [0.6, 0.25, 0], [-0.6, 0.25, 0], [0, 0.25, 0.6], [0, 0.25, -0.6],
  ];
  function probeRoom(x, y, z, wide) {
    if (!roomResolver) return 0;
    const probe = wide ? PROBE_FIRE : PROBE_NEAR;
    for (let i = 0; i < probe.length; i++) {
      const p = probe[i];
      let r;
      try {
        r = roomResolver(x + p[0], y + p[1], z + p[2]);
      } catch (err) {
        console.warn('[lighting] roomAt() threw; falling back to distance-only ' +
          'light selection.', err);
        roomResolver = null;
        return 0;
      }
      if (r !== null && r !== undefined && r !== false) return internRoom(r);
    }
    return 0;
  }

  /**
   * Resolve every anchor's room. Runs once whenever the anchor list or the
   * resolver changes, never per frame.
   */
  function resolveAnchorRooms() {
    anchorRoomsAt = roomEpoch;
    let indoorCount = 0;
    for (let i = 0; i < anchors.length; i++) {
      const a = anchors[i];
      // An anchor that declared its own room is believed without asking. Note
      // `room: null` is a positive assertion of "outdoors" and is honoured too.
      if (a.declaredRoom !== undefined) a.room = internRoom(a.declaredRoom);
      else a.room = probeRoom(a.x, a.y, a.z, a.kind === 'fire');
      const indoor = a.room !== 0 || a.declaredIndoor;
      // `indoor: true` with no room and no resolver: the anchor knows it is
      // inside but not where, so it gets the indoor spec and is exempted from
      // room penalties rather than being penalised against a room it cannot name.
      a.neutral = indoor && a.room === 0;
      a.indoor = indoor;
      a.spec = indoor && a.base.indoor ? a.merged : a.base;
      a.rangeSq = indoor ? INDOOR_ANCHOR_RANGE_SQ : ANCHOR_RANGE_SQ;
      if (indoor) indoorCount++;
    }
    stats.indoorAnchors = indoorCount;
    stats.rooms = nextRoomId - 1;
  }

  /* ----------------------------------------------------------- day state */

  let timeOfDay = DEFAULT_TIME_OF_DAY;
  let nightMix = 0;         // 0 = full day, 1 = night
  let dayMix = 1;
  let elapsed = 0;
  let shadowExtent = 0;     // half-size of the ortho box, metres
  let shadowDistanceUsed = -1;
  let shadowMapUsed = q.shadowMapSize || 2048;

  const stats = {
    poolSize,
    shadowCasters,
    activeLights: 0,
    anchors: 0,
    indoorAnchors: 0,
    rooms: 0,
    roomAware: false,
    indoor: 0,
    playerRoom: 0,
    shadowAnchorKind: '',
    shadowExtent: 0,
    shadowTexelsPerMetre: 0,
    shadowTexelCm: 0,
    sunIntensity: 0,
    /** So the calibration pass can read the fill levels without a debugger. */
    ambientIntensity: 0.02,
    hemiIntensity: HEMI_DAY_INTENSITY,
    /** Peak effective hearth intensity indoors: candela x dayFloor x max flicker. */
    hearthCandela: Math.round(KIND.fire.indoor.candela *
      KIND.fire.indoor.dayFloor * (1 + KIND.fire.flicker * 1.5) * 100) / 100,
    fogNear: FOG_NEAR,
    fogFar: FOG_FAR,
    fogLuminance: 0,
    interiorEnvScale: INTERIOR_ENV.surface,
  };

  /** Base (outdoor) levels applyTime() decides; applyIndoor() trims them. */
  let hemiBase = HEMI_DAY_INTENSITY;
  let ambientBase = 0.02;
  const ambientBaseColour = ambient.color.clone();
  /** Outdoor hemisphere state, written by applyTime(), consumed by applyIndoor(). */
  const hemiSkyBase = HEMI_SKY.clone();
  const hemiGroundBase = HEMI_GROUND.clone();
  const hemiTilt = new THREE.Vector3(0, 1, 0);
  const _hemiAxis = new THREE.Vector3(0, 1, 0);

  /* -------------------------------------------------------------- helpers */

  function normalisedHue(colour, out) {
    const m = Math.max(colour.r, colour.g, colour.b, 1e-5);
    return out.setRGB(colour.r / m, colour.g / m, colour.b / m,
      THREE.LinearSRGBColorSpace);
  }

  function applyTime(t01) {
    timeOfDay = ((t01 % 1) + 1) % 1;

    // sky.js is the authority on the sun; keep them in lockstep even if this
    // is called on its own.
    if (sky && typeof sky.setTimeOfDay === 'function' &&
        Math.abs((sky.timeOfDay === undefined ? -1 : sky.timeOfDay) - timeOfDay) > 1e-5) {
      sky.setTimeOfDay(timeOfDay);
    }

    const dir = (sky && sky.sunDirection) || sunDirectionAt(timeOfDay, _tmpDir);
    const level = sky && typeof sky.sunLevel === 'number' ? sky.sunLevel
      : smoothstep(-0.035, 0.075, dir.y);

    dayMix = smoothstep(-0.06, 0.22, dir.y);
    nightMix = smoothstep(0.20, -0.05, dir.y);

    sun.intensity = SUN.intensity * SUN_GAIN * level;
    if (sky && sky.sunColour) sun.color.copy(sky.sunColour);
    stats.sunIntensity = Math.round(sun.intensity * 1000) / 1000;
    // Below the horizon the light contributes nothing, so stop paying for a
    // full-scene shadow pass. `castShadow` stays true — flipping it would
    // rewrite NUM_DIR_LIGHT_SHADOWS and recompile every material in the world.
    sun.shadow.autoUpdate = sun.intensity > 0.01;

    /* --- hemisphere: the art-directed tints, pulled toward the real sky ---
     * These are the OUTDOOR bases. applyIndoor() is what writes hemi.color and
     * hemi.groundColor, because it has to be able to re-blend them toward the
     * interior tints on every step of the indoor ramp without re-deriving the
     * sky. At indoorMix 0 the values it writes are exactly these. */
    if (sky && sky.skyColour) {
      normalisedHue(sky.skyColour, _tmpCol);
      hemiSkyBase.copy(HEMI_SKY).lerp(_tmpCol, 0.55);
    } else {
      hemiSkyBase.copy(HEMI_SKY);
    }
    if (sky && sky.groundColour) {
      normalisedHue(sky.groundColour, _tmpCol);
      hemiGroundBase.copy(HEMI_GROUND).lerp(_tmpCol, 0.55);
    } else {
      hemiGroundBase.copy(HEMI_GROUND);
    }
    // The axis the interior fill is tilted along: the sun's own bearing, so the
    // cool lobe faces whichever wall the windows are actually taking light from.
    hemiTilt.copy(dir);
    if (hemiTilt.lengthSq() < 1e-6) hemiTilt.set(0, 1, 0);
    hemiTilt.normalize();
    hemiBase = HEMI_NIGHT_INTENSITY +
      (HEMI_DAY_INTENSITY - HEMI_NIGHT_INTENSITY) * dayMix;
    ambientBase = 0.02 + 0.012 * nightMix;
    applyIndoor();

    /* --- fog: the horizon of the sky you are actually looking at, warmed --- */
    // Hue and level are decided separately. The hue comes from the sky (so the
    // haze can never disagree with the band it blends into) pulled toward a warm
    // afternoon haze; the level is a fraction of the sky's own horizon
    // luminance, hard-capped. Note fog.color is read by three with
    // getRGB(workingColorSpace) when rendering into a target, so these are
    // linear values and no conversion happens.
    if (sky && sky.horizonColour) {
      normalisedHue(sky.horizonColour, _fogCol);
      if (dayMix > 0) _fogCol.lerp(FOG_HAZE_HUE, FOG_HAZE_MIX * dayMix);
      const target = Math.min(
        FOG_MAX_LUMINANCE,
        luminance(sky.horizonColour) * FOG_LUMINANCE_SCALE);
      fog.color.copy(_fogCol)
        .multiplyScalar(Math.max(0, target) / Math.max(1e-4, luminance(_fogCol)));
    } else {
      fog.color.setRGB(0.40, 0.38, 0.34, THREE.LinearSRGBColorSpace);
    }
    fog.near = FOG_NEAR;
    fog.far = FOG_FAR;
    stats.fogLuminance = Math.round(luminance(fog.color) * 1000) / 1000;
  }

  /* ------------------------------------------------------- interior mode */

  /**
   * Everything the indoor/outdoor state does to the always-present lights. The
   * real work is still done by the interior materials' own `envMapIntensity`
   * (INTERIOR_ENV) and by the hearth in the light pool. What is left here is the
   * fill, and the whole point of doing it through the hemisphere rather than
   * through the AmbientLight is that a hemisphere has a shape:
   *
   *   level  taken down by INDOOR_HEMI_CUT, because most of the sky is walled off
   *   hue    cool toward the glazing, warm toward the boards
   *   axis   tilted off vertical toward the sun, so the two lobes lie ACROSS the
   *          room instead of merely above and below it
   *
   * Everything below is multiplied by indoorMix, so at indoorMix 0 this function
   * writes exactly the values the exterior was calibrated and signed off with:
   * hemi 0.22 on (HEMI_SKY, HEMI_GROUND) with axis (0,1,0), ambient 0.02 on
   * 0x2a2f38. The exterior seen through a doorway while the player is inside
   * still moves, by ~0.5% of its irradiance — under the threshold of a visible
   * change, which is why this is done globally rather than with two material sets.
   */
  function applyIndoor() {
    const m = indoorMix;
    hemi.intensity = hemiBase * (1 - INDOOR_HEMI_CUT * m);
    hemi.color.copy(hemiSkyBase).lerp(INDOOR_HEMI_SKY, INDOOR_HEMI_TINT_MIX * m);
    hemi.groundColor.copy(hemiGroundBase)
      .lerp(INDOOR_HEMI_GROUND, INDOOR_HEMI_TINT_MIX * m);
    // three reads a HemisphereLight's axis from its world position, so this is
    // the whole tilt. Length is irrelevant (it is normalised in WebGLLights) but
    // it must not be zero, and (0,1,0) must be exact outdoors.
    if (m <= 0) {
      hemi.position.set(0, 1, 0);
    } else {
      _hemiAxis.set(0, 1, 0).lerp(hemiTilt, INDOOR_HEMI_TILT * m);
      if (_hemiAxis.lengthSq() < 1e-6) _hemiAxis.set(0, 1, 0);
      hemi.position.copy(_hemiAxis.normalize());
    }
    hemi.updateMatrixWorld();
    // Black-floor guard only. It is NOT the interior fill; see INDOOR_AMBIENT.
    ambient.intensity = ambientBase + (INDOOR_AMBIENT - ambientBase) * m;
    ambient.color.copy(ambientBaseColour).lerp(INDOOR_AMBIENT_TINT, m);
    stats.indoor = Math.round(m * 100) / 100;
    stats.ambientIntensity = Math.round(ambient.intensity * 1000) / 1000;
    stats.hemiIntensity = Math.round(hemi.intensity * 1000) / 1000;
  }

  /* ------------------------------------------------------ shadow fitting */

  function fitShadow(playerPosition) {
    const outdoorDistance = q.shadowDistance || 80;
    // Indoors, pull the cascade in so the texels land on window bars and door
    // reveals instead of on hills nobody can see from a taproom. Quantised to
    // 0.5 m so the 0.42 s ramp re-fits the ortho box ~20 times, not 25 a second,
    // and so the texel size holds still once the ramp lands.
    const indoorDistance = Math.min(outdoorDistance, INDOOR_SHADOW_DISTANCE);
    const distance = Math.round(
      (outdoorDistance + (indoorDistance - outdoorDistance) * indoorMix) * 2) / 2;
    const mapSize = sun.shadow.mapSize.x;

    if (distance !== shadowDistanceUsed || mapSize !== shadowMapUsed) {
      shadowDistanceUsed = distance;
      shadowMapUsed = mapSize;
      const halfH = SHADOW_WORLD_HEIGHT * 0.5;
      const R = distance * 0.5;
      // Fit a sphere, not a box: a sphere is invariant under the sun's
      // rotation, so the cascade does not change size as the day advances.
      shadowExtent = Math.sqrt(R * R + halfH * halfH);
      const cam = sun.shadow.camera;
      cam.left = -shadowExtent;
      cam.right = shadowExtent;
      cam.top = shadowExtent;
      cam.bottom = -shadowExtent;
      cam.near = 18;
      cam.far = 2 * shadowExtent + 40;
      cam.updateProjectionMatrix();
      stats.shadowExtent = Math.round(shadowExtent * 10) / 10;
      stats.shadowTexelsPerMetre = Math.round(mapSize / (2 * shadowExtent) * 10) / 10;
      stats.shadowTexelCm = Math.round((2 * shadowExtent) / mapSize * 10000) / 100;
    }

    _f.copy((sky && sky.sunDirection) || sunDirectionAt(timeOfDay, _tmpDir)).normalize();
    if (Math.abs(_f.y) > 0.9995) _f.set(0.02, _f.y, 0.02).normalize();

    // Exactly the basis Matrix4.lookAt() will build for the shadow camera:
    // z = f, x = normalize(cross(up, f)), y = cross(f, x). Snapping in any
    // other frame does nothing.
    _r.crossVectors(_worldUp, _f).normalize();
    _u.crossVectors(_f, _r).normalize();

    _centre.copy(playerPosition);
    _centre.y += SHADOW_WORLD_HEIGHT * 0.5 - 3;

    const texel = (2 * shadowExtent) / shadowMapUsed;
    const cr = Math.round(_centre.dot(_r) / texel) * texel;
    const cu = Math.round(_centre.dot(_u) / texel) * texel;
    const cf = _centre.dot(_f);

    _snapped.set(0, 0, 0)
      .addScaledVector(_r, cr)
      .addScaledVector(_u, cu)
      .addScaledVector(_f, cf);

    sun.target.position.copy(_snapped);
    sun.target.updateMatrixWorld();
    sun.position.copy(_snapped).addScaledVector(_f, shadowExtent + 20);
    sun.updateMatrixWorld();
  }

  /* --------------------------------------------------------- light anchors */

  /**
   * @param {Array<{position:number[], colour?:*, color?:*, intensity?:number,
   *   kind?:string, room?:*, indoor?:boolean}>} list
   *
   * `room` and `indoor` are OPTIONAL and are the cheap path to room-aware light
   * selection: an interior stream that stamps `room: <the RoomSpec>` (or any
   * stable key) on the anchors it emits needs nothing else wired anywhere. Absent
   * that, a `roomAt` resolver handed to `setRoomResolver()` is used instead, and
   * absent both, selection is pure distance exactly as before.
   */
  function addLightAnchors(list) {
    if (!Array.isArray(list)) return;
    for (const a of list) {
      if (!a || !a.position) continue;
      const kind = KIND[a.kind] ? a.kind : 'lantern';
      const base = KIND[kind] || KIND_DEFAULT;
      const raw = a.colour !== undefined ? a.colour : a.color;
      const colour = new THREE.Color();
      const authored = raw !== undefined && raw !== null;
      if (!authored) colour.set(0xffb46a);
      else if (raw.isColor) colour.copy(raw);
      else colour.set(raw);
      anchors.push({
        x: a.position[0] || 0,
        y: a.position[1] || 0,
        z: a.position[2] || 0,
        colour,
        /**
         * For a fire, the authored colour is treated as a CORRECTION on the
         * blackbody rather than replacing it, so the temperature swing survives
         * whatever hue the emitting stream asked for.
         */
        authoredColour: authored && base.fire,
        scale: typeof a.intensity === 'number' ? a.intensity : 1,
        kind,
        base,
        /** Pre-merged indoor variant, so update() never builds an object. */
        merged: base.indoor ? { ...base, ...base.indoor } : base,
        spec: base,
        room: 0,
        indoor: false,
        declaredRoom: a.room !== undefined ? a.room : undefined,
        declaredIndoor: !!a.indoor,
        rangeSq: ANCHOR_RANGE_SQ,
        phase: anchorRng.range(0, Math.PI * 2),
        rate: anchorRng.range(0.85, 1.2),
      });
    }
    stats.anchors = anchors.length;
    // New anchors need their rooms resolved on the next tick.
    anchorRoomsAt = -1;
  }

  /**
   * Lantern/window wobble: three fast sines, unchanged — this is what the
   * exterior was signed off with.
   */
  function flickerFor(a, t) {
    const amp = a.spec.flicker;
    if (amp <= 0) return 1;
    const p = a.phase;
    const r = a.rate;
    return 1 +
      amp * 0.55 * Math.sin(t * 8.3 * r + p) +
      amp * 0.30 * Math.sin(t * 13.7 * r + p * 2.3) +
      amp * 0.15 * Math.sin(t * 23.1 * r + p * 3.1);
  }

  /**
   * A fire, which is a different animal from a lantern flame. Three bands, all
   * deterministic off the anchor's Rng phase:
   *
   *   wander  0.9 / 1.63 / 2.71 Hz, mutually incommensurate — the log settling,
   *           the draught turning over. This carries most of the amplitude,
   *           because it is what the eye actually reads as "a fire".
   *   jitter  11.3 / 17.9 Hz at a sixth of the amplitude — the flame edge.
   *   flare   a rectified 0.37 Hz beat that spends ~92% of its time at zero and
   *           then throws a short, sharp lift. Without this a fire animates
   *           evenly and reads as a sine, not a flame.
   *
   * Returns roughly [0.74, 1.45] for the fire kind's amp of 0.30. Second return
   * value (via `_fireT`) is the 0..1 gamut position: brighter = hotter = less
   * red, because that is what a flaring fire does.
   */
  let _fireT = 0.5;
  function fireFlicker(a, t) {
    const amp = a.spec.flicker;
    if (amp <= 0) { _fireT = 0.5; return 1; }
    const p = a.phase;
    const r = a.rate;
    const wander =
      0.55 * Math.sin(t * 0.90 * r + p) +
      0.30 * Math.sin(t * 1.63 * r + p * 1.7) +
      0.15 * Math.sin(t * 2.71 * r + p * 2.9);
    const jitter =
      0.60 * Math.sin(t * 11.3 * r + p * 3.7) +
      0.40 * Math.sin(t * 17.9 * r + p * 5.1);
    const flare = Math.max(0, Math.sin(t * 0.37 * r + p * 0.6) - 0.86) * 7.1;
    const shape = 0.72 * wander + 0.16 * jitter + 0.55 * flare;
    _fireT = Math.max(0, Math.min(1, 0.46 + 0.62 * shape));
    return 1 + amp * shape;
  }

  /**
   * How much this anchor deserves the shadow-casting slot. A hearth in the room
   * the player is standing in wins outright; everything else is ranked so that
   * the map is at least pointed at something worth resolving.
   */
  function shadowWorth(a, d2) {
    if (!a) return -1e3;
    let w = a.kind === 'fire' ? 3 : a.kind === 'lantern' ? 1 : 0.3;
    if (a.indoor && a.room !== 0 && a.room === playerRoom) w += 6;
    else if (a.indoor) w += 1;
    return w - Math.sqrt(d2) * 0.06;
  }

  function updatePool(dt, playerPosition) {
    const n = anchors.length;
    let active = 0;

    if (anchorRoomsAt !== roomEpoch) resolveAnchorRooms();

    if (n > 0) {
      /* --- top-K by room-penalised score, no sort, no allocation --- */
      for (let i = 0; i < poolSize; i++) { bestIdx[i] = -1; bestScore[i] = Infinity; }
      for (let i = 0; i < n; i++) {
        const a = anchors[i];
        const dx = a.x - playerPosition.x;
        const dy = a.y - playerPosition.y;
        const dz = a.z - playerPosition.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        // The range test is on real metres — the penalty decides ranking, not
        // reach, or an indoor anchor could be culled by its own penalty.
        let score = d2 * (a.spec.weight || 1);
        if (a.room !== playerRoom && !a.neutral) {
          if (a.room !== 0 && playerRoom !== 0) {
            // Sideways through a doorway, yes; through a floor, never. A candle
            // in the bedroom above is 3 m from your head with a joist floor in
            // between, and a point light would put its pool straight through the
            // boards onto the taproom floor, which faces it.
            if (d2 > CROSS_ROOM_RANGE_SQ || dy * dy > 2.25) continue;
            score *= PENALTY_CROSS_ROOM;
          } else if (a.room !== 0) score *= PENALTY_OUTSIDE_IN;
          else score *= PENALTY_INSIDE_OUT;
        }
        if (d2 > a.rangeSq) continue;
        if (score >= bestScore[poolSize - 1]) continue;
        let j = poolSize - 1;
        while (j > 0 && bestScore[j - 1] > score) {
          bestScore[j] = bestScore[j - 1];
          bestIdx[j] = bestIdx[j - 1];
          j--;
        }
        bestScore[j] = score;
        bestIdx[j] = i;
      }

      /* --- keep slots that are still in the set, retire the rest --- */
      claimed.fill(0);
      for (let s = 0; s < poolSize; s++) {
        const slot = slots[s];
        if (slot.anchor < 0) continue;
        let found = -1;
        for (let k = 0; k < poolSize; k++) {
          if (bestIdx[k] === slot.anchor) { found = k; break; }
        }
        if (found >= 0 && !claimed[found]) {
          claimed[found] = 1;
          slot.retiring = false;
        } else {
          slot.retiring = true;
        }
      }

      /* --- hand unclaimed anchors to slots that have finished fading out --- */
      for (let k = 0; k < poolSize; k++) {
        if (bestIdx[k] < 0 || claimed[k]) continue;
        for (let s = 0; s < poolSize; s++) {
          const slot = slots[s];
          if (slot.anchor >= 0 && !(slot.retiring && slot.fade <= 0.001)) continue;
          slot.anchor = bestIdx[k];
          slot.retiring = false;
          slot.fade = 0;
          slot.shadowClock = 0;
          slot.shadowWarm = 2;
          claimed[k] = 1;
          break;
        }
      }
    } else {
      for (let s = 0; s < poolSize; s++) slots[s].retiring = true;
    }

    /* --- put the shadow-casting slot on the most deserving anchor -----------
     * `castShadow` is fixed per slot for the lifetime of the run (flipping it
     * rewrites NUM_POINT_LIGHT_SHADOWS and recompiles the village), so the only
     * way to give the hearth a shadow is to move the ANCHOR into the slot that
     * already casts. Swapping the whole slot state — anchor, fade, retiring —
     * is seamless: the light's position, colour and intensity are recomputed
     * from the anchor every frame anyway. Only the shadow map has to be
     * re-rendered, which the clock below does on the next tick. */
    if (shadowCasters > 0 && n > 0) {
      swapClock -= dt;
      if (swapClock <= 0) {
        swapClock = 0.45;
        for (let s = 0; s < poolSize; s++) {
          const slot = slots[s];
          if (slot.anchor < 0) { slot.worth = -1e3; continue; }
          const a = anchors[slot.anchor];
          const dx = a.x - playerPosition.x;
          const dy = a.y - playerPosition.y;
          const dz = a.z - playerPosition.z;
          slot.worth = shadowWorth(a, dx * dx + dy * dy + dz * dz);
        }
        for (let c = 0; c < shadowCasters; c++) {
          let bestS = -1;
          let bestW = slots[c].worth + 0.6;   // hysteresis: do not chatter
          for (let s = shadowCasters; s < poolSize; s++) {
            if (slots[s].worth > bestW) { bestW = slots[s].worth; bestS = s; }
          }
          if (bestS < 0) continue;
          const A = slots[c];
          const B = slots[bestS];
          const anchor = A.anchor, fade = A.fade, retiring = A.retiring, worth = A.worth;
          A.anchor = B.anchor; A.fade = B.fade; A.retiring = B.retiring; A.worth = B.worth;
          B.anchor = anchor; B.fade = fade; B.retiring = retiring; B.worth = worth;
          A.shadowClock = 0;
          A.shadowWarm = 2;
        }
      }
      const held = slots[0].anchor >= 0 ? anchors[slots[0].anchor] : null;
      stats.shadowAnchorKind = held ? (held.indoor ? `${held.kind}/indoor` : held.kind) : '';
    }

    /* --- drive the lights --- */
    const fadeStep = dt / FADE_TIME;
    for (let s = 0; s < poolSize; s++) {
      const slot = slots[s];
      const light = slot.light;

      if (slot.anchor < 0 || slot.retiring) {
        slot.fade = Math.max(0, slot.fade - fadeStep);
        if (slot.fade <= 0 && slot.retiring) { slot.anchor = -1; slot.retiring = false; }
      } else {
        slot.fade = Math.min(1, slot.fade + fadeStep);
      }

      if (slot.anchor < 0 || slot.fade <= 0.0005) {
        if (light.intensity !== 0) light.intensity = 0;
        continue;
      }

      const a = anchors[slot.anchor];
      const spec = a.spec;
      const level = spec.dayFloor + (1 - spec.dayFloor) * nightMix;
      const flick = spec.fire ? fireFlicker(a, elapsed) : flickerFor(a, elapsed);
      const value = spec.candela * a.scale * level * slot.fade * flick;

      light.position.set(a.x, a.y, a.z);
      if (spec.fire) {
        // Colour temperature rides the flicker: 1850 K when the fire has settled
        // back, 2250 K on a flare. An authored anchor colour is applied as a
        // correction on the mid-gamut reference, so a stream that asked for a
        // particular hue keeps it and still gets the breathing.
        fireColourAt(_fireT, _fireCol);
        if (a.authoredColour) {
          _fireCol.setRGB(
            _fireCol.r * (a.colour.r / Math.max(1e-4, FIRE_REF[0])),
            _fireCol.g * (a.colour.g / Math.max(1e-4, FIRE_REF[1])),
            _fireCol.b * (a.colour.b / Math.max(1e-4, FIRE_REF[2])),
            THREE.LinearSRGBColorSpace);
        }
        light.color.copy(_fireCol);
      } else {
        light.color.copy(a.colour);
      }
      light.distance = spec.distance;
      light.intensity = value;
      if (value > 0.02) active++;

      if (light.castShadow) {
        slot.shadowClock -= dt;
        // Nothing at all while the light is dark, and nothing while an indoor
        // anchor's room is not the one the player is in — six cube faces at 5 Hz
        // is the single most expensive thing this file can ask for.
        const worthUpdating = value > 0.05 &&
          (!a.indoor || indoorMix > 0.02 || a.room === playerRoom);
        if (slot.shadowClock <= 0 && worthUpdating) {
          // A point-light shadow is VIEW-INDEPENDENT, and a hearth does not move:
          // the flicker changes its intensity, not its geometry, so re-rendering
          // six cube faces five times a second buys nothing. Two quick refreshes
          // to catch interior geometry that streamed in after the slot was
          // assigned, then 1.1 Hz, which is only there to pick up a barrel the
          // player rolled across the floor. Outdoor anchors keep the old 5 Hz
          // because the props near them are throwable.
          if (slot.shadowWarm > 0) { slot.shadowWarm--; slot.shadowClock = 0.09; }
          else slot.shadowClock = a.indoor ? 0.9 : 0.2;
          // Note there is nothing to set here: WebGLShadowMap overwrites
          // `shadow.camera.far` with `light.distance` for every point light
          // (WebGLShadowMap.js: `const far = light.distance || camera.far`), so
          // the ONLY thing bounding how much of the village a cube render walks
          // is `light.distance` — which is why the indoor spec pulls a fire in
          // from 16 m to 11 m. The line this replaces was dead code.
          light.shadow.needsUpdate = true;
        }
      }
    }

    stats.activeLights = active;
  }

  /* ------------------------------------------------------------ room probe */

  /**
   * One resolver call per ~0.12 s, and only when the player has actually moved.
   * `roomAt` is the interiors stream's code, not ours, and it is presumably an
   * AABB walk over 11 buildings — cheap, but not free, and there is no reason to
   * pay for it 60 times a second when a room is metres across.
   */
  function updateRoom(dt, playerPosition) {
    roomProbeClock -= dt;
    if (roomResolver && !playerRoomForced && roomProbeClock <= 0) {
      // 20 Hz while walking (a threshold has to register promptly), 6 Hz when
      // standing still.
      const moved = _lastProbe.distanceToSquared(playerPosition) > 0.0225; // 0.15 m
      roomProbeClock = moved ? 0.05 : 0.16;
      _lastProbe.copy(playerPosition);
      let r = null;
      try {
        // The eye, not the feet: a player standing in a doorway has their feet
        // on the threshold and their head unambiguously in one room or the other.
        r = roomResolver(playerPosition.x, playerPosition.y + 1.5, playerPosition.z);
      } catch (err) {
        console.warn('[lighting] roomAt() threw; light selection is distance-only ' +
          'from here on.', err);
        roomResolver = null;
      }
      // Interning builds a string key for an object identity, so only do it when
      // the resolver actually returns something new.
      if (r !== lastRoomObject) {
        lastRoomObject = r;
        playerRoom = internRoom(r);
      }
    }
    stats.playerRoom = playerRoom;

    const target = playerRoom !== 0 ? 1 : 0;
    if (indoorMix !== target) {
      const step = dt / INDOOR_RAMP;
      indoorMix = target > indoorMix
        ? Math.min(target, indoorMix + step)
        : Math.max(target, indoorMix - step);
      applyIndoor();
    }
  }

  /* ---------------------------------------------------------------- api */

  applyTime(DEFAULT_TIME_OF_DAY);
  fitShadow(new THREE.Vector3(0, 0, 0));

  return {
    sun,
    ambient,
    hemi,
    fog,
    addLightAnchors,

    setTimeOfDay(t01) {
      applyTime(t01);
    },

    /* ------------------------------------------------------------ interiors */

    /**
     * Hand over the interiors stream's `roomAt(x, y, z)`. It must return a stable
     * identity for the room containing that point (the RoomSpec itself is ideal)
     * and something falsy outdoors. Called once; wire it in main.js right after
     * the interiors chunk is built:
     *
     *     lighting.setRoomResolver(interiors.roomAt);
     *
     * Everything works without it — selection just falls back to pure distance,
     * and interior fires keep the outdoor day floor.
     */
    setRoomResolver(fn) {
      roomResolver = typeof fn === 'function' ? fn : null;
      roomEpoch++;
      anchorRoomsAt = -1;
      stats.roomAware = !!roomResolver;
      // Anchors are re-resolved on the next tick; dropping the resolver leaves
      // only whatever rooms the anchors declared for themselves.
      if (!roomResolver && !playerRoomForced) playerRoom = 0;
      return this;
    },

    /**
     * Override the player's room directly, for a caller that already knows it
     * (a trigger volume, say) and would rather not pay for a resolver call.
     * Passing `undefined` hands control back to the resolver.
     */
    setPlayerRoom(room) {
      if (room === undefined) { playerRoomForced = false; return; }
      playerRoomForced = true;
      playerRoom = internRoom(room);
    },

    /** The env-map scales interior surfaces must be built with. */
    interior: INTERIOR_ENV,
    get indoorMix() { return indoorMix; },

    update(dt, ctx) {
      elapsed += dt;
      const p = (ctx && ctx.playerPosition) || (ctx && ctx.camera && ctx.camera.position);
      if (!p) return;
      updateRoom(dt, p);
      fitShadow(p);
      updatePool(dt, p);
    },

    dispose() {
      scene.remove(sun);
      scene.remove(sun.target);
      scene.remove(hemi);
      scene.remove(ambient);
      for (const slot of slots) {
        slot.light.shadow?.dispose?.();
        slot.light.dispose?.();
        scene.remove(slot.light);
      }
      sun.shadow?.dispose?.();
      sun.dispose?.();
      hemi.dispose?.();
      ambient.dispose?.();
      if (scene.fog === fog) scene.fog = null;
      anchors.length = 0;
    },

    stats,
  };
}
