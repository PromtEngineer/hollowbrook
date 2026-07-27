/**
 * Global tunables and quality presets.
 *
 * Two classes of knob live here:
 *   - BAKE-TIME  : fixed when the world is built (texture resolution, mesh density,
 *                  env map size). Changing them requires a reload.
 *   - RUNTIME    : safe to change every frame (pixel ratio, AO on/off, bloom,
 *                  shadow distance, vegetation cull distance).
 * The performance manager may only touch RUNTIME knobs.
 */

export const WORLD = {
  /** Metres. Everything in the project is authored in metres. */
  unit: 1,
  gravity: -19.6, // 2g — reads better for a first-person game than true 9.81
  /** Fixed physics step. Rendering interpolates between steps. */
  fixedStep: 1 / 60,
  maxSubSteps: 5,
  /** Radius of the flat, walkable village floor. Terrain is guaranteed y=0 inside. */
  villageRadius: 46,
  /** Everything past this is scenery only (no colliders, no interaction). */
  sceneryRadius: 900,
  playerStart: [0, 0, 34],
  // A camera at yaw θ faces (-sinθ, 0, -cosθ), so 0 looks along -Z: from the
  // southern approach, straight into the plaza.
  playerStartYaw: 0,
};

export const PLAYER = {
  eyeHeight: 1.68,
  crouchEyeHeight: 1.02,
  radius: 0.32,
  /** Total capsule height standing (feet to crown). */
  height: 1.8,
  crouchHeight: 1.15,
  walkSpeed: 3.0,
  sprintSpeed: 5.9,
  crouchSpeed: 1.45,
  accelGround: 46,
  accelAir: 7,
  friction: 11,
  jumpVelocity: 6.2,
  maxSlopeDeg: 48,
  stepHeight: 0.45,
  /**
   * Minimum flat width Rapier needs on top of a step before it will autostep up.
   * Rapier's default and our first value was 0.25 m, which is wider than the top
   * tread of a worn doorstep: two buildings (weaver, granary) were unenterable
   * because the player stalled on the threshold and could only get in by
   * jumping, while the doorway itself measured completely clear. 0.12 m is still
   * half a bootprint, so it will not let the player walk up a fence rail.
   */
  stepMinWidth: 0.12,
  /** Capsule skin width handed to the Rapier character controller. */
  skin: 0.02,
  snapToGround: 0.4,
  reach: 3.0,
  carryDistance: 1.55,
  throwImpulse: 9.0,
  maxCarryMass: 40,
  coyoteTime: 0.12,
  jumpBuffer: 0.14,
};

export const CAMERA = {
  fov: 72,
  near: 0.08,
  far: 2200,
};

/**
 * Quality presets. `bake` entries are locked at load, `runtime` entries adapt.
 */
export const QUALITY_PRESETS = {
  low: {
    label: 'Low',
    textureSize: 512,
    envSize: 128,
    shadowMapSize: 1024,
    shadowDistance: 55,
    anisotropy: 4,
    geometryDetail: 0.55,
    vegetationDensity: 0.3,
    thatchDetail: 0,
    pixelRatioCap: 1.0,
    antialias: 'fxaa',
    bloom: 0.45,
    gtao: false,
    softShadows: false,
    contactShadows: false,
    cullDistance: 90,
    grassDistance: 26,
    maxDynamicBodies: 40,
  },
  medium: {
    label: 'Medium',
    textureSize: 1024,
    envSize: 256,
    shadowMapSize: 2048,
    shadowDistance: 72,
    anisotropy: 8,
    geometryDetail: 0.8,
    vegetationDensity: 0.6,
    thatchDetail: 1,
    pixelRatioCap: 1.25,
    antialias: 'smaa',
    bloom: 0.5,
    gtao: false,
    softShadows: true,
    contactShadows: true,
    cullDistance: 140,
    grassDistance: 40,
    maxDynamicBodies: 70,
  },
  high: {
    label: 'High',
    textureSize: 1024,
    envSize: 512,
    shadowMapSize: 4096,
    shadowDistance: 95,
    anisotropy: 16,
    geometryDetail: 1.0,
    vegetationDensity: 1.0,
    thatchDetail: 2,
    pixelRatioCap: 1.5,
    antialias: 'smaa',
    bloom: 0.55,
    gtao: true,
    softShadows: true,
    contactShadows: true,
    cullDistance: 200,
    grassDistance: 58,
    maxDynamicBodies: 110,
  },
  ultra: {
    label: 'Ultra',
    textureSize: 2048,
    envSize: 512,
    shadowMapSize: 4096,
    shadowDistance: 120,
    anisotropy: 16,
    geometryDetail: 1.25,
    vegetationDensity: 1.35,
    thatchDetail: 2,
    pixelRatioCap: 2.0,
    antialias: 'smaa',
    bloom: 0.6,
    gtao: true,
    softShadows: true,
    contactShadows: true,
    cullDistance: 260,
    grassDistance: 74,
    maxDynamicBodies: 150,
  },
};

/** Knobs the performance manager is allowed to change at runtime. */
export const RUNTIME_KNOBS = [
  'pixelRatioCap',
  'shadowDistance',
  'gtao',
  'bloom',
  'cullDistance',
  'grassDistance',
  'contactShadows',
];

export const RENDER = {
  toneMappingExposure: 1.0,
  /** Fog is height-based haze; tuned by the lighting stream. */
  fogNear: 60,
  fogFar: 620,
};

/**
 * Guess a sane starting preset. Deliberately conservative: the performance
 * manager ramps up once it has real frame timings.
 */
export function detectQuality() {
  const url = new URLSearchParams(location.search).get('quality');
  if (url && QUALITY_PRESETS[url]) return url;

  const cores = navigator.hardwareConcurrency || 4;
  const mem = navigator.deviceMemory || 4;
  const mobile = /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);
  if (mobile) return 'low';

  let gpu = '';
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl2') || c.getContext('webgl');
    if (gl) {
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      gpu = (ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)) || '';
      const lose = gl.getExtension('WEBGL_lose_context');
      if (lose) lose.loseContext();
    }
  } catch { /* fall through to core-count heuristic */ }

  const g = gpu.toLowerCase();
  const strong = /apple m[1-9]|rtx (30|40|50)|radeon rx (6|7|9)|arc a7/.test(g);
  const weak = /intel.*(hd|uhd|iris\s*(plus)?\s*6|gma)|mali|adreno|swiftshader|llvmpipe/.test(g);

  if (weak) return 'low';
  if (strong && cores >= 8) return 'high';
  if (cores >= 8 && mem >= 8) return 'high';
  if (cores >= 4) return 'medium';
  return 'low';
}

export function presetFor(name) {
  return { ...QUALITY_PRESETS[name] || QUALITY_PRESETS.medium, name };
}
