/**
 * ============================================================================
 *  MATERIAL LIBRARY
 * ============================================================================
 * The only place in the project that constructs a THREE.Material. Geometry
 * streams ask for a canonical name from `MATERIAL_KEYS`; an unknown name gets a
 * loud magenta debug material and one console warning, so a typo shows up as a
 * screaming surface in the world instead of a thrown exception.
 *
 * Conventions this library assumes and relies on:
 *
 *  - UVs are AUTHORED IN METRES DIVIDED BY worldScale. Every texture set
 *    publishes `worldScale` (metres covered by one uv tile); a wall 6 m wide
 *    using `plaster` (worldScale 2.5) should have u running 0 -> 2.4. Do that
 *    and every surface in the village has a consistent texel density and no
 *    material ever needs a bespoke `repeat`. `materials.worldScale(name)`
 *    returns the number.
 *
 *  - Roughness/metalness/AO all come from ONE packed ORM texture (glTF layout:
 *    r = AO, g = roughness, b = metalness). `material.roughness` and
 *    `material.metalness` are therefore MULTIPLIERS, and default to 1.
 *
 *  - `aoMap` samples uv channel 0 in three >= r151 — no uv2 needed.
 *
 *  - Cut-out foliage carries its alpha in `map.a` and uses `alphaTest`, not
 *    `alphaMap` (three reads alphaMap from the GREEN channel, which would be
 *    wrong) and not `transparent` (which would sort and cost fill rate).
 *
 *  - `setEnvironment` assigns `envMap` on every material rather than leaning on
 *    `scene.environment`. It has to: WebGLRenderer overwrites
 *    `envMapIntensity` with `scene.environmentIntensity` whenever
 *    `material.envMap === null`, so without an explicit envMap the per-material
 *    intensity knob silently does nothing.
 * ============================================================================
 */

import * as THREE from 'three';
import { MATERIAL_KEYS, variantKey } from '../contracts.js';

/** sRGB grey whose LINEAR value is `k` — i.e. a texture multiplier of k. */
function dim(k) {
  const v = Math.max(0, Math.min(255, Math.round(255 * Math.pow(k, 1 / 2.2))));
  return (v << 16) | (v << 8) | v;
}

const SIDES = {
  front: THREE.FrontSide,
  back: THREE.BackSide,
  double: THREE.DoubleSide,
};

/**
 * Canonical definitions.
 *   tex          texture set name (omit for untextured flat materials)
 *   color        tint multiplied into the albedo (sRGB hex)
 *   rough/metal  MULTIPLIERS over the ORM map, or absolute when there is no map
 *   env          envMapIntensity scale relative to the global environment level
 *   foliage      alpha cut-out card: DoubleSide + alphaTest, no ORM
 */
const DEFS = {
  /* ---------------------------------------------------------- walls & frame */
  plaster:       { tex: 'plaster', normalScale: 0.9, env: 0.85 },
  plasterWarm:   { tex: 'plaster', color: 0xfff0d4, normalScale: 0.9, env: 0.85 },
  plasterWorn:   { tex: 'plasterWorn', normalScale: 1.0, env: 0.8 },

  timber:        { tex: 'timber', env: 0.7 },
  timberDark:    { tex: 'timber', color: dim(0.55), env: 0.6 },
  woodBeam:      { tex: 'woodBeam', env: 0.75 },

  stone:         { tex: 'stone', env: 0.85 },
  stoneTrim:     { tex: 'stone', color: 0xe6e2d6, normalScale: 0.7, rough: 0.92, env: 0.95 },
  brick:         { tex: 'brick', env: 0.8 },

  /* ------------------------------------------------------------------ roofs */
  // normalScale > 1: the straw relief is the whole read of a thatched roof in
  // raking late-afternoon sun, and it is the first thing that disappears once
  // the roof is more than a few metres away.
  thatch:        { tex: 'thatch', normalScale: 1.15, env: 0.6, aoIntensity: 1.0 },
  // The ridge cap is a separate, darker sedge laid ACROSS the roof, so the
  // fibre direction is rotated by giving it a stretched repeat.
  // A repeat cannot rotate a texture: the ridge geometry must SWAP its u and v
  // so the straw runs along the ridge instead of down the pitch.
  thatchRidge:   { tex: 'thatch', color: dim(0.62), repeat: [2.2, 0.85], normalScale: 1.35, env: 0.55 },
  roofTile:      { tex: 'roofTile', env: 0.8 },

  /* --------------------------------------------------------------- joinery */
  woodDoor:      { tex: 'woodPlank', color: 0xd2b48c, env: 0.7 },
  woodDoorTeal:  { tex: 'woodPainted', color: 0x2f6f6a, rough: 0.85, env: 0.9 },
  woodPlank:     { tex: 'woodPlank', env: 0.7 },
  woodDark:      { tex: 'woodPlank', color: dim(0.42), env: 0.6 },

  iron:          { tex: 'iron', env: 1.15 },
  ironDark:      { tex: 'iron', color: dim(0.55), rough: 1.1, env: 1.0 },
  copper:        { tex: 'copper', env: 1.1 },
  brass:         { tex: 'brass', env: 1.25 },

  /* --------------------------------------------------------------- glazing */
  // No transmission: a dark, very smooth, high-env surface reads as old glass
  // from outside and costs a fraction of a transmissive pass.
  glassWindow:   { tex: 'glass', color: 0xb6c6cc, rough: 1.0, metal: 0.0, env: 2.6 },
  glassLantern:  {
    tex: 'glass', useOrm: false, color: 0x3a3226, rough: 0.16, metal: 0.0,
    env: 2.0, transparent: true, opacity: 0.45, side: 'double', depthWrite: false,
  },
  lanternEmissive: {
    color: 0x140d06, rough: 0.85, metal: 0.0,
    emissive: 0xffb45a, emissiveIntensity: 3.2, env: 0.3,
  },
  windowGlow: {
    color: 0x0e0b07, rough: 0.9, metal: 0.0,
    emissive: 0xffc884, emissiveIntensity: 1.5, env: 0.3,
  },

  /* ---------------------------------------------------------------- ground */
  cobble:        { tex: 'cobble', env: 0.9 },
  cobbleWorn:    { tex: 'cobbleWorn', env: 0.95 },
  soil:          { tex: 'soil', env: 0.7 },
  grass:         { tex: 'grass', normalScale: 0.8, env: 0.8 },
  // `color` is a MULTIPLIER, so it can only ever darken: 0xd8c8a8 reads as "pale
  // warm earth" but measured as soil x 0.58 linear, which made the lanes the
  // darkest ground in the village. A pale lane has to come from the `soil` set
  // itself (now 0.060 linear, up from 0.036); the tint is just a warm cast.
  dirtPath:      { tex: 'soil', color: 0xf6efe2, normalScale: 0.7, env: 0.8 },

  /* -------------------------------------------------------------- dressing */
  terracotta:    { tex: 'terracotta', env: 0.9 },
  terracottaDark:{ tex: 'terracotta', color: dim(0.55), env: 0.8 },
  fabricAwning:  { tex: 'fabric', side: 'double', env: 0.75 },
  rope:          { tex: 'rope', env: 0.6 },
  candleWax:     { color: 0xf3e7d0, rough: 0.52, metal: 0.0, env: 0.9 },

  /* -------------------------------------------------------------- interiors */
  // `env` is deliberately low across the board. envMapIntensity here is the sky
  // IBL, and an indoor surface can only see the sky through one small window —
  // leaving these at 1.0 lights the inside of a cottage like an open courtyard.
  // The hearth and the window shafts belong to the lighting stream, not to this.
  floorBoard:    { tex: 'floorBoard', normalScale: 1.0, env: 0.5 },
  // Extras, not contract keys: the traffic line and the untrodden corners are the
  // same texture set at a different value, so they cost nothing but a clone and
  // interiors.js can lay them as separate strips without a second bake.
  floorBoardWorn:{ tex: 'floorBoard', color: 0xf4eee4, rough: 0.80, normalScale: 0.8, env: 0.55 },
  floorBoardDark:{ tex: 'floorBoard', color: dim(0.60), rough: 1.06, env: 0.4 },
  flagstone:     { tex: 'flagstone', normalScale: 1.0, env: 0.55 },
  lathPlaster:   { tex: 'lathPlaster', normalScale: 0.85, env: 0.7, aoIntensity: 0.65 },
  lathPlasterSmoked: { tex: 'lathPlaster', color: dim(0.66), normalScale: 0.9, env: 0.55, aoIntensity: 0.8 },
  ceilingBeam:   { tex: 'ceilingBeam', normalScale: 1.1, env: 0.45 },
  hearthStone:   { tex: 'hearthStone', normalScale: 1.05, env: 0.45 },
  soot:          { tex: 'soot', normalScale: 0.8, env: 0.2 },
  linen:         { tex: 'linen', normalScale: 1.2, env: 0.55 },
  sackcloth:     { tex: 'sackcloth', normalScale: 1.3, env: 0.5 },
  strawLitter:   { tex: 'strawLitter', normalScale: 1.15, env: 0.5 },
  // A tankard is a mirror with a rough surface: metalness has to stay at 1 (the
  // ORM's blue channel is already 1) or it turns into grey plastic.
  pewter:        { tex: 'pewter', rough: 1.0, metal: 1.0, env: 1.2 },

  /* ------------------------------------------------------------ vegetation */
  // Two things worth knowing about the foliage entries:
  //  - the faint green emissive stands in for transmission. A back-lit leaf card
  //    is otherwise pure black, which reads as cardboard. It sits far below any
  //    bloom threshold, so it lifts shadowed foliage without glowing.
  //  - normalScale > 1 because the atlases are baked against a nominal 1 m uv
  //    tile, while one cell actually covers a leaf ~10 cm across. Without the
  //    boost the veins and the blade dish are an order of magnitude too flat.
  leaf:          { tex: 'leaf', foliage: true, rough: 0.68, env: 0.9, normalScale: 2.0, emissive: 0x24401a, emissiveIntensity: 0.5 },
  leafDark:      { tex: 'leaf', foliage: true, color: 0x9fbe8c, rough: 0.72, env: 0.75, normalScale: 2.0, emissive: 0x1a3212, emissiveIntensity: 0.45 },
  needle:        { tex: 'needle', foliage: true, rough: 0.74, env: 0.7, normalScale: 1.5, emissive: 0x142a16, emissiveIntensity: 0.4 },
  // Ivy is the darkest foliage in the village: less transmission lift and less
  // env than the broadleaves, or the leaves wash out to a pale sticker green.
  ivy:           { tex: 'ivy', foliage: true, rough: 0.66, env: 0.62, normalScale: 1.7, emissive: 0x12240e, emissiveIntensity: 0.35 },
  flowerRed:     { tex: 'flower', foliage: true, color: 0xc8341f, rough: 0.62, env: 1.0, normalScale: 1.4 },
  flowerPink:    { tex: 'flower', foliage: true, color: 0xe4879f, rough: 0.62, env: 1.0, normalScale: 1.4 },
  flowerYellow:  { tex: 'flower', foliage: true, color: 0xe9bd3b, rough: 0.62, env: 1.0, normalScale: 1.4 },
  flowerWhite:   { tex: 'flower', foliage: true, color: 0xf7f1e4, rough: 0.60, env: 1.0, normalScale: 1.4 },
  grassBlade:    { tex: 'grassBlade', foliage: true, rough: 0.76, env: 0.85, normalScale: 1.4 },
  bark:          { tex: 'bark', env: 0.7 },

  /* --------------------------------------------------------------- scenery */
  hillGrass:     { tex: 'grass', color: 0xd6e0cc, normalScale: 0.35, rough: 1.0, env: 0.9 },
  // Beyond ~200 m a map is pure cost: fog and aerial perspective own the look.
  hillFar:       { color: 0x6f7f57, rough: 0.98, metal: 0.0, env: 1.0 },
  mountain:      { color: 0x6d7787, rough: 1.0, metal: 0.0, env: 1.1 },
};

/* -------------------------------------------------------------------------- */

/**
 * @param {import('../contracts.js').TextureLibrary} textures
 * @param {{quality:Object, renderer:THREE.WebGLRenderer}} opts
 * @returns {import('../contracts.js').MaterialLibrary}
 */
export function createMaterialLibrary(textures, { quality, renderer } = {}) {
  const base = new Map();          // name -> canonical material
  const variants = new Map();      // variantKey -> material
  const texClones = new Map();     // uuid|repeat|offset -> cloned texture
  const owned = new Set();         // every material we made
  const warned = new Set();

  let envMap = null;
  let envIntensity = 1;
  let debugMaterial = null;

  const stats = {
    materials: 0, variants: 0, textureClones: 0,
    missing: [], env: false,
  };

  /* ---------------------------------------------------------------- build */

  function applySet(mat, def) {
    if (!def.tex || !textures?.has?.(def.tex)) return;
    const set = textures.get(def.tex);
    if (!set) return;

    mat.map = set.map || null;
    if (set.normalMap) {
      mat.normalMap = set.normalMap;
      const ns = def.normalScale ?? 1;
      mat.normalScale = new THREE.Vector2(ns, ns);
    }

    // Cut-out cards never take an ORM: a leaf has no meaningful cavity AO and
    // a per-card constant roughness is both cheaper and better behaved.
    const orm = (def.useOrm === false || def.foliage) ? null : set.ormMap;
    if (orm) {
      mat.roughnessMap = orm;
      mat.metalnessMap = orm;
      mat.aoMap = orm;
      mat.aoMapIntensity = def.aoIntensity ?? 0.9;
      // With maps present these are multipliers, not absolutes.
      mat.roughness = def.rough ?? 1.0;
      mat.metalness = def.metal ?? 1.0;
    } else {
      mat.roughness = def.rough ?? 0.85;
      mat.metalness = def.metal ?? 0.0;
    }

    mat.userData.textureSet = def.tex;
    mat.userData.worldScale = set.worldScale || 1;
    mat.userData.atlas = set.atlas || null;
    if (def.repeat) applyRepeat(mat, def.repeat, def.offset);
  }

  function build(name) {
    const def = DEFS[name];
    if (!def) return null;

    const mat = new THREE.MeshStandardMaterial({ name });
    mat.color = new THREE.Color(def.color ?? 0xffffff);
    mat.roughness = def.rough ?? 0.85;
    mat.metalness = def.metal ?? 0.0;
    mat.userData.envScale = def.env ?? 1.0;

    applySet(mat, def);

    if (def.emissive !== undefined) {
      mat.emissive = new THREE.Color(def.emissive);
      mat.emissiveIntensity = def.emissiveIntensity ?? 1.0;
    }

    if (def.foliage) {
      mat.side = THREE.DoubleSide;
      mat.alphaTest = 0.42;
      mat.transparent = false;
      mat.depthWrite = true;
      // Cards must cast from both faces or half the canopy drops its shadow.
      mat.shadowSide = THREE.DoubleSide;
    }

    if (def.side) mat.side = SIDES[def.side] ?? THREE.FrontSide;
    if (def.transparent) {
      mat.transparent = true;
      mat.opacity = def.opacity ?? 1.0;
      mat.depthWrite = def.depthWrite ?? true;
    }
    if (def.depthWrite !== undefined) mat.depthWrite = def.depthWrite;
    if (def.alphaTest !== undefined) mat.alphaTest = def.alphaTest;
    if (def.flatShading) mat.flatShading = true;

    applyEnv(mat);
    owned.add(mat);
    stats.materials++;
    return mat;
  }

  function debug() {
    if (!debugMaterial) {
      debugMaterial = new THREE.MeshStandardMaterial({
        name: 'DEBUG_MISSING',
        color: 0xff00ff,
        emissive: 0x550055,
        roughness: 0.35,
        metalness: 0.0,
      });
      owned.add(debugMaterial);
    }
    return debugMaterial;
  }

  function warnOnce(name) {
    if (warned.has(name)) return;
    warned.add(name);
    stats.missing.push(name);
    console.warn(`[materials] unknown material "${name}" — using the magenta debug material`);
  }

  /* ------------------------------------------------------------------ api */

  function get(name) {
    let m = base.get(name);
    if (m) return m;
    m = build(name);
    if (!m) { warnOnce(name); return debug(); }
    base.set(name, m);
    return m;
  }

  /** Clone a texture purely to change its uv transform. Shares the GPU upload. */
  function repeatTexture(tex, rx, ry, ox, oy) {
    if (!tex) return tex;
    if (rx === 1 && ry === 1 && ox === 0 && oy === 0) return tex;
    const key = `${tex.uuid}|${rx}|${ry}|${ox}|${oy}`;
    let c = texClones.get(key);
    if (!c) {
      c = tex.clone();          // shares `source`, so no second GPU upload
      c.repeat.set(rx, ry);
      c.offset.set(ox, oy);
      texClones.set(key, c);
      stats.textureClones++;
    }
    return c;
  }

  const MAP_SLOTS = ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'alphaMap', 'emissiveMap'];

  function applyRepeat(mat, repeat, offset) {
    const rx = Array.isArray(repeat) ? (repeat[0] ?? 1) : (repeat ?? 1);
    const ry = Array.isArray(repeat) ? (repeat[1] ?? rx) : (repeat ?? 1);
    const ox = Array.isArray(offset) ? (offset[0] ?? 0) : 0;
    const oy = Array.isArray(offset) ? (offset[1] ?? 0) : 0;
    for (const slot of MAP_SLOTS) {
      if (mat[slot]) mat[slot] = repeatTexture(mat[slot], rx, ry, ox, oy);
    }
  }

  /**
   * A cached clone with overrides. Same (name, opts) always returns the same
   * instance so variants still batch.
   */
  function variant(name, opts = {}) {
    const key = variantKey(name, opts);
    const hit = variants.get(key);
    if (hit) return hit;

    const src = get(name);
    const m = src.clone();
    m.name = `${name}|${key.slice(name.length + 1)}`;

    if (opts.color !== undefined) m.color = new THREE.Color(opts.color);
    if (opts.emissive !== undefined) m.emissive = new THREE.Color(opts.emissive);
    if (opts.emissiveIntensity !== undefined) m.emissiveIntensity = opts.emissiveIntensity;
    if (opts.roughness !== undefined) m.roughness = opts.roughness;
    if (opts.metalness !== undefined) m.metalness = opts.metalness;
    if (opts.aoMapIntensity !== undefined) m.aoMapIntensity = opts.aoMapIntensity;
    if (opts.normalScale !== undefined) {
      m.normalScale = new THREE.Vector2(opts.normalScale, opts.normalScale);
    }
    if (opts.vertexColors !== undefined) m.vertexColors = !!opts.vertexColors;
    if (opts.flatShading !== undefined) m.flatShading = !!opts.flatShading;
    if (opts.side !== undefined) {
      m.side = typeof opts.side === 'string' ? (SIDES[opts.side] ?? THREE.FrontSide) : opts.side;
      if (m.shadowSide !== null) m.shadowSide = m.side;
    }
    if (opts.transparent !== undefined) m.transparent = !!opts.transparent;
    if (opts.opacity !== undefined) m.opacity = opts.opacity;
    if (opts.alphaTest !== undefined) m.alphaTest = opts.alphaTest;
    if (opts.depthWrite !== undefined) m.depthWrite = !!opts.depthWrite;
    if (opts.depthTest !== undefined) m.depthTest = !!opts.depthTest;
    if (opts.toneMapped !== undefined) m.toneMapped = !!opts.toneMapped;
    if (opts.wireframe !== undefined) m.wireframe = !!opts.wireframe;
    if (opts.envMapIntensity !== undefined) {
      m.userData.envScale = opts.envMapIntensity;
    }
    if (opts.polygonOffset !== undefined) {
      m.polygonOffset = !!opts.polygonOffset;
      m.polygonOffsetFactor = opts.polygonOffsetFactor ?? -1;
      m.polygonOffsetUnits = opts.polygonOffsetUnits ?? -1;
    }
    if (opts.repeat !== undefined || opts.offset !== undefined) {
      applyRepeat(m, opts.repeat ?? 1, opts.offset);
    }

    applyEnv(m);
    m.needsUpdate = true;      // one compile, then it is as cheap as the base
    variants.set(key, m);
    owned.add(m);
    stats.variants++;
    return m;
  }

  /* --------------------------------------------------------------- env map */

  function applyEnv(mat) {
    if (!mat || !mat.isMeshStandardMaterial) return;
    const scale = mat.userData.envScale ?? 1;
    const intensity = envIntensity * scale;
    if (mat.envMap !== envMap) {
      mat.envMap = envMap;
      mat.needsUpdate = true;    // envMap presence changes the program
    }
    if (mat.envMapIntensity !== intensity) mat.envMapIntensity = intensity;
  }

  /**
   * @param {THREE.Texture} map    prefiltered (PMREM) environment
   * @param {number} intensity     global level; per-material `env` scales it
   */
  function setEnvironment(map, intensity = 1) {
    envMap = map || null;
    envIntensity = intensity;
    stats.env = !!map;
    for (const m of owned) applyEnv(m);
  }

  /* ------------------------------------------------------------- lifecycle */

  function dispose() {
    for (const t of texClones.values()) t.dispose();
    texClones.clear();
    for (const m of owned) m.dispose();
    owned.clear();
    base.clear();
    variants.clear();
    debugMaterial = null;
  }

  /* --------------------------------------------------------------- warm up */

  // Build everything up front: it is cheap (a material is not a shader until
  // something renders with it) and it means a missing definition is reported
  // at load rather than the first time a wall happens to face the camera.
  for (const name of MATERIAL_KEYS) get(name);
  const undefinedKeys = MATERIAL_KEYS.filter((k) => !DEFS[k]);
  if (undefinedKeys.length) {
    console.warn('[materials] contract keys with no definition:', undefinedKeys);
  }
  const extraKeys = Object.keys(DEFS).filter((k) => !MATERIAL_KEYS.includes(k));
  if (extraKeys.length) {
    console.info('[materials] extra (non-contract) materials available:', extraKeys);
  }

  return {
    get,
    variant,
    has: (name) => !!DEFS[name],
    keys: () => Object.keys(DEFS),
    setEnvironment,
    /** Metres covered by one uv tile of this material's texture set. */
    worldScale: (name) => get(name).userData.worldScale || 1,
    /** Atlas layout {cols, rows} for the cut-out materials, else null. */
    atlas: (name) => get(name).userData.atlas || null,
    dispose,
    stats,
    renderer,
  };
}
