/**
 * ============================================================================
 *  THE CONTRACT
 * ============================================================================
 * Every stream in this project is developed independently. This file is the
 * only thing they share. If a module needs something that isn't described
 * here, the contract is wrong — fix the contract, don't reach across modules.
 *
 * Rules:
 *  1. Geometry modules NEVER talk to the physics engine. They emit
 *     `ColliderSpec` objects; the physics stream consumes them.
 *  2. Geometry modules NEVER create materials. They ask the MaterialLibrary
 *     for one of the canonical names below.
 *  3. Nothing outside `lighting/` creates lights.
 *  4. Nothing outside `post/` touches the EffectComposer.
 *  5. Everything is procedural. No network fetches, no external asset files.
 * ============================================================================
 */

import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/* -------------------------------------------------------------------------- */
/* Collider descriptions                                                       */
/* -------------------------------------------------------------------------- */

/**
 * @typedef {Object} ColliderSpec
 * @property {'box'|'sphere'|'capsule'|'cylinder'|'cone'|'trimesh'|'heightfield'} shape
 * @property {[number,number,number]} [halfExtents]  box
 * @property {number} [radius]                       sphere | capsule | cylinder | cone
 * @property {number} [halfHeight]                   capsule | cylinder | cone
 * @property {Float32Array} [vertices]               trimesh (world- or local-space, see position)
 * @property {Uint32Array}  [indices]                trimesh
 * @property {{nrows:number,ncols:number,heights:Float32Array,scale:[number,number,number]}} [field] heightfield
 * @property {[number,number,number]} position       translation (world space for statics)
 * @property {[number,number,number,number]} [quaternion] x,y,z,w  (defaults to identity)
 * @property {number} [friction=0.85]
 * @property {number} [restitution=0.04]
 * @property {string} [tag]      free-form label, surfaces in raycasts + debug
 * @property {boolean} [sensor=false]
 */

export const DEFAULT_FRICTION = 0.85;
export const DEFAULT_RESTITUTION = 0.04;

/** @returns {ColliderSpec} */
export function boxCollider(position, halfExtents, quaternion, extra = {}) {
  return {
    shape: 'box',
    halfExtents: toArr3(halfExtents),
    position: toArr3(position),
    quaternion: toArr4(quaternion),
    ...extra,
  };
}

/** @returns {ColliderSpec} */
export function sphereCollider(position, radius, extra = {}) {
  return { shape: 'sphere', radius, position: toArr3(position), ...extra };
}

/** @returns {ColliderSpec} Cylinder axis is local +Y. `halfHeight` excludes the caps. */
export function cylinderCollider(position, radius, halfHeight, quaternion, extra = {}) {
  return {
    shape: 'cylinder',
    radius,
    halfHeight,
    position: toArr3(position),
    quaternion: toArr4(quaternion),
    ...extra,
  };
}

/** @returns {ColliderSpec} Capsule axis is local +Y. `halfHeight` excludes the caps. */
export function capsuleCollider(position, radius, halfHeight, quaternion, extra = {}) {
  return {
    shape: 'capsule',
    radius,
    halfHeight,
    position: toArr3(position),
    quaternion: toArr4(quaternion),
    ...extra,
  };
}

/**
 * Build a trimesh collider from an Object3D subtree, baked into world space.
 * Use sparingly — trimeshes are the most expensive static collider. Prefer
 * boxes for anything the player can walk into.
 * @returns {ColliderSpec|null}
 */
export function trimeshColliderFromObject(root, extra = {}) {
  root.updateWorldMatrix(true, true);
  const geos = [];
  root.traverse((o) => {
    if (!o.isMesh || o.userData.noCollide) return;
    const g = o.geometry.index ? o.geometry.toNonIndexed() : o.geometry.clone();
    const stripped = new THREE.BufferGeometry();
    stripped.setAttribute('position', g.getAttribute('position').clone());
    stripped.applyMatrix4(o.matrixWorld);
    geos.push(stripped);
    if (g !== o.geometry) g.dispose();
  });
  if (!geos.length) return null;
  const merged = geos.length === 1 ? geos[0] : mergeGeometries(geos, false);
  if (!merged) return null;
  const pos = merged.getAttribute('position');
  const verts = new Float32Array(pos.array);
  const idx = new Uint32Array(verts.length / 3);
  for (let i = 0; i < idx.length; i++) idx[i] = i;
  geos.forEach((g) => g !== merged && g.dispose());
  merged.dispose();
  return {
    shape: 'trimesh',
    vertices: verts,
    indices: idx,
    position: [0, 0, 0],
    quaternion: [0, 0, 0, 1],
    ...extra,
  };
}

/**
 * Convenience: an axis-aligned box collider that exactly wraps a mesh's world
 * bounding box. Cheap and usually the right answer for props.
 * @returns {ColliderSpec}
 */
export function boxColliderFromObject(object, extra = {}) {
  object.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3()).multiplyScalar(0.5);
  const center = box.getCenter(new THREE.Vector3());
  return boxCollider(center, size, null, extra);
}

/* -------------------------------------------------------------------------- */
/* Interactables                                                               */
/* -------------------------------------------------------------------------- */

/**
 * A thing with physics. Geometry streams emit these; the physics stream turns
 * them into rigid bodies and writes transforms back onto `object3D` each frame.
 *
 * `object3D.position/quaternion` at emit time IS the spawn transform. The
 * collider is expressed in the object's LOCAL space (usually centred on 0,0,0).
 *
 * @typedef {Object} InteractableSpec
 * @property {THREE.Object3D} object3D
 * @property {'dynamic'|'kinematic'|'fixed'} body
 * @property {ColliderSpec} collider          local-space
 * @property {number} [mass=8]
 * @property {number} [linearDamping=0.12]
 * @property {number} [angularDamping=0.35]
 * @property {boolean} [grabbable=false]      player can pick it up with E
 * @property {string}  [prompt]               HUD text when looked at
 * @property {JointSpec} [joint]              attaches to the world
 * @property {boolean} [ccd=false]
 * @property {string} [tag]
 */

/**
 * @typedef {Object} JointSpec
 * @property {'revolute'|'spherical'|'fixed'|'rope'|'spring'} type
 * @property {[number,number,number]} anchorWorld   world-space pivot
 * @property {[number,number,number]} [axis]        revolute hinge axis
 * @property {number} [length]                      rope
 * @property {number} [stiffness]                   spring
 * @property {number} [damping]                     spring
 */

/* -------------------------------------------------------------------------- */
/* Interiors                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * A building's shell is generated by `world/buildings.js`, but what goes INSIDE
 * it is generated by `world/interiors.js` (structure) and `world/furnishings.js`
 * (contents). Those two modules must never guess at where a wall is — the shell
 * publishes its own interior as `RoomSpec`s, and they build against that.
 *
 * Local space: +X is the building's long axis, +Z is its front. The origin of a
 * room is the centre of its finished floor.
 *
 * @typedef {Object} RoomSpec
 * @property {string} plotId              matches a PLOTS entry in layout.js
 * @property {number} storey              0 = ground floor
 * @property {number} floorY              world Y of the finished floor
 * @property {number} ceilingY            world Y of the underside of the joists
 * @property {number} width               interior clear width, along local X
 * @property {number} depth               interior clear depth, along local Z
 * @property {[number,number,number]} centre   world centre of the floor
 * @property {number} rotation            the plot's Y rotation
 * @property {number} wallThickness
 * @property {OpeningSpec[]} openings     every door and window in this room
 * @property {boolean} openToRoof         true when the ceiling is the roof underside
 * @property {number} [ridgeY]            world Y of the ridge, if openToRoof
 * @property {string} use                 see ROOM_USES
 */

/**
 * A hole in a wall. `inward` points from the opening into the room, so a module
 * can place a threshold, a reveal or a shaft of light without redoing the
 * building's trigonometry.
 *
 * @typedef {Object} OpeningSpec
 * @property {'door'|'window'} kind
 * @property {'front'|'back'|'left'|'right'} wall
 * @property {number} lx                  centre along that wall, local metres
 * @property {number} sillY               world Y of the sill (floor level for a door)
 * @property {number} width
 * @property {number} height
 * @property {[number,number,number]} centreWorld
 * @property {[number,number,number]} inward   unit vector, into the room
 * @property {boolean} [primary]          the main entrance
 */

/** What a room is for. Drives which furniture set `furnishings.js` places. */
export const ROOM_USES = [
  'hall',      // a plain living room with a hearth
  'kitchen',   // range, dresser, table, hanging herbs
  'shop',      // counter, shelves, scales, stock
  'taproom',   // the inn: bar, benches, casks, big hearth
  'bedroom',   // upper storey: bed, chest, washstand
  'store',     // sacks, crates, barrels — an attic or back room
  'stable',    // stalls, straw, tack
  'bakery',    // brick oven, peels, dough trough, flour
  'workshop',  // bench, tools, part-made goods
];

/**
 * What `buildings.js` adds to its WorldChunk so interiors can be built.
 * @typedef {Object} InteriorHandoff
 * @property {RoomSpec[]} rooms
 * @property {(plotId:string)=>RoomSpec[]} roomsFor
 */

/* -------------------------------------------------------------------------- */
/* Canonical texture + material names                                          */
/* -------------------------------------------------------------------------- */

/**
 * Every procedural texture set the bakery must produce. A "set" is
 * `{ map, normalMap, roughnessMap, aoMap? }` — all THREE.Texture, sRGB on the
 * albedo, linear on the rest.
 */
export const TEXTURE_KEYS = [
  'plaster',      // lime-washed wattle & daub, warm cream, slightly dirty
  'plasterWorn',  // same but grubbier at the base
  'timber',       // dark oak beam, strong grain, adze marks
  'thatch',       // straw roof — combed reed, directional
  'cobble',       // plaza setts, rounded, mortar gaps, wet-ish
  'cobbleWorn',   // smoother, lighter, foot-polished centre of the plaza
  'brick',        // red chimney brick with pale mortar
  'stone',        // grey ashlar / rubble wall for the well and steps
  'woodPlank',    // door and shutter boards, vertical grain
  'woodBeam',     // structural post, lighter than `timber`
  'roofTile',     // clay pantile (a couple of outbuildings)
  'terracotta',   // flower pots
  'soil',         // potting soil, path dirt
  'grass',        // meadow ground cover
  'iron',         // dark wrought iron for hinges, brackets, lantern frames
  'copper',       // patinated teal-green door and roof flashing
  'leaf',         // broadleaf atlas (alpha)
  'needle',       // conifer sprig atlas (alpha)
  'flower',       // mixed blossom atlas (alpha)
  'ivy',          // ivy leaf atlas (alpha)
  'grassBlade',   // grass tuft atlas (alpha)
];

/**
 * Canonical material names. `materials.get(name)` must return one of these; an
 * unknown name returns a loud magenta debug material and logs once, so a typo
 * shows up in the world instead of throwing.
 */
export const MATERIAL_KEYS = [
  // walls & structure
  'plaster', 'plasterWarm', 'plasterWorn',
  'timber', 'timberDark', 'woodBeam',
  'stone', 'stoneTrim', 'brick',
  // roofs
  'thatch', 'thatchRidge', 'roofTile',
  // joinery
  'woodDoor', 'woodDoorTeal', 'woodPlank', 'woodDark',
  'iron', 'ironDark', 'copper', 'brass',
  // glazing
  'glassWindow', 'glassLantern', 'lanternEmissive', 'windowGlow',
  // ground
  'cobble', 'cobbleWorn', 'soil', 'grass', 'dirtPath',
  // dressing
  'terracotta', 'terracottaDark', 'fabricAwning', 'rope', 'candleWax',
  // interiors
  'floorBoard', 'flagstone', 'lathPlaster', 'ceilingBeam', 'hearthStone',
  'soot', 'linen', 'sackcloth', 'strawLitter', 'pewter',
  // vegetation
  'leaf', 'leafDark', 'needle', 'ivy', 'flowerRed', 'flowerPink',
  'flowerYellow', 'flowerWhite', 'grassBlade', 'bark',
  // scenery
  'hillGrass', 'hillFar', 'mountain',
];

/* -------------------------------------------------------------------------- */
/* Library interfaces                                                          */
/* -------------------------------------------------------------------------- */

/**
 * @typedef {Object} TextureSet
 * @property {THREE.Texture} map            albedo, SRGBColorSpace
 * @property {THREE.Texture} normalMap      linear
 * @property {THREE.Texture} roughnessMap   linear, roughness in .g (three reads .g)
 * @property {THREE.Texture} [aoMap]        linear, needs uv2/uv1 on the geometry
 * @property {THREE.Texture} [alphaMap]     for the cut-out foliage atlases
 * @property {number} [worldScale]          metres covered by one uv tile
 */

/**
 * @typedef {Object} TextureLibrary
 * @property {(name:string)=>TextureSet} get
 * @property {(name:string)=>boolean} has
 * @property {()=>void} dispose
 * @property {Object} stats
 */

/**
 * @typedef {Object} MaterialLibrary
 * @property {(name:string)=>THREE.Material} get
 *   Canonical, shared instance. Never mutate what you get back.
 * @property {(name:string, opts:Object)=>THREE.Material} variant
 *   A cached clone with overrides applied — the only sanctioned way to get
 *   `vertexColors`, `side: DoubleSide`, `transparent`, a tint, or a different
 *   uv repeat. Repeated calls with the same (name, opts) return the same
 *   instance, so variants still batch.
 * @property {(envMap:THREE.Texture, intensity:number)=>void} setEnvironment
 * @property {(dt:number, ctx:Object)=>void} [update]
 * @property {()=>void} dispose
 * @property {Object} stats
 */

/**
 * Build a stable cache key for `materials.variant`.
 */
export function variantKey(name, opts) {
  const keys = Object.keys(opts || {}).sort();
  let s = name;
  for (const k of keys) {
    const v = opts[k];
    s += `|${k}=${v && v.isColor ? v.getHexString() : String(v)}`;
  }
  return s;
}

/* -------------------------------------------------------------------------- */
/* Module return shapes                                                        */
/* -------------------------------------------------------------------------- */

/**
 * @typedef {Object} WorldChunk  What every geometry module returns.
 * @property {THREE.Group} group             added to the scene by main.js
 * @property {ColliderSpec[]} colliders      static world collision
 * @property {InteractableSpec[]} [interactables]
 * @property {Array<{position:[number,number,number],color?:number,intensity?:number,kind:string}>} [lightAnchors]
 * @property {(dt:number, ctx:FrameContext)=>void} [update]
 * @property {()=>void} [dispose]
 * @property {Object} [stats]                free-form, shown in the perf HUD
 */

/**
 * @typedef {Object} FrameContext
 * @property {number} elapsed
 * @property {THREE.PerspectiveCamera} camera
 * @property {THREE.Vector3} playerPosition
 * @property {Object} quality   the active preset
 */

/* -------------------------------------------------------------------------- */
/* Small shared helpers                                                        */
/* -------------------------------------------------------------------------- */

export function toArr3(v) {
  if (!v) return [0, 0, 0];
  if (Array.isArray(v)) return [v[0] || 0, v[1] || 0, v[2] || 0];
  return [v.x || 0, v.y || 0, v.z || 0];
}

export function toArr4(q) {
  if (!q) return [0, 0, 0, 1];
  if (Array.isArray(q)) return [q[0] || 0, q[1] || 0, q[2] || 0, q[3] === undefined ? 1 : q[3]];
  return [q.x || 0, q.y || 0, q.z || 0, q.w === undefined ? 1 : q.w];
}

/** Quaternion array for a rotation about +Y. */
export function quatY(radians) {
  const h = radians * 0.5;
  return [0, Math.sin(h), 0, Math.cos(h)];
}

/**
 * Mark an Object3D subtree so raycasts and the interaction system can see it.
 */
export function tagInteractive(object, prompt, kind = 'prop') {
  object.userData.interactive = true;
  object.userData.prompt = prompt;
  object.userData.kind = kind;
  object.traverse((o) => { o.userData.interactiveRoot = object; });
  return object;
}

/** Enable shadows consistently. Big flat ground planes should not cast. */
export function setShadow(root, cast = true, receive = true) {
  root.traverse((o) => {
    if (o.isMesh) { o.castShadow = cast; o.receiveShadow = receive; }
  });
  return root;
}

/** Cheap disposal of a whole subtree (geometries only — materials are shared). */
export function disposeGroup(group) {
  group.traverse((o) => {
    if (o.isMesh || o.isInstancedMesh || o.isPoints || o.isLine) {
      o.geometry?.dispose?.();
    }
  });
  group.clear?.();
}
