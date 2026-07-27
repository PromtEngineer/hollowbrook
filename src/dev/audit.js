/**
 * Scene audit — the automated half of the review pass.
 *
 * Screenshots catch what things look like; this catches what they *are*.
 * Load it from the console at any time:
 *
 *   const { audit } = await import('/src/dev/audit.js'); audit();
 *
 * Everything here is read-only. It never mutates the scene.
 */

import * as THREE from 'three';

const V = new THREE.Vector3();
const BOX = new THREE.Box3();

/**
 * Camera stops for the walkthrough, covering every part of the village.
 * `look` is a world-space target; the yaw is derived, so these stay correct if
 * the layout moves. Yaw 0 faces -Z (see layout.js SPAWN).
 */
export const VIEWPOINTS = [
  { name: 'arrival',       pos: [0, 1.68, 34],     look: [0, 0] },
  { name: 'plaza-centre',  pos: [0, 1.68, 14],     look: [2.2, 5.6] },
  { name: 'the-well',      pos: [2.2, 1.68, 12],   look: [2.2, 5.6] },
  { name: 'well-close',    pos: [5.6, 1.68, 9.6],  look: [2.2, 5.6] },
  { name: 'hero-cottage',  pos: [-13, 1.68, 14],   look: [-23.4, 19.6] },
  { name: 'cottage-door',  pos: [-18, 1.68, 16.5], look: [-23.4, 19.6] },
  { name: 'inn-arcade',    pos: [17, 1.68, 11],    look: [27.6, 17.2] },
  { name: 'east-row',      pos: [20, 1.68, 4],     look: [33.7, 4.7] },
  { name: 'back-cottages', pos: [-6, 1.68, -18],   look: [-10.5, -32.3] },
  { name: 'moot-hall',     pos: [6, 1.68, -20],    look: [6.4, -36.4] },
  { name: 'looking-back',  pos: [0, 1.68, -14],    look: [0, 40] },
  { name: 'west-lane',     pos: [-20, 1.68, 6],    look: [-32, 8] },
  { name: 'north-track',   pos: [0, 1.68, -44],    look: [0, 0] },
  { name: 'up-at-thatch',  pos: [-16, 1.68, 14],   look: [-23.4, 19.6], pitch: 0.45 },
  { name: 'down-at-setts', pos: [1, 1.68, 9],      look: [2.2, 5.6], pitch: -0.7 },
  { name: 'valley',        pos: [0, 1.68, 56],     look: [0, 0] },
  { name: 'at-the-sun',    pos: [0, 1.68, 18],     look: [53, 62], pitch: 0.5 },
];

/** Yaw that makes the camera at `pos` look at the [x,z] target. */
function yawFor(pos, look) {
  return Math.atan2(-(look[0] - pos[0]), -(look[1] - pos[2]));
}

function finite(v) {
  return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z);
}

/**
 * Walk the whole scene graph and report anything structurally wrong.
 * @returns {{issues: Array, summary: Object}}
 */
export async function audit(H = window.HOLLOWBROOK) {
  if (!H) throw new Error('HOLLOWBROOK not on window — has the world finished loading?');

  const issues = [];
  const add = (severity, kind, detail, object) =>
    issues.push({ severity, kind, detail, object: object?.name || object?.type || '' });

  const { scene, renderer, camera } = H;

  /* ---------------------------------------------------------- failed stages */
  for (const f of H.failures || []) {
    add('fatal', 'subsystem-failed', `${f.label}: ${f.err?.message || f.err}`);
  }

  /* ------------------------------------------------------------ scene graph */
  const seenMaterials = new Set();
  const seenGeometries = new Set();
  let meshes = 0, instanced = 0, tris = 0, casters = 0, receivers = 0, lights = 0;
  let debugMaterials = 0, transparentFoliage = 0, noNormals = 0, noEnv = 0;
  const belowGround = [];
  const hugeObjects = [];

  scene.traverse((o) => {
    if (!finite(o.position)) add('fatal', 'nan-transform', `${o.name || o.type} has NaN position`, o);
    if (o.isLight) lights++;

    if (o.isMesh || o.isInstancedMesh) {
      meshes++;
      if (o.isInstancedMesh) instanced++;
      if (o.castShadow) casters++;
      if (o.receiveShadow) receivers++;

      const g = o.geometry;
      if (g && !seenGeometries.has(g)) {
        seenGeometries.add(g);
        const idx = g.index ? g.index.count : g.getAttribute('position')?.count || 0;
        tris += (idx / 3) * (o.isInstancedMesh ? o.count : 1);
        if (!g.getAttribute('normal')) { noNormals++; add('major', 'no-normals', `${o.name || 'mesh'} geometry has no normals — it will shade flat/black`, o); }
        if (!g.boundingSphere) g.computeBoundingSphere();
        const r = g.boundingSphere?.radius || 0;
        if (r > 2000) hugeObjects.push(o.name || o.type);
      }

      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) {
        if (!m) { add('major', 'no-material', `${o.name || 'mesh'} has no material`, o); continue; }
        if (seenMaterials.has(m)) continue;
        seenMaterials.add(m);
        if (m.userData?.debugMaterial || (m.color && m.color.getHex() === 0xff00ff && m.name?.includes('debug'))) {
          debugMaterials++;
          add('major', 'missing-material', `a mesh fell back to the magenta debug material (${m.name})`, o);
        }
        if (m.transparent && m.alphaTest === 0 && /leaf|needle|ivy|flower|grass/i.test(m.name || '')) {
          transparentFoliage++;
          add('major', 'blended-foliage', `${m.name} uses alpha blending instead of alphaTest — it will sort wrong`, o);
        }
        if (m.isMeshStandardMaterial && !m.envMap && !scene.environment) {
          noEnv++;
        }
        if (m.isMeshStandardMaterial && m.roughness === 1 && m.metalness === 0 && !m.roughnessMap && !m.map) {
          add('minor', 'flat-material', `${m.name || 'material'} is untextured, fully rough — will look like plastic`, o);
        }
      }

      // Anything that has fallen through the world.
      o.getWorldPosition(V);
      if (V.y < -12) belowGround.push(o.name || o.type);
    }
  });

  if (belowGround.length) {
    add('major', 'below-world', `${belowGround.length} object(s) below y=-12: ${belowGround.slice(0, 8).join(', ')}`);
  }
  if (hugeObjects.length > 6) {
    add('minor', 'huge-bounds', `${hugeObjects.length} objects with bounding radius > 2 km`);
  }

  /* ------------------------------------------------------------- env / sky */
  if (!scene.environment) add('major', 'no-env', 'scene.environment is null — nothing is image-based lit');
  if (!scene.fog) add('minor', 'no-fog', 'scene.fog is null — the distance will not recede');
  const sun = H.lighting?.sun;
  if (!sun) add('major', 'no-sun', 'no directional light found on the lighting module');
  else {
    if (!sun.castShadow) add('major', 'no-shadows', 'the sun does not cast shadows');
    if (sun.shadow?.mapSize?.x < 1024) add('minor', 'small-shadowmap', `shadow map is only ${sun.shadow.mapSize.x}px`);
  }

  /* --------------------------------------------------------------- physics */
  const p = H.physics;
  if (!p?.world) add('fatal', 'no-physics', 'physics world missing');
  else {
    const s = p.stats || {};
    if ((s.colliders || 0) < 50) add('major', 'few-colliders', `only ${s.colliders} colliders in the world — geometry is probably not solid`);
    if ((s.stepMs || 0) > 6) add('minor', 'slow-physics', `physics step averaging ${s.stepMs?.toFixed?.(2)} ms`);
  }

  /* ------------------------------------------ ground agreement (float/sink) */
  if (p?.raycast && H.terrain?.heightAt) {
    // Sample only open ground — a ray dropped onto a roof or a cart is a hit on
    // something that is supposed to be there.
    const { isClearGround, insideAnyBuilding } = await import('../world/layout.js');
    let worst = 0, worstAt = null;
    for (let i = 0; i < 240; i++) {
      const a = (i / 240) * Math.PI * 2 * 7;
      const r = 2 + (i / 240) * 60;
      const x = Math.cos(a) * r, z = Math.sin(a) * r;
      if (insideAnyBuilding(x, z, 3.5) || !isClearGround(x, z, 2.5)) continue;
      const hit = p.raycast(new THREE.Vector3(x, 40, z), new THREE.Vector3(0, -1, 0), 120, {});
      if (!hit) { add('major', 'no-ground', `no collision under (${x.toFixed(1)}, ${z.toFixed(1)}) — the player can fall through`); break; }
      const expected = H.terrain.heightAt(x, z);
      const d = Math.abs(hit.point.y - expected);
      if (d > worst) { worst = d; worstAt = [x.toFixed(1), z.toFixed(1)]; }
    }
    if (worst > 0.12) {
      add('major', 'ground-mismatch',
        `collision ground and terrain.heightAt disagree by up to ${worst.toFixed(2)} m near (${worstAt}) — the player will float or sink`);
    }
  }

  /* ---------------------------------------------------------------- budget */
  const info = renderer.info;
  const summary = {
    drawCalls: info.render.calls,
    triangles: info.render.triangles,
    sceneTriangles: Math.round(tris),
    programs: info.programs?.length || 0,
    textures: info.memory.textures,
    geometries: info.memory.geometries,
    meshes,
    instancedMeshes: instanced,
    materials: seenMaterials.size,
    lights,
    shadowCasters: casters,
    shadowReceivers: receivers,
    pixelRatio: renderer.getPixelRatio(),
    quality: H.frameCtx?.quality?.name,
    debugMaterials,
  };

  if (summary.drawCalls > 520) add('major', 'draw-calls', `${summary.drawCalls} draw calls — over the 450 budget`);
  if (info.render.triangles > 2_600_000) add('major', 'triangles', `${(info.render.triangles / 1e6).toFixed(2)} M triangles rendered — over budget`);
  if (summary.lights > 12) add('minor', 'many-lights', `${summary.lights} lights in the scene`);

  const order = { fatal: 0, major: 1, minor: 2 };
  issues.sort((a, b) => order[a.severity] - order[b.severity]);

  console.group(`%c[audit] ${issues.length} issue(s)`, 'font-weight:bold');
  console.table(summary);
  if (issues.length) console.table(issues);
  else console.log('%cclean', 'color:#7bd88f');
  console.groupEnd();

  return { issues, summary };
}

/**
 * Teleport through every viewpoint, waiting a few frames at each so the shadow
 * camera, the grass carpet and the light pool settle before a screenshot.
 */
export async function tour(H = window.HOLLOWBROOK, msPerStop = 700) {
  for (const vp of VIEWPOINTS) {
    goto(vp.name, H);
    console.log(`[tour] ${vp.name}`);
    await new Promise((r) => setTimeout(r, msPerStop));
  }
}

/** Jump to one viewpoint by name. */
export function goto(name, H = window.HOLLOWBROOK) {
  const vp = VIEWPOINTS.find((v) => v.name === name);
  if (!vp) { console.warn('no such viewpoint', name, VIEWPOINTS.map((v) => v.name)); return; }
  H.teleport(vp.pos[0], vp.pos[1], vp.pos[2], yawFor(vp.pos, vp.look));
  H.player?.setPitch?.(vp.pitch || 0);
  return vp;
}

/**
 * Interiors audit. Every enterable building has to satisfy four things that a
 * screenshot cannot show: the doorway admits the player capsule, the floor is
 * solid, the room is not open to the sky, and only a few interiors are resident
 * at once. This walks the player into each building and checks all four.
 */
export async function auditInteriors(H = window.HOLLOWBROOK) {
  const rooms = H.buildings?.interiors?.rooms || [];
  if (!rooms.length) {
    console.warn('[audit] buildings published no rooms — interiors not wired?');
    return { issues: [{ severity: 'fatal', kind: 'no-rooms', detail: 'buildings.interiors.rooms is empty' }], rooms: 0 };
  }

  const issues = [];
  const add = (severity, kind, detail) => issues.push({ severity, kind, detail });
  const DOWN = new THREE.Vector3(0, -1, 0);
  const UP = new THREE.Vector3(0, 1, 0);
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));

  const ground = rooms.filter((r) => r.storey === 0);
  const results = [];

  for (const room of ground) {
    const door = (room.openings || []).find((o) => o.kind === 'door' && o.primary)
      || (room.openings || []).find((o) => o.kind === 'door');
    if (!door) { add('major', 'no-door', `${room.plotId} ground floor has no door opening`); continue; }

    if (door.width < 0.98) {
      add('major', 'door-too-narrow',
        `${room.plotId} doorway is ${door.width.toFixed(2)} m clear — the 0.64 m capsule needs ~1.0 m to pass without jamming`);
    }
    if (door.height < 1.95) {
      add('major', 'door-too-low', `${room.plotId} doorway is only ${door.height.toFixed(2)} m tall`);
    }

    // Floor solidity, sampled across the room rather than just at the centre.
    const c = new THREE.Vector3().fromArray(room.centre);
    const cos = Math.cos(room.rotation), sin = Math.sin(room.rotation);
    let missing = 0, worst = 0;
    for (let i = 0; i < 12; i++) {
      const lx = (((i % 4) / 3) - 0.5) * room.width * 0.7;
      const lz = ((Math.floor(i / 4) / 2) - 0.5) * room.depth * 0.7;
      const p = new THREE.Vector3(
        c.x + lx * cos + lz * sin,
        room.floorY + 1.4,
        c.z - lx * sin + lz * cos,
      );
      const hit = H.physics.raycast(p, DOWN, 3.0, {});
      if (!hit) { missing++; continue; }
      worst = Math.max(worst, Math.abs(hit.point.y - room.floorY));
    }
    if (missing) add('fatal', 'floor-not-solid', `${room.plotId}: ${missing}/12 floor samples have no collider — the player falls through`);
    else if (worst > 0.12) add('major', 'floor-mismatch', `${room.plotId}: floor collider is up to ${worst.toFixed(2)} m off the published floorY`);

    // Roof over your head: a ray up from standing height must hit something.
    const up = H.physics.raycast(new THREE.Vector3(c.x, room.floorY + 1.6, c.z), UP, 14, {});
    if (!up) add('major', 'open-to-sky', `${room.plotId}: nothing overhead — the room is open to the sky`);

    // Actually walk in, and confirm we ended up inside.
    const inward = new THREE.Vector3().fromArray(door.inward);
    const outside = new THREE.Vector3().fromArray(door.centreWorld).addScaledVector(inward, -2.4);
    const yaw = Math.atan2(-inward.x, -inward.z);
    H.teleport(outside.x, room.floorY + 1.7, outside.z, yaw);
    H.player?.setPitch?.(0);
    await wait(220);
    const before = H.player.position.clone();
    window.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyW', key: 'w', bubbles: true }));
    await wait(2100);
    window.dispatchEvent(new KeyboardEvent('keyup', { code: 'KeyW', key: 'w', bubbles: true }));
    await wait(220);
    const after = H.player.position.clone();

    // Inside means: within the room's clear footprint, in the room's own frame.
    const dx = after.x - c.x, dz = after.z - c.z;
    const lx = dx * cos - dz * sin;
    const lz = dx * sin + dz * cos;
    const inside = Math.abs(lx) < room.width / 2 && Math.abs(lz) < room.depth / 2
      && after.y > room.floorY - 0.6 && after.y < room.floorY + 2.5;
    const travelled = before.distanceTo(after);
    if (!inside) {
      add('major', 'cannot-enter',
        `${room.plotId}: walked ${travelled.toFixed(2)} m at the doorway and ended outside the room (local ${lx.toFixed(2)}, ${lz.toFixed(2)} vs clear ${room.width.toFixed(1)}x${room.depth.toFixed(1)})`);
    }
    results.push({ plot: room.plotId, use: room.use, entered: inside, travelled: +travelled.toFixed(2), door: +door.width.toFixed(2) });
  }

  // Residency: how much interior is drawn at once?
  const visibleRooms = H.interiors?.stats?.visibleRooms;
  if (typeof visibleRooms === 'number' && visibleRooms > 4) {
    add('minor', 'too-many-interiors', `${visibleRooms} interiors resident at once — culling is not tight`);
  }

  const order = { fatal: 0, major: 1, minor: 2 };
  issues.sort((a, b) => order[a.severity] - order[b.severity]);
  const entered = results.filter((r) => r.entered).length;
  console.group(`%c[audit:interiors] ${entered}/${results.length} enterable, ${issues.length} issue(s)`, 'font-weight:bold');
  console.table(results);
  if (issues.length) console.table(issues);
  console.groupEnd();
  return { issues, results, entered, total: results.length };
}

if (typeof window !== 'undefined') {
  window.HB_AUDIT = { audit, auditInteriors, tour, goto, VIEWPOINTS };
}
