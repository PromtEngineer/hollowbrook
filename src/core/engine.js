/**
 * Renderer / scene / camera ownership.
 *
 * Note on tone mapping: `renderer.toneMapping` is set to ACES here, but three
 * deliberately skips tone mapping in material shaders whenever it renders into
 * a render target (verified in WebGLPrograms). Because everything goes through
 * the EffectComposer, the actual tone map happens once, in OutputPass. Bloom
 * therefore operates on genuinely HDR values. Do not add a second tone map.
 */

import * as THREE from 'three';
import { CAMERA, RENDER } from '../config.js';

export function createEngine(canvas, quality) {
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,      // handled by SMAA/FXAA in post
    alpha: false,
    stencil: false,
    depth: true,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: false,
  });

  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, quality.pixelRatioCap));
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = RENDER.toneMappingExposure;
  renderer.shadowMap.enabled = true;
  // PCFSoftShadowMap is deprecated in three 0.185 — WebGLShadowMap warns and
  // silently swaps it for PCFShadowMap, so asking for it bought hard shadows and
  // a console warning. PCF is now the soft path: hardware sampler2DShadow
  // comparison plus a 5-sample Vogel disk rotated per pixel by interleaved
  // gradient noise, about 20 filtered taps. Softness comes from
  // light.shadow.radius, which main.js sets from quality.softShadows.
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.shadowMap.autoUpdate = true;
  renderer.info.autoReset = false;

  const scene = new THREE.Scene();
  scene.matrixWorldAutoUpdate = true;

  const camera = new THREE.PerspectiveCamera(
    CAMERA.fov,
    window.innerWidth / window.innerHeight,
    CAMERA.near,
    CAMERA.far,
  );
  camera.position.set(0, 1.7, 34);
  camera.rotation.order = 'YXZ';

  const maxAnisotropy = renderer.capabilities.getMaxAnisotropy();

  const resizeListeners = [];
  let resizeRaf = 0;
  function onResize() {
    cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(() => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
      for (const fn of resizeListeners) fn(w, h);
    });
  }
  window.addEventListener('resize', onResize);

  // A lost context is otherwise a silent black screen.
  canvas.addEventListener('webglcontextlost', (e) => {
    e.preventDefault();
    console.error('[engine] WebGL context lost');
    document.getElementById('load-error')?.classList.remove('hidden');
  });

  return {
    renderer,
    scene,
    camera,
    maxAnisotropy: Math.min(maxAnisotropy, quality.anisotropy),
    onResize(fn) { resizeListeners.push(fn); },
    setPixelRatio(r) {
      const target = Math.min(window.devicePixelRatio || 1, r);
      if (Math.abs(renderer.getPixelRatio() - target) < 0.01) return false;
      renderer.setPixelRatio(target);
      renderer.setSize(window.innerWidth, window.innerHeight, false);
      for (const fn of resizeListeners) fn(window.innerWidth, window.innerHeight);
      return true;
    },
    dispose() {
      window.removeEventListener('resize', onResize);
      renderer.dispose();
    },
  };
}
