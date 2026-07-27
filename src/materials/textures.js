/**
 * ============================================================================
 *  PROCEDURAL TEXTURE BAKERY
 * ============================================================================
 * Every surface in Hollowbrook is synthesised on the GPU at load time. A tiny
 * fullscreen-quad pass evaluates a GLSL height/albedo field into an RGBA8
 * render target; the result is read back once and re-uploaded as a DataTexture.
 *
 * Why read back at all? Because three stores the GL texture of a *render
 * target* on the render target itself, not on the texture's `Source`
 * (WebGLTextures.setTexture2D skips `uploadTexture` when
 * `isRenderTargetTexture === true` and binds `__webglTexture` straight from the
 * texture's own properties). Cloning such a texture — which is exactly what
 * `materials.variant({ repeat })` has to do — yields a clone with no
 * `__webglTexture` and renders black. A DataTexture keeps its GL object on the
 * shared `Source`, so clones cost one JS object and zero GPU memory. The
 * readback is the price of a safe, cloneable texture library.
 *
 * Three passes per material, all from ONE shader program (a `uPass` uniform
 * branches at the end) so we pay ~15 shader compiles instead of ~60:
 *   pass 0  albedo   (sRGB bytes, straight out — no colour conversion)
 *   pass 1  normal   (Sobel over the *analytic* height field, not over the
 *                     8-bit albedo alpha, so normals are not quantised)
 *   pass 2  ORM      (r = AO, g = roughness, b = metalness — glTF packing, so
 *                     one texture feeds aoMap/roughnessMap/metalnessMap)
 *
 * Every noise primitive is domain-wrapped (`mod(cell, period)`), so every
 * pattern tiles seamlessly at whatever period it was authored with.
 * ============================================================================
 */

import * as THREE from 'three';
import { TEXTURE_KEYS } from '../contracts.js';

/* ========================================================================== */
/* GLSL — shared prelude                                                       */
/* ========================================================================== */

const PRELUDE = /* glsl */`
precision highp float;

varying vec2 vUv;

uniform int   uPass;    // 0 albedo, 1 normal, 2 orm
uniform float uTexel;   // 1 / size of the target being written
uniform float uNormal;  // relief(m) * size / worldScale  -> physical slope
uniform vec4  uA;
uniform vec4  uB;
uniform vec4  uC;
uniform vec4  uD;

const float PI  = 3.141592653589793;
const float TAU = 6.283185307179586;

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.xx + p3.yz) * p3.zy);
}
vec3 hash32(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yzz) * p3.zyx);
}

/* Tileable value noise. "per" is the lattice period in cells on each axis. */
float vnoise(vec2 x, vec2 per) {
  vec2 i = floor(x), f = fract(x);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash12(mod(i, per));
  float b = hash12(mod(i + vec2(1.0, 0.0), per));
  float c = hash12(mod(i + vec2(0.0, 1.0), per));
  float d = hash12(mod(i + vec2(1.0, 1.0), per));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm2(vec2 p, vec2 per) {
  float s = vnoise(p, per) * 0.5 + vnoise(p * 2.0, per * 2.0) * 0.25;
  return s / 0.75;
}
float fbm(vec2 p, vec2 per) {          // 4 octaves
  float s = 0.0, a = 0.5, n = 0.0;
  vec2 q = p, pe = per;
  for (int i = 0; i < 4; i++) {
    s += a * vnoise(q, pe);
    n += a; q *= 2.0; pe *= 2.0; a *= 0.5;
  }
  return s / n;
}
float fbm6(vec2 p, vec2 per) {         // 6 octaves, for broad soft fields
  float s = 0.0, a = 0.5, n = 0.0;
  vec2 q = p, pe = per;
  for (int i = 0; i < 6; i++) {
    s += a * vnoise(q, pe);
    n += a; q *= 2.0; pe *= 2.0; a *= 0.52;
  }
  return s / n;
}
/* Ridged noise — creases rather than blobs. */
float ridge(vec2 p, vec2 per) {
  float s = 0.0, a = 0.5, n = 0.0;
  vec2 q = p, pe = per;
  for (int i = 0; i < 3; i++) {
    s += a * (1.0 - abs(vnoise(q, pe) * 2.0 - 1.0));
    n += a; q *= 2.0; pe *= 2.0; a *= 0.5;
  }
  return s / n;
}

/*
 * Tileable Worley. Returns:
 *   .x  distance to the nearest feature point
 *   .y  a random id in [0,1) for the owning cell
 *   .z  distance to the cell *border* (Inigo Quilez's second pass) — this is
 *       what makes mortar joints of even width instead of round blobs
 *   .w  a second independent id for the owning cell
 */
vec4 voro(vec2 x, vec2 per, float jitter) {
  vec2 n = floor(x), f = fract(x);
  vec2 mr = vec2(0.0), mg = vec2(0.0);
  float md = 8.0, id = 0.0, id2 = 0.0;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 g = vec2(float(i), float(j));
      vec2 c = mod(n + g, per);
      vec2 o = hash22(c);
      vec2 r = g + 0.5 + (o - 0.5) * jitter - f;
      float d = dot(r, r);
      if (d < md) {
        md = d; mr = r; mg = g;
        id = hash12(c + 11.7);
        id2 = hash12(c + 71.3);
      }
    }
  }
  float ed = 8.0;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 g = mg + vec2(float(i), float(j));
      vec2 c = mod(n + g, per);
      vec2 o = hash22(c);
      vec2 r = g + 0.5 + (o - 0.5) * jitter - f;
      vec2 d = r - mr;
      if (dot(d, d) > 1e-5) ed = min(ed, dot(0.5 * (mr + r), normalize(d)));
    }
  }
  return vec4(sqrt(md), id, ed, id2);
}

vec2 rot2(vec2 p, float a) {
  float c = cos(a), s = sin(a);
  return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}
float sat(float x) { return clamp(x, 0.0, 1.0); }

/* Descending smoothstep. GLSL leaves smoothstep(e0,e1,x) undefined when
   e0 >= e1, so never write it inverted — write istep(hi, lo, x) instead. */
float istep(float hi, float lo, float x) { return 1.0 - smoothstep(lo, hi, x); }
`;

/* Surface template: bodies define hgt(), alb(), orm3(). */
const SURFACE_MAIN = /* glsl */`
float cavity(vec2 uv, float h) {
  float b = 0.0;
  b += hgt(uv + vec2( 0.0070, 0.0));
  b += hgt(uv + vec2(-0.0070, 0.0));
  b += hgt(uv + vec2( 0.0, 0.0070));
  b += hgt(uv + vec2( 0.0,-0.0070));
  b += hgt(uv + vec2( 0.0050, 0.0050));
  b += hgt(uv + vec2(-0.0050,-0.0050));
  b *= 0.1666667;
  float rel = smoothstep(-0.16, 0.10, h - b);
  return clamp(mix(0.30, 1.0, rel) * (0.72 + 0.28 * sat(h)), 0.0, 1.0);
}

void main() {
  if (uPass == 0) {
    float h = hgt(vUv);
    gl_FragColor = vec4(clamp(alb(vUv, h), 0.0, 1.0), 1.0);
  } else if (uPass == 1) {
    float t = uTexel;
    float h00 = hgt(vUv + vec2(-t, -t)), h10 = hgt(vUv + vec2(0.0, -t)), h20 = hgt(vUv + vec2(t, -t));
    float h01 = hgt(vUv + vec2(-t, 0.0)),                                h21 = hgt(vUv + vec2(t, 0.0));
    float h02 = hgt(vUv + vec2(-t,  t)), h12 = hgt(vUv + vec2(0.0,  t)), h22 = hgt(vUv + vec2(t,  t));
    float gx = ((h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02)) * 0.125;
    float gy = ((h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20)) * 0.125;
    vec3 n = normalize(vec3(-gx * uNormal, -gy * uNormal, 1.0));
    gl_FragColor = vec4(n * 0.5 + 0.5, 1.0);
  } else {
    float h = hgt(vUv);
    gl_FragColor = vec4(clamp(orm3(vUv, h, cavity(vUv, h)), 0.0, 1.0), 1.0);
  }
}
`;

/* Atlas template: bodies define hgt(), atlasRGBA(). */
const ATLAS_MAIN = /* glsl */`
void main() {
  if (uPass == 0) {
    gl_FragColor = clamp(atlasRGBA(vUv), 0.0, 1.0);
  } else {
    float t = uTexel;
    float h00 = hgt(vUv + vec2(-t, -t)), h10 = hgt(vUv + vec2(0.0, -t)), h20 = hgt(vUv + vec2(t, -t));
    float h01 = hgt(vUv + vec2(-t, 0.0)),                                h21 = hgt(vUv + vec2(t, 0.0));
    float h02 = hgt(vUv + vec2(-t,  t)), h12 = hgt(vUv + vec2(0.0,  t)), h22 = hgt(vUv + vec2(t,  t));
    float gx = ((h20 + 2.0 * h21 + h22) - (h00 + 2.0 * h01 + h02)) * 0.125;
    float gy = ((h02 + 2.0 * h12 + h22) - (h00 + 2.0 * h10 + h20)) * 0.125;
    vec3 n = normalize(vec3(-gx * uNormal, -gy * uNormal, 1.0));
    gl_FragColor = vec4(n * 0.5 + 0.5, 1.0);
  }
}
`;

const VERTEX = /* glsl */`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

/* ========================================================================== */
/* GLSL — material families                                                    */
/* ========================================================================== */

const FAMILY = {};

/* -------------------------------------------------------------- cobble setts */
/* uA = (cells, wear, damp, jointWidth)   uB = (_, polish, gritty, _) */
FAMILY.cobble = /* glsl */`
vec4 cbCell(vec2 uv) {
  vec2 w = vec2(fbm2(uv * 3.0 + 4.1, vec2(3.0)), fbm2(uv * 3.0 - 2.3, vec2(3.0))) - 0.5;
  return voro(uv * uA.x + w * 0.62, vec2(uA.x), 0.74);
}
float hgt(vec2 uv) {
  vec4 v = cbCell(uv);
  float joint = smoothstep(0.0, uA.w, v.z);
  float dome  = pow(sat(v.z * 2.6), 0.5);
  float proud = (v.y - 0.5) * 0.16 * (1.0 - uA.y * 0.5);
  float h = joint * (0.50 + 0.42 * dome + proud);
  h += (fbm(uv * 44.0, vec2(44.0)) - 0.5) * 0.070 * joint * (1.0 - uA.y * 0.55);
  h += (fbm2(uv * 110.0, vec2(110.0)) - 0.5) * 0.09 * (1.0 - joint) * uB.z;
  h -= (1.0 - joint) * 0.30;                    // joints cut deep, not creased
  return h;
}
vec3 alb(vec2 uv, float h) {
  vec4 v = cbCell(uv);
  float joint = smoothstep(0.0, uA.w, v.z);
  float r = v.y, r2 = v.w;
  float rt = fract(r2 * 7.3);                   // third independent per-cell hash
  /* One STONE TYPE per Voronoi cell — hashed per sett, never per pixel — so the
     plaza reads as mixed salvaged setts rather than one sandy polygon field.
     ALBEDO BUDGET. These are sRGB-encoded values written straight to an RGBA8
     target, so the LINEAR albedo of a value x is about x^2.2. Weathered granite
     setts sit at 0.24-0.34 linear = sRGB 0.50-0.66, which is where this palette
     now lives. Raising the mean to there costs bright-side variation: the whole
     product (palette x per-sett x grain) has to stay under 1.0, and with a mean
     of ~0.59 there is only 1.7x of headroom left. So the variety is carried
     DOWNWARD — the dark ends of each stone type and the darker grain terms —
     and the multipliers keep a low ceiling. Do not scale the palette up without
     pulling the multiplier maxima down by the same factor. */
  vec3 grey  = mix(vec3(0.594, 0.582, 0.567), vec3(0.746, 0.728, 0.688), rt);
  vec3 brown = mix(vec3(0.616, 0.488, 0.372), vec3(0.759, 0.600, 0.428), rt);
  vec3 ochre = mix(vec3(0.690, 0.549, 0.318), vec3(0.770, 0.630, 0.378), rt);
  vec3 slate = vec3(0.475, 0.470, 0.484) * (0.85 + 0.40 * rt);
  vec3 c = grey;
  c = mix(c, brown, step(0.34, r2));
  c = mix(c, ochre, step(0.63, r2));
  c = mix(c, slate, step(0.90, r2));
  c *= 0.90 + 0.34 * r;                                                        // per-sett value
  c *= 0.88 + 0.20 * fbm(uv * 24.0, vec2(24.0));
  c *= 0.93 + 0.08 * fbm2(uv * 90.0, vec2(90.0));
  float polish = smoothstep(0.09, 0.44, v.z) * uB.y;
  c = mix(c, mix(c, vec3(0.762, 0.739, 0.699), 0.40), polish);                  // foot-polished crown
  // The joints stay dark: they are the whole reason the relief reads at all.
  vec3 mortar = vec3(0.205, 0.194, 0.172) * (0.70 + 0.78 * fbm(uv * 80.0, vec2(80.0)));
  c = mix(mortar, c, joint);
  c *= mix(1.0 - 0.34 * uA.z, 1.0, smoothstep(0.0, 0.18, v.z));                 // damp joints
  // No extra "worn = paler" multiplier any more: with the palette at a realistic
  // level, the worn set's thinner joints (uA.w) and stronger polished crown
  // (uB.y) already put it ~0.03 linear above the plaza set, which is the whole
  // difference between a plaza sett and a foot-polished one. The old +10% on top
  // overshot the target band and pushed the road ribbons past the plaza.
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  vec4 v = cbCell(uv);
  float joint = smoothstep(0.0, uA.w, v.z);
  float polish = smoothstep(0.09, 0.44, v.z) * uB.y;
  float rough = mix(0.97, 0.66, joint);
  // Polished-crown roughness FLOOR. 0.30 was authored when the setts measured
  // ~0.10 linear; at 0.28 the same lobe carries 2.8x the energy over a diffuse
  // term that only tripled, and near-camera ground-facing highlights clipped
  // (0.19% of pixels >= 250 in a low close-up). 0.42 keeps the wet-looking sheen
  // on the crowns and puts the peak back inside the tone curve.
  rough = mix(rough, 0.42, polish);
  rough *= 0.90 + 0.22 * fbm2(uv * 26.0, vec2(26.0));
  return vec3(mix(1.0, ao, 0.92), rough, 0.0);
}
`;

/* -------------------------------------------------------------- combed thatch */
/* uA = (straws, courses, weather, clump)   uB = (bleach, endBand, wads, _)
 *
 * Fibres run along V, and V runs DOWN THE PITCH (buildings sweep the roof UV so
 * u = distance along the eave, v = distance down the slope). So every field here
 * varies FAST in u and SLOWLY in v.
 *
 * THREE scales, because one is never enough:
 *   straws (uA.x per uv tile)  the individual reed — carries the raking-light
 *                              relief up close. Must stay under half the normal
 *                              map's Nyquist frequency or the Sobel in
 *                              SURFACE_MAIN straddles a whole period and cancels
 *                              to a flat normal. That is exactly what was wrong.
 *   wads   (uB.z per uv tile)  the bundles the straws are tied in, ~6 cm — the
 *                              scale that survives mip-mapping and still reads
 *                              at 20 m.
 *   clump  (6 per uv tile)     broad thickness variation of the whole coat.
 */
FAMILY.thatch = /* glsl */`
/* Slow sideways drift, so no straw is a ruler-straight line. */
float thWave(vec2 uv) {
  return (fbm2(vec2(uv.x * 3.0, uv.y * 2.0), vec2(3.0, 2.0)) - 0.5) * 0.052
       + (fbm2(vec2(uv.x * 12.0, uv.y * 6.0), vec2(12.0, 6.0)) - 0.5) * 0.017;
}
float thFu(vec2 uv) { return (uv.x + thWave(uv)) * uA.x; }   // straw coordinate
float thBu(vec2 uv) { return (uv.x + thWave(uv)) * uB.z; }   // wad coordinate
float thClump(vec2 uv) { return fbm2(vec2(uv.x * 6.0, uv.y * 2.0), vec2(6.0, 2.0)); }

float hgt(vec2 uv) {
  float ft = fract(thFu(uv)), bt = fract(thBu(uv));
  float fid = mod(floor(thFu(uv)), uA.x);
  float bid = mod(floor(thBu(uv)), uB.z);
  float cv = uv.y * uA.y;
  float cid = mod(floor(cv), uA.y), ct = fract(cv);

  float fr = pow(sin(PI * clamp(ft, 0.002, 0.998)), 0.42);   // one half-round straw
  float br = pow(sin(PI * clamp(bt, 0.002, 0.998)), 0.85);   // the wad it lies in
  float r1 = hash12(vec2(fid, 3.0));
  float b1 = hash12(vec2(bid, 17.0));

  float lap  = smoothstep(0.0, 0.09, ct);                    // crease at the course line
  float lift = smoothstep(0.0, 0.55, ct) * 0.15;             // straw springs off the lap

  // Amplitudes are a real budget: the set's relief maps the whole field to ~7 cm, so a
  // 1 cm straw may own about 0.15 of it. Giving the straws half the field (the
  // obvious thing to write) makes every reed a 3 cm half-pipe and the normal
  // map saturates into noise.
  float h = br * (0.30 + 0.34 * b1)
          + fr * (0.10 + 0.11 * r1) * (0.60 + 0.40 * br);
  h *= 0.70 + 0.60 * thClump(uv) * uA.w;
  h = h * (0.48 + 0.52 * lap) + lift;

  // frayed, broken butts: each straw stops a little short of the course line
  float endJit = hash12(vec2(fid, cid * 1.7 + 5.0));
  h += fr * smoothstep(0.0, 0.26, ct - endJit * 0.28) * 0.13;
  // nodes and kinks along the straw
  h += (fbm2(vec2(uv.x * uA.x * 0.5, uv.y * 80.0), vec2(uA.x * 0.5, 80.0)) - 0.5) * 0.07 * fr;
  return h;
}
vec3 alb(vec2 uv, float h) {
  float ft = fract(thFu(uv)), bt = fract(thBu(uv));
  float fid = mod(floor(thFu(uv)), uA.x);
  float bid = mod(floor(thBu(uv)), uB.z);
  float cv = uv.y * uA.y;
  float ct = fract(cv);
  float r1 = hash12(vec2(fid, 3.0));
  float r2 = hash12(vec2(fid, 91.0));
  float b1 = hash12(vec2(bid, 17.0));

  vec3 gold = vec3(0.788, 0.643, 0.388);   // #c9a463 fresh straw
  vec3 deep = vec3(0.616, 0.478, 0.255);   // #9d7a41 shaded straw
  vec3 pale = vec3(0.862, 0.755, 0.512);   // sun-bleached
  vec3 grey = vec3(0.412, 0.374, 0.306);   // weathered grey-brown

  vec3 c = mix(deep, gold, 0.22 + 0.78 * r1);                       // per-straw tone
  c = mix(c, pale, smoothstep(0.60, 1.0, r2) * (0.30 + 0.50 * uB.x));
  c *= 0.90 + 0.20 * b1;                                            // per-wad tone
  c *= 0.86 + 0.28 * thClump(uv);
  // streaking down the slope: elongated in v, so it never crosses the fibres
  c *= 0.88 + 0.24 * fbm2(vec2(uv.x * 9.0, uv.y * 2.0), vec2(9.0, 2.0));

  float wet = fbm6(vec2(uv.x * 3.0, uv.y * 4.0), vec2(3.0, 4.0));
  c = mix(c, grey, smoothstep(0.56, 0.92, wet) * uA.z);              // weathered patches
  c = mix(c, vec3(0.286, 0.302, 0.222), smoothstep(0.84, 1.0, wet) * uA.z * 0.55); // moss

  // contact darkening in the gaps — the relief itself is in the normal map
  c *= mix(0.60, 1.0, pow(sin(PI * clamp(ft, 0.002, 0.998)), 0.50));
  c *= mix(0.82, 1.0, pow(sin(PI * clamp(bt, 0.002, 0.998)), 0.80));
  c *= mix(0.48, 1.0, smoothstep(0.0, 0.12, ct));                   // shade under the lap
  // the ragged butt ends of the course above catch the sun
  float ends = smoothstep(0.86, 1.0, ct) * uB.y;
  c = mix(c, pale * (0.60 + 0.58 * hash12(vec2(fid, floor(cv) + 0.5))), ends);
  c *= 0.94 + 0.12 * fbm2(vec2(uv.x * uA.x * 0.5, uv.y * 40.0), vec2(uA.x * 0.5, 40.0));
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float ft = fract(thFu(uv)), bt = fract(thBu(uv));
  float ct = fract(uv.y * uA.y);
  float wet = fbm6(vec2(uv.x * 3.0, uv.y * 4.0), vec2(3.0, 4.0));
  float rough = 0.95 - 0.13 * pow(sin(PI * clamp(ft, 0.002, 0.998)), 2.0);
  rough -= 0.05 * pow(sin(PI * clamp(bt, 0.002, 0.998)), 2.0);
  rough = mix(rough, 0.75, smoothstep(0.62, 0.95, wet) * uA.z);      // weathered = smoother
  rough *= 0.97 + 0.06 * fbm2(vec2(uv.x * 20.0, uv.y * 20.0), vec2(20.0));
  float occ = ao * mix(0.50, 1.0, smoothstep(0.0, 0.14, ct));
  return vec3(occ, clamp(rough, 0.74, 0.96), 0.0);
}
`;

/* ------------------------------------------------------- lime-washed plaster */
/* uA = (grime, cracks, trowel, aggregate)  uB = (warmth, patchy, _, _) */
FAMILY.plaster = /* glsl */`
float plCrack(vec2 uv) {
  vec2 w = vec2(fbm2(uv * 4.0, vec2(4.0)), fbm2(uv * 4.0 + 7.3, vec2(4.0))) - 0.5;
  vec4 v = voro(uv * 9.0 + w * 1.6, vec2(9.0), 0.92);
  float thin = 1.0 - smoothstep(0.0, 0.013 + 0.020 * v.w, v.z);
  return thin * smoothstep(0.40, 0.66, fbm(uv * 2.0, vec2(2.0)));
}
float hgt(vec2 uv) {
  float trowel = fbm(vec2(uv.x * 3.0, uv.y * 3.0), vec2(3.0));
  float sweep  = fbm2(vec2(uv.x * 6.0, uv.y * 2.0), vec2(6.0, 2.0));
  float daub   = fbm(uv * 15.0, vec2(15.0));
  float grit   = fbm2(uv * 80.0, vec2(80.0)) - 0.5;
  float pits   = smoothstep(0.70, 0.94, fbm2(uv * 36.0, vec2(36.0)));
  float agg    = smoothstep(0.86, 0.99, hash12(mod(floor(uv * 220.0), vec2(220.0)))) * uA.w;
  float h = 0.5 + (trowel - 0.5) * uA.z + (sweep - 0.5) * uA.z * 0.55
          + (daub - 0.5) * 0.16 + grit * 0.045 - pits * 0.10 + agg * 0.10;
  return h - plCrack(uv) * uA.y * 0.34;
}
vec3 alb(vec2 uv, float h) {
  /* Lime-washed daub is 0.38-0.48 LINEAR, i.e. sRGB 0.66-0.74 — not the 0.82
     this used to be. At 0.55 linear the plaster was the brightest thing in the
     village by a factor of five and it clipped as soon as the ground albedo was
     raised to anything realistic, which is what made an earlier pass milky. */
  vec3 cream = vec3(0.770, 0.722, 0.625);
  vec3 shade = vec3(0.659, 0.611, 0.512);
  float mot = fbm6(uv * 3.0, vec2(3.0));
  vec3 c = mix(shade, cream, smoothstep(0.20, 0.85, mot));
  c = mix(c, cream * vec3(1.02, 0.99, 0.93), uB.x);
  c *= 0.93 + 0.13 * fbm(uv * 22.0, vec2(22.0));
  c *= 0.96 + 0.08 * fbm2(uv * 95.0, vec2(95.0));
  float grimePatch = smoothstep(0.46, 0.86, fbm6(uv * 2.0 + 13.0, vec2(2.0)))
                   * (0.55 + 0.45 * fbm(uv * 9.0, vec2(9.0)));
  c = mix(c, vec3(0.330, 0.310, 0.266), grimePatch * uA.x);
  float streak = smoothstep(0.62, 0.98, fbm2(vec2(uv.x * 26.0, uv.y * 3.0), vec2(26.0, 3.0)));
  c = mix(c, vec3(0.440, 0.414, 0.368), streak * uA.x * 0.55);                   // rain runs
  float crack = plCrack(uv) * uA.y;
  c = mix(c, vec3(0.285, 0.262, 0.226), crack * 0.85);
  float agg = smoothstep(0.90, 1.0, hash12(mod(floor(uv * 220.0), vec2(220.0)))) * uA.w;
  c = mix(c, vec3(0.708, 0.681, 0.620), agg * 0.5);
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float rough = 0.90 + 0.09 * fbm2(uv * 14.0, vec2(14.0));
  rough -= smoothstep(0.55, 0.95, fbm6(uv * 3.0, vec2(3.0))) * 0.10;            // burnished limewash
  rough += plCrack(uv) * uA.y * 0.06;
  return vec3(mix(1.0, ao, 0.75), rough, 0.0);
}
`;

/* ------------------------------------------------------------------- oak wood */
/* uA = (ringFreq, axisSwap, boards, adze)  uB = (tone, checks, nails, weathering)
   uC = (base.rgb, latewood darkness)
   uD = (paintReady, trafficWear, pegHoles, smoked)
 *
 * EXTERIOR ONLY, AND FROZEN. This family bakes `timber`, `woodPlank`,
 * `woodBeam` and `woodPainted`, all four of them calibrated and signed off, so
 * not one character inside the template string below may change — a comment
 * edit is a different shader source and a different program. The interior sets
 * that used to borrow it (floorBoard, ceilingBeam) now have their own family,
 * FAMILY.sawnOak, precisely so that they could be rebuilt without putting the
 * exterior at risk. uD.y (trafficWear), uD.z (pegHoles) and uD.w (smoked) are
 * therefore dead weight now: every remaining set passes 0 for all three, the
 * helpers compile out, and they are left in place only because deleting them
 * would rewrite the string. */
FAMILY.wood = /* glsl */`
vec2 wdG(vec2 uv) { return mix(uv, uv.yx, step(0.5, uA.y)); }

float wdBoard(vec2 g, out float bid, out float gap) {
  if (uA.z < 0.5) { bid = 0.0; gap = 0.0; return 0.0; }
  float bv = g.y * uA.z;
  bid = mod(floor(bv), uA.z);
  float bt = fract(bv);
  float e = min(bt, 1.0 - bt);
  gap = 1.0 - smoothstep(0.0, 0.030, e);
  return bt;
}

float wdGrain(vec2 g, float bid, out float band, out float fleck) {
  float phase = hash12(vec2(bid, 5.0)) * 6.0;
  float warp = fbm(vec2(g.x * 2.0, g.y * 7.0), vec2(2.0, 7.0));
  float s = g.y * uA.x + phase + warp * 2.8
          + (fbm2(vec2(g.x * 5.0, g.y * 3.0), vec2(5.0, 3.0)) - 0.5) * 1.4;
  float r = fract(s);
  band = smoothstep(0.02, 0.26, r) * (1.0 - smoothstep(0.58, 0.90, r));
  fleck = smoothstep(0.60, 0.88, fbm2(vec2(g.x * 80.0, g.y * 9.0), vec2(80.0, 9.0)));
  return r;
}

float wdChecks(vec2 g) {
  return smoothstep(0.78, 0.97, fbm2(vec2(g.x * 3.0, g.y * 46.0), vec2(3.0, 46.0))) * uB.y;
}
float wdNail(vec2 g, float bt) {
  if (uB.z < 0.5) return 0.0;
  vec2 q = vec2(fract(g.x * 3.0) - 0.5, bt - 0.5);
  float d = length(vec2(q.x / 3.0, q.y / max(uA.z, 1.0)));
  return istep(0.016, 0.008, d);
}

/* Traffic polish. Elongated ALONG the grain (g.x) so on a floor it reads as the
   line feet have worn down the room, not as blotches. */
float wdWear(vec2 g) {
  if (uD.y < 0.001) return 0.0;
  float f = fbm2(vec2(g.x * 1.0, g.y * 3.0), vec2(1.0, 3.0));
  return sat(smoothstep(0.40, 0.86, f)) * uD.y;
}
/* Old peg holes in a joist: a third of a sparse lattice, ~2.5 cm across. */
float wdPeg(vec2 g) {
  if (uD.z < 0.001) return 0.0;
  vec2 c = mod(floor(g * 3.0), 3.0);
  vec2 f = fract(g * 3.0) - 0.5;
  vec3 h3 = hash32(c + 4.4);
  if (h3.z > 0.34) return 0.0;
  float d = length((f - (h3.xy - 0.5) * 0.5) / 3.0);
  return istep(0.020, 0.011, d) * uD.z;
}
/* Smoke blackening: a joist over an open hearth is sooty on the underside. */
float wdSmoke(vec2 g) {
  if (uD.w < 0.001) return 0.0;
  // period must be an INTEGER or the lattice does not wrap: mod(cell, 2.5) put a
  // hard seam down every tile edge (measured: 6x the interior gradient).
  return smoothstep(0.46, 0.88, fbm6(g * 3.0 + 21.0, vec2(3.0))) * uD.w;
}

float hgt(vec2 uv) {
  vec2 g = wdG(uv);
  float bid, gap;
  float bt = wdBoard(g, bid, gap);
  float band, fleck;
  wdGrain(g, bid, band, fleck);
  float adze = uA.w * (0.5 - 0.5 * cos(TAU * fract(g.x * 8.0
             + (fbm2(vec2(g.x * 8.0, g.y * 2.0), vec2(8.0, 2.0)) - 0.5) * 0.7)));
  float h = 0.55 + band * 0.10 + fleck * 0.03 + adze * 0.10;
  h += (fbm2(vec2(g.x * 40.0, g.y * 120.0), vec2(40.0, 120.0)) - 0.5) * 0.05;
  h -= wdChecks(g) * 0.35;
  h -= gap * 0.55;
  h += hash12(vec2(bid, 17.0)) * 0.05 * step(0.5, uA.z);
  float nail = wdNail(g, bt);
  h = mix(h, 0.72, nail);
  h -= wdWear(g) * 0.045;                    // feet sand the boards down
  h = mix(h, 0.26, wdPeg(g));
  return h;
}
vec3 alb(vec2 uv, float h) {
  vec2 g = wdG(uv);
  float bid, gap;
  float bt = wdBoard(g, bid, gap);
  float band, fleck;
  wdGrain(g, bid, band, fleck);
  vec3 base = uC.rgb * uB.x;
  vec3 late = base * (1.0 - uC.a);
  vec3 c = mix(late, base * 1.14, band);
  c = mix(c, base * 1.42 + 0.035, fleck * 0.55);                                // medullary rays
  c *= 0.88 + 0.26 * fbm(vec2(g.x * 3.0, g.y * 10.0), vec2(3.0, 10.0));
  c *= 0.94 + 0.12 * fbm2(vec2(g.x * 30.0, g.y * 90.0), vec2(30.0, 90.0));
  c *= mix(0.82, 1.12, hash12(vec2(bid, 23.0)) * step(0.5, uA.z) + (1.0 - step(0.5, uA.z)) * 0.5);
  float weather = smoothstep(0.52, 0.92, fbm6(g * 3.0, vec2(3.0)));
  c = mix(c, mix(c, vec3(0.400, 0.372, 0.330), 0.55), weather * uB.w);          // silvered greying
  c = mix(c, base * 0.24, wdChecks(g));
  c = mix(c, base * 0.16, gap);
  float nail = wdNail(g, bt);
  c = mix(c, vec3(0.115, 0.105, 0.098), nail);
  // Worn boards go PALER, not darker: the traffic line is bare sanded oak while
  // the rest keeps its old wax and dirt.
  c = mix(c, mix(c, vec3(0.596, 0.514, 0.404), 0.62), wdWear(g));
  c = mix(c, vec3(0.132, 0.112, 0.094), wdPeg(g) * 0.9);
  c = mix(c, c * 0.46, wdSmoke(g));
  if (uD.x > 0.5) {
    // Paint-ready: collapse to a mid-grey luminance so material.color can put
    // any paint on it. Tinting the brown map directly would go muddy.
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    c = vec3(clamp(0.42 + l * 1.25, 0.0, 1.0));
  }
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  vec2 g = wdG(uv);
  float bid, gap;
  float bt = wdBoard(g, bid, gap);
  float band, fleck;
  wdGrain(g, bid, band, fleck);
  float rough = mix(0.86, 0.72, band);
  rough = mix(rough, 0.55, fleck * 0.4);
  rough += wdChecks(g) * 0.10;
  rough *= 0.95 + 0.10 * fbm2(g * 18.0, vec2(18.0));
  float metal = wdNail(g, bt);
  rough = mix(rough, 0.48, metal);
  rough = mix(rough, 0.42, wdWear(g) * 0.9);       // polished by boots
  rough = mix(rough, 0.99, wdSmoke(g) * 0.8);      // soot kills the sheen
  return vec3(mix(1.0, ao, 0.85), rough, metal * 0.9);
}
`;

/* ------------------------------------------------------------------ brickwork */
/* uA = (cols, rows, chip, jointWidth)  uB = (burnt, pale, sooty, _) */
FAMILY.brick = /* glsl */`
void brCell(vec2 uv, out vec2 lp, out float bid, out float bid2, out float edge) {
  float rv = uv.y * uA.y;
  float row = floor(rv), ry = fract(rv);
  float off = mod(row, 2.0) * 0.5;
  float cv = uv.x * uA.x + off;
  float col = floor(cv), cx = fract(cv);
  bid  = hash12(vec2(mod(col, uA.x), mod(row, uA.y)));
  bid2 = hash12(vec2(mod(col, uA.x) + 31.0, mod(row, uA.y) + 17.0));
  lp = vec2(cx, ry);
  float ex = min(cx, 1.0 - cx) / uA.x;
  float ey = min(ry, 1.0 - ry) / uA.y;
  edge = min(ex, ey) + (fbm2(uv * 120.0, vec2(120.0)) - 0.5) * uA.z;
}
float hgt(vec2 uv) {
  vec2 lp; float bid, bid2, edge;
  brCell(uv, lp, bid, bid2, edge);
  float m = smoothstep(0.0, uA.w, edge);
  float face = 0.55 + (bid - 0.5) * 0.08
             + (fbm(uv * 55.0, vec2(55.0)) - 0.5) * 0.09
             - smoothstep(0.80, 0.98, fbm2(uv * 34.0, vec2(34.0))) * 0.12;
  float mortar = 0.18 + (fbm2(uv * 150.0, vec2(150.0)) - 0.5) * 0.10;
  return mix(mortar, face, m);
}
vec3 alb(vec2 uv, float h) {
  vec2 lp; float bid, bid2, edge;
  brCell(uv, lp, bid, bid2, edge);
  float m = smoothstep(0.0, uA.w, edge);
  vec3 c = mix(vec3(0.620, 0.290, 0.208), vec3(0.478, 0.231, 0.173), bid);
  c = mix(c, vec3(0.690, 0.440, 0.348), smoothstep(0.80, 1.0, bid2) * uB.y);     // pale flare
  c = mix(c, vec3(0.250, 0.218, 0.230), smoothstep(0.84, 1.0, fract(bid2 * 5.7)) * uB.x); // burnt header
  c *= 0.86 + 0.30 * fbm(uv * 45.0, vec2(45.0));
  c *= 0.93 + 0.14 * fbm2(uv * 130.0, vec2(130.0));
  c = mix(c, c * 0.62, smoothstep(0.70, 0.95, fbm6(uv * 3.0 + 5.0, vec2(3.0))) * uB.z); // soot
  vec3 mortar = vec3(0.700, 0.678, 0.612) * (0.80 + 0.42 * fbm(uv * 140.0, vec2(140.0)));
  c = mix(mortar, c, m);
  c *= mix(0.86, 1.0, smoothstep(0.0, uA.w * 1.8, edge));
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  vec2 lp; float bid, bid2, edge;
  brCell(uv, lp, bid, bid2, edge);
  float m = smoothstep(0.0, uA.w, edge);
  float rough = mix(0.97, 0.80, m) * (0.94 + 0.12 * fbm2(uv * 40.0, vec2(40.0)));
  rough -= smoothstep(0.86, 1.0, fract(bid2 * 5.7)) * uB.x * 0.22;               // burnt bricks glaze
  return vec3(mix(1.0, ao, 0.90), rough, 0.0);
}
`;

/* --------------------------------------------------------------- rubble stone */
/* uA = (colsX, rowsY, jitter, jointWidth)  uB = (lichen, warmth, pitting, _) */
FAMILY.stone = /* glsl */`
vec4 stCell(vec2 uv) {
  vec2 w = vec2(fbm2(uv * 3.0 + 1.7, vec2(3.0)), fbm2(uv * 3.0 - 5.1, vec2(3.0))) - 0.5;
  vec2 p = vec2(uv.x * uA.x, uv.y * uA.y) + w * 0.75;
  // Running bond. The per-row shift must repeat over an even number of rows,
  // or the pattern will not tile in V.
  p.x += mod(floor(uv.y * uA.y), 2.0) * 0.5;
  return voro(p, vec2(uA.x, uA.y), uA.z);
}
float hgt(vec2 uv) {
  vec4 v = stCell(uv);
  float m = smoothstep(0.0, uA.w, v.z);
  float face = 0.55 + (v.y - 0.5) * 0.20
             + (fbm(uv * 26.0, vec2(26.0)) - 0.5) * 0.16
             - smoothstep(0.72, 0.96, fbm2(uv * 60.0, vec2(60.0))) * 0.16 * uB.z;
  float mortar = 0.20 + (fbm2(uv * 120.0, vec2(120.0)) - 0.5) * 0.12;
  float h = mix(mortar, face, m);
  h -= (1.0 - m) * 0.05;
  return h;
}
vec3 alb(vec2 uv, float h) {
  vec4 v = stCell(uv);
  float m = smoothstep(0.0, uA.w, v.z);
  // Grey ashlar / rubble: 0.20-0.24 linear = sRGB 0.50-0.56 for the faces.
  vec3 cool = vec3(0.522, 0.514, 0.484);
  vec3 warm = vec3(0.588, 0.540, 0.452);
  vec3 dark = vec3(0.359, 0.353, 0.339);
  vec3 c = mix(cool, warm, v.y * uB.y);
  c = mix(c, dark, smoothstep(0.66, 1.0, v.w) * 0.7);
  c *= 0.86 + 0.30 * fbm(uv * 30.0, vec2(30.0));
  c *= 0.94 + 0.12 * fbm2(uv * 110.0, vec2(110.0));
  float lich = smoothstep(0.54, 0.84, fbm6(uv * 7.0, vec2(7.0))) * m * uB.x;
  c = mix(c, vec3(0.520, 0.550, 0.345), lich * 0.75);
  float lich2 = smoothstep(0.72, 0.95, fbm(uv * 15.0 + 9.0, vec2(15.0))) * m * uB.x;
  c = mix(c, vec3(0.630, 0.642, 0.494), lich2 * 0.6);
  vec3 mortar = vec3(0.586, 0.569, 0.526) * (0.78 + 0.44 * fbm(uv * 100.0, vec2(100.0)));
  c = mix(mortar, c, m);
  c *= mix(0.80, 1.0, smoothstep(0.0, uA.w * 2.0, v.z));
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  vec4 v = stCell(uv);
  float m = smoothstep(0.0, uA.w, v.z);
  float lich = smoothstep(0.54, 0.84, fbm6(uv * 7.0, vec2(7.0))) * m * uB.x;
  float rough = mix(0.98, 0.84, m) * (0.94 + 0.12 * fbm2(uv * 34.0, vec2(34.0)));
  rough = mix(rough, 0.99, lich);
  return vec3(mix(1.0, ao, 0.92), rough, 0.0);
}
`;

/* ------------------------------------------------------------- clay pantiles */
/* uA = (cols, rows, _, _)  uB = (moss, variation, _, _) */
FAMILY.roofTile = /* glsl */`
float hgt(vec2 uv) {
  float cv = uv.y * uA.y;
  float row = mod(floor(cv), uA.y), ct = fract(cv);
  float uu = uv.x * uA.x;
  float col = mod(floor(uu), uA.x), cx = fract(uu);
  float barrel = sin(PI * clamp(cx, 0.0, 1.0));
  float lap = smoothstep(0.0, 0.10, ct);
  float sag = hash12(vec2(col, row)) * 0.06;
  float h = 0.30 + barrel * 0.45 * lap + smoothstep(0.0, 0.5, ct) * 0.10 + sag;
  h += (fbm2(uv * 70.0, vec2(70.0)) - 0.5) * 0.05;
  h -= (1.0 - lap) * 0.22;
  return h;
}
vec3 alb(vec2 uv, float h) {
  float cv = uv.y * uA.y;
  float row = mod(floor(cv), uA.y), ct = fract(cv);
  float uu = uv.x * uA.x;
  float col = mod(floor(uu), uA.x), cx = fract(uu);
  float id = hash12(vec2(col, row));
  /* Weathered clay pantile is 0.10-0.14 LINEAR. The old palette baked at 0.090,
     which put a whole roof below dry topsoil and killed the barrel read in shade.
     These are the old values x 1.089 in sRGB (= x 1.21 linear). */
  vec3 c = mix(vec3(0.697, 0.351, 0.196), vec3(0.512, 0.259, 0.161), id);
  c = mix(c, vec3(0.414, 0.327, 0.251), smoothstep(0.78, 1.0, fract(id * 6.1)) * uB.y);
  c *= 0.86 + 0.30 * fbm(uv * 26.0, vec2(26.0));
  float moss = smoothstep(0.56, 0.88, fbm6(uv * 6.0, vec2(6.0))) * uB.x;
  c = mix(c, vec3(0.300, 0.340, 0.208), moss * 0.8);
  // Contact shade under the lap only — the barrel curvature belongs to the
  // normal map, baking it into albedo would shade the tile twice.
  c *= mix(0.42, 1.0, smoothstep(0.0, 0.14, ct));
  c *= 0.94 + 0.10 * istep(0.5, 0.0, abs(cx - 0.5));
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float ct = fract(uv.y * uA.y);
  float moss = smoothstep(0.56, 0.88, fbm6(uv * 6.0, vec2(6.0))) * uB.x;
  float rough = 0.78 + 0.16 * fbm2(uv * 24.0, vec2(24.0));
  rough = mix(rough, 0.97, moss);
  return vec3(mix(1.0, ao, 0.88) * mix(0.55, 1.0, smoothstep(0.0, 0.16, ct)), rough, 0.0);
}
`;

/* ------------------------------------------------------------ terracotta pot */
/* uA = (rings, _, chips, _)  uB = (bloom, damp, _, _) */
FAMILY.terracotta = /* glsl */`
float hgt(vec2 uv) {
  float rings = sin(uv.y * uA.x * TAU + fbm2(vec2(uv.x * 3.0, uv.y * uA.x), vec2(3.0, uA.x)) * 2.0);
  float h = 0.55 + rings * 0.035;
  h += (fbm(uv * 60.0, vec2(60.0)) - 0.5) * 0.07;
  h += (fbm2(uv * 160.0, vec2(160.0)) - 0.5) * 0.05;
  h -= smoothstep(0.88, 0.99, fbm2(uv * 12.0, vec2(12.0))) * uA.z * 0.5;   // chipped
  return h;
}
vec3 alb(vec2 uv, float h) {
  vec3 c = mix(vec3(0.620, 0.352, 0.222), vec3(0.760, 0.508, 0.360),
               fbm6(uv * 4.0, vec2(4.0)));
  c *= 0.90 + 0.20 * fbm(uv * 30.0, vec2(30.0));
  float bloom = smoothstep(0.58, 0.92, fbm6(uv * 5.0 + 3.0, vec2(5.0))) * uB.x;
  c = mix(c, vec3(0.800, 0.780, 0.730), bloom * 0.65);                     // lime bloom
  float damp = smoothstep(0.60, 0.95, fbm(uv * 8.0 + 21.0, vec2(8.0))) * uB.y;
  c = mix(c, vec3(0.300, 0.198, 0.150), damp * 0.7);
  c = mix(c, vec3(0.840, 0.620, 0.470), smoothstep(0.90, 1.0, fbm2(uv * 12.0, vec2(12.0))) * uA.z);
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float bloom = smoothstep(0.58, 0.92, fbm6(uv * 5.0 + 3.0, vec2(5.0))) * uB.x;
  float rough = 0.74 + 0.16 * fbm2(uv * 20.0, vec2(20.0));
  rough = mix(rough, 0.95, bloom);
  return vec3(mix(1.0, ao, 0.8), rough, 0.0);
}
`;

/* ------------------------------------------------------------------- topsoil */
/* uA = (clods, stones, straw, _)  uB = (damp, _, _, _) */
FAMILY.soil = /* glsl */`
float hgt(vec2 uv) {
  vec4 v = voro(uv * uA.x, vec2(uA.x), 0.9);
  float clod = smoothstep(0.0, 0.35, v.z) * (0.55 + 0.45 * v.y);
  float h = 0.42 + clod * 0.30;
  h += (fbm(uv * 22.0, vec2(22.0)) - 0.5) * 0.22;
  h += (fbm2(uv * 90.0, vec2(90.0)) - 0.5) * 0.10;
  float st = smoothstep(0.90, 0.995, hash12(mod(floor(uv * 90.0), vec2(90.0))));
  h += st * uA.y * 0.22;
  return h;
}
vec3 alb(vec2 uv, float h) {
  vec4 v = voro(uv * uA.x, vec2(uA.x), 0.9);
  // Dry topsoil is 0.07-0.11 linear; this used to bake at 0.036, which made the
  // dirt lanes and the meadow wear decal the darkest thing in the frame.
  vec3 c = mix(vec3(0.195, 0.143, 0.100), vec3(0.383, 0.288, 0.201), fbm6(uv * 5.0, vec2(5.0)));
  c = mix(c, vec3(0.465, 0.380, 0.275), smoothstep(0.30, 0.9, v.y) * 0.5);
  c *= 0.80 + 0.42 * fbm(uv * 26.0, vec2(26.0));
  c *= 0.92 + 0.16 * fbm2(uv * 110.0, vec2(110.0));
  float st = smoothstep(0.90, 0.995, hash12(mod(floor(uv * 90.0), vec2(90.0))));
  c = mix(c, vec3(0.545, 0.524, 0.481), st * uA.y);                        // grit
  float straw = smoothstep(0.80, 0.97, fbm2(vec2(uv.x * 40.0, uv.y * 9.0), vec2(40.0, 9.0)));
  c = mix(c, vec3(0.545, 0.455, 0.276), straw * uA.z);
  c = mix(c, c * 0.62, smoothstep(0.55, 0.9, fbm6(uv * 3.0 + 8.0, vec2(3.0))) * uB.x);
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float rough = 0.96 - 0.10 * smoothstep(0.55, 0.9, fbm6(uv * 3.0 + 8.0, vec2(3.0))) * uB.x;
  return vec3(mix(1.0, ao, 0.9), rough, 0.0);
}
`;

/* -------------------------------------------------------------- meadow grass */
/* uA = (clumps, blades, dry, _)  uB = (dirt, blue, _, _) */
FAMILY.grass = /* glsl */`
/*
 * Blade streaks. A ROTATED domain does not tile at all, and a sheared one only
 * tiles when the shear is a whole multiple of the x period — so the second
 * layer leans exactly one tile per tile, which is enough to break the
 * verticality of the first.
 */
float grBladesV(vec2 uv) {
  return fbm2(vec2(uv.x * 90.0, uv.y * 14.0), vec2(90.0, 14.0));
}
float grBladesD(vec2 uv) {
  return fbm2(vec2(uv.x * 70.0 + uv.y * 70.0, uv.y * 12.0), vec2(70.0, 12.0));
}
/*
 * TILING. Terrain lays this down at one tile per few metres over a hillside that
 * fills a quarter of the screen, so ANY low-frequency content is a readable
 * repeat: the eye locks onto the per-Voronoi-cell tone and the fbm blotches and
 * sees a hexagonal lattice marching across the hill. The fix is not more noise,
 * it is LESS low-frequency contrast — the variation lives at frequencies the mip
 * chain averages to a flat mean, and the per-cell terms only modulate weakly.
 */
float hgt(vec2 uv) {
  vec4 v = voro(uv * uA.x, vec2(uA.x), 0.85);
  float clump = istep(0.9, 0.0, v.x) * (0.5 + 0.5 * v.y);
  float h = 0.42 + clump * 0.13;
  h += (fbm(uv * 20.0, vec2(20.0)) - 0.5) * 0.16;
  h += (grBladesV(uv) - 0.5) * 0.18 * uA.y;
  h += (grBladesD(uv) - 0.5) * 0.14 * uA.y;
  h += (fbm2(uv * 70.0, vec2(70.0)) - 0.5) * 0.08;
  return h;
}
vec3 alb(vec2 uv, float h) {
  vec4 v = voro(uv * uA.x, vec2(uA.x), 0.85);
  // Meadow grass is 0.10-0.16 LINEAR = sRGB 0.36-0.44 for the mid tone. The old
  // palette measured 0.074 linear, which is darker than wet asphalt and is why
  // the meadow read as one flat dark mass whatever the sun did.
  vec3 deep = vec3(0.205, 0.294, 0.140);
  vec3 mid  = vec3(0.328, 0.452, 0.198);
  vec3 lit  = vec3(0.440, 0.543, 0.257);
  vec3 dry  = vec3(0.536, 0.517, 0.289);
  // Tonal field pushed up in frequency (was uv*4 — a 1 m blotch at meadow scale)
  // and narrowed in range, so distance averages it out instead of tiling it.
  float f = fbm(uv * 11.0, vec2(11.0));
  vec3 c = mix(deep, mid, smoothstep(0.26, 0.70, f));
  c = mix(c, lit, smoothstep(0.62, 0.96, f) * 0.55);
  c = mix(c, dry, smoothstep(0.58, 0.94, fbm(uv * 8.0 + 17.0, vec2(8.0))) * uA.z * 0.7);
  c *= 0.84 + 0.32 * grBladesV(uv) * uA.y + 0.16 * grBladesD(uv) * uA.y;
  c *= 0.90 + 0.20 * fbm(uv * 26.0, vec2(26.0));
  c = mix(c, vec3(0.290, 0.222, 0.150), smoothstep(0.80, 0.98, fbm(uv * 13.0 + 41.0, vec2(13.0))) * uB.x * 0.8);
  c = mix(c, c * vec3(0.86, 0.94, 1.10), uB.y);                            // aerial-perspective shift
  c *= mix(0.90, 1.04, smoothstep(0.2, 0.8, v.y));   // was 0.72..1.06: THE hex lattice
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float rough = 0.88 + 0.10 * fbm2(uv * 16.0, vec2(16.0));
  return vec3(mix(1.0, ao, 0.85), rough, 0.0);
}
`;

/* ------------------------------------------------------------------- bark */
/* uA = (furrows, rings, moss, _)  uB = (tone, _, _, _)  uC = base.rgb */
FAMILY.bark = /* glsl */`
vec4 bkCell(vec2 uv) {
  vec2 w = vec2(fbm2(vec2(uv.x * 4.0, uv.y * 2.0), vec2(4.0, 2.0)),
                fbm2(vec2(uv.x * 4.0 + 3.0, uv.y * 2.0), vec2(4.0, 2.0))) - 0.5;
  return voro(vec2(uv.x * uA.x, uv.y * uA.y) + w * vec2(1.6, 0.7), vec2(uA.x, uA.y), 0.95);
}
float hgt(vec2 uv) {
  vec4 v = bkCell(uv);
  float fis = smoothstep(0.0, 0.09, v.z);                    // deep fissures between plates
  float plate = 0.45 + v.y * 0.22;
  float h = mix(0.10, plate, fis);
  h += (fbm(vec2(uv.x * 10.0, uv.y * 30.0), vec2(10.0, 30.0)) - 0.5) * 0.18 * fis;
  h += (fbm2(vec2(uv.x * 40.0, uv.y * 110.0), vec2(40.0, 110.0)) - 0.5) * 0.09;
  return h;
}
vec3 alb(vec2 uv, float h) {
  vec4 v = bkCell(uv);
  float fis = smoothstep(0.0, 0.09, v.z);
  vec3 c = uC.rgb * uB.x * (0.72 + 0.56 * v.y);
  c *= 0.82 + 0.36 * fbm(vec2(uv.x * 10.0, uv.y * 26.0), vec2(10.0, 26.0));
  c *= 0.92 + 0.16 * fbm2(vec2(uv.x * 40.0, uv.y * 110.0), vec2(40.0, 110.0));
  c = mix(uC.rgb * 0.22, c, fis);
  float moss = smoothstep(0.58, 0.90, fbm6(uv * 5.0, vec2(5.0))) * uA.z * fis;
  c = mix(c, vec3(0.230, 0.290, 0.152), moss * 0.8);
  float lich = smoothstep(0.80, 0.97, fbm(uv * 11.0 + 5.0, vec2(11.0))) * uA.z * fis;
  c = mix(c, vec3(0.560, 0.580, 0.470), lich * 0.6);
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  return vec3(mix(1.0, ao, 0.95), 0.94 + 0.06 * fbm2(uv * 20.0, vec2(20.0)), 0.0);
}
`;

/* ------------------------------------------------------------------- metal */
/* uA = (patina, hammer, rust, pitting)  uB = (roughBase, streak, _, _)
   uC = (base.rgb, _) */
FAMILY.metal = /* glsl */`
float mtPatina(vec2 uv) {
  float f = fbm6(uv * 4.0, vec2(4.0));
  float streak = fbm2(vec2(uv.x * 14.0, uv.y * 3.0), vec2(14.0, 3.0));
  return sat(smoothstep(0.34, 0.72, f * 0.72 + streak * 0.28) * uA.x);
}
float mtRust(vec2 uv) {
  return sat(smoothstep(0.52, 0.86, fbm6(uv * 6.0 + 9.0, vec2(6.0))) * uA.z);
}
float hgt(vec2 uv) {
  vec4 v = voro(uv * uA.y, vec2(uA.y), 0.8);
  float dimple = smoothstep(0.0, 0.55, v.x) * 0.12;          // hammered planish marks
  float h = 0.60 - dimple;
  h -= smoothstep(0.74, 0.96, fbm2(uv * 55.0, vec2(55.0))) * uA.w * 0.30;
  h += (fbm2(uv * 130.0, vec2(130.0)) - 0.5) * 0.04;
  h += mtPatina(uv) * 0.06 + mtRust(uv) * 0.09;
  return h;
}
vec3 alb(vec2 uv, float h) {
  vec3 c = uC.rgb * (0.86 + 0.28 * fbm(uv * 18.0, vec2(18.0)));
  c *= 0.94 + 0.12 * fbm2(uv * 90.0, vec2(90.0));
  float p = mtPatina(uv);
  vec3 verd = mix(vec3(0.220, 0.470, 0.408), vec3(0.372, 0.640, 0.548),
                  fbm(uv * 12.0, vec2(12.0)));
  c = mix(c, verd, p);
  float r = mtRust(uv);
  c = mix(c, mix(vec3(0.330, 0.148, 0.078), vec3(0.480, 0.242, 0.118),
                 fbm(uv * 20.0, vec2(20.0))), r);
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float p = mtPatina(uv);
  float r = mtRust(uv);
  float rough = uB.x * (0.88 + 0.24 * fbm2(uv * 26.0, vec2(26.0)));
  rough = mix(rough, 0.92, max(p, r * 0.9));
  float metal = 1.0 - max(p * 0.85, r * 0.75);
  return vec3(mix(1.0, ao, 0.75), rough, metal);
}
`;

/* -------------------------------------------------------- woven cloth / rope */
/* uA = (mode 0=weave 1=rope, threads, stripes, _)
   uB = (_, _, _, _)  uC = (weft.rgb, _)
   uD = (warp.rgb, fraying)   INTERIOR extension: `linen` and `sackcloth` need a
        ground colour other than the awning's cream. uD = 0 keeps the original
        cream, so `fabric` and `rope` bake exactly what they did before. */
FAMILY.fibre = /* glsl */`
vec3 fbGround() {
  return (uD.x + uD.y + uD.z) > 0.004 ? uD.rgb : vec3(0.845, 0.800, 0.700);
}
float fbFray(vec2 uv) {
  if (uD.w < 0.001) return 0.0;
  return smoothstep(0.70, 0.97, fbm2(vec2(uv.x * 6.0, uv.y * 120.0), vec2(6.0, 120.0))) * uD.w;
}
float fbWeave(vec2 uv, out float over) {
  vec2 q = uv * uA.y;
  vec2 c = floor(q), f = fract(q);
  over = mod(c.x + c.y, 2.0);
  float warp = sin(PI * clamp(f.x, 0.0, 1.0));
  float weft = sin(PI * clamp(f.y, 0.0, 1.0));
  return mix(weft * 0.55 + 0.25, warp * 0.55 + 0.25, over);
}
float fbRope(vec2 uv) {
  float s = fract(uv.x * 3.0 + uv.y * 9.0);
  float strand = pow(sin(PI * s), 0.7);
  float fine = fbm2(vec2(uv.x * 60.0 + uv.y * 180.0, uv.y * 12.0), vec2(60.0, 12.0));
  return strand * 0.7 + fine * 0.22 + 0.1;
}
float hgt(vec2 uv) {
  if (uA.x > 0.5) return fbRope(uv);
  float over;
  float h = fbWeave(uv, over);
  h += (fbm2(uv * 90.0, vec2(90.0)) - 0.5) * 0.10;
  // slubs and loose fibre — the difference between cloth and graph paper
  h += (fbm2(vec2(uv.x * 7.0, uv.y * 130.0), vec2(7.0, 130.0)) - 0.5) * 0.20 * uD.w;
  return h;
}
vec3 alb(vec2 uv, float h) {
  if (uA.x > 0.5) {
    vec3 c = uC.rgb * (0.68 + 0.62 * fbRope(uv));
    c *= 0.88 + 0.26 * fbm(vec2(uv.x * 24.0, uv.y * 60.0), vec2(24.0, 60.0));
    return c;
  }
  float over;
  float w = fbWeave(uv, over);
  float band = step(0.5, fract(uv.x * uA.z));
  vec3 cream = fbGround();
  vec3 c = mix(cream, uC.rgb, band);
  c *= 0.72 + 0.50 * w;
  c *= 0.92 + 0.16 * fbm(uv * 20.0, vec2(20.0));
  c *= 0.94 + 0.12 * fbm2(uv * 120.0, vec2(120.0));
  c = mix(c, c * 1.30 + 0.015, fbFray(uv) * 0.55);          // lifted fibre catches light
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float rough = 0.90 + 0.08 * fbm2(uv * 40.0, vec2(40.0));
  rough += fbFray(uv) * 0.06;
  return vec3(mix(1.0, ao, 0.9), rough, 0.0);
}
`;

/* ------------------------------------------------------------- old glass */
/* uA = (wave, bubbles, _, _) */
FAMILY.glass = /* glsl */`
float hgt(vec2 uv) {
  float h = 0.5 + (fbm(vec2(uv.x * 3.0, uv.y * 7.0), vec2(3.0, 7.0)) - 0.5) * uA.x;
  h += (fbm2(vec2(uv.x * 9.0, uv.y * 2.0), vec2(9.0, 2.0)) - 0.5) * uA.x * 0.6;
  vec4 v = voro(uv * 14.0, vec2(14.0), 0.9);
  h += istep(0.14, 0.0, v.x) * uA.y * 0.35;          // seed bubbles
  return h;
}
vec3 alb(vec2 uv, float h) {
  vec3 c = vec3(0.035, 0.050, 0.055);
  c *= 0.7 + 0.9 * fbm(vec2(uv.x * 3.0, uv.y * 7.0), vec2(3.0, 7.0));
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float rough = 0.045 + 0.075 * fbm(vec2(uv.x * 4.0, uv.y * 9.0), vec2(4.0, 9.0));
  return vec3(1.0, rough, 0.0);
}
`;

/* ========================================================================== */
/* GLSL — interior families                                                    */
/* ========================================================================== */
/*
 * Everything below here is only ever seen from inside a building, which changes
 * two things about how it is authored:
 *
 *  - The light is one small window and one fire, so ALBEDO is doing most of the
 *    work. `lathPlaster` is deliberately the brightest set in the library (it is
 *    the surface that bounces the hearth around the room) and `soot` is the
 *    darkest by an order of magnitude.
 *  - The camera can be 40 cm from the surface (a floor is always underfoot), but
 *    each set covers far less area than a hero exterior set. So: mid-ish
 *    resolution, but relief authored for close inspection.
 */

/* -------------------------------------------------------- riven flagstone floor */
/* uA = (cols, rows, jitter, jointWidth)   uB = (polish, damp, chip, _) */
FAMILY.flagstone = /* glsl */`
vec4 fsCell(vec2 uv) {
  vec2 w = vec2(fbm2(uv * 3.0 + 2.9, vec2(3.0)), fbm2(uv * 3.0 - 4.4, vec2(3.0))) - 0.5;
  vec2 p = vec2(uv.x * uA.x, uv.y * uA.y) + w * 1.15;
  // Random courses, like flags lifted from a quarry floor and laid to fit. The
  // half-cell shift has to repeat over an even number of rows to tile in V.
  p.x += mod(floor(uv.y * uA.y), 2.0) * 0.5;
  return voro(p, vec2(uA.x, uA.y), uA.z);
}
/* 1 in the middle of a slab, 0 at its edge — a flag wears DISHED, not domed. */
float fsDish(vec4 v) { return smoothstep(0.06, 0.55, v.z); }

float hgt(vec2 uv) {
  vec4 v = fsCell(uv);
  float m = smoothstep(0.0, uA.w, v.z);
  float dish = fsDish(v);
  float face = 0.60 + (v.y - 0.5) * 0.10 - dish * 0.20;
  // Riven stone splits along its bedding plane: broad shallow ripples, not pits.
  face += (fbm(uv * 16.0, vec2(16.0)) - 0.5) * 0.13 * (1.0 - 0.55 * dish);
  face += (ridge(uv * 10.0, vec2(10.0)) - 0.5) * 0.09 * (1.0 - 0.60 * dish);
  face -= smoothstep(0.88, 1.0, fbm2(uv * 34.0, vec2(34.0))) * uB.z * 0.34;
  float mortar = 0.24 + (fbm2(uv * 120.0, vec2(120.0)) - 0.5) * 0.12;
  return mix(mortar, face, m);
}
vec3 alb(vec2 uv, float h) {
  vec4 v = fsCell(uv);
  float m = smoothstep(0.0, uA.w, v.z);
  float dish = fsDish(v);
  /* Indoor flags are 0.16-0.20 linear: a shade darker than the exterior ashlar
     (0.213) because centuries of boots, ash and grease have sealed them. */
  vec3 cool = vec3(0.470, 0.462, 0.446);
  vec3 buff = vec3(0.520, 0.484, 0.428);
  vec3 dark = vec3(0.322, 0.316, 0.306);
  vec3 c = mix(cool, buff, sat(v.y * 1.3));
  c = mix(c, dark, smoothstep(0.62, 1.0, v.w) * 0.72);
  c *= 0.88 + 0.26 * fbm(uv * 22.0, vec2(22.0));
  c *= 0.94 + 0.12 * fbm2(uv * 95.0, vec2(95.0));
  c = mix(c, mix(c, vec3(0.392, 0.384, 0.372), 0.45), dish * uB.x);   // polished hollow
  c = mix(c, c * 0.70, smoothstep(0.58, 0.92, fbm6(uv * 4.0 + 6.0, vec2(4.0))) * uB.y);
  vec3 mortar = vec3(0.404, 0.392, 0.360) * (0.78 + 0.44 * fbm(uv * 105.0, vec2(105.0)));
  c = mix(mortar, c, m);
  c *= mix(0.82, 1.0, smoothstep(0.0, uA.w * 2.0, v.z));
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  vec4 v = fsCell(uv);
  float m = smoothstep(0.0, uA.w, v.z);
  float dish = fsDish(v);
  float rough = mix(0.97, 0.86, m) * (0.94 + 0.12 * fbm2(uv * 30.0, vec2(30.0)));
  // The dished middle is the only part of an indoor floor with any sheen at all.
  rough = mix(rough, 0.54, dish * uB.x * 0.85);
  return vec3(mix(1.0, ao, 0.90), rough, 0.0);
}
`;

/* --------------------------------------------------- lime-washed lath & plaster */
/* uA = (laths, flaking, trowel, cobweb)   uB = (grime, lathSag, _, _) */
FAMILY.lathPlaster = /* glsl */`
/* The float coat sags a little between every riven lath, so an interior wall has
   a faint corduroy that only shows in raking window light. */
float lpLath(vec2 uv) {
  float t = uv.y * uA.x + (fbm2(vec2(uv.x * 4.0, uv.y * 2.0), vec2(4.0, 2.0)) - 0.5) * 0.6;
  return 0.5 - 0.5 * cos(TAU * fract(t));
}
/* Distemper lifting in irregular scabs, back to the grey float coat. */
float lpFlake(vec2 uv) {
  vec2 w = vec2(fbm2(uv * 5.0, vec2(5.0)), fbm2(uv * 5.0 + 3.1, vec2(5.0))) - 0.5;
  vec4 v = voro(uv * 13.0 + w * 1.3, vec2(13.0), 0.95);
  return istep(0.34, 0.12, v.x) * smoothstep(0.52, 0.80, fbm(uv * 3.0 + 7.0, vec2(3.0))) * uA.y;
}
/* Cobweb: thin pale filaments, sparse, and only where the noise says "corner". */
float lpWeb(vec2 uv) {
  float f = ridge(uv * 26.0, vec2(26.0));
  return smoothstep(0.86, 0.99, f) * smoothstep(0.66, 0.95, fbm(uv * 2.0 + 19.0, vec2(2.0))) * uA.w;
}
float hgt(vec2 uv) {
  float trowel = fbm(uv * 4.0, vec2(4.0));
  float sweep = fbm2(vec2(uv.x * 7.0, uv.y * 2.0), vec2(7.0, 2.0));
  float h = 0.56 + (trowel - 0.5) * uA.z + (sweep - 0.5) * uA.z * 0.50;
  h += (lpLath(uv) - 0.5) * uB.y;
  h += (fbm(uv * 26.0, vec2(26.0)) - 0.5) * 0.06;
  h += (fbm2(uv * 90.0, vec2(90.0)) - 0.5) * 0.035;
  h -= lpFlake(uv) * 0.26;
  return h;
}
vec3 alb(vec2 uv, float h) {
  /* LIMEWASH INDOORS: 0.44-0.50 linear, the brightest albedo in the library. It
     is what carries a single window and one fire around a whole room, so it is
     brighter AND cleaner than the exterior 'plaster' (0.421), which has forty
     years of rain streaks on it. Do not "dirty this up to match" — the exterior
     set is dirty because it is outside. */
  vec3 wash  = vec3(0.764, 0.752, 0.721);
  vec3 shade = vec3(0.680, 0.667, 0.635);
  float mot = fbm6(uv * 3.0, vec2(3.0));
  vec3 c = mix(shade, wash, smoothstep(0.22, 0.86, mot));
  c *= 0.965 + 0.070 * fbm(uv * 20.0, vec2(20.0));
  c *= 0.980 + 0.040 * fbm2(uv * 88.0, vec2(88.0));
  c *= 0.985 + 0.030 * lpLath(uv);
  c = mix(c, vec3(0.472, 0.440, 0.396), lpFlake(uv) * 0.85);
  float grime = smoothstep(0.52, 0.92, fbm6(uv * 2.0 + 11.0, vec2(2.0))) * uB.x;
  c = mix(c, vec3(0.520, 0.500, 0.470), grime * 0.60);
  c = mix(c, vec3(0.800, 0.795, 0.782), lpWeb(uv) * 0.50);
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float rough = 0.92 + 0.06 * fbm2(uv * 15.0, vec2(15.0));
  rough -= smoothstep(0.55, 0.95, fbm6(uv * 3.0, vec2(3.0))) * 0.08;   // burnished limewash
  rough += lpFlake(uv) * 0.05;
  // Light AO: this surface is the room's bounce card, and baked cavity shade on
  // a nearly flat wall only ever makes an interior murkier.
  return vec3(mix(1.0, ao, 0.62), rough, 0.0);
}
`;

/* --------------------------------------------------- fire-blackened hearth stone */
/* uA = (cols, rows, chip, jointWidth)   uB = (soot, crazing, spall, _) */
FAMILY.hearthStone = /* glsl */`
void hsCell(vec2 uv, out float id, out float id2, out float edge) {
  float rv = uv.y * uA.y;
  float row = floor(rv), ry = fract(rv);
  float cv = uv.x * uA.x + mod(row, 2.0) * 0.5;
  float col = floor(cv), cx = fract(cv);
  id  = hash12(vec2(mod(col, uA.x), mod(row, uA.y)));
  id2 = hash12(vec2(mod(col, uA.x) + 13.0, mod(row, uA.y) + 29.0));
  float ex = min(cx, 1.0 - cx) / uA.x;
  float ey = min(ry, 1.0 - ry) / uA.y;
  edge = min(ex, ey) + (fbm2(uv * 130.0, vec2(130.0)) - 0.5) * uA.z;
}
/* Heat crazing: a fine dark web, tight and only over the hottest patches. */
float hsCraze(vec2 uv) {
  vec2 w = vec2(fbm2(uv * 6.0, vec2(6.0)), fbm2(uv * 6.0 + 5.0, vec2(6.0))) - 0.5;
  vec4 v = voro(uv * 15.0 + w * 1.5, vec2(15.0), 0.95);
  return istep(0.018, 0.004, v.z) * smoothstep(0.30, 0.72, fbm(uv * 4.0, vec2(4.0))) * uB.y;
}
float hsSoot(vec2 uv) {
  return smoothstep(0.40, 0.86, fbm6(uv * 3.0 + 4.0, vec2(3.0))) * uB.x;
}
float hgt(vec2 uv) {
  float id, id2, edge;
  hsCell(uv, id, id2, edge);
  float m = smoothstep(0.0, uA.w, edge);
  float face = 0.58 + (id - 0.5) * 0.07
             + (fbm(uv * 30.0, vec2(30.0)) - 0.5) * 0.10
             - smoothstep(0.76, 0.97, fbm2(uv * 26.0, vec2(26.0))) * uB.z * 0.26;   // spalled
  float mortar = 0.22 + (fbm2(uv * 140.0, vec2(140.0)) - 0.5) * 0.10;
  return mix(mortar, face, m) - hsCraze(uv) * 0.16;
}
vec3 alb(vec2 uv, float h) {
  float id, id2, edge;
  hsCell(uv, id, id2, edge);
  float m = smoothstep(0.0, uA.w, edge);
  /* 0.08-0.12 linear. Dressed sandstone starts near the exterior 'stone' set and
     is then dragged down by three generations of smoke. */
  vec3 c = mix(vec3(0.478, 0.458, 0.427), vec3(0.382, 0.367, 0.353), id);
  c = mix(c, vec3(0.530, 0.497, 0.446), smoothstep(0.82, 1.0, id2) * 0.6);   // a fresher stone
  c *= 0.86 + 0.30 * fbm(uv * 34.0, vec2(34.0));
  c *= 0.93 + 0.14 * fbm2(uv * 120.0, vec2(120.0));
  vec3 mortar = vec3(0.485, 0.470, 0.439) * (0.76 + 0.46 * fbm(uv * 130.0, vec2(130.0)));
  c = mix(mortar, c, m);
  c = mix(c, vec3(0.176, 0.170, 0.166), hsSoot(uv) * 0.82);                  // smoke wash
  c = mix(c, vec3(0.135, 0.128, 0.124), hsCraze(uv) * 0.8);
  c *= mix(0.84, 1.0, smoothstep(0.0, uA.w * 1.8, edge));
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float id, id2, edge;
  hsCell(uv, id, id2, edge);
  float m = smoothstep(0.0, uA.w, edge);
  float rough = mix(0.98, 0.88, m) * (0.94 + 0.10 * fbm2(uv * 36.0, vec2(36.0)));
  rough = mix(rough, 0.99, hsSoot(uv));           // soot is the flattest thing there is
  return vec3(mix(1.0, ao, 0.92), rough, 0.0);
}
`;

/* --------------------------------------------------------------- soot deposit */
/* uA = (flocc, tar, ash, _) */
FAMILY.soot = /* glsl */`
float stTar(vec2 uv) {
  return smoothstep(0.62, 0.90, fbm6(uv * 5.0 + 3.0, vec2(5.0))) * uA.y;
}
float hgt(vec2 uv) {
  float h = 0.50 + (fbm(uv * 12.0, vec2(12.0)) - 0.5) * 0.26 * uA.x;
  h += (fbm2(uv * 55.0, vec2(55.0)) - 0.5) * 0.16 * uA.x;
  h += (fbm2(uv * 150.0, vec2(150.0)) - 0.5) * 0.07;
  h += stTar(uv) * 0.08;                      // tarry runs stand slightly proud
  return h;
}
vec3 alb(vec2 uv, float h) {
  /* 0.02-0.035 linear. Soot is the blackest common surface in the world — a
     chimney breast is darker than anything outdoors by a factor of six. */
  vec3 c = vec3(0.176, 0.170, 0.166);
  c *= 0.72 + 0.56 * fbm(uv * 14.0, vec2(14.0));
  c *= 0.90 + 0.20 * fbm2(uv * 90.0, vec2(90.0));
  c = mix(c, vec3(0.315, 0.305, 0.296), smoothstep(0.74, 0.96, fbm(uv * 9.0 + 17.0, vec2(9.0))) * uA.z * 0.55);
  c = mix(c, vec3(0.120, 0.116, 0.118), stTar(uv) * 0.80);
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float rough = 0.97 - 0.05 * fbm2(uv * 30.0, vec2(30.0));
  rough = mix(rough, 0.42, stTar(uv) * 0.9);
  return vec3(mix(1.0, ao, 0.85), rough, 0.0);
}
`;

/* ----------------------------------------------------------- loose straw litter */
/* uA = (straws, _, dirt, _)   uB = (chaff, damp, _, _)
 *
 * TILING: a rotated domain does not tile at all, so the crossing layers are
 * built from INTEGER shears of uv — (u), (u+v), (u-v) — which do. The straw
 * count must stay even so the warp's noise period (n/2) is a whole number.
 */
FAMILY.strawLitter = /* glsl */`
float slLayer(vec2 uv, vec2 across, vec2 along, float n, float seed) {
  float a = dot(uv, across) * n;
  float l = dot(uv, along);
  a += (fbm2(vec2(l * 5.0, a * 0.5), vec2(5.0, n * 0.5)) - 0.5) * 1.0;
  float t = fract(a);
  float id = mod(floor(a), n);
  float body = pow(sin(PI * clamp(t, 0.002, 0.998)), 4.0);          // one thin straw
  float seg = smoothstep(0.30, 0.55, fbm2(vec2(l * 18.0, a), vec2(18.0, n)));  // broken lengths
  return body * seg * (0.60 + 0.80 * hash12(vec2(id, seed)));
}
float slStraw(vec2 uv) {
  float s = slLayer(uv, vec2(0.0, 1.0), vec2(1.0, 0.0), uA.x, 3.0);
  s = max(s, slLayer(uv, vec2(1.0, 1.0), vec2(1.0, -1.0), uA.x * 0.5, 17.0) * 0.85);
  s = max(s, slLayer(uv, vec2(0.0, 1.0), vec2(1.0, 0.0), uA.x * 0.5, 41.0) * 0.90);
  return sat(s);
}
float slCover(vec2 uv) {
  return smoothstep(0.08 + uA.z * 0.30, 0.50 + uA.z * 0.30, slStraw(uv));
}
float hgt(vec2 uv) {
  float s = slStraw(uv);
  float h = 0.34 + s * 0.42;
  h += (fbm(uv * 20.0, vec2(20.0)) - 0.5) * 0.10;
  h += (fbm2(uv * 90.0, vec2(90.0)) - 0.5) * 0.06;
  return h;
}
vec3 alb(vec2 uv, float h) {
  /* 0.18-0.24 linear: bright straw over dark trodden earth, and the mean lands
     between the two. Fresh straw alone would be as bright as the plaster. */
  vec3 dirt = mix(vec3(0.232, 0.184, 0.135), vec3(0.346, 0.279, 0.200), fbm6(uv * 6.0, vec2(6.0)));
  vec3 pale = vec3(0.732, 0.635, 0.413);
  vec3 gold = vec3(0.635, 0.521, 0.311);
  vec3 straw = mix(gold, pale, fbm2(vec2(uv.x * 18.0, uv.y * 5.0), vec2(18.0, 5.0)));
  straw *= 0.86 + 0.28 * fbm(uv * 30.0, vec2(30.0));
  vec3 c = mix(dirt, straw, slCover(uv));
  float chaff = smoothstep(0.88, 1.0, hash12(mod(floor(uv * 150.0), vec2(150.0)))) * uB.x;
  c = mix(c, pale * 0.92, chaff * 0.70);
  c = mix(c, c * 0.66, smoothstep(0.60, 0.92, fbm6(uv * 3.0 + 9.0, vec2(3.0))) * uB.y);  // trodden damp
  return c;
}
vec3 orm3(vec2 uv, float h, float ao) {
  float cover = slCover(uv);
  float rough = mix(0.98, 0.90, cover) * (0.96 + 0.08 * fbm2(uv * 26.0, vec2(26.0)));
  return vec3(mix(1.0, ao, 0.92), rough, 0.0);
}
`;

/* ========================================================================== */
/* GLSL — alpha cut-out atlases                                                */
/* ========================================================================== */

const ATLAS_HELPERS = /* glsl */`
/* uA = (cols, rows, _, _) */
vec2 atCell(vec2 uv, out vec2 cid) {
  vec2 g = uv * vec2(uA.x, uA.y);
  cid = floor(g);
  return (fract(g) - 0.5) * 2.0;      // local coords in [-1,1]
}
float atAA() { return max(uTexel * 2.0 * uA.x * 1.4, 0.004); }
`;

/* -------------------------------------------------------------- broadleaf/ivy */
/* uA=(cols,rows,_,_)  uB=(lobes, ivyMode, clusterRow, _)  uC=(leaf.rgb,_) */
FAMILY.leafAtlas = ATLAS_HELPERS + /* glsl */`
/* Signed "inside" amount for one leaf placed at o, rotated a, scaled s. */
float leafBlade(vec2 q, vec2 o, float a, float s, float seed, out float t, out float across) {
  vec2 p = rot2((q - o) / s, -a);
  t = (p.y + 0.92) / 1.84;
  float base, lobe, wide;
  if (uB.y > 0.5) {
    /* Ivy: palmate — broadest near the heart-shaped base, 3-5 pointed lobes. */
    base = sin(PI * pow(clamp(t, 0.0, 1.0), 0.42));
    lobe = 0.66 + 0.52 * abs(cos(uB.x * PI * clamp(t, 0.0, 1.0)));
    lobe *= 1.0 - smoothstep(0.82, 1.0, t) * 0.55;
    wide = 1.42;
  } else {
    /* Broadleaf: widest around a third of the way up, tapering to a point. */
    base = sin(PI * pow(clamp(t, 0.0, 1.0), 0.72));
    lobe = 1.0 + 0.14 * sin(uB.x * TAU * t + seed * 6.0);
    wide = 1.0;
  }
  float w = base * lobe * wide * (0.34 + 0.10 * hash12(vec2(seed, 4.0)));
  across = p.x / max(w, 1e-4);
  float d = abs(p.x) - w;
  float m = istep(atAA(), -atAA(), d) * step(0.0, t) * step(t, 1.0);
  float stem = istep(0.024, 0.012, abs(p.x)) * step(-1.0, p.y) * step(p.y, -0.52);
  return max(m, stem);
}

void leafSet(vec2 q, vec2 cid, out float mask, out float t, out float across, out float vein, out float seed) {
  mask = 0.0; t = 0.5; across = 0.0; vein = 0.0;
  seed = hash12(cid + 3.7);
  float n = (cid.y > 0.5) ? 3.0 : 1.0;     // top row: small sprigs of three
  for (int i = 0; i < 3; i++) {
    if (float(i) >= n) break;
    float fi = float(i);
    float sd = hash12(cid + fi * 7.3 + 1.1);
    float ang = (n > 1.5) ? (fi - 1.0) * 0.85 + (sd - 0.5) * 0.5
                          : (sd - 0.5) * 1.2;
    vec2 off = (n > 1.5) ? vec2((fi - 1.0) * 0.42, -0.18 + sd * 0.30) : vec2(0.0, -0.02);
    float sc = (n > 1.5) ? 0.52 + sd * 0.16 : 0.80 + sd * 0.18;
    float tt, aa;
    float m = leafBlade(q, off, ang, sc, sd, tt, aa);
    if (m > mask) { mask = m; t = tt; across = aa; seed = sd; }
  }
  float lat = sin((t * 9.0 + across * 3.0) * PI);
  vein = max(smoothstep(0.86, 1.0, 1.0 - abs(across)), smoothstep(0.80, 1.0, abs(lat)) * 0.55);
}

float hgt(vec2 uv) {
  vec2 cid; vec2 q = atCell(uv, cid);
  float mask, t, across, vein, seed;
  leafSet(q, cid, mask, t, across, vein, seed);
  float dish = 1.0 - across * across;                 // blade curls up from the midrib
  return 0.35 + mask * (0.30 * sat(dish) + vein * 0.22 + 0.10 * sin(t * 7.0));
}
vec4 atlasRGBA(vec2 uv) {
  vec2 cid; vec2 q = atCell(uv, cid);
  float mask, t, across, vein, seed;
  leafSet(q, cid, mask, t, across, vein, seed);
  vec3 deep = uC.rgb * 0.62;
  vec3 mid  = uC.rgb;
  vec3 lite = uC.rgb * 1.32 + vec3(0.045, 0.070, 0.020);
  vec3 c = mix(deep, mid, sat(0.35 + 0.65 * (1.0 - abs(across))));
  c = mix(c, lite, smoothstep(0.45, 1.0, t) * 0.45);                     // young tip
  c = mix(c, lite * 1.06, vein * 0.55);                                  // veins catch the light
  c *= 0.86 + 0.28 * fbm(uv * 40.0, vec2(40.0));
  c = mix(c, vec3(0.520, 0.470, 0.180), smoothstep(0.72, 1.0, seed) * smoothstep(0.6, 1.0, t) * 0.5);
  c *= 0.84 + 0.32 * hash12(cid + 9.1);
  return vec4(c, mask);
}
`;

/* ----------------------------------------------------------------- ivy leaves */
/* uA=(cols,rows,_,_)  uB=(lobeK, nMin, nExtra, vein)  uC=(leaf.rgb,_)
 *
 * ONE CELL IS A CLUSTER of uB.y..uB.y+uB.z small leaves, never a single big one:
 * an ivy card on a wall is ~30 cm across, and one 30 cm leaf reads as a sticker.
 * The silhouette is the real palmate ivy outline — a polar radius with 5 lobes,
 * deep sinuses between them and a heart-shaped notch at the petiole — so the
 * alpha edge is a clean curve rather than a torn-paper fbm boundary.
 */
FAMILY.ivyAtlas = ATLAS_HELPERS + /* glsl */`
float ivLeaf(vec2 q, vec2 o, float rot, float s, float sd, out float rr, out float vein) {
  vec2 p = rot2((q - o) / s, -rot);
  float stem = istep(0.055, 0.024, abs(p.x)) * step(-1.05, p.y) * step(p.y, -0.44);
  vec2 b = p - vec2(0.0, -0.10);          // blade centre sits above the petiole
  float r = length(b);
  float a = atan(b.x, b.y);               // 0 points at the apex lobe
  float k = abs(cos(uB.x * a));           // uB.x = 2.5 -> 5 lobes at 0, +-72, +-144
  float R = 0.60 + 0.42 * pow(k, 0.55);
  R *= 1.0 + 0.26 * exp(-a * a * 3.2);                 // apex lobe runs longer
  R *= 1.0 - 0.52 * smoothstep(2.42, PI, abs(a));      // heart notch at the base
  R *= (0.84 + 0.26 * sd) * 0.86;
  float aa = atAA() / max(s, 1e-3);
  rr = sat(r / max(R, 1e-4));
  vein = pow(k, 24.0) * smoothstep(0.05, 0.30, rr) * istep(1.0, 0.84, rr);
  return max(istep(aa, -aa, r - R), stem);
}

void ivSet(vec2 q, vec2 cid, out float mask, out float rr, out float vein, out float sd) {
  mask = 0.0; rr = 0.6; vein = 0.0; sd = hash12(cid + 3.7);
  float n = uB.y + floor(hash12(cid + 12.3) * (uB.z + 0.999));
  for (int i = 0; i < 9; i++) {
    if (float(i) >= n) break;
    float fi = float(i);
    vec3 h3 = hash32(cid * 4.7 + fi * 1.37 + 0.9);
    float ang = (fi / max(n, 1.0)) * TAU + (h3.x - 0.5) * 1.3;
    float rad = 0.16 + 0.30 * h3.y;
    vec2 off = vec2(cos(ang) * rad, sin(ang) * rad * 0.92);
    // Apex points away from the middle of the cluster, petiole inward.
    float rot = PI * 0.5 - ang + (h3.x - 0.5) * 0.9;
    float sc = 0.19 + 0.09 * h3.z;
    float r2, vn;
    float m = ivLeaf(q, off, rot, sc, h3.z, r2, vn);
    // Painter's order: a later leaf overlaps the ones before it.
    if (m > 0.5) { rr = r2; vein = vn; sd = h3.y; }
    mask = max(mask, m);
  }
}

float hgt(vec2 uv) {
  vec2 cid; vec2 q = atCell(uv, cid);
  float mask, rr, vein, sd;
  ivSet(q, cid, mask, rr, vein, sd);
  float dish = 1.0 - rr * rr;                          // blade cups up from the veins
  return 0.32 + mask * (0.26 * sat(dish) + vein * 0.30);
}
vec4 atlasRGBA(vec2 uv) {
  vec2 cid; vec2 q = atCell(uv, cid);
  float mask, rr, vein, sd;
  ivSet(q, cid, mask, rr, vein, sd);
  vec3 mid  = uC.rgb;                                  // ~#325d2c
  vec3 deep = uC.rgb * 0.60;                           // ~#1e3d1c
  vec3 lite = mix(uC.rgb, vec3(0.372, 0.482, 0.276), 0.60);
  vec3 c = mix(mid, deep, smoothstep(0.46, 1.0, rr));  // margin darker than the middle
  c = mix(c, lite, vein * uB.w);                       // pale palmate veins
  c *= 0.90 + 0.20 * sd;                               // per-leaf tone
  c *= 0.92 + 0.16 * fbm(uv * 60.0, vec2(60.0));
  float old = smoothstep(0.88, 1.0, sd);
  c = mix(c, vec3(0.318, 0.242, 0.128), old * smoothstep(0.55, 1.0, rr) * 0.55); // bronzed margins
  c *= 0.90 + 0.20 * hash12(cid + 9.1);
  return vec4(c, mask);
}
`;

/* ----------------------------------------------------------- conifer needles */
/* uA=(cols,rows,_,_)  uB=(needles, spread, _, _)  uC=(needle.rgb,_) */
FAMILY.needleAtlas = ATLAS_HELPERS + /* glsl */`
float ndSide(vec2 q, float side, float seed, out float along) {
  float ang = uB.y;                     // radians from the stem
  float sn = sin(ang), cs = cos(ang);
  float x = q.x * side;
  along = x / max(sn, 0.08);
  float y0 = q.y - x * (cs / max(sn, 0.08));
  float k = (y0 + 1.0) * uB.x * 0.5;
  float i = floor(k), f = fract(k) - 0.5;
  float L = 0.42 + 0.36 * hash12(vec2(i, side + seed));
  float taper = 1.0 - smoothstep(0.0, L, along);
  float halfw = 0.32 * taper + 0.02;
  float m = istep(halfw, halfw * 0.45, abs(f));
  m *= step(0.0, along) * step(along, L) * step(0.0, x);
  // Only sprout needles from the middle of the stem, so none of them can reach
  // past the top of the atlas cell and bleed into its neighbour at low mips.
  m *= step(-0.88, y0) * step(y0, 0.42);
  return m;
}
float ndMask(vec2 q, vec2 cid, out float along, out float stem) {
  float seed = hash12(cid + 5.5);
  vec2 p = rot2(q, (seed - 0.5) * 0.5);
  p.y += 0.05;
  float a1, a2;
  float m = max(ndSide(p, 1.0, seed, a1), ndSide(p, -1.0, seed, a2));
  along = max(a1, a2);
  // Inset from the cell border: at low mip levels a shape touching the edge
  // bleeds into the neighbouring atlas cell.
  stem = istep(0.028, 0.014, abs(p.x)) * step(-0.92, p.y) * step(p.y, 0.72);
  return max(m, stem);
}
float hgt(vec2 uv) {
  vec2 cid; vec2 q = atCell(uv, cid);
  float along, stem;
  float m = ndMask(q, cid, along, stem);
  return 0.35 + m * 0.26 + stem * 0.18;
}
vec4 atlasRGBA(vec2 uv) {
  vec2 cid; vec2 q = atCell(uv, cid);
  float along, stem;
  float m = ndMask(q, cid, along, stem);
  vec3 c = uC.rgb * (0.72 + 0.52 * fbm(uv * 50.0, vec2(50.0)));
  c = mix(c, uC.rgb * 1.45 + vec3(0.02, 0.05, 0.01), sat(along * 1.3) * 0.5);
  c = mix(c, vec3(0.240, 0.180, 0.110), stem * 0.8);
  c *= 0.82 + 0.34 * hash12(cid + 2.3);
  return vec4(c, m);
}
`;

/* ------------------------------------------------------------------ blossom */
/* uA=(cols,rows,_,_)  uB=(petals, _, _, _) */
FAMILY.flowerAtlas = ATLAS_HELPERS + /* glsl */`
float flHead(vec2 q, vec2 o, float s, float petals, float seed, out float r, out float pet) {
  vec2 p = (q - o) / s;
  r = length(p);
  float a = atan(p.y, p.x) + seed * TAU;
  pet = 0.5 + 0.5 * cos(a * petals);
  float edge = 0.55 * (0.52 + 0.48 * pow(pet, 0.6));
  float aa = atAA() / max(s, 1e-3);
  return istep(aa, -aa, r - edge);
}
void flSet(vec2 q, vec2 cid, out float mask, out float r, out float pet, out float seed, out float core) {
  seed = hash12(cid + 1.9);
  mask = 0.0; r = 1.0; pet = 0.0; core = 0.0;
  float n = (cid.y > 0.5) ? 3.0 : 1.0;      // top row: clusters
  for (int i = 0; i < 3; i++) {
    if (float(i) >= n) break;
    float fi = float(i);
    float sd = hash12(cid + fi * 4.7 + 2.2);
    vec2 off = (n > 1.5) ? vec2(cos(fi * 2.2 + sd) * 0.38, sin(fi * 2.2 + sd) * 0.34) : vec2(0.0);
    float sc = (n > 1.5) ? 0.60 + sd * 0.18 : 1.0;
    float rr, pp;
    float m = flHead(q, off, sc, uB.x + floor(sd * 2.0), sd, rr, pp);
    if (m > mask) { mask = m; r = rr; pet = pp; seed = sd; }
  }
  core = istep(0.20, 0.06, r);
}
float hgt(vec2 uv) {
  vec2 cid; vec2 q = atCell(uv, cid);
  float mask, r, pet, seed, core;
  flSet(q, cid, mask, r, pet, seed, core);
  return 0.35 + mask * (0.16 + 0.22 * (1.0 - sat(r * 1.6))) + core * 0.16;
}
vec4 atlasRGBA(vec2 uv) {
  vec2 cid; vec2 q = atCell(uv, cid);
  float mask, r, pet, seed, core;
  flSet(q, cid, mask, r, pet, seed, core);
  /* Petals are near-white so material.color can paint them red / pink /
     yellow / white; the throat keeps a little warmth. */
  vec3 c = mix(vec3(0.96, 0.94, 0.92), vec3(0.80, 0.76, 0.74), sat(r * 1.4));
  c = mix(c, vec3(0.88, 0.80, 0.66), istep(0.42, 0.05, r) * 0.7);
  c *= 0.92 + 0.14 * fbm(uv * 60.0, vec2(60.0));
  c = mix(c, vec3(0.94, 0.78, 0.24), core * 0.85);                    // stamens
  c *= 0.90 + 0.18 * hash12(cid + 7.7);
  return vec4(c, mask);
}
`;

/* --------------------------------------------------------------- grass blades */
/* uA=(cols,rows,_,_)  uB=(bladeW, curve, _, _)  uC=(blade.rgb,_) */
FAMILY.grassAtlas = ATLAS_HELPERS + /* glsl */`
float gbBlade(vec2 q, float ox, float curve, float w, float h, out float t, out float rib) {
  t = (q.y + 1.0) / (1.0 + h);
  float cx = ox + curve * t * t;
  float ww = w * (1.0 - 0.85 * t) * (0.55 + 0.45 * (1.0 - t));
  rib = 1.0 - sat(abs(q.x - cx) / max(ww, 1e-4));
  float m = istep(atAA(), -atAA(), abs(q.x - cx) - ww);
  return m * step(0.0, t) * step(t, 1.0);
}
void gbSet(vec2 q, vec2 cid, out float mask, out float t, out float rib, out float seed) {
  seed = hash12(cid + 4.3);
  mask = 0.0; t = 0.0; rib = 0.0;
  float n = (cid.y > 0.5) ? 5.0 : 1.0;
  for (int i = 0; i < 5; i++) {
    if (float(i) >= n) break;
    float fi = float(i);
    float sd = hash12(cid + fi * 3.1 + 0.7);
    float ox = (n > 1.5) ? (fi - 2.0) * 0.13 + (sd - 0.5) * 0.10 : (sd - 0.5) * 0.10;
    float cv = (n > 1.5) ? (fi - 2.0) * 0.26 + (sd - 0.5) * 0.30 : (sd - 0.5) * uB.y * 2.0;
    float w  = uB.x * (0.7 + 0.6 * sd);
    // Keep the tip inside the cell: t reaches 1 at q.y == hh, and q.y tops out
    // at 1, so hh must stay below 1 or the blade is clipped flat at the border.
    float hh = (n > 1.5) ? 0.45 + sd * 0.45 : 0.70 + sd * 0.22;
    float tt, rr;
    float m = gbBlade(q, ox, cv, w, hh, tt, rr);
    if (m > mask) { mask = m; t = tt; rib = rr; seed = sd; }
  }
}
float hgt(vec2 uv) {
  vec2 cid; vec2 q = atCell(uv, cid);
  float mask, t, rib, seed;
  gbSet(q, cid, mask, t, rib, seed);
  return 0.4 + mask * rib * 0.35;
}
vec4 atlasRGBA(vec2 uv) {
  vec2 cid; vec2 q = atCell(uv, cid);
  float mask, t, rib, seed;
  gbSet(q, cid, mask, t, rib, seed);
  /* Per-BLADE hue, not just per-blade value: a tuft of identical saturated
     green spikes reads as neon plastic. Real turf runs olive to straw to a
     cold blue-green within one clump. */
  // Same albedo scale as FAMILY.grass — a tuft measurably darker than the ground
  // it sits on reads as a dark spike, which is what the old 0.061 linear did.
  float hv = hash12(vec2(seed * 37.0, cid.x * 5.3 + cid.y * 11.7));
  vec3 base = uC.rgb;
  base = mix(base, vec3(0.397, 0.403, 0.215), smoothstep(0.20, 0.62, hv));        // olive
  base = mix(base, vec3(0.533, 0.479, 0.267), smoothstep(0.74, 1.00, hv) * 0.85); // straw
  base = mix(base, vec3(0.253, 0.389, 0.288), istep(0.28, 0.00, hv) * 0.70);      // blue-green
  vec3 root = base * 0.58;
  vec3 tip  = base * 1.18 + vec3(0.05, 0.05, 0.015);
  vec3 c = mix(root, base, smoothstep(0.0, 0.45, t));
  c = mix(c, tip, smoothstep(0.45, 1.0, t));
  c = mix(c, c * 1.12, rib * 0.5);
  c = mix(c, vec3(0.639, 0.571, 0.294), smoothstep(0.80, 1.0, seed) * smoothstep(0.5, 1.0, t) * 0.6);
  c *= 0.88 + 0.24 * hash12(cid + 6.1);
  return vec4(c, mask);
}
`;

/* ========================================================================== */
/* The set table                                                               */
/* ========================================================================== */

/**
 * tier   -> resolution class (see planSizes)
 * relief -> peak-to-peak height of the field in METRES; drives normal strength
 * uA/uB/uC -> family parameters (see each family's header comment)
 */
const SETS = [
  /* ---- hero ---- */
  { key: 'cobble', family: 'cobble', tier: 'hero', worldScale: 2.0, relief: 0.036,
    uA: [12, 0.0, 0.55, 0.055], uB: [0, 0.55, 1.0, 0] },
  // straws: 96 per uv tile. The normal pass for a hero set is base/2 texels, so
  // that is ~5 texels per straw — above the Sobel's Nyquist. At the old 220 the
  // kernel straddled a whole period, the gradient cancelled, and the roofs came
  // out smooth and olive. wads: 16 per tile = 6 fibres each, the scale that
  // survives mip-mapping and still reads at 20 m.
  { key: 'thatch', family: 'thatch', tier: 'hero', worldScale: 1.6, relief: 0.056,
    uA: [96, 4, 0.45, 1.0], uB: [0.55, 0.60, 16, 0], directional: 'fibres along V (V runs down the pitch)' },
  { key: 'plaster', family: 'plaster', tier: 'hero', worldScale: 2.5, relief: 0.008,
    uA: [0.18, 0.55, 0.55, 0.35], uB: [0.35, 0, 0, 0] },
  // uC.rgb is the sRGB-encoded heartwood tone. At the old (0.290, 0.208, 0.142)
  // the set measured 0.038 linear — dark oak, but so dark that none of the baked
  // grain was visible. (0.372, 0.268, 0.184) is ~0.065 linear: still clearly
  // dark oak, with the ring figure and the adze marks actually readable.
  { key: 'timber', family: 'wood', tier: 'hero', worldScale: 1.2, relief: 0.011,
    uA: [7, 0.0, 0.0, 0.55], uB: [1.0, 0.55, 0.0, 0.25],
    uC: [0.372, 0.268, 0.184, 0.42] , directional: 'grain along U'},

  /* ---- mid ---- */
  { key: 'cobbleWorn', family: 'cobble', tier: 'mid', worldScale: 2.2, relief: 0.026,
    uA: [11, 1.0, 0.20, 0.042], uB: [0, 0.80, 0.55, 0] },
  // grime 0.75 -> 0.52 and cracks 0.95 -> 0.82: with the palette dropped to a
  // realistic limewash the old grime load put this a fifth below `plaster`
  // instead of just under it.
  { key: 'plasterWorn', family: 'plaster', tier: 'mid', worldScale: 2.5, relief: 0.010,
    uA: [0.52, 0.82, 0.70, 0.50], uB: [0.10, 0, 0, 0] },
  { key: 'brick', family: 'brick', tier: 'mid', worldScale: 1.35, relief: 0.014,
    uA: [6, 18, 0.010, 0.011], uB: [0.55, 0.65, 0.35, 0] },
  { key: 'stone', family: 'stone', tier: 'mid', worldScale: 2.2, relief: 0.030,
    uA: [5, 4, 0.85, 0.035], uB: [0.75, 0.6, 0.8, 0] },
  // DRY SAWN SOFTWOOD is 0.15-0.22 linear — the single biggest miss in the
  // library. At the old (0.400, 0.288, 0.188) this baked at 0.075, half of
  // `stone` and barely above `soil`, so a stack of crates standing in the inn's
  // arcade (shade, no direct sun) collapsed into one featureless dark block.
  // (0.536, 0.386, 0.252) is ~0.145 linear: boards, gaps and nail heads all read
  // in shade, and the brightest medullary fleck still lands short of clipping.
  { key: 'woodPlank', family: 'wood', tier: 'mid', worldScale: 1.0, relief: 0.009,
    uA: [6, 1.0, 6, 0.0], uB: [1.0, 0.45, 1.0, 0.55],
    uC: [0.561, 0.404, 0.264, 0.42] , directional: 'boards + grain along V'},
  // Oak beams are 0.09-0.13 linear. Ordering that must hold: timber (0.0615,
  // deliberately dark oak) < woodBeam < woodPlank.
  { key: 'woodBeam', family: 'wood', tier: 'mid', worldScale: 1.4, relief: 0.013,
    uA: [6, 0.0, 0.0, 0.85], uB: [1.0, 0.60, 0.0, 0.35],
    uC: [0.446, 0.333, 0.216, 0.40] , directional: 'grain along U'},
  { key: 'roofTile', family: 'roofTile', tier: 'mid', worldScale: 1.4, relief: 0.050,
    uA: [7, 5, 0, 0], uB: [0.55, 0.7, 0, 0] , directional: 'courses along V (V runs down the pitch)'},
  { key: 'terracotta', family: 'terracotta', tier: 'mid', worldScale: 0.6, relief: 0.004,
    uA: [9, 0, 0.25, 0], uB: [0.55, 0.45, 0, 0] , directional: 'throwing rings across V (V is the pot axis)'},
  { key: 'soil', family: 'soil', tier: 'mid', worldScale: 1.2, relief: 0.020,
    uA: [14, 0.55, 0.45, 0], uB: [0.5, 0, 0, 0] },
  // worldScale matches terrain's meadow tile (TILE.grass) so the baked normal
  // strength is the real physical slope. See the tiling note in FAMILY.grass.
  { key: 'grass', family: 'grass', tier: 'mid', worldScale: 4.6, relief: 0.024,
    uA: [9, 1.0, 0.30, 0], uB: [0.20, 0.0, 0, 0] },
  // Tree bark is 0.07-0.11 linear. The old (0.300, 0.242, 0.182) baked at 0.049,
  // which made every trunk in the village a silhouette against the lifted ground
  // and threw away the whole plate-and-fissure relief the family bakes.
  { key: 'bark', family: 'bark', tier: 'mid', worldScale: 1.1, relief: 0.028,
    uA: [7, 3, 0.55, 0], uB: [1.0, 0, 0, 0], uC: [0.366, 0.295, 0.222, 0], directional: 'furrows along V' },

  /* ---- low ---- */
  { key: 'iron', family: 'metal', tier: 'low', worldScale: 0.5, relief: 0.004,
    uA: [0.0, 7, 0.35, 0.6], uB: [0.52, 0, 0, 0], uC: [0.160, 0.152, 0.146, 0] },
  { key: 'copper', family: 'metal', tier: 'low', worldScale: 0.6, relief: 0.003,
    uA: [0.80, 6, 0.0, 0.3], uB: [0.36, 0, 0, 0], uC: [0.560, 0.340, 0.212, 0] },
  { key: 'brass', family: 'metal', tier: 'low', worldScale: 0.4, relief: 0.003,
    uA: [0.0, 8, 0.06, 0.25], uB: [0.32, 0, 0, 0], uC: [0.680, 0.540, 0.240, 0] },
  { key: 'woodPainted', family: 'wood', tier: 'low', worldScale: 1.0, relief: 0.009,
    uA: [6, 1.0, 6, 0.0], uB: [1.0, 0.40, 1.0, 0.0],
    uC: [0.400, 0.288, 0.188, 0.42], uD: [1, 0, 0, 0] , directional: 'boards + grain along V'},
  { key: 'fabric', family: 'fibre', tier: 'low', worldScale: 0.8, relief: 0.002,
    uA: [0.0, 90, 8, 0], uB: [0, 0, 0, 0], uC: [0.520, 0.170, 0.145, 0] , directional: 'stripes along U'},
  { key: 'rope', family: 'fibre', tier: 'low', worldScale: 0.25, relief: 0.006,
    uA: [1.0, 0, 0, 0], uB: [0, 0, 0, 0], uC: [0.560, 0.462, 0.300, 0] , directional: 'lay along V'},
  { key: 'glass', family: 'glass', tier: 'low', worldScale: 1.0, relief: 0.0015,
    uA: [0.35, 0.25, 0, 0], uB: [0, 0, 0, 0] },

  /* ---- interiors ----
   * Two cheap tiers of their own. Eleven buildings' worth of interior surface is
   * a lot of new keys, and the exterior library already eats 80 of the 92 MB
   * budget at 'high'; if these went in at 'mid' the planner would shrink the
   * WHOLE library to 768 and drop the resolution of every calibrated hero set.
   * They can afford it: an interior is one small window's worth of light, seen
   * at 2-4 m, and never at the 30 m where a repeat becomes obvious. */
  // 6 boards over a 1.6 m tile = 0.267 m: real wide oak boards. Grain runs along
  // the board (V), nails every 0.53 m, uD.y puts the traffic polish along it.
  //
  // ringFreq (uA.x) is 26, not the 9 this shipped with. wdGrain adds up to ~3.5
  // ring periods of domain warp to the ring coordinate, so the grain wanders by
  // 3.5/ringFreq of a board's width: at 9 that is 39% and the boards read as
  // swirling marble from the 1-3 m an interior floor is actually seen from. At 26
  // it is 13% — long straight lines with a slight waver, which is what
  // quarter-sawn oak looks like. 26 rings still lands ~10 texels apart in the
  // interior tier's normal pass, above the Sobel's Nyquist (see the thatch note).
  { key: 'floorBoard', family: 'wood', tier: 'interior', worldScale: 1.6, relief: 0.010,
    uA: [26, 1.0, 6, 0.0], uB: [1.0, 0.42, 1.0, 0.30],
    uC: [0.443, 0.338, 0.232, 0.44], uD: [0, 0.85, 0, 0],
    directional: 'boards + grain along V' },
  // 3 x 4 cells over 2.4 m = flags roughly 0.8 x 0.6 m, jitter 0.85 so no two
  // are the same shape.
  { key: 'flagstone', family: 'flagstone', tier: 'interior', worldScale: 2.4, relief: 0.026,
    uA: [3, 4, 0.85, 0.030], uB: [0.85, 0.35, 0.45, 0] },
  // 72 laths over 3.2 m = 44 mm pitch, ~5.4 texels each in the 384 normal pass —
  // still above the Sobel's Nyquist (see the thatch note; below ~5 it cancels flat).
  //
  // flaking (uA.y) is 0.10, down from 0.45. lpFlake is a VORONOI field, so at 0.45
  // the distemper scabs joined up into a network of dark irregular polygons and the
  // walls read as crazed dried mud rather than limewash — the single worst thing
  // about the interiors on first look. Cobweb comes down with it. worldScale 2.5 ->
  // 3.2 also pushes the 13-cell voronoi period out to ~25 cm and the tile repeat
  // past the width of most wall panels, so the eye stops finding the pattern.
  { key: 'lathPlaster', family: 'lathPlaster', tier: 'interior', worldScale: 3.2, relief: 0.009,
    uA: [72, 0.10, 0.32, 0.16], uB: [0.22, 0.045, 0, 0] },
  // ringFreq 22 for the same reason as floorBoard: at 6 the warp moved the grain
  // across 58% of the joist and the ceiling read as dark swirling liquid under
  // the hearth light. Joists are seen closer than a floor and almost always in
  // raking firelight, so this is the surface the swirl was most obvious on.
  { key: 'ceilingBeam', family: 'wood', tier: 'interiorLow', worldScale: 1.4, relief: 0.014,
    uA: [22, 0.0, 0.0, 0.95], uB: [1.0, 0.55, 0.0, 0.15],
    uC: [0.385, 0.278, 0.191, 0.44], uD: [0, 0, 1.0, 0.35],
    directional: 'grain along U' },
  { key: 'hearthStone', family: 'hearthStone', tier: 'interiorLow', worldScale: 1.2, relief: 0.016,
    uA: [3, 5, 0.012, 0.016], uB: [0.65, 0.55, 0.45, 0] },
  { key: 'soot', family: 'soot', tier: 'interiorLow', worldScale: 0.8, relief: 0.004,
    uA: [0.9, 0.45, 0.35, 0] },
  // uD carries the ground colour (undyed flax) and uD.w the fraying; uC is only
  // a hair different so the woven stripe reads without becoming a ticking.
  { key: 'linen', family: 'fibre', tier: 'interiorLow', worldScale: 0.5, relief: 0.0028,
    uA: [0.0, 40, 1, 0], uC: [0.620, 0.598, 0.545, 0],
    uD: [0.665, 0.640, 0.585, 0.55], directional: 'weave' },
  { key: 'sackcloth', family: 'fibre', tier: 'interiorLow', worldScale: 0.6, relief: 0.0055,
    uA: [0.0, 34, 1, 0], uC: [0.442, 0.372, 0.250, 0],
    uD: [0.480, 0.408, 0.278, 1.0], directional: 'weave' },
  { key: 'strawLitter', family: 'strawLitter', tier: 'interiorLow', worldScale: 1.0, relief: 0.020,
    uA: [40, 0, 0.35, 0], uB: [0.45, 0.30, 0, 0], directional: 'straws along U' },
  // Pewter is a real metal: metalness 1 out of the ORM, roughness 0.37-0.47, and
  // its albedo only means anything as a specular colour.
  { key: 'pewter', family: 'metal', tier: 'interiorLow', worldScale: 0.35, relief: 0.0025,
    uA: [0.0, 9, 0.0, 0.30], uB: [0.42, 0, 0, 0], uC: [0.790, 0.780, 0.760, 0] },

  /* ---- alpha atlases ---- */
  // FOLIAGE ALBEDO. Broadleaf is 0.09-0.14 linear and conifer needle 0.06-0.10 —
  // leaves are far more reflective than they look, because a lit canopy is mostly
  // transmission. These uC values are measured over the texels that pass
  // alphaTest (0.42), which is the only average that means anything on a cut-out
  // card: the RGB under a transparent texel is never shaded.
  { key: 'leaf', family: 'leafAtlas', tier: 'atlas', worldScale: 1, relief: 0.010,
    atlas: [4, 2], uA: [4, 2, 0, 0], uB: [3.0, 0.0, 1, 0], uC: [0.221, 0.359, 0.133, 0] , directional: 'petiole at V=0'},
  // 2 x 2 — square cells (a 4 x 2 cell is twice as tall as it is wide, which
  // stretched every leaf on the square cards the buildings stream uses). Each
  // cell is a cluster of 5-9 leaves; uB = (lobeK, nMin, nExtra, veinStrength).
  { key: 'ivy', family: 'ivyAtlas', tier: 'atlas', worldScale: 1, relief: 0.012,
    atlas: [2, 2], uA: [2, 2, 0, 0], uB: [2.5, 5.0, 4.0, 1.0], uC: [0.220, 0.410, 0.193, 0], directional: 'cluster, no single up axis'},
  // needle was the darkest surface in the whole world at 0.032 linear — darker
  // than wrought iron — so every conifer read as a flat black cut-out.
  { key: 'needle', family: 'needleAtlas', tier: 'atlas', worldScale: 1, relief: 0.006,
    atlas: [2, 2], uA: [2, 2, 0, 0], uB: [26.0, 1.05, 0, 0], uC: [0.171, 0.308, 0.186, 0] , directional: 'stem along V'},
  { key: 'flower', family: 'flowerAtlas', tier: 'atlas', worldScale: 1, relief: 0.006,
    atlas: [4, 2], uA: [4, 2, 0, 0], uB: [5.0, 0, 0, 0] , directional: 'none'},
  { key: 'grassBlade', family: 'grassAtlas', tier: 'atlas', worldScale: 1, relief: 0.006,
    atlas: [4, 2], uA: [4, 2, 0, 0], uB: [0.13, 0.35, 0, 0], uC: [0.335, 0.432, 0.226, 0] , directional: 'base at V=0'},
];

/* Fractions of the planning base size, per tier: [albedo, normal, orm]. */
const TIER_FRACTION = {
  hero:  [1.0, 0.5, 0.5],
  mid:   [0.5, 0.5, 0.25],
  low:   [0.25, 0.25, 0.25],
  atlas: [0.5, 0.25, 0.0],
  // Interiors: 0.375 = 384 at base 1024, i.e. 240 texels/m on a floor. Ten new
  // sets at 'mid' would have cost 30 MB and forced the planner to shrink the
  // whole library; these two tiers cost 10.3 MB together.
  interior:    [0.375, 0.375, 0.1875],
  interiorLow: [0.25,  0.25,  0.125],
};

// 'ultra' went 200 -> 224 purely so the interior sets do not push the planner off
// its 1536 base and quietly halve the resolution of the calibrated hero sets.
const BUDGET_MB = { low: 26, medium: 92, high: 92, ultra: 224 };

/* ========================================================================== */
/* Bakery                                                                      */
/* ========================================================================== */

/* WebGL2 filters and mips NPOT textures fine, so sizes only have to be a
 * multiple of 64 — that lets the planner land on 1536 instead of jumping
 * straight from 2048 to 1024. */
const snap = (n) => Math.max(64, Math.min(2048, Math.round(n / 64) * 64));
const mib = (size) => (size * size * 4 * 4) / 3 / 1048576; // +33% for the mip chain

/** Choose per-tier resolutions and shrink until the estimate fits the budget. */
function planSizes(textureSize, budget) {
  let base = snap(textureSize);
  for (let guard = 0; guard < 8; guard++) {
    const sizes = {};
    for (const t of Object.keys(TIER_FRACTION)) {
      sizes[t] = TIER_FRACTION[t].map((f) => (f > 0 ? snap(base * f) : 0));
    }
    let total = 0;
    for (const s of SETS) {
      const [a, n, o] = sizes[s.tier];
      total += mib(a) + (n ? mib(n) : 0) + (o ? mib(o) : 0);
    }
    if (total <= budget || base <= 192) return { base, sizes, estimateMB: total };
    base = snap(base * 0.75);
  }
  return null;
}

/**
 * Build the whole procedural texture library.
 *
 * @param {THREE.WebGLRenderer} renderer
 * @param {{quality:Object, anisotropy:number}} opts
 * @returns {Promise<import('../contracts.js').TextureLibrary>}
 */
export async function createTextureLibrary(renderer, { quality, anisotropy } = {}) {
  const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const qname = quality?.name || 'high';
  const textureSize = quality?.textureSize || 1024;
  const budget = BUDGET_MB[qname] ?? 92;
  const plan = planSizes(textureSize, budget) || { base: 256, sizes: null, estimateMB: 0 };

  const maxAniso = renderer?.capabilities?.getMaxAnisotropy?.() ?? 8;
  const aniso = Math.max(1, Math.min(anisotropy || 8, maxAniso));

  /** @type {Map<string, Object>} */
  const sets = new Map();
  const owned = [];
  const stats = {
    baked: 0, textures: 0, memoryMB: 0, cpuMB: 0,
    bakeMs: 0, base: plan.base, anisotropy: aniso, failed: [],
  };

  if (!renderer || !plan.sizes) {
    console.warn('[textures] no renderer or no viable size plan — using flat fallbacks');
    for (const s of SETS) sets.set(s.key, flatSet(s, aniso));
    return finish();
  }

  /* ---- bake rig -------------------------------------------------------- */
  const quadGeo = new THREE.PlaneGeometry(2, 2);
  const quad = new THREE.Mesh(quadGeo, null);
  quad.frustumCulled = false;
  const bakeScene = new THREE.Scene();
  bakeScene.add(quad);
  const bakeCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);

  const targets = new Map();
  const getTarget = (size) => {
    let rt = targets.get(size);
    if (!rt) {
      rt = new THREE.WebGLRenderTarget(size, size, {
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
        colorSpace: THREE.NoColorSpace,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        depthBuffer: false,
        stencilBuffer: false,
        generateMipmaps: false,
      });
      targets.set(size, rt);
    }
    return rt;
  };

  const prevTarget = renderer.getRenderTarget();
  const prevAutoClear = renderer.autoClear;
  const prevClear = new THREE.Color();
  renderer.getClearColor(prevClear);
  const prevClearAlpha = renderer.getClearAlpha();
  renderer.autoClear = true;
  renderer.setClearColor(0x000000, 1);

  const programs = new Map();

  try {
    let n = 0;
    for (const spec of SETS) {
      try {
        sets.set(spec.key, bakeSet(spec));
      } catch (err) {
        console.warn(`[textures] "${spec.key}" failed to bake:`, err);
        stats.failed.push(spec.key);
        sets.set(spec.key, flatSet(spec, aniso));
      }
      // Let the loading bar breathe every few sets; a 1 s frozen tab reads as a hang.
      if ((++n & 3) === 0) await new Promise((r) => setTimeout(r, 0));
    }
  } finally {
    for (const m of programs.values()) m.dispose();
    for (const rt of targets.values()) rt.dispose();
    quadGeo.dispose();
    bakeScene.remove(quad);
    renderer.setRenderTarget(prevTarget);
    renderer.autoClear = prevAutoClear;
    renderer.setClearColor(prevClear, prevClearAlpha);
    renderer.info?.reset?.();
  }

  return finish();

  /* ---------------------------------------------------------------------- */

  function material(spec) {
    const key = spec.family;
    let m = programs.get(key);
    if (!m) {
      const isAtlas = spec.tier === 'atlas';
      m = new THREE.ShaderMaterial({
        vertexShader: VERTEX,
        fragmentShader: PRELUDE + FAMILY[key] + (isAtlas ? ATLAS_MAIN : SURFACE_MAIN),
        uniforms: {
          uPass: { value: 0 },
          uTexel: { value: 1 / 1024 },
          uNormal: { value: 1 },
          uA: { value: new THREE.Vector4() },
          uB: { value: new THREE.Vector4() },
          uC: { value: new THREE.Vector4() },
          uD: { value: new THREE.Vector4() },
        },
        depthTest: false,
        depthWrite: false,
      });
      programs.set(key, m);
    }
    m.uniforms.uA.value.fromArray(spec.uA || [0, 0, 0, 0]);
    m.uniforms.uB.value.fromArray(spec.uB || [0, 0, 0, 0]);
    m.uniforms.uC.value.fromArray(spec.uC || [0, 0, 0, 0]);
    m.uniforms.uD.value.fromArray(spec.uD || [0, 0, 0, 0]);
    return m;
  }

  function renderPass(mat, size, pass, normalStrength) {
    const rt = getTarget(size);
    mat.uniforms.uPass.value = pass;
    mat.uniforms.uTexel.value = 1 / size;
    mat.uniforms.uNormal.value = normalStrength;
    quad.material = mat;
    renderer.setRenderTarget(rt);
    renderer.clear(true, false, false);
    renderer.render(bakeScene, bakeCam);
    const buf = new Uint8Array(size * size * 4);
    renderer.readRenderTargetPixels(rt, 0, 0, size, size, buf);
    stats.cpuMB += buf.byteLength / 1048576;
    return buf;
  }

  function makeTexture(buf, size, srgb, wrap) {
    const t = new THREE.DataTexture(buf, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.wrapS = t.wrapT = wrap;
    t.magFilter = THREE.LinearFilter;
    t.minFilter = THREE.LinearMipmapLinearFilter;
    t.generateMipmaps = true;
    t.anisotropy = aniso;
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.flipY = false;             // GL rows already come back bottom-up == v=0
    t.needsUpdate = true;
    owned.push(t);
    stats.textures++;
    stats.memoryMB += mib(size);
    // Force the upload now so the first frame is not a stutter.
    renderer.initTexture?.(t);
    return t;
  }

  function bakeSet(spec) {
    const [aSize, nSize, oSize] = plan.sizes[spec.tier];
    const mat = material(spec);
    const wrap = spec.tier === 'atlas' ? THREE.ClampToEdgeWrapping : THREE.RepeatWrapping;
    const ws = spec.worldScale || 1;

    const albedo = makeTexture(renderPass(mat, aSize, 0, 1), aSize, true, wrap);

    // Physically consistent slope: relief(m) per texel(m). Independent of the
    // chosen resolution except that a finer texel resolves a steeper local slope.
    const strength = Math.min(64, (spec.relief * nSize) / ws);
    const normal = makeTexture(renderPass(mat, nSize, 1, strength), nSize, false, wrap);

    let orm = null;
    if (oSize > 0) orm = makeTexture(renderPass(mat, oSize, 2, 1), oSize, false, wrap);

    stats.baked++;
    return {
      map: albedo,
      normalMap: normal,
      roughnessMap: orm,
      metalnessMap: orm,
      aoMap: orm,
      ormMap: orm,
      alphaMap: null,          // cut-outs carry alpha in map.a — use alphaTest
      hasAlphaInMap: spec.tier === 'atlas',
      worldScale: ws,
      relief: spec.relief,
      atlas: spec.atlas ? { cols: spec.atlas[0], rows: spec.atlas[1] } : null,
      directional: spec.directional || null,
      sizes: [aSize, nSize, oSize],
    };
  }

  function finish() {
    stats.bakeMs = Math.round(((typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0) * 10) / 10;
    stats.memoryMB = Math.round(stats.memoryMB * 10) / 10;
    stats.cpuMB = Math.round(stats.cpuMB * 10) / 10;
    stats.estimateMB = Math.round((plan.estimateMB || 0) * 10) / 10;

    const missing = TEXTURE_KEYS.filter((k) => !sets.has(k));
    if (missing.length) console.warn('[textures] contract keys missing:', missing);

    return {
      get(name) {
        const s = sets.get(name);
        if (!s) {
          warnMissing(name);
          return sets.get('plaster') || null;
        }
        return s;
      },
      has: (name) => sets.has(name),
      keys: () => [...sets.keys()],
      dispose() {
        for (const t of owned) t.dispose();
        owned.length = 0;
        sets.clear();
      },
      stats,
    };
  }
}

/* -------------------------------------------------------------------------- */

const warnedTextures = new Set();
function warnMissing(name) {
  if (warnedTextures.has(name)) return;
  warnedTextures.add(name);
  console.warn(`[textures] unknown texture set "${name}" — substituting plaster`);
}

/** 4x4 flat stand-ins, so a GPU failure degrades to "ugly" rather than "dead". */
function flatSet(spec, aniso) {
  const size = 4;
  const px = size * size;
  const alb = new Uint8Array(px * 4);
  const nrm = new Uint8Array(px * 4);
  const orm = new Uint8Array(px * 4);
  for (let i = 0; i < px; i++) {
    alb[i * 4] = 150; alb[i * 4 + 1] = 140; alb[i * 4 + 2] = 125; alb[i * 4 + 3] = 255;
    nrm[i * 4] = 128; nrm[i * 4 + 1] = 128; nrm[i * 4 + 2] = 255; nrm[i * 4 + 3] = 255;
    orm[i * 4] = 255; orm[i * 4 + 1] = 200; orm[i * 4 + 2] = 0; orm[i * 4 + 3] = 255;
  }
  const mk = (data, srgb) => {
    const t = new THREE.DataTexture(data, size, size, THREE.RGBAFormat, THREE.UnsignedByteType);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.magFilter = THREE.LinearFilter;
    t.minFilter = THREE.LinearFilter;
    t.generateMipmaps = false;
    t.anisotropy = aniso;
    t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    t.flipY = false;
    t.needsUpdate = true;
    return t;
  };
  const o = mk(orm, false);
  return {
    map: mk(alb, true),
    normalMap: mk(nrm, false),
    roughnessMap: o,
    metalnessMap: o,
    aoMap: o,
    ormMap: o,
    alphaMap: null,
    hasAlphaInMap: false,
    worldScale: spec.worldScale || 1,
    relief: spec.relief || 0,
    atlas: spec.atlas ? { cols: spec.atlas[0], rows: spec.atlas[1] } : null,
    directional: spec.directional || null,
    sizes: [size, size, size],
    fallback: true,
  };
}

/** The atlas layouts, published so the vegetation stream can pick a cell. */
export const ATLAS_LAYOUT = {
  leaf: { cols: 4, rows: 2, note: 'row 0 = single broadleaf, row 1 = 3-leaf sprig' },
  // Every cell is a CLUSTER of 5-9 small palmate leaves — there is no
  // single-leaf cell. A card sampling one cell wants to be ~0.30 m; a card
  // sampling the whole 0..1 atlas (four clusters) wants to be ~0.60 m.
  ivy: { cols: 2, rows: 2, note: 'all four cells are 5-9 leaf clusters; no up axis' },
  needle: { cols: 2, rows: 2, note: 'conifer sprigs, stem along +V' },
  flower: { cols: 4, rows: 2, note: 'row 0 = single blossom, row 1 = 3-head cluster; near-white, tint with material.color' },
  grassBlade: { cols: 4, rows: 2, note: 'row 0 = single blade, row 1 = 5-blade tuft; base at V=0' },
};

/** UV helper: bottom-left corner + size of one atlas cell. Row 0 is at V=0. */
export function atlasCell(layout, col, row) {
  const c = ((col % layout.cols) + layout.cols) % layout.cols;
  const r = ((row % layout.rows) + layout.rows) % layout.rows;
  return {
    offset: [c / layout.cols, r / layout.rows],
    repeat: [1 / layout.cols, 1 / layout.rows],
  };
}
