/**
 * Deterministic seeded randomness. Every generator in the project must take an
 * Rng instance (or a seed) so that the world is byte-identical between reloads —
 * a defect you can't reproduce is a defect you can't fix.
 */

/** mulberry32 — fast, good enough distribution for content generation. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Hash a string into a 32-bit seed (FNV-1a). */
export function hashSeed(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

export class Rng {
  constructor(seed = 1337) {
    this.seed = typeof seed === 'string' ? hashSeed(seed) : seed >>> 0;
    this._next = mulberry32(this.seed);
  }
  /** [0,1) */
  next() { return this._next(); }
  /** [min,max) */
  range(min, max) { return min + (max - min) * this._next(); }
  /** integer in [min,max] */
  int(min, max) { return Math.floor(min + (max - min + 1) * this._next()); }
  /** symmetric [-a,a] */
  sym(a) { return (this._next() * 2 - 1) * a; }
  bool(p = 0.5) { return this._next() < p; }
  pick(arr) { return arr[Math.floor(this._next() * arr.length) % arr.length]; }
  /** Fisher–Yates, in place. */
  shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(this._next() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
  /** A child generator, so adding a call in one system can't shift another. */
  fork(tag) { return new Rng((this.seed ^ hashSeed(String(tag))) >>> 0); }
}

/** 2D value noise with smooth interpolation — deterministic, no allocation. */
export function valueNoise2D(seed = 0) {
  const perm = new Uint8Array(512);
  const rnd = mulberry32(seed || 1);
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    const t = p[i]; p[i] = p[j]; p[j] = t;
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];

  const grad = (h, x, y) => {
    switch (h & 3) {
      case 0: return x + y;
      case 1: return -x + y;
      case 2: return x - y;
      default: return -x - y;
    }
  };
  const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10);

  return function noise(x, y) {
    const xi = Math.floor(x) & 255;
    const yi = Math.floor(y) & 255;
    const xf = x - Math.floor(x);
    const yf = y - Math.floor(y);
    const u = fade(xf), v = fade(yf);
    const aa = perm[perm[xi] + yi];
    const ab = perm[perm[xi] + yi + 1];
    const ba = perm[perm[xi + 1] + yi];
    const bb = perm[perm[xi + 1] + yi + 1];
    const x1 = grad(aa, xf, yf) + u * (grad(ba, xf - 1, yf) - grad(aa, xf, yf));
    const x2 = grad(ab, xf, yf - 1) + u * (grad(bb, xf - 1, yf - 1) - grad(ab, xf, yf - 1));
    return (x1 + v * (x2 - x1)) * 0.5; // roughly [-1,1]
  };
}

/** Fractal Brownian motion over a noise function. */
export function fbm2D(noise, x, y, octaves = 4, lacunarity = 2.02, gain = 0.5) {
  let amp = 0.5, freq = 1, sum = 0, norm = 0;
  for (let i = 0; i < octaves; i++) {
    sum += amp * noise(x * freq, y * freq);
    norm += amp;
    amp *= gain;
    freq *= lacunarity;
  }
  return sum / (norm || 1);
}
