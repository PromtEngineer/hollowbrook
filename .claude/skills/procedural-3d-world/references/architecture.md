# Architecture — building a world with parallel agents

The organising problem is not rendering. It is that a world of this size does not fit in one
context, and the subsystems are mutually entangled: geometry needs materials, physics needs
geometry, lighting needs to know what emits light, the furnisher needs to know where the walls
are. Build them in sequence and it takes forever; build them in parallel without a contract and
they do not integrate.

The answer is to make the entanglement **explicit and one-directional**, in a contract written
before any subsystem exists.

---

## The dependency inversions that make it work

Three rules, each of which removes a whole class of cross-module coupling:

**1. Geometry never touches physics.** Modules that build meshes emit *collider descriptions* —
plain data — alongside their `THREE.Group`. The physics module consumes those descriptions and
knows nothing about how the geometry was made.

```js
{ shape: 'box', halfExtents: [1.2, 1.5, 0.3], position: [x, y, z],
  quaternion: [0, 0, 0, 1], friction: 0.85, tag: 'building-wall' }
```

`tag` is not decoration. When a raycast tells you `door-weaver-0` is blocking, you have a
diagnosis; when it tells you "something at 1.62 m", you have a search.

**2. Geometry never creates materials.** It asks a library for one *by name* from a canonical
list, and gets a shared instance back. Variants (vertex colours, double-sided, a tint, a
different UV repeat) come from a cached `variant(name, opts)` so they still batch. An unknown
name returns a loud magenta debug material and warns **once** — so a typo shows up in the world
and in the automated audit rather than throwing at build time.

**3. A producer publishes what it knows; a consumer never re-derives it.** The clearest example:
the module that builds the shells also publishes their interiors as `RoomSpec` — clear interior
dimensions, floor and ceiling heights, and every door and window opening with an inward normal.
Two other modules furnish those rooms without ever recomputing where a wall is.

The corollary is a real defect: when the module that built the staircases published *no* keep-out
volumes, the furnisher guessed for 14 rooms and placed furniture into the stairs. Publish what
you know.

---

## What goes in the contract

Write all of this before dispatching anyone.

| File | Contents | Why first |
|---|---|---|
| `contracts.js` | collider spec, interactable spec, canonical material names, hand-off types, small shared helpers | the only thing every stream imports |
| `layout.js` | the authored plan: every position, rotation, footprint, road, sun bearing | one source of truth for placement |
| `config.js` | quality presets, player tunables, world constants | shared numbers, and the bake/runtime split |
| `main.js` | the integration | fixes each module's signature by *using* it |

**Write `main.js` before the modules it imports.** It is the cheapest way to force the contract
to be concrete: you cannot write the call site without deciding exactly what each factory takes
and returns. Every stream then reads it to see how it will be invoked.

Separate **bake-time** knobs (texture resolution, mesh density, env map size — fixed at load)
from **runtime** knobs (pixel ratio, AO on/off, shadow distance, cull distances). Only the
latter may be touched by an adaptive performance manager. Getting this wrong produces a manager
that tries to change something it cannot.

### The uniform module shape

Every geometry stream returns the same thing, so integration is a loop rather than a special
case per module:

```js
{
  group,           // THREE.Group — the integrator adds it to the scene
  colliders,       // ColliderSpec[]   — static world collision
  interactables,   // InteractableSpec[] — physics bodies, optional
  lightAnchors,    // [{ position, colour, intensity, kind }] — optional
  update(dt, ctx), // optional
  dispose(),       // optional
  stats,           // free-form — this is what you measure later
}
```

`stats` is not an afterthought. It is how every later review pass attributes cost and behaviour;
make each module report its own object counts, triangle counts and any internal state a reviewer
would otherwise have to guess at (which room is occupied, how many LOD instances are live).

---

## Isolation at the seams

Each subsystem is built behind a stage that catches, logs, names the failure, and substitutes an
inert stub:

```js
async function stage(fraction, label, fn, stub) {
  hud.progress(fraction, label);
  await yieldToBrowser();                    // rAF raced against a timer — see below
  try { return await fn(); }
  catch (err) {
    console.error(`stage "${label}" failed:`, err);
    failures.push({ label, err });
    return typeof stub === 'function' ? stub() : stub;
  }
}
```

One broken module then produces a world with a hole in it and a named failure in
`window.MYWORLD.failures`, not a black screen. During a parallel build, where any given module
may be mid-edit, this is the difference between "three streams are still broken" and "nothing
works and I cannot tell why".

Yield between stages by racing `requestAnimationFrame` against a timer — rAF alone stops firing
in a background tab and the load stalls forever if the user switches away.

---

## Dispatching the streams

One agent per stream, each owning its files **exclusively**. Overlapping ownership is the one
thing that reliably destroys work.

Every brief should carry:

- the contract files to read first, in full;
- the reference image or brief, described in words — the agents cannot see it;
- **verified** library API notes (see `three-api-notes.md`), because otherwise several agents
  will independently guess the same wrong thing;
- explicit budgets (draw calls, triangles, memory, build time);
- the house style, so 26k lines don't read like nine different authors;
- an instruction to report **what is still weak** — that list is the first review pass's input,
  and good agents are unnervingly accurate about their own weak spots. One flagged the exact
  albedo values that later turned out to be the project's biggest visual problem.

Ask for structured reports: what was built, the numbers, what is weak, and **what was assumed
about other streams**. That last field catches integration mismatches before they ship.

### What this costs

Be honest about the failure modes, because they are real:

- **Cross-module drift.** Two modules independently hard-code the same constant and one changes.
  Both integration bugs in this project were of this kind (an atlas grid copied instead of
  imported; a value published on one object and read from another). Neither was visible in code
  review; both were found by runtime measurement.
- **Agents may not report.** Several were killed by transient API errors *after* writing their
  files — work on disk, no report. Verify the disk, not the report.
- **Partial landings.** One agent left a comment describing a refactor it had not finished. If a
  claim matters, check the code, not the comment.

---

## The review loop

Integrate, then iterate. Each pass:

1. **Automated audit** for structural defects (see `verification.md`).
2. **Walk it** and capture viewpoints.
3. **Isolate each defect by measurement** before assigning it to a stream — toggle candidates
   off and re-measure. The visible symptom rarely names the responsible subsystem.
4. **Dispatch one fix agent per affected file**, giving it the measured evidence, and an explicit
   list of *what is already good and must not regress*. Without that list, agents helpfully
   "improve" calibrated work.

Five loops was the right number here. Loops 1–3 fixed real defects; 4–5 were polish. The loop
terminates when a full pass produces nothing — not when you run out of ideas.

Two notes on running fix passes:

- **Give agents measurements, not adjectives.** "The ground is too dark" produces guesswork.
  "Ground-half mean screen luminance is 0.243, disabling shadows moves it 4%, baked cobble albedo
  is 0.099 linear against a real-world 0.24–0.34" produces a correct fix on the first attempt.
- **Expect to be corrected, and let it happen.** Instructed to rest doors 50–70° ajar, an agent
  computed that a 1.08 m leaf at 60° in a 1.20 m opening leaves a 0.64 m corridor — exactly the
  player capsule's diameter — and used ~85° instead. The instruction was wrong; the measurement
  was right. Brief for the *goal* and the *constraint*, and let the agent solve the geometry.
