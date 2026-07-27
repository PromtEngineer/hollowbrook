Original prompt: refine and improve this, maintain or improve quality, reduce unneeded renders, improve framewrate and textures, use blender skills to massively improve and enhance this game

## 2026-07-27

- Began from the live `2389491` worktree; it was clean.
- Confirmed Hollowbrook is a deterministic, asset-free Three.js/Rapier village with adaptive quality, generated PBR texture sets, instanced repetition, and room culling.
- Current focus: establish visual/performance baselines, identify repeated frame work and close-range material weaknesses, then apply Blender-informed geometry/material/export decisions only where they improve the shipped web game.
- Baseline at `?quality=high`, 1440x900 on the Apple M3 Max: 508 submitted draws, 2,426,450 submitted triangles, about 60 FPS. GTAO alone repeated 236 draws / 1,200,132 triangles and measured about 4.7 ms GPU.
- Implemented room-aware GTAO: requested by High/Ultra but physically inserted only indoors. Outdoor result is 272 draws / 1,226,318 triangles, with the same clean scene audit and 60 FPS; GTAO returns automatically in a room.
- Bloom now runs its deliberately blurred pyramid at half resolution on High; full-screen appearance is preserved while avoiding full-resolution blur fill.
- Hidden tabs skip rendering, and the settings overlay renders the live background at 15 Hz. Browser instrumentation observed 14 paints over 1.1 seconds while paused.
- Recalibrated cobble, exterior oak/planks, and thatch albedo/relief. Removed repeated eye-shaped wood figure and strong baked-in colour shading; reduced inked mortar and thatch course bands.
- Ran a Blender 5 material/silhouette/export study for oak, thatch, and curved well masonry: 1,190 triangles, 86.2 KB GLB. Applied its compressed eave turn-under proportions back to the procedural roof mesh without adding runtime assets.
- Added `window.render_game_to_text()` for concise live state and automated checks.

## TODO

- Completed matched arrival, cottage, thatch, and stonework captures after the
  eave silhouette pass; assembled side-by-side comparison images in `output/`.
- Final high-quality browser checks passed walking, sprinting, jumping,
  crouching, lantern, performance panel, settings pause/resume, text state,
  indoor/outdoor AO transition, and the scene audit with no subsystem failures.
- Stable outdoor samples remained display-capped at about 60 FPS. The measured
  post chain fell from about 15.82 ms to 9.60 ms GPU in the comparable pass
  profile, while submitted work fell by 236 draws and 1,200,132 triangles.
- The supplied web-game client was also run in both headless and headed modes.
  Its isolated Chromium either timed out before `DOMContentLoaded` or found the
  Enter button while the loading overlay still hid it; the persistent
  Playwright browser loaded and exercised the same URL successfully.
- `npm run build` and `git diff --check` pass after the final review.
