/**
 * Thin wrapper over the DOM overlay. Passed to the controls and performance
 * streams so they never query the document directly.
 */

export function createHud() {
  const el = {
    hud: document.getElementById('hud'),
    crosshair: document.getElementById('crosshair'),
    prompt: document.getElementById('prompt'),
    carrying: document.getElementById('carrying'),
    perf: document.getElementById('perf'),
    loading: document.getElementById('loading'),
    loadBar: document.getElementById('load-bar'),
    loadTask: document.getElementById('load-task'),
    loadError: document.getElementById('load-error'),
    start: document.getElementById('start'),
    startBtn: document.getElementById('start-btn'),
    menu: document.getElementById('menu'),
    resumeBtn: document.getElementById('resume-btn'),
  };

  let lastPrompt = null;
  let lastCarry = null;

  return {
    el,

    progress(fraction, task) {
      if (el.loadBar) el.loadBar.style.width = `${Math.round(fraction * 100)}%`;
      if (task && el.loadTask) el.loadTask.textContent = task;
    },

    fail(message) {
      if (!el.loadError) return;
      el.loadError.textContent = String(message?.stack || message);
      el.loadError.classList.remove('hidden');
      if (el.loadTask) el.loadTask.textContent = 'failed to enter the village';
    },

    finishLoading() {
      el.loading?.classList.add('fade');
      setTimeout(() => el.loading?.classList.add('hidden'), 750);
      el.start?.classList.remove('hidden');
    },

    showHud(on) { el.hud?.classList.toggle('hidden', !on); },

    setPrompt(text) {
      if (text === lastPrompt) return;
      lastPrompt = text;
      if (!el.prompt) return;
      el.prompt.textContent = text || '';
      el.prompt.classList.toggle('show', !!text);
      el.crosshair?.classList.toggle('active', !!text);
    },

    setCarrying(text) {
      if (text === lastCarry) return;
      lastCarry = text;
      if (!el.carrying) return;
      el.carrying.textContent = text ? `carrying ${text} — [E] drop · [LMB] throw` : '';
      el.carrying.classList.toggle('show', !!text);
    },

    setPerf(text) {
      if (el.perf) el.perf.textContent = text;
    },

    togglePerf(force) {
      const hidden = el.perf?.classList.toggle('hidden', force === undefined ? undefined : !force);
      return !hidden;
    },
  };
}
