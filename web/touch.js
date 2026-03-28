// VibeBoy — Touch controls module

/**
 * Set up touch controls for mobile devices.
 * @param {object} state - shared emulator state object
 * @param {object} btnConstants - { btn_a, btn_b, btn_up, btn_down, btn_left, btn_right, btn_start, btn_select }
 */
export function setupTouchControls(state, btnConstants) {
  const touchControls = document.getElementById('touch-controls');
  const canvas = document.getElementById('screen');
  const canvas2d = document.getElementById('screen-2d');
  const isTouchDevice = ('ontouchstart' in window) || (navigator.maxTouchPoints > 0);

  if (!isTouchDevice) return;

  touchControls.classList.add('visible');

  // Prevent pull-to-refresh and overscroll
  document.body.style.overscrollBehavior = 'none';
  document.body.style.touchAction = 'none';

  const touchBtnMap = {
    up: btnConstants.btn_up,
    down: btnConstants.btn_down,
    left: btnConstants.btn_left,
    right: btnConstants.btn_right,
    a: btnConstants.btn_a,
    b: btnConstants.btn_b,
    start: btnConstants.btn_start,
    select: btnConstants.btn_select,
  };

  // Track active touches per button element
  const activeTouches = new Map();

  function handleTouchStart(e) {
    e.preventDefault();
    const el = e.target.closest('[data-btn]');
    if (!el) return;
    const btnName = el.dataset.btn;
    const btnId = touchBtnMap[btnName];
    if (btnId === undefined) return;

    for (const t of e.changedTouches) {
      if (!activeTouches.has(el)) activeTouches.set(el, new Set());
      activeTouches.get(el).add(t.identifier);
    }

    el.classList.add('active');
    if (state.emu) state.emu.set_button(btnId, true);
  }

  function handleTouchEnd(e) {
    e.preventDefault();
    for (const t of e.changedTouches) {
      for (const [el, ids] of activeTouches) {
        if (ids.has(t.identifier)) {
          ids.delete(t.identifier);
          if (ids.size === 0) {
            activeTouches.delete(el);
            el.classList.remove('active');
            const btnName = el.dataset.btn;
            const btnId = touchBtnMap[btnName];
            if (btnId !== undefined && state.emu) state.emu.set_button(btnId, false);
          }
        }
      }
    }
  }

  function handleTouchMove(e) {
    e.preventDefault();
    for (const t of e.changedTouches) {
      const elUnder = document.elementFromPoint(t.clientX, t.clientY);
      const btnUnder = elUnder ? elUnder.closest('[data-btn]') : null;

      for (const [el, ids] of activeTouches) {
        if (ids.has(t.identifier) && el !== btnUnder) {
          ids.delete(t.identifier);
          if (ids.size === 0) {
            activeTouches.delete(el);
            el.classList.remove('active');
            const btnName = el.dataset.btn;
            const btnId = touchBtnMap[btnName];
            if (btnId !== undefined && state.emu) state.emu.set_button(btnId, false);
          }
        }
      }

      if (btnUnder) {
        const btnName = btnUnder.dataset.btn;
        const btnId = touchBtnMap[btnName];
        if (btnId !== undefined) {
          if (!activeTouches.has(btnUnder)) activeTouches.set(btnUnder, new Set());
          if (!activeTouches.get(btnUnder).has(t.identifier)) {
            activeTouches.get(btnUnder).add(t.identifier);
            btnUnder.classList.add('active');
            if (state.emu) state.emu.set_button(btnId, true);
          }
        }
      }
    }
  }

  touchControls.addEventListener('touchstart', handleTouchStart, { passive: false });
  touchControls.addEventListener('touchend', handleTouchEnd, { passive: false });
  touchControls.addEventListener('touchcancel', handleTouchEnd, { passive: false });
  touchControls.addEventListener('touchmove', handleTouchMove, { passive: false });

  // Prevent default on canvas touches to avoid scrolling
  canvas.addEventListener('touchstart', e => e.preventDefault(), { passive: false });
  canvas2d.addEventListener('touchstart', e => e.preventDefault(), { passive: false });

  // Tap canvas to toggle fullscreen
  function fullscreenOnTap(e) {
    if (e.changedTouches.length === 1 && !document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
    }
  }
  canvas.addEventListener('touchend', fullscreenOnTap);
  canvas2d.addEventListener('touchend', fullscreenOnTap);
}
