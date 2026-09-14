/**
 * ActivePlaybackController
 * Centralized singleton enforcing strict single-player playback across the entire app.
 *
 * Requirements:
 * 1. Maintain a centralized singleton/store.
 * 2. When any clip/player starts playing, emit a global STOP_ALL_MEDIA event.
 * 3. Pause, mute, and detach previous media instances immediately before
 *    mounting or playing the new target video. Never allow simultaneous playback.
 */

class ActivePlaybackManager {
    constructor() {
        this.activeId = null;
        this.activeElement = null;
        this.registered = new Map(); // id -> { element, onStop, pause }
        this.listeners = new Set();
        this._initGlobalCapture();
    }

    /**
     * Intercept any native play events on document in capture phase
     * so third-party players, Remotion players, and manual <video> elements
     * automatically trigger mutual exclusion without boilerplate.
     */
    _initGlobalCapture() {
        if (typeof window === 'undefined' || typeof document === 'undefined') return;

        document.addEventListener(
            'play',
            (event) => {
                const target = event.target;
                if (target && (target.tagName === 'VIDEO' || target.tagName === 'AUDIO')) {
                    const id = target.dataset?.playerId || target.id || `dom-media-${Math.random().toString(36).slice(2, 8)}`;
                    this.claimPlayback(id, { element: target });
                }
            },
            true // Capture phase: runs before component onPlay handlers
        );
    }

    /**
     * Register a player instance with callbacks to pause/mute it.
     * Returns an unregister cleanup function.
     */
    register(id, { element = null, onStop = null, pause = null } = {}) {
        this.registered.set(id, { element, onStop, pause });
        return () => {
            this.unregister(id);
        };
    }

    unregister(id) {
        if (this.activeId === id) {
            this.activeId = null;
            this.activeElement = null;
        }
        this.registered.delete(id);
        this._notify();
    }

    /**
     * Claim active playback for a specific player ID.
     * Stops, pauses, and mutes all other media instances immediately.
     */
    claimPlayback(id, { element = null, onStop = null, pause = null } = {}) {
        if (this.activeId === id && this.activeElement === element) {
            return;
        }

        const previousId = this.activeId;
        const previousElement = this.activeElement;

        // 1. Pause and mute previous active element immediately
        if (previousElement && previousElement !== element) {
            try {
                previousElement.pause();
                previousElement.muted = true;
            } catch (e) {
                // Ignore transient DOM aborts
            }
        }

        // 2. Stop all other registered players
        for (const [regId, reg] of this.registered.entries()) {
            if (regId !== id) {
                try {
                    if (reg.pause) reg.pause();
                    if (reg.onStop) reg.onStop();
                    if (reg.element && reg.element !== element) {
                        reg.element.pause();
                        reg.element.muted = true;
                    }
                } catch (e) {
                    // Ignore transient errors
                }
            }
        }

        // 3. Scan DOM for any rogue <video> or <audio> currently playing
        if (typeof document !== 'undefined') {
            const allMedia = document.querySelectorAll('video, audio');
            allMedia.forEach((media) => {
                if (media !== element && !media.paused) {
                    try {
                        media.pause();
                        media.muted = true;
                    } catch (e) {
                        // Ignore
                    }
                }
            });
        }

        // 4. Emit global STOP_ALL_MEDIA custom event
        if (typeof window !== 'undefined') {
            window.dispatchEvent(
                new CustomEvent('STOP_ALL_MEDIA', {
                    detail: {
                        activeId: id,
                        previousId,
                        activeElement: element,
                    },
                })
            );
        }

        // 5. Update registered entry if element provided
        if (id && (element || onStop || pause)) {
            const existing = this.registered.get(id) || {};
            this.registered.set(id, {
                element: element || existing.element || null,
                onStop: onStop || existing.onStop || null,
                pause: pause || existing.pause || null,
            });
        }

        this.activeId = id;
        this.activeElement = element;
        this._notify();
    }

    /**
     * Explicitly stop all media across the application.
     */
    stopAll(exceptId = null) {
        for (const [regId, reg] of this.registered.entries()) {
            if (regId !== exceptId) {
                try {
                    if (reg.pause) reg.pause();
                    if (reg.onStop) reg.onStop();
                    if (reg.element) {
                        reg.element.pause();
                        reg.element.muted = true;
                    }
                } catch (e) {}
            }
        }

        if (typeof document !== 'undefined') {
            const allMedia = document.querySelectorAll('video, audio');
            allMedia.forEach((media) => {
                const isExcept = exceptId && media.dataset?.playerId === exceptId;
                if (!isExcept && !media.paused) {
                    try {
                        media.pause();
                        media.muted = true;
                    } catch (e) {}
                }
            });
        }

        if (typeof window !== 'undefined') {
            window.dispatchEvent(
                new CustomEvent('STOP_ALL_MEDIA', {
                    detail: { activeId: exceptId },
                })
            );
        }

        if (!exceptId || this.activeId !== exceptId) {
            this.activeId = exceptId;
            this.activeElement = exceptId ? this.registered.get(exceptId)?.element || null : null;
        }
        this._notify();
    }

    subscribe(callback) {
        this.listeners.add(callback);
        return () => this.listeners.delete(callback);
    }

    _notify() {
        for (const cb of this.listeners) {
            try {
                cb({
                    activeId: this.activeId,
                    activeElement: this.activeElement,
                });
            } catch (e) {}
        }
    }
}

export const ActivePlaybackController = new ActivePlaybackManager();
