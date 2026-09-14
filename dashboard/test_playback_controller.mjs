import assert from 'node:assert';

// Mock DOM environment for Node.js
class MockMediaElement {
    constructor(id) {
        this.id = id;
        this.paused = false;
        this.muted = false;
        this.tagName = 'VIDEO';
    }
    pause() {
        this.paused = true;
    }
    play() {
        this.paused = false;
    }
}

class MockCustomEvent {
    constructor(type, init = {}) {
        this.type = type;
        this.detail = init.detail || {};
    }
}

const windowListeners = new Map();
global.window = {
    addEventListener: (type, fn) => {
        if (!windowListeners.has(type)) windowListeners.set(type, new Set());
        windowListeners.get(type).add(fn);
    },
    removeEventListener: (type, fn) => {
        windowListeners.get(type)?.delete(fn);
    },
    dispatchEvent: (evt) => {
        const fns = windowListeners.get(evt.type);
        if (fns) {
            for (const fn of fns) fn(evt);
        }
        return true;
    },
    CustomEvent: MockCustomEvent,
};
global.CustomEvent = MockCustomEvent;

const domMedia = [];
global.document = {
    addEventListener: () => {},
    querySelectorAll: (sel) => {
        if (sel === 'video, audio') return domMedia;
        return [];
    },
};

// Import controller
const { ActivePlaybackController } = await import('./src/lib/activePlayback.js');

console.log('Testing ActivePlaybackController...');

// Test 1: Register and claim playback
const media1 = new MockMediaElement('video-1');
const media2 = new MockMediaElement('video-2');
domMedia.push(media1, media2);

let stopEvents = [];
window.addEventListener('STOP_ALL_MEDIA', (e) => {
    stopEvents.push(e.detail);
});

let media1Paused = false;
let media2Paused = false;

const unreg1 = ActivePlaybackController.register('player-1', {
    element: media1,
    pause: () => { media1Paused = true; media1.pause(); },
});

const unreg2 = ActivePlaybackController.register('player-2', {
    element: media2,
    pause: () => { media2Paused = true; media2.pause(); },
});

// Player 1 claims playback
ActivePlaybackController.claimPlayback('player-1', { element: media1 });
assert.strictEqual(ActivePlaybackController.activeId, 'player-1');
assert.strictEqual(media1.paused, false);

// Now Player 2 claims playback -> Player 1 must be paused and muted immediately!
ActivePlaybackController.claimPlayback('player-2', { element: media2 });
assert.strictEqual(ActivePlaybackController.activeId, 'player-2');
assert.strictEqual(media1Paused, true, 'Player 1 pause callback must be invoked');
assert.strictEqual(media1.paused, true, 'Player 1 media must be paused');
assert.strictEqual(media1.muted, true, 'Player 1 media must be muted');

// STOP_ALL_MEDIA event must have fired
assert.ok(stopEvents.length > 0, 'STOP_ALL_MEDIA event must be emitted');
const lastEvent = stopEvents[stopEvents.length - 1];
assert.strictEqual(lastEvent.activeId, 'player-2');

// Test 3: stopAll
ActivePlaybackController.stopAll();
assert.strictEqual(media2Paused, true, 'Player 2 must be paused when stopAll is called');
assert.strictEqual(ActivePlaybackController.activeId, null);

// Cleanup
unreg1();
unreg2();
assert.strictEqual(ActivePlaybackController.registered.size, 0);

console.log('✅ All ActivePlaybackController tests passed successfully!');
