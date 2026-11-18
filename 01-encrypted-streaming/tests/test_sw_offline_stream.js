const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const test = require('node:test');

const {
    HEADER_LEN,
    IV_LEN,
    hexToBytes,
    computeCtrBlockCounter,
    parseHttpRange,
    handleOfflineAudioStream,
} = require('../sw-offline-stream.js');

/**
 * Helper that produces an MSD1-encrypted Blob matching Python's `encrypt_audio_bytes`.
 */
function createEncryptedMsd1Blob(plaintextBuffer, keyHex, ivBuffer) {
    const keyBytes = Buffer.from(keyHex, 'hex');
    const cipher = crypto.createCipheriv('aes-128-ctr', keyBytes, ivBuffer);
    const ciphertext = Buffer.concat([cipher.update(plaintextBuffer), cipher.final()]);
    const magic = Buffer.from('MSD1', 'ascii');
    const fullBuffer = Buffer.concat([magic, ivBuffer, ciphertext]);
    return new Blob([fullBuffer], { type: 'application/octet-stream' });
}

test('hexToBytes validates 32-char hexadecimal keys strictly', () => {
    const validKey = '000102030405060708090a0b0c0d0e0f';
    const bytes = hexToBytes(validKey);
    assert.ok(bytes instanceof Uint8Array);
    assert.equal(bytes.byteLength, 16);
    assert.equal(bytes[0], 0x00);
    assert.equal(bytes[15], 0x0f);

    // Reject non-hex characters or wrong lengths
    assert.equal(hexToBytes('zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz'), null);
    assert.equal(hexToBytes('000102'), null);
    assert.equal(hexToBytes(null), null);
});

test('computeCtrBlockCounter handles 128-bit big-endian addition and wrap-around', () => {
    const maxIv = new Uint8Array(16).fill(0xff);
    const wrapped = computeCtrBlockCounter(maxIv, 1);
    assert.deepEqual(wrapped, new Uint8Array(16).fill(0x00));

    const wrappedPlus2 = computeCtrBlockCounter(maxIv, 2);
    const expected = new Uint8Array(16).fill(0x00);
    expected[15] = 0x01;
    assert.deepEqual(wrappedPlus2, expected);
});

test('parseHttpRange validates RFC 7233 ranges and rejects malformed/unsatisfiable ranges', () => {
    const total = 10000;

    // Explicit range
    const r1 = parseHttpRange('bytes=100-499', total);
    assert.deepEqual(r1, { valid: true, isPartial: true, start: 100, end: 499 });

    // Open-ended range capped by maxChunkSize
    const r2 = parseHttpRange('bytes=500-', total, 1024);
    assert.deepEqual(r2, { valid: true, isPartial: true, start: 500, end: 1523 });

    // Suffix range
    const r3 = parseHttpRange('bytes=-250', total);
    assert.deepEqual(r3, { valid: true, isPartial: true, start: 9750, end: 9999 });

    // Invalid ranges -> 416
    assert.equal(parseHttpRange('bytes=10000-10050', total).valid, false);
    assert.equal(parseHttpRange('bytes=500-100', total).valid, false);
    assert.equal(parseHttpRange('bytes=-0', total).valid, false);
    assert.equal(parseHttpRange('bytes=-', total).valid, false);
    assert.equal(parseHttpRange('items=0-5', total).valid, false);
});

test('handleOfflineAudioStream performs O(1) random-access AES-128-CTR range decryption', async () => {
    const keyHex = '000102030405060708090a0b0c0d0e0f';
    const iv = Buffer.from('fedcba98765432100123456789abcdef', 'hex');
    const plaintext = Buffer.alloc(4099);
    for (let i = 0; i < plaintext.length; i++) {
        plaintext[i] = (i * 41 + 13) & 0xff;
    }

    const encryptedBlob = createEncryptedMsd1Blob(plaintext, keyHex, iv);
    const mockDb = new Map([
        ['track-42', { id: 'track-42', audioBlob: encryptedBlob, streamKey: keyHex, filename: 'song.mp3' }],
    ]);
    const lookup = async (id) => mockDb.get(id) || null;

    // Test aligned and unaligned byte ranges across 16-byte AES block boundaries
    const rangesToTest = [
        [0, 15],
        [1, 14],
        [7, 39],
        [15, 16],
        [511, 2047],
        [4000, 4098],
    ];

    for (const [start, end] of rangesToTest) {
        const req = new Request('https://museed.example/offline-stream/track-42', {
            method: 'GET',
            headers: { Range: `bytes=${start}-${end}` },
        });
        const res = await handleOfflineAudioStream(req, lookup);
        assert.equal(res.status, 206);
        assert.equal(res.headers.get('Content-Range'), `bytes ${start}-${end}/${plaintext.length}`);

        const bodyBuf = Buffer.from(await res.arrayBuffer());
        assert.deepEqual(bodyBuf, plaintext.subarray(start, end + 1), `Mismatch on range ${start}-${end}`);
    }

    // Verify 416 Range Not Satisfiable on out-of-bounds range
    const badReq = new Request('https://museed.example/offline-stream/track-42', {
        method: 'GET',
        headers: { Range: `bytes=${plaintext.length}-${plaintext.length + 100}` },
    });
    const badRes = await handleOfflineAudioStream(badReq, lookup);
    assert.equal(badRes.status, 416);
});
