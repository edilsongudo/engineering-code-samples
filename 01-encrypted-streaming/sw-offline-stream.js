/**
 * Museed Offline Audio Streaming Service Worker
 *
 * Intercepts `/offline-stream/:trackId` and `/offline-hls/:trackId/:resource` requests
 * from the browser's HTML5 `<audio>` / MSE player and performs O(1) on-the-fly byte-range
 * decryption of AES-128-CTR (`MSD1`) audio blobs stored in IndexedDB.
 *
 * By decrypting only the requested 512 KB range window rather than the entire audio file,
 * this worker avoids holding multi-megabyte uncompressed audio buffers in memory and
 * enables instant random-access seeking offline.
 *
 * Security Note:
 *   AES-128-CTR is a seekable stream cipher mode providing confidentiality. It does not
 *   provide built-in message authentication (unlike AES-GCM).
 */

const DB_NAME = 'museed_offline_db';
const STORE_TRACKS = 'tracks';
const HEADER_LEN = 4;
const IV_LEN = 16;
const AES_BLOCK_SIZE = 16;
const CHUNK_SIZE = 512 * 1024; // 512 KB streaming window

// In-memory cache for imported Web Crypto AES-CTR keys
const keyCache = new Map();

/**
 * Converts a 32-character hexadecimal key string to a 16-byte Uint8Array.
 * Returns null if the string is not a valid 32-char hex string.
 */
function hexToBytes(hex) {
    if (typeof hex !== 'string') return null;
    const cleanHex = hex.trim();
    if (!/^[0-9a-fA-F]{32}$/.test(cleanHex)) return null;
    const bytes = new Uint8Array(16);
    for (let i = 0; i < 16; i++) {
        bytes[i] = parseInt(cleanHex.substring(i * 2, i * 2 + 2), 16);
    }
    return bytes;
}

/**
 * Computes the 128-bit big-endian AES-CTR counter block for a given 16-byte IV
 * and 16-byte block index: `(BigInt(IV) + BigInt(blockIndex)) mod 2^128`.
 */
function computeCtrBlockCounter(ivBytes, blockIndex) {
    if (!(ivBytes instanceof Uint8Array) || ivBytes.byteLength !== IV_LEN) {
        throw new Error(`IV must be a ${IV_LEN}-byte Uint8Array`);
    }
    if (!Number.isInteger(blockIndex) || blockIndex < 0) {
        throw new Error('blockIndex must be a non-negative integer');
    }

    let ivBig = 0n;
    for (let i = 0; i < IV_LEN; i++) {
        ivBig = (ivBig << 8n) | BigInt(ivBytes[i]);
    }
    const blockCounterBig = (ivBig + BigInt(blockIndex)) % (1n << 128n);
    const blockIv = new Uint8Array(IV_LEN);
    for (let i = IV_LEN - 1; i >= 0; i--) {
        blockIv[i] = Number((blockCounterBig >> BigInt((IV_LEN - 1 - i) * 8)) & 0xffn);
    }
    return blockIv;
}

/**
 * Parses and validates an RFC 7233 HTTP Range header against `totalSize`.
 *
 * Returns:
 *   - `{ valid: true, isPartial: false, start: 0, end: min(maxChunkSize - 1, totalSize - 1) }`
 *     when no Range header is present (streams initial chunk as 206 so `<audio>` discovers byte-range support).
 *   - `{ valid: true, isPartial: true, start, end }` for a valid satisfiable byte range.
 *   - `{ valid: false, status: 416 }` when the range is malformed or unsatisfiable.
 */
function parseHttpRange(rangeHeader, totalSize, maxChunkSize = CHUNK_SIZE) {
    if (!Number.isInteger(totalSize) || totalSize <= 0) {
        return { valid: false, status: 416 };
    }

    if (!rangeHeader || typeof rangeHeader !== 'string' || !rangeHeader.trim()) {
        return {
            valid: true,
            isPartial: false,
            start: 0,
            end: Math.min(maxChunkSize - 1, totalSize - 1),
        };
    }

    const cleaned = rangeHeader.trim();
    if (!cleaned.startsWith('bytes=')) {
        return { valid: false, status: 416 };
    }

    const match = cleaned.match(/^bytes=(\d*)-(\d*)$/);
    if (!match) {
        return { valid: false, status: 416 };
    }

    const [, startStr, endStr] = match;
    if (!startStr && !endStr) {
        return { valid: false, status: 416 };
    }

    let start = 0;
    let end = totalSize - 1;

    if (!startStr) {
        // Suffix range: bytes=-500
        const suffixLength = parseInt(endStr, 10);
        if (!Number.isFinite(suffixLength) || suffixLength <= 0) {
            return { valid: false, status: 416 };
        }
        start = Math.max(0, totalSize - suffixLength);
        end = totalSize - 1;
    } else {
        start = parseInt(startStr, 10);
        if (!Number.isFinite(start) || start < 0 || start >= totalSize) {
            return { valid: false, status: 416 };
        }

        if (endStr) {
            const requestedEnd = parseInt(endStr, 10);
            if (!Number.isFinite(requestedEnd) || requestedEnd < start) {
                return { valid: false, status: 416 };
            }
            end = Math.min(requestedEnd, totalSize - 1);
        } else {
            // Open-ended range (bytes=start-): cap at maxChunkSize window
            end = Math.min(start + maxChunkSize - 1, totalSize - 1);
        }
    }

    return { valid: true, isPartial: true, start, end };
}

/**
 * Imports and caches an AES-CTR CryptoKey using Web Crypto API.
 */
async function getCryptoKey(hexKey) {
    if (!hexKey) return null;
    const cleanKey = String(hexKey).trim().toLowerCase();
    if (keyCache.has(cleanKey)) return keyCache.get(cleanKey);

    const keyBytes = hexToBytes(cleanKey);
    if (!keyBytes) return null;

    try {
        const cryptoObj = (typeof self !== 'undefined' && self.crypto) ? self.crypto : globalThis.crypto;
        const key = await cryptoObj.subtle.importKey(
            'raw',
            keyBytes,
            { name: 'AES-CTR' },
            false,
            ['decrypt']
        );
        keyCache.set(cleanKey, key);
        return key;
    } catch (e) {
        console.error('[SW Offline Stream] Failed to import crypto key:', e);
        return null;
    }
}

/**
 * Retrieves a track record from IndexedDB.
 */
function getTrackRecord(trackId) {
    return new Promise((resolve) => {
        const idb = (typeof self !== 'undefined' && self.indexedDB) ? self.indexedDB : globalThis.indexedDB;
        if (!idb) return resolve(null);

        const req = idb.open(DB_NAME);
        req.onerror = () => resolve(null);
        req.onsuccess = () => {
            const db = req.result;
            if (!db.objectStoreNames.contains(STORE_TRACKS)) {
                db.close();
                return resolve(null);
            }
            try {
                const tx = db.transaction(STORE_TRACKS, 'readonly');
                const store = tx.objectStore(STORE_TRACKS);
                const getReq = store.get(String(trackId));
                getReq.onsuccess = () => {
                    db.close();
                    resolve(getReq.result || null);
                };
                getReq.onerror = () => {
                    db.close();
                    resolve(null);
                };
            } catch {
                db.close();
                resolve(null);
            }
        };
    });
}

/**
 * Decrypts a byte range `[start, end]` from an MSD1 AES-128-CTR encrypted Blob.
 */
async function decryptBlobRange(blob, ivBytes, cryptoKey, start, end) {
    const blockIndex = Math.floor(start / AES_BLOCK_SIZE);
    const blockOffset = start % AES_BLOCK_SIZE;
    const cipherStart = (HEADER_LEN + IV_LEN) + (blockIndex * AES_BLOCK_SIZE);
    const cipherEnd = (HEADER_LEN + IV_LEN) + end + 1;

    const cipherSlice = await blob.slice(cipherStart, cipherEnd).arrayBuffer();
    const blockIv = computeCtrBlockCounter(ivBytes, blockIndex);

    const cryptoObj = (typeof self !== 'undefined' && self.crypto) ? self.crypto : globalThis.crypto;
    const decryptedBlock = await cryptoObj.subtle.decrypt(
        { name: 'AES-CTR', counter: blockIv, length: 128 },
        cryptoKey,
        cipherSlice
    );

    return decryptedBlock.slice(blockOffset, blockOffset + (end - start + 1));
}

/**
 * Handles `/offline-stream/:trackId` requests with RFC 7233 Range streaming.
 */
async function handleOfflineAudioStream(request, recordLookup = getTrackRecord) {
    const url = new URL(request.url);
    const parts = url.pathname.split('/').filter(Boolean);
    const trackIdIndex = parts.indexOf('offline-stream');
    const trackId = trackIdIndex !== -1 && parts[trackIdIndex + 1]
        ? decodeURIComponent(parts[trackIdIndex + 1])
        : null;

    if (!trackId) {
        return new Response('Track ID missing', { status: 400 });
    }

    const record = await recordLookup(trackId);
    if (!record || !record.audioBlob) {
        return new Response('Offline audio track not found in local database', { status: 404 });
    }

    const blob = record.audioBlob;

    // Retrieve decryption key from IndexedDB record (avoiding sensitive keys in query params)
    let streamKey = record.streamKey || null;
    if (!streamKey) {
        const globalKeyRecord = await recordLookup('__museed_global_stream_key__');
        if (globalKeyRecord && globalKeyRecord.key) {
            streamKey = globalKeyRecord.key;
        }
    }

    // Detect MSD1 encryption header (4-byte 'MSD1' + 16-byte IV)
    let isEncrypted = false;
    let ivBytes = null;
    let totalPlaintextSize = blob.size;

    if (blob.size >= HEADER_LEN + IV_LEN) {
        const headerSlice = await blob.slice(0, HEADER_LEN + IV_LEN).arrayBuffer();
        const headerView = new Uint8Array(headerSlice);
        if (
            headerView[0] === 0x4D && // 'M'
            headerView[1] === 0x53 && // 'S'
            headerView[2] === 0x44 && // 'D'
            headerView[3] === 0x31    // '1'
        ) {
            isEncrypted = true;
            ivBytes = headerView.slice(HEADER_LEN, HEADER_LEN + IV_LEN);
            totalPlaintextSize = blob.size - (HEADER_LEN + IV_LEN);
        }
    }

    if (totalPlaintextSize <= 0) {
        return new Response(null, {
            status: 416,
            statusText: 'Range Not Satisfiable',
            headers: { 'Content-Range': 'bytes */0' },
        });
    }

    const trackFilename = record.metadata?.filename || record.metadata?.file || record.filename || '';
    const mimeType = trackFilename.toLowerCase().endsWith('.m4a') ? 'audio/mp4' : 'audio/mpeg';

    if (request.method === 'HEAD') {
        return new Response(null, {
            status: 200,
            headers: {
                'Content-Type': mimeType,
                'Content-Length': String(totalPlaintextSize),
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'no-cache, no-store',
            },
        });
    }

    const rangeHeader = request.headers.get('range');
    const parsedRange = parseHttpRange(rangeHeader, totalPlaintextSize, CHUNK_SIZE);
    if (!parsedRange.valid) {
        return new Response(null, {
            status: 416,
            statusText: 'Range Not Satisfiable',
            headers: {
                'Content-Range': `bytes */${totalPlaintextSize}`,
            },
        });
    }

    const { start, end } = parsedRange;

    if (!isEncrypted) {
        const buffer = await blob.slice(start, end + 1).arrayBuffer();
        return new Response(buffer, {
            status: 206,
            statusText: 'Partial Content',
            headers: {
                'Content-Type': mimeType,
                'Content-Range': `bytes ${start}-${end}/${totalPlaintextSize}`,
                'Content-Length': String(buffer.byteLength),
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'no-cache, no-store',
            },
        });
    }

    const cryptoKey = await getCryptoKey(streamKey);
    if (!cryptoKey) {
        return new Response('Missing or invalid decryption key for offline track', { status: 401 });
    }

    try {
        const plaintextSlice = await decryptBlobRange(blob, ivBytes, cryptoKey, start, end);
        return new Response(plaintextSlice, {
            status: 206,
            statusText: 'Partial Content',
            headers: {
                'Content-Type': mimeType,
                'Content-Range': `bytes ${start}-${end}/${totalPlaintextSize}`,
                'Content-Length': String(plaintextSlice.byteLength),
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'no-cache, no-store',
            },
        });
    } catch (err) {
        console.error('[SW Offline Stream] Decryption failed:', err);
        return new Response('Decryption failed', { status: 500 });
    }
}

/**
 * Handles `/offline-hls/:trackId/:resource` requests for offline HLS VOD playback.
 */
async function handleOfflineHlsRequest(request, recordLookup = getTrackRecord) {
    const url = new URL(request.url);
    const parts = url.pathname.split('/').filter(Boolean);
    const hlsIndex = parts.indexOf('offline-hls');
    const trackId = hlsIndex !== -1 && parts[hlsIndex + 1] ? decodeURIComponent(parts[hlsIndex + 1]) : null;
    const resource = hlsIndex !== -1 && parts[hlsIndex + 2] ? parts[hlsIndex + 2] : 'playlist.m3u8';

    if (!trackId) {
        return new Response('Track ID missing', { status: 400 });
    }

    const record = await recordLookup(trackId);
    if (!record) {
        return new Response('Offline HLS track not found in local database', { status: 404 });
    }

    if (resource === 'playlist.m3u8') {
        if (!record.m3u8Manifest) {
            return new Response('Offline HLS manifest not found', { status: 404 });
        }
        const encodedId = encodeURIComponent(trackId);
        const keyUrl = `${url.origin}/offline-hls/${encodedId}/key.bin`;
        const tsUrl = `${url.origin}/offline-hls/${encodedId}/playlist.ts`;

        const rewrittenLines = record.m3u8Manifest.split(/\r?\n/).map(line => {
            const trimmed = line.trim();
            if (!trimmed) return line;
            if (trimmed.startsWith('#EXT-X-KEY')) {
                return line.replace(/URI="[^"]+"/, `URI="${keyUrl}"`);
            }
            if (!trimmed.startsWith('#')) {
                return tsUrl;
            }
            return line;
        });

        return new Response(rewrittenLines.join('\n'), {
            status: 200,
            headers: {
                'Content-Type': 'application/vnd.apple.mpegurl',
                'Cache-Control': 'no-cache, no-store',
            },
        });
    }

    if (resource === 'key.bin') {
        if (!record.hlsKey) {
            return new Response('Offline HLS key not found', { status: 404 });
        }
        const keyBytes = record.hlsKey instanceof Uint8Array ? record.hlsKey : new Uint8Array(record.hlsKey);
        return new Response(keyBytes, {
            status: 200,
            headers: {
                'Content-Type': 'application/octet-stream',
                'Content-Length': String(keyBytes.byteLength),
                'Cache-Control': 'no-cache, no-store',
            },
        });
    }

    if (resource === 'playlist.ts' || resource.endsWith('.ts')) {
        const blob = record.audioBlob;
        if (!blob) {
            return new Response('Offline HLS transport stream not found', { status: 404 });
        }
        const totalSize = blob.size;

        if (request.method === 'HEAD') {
            return new Response(null, {
                status: 200,
                headers: {
                    'Content-Type': 'video/mp2t',
                    'Content-Length': String(totalSize),
                    'Accept-Ranges': 'bytes',
                    'Cache-Control': 'no-cache, no-store',
                },
            });
        }

        const rangeHeader = request.headers.get('range');
        if (rangeHeader) {
            const parsedRange = parseHttpRange(rangeHeader, totalSize, totalSize);
            if (!parsedRange.valid) {
                return new Response(null, {
                    status: 416,
                    statusText: 'Range Not Satisfiable',
                    headers: { 'Content-Range': `bytes */${totalSize}` },
                });
            }
            const { start, end } = parsedRange;
            const sliceBuffer = await blob.slice(start, end + 1).arrayBuffer();
            return new Response(sliceBuffer, {
                status: 206,
                statusText: 'Partial Content',
                headers: {
                    'Content-Type': 'video/mp2t',
                    'Content-Range': `bytes ${start}-${end}/${totalSize}`,
                    'Content-Length': String(sliceBuffer.byteLength),
                    'Accept-Ranges': 'bytes',
                    'Cache-Control': 'no-cache, no-store',
                },
            });
        }

        const fullBuffer = await blob.arrayBuffer();
        return new Response(fullBuffer, {
            status: 200,
            headers: {
                'Content-Type': 'video/mp2t',
                'Content-Length': String(fullBuffer.byteLength),
                'Accept-Ranges': 'bytes',
                'Cache-Control': 'no-cache, no-store',
            },
        });
    }

    return new Response('Not found', { status: 404 });
}

if (typeof self !== 'undefined' && typeof self.addEventListener === 'function') {
    self.addEventListener('fetch', (event) => {
        const url = new URL(event.request.url);
        if (url.pathname.includes('/offline-hls/')) {
            event.respondWith(handleOfflineHlsRequest(event.request));
        } else if (url.pathname.includes('/offline-stream/')) {
            event.respondWith(handleOfflineAudioStream(event.request));
        }
    });
}

// Export helpers for unit and cross-language interoperability testing in Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        HEADER_LEN,
        IV_LEN,
        AES_BLOCK_SIZE,
        CHUNK_SIZE,
        hexToBytes,
        computeCtrBlockCounter,
        parseHttpRange,
        getCryptoKey,
        decryptBlobRange,
        handleOfflineAudioStream,
        handleOfflineHlsRequest,
    };
}
