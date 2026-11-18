# Encrypted Audio Streaming (`audio_crypto.py` & `sw-offline-stream.js`)

## The Problem

In Museed, audio files are stored encrypted (`MSD1` format), including tracks downloaded to IndexedDB for offline playback.

Originally, playing an encrypted track in the browser meant reading the full file into memory, decrypting the entire buffer, and creating an in-memory `Blob` URL. That caused two issues:

1. Decrypting full tracks (especially long DJ mixes) into RAM at once caused high memory usage and tab crashes on mobile browsers.
2. HTML5 `<audio>` elements use HTTP `Range: bytes=start-end` requests to seek. When a user scrubs to the middle of a track, we need to decrypt starting from an arbitrary byte offset without decrypting everything before it.

## The Solution

We used **AES-128-CTR** with a 20-byte header (`4-byte 'MSD1'` magic signature + `16-byte IV`) prepended to the ciphertext.

Because AES-CTR encrypts sequential 16-byte counter blocks (`IV + block_index`), any block in the file can be decrypted independently. Both the Python backend (`audio_crypto.py`) and the browser Service Worker (`sw-offline-stream.js`) intercept byte-range requests, compute the counter offset for the requested slice, decrypt only that window, and return `HTTP 206 Partial Content`.

## How It Works

- **Counter offset math (`compute_ctr_block_counter` / `computeCtrBlockCounter`):** For a requested plaintext offset `start`, we compute `block_idx = start // 16` and `block_offset = start % 16`. We add `block_idx` to the 128-bit big-endian IV modulo $2^{128}$, decrypt from that block boundary, and discard the leading `block_offset` bytes.
- **Backend streaming (`AudioSourceReader`):** Reads encrypted or unencrypted tracks from disk, Django storage (S3), or HTTP URLs, yielding decrypted 64 KB chunks.
- **Offline Service Worker (`sw-offline-stream.js`):** Intercepts `/offline-stream/:trackId` requests, slices a 512 KB window from the IndexedDB `Blob` (`blob.slice()`), decrypts it via Web Crypto (`crypto.subtle`), and returns `206 Partial Content`. It also handles `/offline-hls/:trackId/*` to rewrite `.m3u8` manifests and serve local transport streams.

## Tests & Limitations

```bash
python3 -m unittest discover -s tests -v
node --test tests/test_sw_offline_stream.js
```

**Production vs. Sample Changes:**
- Removed the fallback to a default secret key and the `?key=` URL query parameter in the Service Worker.
- Added RFC 7233 `Range` validation (`416 Range Not Satisfiable` on inverted/out-of-bounds ranges) and handling for upstream HTTP servers that return `200` instead of `206`.
- Added CommonJS exports to `sw-offline-stream.js` so its functions can be tested in Node.js.

**Limitations:**
- AES-CTR provides confidentiality and random-access seeking, not message authentication (unlike AES-GCM). It does not detect active tampering with stored ciphertext unless verified separately.
- The Django view that checks listener permissions and the IndexedDB download manager that stores tracks are omitted.
