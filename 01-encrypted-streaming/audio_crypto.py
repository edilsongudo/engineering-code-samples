"""
Encrypted Audio Streaming & Seekable AES-128-CTR Engine.

Provides zero-overhead AES-128-CTR encryption, per-track HMAC-SHA256 key derivation,
RFC 7233 HTTP Range header parsing, and constant-memory O(1) random-access
decryption across local files, cloud object storage, and remote HTTP streams.

Security Note on AES-CTR:
    AES-128-CTR is a stream cipher mode chosen specifically for O(1) random-access
    seeking (any 16-byte block can be decrypted independently by advancing the
    128-bit counter by `block_index = byte_offset // 16`).
    Note that AES-CTR provides confidentiality only, not cryptographic integrity
    or authenticity (unlike AES-GCM). If tamper resistance against active storage
    modification is required, an external HMAC or signed manifest should be verified.
"""

from dataclasses import dataclass
import hashlib
import hmac
import logging
import os
import re
from typing import Callable, Iterator, Optional, Tuple

import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

try:
    from django.conf import settings as django_settings
    from django.core.files.storage import default_storage as django_default_storage
except ImportError:
    django_settings = None  # type: ignore[assignment]
    django_default_storage = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

MAGIC_HEADER = b"MSD1"
HEADER_LEN = 4
IV_LEN = 16
KEY_LEN = 16
AES_BLOCK_SIZE = 16
DEFAULT_CHUNK_SIZE = 65536  # 64 KB streaming buffer

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class RangeNotSatisfiableError(ValueError):
    """Raised when a requested HTTP byte range falls outside the resource bounds (HTTP 416)."""

    def __init__(self, message: str, total_size: int) -> None:
        super().__init__(message)
        self.total_size = total_size


@dataclass(frozen=True)
class ByteRange:
    """Represents a validated inclusive byte range [start, end] and whether it is a partial request."""

    start: int
    end: int
    total_size: int
    is_partial: bool

    @property
    def content_length(self) -> int:
        return self.end - self.start + 1

    @property
    def content_range_header(self) -> str:
        return f"bytes {self.start}-{self.end}/{self.total_size}"


def _get_setting(name: str, default: Optional[str] = None) -> Optional[str]:
    """Retrieves a configuration setting from Django settings (if active) or environment variables."""
    if django_settings is not None:
        try:
            if getattr(django_settings, "configured", False):
                val = getattr(django_settings, name, None)
                if val:
                    return str(val)
        except Exception:
            pass
    return os.getenv(name, default)


def resolve_local_audio_path(raw_file: str) -> Optional[str]:
    """
    Resolves whether `raw_file` corresponds to an existing local file on disk.
    Checks direct paths as well as MEDIA_ROOT and BASE_DIR when configured.
    Returns the path if found locally, or None for remote URLs / missing files.
    """
    if not raw_file or raw_file.startswith(("http://", "https://")):
        return None

    raw_file_str = raw_file.strip()
    candidates = [raw_file_str]

    media_root = _get_setting("MEDIA_ROOT")
    base_dir = _get_setting("BASE_DIR")

    if media_root:
        candidates.append(os.path.join(media_root, raw_file_str.lstrip("/")))
        stripped_media = raw_file_str.replace("/media/", "").replace("media/", "").lstrip("/")
        candidates.append(os.path.join(media_root, stripped_media))
    if base_dir:
        candidates.append(os.path.join(base_dir, raw_file_str.lstrip("/")))

    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def get_audio_encryption_key() -> bytes:
    """
    Returns the 16-byte (128-bit) symmetric AES master key.
    Reads `AUDIO_ENCRYPTION_KEY` (either a 32-character hex string or arbitrary passphrase),
    or derives a deterministic 16-byte key from `SECRET_KEY`.
    Raises RuntimeError if neither key is configured.
    """
    configured_key = _get_setting("AUDIO_ENCRYPTION_KEY")
    if configured_key:
        cleaned = configured_key.strip()
        if len(cleaned) == 32:
            try:
                return bytes.fromhex(cleaned)
            except ValueError:
                pass
        return hashlib.sha256(cleaned.encode("utf-8")).digest()[:KEY_LEN]

    secret_key = _get_setting("SECRET_KEY")
    if secret_key and secret_key.strip():
        return hashlib.sha256(b"museed_audio_crypto_" + secret_key.strip().encode("utf-8")).digest()[:KEY_LEN]

    raise RuntimeError(
        "Audio encryption key is not configured. Set AUDIO_ENCRYPTION_KEY or SECRET_KEY in the environment."
    )


def get_stream_key_hex() -> str:
    """Returns the 128-bit master audio encryption key as a 32-character hexadecimal string."""
    return get_audio_encryption_key().hex()


def get_track_hls_key(audio_id: int) -> bytes:
    """
    Derives a deterministic 16-byte (128-bit) AES key unique to a specific audio track
    using HMAC-SHA256 over the platform master key.
    Ensures compromising a single track key does not expose other tracks in the catalog.
    """
    master_key = get_audio_encryption_key()
    msg = f"museed-hls-key-audio-{int(audio_id)}".encode("utf-8")
    return hmac.new(master_key, msg, hashlib.sha256).digest()[:KEY_LEN]


def is_encrypted_audio(data: bytes) -> bool:
    """Checks whether the byte stream begins with the 4-byte 'MSD1' magic header."""
    return bool(data and len(data) >= HEADER_LEN and data.startswith(MAGIC_HEADER))


def encrypt_audio_bytes(data: bytes, key: Optional[bytes] = None, iv: Optional[bytes] = None) -> bytes:
    """
    Encrypts audio data using AES-128-CTR with zero ciphertext expansion.
    Prepends the 4-byte 'MSD1' header and a 16-byte random IV:
        [4B 'MSD1'][16B IV][N-byte AES-128-CTR Ciphertext]
    If the buffer is empty or already starts with 'MSD1', returns it unchanged.
    """
    if not data or is_encrypted_audio(data):
        return data

    aes_key = key if key is not None else get_audio_encryption_key()
    if len(aes_key) != KEY_LEN:
        raise ValueError(f"AES-128 key must be exactly {KEY_LEN} bytes, got {len(aes_key)}.")

    nonce_iv = iv if iv is not None else os.urandom(IV_LEN)
    if len(nonce_iv) != IV_LEN:
        raise ValueError(f"AES-CTR IV must be exactly {IV_LEN} bytes, got {len(nonce_iv)}.")

    cipher = Cipher(algorithms.AES(aes_key), modes.CTR(nonce_iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(data) + encryptor.finalize()

    return MAGIC_HEADER + nonce_iv + ciphertext


def decrypt_audio_bytes(data: bytes, key: Optional[bytes] = None) -> bytes:
    """
    Decrypts an 'MSD1' AES-128-CTR encrypted buffer.
    If the buffer does not start with 'MSD1', returns it unchanged for backward compatibility
    with legacy unencrypted tracks.
    """
    if not data or not is_encrypted_audio(data):
        return data

    if len(data) < (HEADER_LEN + IV_LEN):
        raise ValueError("Corrupt encrypted audio buffer: shorter than 20-byte MSD1 header + IV.")

    aes_key = key if key is not None else get_audio_encryption_key()
    if len(aes_key) != KEY_LEN:
        raise ValueError(f"AES-128 key must be exactly {KEY_LEN} bytes, got {len(aes_key)}.")

    iv = data[HEADER_LEN : HEADER_LEN + IV_LEN]
    ciphertext = data[HEADER_LEN + IV_LEN :]

    cipher = Cipher(algorithms.AES(aes_key), modes.CTR(iv))
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()


def compute_ctr_block_counter(iv: bytes, block_index: int) -> bytes:
    """
    Computes the 128-bit big-endian AES-CTR counter value for an arbitrary 16-byte block offset:
        counter = (int(IV) + block_index) mod 2^128
    """
    if len(iv) != IV_LEN:
        raise ValueError(f"IV must be {IV_LEN} bytes, got {len(iv)}.")
    if block_index < 0:
        raise ValueError("block_index must be non-negative.")

    iv_int = int.from_bytes(iv, "big")
    counter_int = (iv_int + block_index) % (1 << 128)
    return counter_int.to_bytes(IV_LEN, "big")


def parse_http_range_header(
    range_header: Optional[str],
    total_size: int,
    max_open_chunk_size: Optional[int] = None,
) -> ByteRange:
    """
    Parses and validates an RFC 7233 HTTP Range header (e.g. 'bytes=0-1023', 'bytes=1024-', 'bytes=-512').

    - If `range_header` is empty or None, returns the full range [0, total_size - 1] with `is_partial=False`.
    - If `max_open_chunk_size` is provided and the client requests an open-ended range ('bytes=start-'),
      caps `end` to `min(start + max_open_chunk_size - 1, total_size - 1)`.
    - Raises `RangeNotSatisfiableError` (HTTP 416) if the range is syntactically invalid for `bytes=`,
      if `start >= total_size`, `start > end`, or if a suffix range requests 0 bytes on an empty resource.
    """
    if total_size <= 0:
        raise RangeNotSatisfiableError("Resource is empty (0 bytes).", total_size=0)

    if not range_header or not range_header.strip():
        return ByteRange(start=0, end=total_size - 1, total_size=total_size, is_partial=False)

    cleaned = range_header.strip()
    if not cleaned.startswith("bytes="):
        return ByteRange(start=0, end=total_size - 1, total_size=total_size, is_partial=False)

    match = _RANGE_RE.match(cleaned)
    if not match:
        raise RangeNotSatisfiableError(f"Malformed Range header: {cleaned!r}", total_size=total_size)

    start_str, end_str = match.group(1), match.group(2)
    if not start_str and not end_str:
        raise RangeNotSatisfiableError("Empty byte range specification.", total_size=total_size)

    if not start_str:
        # Suffix byte-range-spec: "bytes=-500" (last N bytes)
        suffix_length = int(end_str)
        if suffix_length <= 0:
            raise RangeNotSatisfiableError("Suffix range length must be > 0.", total_size=total_size)
        start = max(0, total_size - suffix_length)
        end = total_size - 1
    else:
        start = int(start_str)
        if start >= total_size:
            raise RangeNotSatisfiableError(
                f"Range start {start} exceeds resource size {total_size}.",
                total_size=total_size,
            )
        if end_str:
            end = int(end_str)
            if end < start:
                raise RangeNotSatisfiableError(
                    f"Range end {end} is less than range start {start}.",
                    total_size=total_size,
                )
            end = min(end, total_size - 1)
        else:
            if max_open_chunk_size is not None and max_open_chunk_size > 0:
                end = min(start + max_open_chunk_size - 1, total_size - 1)
            else:
                end = total_size - 1

    return ByteRange(start=start, end=end, total_size=total_size, is_partial=True)


class AudioSourceReader:
    """
    Streaming audio reader supporting arbitrary byte-range requests (HTTP 206)
    with on-the-fly AES-128-CTR decryption in fixed-size chunks (default 64 KB).
    Transparently supports local filesystem paths, Django storage backends (S3),
    and remote HTTP/HTTPS URLs.
    """

    def __init__(
        self,
        raw_file_or_url: str,
        storage_backend: Optional[object] = None,
        http_getter: Callable[..., requests.Response] = requests.get,
        http_header: Callable[..., requests.Response] = requests.head,
    ) -> None:
        raw_str = (raw_file_or_url or "").strip()
        if not raw_str:
            raise ValueError("No audio file path or URL provided.")

        self._storage = storage_backend if storage_backend is not None else django_default_storage
        self._http_get = http_getter
        self._http_head = http_header

        self.local_path = resolve_local_audio_path(raw_str)
        self.storage_name: Optional[str] = None
        self.url: Optional[str] = None
        self.total_size: int = 0
        self.header_bytes: bytes = b""

        if self.local_path and os.path.exists(self.local_path):
            self.total_size = os.path.getsize(self.local_path)
            with open(self.local_path, "rb") as f:
                self.header_bytes = f.read(HEADER_LEN + IV_LEN)
        elif raw_str.startswith(("http://", "https://")):
            self.url = raw_str
            self.header_bytes, self.total_size = self._probe_remote_url(self.url)
        else:
            cleaned = raw_str.lstrip("/")
            if self._storage is not None and self._storage.exists(cleaned):
                self.storage_name = cleaned
                self.total_size = self._storage.size(cleaned)
                with self._storage.open(cleaned, "rb") as f:
                    self.header_bytes = f.read(HEADER_LEN + IV_LEN)
            else:
                raise FileNotFoundError(f"Cannot resolve audio source: {raw_str}")

        # A valid encrypted MSD1 file must have at least the 20-byte header + IV
        self.is_encrypted = (
            self.total_size >= (HEADER_LEN + IV_LEN)
            and len(self.header_bytes) >= (HEADER_LEN + IV_LEN)
            and is_encrypted_audio(self.header_bytes)
        )
        if self.is_encrypted:
            self.iv: Optional[bytes] = self.header_bytes[HEADER_LEN : HEADER_LEN + IV_LEN]
            self.plaintext_size = max(0, self.total_size - (HEADER_LEN + IV_LEN))
        else:
            self.iv = None
            self.plaintext_size = self.total_size

    def _probe_remote_url(self, url: str) -> Tuple[bytes, int]:
        """
        Fetches only the first 20 bytes (MSD1 header + IV) using a streaming Range request
        and determines the total remote file size without downloading the body if the origin
        ignores the Range header (returns HTTP 200).
        """
        probe_len = HEADER_LEN + IV_LEN
        with self._http_get(
            url,
            headers={"Range": f"bytes=0-{probe_len - 1}"},
            stream=True,
            timeout=15,
        ) as resp:
            resp.raise_for_status()

            header_chunks = []
            bytes_collected = 0
            for chunk in resp.iter_content(chunk_size=probe_len):
                if not chunk:
                    break
                needed = probe_len - bytes_collected
                header_chunks.append(chunk[:needed])
                bytes_collected += min(len(chunk), needed)
                if bytes_collected >= probe_len:
                    break
            header_bytes = b"".join(header_chunks)

            content_range = resp.headers.get("Content-Range")
            if content_range and "/" in content_range:
                size_part = content_range.split("/")[-1].strip()
                if size_part.isdigit():
                    return header_bytes, int(size_part)

            if resp.status_code == 200 and resp.headers.get("Content-Length"):
                return header_bytes, int(resp.headers["Content-Length"])

        head_resp = self._http_head(url, timeout=15)
        head_resp.raise_for_status()
        total_size = int(head_resp.headers.get("Content-Length", len(header_bytes)))
        return header_bytes, total_size

    def _read_raw_chunks(
        self,
        read_start: int,
        total_bytes: int,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Iterator[bytes]:
        """Streams raw ciphertext/plaintext bytes from local disk, storage backend, or remote URL."""
        if total_bytes <= 0:
            return

        if self.local_path:
            with open(self.local_path, "rb") as f:
                f.seek(read_start)
                rem = total_bytes
                while rem > 0:
                    chunk = f.read(min(rem, chunk_size))
                    if not chunk:
                        break
                    rem -= len(chunk)
                    yield chunk

        elif self.storage_name and self._storage is not None:
            with self._storage.open(self.storage_name, "rb") as f:
                f.seek(read_start)
                rem = total_bytes
                while rem > 0:
                    chunk = f.read(min(rem, chunk_size))
                    if not chunk:
                        break
                    rem -= len(chunk)
                    yield chunk

        elif self.url:
            read_end = read_start + total_bytes - 1
            with self._http_get(
                self.url,
                headers={"Range": f"bytes={read_start}-{read_end}"},
                stream=True,
                timeout=30,
            ) as resp:
                resp.raise_for_status()
                # If the upstream server honours the Range header (HTTP 206), stream directly.
                # If the upstream server ignores Range and returns HTTP 200, skip the leading
                # `read_start` bytes and stop after `total_bytes` so the AES-CTR keystream stays aligned.
                skip_upstream = 0 if resp.status_code == 206 else read_start
                rem = total_bytes

                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    if skip_upstream > 0:
                        if len(chunk) <= skip_upstream:
                            skip_upstream -= len(chunk)
                            continue
                        chunk = chunk[skip_upstream:]
                        skip_upstream = 0

                    if len(chunk) > rem:
                        chunk = chunk[:rem]

                    if chunk:
                        rem -= len(chunk)
                        yield chunk

                    if rem <= 0:
                        break

    def read_range(
        self,
        start: int,
        end: int,
        key: Optional[bytes] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> Iterator[bytes]:
        """
        Yields decrypted plaintext chunks for the inclusive byte range [start, end].

        For AES-128-CTR encrypted files:
          1. Aligns the read offset to the enclosing 16-byte AES block boundary (`block_idx = start // 16`).
          2. Computes the block's 128-bit counter in O(1) time: `(IV + block_idx) mod 2^128`.
          3. Streams and decrypts in `chunk_size` windows, discarding the initial `start % 16` alignment bytes
             and truncating at `end - start + 1` bytes.
        """
        if start < 0 or start > end or start >= self.plaintext_size:
            return

        end = min(end, self.plaintext_size - 1)
        length = end - start + 1

        if not self.is_encrypted:
            yield from self._read_raw_chunks(start, length, chunk_size)
            return

        aes_key = key if key is not None else get_audio_encryption_key()
        if len(aes_key) != KEY_LEN:
            raise ValueError(f"AES-128 key must be {KEY_LEN} bytes, got {len(aes_key)}.")

        assert self.iv is not None
        block_idx = start // AES_BLOCK_SIZE
        block_offset = start % AES_BLOCK_SIZE
        read_start = (HEADER_LEN + IV_LEN) + (block_idx * AES_BLOCK_SIZE)
        bytes_to_read = length + block_offset

        counter_bytes = compute_ctr_block_counter(self.iv, block_idx)
        cipher = Cipher(algorithms.AES(aes_key), modes.CTR(counter_bytes))
        decryptor = cipher.decryptor()

        skip_bytes = block_offset
        rem_to_yield = length

        for raw_chunk in self._read_raw_chunks(read_start, bytes_to_read, chunk_size):
            dec = decryptor.update(raw_chunk)
            if skip_bytes > 0:
                if len(dec) <= skip_bytes:
                    skip_bytes -= len(dec)
                    continue
                dec = dec[skip_bytes:]
                skip_bytes = 0

            if len(dec) > rem_to_yield:
                dec = dec[:rem_to_yield]

            if dec:
                rem_to_yield -= len(dec)
                yield dec

            if rem_to_yield <= 0:
                break

        tail = decryptor.finalize()
        if tail and rem_to_yield > 0:
            if skip_bytes < len(tail):
                tail = tail[skip_bytes:]
                yield tail[:rem_to_yield]
