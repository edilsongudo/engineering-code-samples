import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from audio_crypto import (
    AES_BLOCK_SIZE,
    HEADER_LEN,
    IV_LEN,
    MAGIC_HEADER,
    AudioSourceReader,
    RangeNotSatisfiableError,
    compute_ctr_block_counter,
    decrypt_audio_bytes,
    encrypt_audio_bytes,
    get_audio_encryption_key,
    get_stream_key_hex,
    get_track_hls_key,
    is_encrypted_audio,
    parse_http_range_header,
)


class TestAudioCrypto(unittest.TestCase):
    def setUp(self) -> None:
        self.key_hex = "000102030405060708090a0b0c0d0e0f"
        self.key_bytes = bytes.fromhex(self.key_hex)
        os.environ["AUDIO_ENCRYPTION_KEY"] = self.key_hex

    def test_key_derivation_and_track_hls_keys(self) -> None:
        self.assertEqual(get_audio_encryption_key(), self.key_bytes)
        self.assertEqual(get_stream_key_hex(), self.key_hex)

        k1 = get_track_hls_key(101)
        k2 = get_track_hls_key(102)
        self.assertEqual(len(k1), 16)
        self.assertEqual(len(k2), 16)
        self.assertNotEqual(k1, k2)
        self.assertEqual(get_track_hls_key(101), k1)

    def test_encrypt_and_decrypt_roundtrip(self) -> None:
        plaintext = os.urandom(2049)  # Unaligned length (not a multiple of 16)
        encrypted = encrypt_audio_bytes(plaintext, key=self.key_bytes)

        self.assertTrue(is_encrypted_audio(encrypted))
        self.assertEqual(encrypted[:HEADER_LEN], MAGIC_HEADER)
        self.assertEqual(len(encrypted), HEADER_LEN + IV_LEN + len(plaintext))

        # Idempotent when already encrypted
        self.assertEqual(encrypt_audio_bytes(encrypted, key=self.key_bytes), encrypted)

        decrypted = decrypt_audio_bytes(encrypted, key=self.key_bytes)
        self.assertEqual(decrypted, plaintext)

    def test_unencrypted_legacy_passthrough(self) -> None:
        raw_mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x1f" + os.urandom(500)
        self.assertFalse(is_encrypted_audio(raw_mp3))
        self.assertEqual(decrypt_audio_bytes(raw_mp3, key=self.key_bytes), raw_mp3)

    def test_empty_and_corrupt_buffers(self) -> None:
        self.assertEqual(encrypt_audio_bytes(b"", key=self.key_bytes), b"")
        self.assertEqual(decrypt_audio_bytes(b"", key=self.key_bytes), b"")

        # Truncated MSD1 buffer (< 20 bytes)
        with self.assertRaises(ValueError):
            decrypt_audio_bytes(b"MSD1short", key=self.key_bytes)

    def test_ctr_counter_wrapping_at_128_bit_boundary(self) -> None:
        max_iv = b"\xff" * 16
        wrapped = compute_ctr_block_counter(max_iv, 1)
        self.assertEqual(wrapped, b"\x00" * 16)
        wrapped_two = compute_ctr_block_counter(max_iv, 2)
        self.assertEqual(wrapped_two, (b"\x00" * 15) + b"\x01")

    def test_random_access_range_decryption_aligned_and_unaligned(self) -> None:
        # 10,003 bytes of deterministic pseudo-random data
        plaintext = bytes((i * 73 + 19) % 256 for i in range(10003))
        encrypted = encrypt_audio_bytes(plaintext, key=self.key_bytes)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(encrypted)
            tmp_path = tmp.name

        try:
            reader = AudioSourceReader(tmp_path)
            self.assertTrue(reader.is_encrypted)
            self.assertEqual(reader.plaintext_size, len(plaintext))

            test_ranges = [
                (0, 0),                      # Single first byte
                (0, 15),                     # Exact first AES block
                (16, 31),                    # Exact second AES block
                (1, 14),                     # Unaligned sub-block inside block 0
                (7, 23),                     # Unaligned crossing block 0 -> block 1 boundary
                (15, 16),                    # Last byte of block 0 and first byte of block 1
                (133, 4099),                 # Multi-block unaligned slice
                (len(plaintext) - 1, len(plaintext) - 1),  # Final single byte
                (len(plaintext) - 37, len(plaintext) - 1), # Unaligned tail to EOF
                (0, len(plaintext) - 1),     # Full file
            ]

            for start, end in test_ranges:
                # Test with small 37-byte chunks to stress chunk boundary + block offset logic
                decrypted_slice = b"".join(
                    reader.read_range(start, end, key=self.key_bytes, chunk_size=37)
                )
                expected_slice = plaintext[start : end + 1]
                self.assertEqual(
                    decrypted_slice,
                    expected_slice,
                    f"Mismatch on range [{start}, {end}]",
                )

            # Out-of-bounds or inverted ranges to read_range yield empty
            self.assertEqual(b"".join(reader.read_range(-5, 10, key=self.key_bytes)), b"")
            self.assertEqual(b"".join(reader.read_range(100, 50, key=self.key_bytes)), b"")
            self.assertEqual(
                b"".join(reader.read_range(len(plaintext), len(plaintext) + 10, key=self.key_bytes)),
                b"",
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_http_range_header_parsing(self) -> None:
        total = 5000

        # No range header -> full file, is_partial=False
        r_full = parse_http_range_header(None, total)
        self.assertFalse(r_full.is_partial)
        self.assertEqual((r_full.start, r_full.end), (0, 4999))

        # Standard explicit range
        r_explicit = parse_http_range_header("bytes=100-499", total)
        self.assertTrue(r_explicit.is_partial)
        self.assertEqual((r_explicit.start, r_explicit.end, r_explicit.content_length), (100, 499, 400))
        self.assertEqual(r_explicit.content_range_header, "bytes 100-499/5000")

        # Open range with chunk cap
        r_open = parse_http_range_header("bytes=1000-", total, max_open_chunk_size=1024)
        self.assertEqual((r_open.start, r_open.end), (1000, 2023))

        # Suffix range
        r_suffix = parse_http_range_header("bytes=-500", total)
        self.assertEqual((r_suffix.start, r_suffix.end), (4500, 4999))

        # Clamping end > total_size - 1
        r_clamp = parse_http_range_header("bytes=4900-9999", total)
        self.assertEqual((r_clamp.start, r_clamp.end), (4900, 4999))

        # Invalid / unsatisfiable ranges must raise RangeNotSatisfiableError (HTTP 416)
        invalid_headers = [
            "bytes=5000-5010",  # start >= total_size
            "bytes=500-100",    # end < start
            "bytes=-0",         # 0-length suffix
            "bytes=-",          # empty start and end
            "bytes=abc-def",    # non-numeric
            "bytes=0-10, 20-30",  # multi-part range not supported
        ]
        for hdr in invalid_headers:
            with self.assertRaises(RangeNotSatisfiableError, msg=f"Expected 416 for {hdr}"):
                parse_http_range_header(hdr, total)

    def test_remote_url_fallback_when_upstream_returns_200_instead_of_206(self) -> None:
        """
        Verifies that if an upstream HTTP server ignores the `Range` header and returns
        HTTP 200 with the full file body, `AudioSourceReader` still extracts only the
        requested byte slice without corrupting the AES-CTR keystream alignment.
        """
        plaintext = bytes((i * 31 + 7) % 256 for i in range(512))
        encrypted = encrypt_audio_bytes(plaintext, key=self.key_bytes)

        def fake_http_get(url: str, headers=None, stream=False, timeout=15):
            resp = MagicMock()
            resp.status_code = 200  # Server ignored Range header!
            resp.headers = {"Content-Length": str(len(encrypted))}
            resp.raise_for_status = MagicMock()
            resp.iter_content = lambda chunk_size=64: (
                encrypted[i : i + chunk_size] for i in range(0, len(encrypted), chunk_size)
            )
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        reader = AudioSourceReader(
            "https://cdn.example.com/audio/track_1.mp3",
            http_getter=fake_http_get,
        )
        self.assertTrue(reader.is_encrypted)
        self.assertEqual(reader.plaintext_size, len(plaintext))

        # Unaligned mid-stream seek
        start, end = 73, 290
        result = b"".join(reader.read_range(start, end, key=self.key_bytes, chunk_size=32))
        self.assertEqual(result, plaintext[start : end + 1])


if __name__ == "__main__":
    unittest.main()
