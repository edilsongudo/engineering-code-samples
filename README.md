# Selected Engineering Code Samples

These modules were extracted from **Museed**, an audio streaming and creator platform. Each directory isolates a specific subsystem from the codebase along with unit tests.

| Sample | Focus |
|--------|-------|
| [`01-encrypted-streaming/`](./01-encrypted-streaming/) | Random-access AES-128-CTR audio decryption and HTTP `Range` streaming across Python and a browser Service Worker |
| [`02-ai-web-intelligence/`](./02-ai-web-intelligence/) | Decorator-based AI tool registry and web content extraction tools (Google Search, webpages, Reddit threads) |
| [`03-automix-engine/`](./03-automix-engine/) | Track sequencing by musical features, Float32 audio transitions, and streaming output to FFmpeg |

## Running the Tests

Each sample includes a `README.md` covering the problem it solves, how the implementation works, and what changed when extracting it from the full Django/React application.

```bash
# Sample 01 (Python + Node.js 18+)
cd 01-encrypted-streaming
python3 -m unittest discover -s tests -v
node --test tests/test_sw_offline_stream.js

# Sample 02 (Python)
cd ../02-ai-web-intelligence
python3 -m unittest discover -s tests -v

# Sample 03 (Python)
cd ../03-automix-engine
python3 -m unittest discover -s tests -v
```
