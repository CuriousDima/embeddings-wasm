# e5-wasm (Candle + Chrome Extension)

This project runs **`intfloat/e5-small-v2`** embedding inference fully local in Chrome:

- Rust inference engine built with HuggingFace **Candle**.
- Compiled to **WebAssembly** via `wasm-pack`.
- Loaded from a Chrome extension popup using Chrome APIs (`chrome.runtime.getURL`).
- Model + tokenizer are loaded locally from extension-packaged files (`model/*`).

## 1) Build WASM

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build --target web --release --out-dir chrome-extension/pkg
```

## 2) Download the E5-small-v2 files

```bash
./scripts/download_e5_small_v2.sh chrome-extension/model
```

This downloads:

- `config.json`
- `tokenizer.json`
- `model.safetensors`

## 3) Load in Chrome

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select `chrome-extension/`
5. Open the extension popup and click **Generate embedding**

## How Chrome APIs are used

`chrome-extension/popup.js` uses:

- `chrome.runtime.getURL("pkg/e5_wasm.js")` to load the generated JS/WASM wrapper.
- `chrome.runtime.getURL("pkg/e5_wasm_bg.wasm")` to initialize WASM bytes.
- `chrome.runtime.getURL("model")` as the base path for local model files.

Inside WASM, Rust calls browser `fetch` (`web-sys`) to load:

- `model/config.json`
- `model/tokenizer.json`
- `model/model.safetensors`

No server-side inference is required.

## Rust API

WASM exports `E5Embedder`:

- `E5Embedder.fromModelBaseUrl(baseUrl)` → async constructor.
- `embed(["query: ..."])` → returns `{ dimensions, vectors }`.

Embedding logic:

1. Tokenize with `tokenizers`.
2. Run BERT forward pass with Candle (E5 backbone).
3. Mean-pool with attention mask.
4. L2 normalize output vectors.

## Notes

- First run may take a while due to model weight loading.
- Keep model files local to satisfy local-only inference requirement.
