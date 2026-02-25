# e5-wasm (Candle + Browser App)

This project runs **`intfloat/e5-small-v2`** embedding inference fully local in a small browser app:

- Rust inference engine built with HuggingFace **Candle**.
- Compiled to **WebAssembly** via `wasm-pack`.
- Loaded from a simple static HTML/JS app in `web-app/`.
- Model + tokenizer are loaded locally from `web-app/model/*` over HTTP.

## 1) Build WASM

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build --target web --release --out-dir web-app/pkg
```

## 2) Download E5-small-v2 model files

```bash
./scripts/download_e5_small_v2.sh web-app/model
```

This downloads:

- `config.json`
- `tokenizer.json`
- `model.safetensors`

## 3) Serve the app (Python HTTP)

```bash
python3 scripts/serve_web_app.py
```

Then open: <http://127.0.0.1:8000>

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
