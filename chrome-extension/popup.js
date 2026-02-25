const output = document.getElementById("output");
const runBtn = document.getElementById("run");
const input = document.getElementById("input");

let embedder;

async function init() {
  output.textContent = "Loading WASM module...";

  const wasmEntrypoint = chrome.runtime.getURL("pkg/e5_wasm.js");
  const wasmModule = await import(wasmEntrypoint);

  await wasmModule.default(chrome.runtime.getURL("pkg/e5_wasm_bg.wasm"));

  output.textContent = "Loading tokenizer + model weights (first run can take time)...";

  const modelBaseUrl = chrome.runtime.getURL("model");
  embedder = await wasmModule.E5Embedder.fromModelBaseUrl(modelBaseUrl);

  output.textContent = "Ready. Click \"Generate embedding\".";
}

runBtn.addEventListener("click", () => {
  if (!embedder) {
    output.textContent = "Still loading model...";
    return;
  }

  const text = input.value.trim();
  if (!text) {
    output.textContent = "Please enter text.";
    return;
  }

  const result = embedder.embed([`query: ${text}`]);
  const vector = result.vectors[0] || [];

  output.textContent = JSON.stringify(
    {
      dimensions: result.dimensions,
      first10: vector.slice(0, 10),
    },
    null,
    2,
  );
});

init().catch((err) => {
  output.textContent = `Initialization failed:\n${err?.stack || err}`;
  console.error(err);
});
