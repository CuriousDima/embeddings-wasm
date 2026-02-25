use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use js_sys::{Array, Promise};
use serde::Serialize;
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

#[derive(Debug, Serialize)]
struct EmbeddingResult {
    dimensions: usize,
    vectors: Vec<Vec<f32>>,
}

#[wasm_bindgen]
pub struct E5Embedder {
    tokenizer: Tokenizer,
    model: BertModel,
    device: Device,
}

#[wasm_bindgen]
impl E5Embedder {
    #[wasm_bindgen(js_name = fromModelBaseUrl)]
    pub fn from_model_base_url(model_base_url: String) -> Promise {
        wasm_bindgen_futures::future_to_promise(async move {
            let embedder = Self::load(&model_base_url)
                .await
                .map_err(|e| JsValue::from_str(&format!("failed to initialize E5 embedder: {e:#}")))?;
            Ok(JsValue::from(embedder))
        })
    }

    #[wasm_bindgen]
    pub fn embed(&self, input_texts: Array) -> Result<JsValue, JsValue> {
        let texts: Vec<String> = input_texts
            .iter()
            .map(|v| {
                v.as_string()
                    .ok_or_else(|| JsValue::from_str("all inputs must be strings"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let vectors = embed_with_e5(&self.tokenizer, &self.model, &self.device, &texts)
            .map_err(|e| JsValue::from_str(&format!("embedding failed: {e:#}")))?;

        let result = EmbeddingResult {
            dimensions: vectors.first().map(|v| v.len()).unwrap_or_default(),
            vectors,
        };

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
    }
}

impl E5Embedder {
    async fn load(model_base_url: &str) -> Result<Self> {
        let config_bytes = fetch_binary(&format!("{model_base_url}/config.json")).await?;
        let tokenizer_bytes = fetch_binary(&format!("{model_base_url}/tokenizer.json")).await?;
        let weights_bytes = fetch_binary(&format!("{model_base_url}/model.safetensors")).await?;

        let config: BertConfig =
            serde_json::from_slice(&config_bytes).context("could not parse config.json")?;

        let mut tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("could not build tokenizer")?;

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 512,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            ..Default::default()
        }));

        let device = Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(weights_bytes, DType::F32, &device)
            .context("failed to read model.safetensors")?;
        let model = BertModel::load(vb, &config).context("failed to load BERT model")?;

        Ok(Self {
            tokenizer,
            model,
            device,
        })
    }
}

fn embed_with_e5(
    tokenizer: &Tokenizer,
    model: &BertModel,
    device: &Device,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let prefixed: Vec<String> = texts
        .iter()
        .map(|text| {
            if text.starts_with("query: ") || text.starts_with("passage: ") {
                text.clone()
            } else {
                format!("query: {text}")
            }
        })
        .collect();

    let encodings = tokenizer
        .encode_batch(prefixed, true)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?;

    let token_ids: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_ids().to_vec()).collect();
    let attention: Vec<Vec<u32>> = encodings.iter().map(|e| e.get_attention_mask().to_vec()).collect();

    let input_ids = Tensor::new(token_ids, device)?;
    let token_type_ids = input_ids.zeros_like()?;
    let attention_mask = Tensor::new(attention, device)?;

    let hidden = model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
    let pooled = masked_mean_pooling(hidden, attention_mask)?;
    let normalized = l2_normalize(pooled)?;

    Ok(normalized.to_vec2::<f32>()?)
}

fn masked_mean_pooling(token_embeddings: Tensor, attention_mask: Tensor) -> Result<Tensor> {
    let expanded = attention_mask.unsqueeze(2)?;
    let masked = token_embeddings.broadcast_mul(&expanded)?;

    let sum_embeddings = masked.sum(1)?;
    let sum_mask = expanded.sum(1)?.clamp(1e-9f64, f64::INFINITY)?;

    Ok(sum_embeddings.broadcast_div(&sum_mask)?)
}

fn l2_normalize(embeddings: Tensor) -> Result<Tensor> {
    let squared = embeddings.sqr()?;
    let summed = squared.sum(1)?;
    let norms = summed.sqrt()?.unsqueeze(1)?;
    Ok(embeddings.broadcast_div(&norms)?)
}

async fn fetch_binary(url: &str) -> Result<Vec<u8>> {
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts)
        .map_err(|e| anyhow::anyhow!("request construction failed: {e:?}"))?;

    let window = web_sys::window().context("window was not available")?;
    let response_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| anyhow::anyhow!("fetch failed: {e:?}"))?;
    let response: Response = response_value
        .dyn_into()
        .map_err(|_| anyhow::anyhow!("failed to cast fetch response"))?;

    if !response.ok() {
        return Err(anyhow::anyhow!(
            "download failed for {url}: HTTP {}",
            response.status()
        ));
    }

    let array_buffer = JsFuture::from(
        response
            .array_buffer()
            .map_err(|_| anyhow::anyhow!("array_buffer() failed"))?,
    )
    .await
    .map_err(|e| anyhow::anyhow!("array_buffer await failed: {e:?}"))?;

    let data = js_sys::Uint8Array::new(&array_buffer);
    Ok(data.to_vec())
}
