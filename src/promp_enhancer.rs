#![allow(dead_code)]

use anyhow::Result;
use mistralrs::{Model, ModelDType, RequestBuilder, TextMessageRole, TextModelBuilder};
use std::time::{Duration, Instant};

/// Format a `Duration` as `Xm Ys` (e.g. "2m 30.5s") or just `Ys` when under a minute.
fn fmt_duration(d: Duration) -> String {
    let total_secs = d.as_secs_f64();
    let mins = (total_secs / 60.0).floor() as u64;
    let secs = total_secs - (mins as f64 * 60.0);
    if mins > 0 {
        format!("{}m {:.1}s", mins, secs)
    } else {
        format!("{:.1}s", secs)
    }
}

const DEFAULT_MODEL: &str = "microsoft/Phi-3.5-mini-instruct";

/// CLIP (used by FLUX.1-schnell) has a hard limit of 77 tokens (including
/// BOS/EOS), so the enhanced prompt must stay under ~50 words to be safe.
const SYSTEM_PROMPT: &str = r#"You are a prompt enhancer for image generation models. Given a short description, expand it into a vivid image generation prompt. Keep artistic style references if provided. Add lighting, composition, and atmosphere details. The result MUST be under 50 words. Output ONLY the enhanced prompt, no explanation, no quotes."#;

/// Maximum number of CLIP tokens the diffusion model accepts (including BOS/EOS).
const MAX_CLIP_TOKENS: usize = 77;

/// Conservative word-count ceiling so the prompt fits within [`MAX_CLIP_TOKENS`].
/// CLIP roughly tokenises at the word level; 50 words ≈ 55-65 CLIP tokens,
/// leaving headroom for BOS/EOS and occasional sub-word splits.
const MAX_PROMPT_WORDS: usize = 50;

/// A self-contained prompt enhancer that owns a text generation model.
///
/// Replicates the behavior of `Gustavosta/MagicPrompt-Stable-Diffusion` (a GPT-2
/// fine-tune) by using a small instruction-following model with a system prompt
/// that instructs it to expand short descriptions into rich image generation prompts.
pub struct PromptEnhancer {
    model: Model,
    system_prompt: String,
}

impl PromptEnhancer {
    /// Build a new `PromptEnhancer` using the default model (`microsoft/Phi-3.5-mini-instruct`)
    /// with Q4K in-situ quantization.
    pub async fn new() -> Result<Self> {
        Self::with_model(DEFAULT_MODEL).await
    }

    /// Build a `PromptEnhancer` with a specific HuggingFace model ID.
    ///
    /// The model must be a text/instruction model supported by mistral.rs
    /// (e.g. Phi-3, Qwen2, Llama, Mistral).
    pub async fn with_model(model_id: &str) -> Result<Self> {
        let model = TextModelBuilder::new(model_id)
            .with_dtype(ModelDType::F16)
            .with_logging()
            .build()
            .await?;

        Ok(Self {
            model,
            system_prompt: SYSTEM_PROMPT.to_string(),
        })
    }

    /// Override the default system prompt used for enhancement.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Enhance a seed prompt into a detailed image generation prompt.
    ///
    /// If the model fails to produce a meaningful expansion (result is too short
    /// or identical to input), the original seed prompt is returned as-is.
    pub async fn enhance(&self, seed_prompt: &str) -> Result<String> {
        let request = RequestBuilder::new()
            .set_sampler_temperature(0.9)
            .set_sampler_topp(0.95)
            // Keep generation short so the result fits within CLIP's 77-token
            // window after tokenisation.
            .set_sampler_max_len(80)
            .add_message(TextMessageRole::System, &self.system_prompt)
            .add_message(TextMessageRole::User, seed_prompt);

        let response = self.model.send_chat_request(request).await?;

        let enhanced = response.choices[0]
            .message
            .content
            .as_ref()
            .map(|c| c.trim().to_string())
            .unwrap_or_default();

        // Fallback to the seed prompt if the model returned something too short
        if enhanced.len() <= seed_prompt.len() + 4 {
            Ok(truncate_to_words(seed_prompt, MAX_PROMPT_WORDS))
        } else {
            Ok(truncate_to_words(&enhanced, MAX_PROMPT_WORDS))
        }
    }

    /// Build a seed prompt from a song title and style descriptor,
    /// then enhance it.
    ///
    /// This is a convenience wrapper matching the Python
    /// `generate_improved_prompt` workflow.
    pub async fn enhance_for_song(&self, song_title: &str, style: Option<&str>) -> Result<String> {
        let seed = match style {
            Some(s) => format!("{song_title}, {s}"),
            None => song_title.to_string(),
        };
        self.enhance(&seed).await
    }

    /// Return a reference to the underlying `Model` (e.g. for reuse or inspection).
    pub fn model(&self) -> &Model {
        &self.model
    }
}

/// Truncate `text` to at most `max_words` whitespace-separated words.
///
/// This is a safety net so that prompts never exceed CLIP's 77-token limit.
fn truncate_to_words(text: &str, max_words: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= max_words {
        return text.to_string();
    }
    words[..max_words].join(" ")
}

/// Run the prompt enhancer as a standalone example.
///
/// Loads a text model, takes a seed prompt, and prints the enhanced version.
pub async fn run(prompt: Option<String>) -> Result<()> {
    let seed = prompt.unwrap_or_else(|| {
        "Detective Conan Main Theme, in the style of Raden Saleh, \
         trending on artstation, highly detailed"
            .to_string()
    });

    println!("Loading prompt enhancer model ({DEFAULT_MODEL})...");
    let start = Instant::now();
    let enhancer = PromptEnhancer::new().await?;
    let load_elapsed = start.elapsed();
    println!("Model loaded in {}", fmt_duration(load_elapsed));

    println!("\nSeed prompt:\n  \"{seed}\"\n");

    let enhance_start = Instant::now();
    let enhanced = enhancer.enhance(&seed).await?;
    let enhance_elapsed = enhance_start.elapsed();

    println!("Enhanced prompt ({}):", fmt_duration(enhance_elapsed));
    println!("  \"{enhanced}\"");

    Ok(())
}
