#![allow(dead_code)]

use anyhow::Result;
use mistralrs::{IsqType, Model, ModelDType, RequestBuilder, TextMessageRole, TextModelBuilder};
use std::fmt;
use std::time::{Duration, Instant};

// ── Model presets ────────────────────────────────────────────────────────────

/// Available prompt-enhancer model presets.
///
/// Each variant carries the HuggingFace model ID and the optimal loading
/// strategy (dtype / in-situ quantization) for its size class.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum EnhancerModel {
    /// Gemma 3n E2B — smallest, best for iPhone / on-device (~1.5 GB with Q4K).
    #[value(name = "gemma-e2b")]
    GemmaE2b,

    /// Gemma 3n E4B — balanced quality & size for macOS desktop (F16).
    #[default]
    #[value(name = "gemma-e4b")]
    GemmaE4b,

    /// Phi-3.5-mini — strongest quality, larger memory footprint (~2.8 GB with Q4K).
    #[value(name = "phi-3.5-mini")]
    Phi35Mini,
}

impl EnhancerModel {
    /// HuggingFace model identifier.
    pub fn model_id(self) -> &'static str {
        match self {
            Self::GemmaE2b => "google/gemma-3n-E2B-it",
            Self::GemmaE4b => "google/gemma-3n-E4B-it",
            Self::Phi35Mini => "microsoft/Phi-3.5-mini-instruct",
        }
    }

    /// Human-readable label used in log messages.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::GemmaE2b => "Gemma 3n E2B",
            Self::GemmaE4b => "Gemma 3n E4B",
            Self::Phi35Mini => "Phi-3.5-mini",
        }
    }

    /// Approximate memory footprint with the chosen loading strategy.
    pub fn approx_memory(self) -> &'static str {
        match self {
            Self::GemmaE2b => "~1.5 GB (Q4K)",
            Self::GemmaE4b => "~8 GB (F16)",
            Self::Phi35Mini => "~2.8 GB (Q4K)",
        }
    }

    /// Build a [`TextModelBuilder`] with the optimal dtype / ISQ settings for
    /// this preset.
    fn configure_builder(self) -> TextModelBuilder {
        let builder = TextModelBuilder::new(self.model_id()).with_logging();

        match self {
            // E2B is the "on-device" pick — quantise aggressively to fit in
            // iPhone memory alongside the diffusion model.
            Self::GemmaE2b => builder.with_dtype(ModelDType::Auto).with_isq(IsqType::Q4K),

            // E4B in full F16 — the sweet spot on a Mac with ≥16 GB RAM.
            Self::GemmaE4b => builder.with_dtype(ModelDType::F16),

            // Phi-3.5-mini at 3.8 B params is too large for F16 on most
            // laptops, so default to Q4K like the upstream examples.
            Self::Phi35Mini => builder.with_dtype(ModelDType::Auto).with_isq(IsqType::Q4K),
        }
    }
}

impl fmt::Display for EnhancerModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.display_name(), self.model_id())
    }
}

// ── Constants ────────────────────────────────────────────────────────────────

/// CLIP (used by FLUX.1-schnell) has a hard limit of 77 tokens (including
/// BOS/EOS), so the enhanced prompt must stay under ~50 words to be safe.
const SYSTEM_PROMPT: &str = r#"You are a prompt enhancer for image generation models. Given a short description, expand it into a vivid image generation prompt. Keep artistic style references if provided. Add lighting, composition, and atmosphere details. The result MUST be under 50 words. Output ONLY the enhanced prompt, no explanation, no quotes."#;

/// Maximum number of CLIP tokens the diffusion model accepts (including BOS/EOS).
const MAX_CLIP_TOKENS: usize = 77;

/// Conservative word-count ceiling so the prompt fits within [`MAX_CLIP_TOKENS`].
/// CLIP roughly tokenises at the word level; 50 words ≈ 55-65 CLIP tokens,
/// leaving headroom for BOS/EOS and occasional sub-word splits.
const MAX_PROMPT_WORDS: usize = 50;

// ── Helpers ──────────────────────────────────────────────────────────────────

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

// ── PromptEnhancer ───────────────────────────────────────────────────────────

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
    /// Build a new `PromptEnhancer` using the **default** preset
    /// ([`EnhancerModel::GemmaE4b`]).
    pub async fn new() -> Result<Self> {
        Self::from_preset(EnhancerModel::default()).await
    }

    /// Build a `PromptEnhancer` from one of the built-in [`EnhancerModel`]
    /// presets.  Each preset applies the optimal dtype / ISQ configuration
    /// automatically.
    pub async fn from_preset(preset: EnhancerModel) -> Result<Self> {
        let model = preset.configure_builder().build().await?;

        Ok(Self {
            model,
            system_prompt: SYSTEM_PROMPT.to_string(),
        })
    }

    /// Build a `PromptEnhancer` with an arbitrary HuggingFace model ID.
    ///
    /// The model must be a text/instruction model supported by mistral.rs
    /// (e.g. Gemma, Qwen2, Llama, Mistral).  Loads with F16 dtype and no ISQ —
    /// use [`from_preset`](Self::from_preset) for optimised defaults.
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

// ── Standalone CLI entry-point ───────────────────────────────────────────────

/// Run the prompt enhancer as a standalone example.
///
/// Loads a text model, takes a seed prompt, and prints the enhanced version.
pub async fn run(prompt: Option<String>, model: Option<EnhancerModel>) -> Result<()> {
    let preset = model.unwrap_or_default();

    let seed = prompt.unwrap_or_else(|| {
        "Detective Conan Main Theme, in the style of Raden Saleh, \
         trending on artstation, highly detailed"
            .to_string()
    });

    println!("Loading prompt enhancer model: {preset}");
    println!("  Memory estimate: {}", preset.approx_memory());
    let start = Instant::now();
    let enhancer = PromptEnhancer::from_preset(preset).await?;
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
