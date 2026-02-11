#![allow(dead_code)]

use anyhow::Result;
use mistralrs::{IsqType, Model, RequestBuilder, TextMessageRole, TextModelBuilder};
use std::time::Instant;

const DEFAULT_MODEL: &str = "microsoft/Phi-3.5-mini-instruct";

const SYSTEM_PROMPT: &str = r#"You are a prompt enhancer for image generation models. Given a short description, expand it into a detailed, vivid, and creative image generation prompt. Keep the artistic style references if provided. Add details about lighting, composition, medium, and atmosphere. Output ONLY the enhanced prompt text, nothing else. Do not include any explanation, preamble, or quotes."#;

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
            .with_isq(IsqType::Q4K)
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
            .set_sampler_max_len(150)
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
            Ok(seed_prompt.to_string())
        } else {
            Ok(enhanced)
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
    println!("Model loaded in {:.1}s", load_elapsed.as_secs_f32());

    println!("\nSeed prompt:\n  \"{seed}\"\n");

    let enhance_start = Instant::now();
    let enhanced = enhancer.enhance(&seed).await?;
    let enhance_elapsed = enhance_start.elapsed();

    println!("Enhanced prompt ({:.1}s):", enhance_elapsed.as_secs_f32());
    println!("  \"{enhanced}\"");

    Ok(())
}
