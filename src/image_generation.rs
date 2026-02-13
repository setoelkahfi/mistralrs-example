use anyhow::Result;
use mistralrs::{
    DiffusionGenerationParams, DiffusionLoaderType, DiffusionModelBuilder,
    ImageGenerationResponseFormat, ModelDType,
};
use std::time::{Duration, Instant};

use crate::promp_enhancer::{EnhancerModel, PromptEnhancer};

/// Maximum number of whitespace-separated words to send to the diffusion model.
/// CLIP (used by FLUX.1-schnell) has a hard 77-token limit; keeping prompts
/// under 50 words provides safe headroom for BOS/EOS and sub-word splits.
const MAX_PROMPT_WORDS: usize = 50;

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

const DEFAULT_MODEL: &str = "black-forest-labs/FLUX.1-schnell";
const DEFAULT_LOADER: DiffusionLoaderType = DiffusionLoaderType::FluxOffloaded;

/// Run image generation, optionally enhancing a seed prompt first.
///
/// - If `prompt` is provided it is used directly (no enhancement).
/// - If `seed` is provided the prompt enhancer expands it before generation.
/// - If neither is provided a built-in default prompt is used.
pub async fn run(
    prompt: Option<String>,
    seed: Option<String>,
    enhancer_model: Option<EnhancerModel>,
) -> Result<()> {
    // ── Resolve the final prompt ────────────────────────────────────────
    let prompt = if let Some(p) = prompt {
        // Direct prompt — use as-is.
        p
    } else if let Some(seed_text) = seed {
        // Seed provided — enhance it first.
        let preset = enhancer_model.unwrap_or_default();
        println!("Loading prompt enhancer model: {preset}");
        println!("  Memory estimate: {}", preset.approx_memory());
        let enhancer_start = Instant::now();
        let enhancer = PromptEnhancer::from_preset(preset).await?;
        let enhancer_load = enhancer_start.elapsed();
        println!("Prompt enhancer loaded in {}", fmt_duration(enhancer_load));

        println!("\nSeed prompt:\n  \"{seed_text}\"\n");

        let enhance_start = Instant::now();
        let enhanced = enhancer.enhance(&seed_text).await?;
        let enhance_elapsed = enhance_start.elapsed();

        println!(
            "Enhanced prompt ({}):\n  \"{enhanced}\"\n",
            fmt_duration(enhance_elapsed)
        );
        enhanced
    } else {
        // Fallback default.
        "A majestic castle on a cliff overlooking the sea at sunset, \
         highly detailed, digital painting, trending on artstation, in the style of Raden Saleh"
            .to_string()
    };

    // ── Load diffusion model ────────────────────────────────────────────
    println!("Loading diffusion model ({DEFAULT_MODEL})...");
    let load_start = Instant::now();
    let model = DiffusionModelBuilder::new(DEFAULT_MODEL, DEFAULT_LOADER)
        .with_dtype(ModelDType::BF16)
        .with_logging()
        .build()
        .await?;
    let load_elapsed = load_start.elapsed();
    println!("Model loaded in {}", fmt_duration(load_elapsed));

    // ── Truncate to fit CLIP's 77-token window ──────────────────────────
    let prompt = truncate_to_words(&prompt, MAX_PROMPT_WORDS);

    // ── Generate image ──────────────────────────────────────────────────
    println!("\nGenerating image for prompt:\n  \"{prompt}\"");

    let start = Instant::now();
    let response = model
        .generate_image(
            &prompt,
            ImageGenerationResponseFormat::Url,
            DiffusionGenerationParams::default(),
        )
        .await?;
    let elapsed = start.elapsed();

    let path = response.data[0]
        .url
        .as_ref()
        .expect("expected image URL in response");

    println!(
        "Done! Image generation took {}.\nImage saved at: {path}",
        fmt_duration(elapsed)
    );

    Ok(())
}

/// Truncate `text` to at most `max_words` whitespace-separated words.
///
/// Acts as a final safety net so prompts never exceed CLIP's 77-token limit.
fn truncate_to_words(text: &str, max_words: usize) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() <= max_words {
        return text.to_string();
    }
    words[..max_words].join(" ")
}
