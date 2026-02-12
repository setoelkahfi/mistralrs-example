use anyhow::Result;
use mistralrs::{
    DiffusionGenerationParams, DiffusionLoaderType, DiffusionModelBuilder,
    ImageGenerationResponseFormat,
};
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

const DEFAULT_MODEL: &str = "black-forest-labs/FLUX.1-schnell";
const DEFAULT_LOADER: DiffusionLoaderType = DiffusionLoaderType::FluxOffloaded;

/// Run standalone image generation with a direct prompt (no enhancement).
pub async fn run(prompt: Option<String>) -> Result<()> {
    let prompt = prompt.unwrap_or_else(|| {
        "A majestic castle on a cliff overlooking the sea at sunset, \
         highly detailed, digital painting, trending on artstation, in style of Raden Saleh"
            .to_string()
    });

    println!("Loading diffusion model ({DEFAULT_MODEL})...");
    let model = DiffusionModelBuilder::new(DEFAULT_MODEL, DEFAULT_LOADER)
        .with_logging()
        .build()
        .await?;

    println!("Generating image for prompt:\n  \"{prompt}\"");

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
