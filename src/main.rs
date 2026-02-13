use anyhow::Result;
use clap::{Parser, Subcommand};

mod image_generation;
mod promp_enhancer;

use promp_enhancer::EnhancerModel;

#[derive(Parser)]
#[command(name = "mistralrs-example")]
#[command(about = "mistral.rs examples — image generation & prompt enhancement")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate an image using a diffusion model (FLUX.1-schnell).
    ///
    /// You can provide a fully-formed prompt with `--prompt`, or a short seed
    /// with `--seed` which will be auto-enhanced by the prompt enhancer before
    /// image generation.
    ///
    /// Examples:
    ///   cargo run -- image
    ///   cargo run -- image --prompt "A cat riding a bicycle on the moon"
    ///   cargo run -- image --seed "lonely astronaut, watercolor"
    ///   cargo run -- image --seed "lonely astronaut" --model gemma-e2b
    Image {
        /// A fully-formed prompt to use directly for image generation.
        /// Mutually exclusive with --seed.
        #[arg(short, long, conflicts_with = "seed")]
        prompt: Option<String>,

        /// A short seed prompt that will be enhanced by the prompt enhancer
        /// before being sent to the diffusion model.
        /// Mutually exclusive with --prompt.
        #[arg(short, long, conflicts_with = "prompt")]
        seed: Option<String>,

        /// Which text model to use for prompt enhancement.
        /// Only used when --seed is provided.
        ///
        /// Possible values:
        ///   gemma-e2b    — Gemma 3n E2B, smallest (~1.5 GB Q4K), best for iPhone
        ///   gemma-e4b    — Gemma 3n E4B, balanced (~8 GB F16) [default]
        ///   phi-3.5-mini — Phi-3.5-mini, strongest quality (~2.8 GB Q4K)
        #[arg(short, long, value_enum)]
        model: Option<EnhancerModel>,
    },

    /// Enhance a short prompt into a detailed image-generation prompt
    /// using a small instruction-following text model.
    ///
    /// Examples:
    ///   cargo run -- prompt
    ///   cargo run -- prompt --seed "A lonely astronaut, watercolor"
    ///   cargo run -- prompt --model gemma-e2b
    ///   cargo run -- prompt --model phi-3.5-mini --seed "cyberpunk city"
    Prompt {
        /// The seed prompt to enhance.
        /// If omitted a default seed is used.
        #[arg(short, long)]
        seed: Option<String>,

        /// Which text model to use for prompt enhancement.
        ///
        /// Possible values:
        ///   gemma-e2b    — Gemma 3n E2B, smallest (~1.5 GB Q4K), best for iPhone
        ///   gemma-e4b    — Gemma 3n E4B, balanced (~8 GB F16) [default]
        ///   phi-3.5-mini — Phi-3.5-mini, strongest quality (~2.8 GB Q4K)
        #[arg(short, long, value_enum)]
        model: Option<EnhancerModel>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Image {
            prompt,
            seed,
            model,
        } => image_generation::run(prompt, seed, model).await,
        Command::Prompt { seed, model } => promp_enhancer::run(seed, model).await,
    }
}
