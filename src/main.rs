use anyhow::Result;
use clap::{Parser, Subcommand};

mod image_generation;
mod promp_enhancer;

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
    Image {
        /// A fully-formed prompt to use directly for image generation.
        /// Mutually exclusive with --seed.
        #[arg(short, long, conflicts_with = "seed")]
        prompt: Option<String>,

        /// A short seed prompt that will be enhanced by the prompt enhancer
        /// (Phi-3.5-mini) before being sent to the diffusion model.
        /// Mutually exclusive with --prompt.
        #[arg(short, long, conflicts_with = "prompt")]
        seed: Option<String>,
    },

    /// Enhance a short prompt into a detailed image-generation prompt
    /// using a small instruction-following text model (Phi-3.5-mini).
    ///
    /// Example:
    ///   cargo run -- prompt
    ///   cargo run -- prompt --seed "A lonely astronaut, watercolor"
    Prompt {
        /// The seed prompt to enhance.
        /// If omitted a default seed is used.
        #[arg(short, long)]
        seed: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Image { prompt, seed } => image_generation::run(prompt, seed).await,
        Command::Prompt { seed } => promp_enhancer::run(seed).await,
    }
}
