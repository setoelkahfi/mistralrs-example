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
    /// Example:
    ///   cargo run -- image
    ///   cargo run -- image --prompt "A cat riding a bicycle on the moon"
    Image {
        /// The prompt to use for image generation.
        /// If omitted a default prompt is used.
        #[arg(short, long)]
        prompt: Option<String>,
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
        Command::Image { prompt } => image_generation::run(prompt).await,
        Command::Prompt { seed } => promp_enhancer::run(seed).await,
    }
}
