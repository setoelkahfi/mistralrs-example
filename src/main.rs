use anyhow::Result;
use mistralrs::{
    DiffusionGenerationParams, DiffusionLoaderType, DiffusionModelBuilder,
    ImageGenerationResponseFormat,
};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    let model = DiffusionModelBuilder::new(
        "black-forest-labs/FLUX.1-schnell",
        DiffusionLoaderType::FluxOffloaded,
    )
    .with_logging()
    .build()
    .await?;

    let start = Instant::now();
    // let prompt = "Horse, in the style of Raden Saleh, Cedric Peyravernay, Peter Mohrbacher, george clausen, artgerm, mixed media on toned paper, 2021, very detailed, coffee art.";
    // let prompt = "Abdel Kader - Ahmed Alshaiba ft Mazen Samih, Ahmed Mounib, in the style of Raden Saleh, face symmetry, Wadim Kashin, artgerm, face symmetry, trending on artstation";
    let prompt = "Kelana, in the style of Raden Saleh, by Joao Ruas, by Artgerm";

    let response = model
        .generate_image(
            prompt.to_string(),
            ImageGenerationResponseFormat::Url,
            DiffusionGenerationParams::default(),
        )
        .await?;

    let finished = Instant::now();

    println!(
        "Done! Took {} s. Image saved at: {}",
        finished.duration_since(start).as_secs_f32(),
        response.data[0].url.as_ref().unwrap()
    );

    Ok(())
}
