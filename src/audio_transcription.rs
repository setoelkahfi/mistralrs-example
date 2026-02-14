#![allow(dead_code)]

use anyhow::{Context, Result};
use mistralrs::{
    AudioInput, IsqType, Model, ModelDType, RequestBuilder, TextMessageRole, VisionModelBuilder,
};
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// ── Model presets ────────────────────────────────────────────────────────────

/// Available Gemma 3n model presets for audio transcription.
///
/// Both variants use a full 128-bin mel conformer audio encoder, which provides
/// high spectral resolution — ideal for noisy / artifact-heavy audio such as
/// demucs vocal stems.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum TranscriptionModel {
    /// Gemma 3n E2B — smallest (~1.5 GB with Q4K), fastest inference.
    #[value(name = "gemma-e2b")]
    GemmaE2b,

    /// Gemma 3n E4B — balanced quality & size for macOS desktop (F16).
    #[default]
    #[value(name = "gemma-e4b")]
    GemmaE4b,
}

impl TranscriptionModel {
    /// HuggingFace model identifier.
    pub fn model_id(self) -> &'static str {
        match self {
            Self::GemmaE2b => "google/gemma-3n-E2B-it",
            Self::GemmaE4b => "google/gemma-3n-E4B-it",
        }
    }

    /// Human-readable label used in log messages.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::GemmaE2b => "Gemma 3n E2B",
            Self::GemmaE4b => "Gemma 3n E4B",
        }
    }

    /// Approximate memory footprint with the chosen loading strategy.
    pub fn approx_memory(self) -> &'static str {
        match self {
            Self::GemmaE2b => "~1.5 GB (Q4K)",
            Self::GemmaE4b => "~8 GB (F16)",
        }
    }

    /// Build the [`Model`] with the optimal dtype / ISQ settings for this
    /// preset.
    ///
    /// Gemma 3n uses `Gemma3nForConditionalGeneration` (a multimodal
    /// architecture that includes a conformer audio encoder), so mistral.rs
    /// classifies it as a **vision** model.  We load it via
    /// [`VisionModelBuilder`].
    async fn build_model(self) -> Result<Model> {
        match self {
            Self::GemmaE2b => {
                VisionModelBuilder::new(self.model_id())
                    .with_isq(IsqType::Q4K)
                    .with_logging()
                    .build()
                    .await
            }
            Self::GemmaE4b => {
                VisionModelBuilder::new(self.model_id())
                    .with_dtype(ModelDType::F16)
                    .with_logging()
                    .build()
                    .await
            }
        }
    }
}

impl fmt::Display for TranscriptionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.display_name(), self.model_id())
    }
}

// ── Constants ────────────────────────────────────────────────────────────────

/// System prompt that steers the model toward faithful word-for-word
/// transcription rather than summarisation or description.
const TRANSCRIPTION_SYSTEM_PROMPT: &str = "\
You are a precise audio transcription assistant. \
Your task is to listen to the audio and produce an exact, word-for-word transcription of everything that is spoken or sung. \
Follow these rules strictly:\n\
1. Transcribe every word exactly as you hear it.\n\
2. Use standard punctuation (periods, commas, question marks).\n\
3. Start a new line for each distinct sentence or phrase.\n\
4. If a section is unintelligible, write [inaudible].\n\
5. Do NOT add any commentary, explanation, or description — output ONLY the transcription.";

/// A simpler user-level instruction appended when the caller does not supply
/// a custom prompt.
const DEFAULT_USER_PROMPT: &str = "Transcribe the vocals in this audio exactly, word for word.";

// ── AudioTranscriber ─────────────────────────────────────────────────────────

/// A self-contained audio transcriber built on Gemma 3n's conformer audio
/// encoder.
///
/// Designed for transcribing vocal stems produced by source-separation tools
/// like demucs.  The 128-bin mel spectrogram + conformer encoder gives higher
/// spectral resolution than typical 80-bin ASR front-ends, which helps with
/// the artefacts present in separated vocals.
pub struct AudioTranscriber {
    model: Model,
    system_prompt: String,
}

impl AudioTranscriber {
    /// Build a new `AudioTranscriber` using the **default** preset
    /// ([`TranscriptionModel::GemmaE4b`]).
    pub async fn new() -> Result<Self> {
        Self::from_preset(TranscriptionModel::default()).await
    }

    /// Build an `AudioTranscriber` from one of the built-in
    /// [`TranscriptionModel`] presets.
    pub async fn from_preset(preset: TranscriptionModel) -> Result<Self> {
        let model = preset.build_model().await?;
        Ok(Self {
            model,
            system_prompt: TRANSCRIPTION_SYSTEM_PROMPT.to_string(),
        })
    }

    /// Override the default system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }

    /// Transcribe audio from raw bytes (WAV, MP3, OGG, FLAC — anything
    /// symphonia can decode).
    ///
    /// `user_prompt` lets the caller customise the instruction sent alongside
    /// the audio.  Pass `None` to use the default transcription instruction.
    pub async fn transcribe_bytes(
        &self,
        audio_bytes: &[u8],
        user_prompt: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let audio = AudioInput::from_bytes(audio_bytes).context("Failed to decode audio bytes")?;
        self.transcribe_audio(audio, user_prompt).await
    }

    /// Transcribe a WAV file on disk.
    pub async fn transcribe_wav(
        &self,
        path: impl AsRef<Path>,
        user_prompt: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let path = path.as_ref();
        let audio = AudioInput::read_wav(
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("Path is not valid UTF-8: {}", path.display()))?,
        )
        .with_context(|| format!("Failed to read WAV file: {}", path.display()))?;
        self.transcribe_audio(audio, user_prompt).await
    }

    /// Transcribe audio from any file format supported by symphonia (WAV, MP3,
    /// OGG, FLAC, etc.) by reading the file into memory first.
    pub async fn transcribe_file(
        &self,
        path: impl AsRef<Path>,
        user_prompt: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("Failed to read audio file: {}", path.display()))?;
        self.transcribe_bytes(&bytes, user_prompt).await
    }

    /// Core transcription method that takes a decoded [`AudioInput`].
    async fn transcribe_audio(
        &self,
        audio: AudioInput,
        user_prompt: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let sample_rate = audio.sample_rate;
        let channels = audio.channels;
        let num_samples = audio.samples.len();
        let duration_secs = num_samples as f64 / (sample_rate as f64 * channels as f64);

        let user_text = user_prompt.unwrap_or(DEFAULT_USER_PROMPT);

        let request = RequestBuilder::new()
            .set_sampler_temperature(0.0)
            .add_message(TextMessageRole::System, &self.system_prompt)
            .add_audio_message(TextMessageRole::User, user_text, vec![audio], &self.model)?;

        let start = Instant::now();
        let response = self.model.send_chat_request(request).await?;
        let inference_elapsed = start.elapsed();

        let text = response.choices[0]
            .message
            .content
            .as_ref()
            .map(|c| c.trim().to_string())
            .unwrap_or_default();

        Ok(TranscriptionResult {
            text,
            audio_duration_secs: duration_secs,
            inference_duration: inference_elapsed,
            sample_rate,
            channels,
        })
    }

    /// Return a reference to the underlying `Model`.
    pub fn model(&self) -> &Model {
        &self.model
    }
}

// ── TranscriptionResult ──────────────────────────────────────────────────────

/// The output of a transcription, including the text and timing metadata.
pub struct TranscriptionResult {
    /// The transcribed text.
    pub text: String,
    /// Duration of the input audio in seconds.
    pub audio_duration_secs: f64,
    /// Wall-clock time the model spent generating the transcription.
    pub inference_duration: Duration,
    /// Sample rate of the input audio (before any resampling by the model).
    pub sample_rate: u32,
    /// Number of channels in the input audio.
    pub channels: u16,
}

impl TranscriptionResult {
    /// Real-time factor: `inference_time / audio_duration`.
    ///
    /// Values below 1.0 mean the model transcribes faster than real-time.
    pub fn real_time_factor(&self) -> f64 {
        if self.audio_duration_secs > 0.0 {
            self.inference_duration.as_secs_f64() / self.audio_duration_secs
        } else {
            f64::INFINITY
        }
    }
}

impl fmt::Display for TranscriptionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "── Transcription ──")?;
        writeln!(f, "{}", self.text)?;
        writeln!(f, "───────────────────")?;
        writeln!(
            f,
            "Audio duration : {:.1}s ({} Hz, {} ch)",
            self.audio_duration_secs, self.sample_rate, self.channels,
        )?;
        writeln!(
            f,
            "Inference time : {}",
            fmt_duration(self.inference_duration),
        )?;
        write!(f, "Real-time factor: {:.2}x", self.real_time_factor())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Format a `Duration` as `Xm Ys` (e.g. "2m 30.5s") or just `Ys` when under
/// a minute.
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

// ── Standalone CLI entry-point ───────────────────────────────────────────────

/// Run audio transcription as a standalone CLI example.
///
/// Loads Gemma 3n, reads the audio file at the given path, and prints the
/// transcription along with timing statistics.
pub async fn run(
    audio_path: PathBuf,
    model: Option<TranscriptionModel>,
    user_prompt: Option<String>,
) -> Result<()> {
    let preset = model.unwrap_or_default();

    // Validate input file exists
    if !audio_path.exists() {
        anyhow::bail!("Audio file not found: {}", audio_path.display());
    }

    println!("Loading transcription model: {preset}");
    println!("  Memory estimate: {}", preset.approx_memory());

    let load_start = Instant::now();
    let transcriber = AudioTranscriber::from_preset(preset).await?;
    let load_elapsed = load_start.elapsed();
    println!("Model loaded in {}\n", fmt_duration(load_elapsed));

    println!("Transcribing: {}", audio_path.display());

    let result = transcriber
        .transcribe_file(&audio_path, user_prompt.as_deref())
        .await?;

    println!("\n{result}");

    Ok(())
}
