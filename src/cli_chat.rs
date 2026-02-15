use anyhow::Result;
use mistralrs::{
    IsqType, Model, ModelDType, RequestBuilder, TextMessageRole, TextModelBuilder,
    VisionModelBuilder,
};
use std::fmt;
use std::io::{self, Write};
use std::time::{Duration, Instant};

/// Available chat model presets.
///
/// These match the presets used by `promp_enhancer.rs` so both modules use
/// identical model IDs and loading strategies.
use crate::promp_enhancer::EnhancerModel;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChatModel {
    /// Gemma 3n E2B — smallest, best for iPhone / on-device (~1.5 GB with Q4K).
    GemmaE2b,

    /// Gemma 3n E4B — balanced quality & size for macOS desktop (F16).
    #[default]
    GemmaE4b,

    /// Phi-3.5-mini — strongest quality, larger memory footprint (~2.8 GB with Q4K).
    Phi35Mini,
}

impl From<EnhancerModel> for ChatModel {
    fn from(value: EnhancerModel) -> Self {
        match value {
            EnhancerModel::GemmaE2b => Self::GemmaE2b,
            EnhancerModel::GemmaE4b => Self::GemmaE4b,
            EnhancerModel::Phi35Mini => Self::Phi35Mini,
        }
    }
}

impl ChatModel {
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

    /// Build the [`Model`] with preset-specific dtype / ISQ settings.
    ///
    /// Gemma 3n variants use a multimodal architecture and are loaded through
    /// [`VisionModelBuilder`] even for text chat.
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
            Self::Phi35Mini => {
                TextModelBuilder::new(self.model_id())
                    .with_isq(IsqType::Q4K)
                    .with_logging()
                    .build()
                    .await
            }
        }
    }
}

impl fmt::Display for ChatModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.display_name(), self.model_id())
    }
}

/// Format a `Duration` as `Xm Ys` (e.g. "2m 30.5s") or `Ys` under a minute.
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

/// A single conversation message.
#[derive(Clone, Debug)]
struct ChatTurn {
    role: TextMessageRole,
    content: String,
}

/// Interactive chat session state.
pub struct CliChat {
    model: Model,
    system_prompt: String,
    history: Vec<ChatTurn>,
    temperature: f64,
    top_p: f64,
    max_len: usize,
}

impl CliChat {
    /// Build a chat session from model preset.
    pub async fn from_preset(model: ChatModel, system_prompt: Option<String>) -> Result<Self> {
        let loaded = model.build_model().await?;
        Ok(Self {
            model: loaded,
            system_prompt: system_prompt.unwrap_or_else(|| {
                "You are a helpful, concise assistant. Answer clearly and accurately.".to_string()
            }),
            history: Vec::new(),
            temperature: 0.7,
            top_p: 0.95,
            max_len: 512,
        })
    }

    /// Send one user message and return assistant response.
    pub async fn send(&mut self, user_message: &str) -> Result<String> {
        let mut request = RequestBuilder::new()
            .set_sampler_temperature(self.temperature)
            .set_sampler_topp(self.top_p)
            .set_sampler_max_len(self.max_len)
            .add_message(TextMessageRole::System, &self.system_prompt);

        // Replay prior conversation for context.
        for turn in &self.history {
            request = request.add_message(turn.role.clone(), &turn.content);
        }

        // Add current user turn.
        request = request.add_message(TextMessageRole::User, user_message);

        let response = self.model.send_chat_request(request).await?;
        let assistant = response.choices[0]
            .message
            .content
            .as_ref()
            .map(|c| c.trim().to_string())
            .unwrap_or_else(|| String::from("(empty response)"));

        // Persist turn history.
        self.history.push(ChatTurn {
            role: TextMessageRole::User,
            content: user_message.to_string(),
        });
        self.history.push(ChatTurn {
            role: TextMessageRole::Assistant,
            content: assistant.clone(),
        });

        Ok(assistant)
    }

    /// Clear conversation history but keep loaded model and system prompt.
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

/// Run an interactive CLI chat session.
///
/// Commands:
/// - `/help`  : show command help
/// - `/clear` : clear chat history
/// - `/exit`  : quit
/// - `/quit`  : quit
pub async fn run(model: Option<EnhancerModel>) -> Result<()> {
    let preset = model.unwrap_or_default();
    let preset: ChatModel = preset.into();

    println!("Loading chat model: {preset}");
    println!("  Memory estimate: {}", preset.approx_memory());

    let load_start = Instant::now();
    let mut chat = CliChat::from_preset(preset, None).await?;
    println!("Model loaded in {}", fmt_duration(load_start.elapsed()));

    println!();
    println!("Interactive chat is ready.");
    println!("Type your message and press Enter.");
    println!("Commands: /help, /clear, /exit, /quit");
    println!();

    let stdin = io::stdin();

    loop {
        print!("you> ");
        io::stdout().flush()?;

        let mut input = String::new();
        let n = stdin.read_line(&mut input)?;
        if n == 0 {
            // EOF (Ctrl-D / piped input end).
            println!("\nExiting.");
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        match input {
            "/exit" | "/quit" => {
                println!("Exiting.");
                break;
            }
            "/help" => {
                println!("Commands:");
                println!("  /help   Show this help");
                println!("  /clear  Clear chat history");
                println!("  /exit   Quit");
                println!("  /quit   Quit");
                continue;
            }
            "/clear" => {
                chat.clear();
                println!("History cleared.");
                continue;
            }
            _ => {}
        }

        let turn_start = Instant::now();
        let reply = chat.send(input).await?;
        let elapsed = turn_start.elapsed();

        println!("assistant> {}", reply);
        println!("(latency: {})", fmt_duration(elapsed));
        println!();
    }

    Ok(())
}
