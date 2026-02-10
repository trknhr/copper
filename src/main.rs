use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use tracing::Level;

#[derive(Parser, Debug)]
#[command(
    name = "copper",
    version,
    about = "Minimal GPT-2 runner (Candle backend)"
)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,

    #[arg(long)]
    prompt: String,

    #[arg(long, default_value_t = 16)]
    max_new_tokens: usize,

    #[arg(long, value_enum, default_value_t = OutputMode::Stream)]
    output: OutputMode,

    #[arg(long, value_enum, default_value_t = LogLevel::Info)]
    log_level: LogLevel,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum OutputMode {
    Stream,
    Buffered,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl From<LogLevel> for Level {
    fn from(value: LogLevel) -> Self {
        match value {
            LogLevel::Error => Level::ERROR,
            LogLevel::Warn => Level::WARN,
            LogLevel::Info => Level::INFO,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Trace => Level::TRACE,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    tracing_subscriber::fmt()
        .with_max_level(Level::from(args.log_level))
        .with_target(false)
        .init();

    match args.output {
        OutputMode::Stream => {
            copper::run_stream(&args.model_dir, &args.prompt, args.max_new_tokens)?;
        }
        OutputMode::Buffered => {
            let out = copper::run(&args.model_dir, &args.prompt, args.max_new_tokens)?;
            println!("{out}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_output_modes() {
        let stream = Args::try_parse_from([
            "copper",
            "--model-dir",
            "/tmp/gpt2",
            "--prompt",
            "hello",
            "--output",
            "stream",
        ])
        .expect("stream mode should parse");
        assert_eq!(stream.output, OutputMode::Stream);

        let buffered = Args::try_parse_from([
            "copper",
            "--model-dir",
            "/tmp/gpt2",
            "--prompt",
            "hello",
            "--output",
            "buffered",
        ])
        .expect("buffered mode should parse");
        assert_eq!(buffered.output, OutputMode::Buffered);
    }

    #[test]
    fn cli_rejects_invalid_output_mode() {
        let bad = Args::try_parse_from([
            "copper",
            "--model-dir",
            "/tmp/gpt2",
            "--prompt",
            "hello",
            "--output",
            "invalid",
        ]);
        assert!(bad.is_err());
    }
}
