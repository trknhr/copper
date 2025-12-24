use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

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

    #[arg(long, default_value_t = true)]
    stream: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.stream {
        copper::run_stream(&args.model_dir, &args.prompt, args.max_new_tokens)?;
    } else {
        let out = copper::run(&args.model_dir, &args.prompt, args.max_new_tokens)?;
        println!("{out}");
    }
    Ok(())
}
