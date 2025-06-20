use std::{fs::File, path::PathBuf, time::Instant};

use clap::Parser;
use env_logger::Builder;
use log::{error, info, warn};
use memory_stats::memory_stats;
use pbrt_rs::{integrators::Integrate, scene_parsing::create_scene_integrator};

fn main() {
    let args = CliArgs::parse();

    let log_env = env_logger::Env::default().default_filter_or("info");
    Builder::from_env(log_env).format_target(false).init();
    info!("Initialized logger.");

    info!("Scene file to read: {}", args.scene_file.display());
    log_memory_usage();

    let render_result = render_cpu(&args);
    if let Err(err) = render_result {
        error!("Render failed:");
        error!("{}", err);
        return;
    }

    log_memory_usage();
}

fn render_cpu(args: &CliArgs) -> anyhow::Result<()> {
    let scene_file = File::open(args.scene_file.clone())?;
    let mut integrator = create_scene_integrator(scene_file, args.out_file.clone(), false)?;

    info!("Render begin");
    let start = Instant::now();

    integrator.render();

    let secs_elapsed = start.elapsed().as_secs();
    info!("Render complete");

    let hours = secs_elapsed / 3600;
    let mins = secs_elapsed % 3600 / 60;
    let secs = secs_elapsed % 60;
    info!("Render took {hours}h {mins}m {secs}s");

    log_memory_usage();

    info!("Dropping scene integrator");
    Ok(())
}

#[derive(Parser)]
struct CliArgs {
    /// .pbrt scene file to render.
    scene_file: PathBuf,
    /// The file to output the resulting render to.
    #[arg(short, long = "out")]
    out_file: Option<PathBuf>,
}

fn log_memory_usage() {
    if let Some(usage) = memory_stats() {
        info!(
            "Current physical memory usage: {:.1} MiB",
            usage.physical_mem as f32 / (1024.0 * 1024.0)
        );
        info!(
            "Current virtual memory usage: {:.1} MiB",
            usage.virtual_mem as f32 / (1024.0 * 1024.0)
        );
    } else {
        warn!("Couldn't get the current memory usage");
    }
}
