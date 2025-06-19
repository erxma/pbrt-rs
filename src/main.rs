use std::{fs::File, path::PathBuf, time::Instant};

use clap::Parser;
use log::{error, info};
use pbrt_rs::{integrators::Integrate, scene_parsing::create_scene_integrator};

fn main() {
    let args = CliArgs::parse();

    let log_env = env_logger::Env::default().default_filter_or("info");
    env_logger::init_from_env(log_env);
    info!("Initialized logger.");

    let start = Instant::now();

    let render_result = render_cpu(&args);
    if let Err(err) = render_result {
        error!("Render failed:");
        error!("{}", err);
        return;
    }

    let secs_elapsed = start.elapsed().as_secs();
    let hours = secs_elapsed / 3600;
    let mins = secs_elapsed % 3600 / 60;
    let secs = secs_elapsed % 60;

    info!("Render took {hours}h {mins}m {secs}s.");
}

fn render_cpu(args: &CliArgs) -> anyhow::Result<()> {
    let scene_file = File::open(args.scene_file.clone())?;
    let mut integrator = create_scene_integrator(scene_file, args.out_file.clone(), false)?;

    integrator.render();

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
