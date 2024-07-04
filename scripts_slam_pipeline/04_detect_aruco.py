"""
This script processes video files to detect ArUco markers and save the results in a pickle file.
It takes as input a directory containing video files, a camera intrinsics JSON file, and an ArUco configuration YAML file.
The script uses concurrent processing to handle multiple video files in parallel and provides a progress bar to monitor the processing status.

Usage:
    python 04_detect_aruco.py \
        -i data/dataset/demos \
        -ci example/calibration/gopro_intrinsics_2_7k.json \
        -ac example/calibration/aruco_config.yaml
"""

import os
import pathlib
import click
import multiprocessing
import subprocess
import concurrent.futures
from rich import print
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

@click.command()
@click.option('-i', '--input_dir', required=True, help='Directory for demos folder')
@click.option('-ci', '--camera_intrinsics', required=True, help='Camera intrinsics json file (2.7k)')
@click.option('-ac', '--aruco_yaml', required=True, help='Aruco config yaml file')
@click.option('-n', '--num_workers', type=int, default=None)

def main(input_dir, camera_intrinsics, aruco_yaml, num_workers):
    input_dir = pathlib.Path(os.path.expanduser(input_dir))
    print(f'[blue]input_dir:[/blue] {input_dir}')
    input_video_dirs = [x.parent for x in input_dir.glob('*/raw_video.mp4')]
    print(f'[green]Found {len(input_video_dirs)} video dirs[/green]')
    
    assert os.path.isfile(camera_intrinsics), f"[red]Camera intrinsics file {camera_intrinsics} does not exist.[/red]"
    assert os.path.isfile(aruco_yaml), f"[red]Aruco config file {aruco_yaml} does not exist.[/red]"

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    script_path = pathlib.Path(__file__).parent.parent.joinpath('scripts', 'detect_aruco.py')

    def process_video(video_dir):
        video_dir = video_dir.absolute()
        video_path = video_dir.joinpath('raw_video.mp4')
        pkl_path = video_dir.joinpath('tag_detection.pkl')

        if pkl_path.is_file():
            print(f"[yellow]tag_detection.pkl already exists, skipping {video_dir.name}[/yellow]")
            return True

        cmd = [
            'python', script_path,
            '--input', str(video_path),
            '--output', str(pkl_path),
            '--intrinsics_json', camera_intrinsics,
            '--aruco_yaml', aruco_yaml,
            '--num_workers', '1'
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"[red]Error processing {video_dir.name}: {result.stderr.decode('utf-8')}[/red]")
            return False

        return pkl_path.is_file()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} files processed"),
    ) as progress:
        task = progress.add_task("[cyan]Processing videos...", total=len(input_video_dirs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for video_dir in input_video_dirs:
                if len(futures) >= num_workers:
                    completed, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                    progress.update(task, advance=len(completed))

                futures.add(executor.submit(process_video, video_dir))

            completed, futures = concurrent.futures.wait(futures)
            progress.update(task, advance=len(completed))

    print("[bold green]Done![/bold green]")

if __name__ == "__main__":
    main()
