"""
This script runs SLAM tag and gripper range calibrations for specified session directories.
It processes video directories, generates calibration files, and ensures that the necessary input files are present.

Usage:
    python scripts_slam_pipeline/05_run_calibrations.py data/dataset
"""
import pathlib
import click
import subprocess
from rich import print

@click.command()
@click.argument('session_dir', nargs=-1)
def main(session_dir):
    script_dir = pathlib.Path(__file__).parent.parent.joinpath('scripts')
    
    for session in session_dir:
        session = pathlib.Path(session)
        print(f'[blue]Session path:[/blue] {session}')
        demos_dir = session.joinpath('demos')
        print(f'[blue]Demos path:[/blue] {demos_dir}')
        mapping_dir = demos_dir.joinpath('mapping')
        print(f'[blue]Mapping path:[/blue] {mapping_dir}')
        slam_tag_path = mapping_dir.joinpath('tx_slam_tag.json')
        print(f'[blue]SLAM tag path:[/blue] {slam_tag_path}')
            
        # Run SLAM tag calibration
        script_path = script_dir.joinpath('calibrate_slam_tag.py')
        print(f'[blue]Script path (calibrate_slam_tag.py):[/blue] {script_path}')
        assert script_path.is_file(), f"[red]Script file {script_path} does not exist.[/red]"
        
        tag_path = mapping_dir.joinpath('tag_detection.pkl')
        print(f'[blue]Tag path (tag_detection.pkl):[/blue] {tag_path}')
        assert tag_path.is_file(), f"[red]Tag file {tag_path} does not exist.[/red]"
        
        csv_path = mapping_dir.joinpath('camera_trajectory.csv')
        print(f'[blue]CSV path (camera_trajectory.csv):[/blue] {csv_path}')
        if not csv_path.is_file():
            csv_path = mapping_dir.joinpath('mapping_camera_trajectory.csv')
            print("[yellow]camera_trajectory.csv not found! Using mapping_camera_trajectory.csv[/yellow]")
        assert csv_path.is_file(), f"[red]CSV file {csv_path} does not exist.[/red]"
        
        cmd = [
            'python', str(script_path),
            '--tag_detection', str(tag_path),
            '--csv_trajectory', str(csv_path),
            '--output', str(slam_tag_path),
            '--keyframe_only'
        ]
        subprocess.run(cmd)
        
        # Run gripper range calibration
        script_path = script_dir.joinpath('calibrate_gripper_range.py')
        print(f'[blue]Script path (calibrate_gripper_range.py):[/blue] {script_path}')
        assert script_path.is_file(), f"[red]Script file {script_path} does not exist.[/red]"
        
        for gripper_dir in demos_dir.glob("gripper_calibration*"):
            gripper_range_path = gripper_dir.joinpath('gripper_range.json')
            tag_path = gripper_dir.joinpath('tag_detection.pkl')
            assert tag_path.is_file(), f"[red]Tag file {tag_path} does not exist in {gripper_dir}.[/red]"
            cmd = [
                'python', str(script_path),
                '--input', str(tag_path),
                '--output', str(gripper_range_path)
            ]
            subprocess.run(cmd)

if __name__ == "__main__":
    main()
