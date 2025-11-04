"""
Script to scan audio files in a directory and output metadata to CSV.

Usage:
    python script/dump_dataset.py <directory> [--output <csv_path>] [--pattern <glob_pattern>]

Example:
    python script/dump_dataset.py /path/to/audio --output dataset.csv --pattern "**/*.wav"
"""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import Path

import soundfile as sf
from tqdm import tqdm


def process_audio_file(audio_path: str, audio_dir: Path) -> dict | None:
    """Process a single audio file and return its metadata."""
    try:
        f = sf.SoundFile(audio_path)

        try:
            relative_path = str(Path(audio_path).relative_to(audio_dir))
        except ValueError:
            # If not relative, use absolute path
            relative_path = audio_path

        return {"path": relative_path, "length": f.frames, "sample_rate": f.samplerate}

    except Exception as e:
        tqdm.write(f"Warning: Failed to process {audio_path}: {e}")
        return None


def scan_audio_files(audio_dir: Path, pattern: str, max_workers: int = 8) -> list[dict]:
    # Find all audio files using glob pattern
    audio_files = list(glob(str(audio_dir / pattern), recursive=True))
    audio_files.sort()
    print(f"Found {len(audio_files)} audio files. Processing with {max_workers} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [
            result
            for result in tqdm(
                executor.map(lambda audio_path: process_audio_file(audio_path, audio_dir), audio_files),
                total=len(audio_files),
                desc="Processing",
            )
            if result is not None
        ]

    return results


def add_audio_ids(results: list[dict]) -> list[dict]:
    """Add unique audio_id to each result based on file stem, handling duplicates."""
    stem_counts = {}
    for result in results:
        stem = Path(result["path"]).stem
        if stem not in stem_counts:
            stem_counts[stem] = 0
            result["audio_id"] = stem
        else:
            stem_counts[stem] += 1
            result["audio_id"] = f"{stem}_{stem_counts[stem]}"
    return results


def write_csv(results: list[dict], output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_id", "path", "length", "sample_rate"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Scan audio files and generate metadata CSV")
    parser.add_argument("audio_dir", type=Path, help="Directory to scan for audio files")
    parser.add_argument("--output", "-o", type=Path, help="Output CSV file path")
    parser.add_argument(
        "--pattern", "-p", type=str, default="**/*.wav", help="Glob pattern for audio files (default: **/*.wav)"
    )
    parser.add_argument("--workers", "-w", type=int, default=8, help="Number of parallel workers (default: 8)")
    args = parser.parse_args()

    # Validate directory
    if not args.audio_dir.exists():
        print(f"Error: Directory {args.audio_dir} does not exist.")
        return 1
    if not args.audio_dir.is_dir():
        print(f"Error: {args.audio_dir} is not a directory.")
        return 1

    print(f"Scanning directory {args.audio_dir} with pattern '{args.pattern}'...")

    # Scan audio files
    results = scan_audio_files(args.audio_dir, args.pattern, max_workers=args.workers)

    # Add audio IDs
    results = add_audio_ids(results)

    # Write results
    if results:
        write_csv(results, args.output)

        print("Summary:")
        print(f"  Total files: {len(results)}")
        sample_rates = set(r["sample_rate"] for r in results)
        print(f"  Sample rates: {sorted(sample_rates)}")
        total_length = sum(r["length"] for r in results)
        print(f"  Total samples: {total_length:,}")
        total_hours = sum(r["length"] / r["sample_rate"] for r in results) / 3600
        print(f"  Total duration: {total_hours:.2f} hours")
    else:
        print("No audio files found.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
