"""Export model weights from a checkpoint to a safetensors file."""

import argparse
from pathlib import Path

from safetensors.torch import save_file

from kanade_tokenizer.pipeline import KanadePipeline


def main():
    parser = argparse.ArgumentParser(description="Export model weights to safetensors format")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file (.ckpt)")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration file (.yaml)")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for the safetensors file (default: same name as checkpoint with .safetensors extension)",
    )
    args = parser.parse_args()

    # Load the pipeline from checkpoint
    print(f"Loading pipeline from {args.checkpoint_path}...")
    pipeline = KanadePipeline.from_pretrained(args.config, args.checkpoint_path)

    # Get the model state dict
    state_dict = pipeline.model.weights_to_save()

    # Convert all tensors to CPU if needed
    state_dict = {k: v.cpu() for k, v in state_dict.items()}

    # Determine output path
    if args.output is None:
        checkpoint_path = Path(args.checkpoint_path)
        output_path = checkpoint_path.with_suffix(".safetensors")
    else:
        output_path = Path(args.output)

    # Save to safetensors
    print(f"Saving weights to {output_path}...")
    save_file(state_dict, str(output_path))
    print(f"Successfully exported {len(state_dict)} tensors to {output_path}")
    for key in state_dict.keys():
        print(f" - {key}")


if __name__ == "__main__":
    main()
