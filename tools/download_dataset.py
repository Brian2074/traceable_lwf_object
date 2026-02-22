import argparse
import os
import shutil
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download YOLO dataset from Hugging Face")
    parser.add_argument("--repo", type=str, default="ARG-NCTU/spscd_plus_buoy_yolo", help="HF repo ID")
    parser.add_argument("--output_dir", type=str, default="datasets/spscd", help="Output directory")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of FL clients to split into")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo} from Hugging Face...")
    # This downloads the full tree. If it's private, you need `huggingface-cli login` first.
    # We enable `resume_download=True` and set `max_workers` so large datasets won't break 
    # and will download much faster in parallel.
    dataset_path = snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=args.output_dir,
        resume_download=True,
        max_workers=8
        # We removed allow_patterns so we don't accidentally skip 'train' or subfolders
    )
    print(f"Dataset downloaded to {dataset_path}")

    # Generate client-specific yaml files
    # Assuming there's a base data.yaml or dataset.yaml
    yaml_files = list(out_path.glob("*.yaml"))
    
    if not yaml_files:
        print("Warning: No .yaml config found in the downloaded dataset.")
        print("You will need a data.yaml for YOLOv8 to train.")
    else:
        base_yaml = yaml_files[0]
        print(f"Found base config: {base_yaml.name}. Generating client configs...")
        
        with open(base_yaml, "r") as f:
            base_content = f.read()
            
        # Optional: Split the data into client subdirectories here, 
        # but the simplest way is to give every client the same config for now 
        # (or define path exactly)
        
        for i in range(args.num_clients):
            client_yaml = out_path / f"client_{i}.yaml"
            if not client_yaml.exists():
                shutil.copy(base_yaml, client_yaml)
                print(f"Created {client_yaml}")

if __name__ == "__main__":
    main()
