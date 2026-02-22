import os
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    exit(1)

def main():
    parser = argparse.ArgumentParser("HuggingFace to YOLO Converter")
    parser.add_argument("--repo", type=str, required=True, help="Hugging Face repo ID")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset {args.repo} from Hugging Face...")
    ds = load_dataset(args.repo)
    
    all_class_names = []
    
    # Pre-scan to gather all unique classes if they are strings
    print("Pre-scanning to find unique classes...")
    for split, dataset in ds.items():
        if 'objects' in dataset.features:
            for item in dataset:
                cats = item['objects'].get('category', [])
                for c in cats:
                    if isinstance(c, str) and c not in all_class_names:
                        all_class_names.append(c)
    
    all_class_names = sorted(all_class_names)
    cat_to_id = {c: i for i, c in enumerate(all_class_names)}
    
    # We will iterate through splits
    for split, dataset in ds.items():
        print(f"--- Processing split: {split} ---")
        
        split_dir = out_dir / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        if 'objects' not in dataset.features:
            print(f"Warning: 'objects' column not found in split {split}. Skipping...")
            continue
            
        print(f"Found {len(dataset)} examples in split {split}")
        
        for idx, item in enumerate(tqdm(dataset, desc=f"Converting {split}")):
            img = item['image']  # PIL Image
            if not getattr(img, 'mode', None):
                img = img.convert("RGB")
                
            img_w, img_h = img.size
            
            # Save the image
            filename = item.get('file_name', item.get('image_id', f"{idx:06d}"))
            if isinstance(filename, int) or not filename:
                filename = f"{idx:06d}"
            
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename = f"{filename}.jpg"
                
            base_name = os.path.splitext(filename)[0]
            img_path = images_dir / filename
            label_path = labels_dir / f"{base_name}.txt"
            
            img.save(img_path)
            
            objects = item['objects']
            bboxes = objects.get('bbox', [])
            categories = objects.get('category', [])
            
            with open(label_path, 'w') as f:
                for bbox, cat in zip(bboxes, categories):
                    cat_id = cat_to_id.get(cat, 0) if isinstance(cat, str) else cat
                    
                    x_min, y_min, w, h = bbox
                    
                    x_c = (x_min + w / 2) / img_w
                    y_c = (y_min + h / 2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    
                    x_c = max(0.0, min(1.0, x_c))
                    y_c = max(0.0, min(1.0, y_c))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))
                    
                    f.write(f"{cat_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    
    if not all_class_names:
        print("Warning: Could not automatically detect class names. Generating dummy names.")
        all_class_names = [f"class_{i}" for i in range(100)] # Hack if we don't know nc
        
    print(f"Identified {len(all_class_names)} classes: {all_class_names}")

    # Generate data.yaml for YOLO
    yaml_dict = {}
    if (out_dir / "train").exists(): yaml_dict["train"] = "train/images"
    if (out_dir / "valid").exists(): yaml_dict["val"] = "valid/images"
    elif (out_dir / "validation").exists(): yaml_dict["val"] = "validation/images"
    if (out_dir / "test").exists(): yaml_dict["test"] = "test/images"
    
    yaml_dict["nc"] = len(all_class_names)
    yaml_dict["names"] = all_class_names
    
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
        
    print(f"Successfully created YOLO configuration at {yaml_path}")
    
    # Generate client variations for FedRep
    for i in range(10): # up to 10 clients for convenience
        c_path = out_dir / f"client_{i}.yaml"
        with open(c_path, "w") as f:
            yaml.dump(yaml_dict, f, sort_keys=False)

if __name__ == "__main__":
    main()
