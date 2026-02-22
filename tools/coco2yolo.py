import json
import os
import argparse
from pathlib import Path
import yaml

def convert_coco_json(json_path, images_dir, out_labels_dir):
    """
    Parses a COCO instances JSON file and converts it to YOLO format text files.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        print(f"No JSON found at {json_path}. Skipping.")
        return [], {}
    
    out_labels_dir = Path(out_labels_dir)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
        
    if 'categories' not in coco_data or 'images' not in coco_data or 'annotations' not in coco_data:
        print(f"Invalid COCO format in {json_path}")
        return [], {}

    # Map category_id to a continuous 0-indexed YOLO ID
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    sorted_cat_ids = sorted(list(category_id_to_name.keys()))
    cat_id_to_yolo_id = {old_id: new_id for new_id, old_id in enumerate(sorted_cat_ids)}
    yolo_id_to_name = {cat_id_to_yolo_id[old]: name for old, name in category_id_to_name.items()}

    # Map image_id to file_name and dimensions
    image_id_to_info = {
        img['id']: {
            'file_name': img['file_name'],
            'width': img['width'],
            'height': img['height']
        } for img in coco_data['images']
    }

    # Group annotations by image_id
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        
    print(f"Loaded {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations from {json_path.name}")
        
    # Create YOLO label files
    processed = 0
    for img_id, info in image_id_to_info.items():
        base_name = os.path.splitext(info['file_name'])[0]
        label_file = out_labels_dir / f"{base_name}.txt"
        
        img_w = info['width']
        img_h = info['height']
        
        anns = img_to_anns.get(img_id, [])
        
        # We write empty files for images with no annotations (negative samples)
        with open(label_file, 'w') as f:
            for ann in anns:
                cat_id = ann['category_id']
                if cat_id not in cat_id_to_yolo_id:
                    continue
                yolo_id = cat_id_to_yolo_id[cat_id]
                
                # COCO bbox: [top_left_x, top_left_y, width, height]
                x_tl, y_tl, w, h = ann['bbox']
                
                # YOLO bbox: [center_x, center_y, width, height] normalized
                x_c = (x_tl + w / 2) / img_w
                y_c = (y_tl + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                x_c = max(0.0, min(1.0, x_c))
                y_c = max(0.0, min(1.0, y_c))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))
                
                f.write(f"{yolo_id} {x_c:.6f} {y_c:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        processed += 1
        
    print(f"Generated {processed} YOLO label files in {out_labels_dir}")
    
    class_names = [yolo_id_to_name[i] for i in range(len(yolo_id_to_name))]
    return class_names, dict(yolo_id_to_name)

def main():
    parser = argparse.ArgumentParser("COCO to YOLO Converter")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Base directory of the downloaded COCO dataset")
    args = parser.parse_args()
    
    base_dir = Path(args.dataset_dir)
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist.")
        return
        
    # Hugging Face COCO wrappers usually have exactly one annotation file per split
    # For instance 'train/_annotations.coco.json'
    splits = ["train", "valid", "validation", "test"]
    all_class_names = []
    
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
            
        json_path = split_dir / "_annotations.coco.json"
        
        if not json_path.exists():
            # Sometimes it's a generic result.json or something else if it's straight from roboflow
            potential_jsons = list(split_dir.glob("*.json"))
            if potential_jsons:
                json_path = potential_jsons[0]
            else:
                continue
                
        images_dir = split_dir / "images"
        out_labels_dir = split_dir / "labels"
        
        print(f"--- Processing split: {split} ({json_path.name}) ---")
        class_names, _ = convert_coco_json(json_path, images_dir, out_labels_dir)
        if class_names and not all_class_names:
            all_class_names = class_names
            
    if not all_class_names:
        print("Could not find any categories to build a data.yaml. Did conversion fail?")
        return
        
    # Write yaml config
    yaml_dict = {}
    if (base_dir / "train").exists(): yaml_dict["train"] = "train/images"
    if (base_dir / "valid").exists(): yaml_dict["val"] = "valid/images"
    elif (base_dir / "validation").exists(): yaml_dict["val"] = "validation/images"
    if (base_dir / "test").exists(): yaml_dict["test"] = "test/images"
    
    yaml_dict["nc"] = len(all_class_names)
    yaml_dict["names"] = all_class_names
    
    yaml_path = base_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)
        
    print(f"Successfully created base YOLO configuring at {yaml_path}")
    
    # Generate client variations for FedRep
    for i in range(10): # up to 10 clients for convenience
        c_path = base_dir / f"client_{i}.yaml"
        with open(c_path, "w") as f:
            yaml.dump(yaml_dict, f, sort_keys=False)

if __name__ == "__main__":
    main()
