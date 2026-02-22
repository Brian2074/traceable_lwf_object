import os
import shutil
import argparse
import yaml
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def get_image_classes(label_path):
    """Read a YOLO label file and return the unique classes present."""
    if not os.path.exists(label_path):
        return []
    classes = set()
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.add(int(parts[0]))
    return list(classes)

def create_non_iid_split(images_dir, labels_dir, num_clients, num_classes, alpha=0.5):
    """
    Distribute images among clients using a Dirichlet distribution over classes.
    Because an image can have multiple classes, we primary assign it based on 
    one of its classes, or we just sample the primary class.
    """
    # 1. Group images by their "primary" class
    # For simplicity, we just use the first class found in the label file
    class_to_images = defaultdict(list)
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print("Scanning images to determine class distribution...")
    for img_name in tqdm(all_images):
        base_name = os.path.splitext(img_name)[0]
        label_name = f"{base_name}.txt"
        label_path = os.path.join(labels_dir, label_name)
        
        classes = get_image_classes(label_path)
        if classes:
            primary_class = classes[0]  # Just use the first object's class as primary
        else:
            primary_class = 0 # Fallback if empty
            
        class_to_images[primary_class].append(img_name)

    # 2. Distribute indices Dirichlet
    client_image_lists = [[] for _ in range(num_clients)]
    
    for c_id in range(num_classes):
        images_in_class = class_to_images[c_id]
        np.random.shuffle(images_in_class)
        
        # Sample Dirichlet distribution for this class
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions into counts
        counts = (proportions * len(images_in_class)).astype(int)
        
        # Fix rounding dropping elements
        remainder = len(images_in_class) - counts.sum()
        for i in range(remainder):
            counts[i % num_clients] += 1
            
        start = 0
        for client_idx in range(num_clients):
            end = start + counts[client_idx]
            client_image_lists[client_idx].extend(images_in_class[start:end])
            start = end

    return client_image_lists

def main():
    parser = argparse.ArgumentParser("YOLO Dataset Non-IID Dirichlet Splitter")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to base dataset dir (e.g. datasets/spscd_coco_yolo)")
    parser.add_argument("--clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha (smaller = more non-IID)")
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    data_yaml_path = base_dir / "data.yaml"
    
    if not data_yaml_path.exists():
        print(f"Error: {data_yaml_path} not found.")
        return

    with open(data_yaml_path, 'r') as f:
        data_info = yaml.safe_load(f)
        
    num_classes = data_info.get('nc', 0)
    
    # We only split the 'train' folder. Validation remains global per the user's setup usually, 
    # but for true FL each client could have local validation. Here we create client-specific train dirs.
    source_train_images = base_dir / "train" / "images"
    source_train_labels = base_dir / "train" / "labels"
    
    if not source_train_images.exists():
        print(f"Error: {source_train_images} not found.")
        return

    print(f"Starting Dirichlet split (alpha={args.alpha}) for {args.clients} clients...")
    client_splits = create_non_iid_split(
        images_dir=source_train_images,
        labels_dir=source_train_labels,
        num_clients=args.clients,
        num_classes=num_classes,
        alpha=args.alpha
    )
    
    # Check absolute path
    abs_base = base_dir.absolute()
    
    for client_id, img_list in enumerate(client_splits):
        client_dir = base_dir / f"client_{client_id}_data"
        c_images = client_dir / "images"
        c_labels = client_dir / "labels"
        c_images.mkdir(parents=True, exist_ok=True)
        c_labels.mkdir(parents=True, exist_ok=True)
        
        print(f"Copying {len(img_list)} images for Client {client_id}...")
        for img_name in tqdm(img_list, leave=False):
            base_name = os.path.splitext(img_name)[0]
            label_name = f"{base_name}.txt"
            
            # Copy image
            src_img = source_train_images / img_name
            dst_img = c_images / img_name
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                
            # Copy label
            src_lbl = source_train_labels / label_name
            dst_lbl = c_labels / label_name
            if src_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
                
        # Generate client_X.yaml mapping to these new copied folders
        client_yaml_path = base_dir / f"client_{client_id}.yaml"
        client_info = data_info.copy()
        
        # Point to the new local slice (use relative paths from the dataset root)
        client_info['train'] = f"client_{client_id}_data/images"
        
        # Ensure validation and test use relative paths from the dataset root
        if 'val' in client_info and '/' in client_info['val'] and client_info['val'].startswith('/'):
            client_info['val'] = os.path.relpath(client_info['val'], abs_base)
        if 'test' in client_info and '/' in client_info['test'] and client_info['test'].startswith('/'):
            client_info['test'] = os.path.relpath(client_info['test'], abs_base)
            
        # VERY IMPORTANT for Docker: Tell YOLO that these relative paths start at /app/datasets/...
        # This overrides whatever directory YOLO is currently executing from.
        docker_base_name = base_dir.name
        client_info['path'] = f"/app/datasets/{docker_base_name}"
            
        with open(client_yaml_path, 'w') as f:
            yaml.dump(client_info, f, sort_keys=False)
            
        print(f"Created config for Client {client_id} -> {client_yaml_path}")

    print("\nNon-IID Splitting Complete!")

if __name__ == "__main__":
    main()
