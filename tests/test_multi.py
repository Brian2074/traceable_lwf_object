from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    
    # Run 1
    model.train(
        data="datasets/spscd_coco_yolo/client_0.yaml",
        epochs=1,
        project="test_multi_train",
        name="run1",
        freeze=[22],
        exist_ok=True,
        workers=0
    )
    
    print("\n\nRUN 1 DONE. STARTING RUN 2\n\n")
    
    # Try restoring model key if it's missing
    # model.overrides['model'] = "yolov8n.pt" # This might reset weights!
    # A safer way might be to load the checkpoint from run 1
    
    # Run 2
    try:
        model.train(
            data="datasets/spscd_coco_yolo/client_0.yaml",
            epochs=1,
            project="test_multi_train",
            name="run2",
            freeze=22,
            exist_ok=True,
            workers=0
        )
        print("Run 2 succeeded without fixing overrides!")
    except Exception as e:
        print(f"Run 2 failed: {e}")
        
        print("\n\nAttempting fix...\n\n")
        model.overrides['model'] = model.ckpt_path or "yolov8n.pt"
        
        try:
            model.train(
                data="datasets/spscd_coco_yolo/client_0.yaml",
                epochs=1,
                project="test_multi_train",
                name="run2_fixed",
                freeze=22,
                exist_ok=True,
                workers=0
            )
            print("Run 2 succeeded with override fix!")
        except Exception as e2:
            print(f"Run 2 still failed: {e2}")

if __name__ == "__main__":
    main()
