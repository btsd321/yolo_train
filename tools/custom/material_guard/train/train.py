from ultralytics import YOLO

if __name__ == '__main__':
    dataset_config_file = "D:/Project/yolo_train/tools/custom/material_guard/config/material_guard_cls.yaml"
    model_config_file = "D:/Project/yolo_train/tools/custom/material_guard/config/yolo26-cls.yaml"

    # Load a YOLO26n-cls model
    model = YOLO(model_config_file, task="classify")

    # Train the model on the material guard dataset
    results = model.train(data=dataset_config_file,
                          epochs=100,
                          imgsz=224,
                          batch=32,
                          device=0,
                          project="output",
                          name="material_guard_cls",
                          exist_ok=True
                          )

    # # Run inference with the trained model
    # results = model("path/to/image.jpg")
