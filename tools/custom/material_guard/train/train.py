from ultralytics import YOLO

if __name__ == '__main__':
    # 分类任务: data 直接传数据集目录路径（包含 train/val/test 子目录），不需要 YAML 配置文件
    dataset_dir = "D:/Data/material_guard/material_guard_dataset"
    model_config_file = "D:/Project/yolo_train/tools/custom/material_guard/config/yolo26m-cls.yaml"

    # Load a YOLO26m-cls model
    model = YOLO(model_config_file, task="classify")

    # Train the model on the material guard dataset
    results = model.train(data=dataset_dir,
                          epochs=300,
                          imgsz=640,
                          batch=8,
                          device=0,
                          project="output",
                          name="material_guard_cls",
                          exist_ok=True
                          )

    # # Run inference with the trained model
    # results = model("path/to/image.jpg")
