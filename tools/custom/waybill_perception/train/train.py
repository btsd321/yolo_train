from ultralytics import YOLO

if __name__ == '__main__':
    dataset_config_file = f"D:/Project/yolo_train/tools/custom/waybill_perception/config/dataset.yaml"
    model_config_file = f"D:/Project/yolo_train/tools/custom/waybill_perception/config/yolo26m-obb.yaml"

    # Load a COCO-pretrained YOLO26n model
    model = YOLO(model_config_file, task="obb")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data=dataset_config_file,
                          epochs=300,
                          imgsz=640,
                          batch=16,
                          device=0,
                          project="output",
                          name="waybill_perception_obb",
                          exist_ok=True
                          )

    # # Run inference with the YOLO26n model on the 'bus.jpg' image
    # results = model("path/to/bus.jpg")