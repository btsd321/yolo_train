import os
os.environ["YOLO_CONFIG_DIR"] = os.path.join(os.path.dirname(__file__), "../../../../.ultralytics")

from ultralytics import YOLO

if __name__ == '__main__':
    dataset_config_file = f"D:\\Project\\yolo_train\\tools\\custom\\linden_perception\\config\\dataset.yaml"
    model_config_file = f"D:\\Project\\yolo_train\\tools\\custom\\linden_perception\\config\\yolo26m-seg.yaml"

    # Load a COCO-pretrained YOLO26n model
    model = YOLO(model_config_file, task="segment")

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data=dataset_config_file,
                          epochs=300,
                          imgsz=640,
                          batch=4,
                          device=0,
                          project="output",
                          name="linden_perception_seg",
                          exist_ok=True
                          )

    # # Run inference with the YOLO26n model on the 'bus.jpg' image
    # results = model("path/to/bus.jpg")