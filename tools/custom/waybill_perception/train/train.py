from ultralytics import YOLO

dataset_config_file = f"/home/lixinlong/Project/yolo_train/Data/dataset/waybill_perception/dataset_config.yaml"
model_config_file = f"/home/lixinlong/Project/yolo_train/Data/dataset/waybill_perception/yolo26m-obb.yaml"

# Load a COCO-pretrained YOLO26n model
model = YOLO(model_config_file, task="obb")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data=dataset_config_file, 
                      epochs=300, 
                      imgsz=960,
                      batch=16,
                      device=0,
                      project="/home/lixinlong/Project/yolo_train/Data/output/waybill_perception",
                      name="yolo26m-obb",
                      exist_ok=True
                      )

# Run inference with the YOLO26n model on the 'bus.jpg' image
results = model("path/to/bus.jpg")