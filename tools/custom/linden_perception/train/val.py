import os
os.environ["YOLO_CONFIG_DIR"] = os.path.join(os.path.dirname(__file__), "../../../../.ultralytics")

from ultralytics import YOLO

if __name__ == '__main__':
    dataset_config_file = f"D:\\Project\\yolo_train\\tools\\custom\\linden_perception\\config\\dataset.yaml"
    # 待评估的模型权重
    model_weights = f"D:\\Project\\yolo_train\\runs\\segment\\linden_perception\\yolo26m_train_20260813\\weights\\best.pt"

    # 加载训练好的模型
    model = YOLO(model_weights, task="segment")

    # 在验证集上评估模型，参数与训练保持一致以保证结果可比性
    # 官方推荐：使用独立的 val 划分、与训练一致的 imgsz、iou 阈值
    metrics = model.val(data=dataset_config_file,
                        split="val",
                        imgsz=640,
                        batch=16,
                        device=0,
                        iou=0.7,            # 与训练一致
                        max_det=300,
                        plots=True,         # 生成混淆矩阵、PR 曲线等可视化
                        save_json=False,
                        project="runs/segment/linden_perception",
                        name="yolo26m_val_20260813",
                        exist_ok=True
                        )

    # 打印关键分割评估指标
    print("\n================ Validation Metrics ================")
    print(f"Precision (P):     {metrics.box.mp:.4f}")
    print(f"Recall (R):        {metrics.box.mr:.4f}")
    print(f"mAP50:             {metrics.box.map50:.4f}")
    print(f"mAP50-95:          {metrics.box.map:.4f}")
    print(f"Mask mAP50:        {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95:     {metrics.seg.map:.4f}")
    print("===================================================")
