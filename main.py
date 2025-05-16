from ultralytics import YOLO
# training
model = YOLO("ultralytics/backbone/EfficientNet.yaml")
results = model.train(data="F:/NCKH/dataset/pomegranate_data", epochs=3, imgsz=640)