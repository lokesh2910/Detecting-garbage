from ultralytics import YOLO
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    model = YOLO("yolov8m.pt")  # detection model
    model.train(data="C:/Users/Lokesh/PycharmProjects/PythonProject/litter.yaml", epochs=5)
