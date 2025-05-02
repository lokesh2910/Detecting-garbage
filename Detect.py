from ultralytics import YOLO
import cv2

# Load your trained model (raw string used)
model = YOLO(r"C:\Users\Lokesh\PycharmProjects\GreenEye\runs\detect\train36\weights\best.pt")

# Load a sample image
image_path = r"C:\Users\Lokesh\PycharmProjects\GreenEye\runs\detect\exp2\garbage-2729608_640.jpg"
img = cv2.imread(image_path)

# Run inference (prediction)
results = model.predict(source=image_path, save=True)

# Optional: Show image (after prediction is saved)
predicted_img = cv2.imread(r"C:\Users\Lokesh\PycharmProjects\GreenEye\runs\detect\exp2\garbage-2729608_640.jpg")
cv2.imshow('Prediction', predicted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
