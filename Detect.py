from ultralytics import YOLO
import cv2
import os
from screeninfo import get_monitors

# Load both models using full paths
model1 = YOLO(r"C:\Users\Lokesh\PycharmProjects\PythonProject\best.pt")
model2 = YOLO(r"C:\Users\Lokesh\OneDrive\Desktop\best.pt")

# Image filenames to process
image_filenames = ["1.jpeg", "2.jpeg", "3.jpeg"]

# Input image folder path
input_folder = r"C:\Users\Lokesh\PycharmProjects\GreenEye\Camera"

# Get screen resolution
monitor = get_monitors()[0]
screen_width, screen_height = monitor.width, monitor.height
margin = 100

# Loop through each image
for filename in image_filenames:
    image_path = os.path.join(input_folder, filename)
    name_without_ext = os.path.splitext(filename)[0]

    for idx, model in enumerate([model1, model2], start=1):
        results = model.predict(
            source=image_path,
            save=True,
            project=r"C:\Users\Lokesh\PycharmProjects\GreenEye\runs\detect",
            name=f"predict_model{idx}_{name_without_ext}",
            exist_ok=True
        )

        save_dir = results[0].save_dir
        predicted_files = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]

        if predicted_files:
            predicted_img_path = os.path.join(save_dir, predicted_files[0])
            predicted_img = cv2.imread(predicted_img_path)

            if predicted_img is not None:
                img_height, img_width = predicted_img.shape[:2]
                width_ratio = (screen_width - margin) / img_width
                height_ratio = (screen_height - margin) / img_height
                scale_ratio = min(width_ratio, height_ratio, 1.0)

                new_width = int(img_width * scale_ratio)
                new_height = int(img_height * scale_ratio)
                resized_img = cv2.resize(predicted_img, (new_width, new_height))

                cv2.imshow(f'Model {idx} Prediction - {filename}', resized_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Unable to load predicted image: {predicted_img_path}")
        else:
            print(f"No predicted images found in: {save_dir}")
