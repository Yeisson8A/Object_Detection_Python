import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import cv2

# Especificar la configuración del detector de objetos
options = vision.ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path="./Models/efficientdet_lite_float32.tflite"),
    max_results=10,
    score_threshold=0.2,
    running_mode=vision.RunningMode.IMAGE)
detector = vision.ObjectDetector.create_from_options(options)

# Leer la imagen de entrada
image = cv2.imread("./Data/Imagen_1.jpg")

# Convertir imagen a RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# Detectar objetos sobre la imagen
detection_result = detector.detect(image_rgb)

color = (100, 255, 0)
text_color = (255, 255, 255)
# Leer cada una de las detecciones obtenidas
for detection in detection_result.detections:
    # Bounding box
    bbox = detection.bounding_box
    bbox_x, bbox_y, bbox_w, bbox_h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
    # Score y Category name
    category = detection.categories[0]
    score = category.score * 100
    category_name = category.category_name
    print(f"{category_name}: {score:.2f}%")
    cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y - 30), color, -1)
    cv2.rectangle(image, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), color, 2)
    cv2.putText(image, f"{category_name}: {score:.2f}%", (bbox_x + 5, bbox_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, text_color, 2)

# Visualizar imagen
image_scale = cv2.resize(image, (540, 540))
cv2.imshow("Image", image_scale)
cv2.waitKey(0)
cv2.destroyAllWindows()