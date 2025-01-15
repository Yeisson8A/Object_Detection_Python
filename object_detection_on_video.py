import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import cv2

# Especificar la configuración del detector de objetos
options = vision.ObjectDetectorOptions(
     base_options=BaseOptions(model_asset_path="./Models/efficientdet_lite_float32.tflite"),
     max_results=10,
     score_threshold=0.15,
     running_mode=vision.RunningMode.VIDEO)
detector = vision.ObjectDetector.create_from_options(options)

# Leer el video de entrada
cap = cv2.VideoCapture("./Data/Video_1.mp4")
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)

color = (100, 255, 0)
text_color = (255, 255, 255)
# Recorrer cada fotograma
for frame_index in range(int(frame_count)):
     ret, frame = cap.read()
     if ret == False:
          break
     
     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     frame_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
     # Calcular la marca temporal del frame actual (en milisegundos)
     frame_timestamp_ms = int(1000 * frame_index / fps)
     # Detectar objetos sobre el frame
     detection_result = detector.detect_for_video(frame_rgb, frame_timestamp_ms)

     # Leer cada una de las detecciones obtenidas
     for detection in detection_result.detections:
          bbox = detection.bounding_box
          bbox_x, bbox_y, bbox_w, bbox_h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
          category = detection.categories[0]
          score = category.score*100
          category_name = category.category_name
          print(f"{category_name}: {score:.2f}%")
          cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y - 30), color, -1)
          cv2.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), color, 2)
          cv2.putText(frame, f"{category_name} {score:.2f}%", (bbox_x + 5, bbox_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                         0.6, text_color, 2)
     
     # Visualizar video
     cv2.imshow('Video', frame)
     # waitKey specifies the value in millisecond 
     # for which a particular frame will be shown
     k = cv2.waitKey(60)
 
     # if 'q' is pressed then above while loop will
     # break and video reading and writing
     # process will stop
     if k == ord("q"):
        break

# Liberar objetos
cap.release()
cv2.destroyAllWindows()