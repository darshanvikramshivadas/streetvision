from ultralytics import YOLO
import cv2
import numpy as np


def estimate_pothole_depth(width, height, image_width, image_height):
    sensor_height = 10
    average_lane_width = 3000

    pothole_width_mm = (width / image_width) * average_lane_width
    pothole_height_mm = (height / image_height) * sensor_height

    base_area_mm2 = pothole_width_mm * pothole_height_mm

    volume_mm3 = base_area_mm2 * height
    depth_mm = (volume_mm3 * 3) / base_area_mm2

    depth_meters = depth_mm / 1000

    return depth_meters


model = YOLO("best.pt")
class_names = model.names
cap = cv2.VideoCapture('sl1.mp4')
count = 0

focal_length = 1500
distance = 1000

while True:
    ret, img = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = model.predict(img)

    for r in results:
        boxes = r.boxes
        masks = r.masks
        
    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = class_names[d]
                x, y, width, height = cv2.boundingRect(contour)
                cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                
                if c == "street-light":
                    cv2.putText(img, "Street Light", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    # Estimate pothole depth
                    depth = estimate_pothole_depth(width, height, focal_length, distance)
                    
                    # Categorize pothole based on depth
                    if depth < 0.1:  # Example depth thresholds
                        category = "Shallow"
                    elif depth < 0.3:
                        category = "Moderate"
                    else:
                        category = "Deep"
                    
                    pothole_width_meters = (width * distance) / (focal_length * 1000)
                    pothole_height_meters = (height * distance) / (focal_length * 1000)
                    
                    cv2.putText(img, f"Depth: {depth:.2f} meters ({category})", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(img, f"Dimensions: {pothole_width_meters:.2f} meters x {pothole_height_meters:.2f} meters", (x, y - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                 
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
