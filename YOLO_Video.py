from ultralytics import YOLO
import cv2
import math

def video_detection(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Load the YOLO model
    model = YOLO("ppe.pt")
    classNames = ['Protective Helmet', 'Shield', 'Jacket', 'Dust Mask', 'Eye Wear', 'Glove', 'Protective Boots']

    # Initialize video writer
    out = cv2.VideoWriter('static/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while True:
        success, img = cap.read()
        if not success:
            break

        # Run YOLO model
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf}'

                color = (0, 255, 0) if class_name == 'Dust Mask' else (85, 45, 255)
                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(img, label, (x1, y1 - 10), 0, 0.5, color, 2)

        out.write(img)

    cap.release()
    out.release()
