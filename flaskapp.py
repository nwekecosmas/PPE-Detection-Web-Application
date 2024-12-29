import os
import cv2
from flask import Flask, render_template, request, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Set up the upload folder and detected folder
UPLOAD_FOLDER = os.path.join('static', 'upload')
DETECTED_FOLDER = os.path.join('static', 'detected')

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)  # Create the upload directory if it doesn't exist
# if not os.path.exists(DETECTED_FOLDER):
#     os.makedirs(DETECTED_FOLDER)  # Create the detected directory if it doesn't exist

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER

# Initialize YOLO model
model = YOLO('ppe.pt')  # Trained model

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            filename = secure_filename(f.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()

            if file_extension not in ['jpg', 'jpeg', 'png', 'mp4']:
                return "Unsupported file type", 400

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(filepath)

            uploaded_image_url = url_for('display', filename=filename)

            if file_extension in ['jpg', 'jpeg', 'png']:
                img = cv2.imread(filepath)
                results = model(img)

                # Save the detected image in the detected folder
                detected_image_filename = 'detected_' + filename
                detected_image_path = os.path.join(app.config['DETECTED_FOLDER'], detected_image_filename)

                # Plot detections on the image
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int
                        conf = box.conf[0].item()
                        cls = int(box.cls[0])
                        label = f'{model.names[cls]} {conf:.2f}'
                        color = (0, 255, 0)  # Green for detected objects
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                cv2.imwrite(detected_image_path, img)  # Save the image with detections
                detected_image_url = url_for('display_detected', filename=detected_image_filename)

                return render_template('firstpage.html', uploaded_image_path=uploaded_image_url, detected_image_path=detected_image_url)

            elif file_extension == 'mp4':
                video_path = filepath
                cap = cv2.VideoCapture(video_path)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_path = os.path.join(app.config['DETECTED_FOLDER'], 'detected_output.mp4')
                out = cv2.VideoWriter(out_path, fourcc, 30, (frame_width, frame_height))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame)
                    res_plotted = results[0].plot()
                    out.write(res_plotted)

                cap.release()
                out.release()

                return render_template('firstpage.html', video_feed=True, video_path=url_for('static', filename='detected/detected_output.mp4'))

    return render_template('firstpage.html')

@app.route('/video_feed')
def video_feed():
    def gen_frames():
        cap = cv2.VideoCapture(0)  # Open the default camera
        while True:
            success, frame = cap.read()
            if not success:
                break
            else:
                results = model(frame)  # Detect objects in the frame
                res_plotted = results[0].plot()
                ret, buffer = cv2.imencode('.jpg', res_plotted)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Streaming the frame
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/detected/<filename>')
def display_detected(filename):
    return send_from_directory(app.config['DETECTED_FOLDER'], filename)

@app.route('/static/<filename>')
def display(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
