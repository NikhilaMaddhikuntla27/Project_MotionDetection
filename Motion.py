from flask import Flask, Response, render_template
import cv2

# Initialize Flask app
app = Flask(__name__)

# Initialize MOG2 background subtractor
mog2 = cv2.createBackgroundSubtractorMOG2()

def generate_frames():
    # Open the default camera (or provide a video file path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply MOG2 to detect motion
        fg_mask = mog2.apply(frame)

        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', fg_mask)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    # Render the HTML template
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Stream the motion detection output
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Create a templates folder and add index.html
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create a simple HTML file
    with open('templates/index.html', 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Motion Detection</title>
        </head>
        <body>
            <h1>AI-Powered Motion Detection</h1>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
        </body>
        </html>
        ''')

    # Run the Flask app
    app.run(host='0.0.0.0', port=5001)