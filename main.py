from flask import Flask, render_template, flash, redirect, request, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import requests
app = Flask(__name__)
app.config['SECRET_KEY'] = '1d0b7f7183d919a41952fe68884fad8d'
thres=0.45
classNames=[]
classFile='./static/coco.names'
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8)
public_ip=requests.get('https://api.ipify.org').text

@app.route('/')
def layout():
    return render_template('layout.html')

@app.route('/video')
def video():
    # Define the MIME type for an MJPEG stream
    mjpeg_type = 'multipart/x-mixed-replace; boundary=frame'

    def generate_frames():
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()

            # Perform the hand detection inference
            hands, image = detector.findHands(frame)

            # Draw the detected hands on the frame
            # ...

            # Encode the frame as a JPEG image
            _, jpeg = cv2.imencode('.jpg', image)

            # Yield the JPEG-encoded image as a byte string
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    # Return the MJPEG stream as a Flask Response object
    return Response(generate_frames(), mimetype=mjpeg_type)

@app.route('/show_video')
def show_video():
    return render_template('video.html')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/object')
def object():
    # Define the MIME type for an MJPEG stream
    mjpeg_type = 'multipart/x-mixed-replace; boundary=frame'

    def generate_frame():
        with open(classFile,'rt') as f:
            classNames=f.read().rstrip('\n').split('\n')
        configPath='./static/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath='./static/frozen_inference_graph.pb'
        net=cv2.dnn_DetectionModel(weightsPath,configPath)
        net.setInputSize(320,320)
        net.setInputScale(1.0/127.5)
        net.setInputMean((127.5,127.5,127.5))
        net.setInputSwapRB(True)
        while True:
            ret,frame=cap.read()
            classIds,confs,bbox=net.detect(frame,confThreshold=thres)
            if len(classIds)!=0:
                for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                    cv2.rectangle(frame,box,color=(0,255,0),thickness=2)
                    cv2.putText(frame,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(frame,str(round(confidence*100,2)),(box[0]+280,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                # Draw the detected hands on the frame
                # ...

                # Encode the frame as a JPEG image
            _, jpg = cv2.imencode('.jpg', frame)

                # Yield the JPEG-encoded image as a byte string
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

    # Return the MJPEG stream as a Flask Response object
    return Response(generate_frame(), mimetype=mjpeg_type)
@app.route('/show_object')
def show_object():
    return render_template('object.html')
@app.route('/face')
def face():
    return render_template('face.html')
@app.route('/plate')
def plate():
    return render_template('plate.html')
@app.route('/emotion')
def emotion():
    return render_template('emotion.html')
@app.route('/color')
def color():
    return render_template('color.html')
if __name__ == '__main__':
    app.run(debug=True,host='10.10.20.62',port=3500)

