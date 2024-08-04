import os
import cv2
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

def detect_movement(frame1, frame2):
    # Crop the frames to exclude the timestamp area
    crop_y = 50  # adjust this value to match the height of the timestamp area
    frame1_cropped = frame1[crop_y:, :]  # crop the frame from the top
    frame2_cropped = frame2[crop_y:, :]  # crop the frame from the top

    # Process the cropped frames
    diff = cv2.absdiff(frame1_cropped, frame2_cropped)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def save_photo(frame):
    current_time = time.time()
    if not hasattr(save_photo, "last_save_time"):
        save_photo.last_save_time = 0
    if current_time - save_photo.last_save_time >= 3:
        # Create the output directory if it doesn't exist
        if not os.path.exists("data/output"):
            os.makedirs("data/output")
        # Save the image with a timestamp-based file name
        cv2.imwrite(f"data/output/img_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg", frame)
        save_photo.last_save_time = current_time

def read_rtsp_stream(rtsp_url):
    # Initialize the video capture object
    video_capture = cv2.VideoCapture(rtsp_url)
    
    # Check if the stream is opened
    if not video_capture.isOpened():
        print("Error: Could not open RTSP stream")
        return
    
    ret, frame1 = video_capture.read()
    ret, frame2 = video_capture.read()
    
    # Read the stream
    while True:
        contours = detect_movement(frame1, frame2)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 900:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            save_photo(frame1)
        
        cv2.imshow('RTSP Stream', frame1)
        frame1 = frame2
        ret, frame2 = video_capture.read()
        
        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video_capture object
    video_capture.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    camera_ip= os.getenv("CAM_IP")
    port= os.getenv("STREAM_PORT")
    username= os.getenv("CAM_USERNAME")
    password= os.getenv("CAM_PASSWORD")

    read_rtsp_stream(f"rtsp://{username}:{password}@{camera_ip}:{port}/stream1")