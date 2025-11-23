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

def read_rtsp_stream(rtsp_url, show_gui=False):
    # Initialize the video capture object
    video_capture = cv2.VideoCapture(rtsp_url)

    # Check if the stream is opened
    if not video_capture.isOpened():
        raise ConnectionError("Could not open RTSP stream")

    ret, frame1 = video_capture.read()
    ret, frame2 = video_capture.read()

    if not ret:
        video_capture.release()
        raise ConnectionError("Could not read initial frames from stream")

    # Read the stream
    while True:
        contours = detect_movement(frame1, frame2)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 900:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            save_photo(frame1)

        if show_gui:
            cv2.imshow('RTSP Stream', frame1)

        frame1 = frame2
        ret, frame2 = video_capture.read()

        # Check if frame was read successfully
        if not ret:
            video_capture.release()
            if show_gui:
                cv2.destroyAllWindows()
            raise ConnectionError("Lost connection to RTSP stream")

        # Exit on key press (only in GUI mode)
        if show_gui and cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            return  # Normal exit, don't raise exception
        elif not show_gui:
            # In headless mode, add a small delay to prevent CPU overload
            time.sleep(0.01)

    # Release the video_capture object
    video_capture.release()
    if show_gui:
        cv2.destroyAllWindows()



def main():
    """Main function with automatic reconnection logic"""
    camera_ip = os.getenv("CAM_IP")
    username = os.getenv("CAM_USERNAME")
    password = os.getenv("CAM_PASSWORD")

    # Check if HEADLESS mode is enabled (default: True - no GUI)
    headless = os.getenv("HEADLESS", "True").lower() in ["true", "1", "yes"]
    show_gui = not headless

    # Build RTSP URL
    rtsp_url = f"rtsp://{username}:{password}@{camera_ip}/stream1"

    retry_delay = 10  # seconds 
    attempt = 0

    print("=== Sistema de Detección de Movimiento ===")
    print(f"Cámara: {camera_ip}")
    print(f"Modo: {'HEADLESS (sin ventana)' if headless else 'GUI (con ventana)'}")
    if show_gui:
        print("Presiona 'q' para salir\n")
    else:
        print("Presiona Ctrl+C para salir\n")

    while True:
        attempt += 1
        try:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Intento #{attempt}: Conectando a cámara...")
            read_rtsp_stream(rtsp_url, show_gui=show_gui)
            # If we get here, user pressed 'q' to exit normally (only in GUI mode)
            print("\nSaliendo del programa...")
            break

        except ConnectionError as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error de conexión: {e}")
            print(f"Reintentando en {retry_delay} segundos...\n")
            time.sleep(retry_delay)

        except KeyboardInterrupt:
            print("\n\nInterrupción detectada (Ctrl+C)")
            print("Saliendo del programa...")
            break

        except Exception as e:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error inesperado: {e}")
            print(f"Reintentando en {retry_delay} segundos...\n")
            time.sleep(retry_delay)


if __name__ == '__main__':
    main()