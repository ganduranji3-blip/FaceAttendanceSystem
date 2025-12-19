import cv2
import os
import time

def create_dataset():
    # 1. Setup Camera
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    
    # 2. Use OpenCV's built-in Haar Cascade for face detection
    # Note: face_recognition library is better for recognition, but for capturing
    # simple frontal faces quickly, Haar cascade is lightweight and fast.
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 3. Get User Input
    print("\n[INFO] Initializing Face Capture. Look at the camera and wait ...")
    face_name = input("Enter User Name (e.g., Nikhil): ").strip()
    face_id = input("Enter User ID (e.g., 101): ").strip()
    
    folder_name = f"{face_name}_{face_id}"
    dataset_path = os.path.join("../dataset", folder_name)
    
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"[INFO] Created folder: {dataset_path}")
    else:
        print(f"[INFO] Folder already exists. Appending new images.")

    print("\n[INFO] Starting camera... Press 'q' to quit early if needed.")
    
    count = 0
    max_images = 30  # Number of images to capture
    
    while(True):
        ret, img = cam.read()
        if not ret:
            print("[ERROR] Failed to capture image")
            break
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            # Draw rectangle around face (visual feedback)
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            
            # Save the captured image into the datasets folder
            count += 1
            file_name = f"{face_name}.{face_id}.{count}.jpg"
            file_path = os.path.join(dataset_path, file_name)
            
            # Save the gray face area (better for training) or full color
            # We'll save the full color face region
            cv2.imwrite(file_path, img[y:y+h,x:x+w])
            
            print(f"[INFO] Saved {file_name} ({count}/{max_images})")

        cv2.imshow('Face Capture', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= max_images: # Take 30 face sample and stop video
            break

    # Cleanup
    print("\n[INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    create_dataset()
