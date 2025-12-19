import face_recognition
import cv2
import pickle
import os
import datetime
import time
import pandas as pd
from hardware import buzz_success, buzz_error, display_message, cleanup

# --- Constants ---
ENCODINGS_PATH = "../encodings/face_encodings.pickle"
REPORTS_DIR = "../attendance_reports"
TOLERANCE = 0.45  # Lower is stricter (0.6 is default)

def mark_attendance(name, user_id, lecture_name):
    """
    Marks attendance in an Excel file.
    Returns:
        True: If marked successfully
        False: If already marked for this lecture
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    filename = f"attendance_{date_str}.xlsx"
    file_path = os.path.join(REPORTS_DIR, filename)

    # Initialize or Load Dataframe
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["ID", "Name", "Date", "Time", "Lecture"])
    else:
        df = pd.read_excel(file_path)

    # Check duplicates for same ID and Lecture
    already_present = df[
        (df['ID'] == user_id) & 
        (df['Lecture'] == lecture_name)
    ]

    if not already_present.empty:
        return False

    # Append new record
    new_record = pd.DataFrame([{
        "ID": user_id,
        "Name": name,
        "Date": date_str,
        "Time": time_str,
        "Lecture": lecture_name
    }])
    
    # Use pd.concat instead of append (deprecated)
    df = pd.concat([df, new_record], ignore_index=True)
    df.to_excel(file_path, index=False)
    
    return True

def main():
    # 1. Load Encodings
    print("[INFO] Loading encodings...")
    try:
        data = pickle.loads(open(ENCODINGS_PATH, "rb").read())
    except FileNotFoundError:
        print("[ERROR] Encodings file not found. Run train_model.py first.")
        return

    # 2. Setup Camera
    print("[INFO] Starting Camera...")
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)
    video_capture.set(4, 480)

    # 3. Get Lecture Info
    lecture_name = input("Enter Lecture Name (e.g., Math_101): ").strip()
    display_message("System Ready", f"Lec: {lecture_name}")
    time.sleep(2)

    print("[INFO] Attendance System Started. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=TOLERANCE)
            name = "Unknown"
            user_id = "N/A"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(data["encodings"], face_encoding)
            
            best_match_index = -1
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()

            if best_match_index != -1 and matches[best_match_index]:
                name = data["names"][best_match_index]
                user_id = data["ids"][best_match_index]
                
                # --- LOGIC FOR ATTENDANCE ---
                status = mark_attendance(name, user_id, lecture_name)
                
                if status:
                    # New attendance
                    msg1 = f"Welcome {name}"
                    msg2 = "Marked: Success"
                    print(f"[SUCCESS] {name} ({user_id}) marked.")
                    display_message(msg1, msg2)
                    buzz_success()
                else:
                    # Already marked
                    msg1 = f"Hi {name}"
                    msg2 = "Already Marked"
                    # print(f"[INFO] {name} already present.") # Optional: Reduce spam
                    display_message(msg1, msg2)
                    # buzz_error() # Optional: beep on duplicate scan?

            else:
                display_message("Unknown Face", "Access Denied")
            
            face_names.append(f"{name} ({user_id})")

        # Display the results on the screen (GUI)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            color = (0, 255, 0) if "Unknown" not in name else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        cv2.imshow('Attendance System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup()
