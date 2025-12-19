import face_recognition
import pickle
import os
import cv2

# Path to the dataset
DATASET_PATH = "../dataset"
ENCODINGS_PATH = "../encodings/face_encodings.pickle"

def train_encodings():
    print("[INFO] Quantifying faces...")
    imagePaths = []
    
    # Loop over the directory structure
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                imagePaths.append(path)

    knownEncodings = []
    knownNames = []
    knownIDs = []

    total_images = len(imagePaths)
    
    for (i, imagePath) in enumerate(imagePaths):
        print(f"[INFO] Processing image {i + 1}/{total_images}: {imagePath}")
        
        # Extract the person name and ID from the folder name
        # Folder structure: dataset/Name_ID/image.jpg
        # We can also get it from the parent directory name
        path_parts = imagePath.split(os.path.sep)
        folder_name = path_parts[-2] # e.g., "Nikhil_101"
        
        try:
            name, user_id = folder_name.split("_")
        except ValueError:
            print(f"[WARN] Folder {folder_name} format incorrect. Expected Name_ID. Using folder name as Name.")
            name = folder_name
            user_id = "Unknown"

        # Load the input image and convert it from BGR (OpenCV ordering)
        # to RGB (dlib ordering)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model="hog")

        # Compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
            knownIDs.append(user_id)

    print("[INFO] Serializing encodings...")
    data = {
        "encodings": knownEncodings, 
        "names": knownNames,
        "ids": knownIDs
    }
    
    # Ensure encodings directory exists
    if not os.path.exists("../encodings"):
        os.makedirs("../encodings")
        
    f = open(ENCODINGS_PATH, "wb")
    f.write(pickle.dumps(data))
    f.close()
    print(f"[INFO] Encodings saved to {ENCODINGS_PATH}")

if __name__ == "__main__":
    train_encodings()
