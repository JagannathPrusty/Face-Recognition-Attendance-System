import cv2
import os
from train_model import train_model

from database import get_connection
from config import PHONE_CAMERA_URL

# -----------------------------
# Project Paths
# -----------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

cascade_path = os.path.join(
    BASE_DIR,
    "assets",
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(
    cascade_path
)

if face_detector.empty():
    print("Error: Cannot load Haar Cascade.")
    exit()

# -----------------------------
# Student Details
# -----------------------------

student_name = input(
    "Enter Student Name : "
).strip()

roll_number = input(
    "Enter Roll Number : "
).strip().upper()

department = input(
    "Enter Department : "
).strip()

# Use Roll Number as folder name
folder_name = roll_number

dataset_folder = os.path.join(
    BASE_DIR,
    "dataset",
    folder_name
)

# -----------------------------
# MySQL Connection
# -----------------------------

conn = get_connection()
cursor = conn.cursor()

# Check duplicate Roll Number

cursor.execute(
    """
    SELECT id
    FROM students
    WHERE roll_number = %s
    """,
    (roll_number,)
)

if cursor.fetchone():

    print(
        "\nStudent already exists!"
    )

    cursor.close()
    conn.close()

    exit()

# Create Dataset Folder

os.makedirs(
    dataset_folder,
    exist_ok=True
)

# Insert Student

cursor.execute(
    """
    INSERT INTO students
    (
        student_name,
        roll_number,
        department,
        image_folder
    )

    VALUES
    (
        %s,
        %s,
        %s,
        %s
    )
    """,

    (
        student_name,
        roll_number,
        department,
        folder_name
    )
)

conn.commit()

print("\nStudent Registered Successfully.")

# -----------------------------
# Open Camera
# -----------------------------

cap = cv2.VideoCapture(
    PHONE_CAMERA_URL
)

if not cap.isOpened():

    print(
        "Cannot Open Camera."
    )

    cursor.close()
    conn.close()

    exit()

MAX_IMAGES = 200

count = 0

frame_skip = 0

print("\nCapturing Images...\n")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2

        )

        frame_skip += 1

        if frame_skip % 5 != 0:
            continue

        face = gray[
            y:y + h,
            x:x + w
        ]

        face = cv2.resize(
            face,
            (200, 200)
        )

        image_path = os.path.join(

            dataset_folder,

            f"{count + 1}.jpg"

        )

        cv2.imwrite(

            image_path,

            face

        )

        count += 1

        print(
            f"Captured {count}"
        )

        if count >= MAX_IMAGES:
            break

    cv2.putText(

        frame,

        f"{count}/{MAX_IMAGES}",

        (20, 40),

        cv2.FONT_HERSHEY_SIMPLEX,

        1,

        (0, 255, 0),

        2

    )

    cv2.imshow(

        "Capture Faces",

        frame

    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if count >= MAX_IMAGES:
        break

cap.release()

cv2.destroyAllWindows()

cursor.close()

conn.close()

print("\nImages Captured Successfully.")


print(
    "Saved To:",
    dataset_folder
)
print("\nRetraining model...")

success = train_model()

if success:
    print("Model updated successfully.")
else:
    print("Model update failed.")


