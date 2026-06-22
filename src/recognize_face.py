# import cv2
# import os
# import time
# from attendance import mark_attendance
# from database import get_connection
# from config import PHONE_CAMERA_URL

# # ---------------------------------
# # Project Paths
# # ---------------------------------

# BASE_DIR = os.path.dirname(
#     os.path.dirname(
#         os.path.abspath(__file__)
#     )
# )

# cascade_path = os.path.join(
#     BASE_DIR,
#     "assets",
#     "haarcascade_frontalface_default.xml"
# )

# trainer_path = os.path.join(
#     BASE_DIR,
#     "trainer",
#     "trainer.yml"
# )

# # ---------------------------------
# # Load Haar Cascade
# # ---------------------------------

# face_detector = cv2.CascadeClassifier(
#     cascade_path
# )

# if face_detector.empty():

#     print("Cannot load Haar Cascade.")

#     exit()

# # ---------------------------------
# # Check Model
# # ---------------------------------

# if not os.path.exists(trainer_path):

#     print("trainer.yml not found.")

#     print("Please train the model first.")

#     exit()

# # ---------------------------------
# # Load LBPH Model
# # ---------------------------------

# recognizer = cv2.face.LBPHFaceRecognizer_create()

# recognizer.read(
#     trainer_path
# )

# # ---------------------------------
# # Database Connection
# # ---------------------------------

# conn = get_connection()

# cursor = conn.cursor()

# # ---------------------------------
# # Open Camera
# # ---------------------------------

# cap = cv2.VideoCapture(
#     PHONE_CAMERA_URL
# )

# if not cap.isOpened():

#     print("Cannot open camera.")

#     cursor.close()

#     conn.close()

#     exit()

# print("\nFace Recognition Started...\n")

# while True:

#     ret, frame = cap.read()

#     if not ret:
#         break

#     gray = cv2.cvtColor(

#         frame,

#         cv2.COLOR_BGR2GRAY

#     )

#     faces = face_detector.detectMultiScale(

#         gray,

#         scaleFactor=1.3,

#         minNeighbors=5

#     )

#     for (x, y, w, h) in faces:

#         face = gray[
#             y:y+h,
#             x:x+w
#         ]

#         face = cv2.resize(

#             face,

#             (200, 200)

#         )

#         student_id, confidence = recognizer.predict(
#             face
#         )

#         # ---------------------------------
#         # Threshold
#         # ---------------------------------

#         if confidence < 55:

#             cursor.execute(

#                 """
#                 SELECT
#                     student_name,
#                     roll_number
#                 FROM students
#                 WHERE id = %s
#                 """,

#                 (student_id,)

#             )

#             result = cursor.fetchone()

#             if result:

#                 student_name = result[0]

#                 roll_number = result[1]

#                 display_text = (
#                     f"{student_name}"
#                 )

#                 print(

#                     f"Recognized : "

#                     f"{student_name} "

#                     f"({roll_number}) "

#                     f"Confidence : "

#                     f"{confidence:.2f}"

#                 )

#             else:

#                 display_text = "Unknown"

#         else:

#             display_text = "Unknown"

#         cv2.rectangle(

#             frame,

#             (x, y),

#             (x + w, y + h),

#             (0, 255, 0),

#             2

#         )

#         cv2.putText(

#             frame,

#             display_text,

#             (x, y - 10),

#             cv2.FONT_HERSHEY_SIMPLEX,

#             0.8,

#             (255, 0, 0),

#             2

#         )

#         cv2.putText(

#             frame,

#             f"Conf: {confidence:.2f}",

#             (x, y + h + 25),

#             cv2.FONT_HERSHEY_SIMPLEX,

#             0.7,

#             (0, 255, 255),

#             2

#         )

#     cv2.imshow(

#         "Face Recognition",

#         frame

#     )

#     if cv2.waitKey(1) & 0xFF == ord("q"):

#         break

# cap.release()

# cv2.destroyAllWindows()

# cursor.close()

# conn.close()

# print("\nRecognition Stopped.")



import cv2
import os

from database import get_connection
from config import PHONE_CAMERA_URL
from attendance import mark_attendance

# ---------------------------------
# Project Paths
# ---------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

cascade_path = os.path.join(
    BASE_DIR,
    "assets",
    "haarcascade_frontalface_default.xml"
)

trainer_path = os.path.join(
    BASE_DIR,
    "trainer",
    "trainer.yml"
)

# ---------------------------------
# Load Haar Cascade
# ---------------------------------

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("Error: Cannot load Haar Cascade.")
    exit()

# ---------------------------------
# Check Trained Model
# ---------------------------------

if not os.path.exists(trainer_path):

    print("trainer.yml not found.")
    print("Please train the model first.")

    exit()

# ---------------------------------
# Load Face Recognizer
# ---------------------------------

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read(trainer_path)

# ---------------------------------
# Database Connection
# ---------------------------------

conn = get_connection()

cursor = conn.cursor()

# ---------------------------------
# Open Camera
# ---------------------------------

cap = cv2.VideoCapture(PHONE_CAMERA_URL)

if not cap.isOpened():

    print("Cannot Open Camera.")

    cursor.close()
    conn.close()

    exit()

print("\n===================================")
print(" Face Recognition Started")
print(" Press Q to Exit")
print("===================================\n")

# ---------------------------------
# Recognition Loop
# ---------------------------------

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
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        # Ignore very small faces

        if w < 100 or h < 100:
            continue

        face = gray[
            y:y + h,
            x:x + w
        ]

        face = cv2.resize(
            face,
            (200, 200)
        )

        student_id, confidence = recognizer.predict(face)

        display_text = "Unknown"
        attendance_status = ""

        # ---------------------------------
        # Recognition Threshold
        # ---------------------------------

        if confidence < 55:

            cursor.execute(
                """
                SELECT
                    student_name,
                    roll_number
                FROM students
                WHERE id=%s
                """,
                (student_id,)
            )

            result = cursor.fetchone()

            if result:

                student_name = result[0]
                roll_number = result[1]

                display_text = student_name

                # -----------------------------
                # Mark Attendance
                # -----------------------------

                attendance_status = mark_attendance(
                    student_id
                )

                print("\n-----------------------------")
                print("Student :", student_name)
                print("Roll No :", roll_number)
                print("Confidence :", round(confidence, 2))
                print("Status :", attendance_status)
                print("-----------------------------")

            else:

                display_text = "Unknown"

        # ---------------------------------
        # Draw Rectangle
        # ---------------------------------

        cv2.rectangle(

            frame,

            (x, y),

            (x + w, y + h),

            (0, 255, 0),

            2

        )

        # ---------------------------------
        # Student Name
        # ---------------------------------

        cv2.putText(

            frame,

            display_text,

            (x, y - 15),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.8,

            (255, 0, 0),

            2

        )

        # ---------------------------------
        # Attendance Status
        # ---------------------------------

        if attendance_status != "":

            cv2.putText(

                frame,

                attendance_status,

                (x, y + h + 25),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2

            )

        # ---------------------------------
        # Confidence
        # ---------------------------------

        cv2.putText(

            frame,

            f"Conf: {confidence:.2f}",

            (x, y + h + 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 255, 255),

            2

        )

    cv2.imshow(
        "Face Recognition Attendance System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------------------------
# Cleanup
# ---------------------------------

cap.release()

cv2.destroyAllWindows()

cursor.close()

conn.close()

print("\nRecognition Stopped Successfully.")

