import os
import cv2
import numpy as np
from PIL import Image

from database import get_connection

# =====================================================
# Project Paths
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset"
)

TRAINER_PATH = os.path.join(
    BASE_DIR,
    "trainer",
    "trainer.yml"
)


# =====================================================
# Train Model Function
# =====================================================

def train_model():

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            student_name,
            image_folder
        FROM students
        ORDER BY id
        """
    )

    students = cursor.fetchall()

    print("\n===================================")
    print("       TRAINING STARTED")
    print("===================================\n")

    # ---------------------------------

    for student_id, student_name, image_folder in students:

        folder_path = os.path.join(
            DATASET_PATH,
            image_folder
        )

        if not os.path.exists(folder_path):

            print(
                f"[SKIPPED] {student_name}"
            )

            print(
                f"Folder Not Found -> {folder_path}\n"
            )

            continue

        print(
            f"Reading Images : {student_name}"
        )

        image_count = 0

        for file_name in os.listdir(folder_path):

            image_path = os.path.join(
                folder_path,
                file_name
            )

            try:

                image = Image.open(
                    image_path
                ).convert("L")

                image = image.resize(
                    (200, 200)
                )

                image_np = np.array(
                    image,
                    dtype="uint8"
                )

                faces.append(
                    image_np
                )

                labels.append(
                    student_id
                )

                image_count += 1

            except Exception as e:

                print(
                    f"Error Reading : {image_path}"
                )

                print(e)

        print(
            f"Loaded {image_count} Images\n"
        )

    # ---------------------------------

    if len(faces) == 0:

        print("\nNo Images Found For Training.\n")

        cursor.close()
        conn.close()

        return False

    print("\nTraining Model...\n")

    recognizer.train(

        faces,

        np.array(labels)

    )

    os.makedirs(

        os.path.dirname(TRAINER_PATH),

        exist_ok=True

    )

    recognizer.save(

        TRAINER_PATH

    )

    print("===================================")
    print(" Model Trained Successfully")
    print("===================================")
    print(TRAINER_PATH)
    print("===================================\n")

    cursor.close()
    conn.close()

    return True


# =====================================================
# Run Directly
# =====================================================

if __name__ == "__main__":

    success = train_model()

    if success:

        print("Training Completed Successfully.")

    else:

        print("Training Failed.")

