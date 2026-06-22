from database import get_connection
import time

# -----------------------------
# Cooldown Dictionary
# -----------------------------

last_seen = {}

COOLDOWN = 5   # seconds


def mark_attendance(student_id):

    current_time = time.time()

    # -----------------------------
    # Cooldown Check
    # -----------------------------

    if student_id in last_seen:

        if current_time - last_seen[student_id] < COOLDOWN:

            return "Cooldown"

    conn = get_connection()

    cursor = conn.cursor()

    # -----------------------------
    # Already Marked Today ?
    # -----------------------------

    cursor.execute(
        """
        SELECT id

        FROM attendance

        WHERE student_id = %s

        AND attendance_date = CURDATE()
        """,
        (student_id,)
    )

    result = cursor.fetchone()

    if result:

        cursor.close()

        conn.close()

        last_seen[student_id] = current_time

        return "Already Marked"

    # -----------------------------
    # Insert Attendance
    # -----------------------------

    cursor.execute(
        """
        INSERT INTO attendance
        (
            student_id,
            attendance_date,
            attendance_time,
            status
        )

        VALUES
        (
            %s,
            CURDATE(),
            CURTIME(),
            'Present'
        )
        """,
        (student_id,)
    )

    conn.commit()

    cursor.close()

    conn.close()

    last_seen[student_id] = current_time

    return "Attendance Marked"

