"""Main program for the exercise form detector (push-ups, squats, sit-ups).

This file opens the webcam, uses MediaPipe to find body landmarks, checks the
user's form, and counts repetitions. The main idea is that joint angles can show
whether the body is in a good position for the chosen exercise.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import BasicPoseModule as pm
import FrontView
import SideView
import SitUp
import Squat
from RepCounter import RepCounter


def select_exercise():
    """Let the user choose which exercise to track."""
    while True:
        selection = input(
            "Select exercise - PUSH-UP (P), SQUAT (S) or SIT-UP (U): "
        )
        selection = selection.strip().lower()
        if selection in ("p", "pushup", "push-up", "push up"):
            return "pushup"
        if selection in ("s", "squat"):
            return "squat"
        if selection in ("u", "situp", "sit-up", "sit up"):
            return "situp"
        print("Please enter P for push-up, S for squat or U for sit-up.")


def select_mode():
    """Let the user choose the view used to judge form.

    Side view is better for checking elbow and body angles. Front view compares
    the left and right arm positions.
    """
    while True:
        # This choice changes which angle rules the program uses for the check.
        selection = input("Select push-up view - SIDE VIEW (S) or FRONT VIEW (F): ")
        selection = selection.strip().lower()
        if selection in ("s", "side"):
            return "side"
        if selection in ("f", "front"):
            return "front"
        print("Please enter S for side view or F for front view.")


def main():
    """Run the webcam loop and display form feedback and rep count."""
    # The selected exercise decides which form rules and rep angles are used.
    exercise = select_exercise()

    # Push-ups can be checked from the front or the side. Squats and sit-ups are
    # always checked from the side because the key angles are easiest to see there.
    if exercise == "pushup":
        view_mode = select_mode()
    else:
        view_mode = "side"

    # These fonts are used to draw text directly onto the video frame.
    font_paths = (
        r"C:\Windows\Fonts\times.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Microsoft/Times New Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    )
    for font_path in font_paths:
        try:
            feedback_font = ImageFont.truetype(font_path, 40)
            angle_font = ImageFont.truetype(font_path, 28)
            break
        except OSError:
            continue
    else:
        feedback_font = ImageFont.load_default()
        angle_font = ImageFont.load_default()

    # This object gives us the pose landmarks and helper functions for angle calculations.
    detector = pm.poseDetector()

    # Open the default webcam.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not access camera")
        cap.release()
        detector.pose.close()
        return

    # The rep counter tracks the movement phase so it can count full cycles.
    # Each exercise has its own up/down angle limits.
    if exercise == "squat":
        counter = RepCounter(Squat.SQUAT_UP_ANGLE, Squat.SQUAT_DOWN_ANGLE)
        window_name = "Squat Form Detector"
    elif exercise == "situp":
        counter = RepCounter(SitUp.SITUP_UP_ANGLE, SitUp.SITUP_DOWN_ANGLE)
        window_name = "Sit-Up Form Detector"
    else:
        counter = RepCounter()
        window_name = "Push-Up Form Detector"

    try:
        while True:
            # Read one camera frame. If the frame is missing, the loop stops.
            success, img = cap.read()
            if not success:
                print("Could not read from camera")
                break

            # findPose() draws the body skeleton, and findPosition() stores the
            # landmark data needed for angle checks.
            img = detector.findPose(img, True)
            lm_list = detector.findPosition(img, False)

            # This text is replaced by feedback such as "GOOD FORM" or "FIX FORM"
            # when a pose is found.
            feedback = "NO POSE DETECTED"
            feedback_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(feedback_img)

            # If MediaPipe finds a pose, we can judge the body position.
            if lm_list:
                if exercise == "squat":
                    # Squat check uses the knee angle and how far the back leans.
                    result = Squat.detect_squat(detector, img)
                elif exercise == "situp":
                    # Sit-up check uses the hip angle and the knee bend.
                    result = SitUp.detect_situp(detector, img)
                elif view_mode == "side":
                    # Side view checks the elbow, shoulder, and hip angles.
                    # Example: a good down position usually has a smaller elbow angle.
                    result = SideView.detect_side_view(detector, img)
                else:
                    # Front view compares the left and right elbow angles.
                    result = FrontView.detect_front_view(detector, img)

                view_feedback, angles, counter_angle = result
                if view_feedback is not None:
                    feedback = view_feedback

                # Each angle value is drawn on the video so the user can see what
                # the program is measuring.
                for y, label, angle in angles:
                    draw.text(
                        (20, y),
                        f"{label}: {int(angle)}",
                        font=angle_font,
                        fill=(0, 255, 0)
                    )

                # The counter only receives the main angle for the selected view.
                # A full rep is counted when the movement moves through the up/down
                # angles in the right order.
                if counter_angle is not None:
                    counter.update(counter_angle)

            # Display the final feedback and rep count on the screen.
            draw.text(
                (20, 150), feedback, font=feedback_font, fill=(0, 255, 0)
            )
            draw.text(
                (20, 205), f"{exercise.upper()}  |  Reps: {counter.reps}  |  {view_mode.upper()} VIEW",
                font=angle_font, fill=(0, 255, 0)
            )

            # Convert the image back so OpenCV can show it in the window.
            img = cv2.cvtColor(np.asarray(feedback_img), cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, img)

            # Press q to close the camera and exit the program.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Always close the webcam and window, even if the loop stops early.
        cap.release()
        detector.pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
