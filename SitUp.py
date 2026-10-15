"""Check sit-up form using the side camera view.

Side view is used because the angle between the upper body and the thighs
(the hip angle) clearly changes as the person lies down and sits up.
"""

from BasicPoseModule import (
    VISIBILITY_THRESHOLD, get_reliable_angle, pick_visible_side
)

# The hip angle is measured at the hip between the shoulder and the knee.
# Lying down with bent knees gives a larger angle; sitting up gives a smaller one.
SITUP_UP_ANGLE = 120    # lying back (start/end of each rep)
SITUP_DOWN_ANGLE = 80   # sitting up (top of each rep)

# Sit-ups are normally done with the knees bent. A knee angle above this value
# means the legs are too straight.
MAX_KNEE_ANGLE = 130

# (shoulder, hip, knee, ankle) landmark ids for each side of the body.
LEFT_SIDE = (11, 23, 25, 27)
RIGHT_SIDE = (12, 24, 26, 28)


def detect_situp(detector, img):
    """Return the sit-up form result and the angles used for display."""
    shoulder_id, hip_id, knee_id, ankle_id = pick_visible_side(
        detector, LEFT_SIDE, RIGHT_SIDE
    )

    if not all(
        detector.isLandmarkVisible(point, VISIBILITY_THRESHOLD)
        for point in (shoulder_id, hip_id, knee_id, ankle_id)
    ):
        return None, (), None

    hip = get_reliable_angle(detector, img, (shoulder_id, hip_id, knee_id))
    knee = get_reliable_angle(detector, img, (hip_id, knee_id, ankle_id))
    if hip is None or knee is None:
        return None, (), None

    # Good form means the knees stay bent while the upper body moves.
    if knee <= MAX_KNEE_ANGLE:
        feedback = "GOOD FORM"
    else:
        feedback = "FIX FORM"

    angles = (
        (20, "Hip", hip),
        (60, "Knee", knee),
    )
    return feedback, angles, hip
