"""Check squat form using the side camera view.

Side view is used because the knee bend and the lean of the back are easiest to
see from the side. The knee angle is the main angle used to count repetitions.
"""

import math

from BasicPoseModule import (
    VISIBILITY_THRESHOLD, get_reliable_angle, pick_visible_side
)

# A larger knee angle means the legs are straighter (standing). A smaller knee
# angle means the person has squatted down.
SQUAT_UP_ANGLE = 160
SQUAT_DOWN_ANGLE = 100

# The back may lean forward during a squat, but too much lean is marked as bad
# form. This is measured from a straight vertical line, in degrees.
MAX_TORSO_LEAN = 60

# (shoulder, hip, knee, ankle) landmark ids for each side of the body.
LEFT_SIDE = (11, 23, 25, 27)
RIGHT_SIDE = (12, 24, 26, 28)


def detect_squat(detector, img):
    """Return the squat form result and the angles used for display."""
    shoulder_id, hip_id, knee_id, ankle_id = pick_visible_side(
        detector, LEFT_SIDE, RIGHT_SIDE
    )

    # If any of these points are hidden, the angle values cannot be trusted.
    if not all(
        detector.isLandmarkVisible(point, VISIBILITY_THRESHOLD)
        for point in (shoulder_id, hip_id, knee_id, ankle_id)
    ):
        return None, (), None

    knee = get_reliable_angle(detector, img, (hip_id, knee_id, ankle_id))
    hip = get_reliable_angle(detector, img, (shoulder_id, hip_id, knee_id))
    if knee is None or hip is None:
        return None, (), None

    # Torso lean: angle of the shoulder-to-hip line compared with vertical.
    # 0 means perfectly upright and 90 means the back is flat.
    _, shoulder_x, shoulder_y = detector.lmList[shoulder_id]
    _, hip_x, hip_y = detector.lmList[hip_id]
    lean = math.degrees(
        math.atan2(abs(shoulder_x - hip_x), abs(shoulder_y - hip_y))
    )

    # Bad form is flagged whenever the back leans too far forward. Depth is
    # enforced by the rep counter: a rep only counts if the knee angle reaches
    # SQUAT_DOWN_ANGLE, so a shallow squat will not be counted.
    if lean > MAX_TORSO_LEAN:
        feedback = "FIX FORM"
    else:
        feedback = "GOOD FORM"

    angles = (
        (20, "Knee", knee),
        (60, "Hip", hip),
        (100, "Back lean", lean),
    )
    return feedback, angles, knee
