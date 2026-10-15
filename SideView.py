"""Check push-up form using the side camera view.

Side view is useful because it shows the body profile. This makes the elbow,
shoulder, and hip angles easier to interpret than in a front camera image.
"""

from BasicPoseModule import VISIBILITY_THRESHOLD, get_reliable_angle
from RepCounter import DOWN_ANGLE, UP_ANGLE


SIDE_HIP_ANGLE = 160
SIDE_SHOULDER_ANGLE = 40


def detect_side_view(detector, img):
    """Return the side-view form result and the angles used for display."""
    # These points are needed for a side check. If they are not visible enough, we
    # cannot trust the body angle values.
    required_points = (11, 13, 15, 23, 25)
    if not all(
        detector.isLandmarkVisible(point, VISIBILITY_THRESHOLD)
        for point in required_points
    ):
        return None, (), None

    # These angles help classify the arm position and the overall body alignment.
    elbow = get_reliable_angle(detector, img, (11, 13, 15))
    shoulder = get_reliable_angle(detector, img, (13, 11, 23))
    hip = get_reliable_angle(detector, img, (11, 23, 25))

    if not all(angle is not None for angle in (elbow, shoulder, hip)):
        return None, (), None

    # In the lower push-up position, the elbow is usually bent and the hip stays
    # open. In the top position, the arm is mostly straight while the shoulder and
    # hip still remain in a good alignment.
    if elbow <= DOWN_ANGLE and hip > SIDE_HIP_ANGLE:
        feedback = "GOOD FORM"
    elif (
        elbow >= UP_ANGLE
        and shoulder > SIDE_SHOULDER_ANGLE
        and hip > SIDE_HIP_ANGLE
    ):
        feedback = "GOOD FORM"
    else:
        feedback = "FIX FORM"

    angles = (
        (20, "Elbow", elbow),
        (60, "Shoulder", shoulder),
        (100, "Hip", hip),
    )
    return feedback, angles, elbow
