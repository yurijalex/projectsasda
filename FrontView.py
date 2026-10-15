"""Check push-up form using the front camera view.

Front view is useful because both arms can be compared at the same time. If the
left and right elbows do not match, the body may be twisting or one side may be
out of position.
"""

from BasicPoseModule import get_reliable_angle
from RepCounter import DOWN_ANGLE, UP_ANGLE


MAX_FRONT_ANGLE_DIFFERENCE = 45


def detect_front_view(detector, img):
    """Return a form result based on the user's left and right elbow angles."""
    # The elbow joints are measured with depth awareness because the person is
    # facing the camera and the arm positions may not be perfectly flat in 2D.
    left_elbow = get_reliable_angle(
        detector, img, (11, 13, 15), use_depth=True
    )
    right_elbow = get_reliable_angle(
        detector, img, (12, 14, 16), use_depth=True
    )
    arm_angles = [
        angle for angle in (left_elbow, right_elbow)
        if angle is not None
    ]

    feedback = None
    if len(arm_angles) == 2:
        # If both elbows are visible and close to each other, we average them for
        # one main angle. This makes the system less sensitive to small side-to-side
        # differences.
        if abs(left_elbow - right_elbow) <= MAX_FRONT_ANGLE_DIFFERENCE:
            elbow = (left_elbow + right_elbow) / 2
        else:
            elbow = None
            feedback = "FIX FORM"
    elif arm_angles:
        elbow = arm_angles[0]
    else:
        elbow = None

    if elbow is not None:
        angles = []
        if left_elbow is not None:
            angles.append((20, "Left elbow", left_elbow))
        if right_elbow is not None:
            angles.append((60, "Right elbow", right_elbow))
        label = "Mean" if len(arm_angles) == 2 else "Single arm"
        angles.append((100, label, elbow))

        # A good push-up usually means both elbows are either near the low-angle
        # position or the high-angle position together. If one arm is different, the
        # form is marked as bad.
        if len(arm_angles) == 2 and (
            (left_elbow <= DOWN_ANGLE and right_elbow <= DOWN_ANGLE)
            or (left_elbow >= UP_ANGLE and right_elbow >= UP_ANGLE)
        ):
            feedback = "GOOD FORM"
        elif len(arm_angles) == 1 and (
            elbow <= DOWN_ANGLE or elbow >= UP_ANGLE
        ):
            feedback = "GOOD FORM"
        else:
            feedback = "FIX FORM"

        return feedback, tuple(angles), elbow

    return feedback, (), None
