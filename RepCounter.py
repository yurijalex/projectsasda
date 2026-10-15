import math

# These angle values define the push-up position used by the counter.
# A larger value means the arm is more straightened, while a smaller value means
# the elbow is more bent.
UP_ANGLE = 160
DOWN_ANGLE = 90


class RepCounter:
    """Track one exercise cycle (high angle -> low angle -> high angle).

    Push-ups, squats and sit-ups all follow the same pattern, so the same state
    machine is reused. Only the two angle limits change per exercise.
    """
    def __init__(self, up_angle=UP_ANGLE, down_angle=DOWN_ANGLE):
        # "up" is the extended/starting position (high angle) and "down" is the
        # most bent position (low angle) for the chosen exercise.
        self.up_angle = up_angle
        self.down_angle = down_angle
        # The program moves through phases to detect one complete repetition:
        # waiting_up -> up -> down -> up -> ...
        # This lets the program ignore random small movements and count only a full
        # push-up motion.
        self.phase = "waiting_up"
        self.reps = 0

    def update(self, angle):
        """Update the rep count based on the main joint angle."""
        if angle is None or not math.isfinite(angle):
            return self.reps

        # Example: if the arm starts up, the program waits until it goes down,
        # then waits until it comes back up again before counting a repetition.
        if self.phase == "waiting_up":
            if angle >= self.up_angle:
                self.phase = "up"
        elif self.phase == "up":
            if angle <= self.down_angle:
                self.phase = "down"
        elif self.phase == "down" and angle >= self.up_angle:
            self.reps += 1
            self.phase = "up"

        return self.reps
