import cv2
import mediapipe as mp
import math

# MediaPipe gives each landmark a visibility score. If a body point is hidden or
# blurry, we should not trust it for angle detection.
VISIBILITY_THRESHOLD = 0.5


def get_reliable_angle(detector, img, points, use_depth=False):
    #Return a joint angle only when all points are visible enough to trust.
    # If a required landmark is missing, the angle is unreliable and can cause a
    # false form result. This check prevents that.
    if not all(
        detector.isLandmarkVisible(point, VISIBILITY_THRESHOLD)
        for point in points
    ):
        return None

    angle = detector.findAngle(
        img, points[0], points[1], points[2], use_depth=use_depth
    )
    return angle if math.isfinite(angle) else None


def pick_visible_side(detector, left_points, right_points):
    #Return whichever set of landmarks (left or right body side) is more visible.
    # In a side view only one side of the body faces the camera, so we use the
    # side the camera can see best instead of always assuming the left side.
    left_score = sum(detector.lmVisibility.get(p, 0.0) for p in left_points)
    right_score = sum(detector.lmVisibility.get(p, 0.0) for p in right_points)
    return left_points if left_score >= right_score else right_points


class poseDetector() :
    #Wrap MediaPipe pose detection and provide helper methods for measuring angles.
    
    def __init__(self, mode=False, complexity=1, smooth_landmarks=True,
                 enable_segmentation=False, smooth_segmentation=True,
                 detectionCon=0.5, trackCon=0.5):
        #Store pose settings and prepare the landmark tracking data.
        self.mode = mode 
        self.complexity = complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.lmList = []
        self.lmVisibility = {}
        self.lmDepth = {}
        self.results = None
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_landmarks,
                                     self.enable_segmentation, self.smooth_segmentation,
                                     self.detectionCon, self.trackCon)
        
        
    def findPose (self, img, draw=True):
        #Run MediaPipe pose detection and draw the skeleton if requested.
        # MediaPipe expects RGB images, so we convert the frame before processing.
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results and self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
                
        return img
    
    def findPosition(self, img, draw=True):
        """Save landmark coordinates and visibility values for later angle checks."""
        # self.lmList stores each landmark as [id, x, y]. This makes it easy to
        # access body points like the elbow or shoulder in later calculations.
        self.lmList = []
        self.lmVisibility = {}
        self.lmDepth = {}
        if self.results and self.results.pose_landmarks:
            h, w = img.shape[:2]
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                self.lmVisibility[id] = lm.visibility
                self.lmDepth[id] = lm.z * w
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return self.lmList

    def isLandmarkVisible(self, landmark_id, threshold=0.5):
        """Check whether a landmark is visible enough to trust."""
        return self.lmVisibility.get(landmark_id, 0.0) >= threshold

    def findAngle(self, img, p1, p2, p3, draw=True, use_depth=False):
        """Measure the angle at the middle joint using 2D or depth-aware 3D math."""
        # We use the three points as a bend: p1-p2-p3. Point p2 is the joint
        # being measured, such as the elbow or shoulder.
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        if use_depth:
            # For front-view checks, the z value is used to get a better estimate of
            # depth when the subject is angled away from the camera.
            z1, z2, z3 = (self.lmDepth.get(point_id) for point_id in (p1, p2, p3))
            if z1 is None or z2 is None or z3 is None:
                return float("nan")
            first = (x1 - x2, y1 - y2, z1 - z2)
            second = (x3 - x2, y3 - y2, z3 - z2)
            first_length = math.sqrt(sum(component ** 2 for component in first))
            second_length = math.sqrt(sum(component ** 2 for component in second))
            if first_length == 0 or second_length == 0:
                return float("nan")
            cosine = sum(a * b for a, b in zip(first, second)) / (
                first_length * second_length
            )
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        else:
            # In the simpler 2D version, we compare the direction of the two body
            # lines and convert the result to degrees.
            first_length = math.hypot(x1 - x2, y1 - y2)
            second_length = math.hypot(x3 - x2, y3 - y2)
            if first_length == 0 or second_length == 0:
                return float("nan")
            angle = math.degrees(
                math.atan2(y3-y2, x3-x2) -
                math.atan2(y1-y2, x1-x2)
            )

            if angle < 0:
                angle += 360

            if angle > 180:
                angle = 360 - angle

        return angle