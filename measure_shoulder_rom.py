"""
INSTRUCTIONS FOR MY IDIOT SELF TO RUN THIS: 
GO TERMINAL...

source ~/realsense_pose_venv312/bin/activate (virtual environment for this project because mediapipe is picky)
cd ~/projects/shoulder-rom-realsense
python measure_shoulder_rom.py

YOUREWELCOMEOK
"""



#this might only work on fedora i dunno :D

#necessary libraries
import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp #this gives our pretrained shoulder model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MATH STUFF ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
NORM: takes in a vector v, scaling it to have a length of one. gives us our unit vectors.

ANGLE_DEG: takes in two angles, normalizes them both, takes the dot product and uses that to give the angle  (from cosine :D)

SIGNED_ANGLE_DEG: gives a signed angle from a to b around an axis_unit (for IE/ER)
"""
def norm(v): 
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    return v / n

def angle_deg(a, b):
    a = norm(a)
    b = norm(b)
    if a is None or b is None:
        return None
    return math.degrees(math.acos(np.clip(float(np.dot(a, b)), -1.0, 1.0)))

def signed_angle_deg(axis_unit, a, b):
    axis_unit = norm(axis_unit)
    a = norm(a)
    b = norm(b)
    if axis_unit is None or a is None or b is None:
        return None
    return math.degrees(math.atan2(float(np.dot(axis_unit, np.cross(a, b))), float(np.dot(a, b))))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ MEDIAPIPE POSE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

#This is my specific path lol you'd replace it with yours
MODEL_PATH = "/home/cait/mp_models/pose_landmarker_lite.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)
pose = PoseLandmarker.create_from_options(options)

# gives indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,    R_ELBOW    = 13, 14
L_WRIST,    R_WRIST    = 15, 16
L_HIP,      R_HIP      = 23, 24

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SETTING UP THE CAMERA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

profile = pipeline.start(config)
align = rs.align(rs.stream.color)

intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# ----------------- measurement state -----------------
#because this is a janky prototype, i stole like all of this from someone's github thanks my goat 
flexion_max = 0.0
rot_min =  999.0   # ER extreme
rot_max = -999.0   # IR extreme

print("\nPress ESC to quit\n")

def deproject(depth_frame, px, py):
    d = float(depth_frame.get_distance(px, py))
    if d <= 0:
        return None
    # returns meters in camera coordinates: +x right, +y down, +z forward
    return np.array(rs.rs2_deproject_pixel_to_point(intr, [px, py], d), dtype=np.float32)

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            continue

        bgr = np.asanyarray(color.get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = bgr.shape

        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = pose.detect_for_video(mp_image, timestamp_ms)

        if not res.pose_landmarks:
            cv2.imshow("feed", bgr)
            if cv2.waitKey(1) == 27:
                break
            continue

        lm = res.pose_landmarks[0]

        def get3d(idx):
            x = float(lm[idx].x)
            y = float(lm[idx].y)
            px = int(x * w)
            py = int(y * h)
            if px < 0 or py < 0 or px >= w or py >= h:
                return None
            # simple depth deprojection at that pixel
            return deproject(depth, px, py)

        # Get torso + right arm points
        LS = get3d(L_SHOULDER)
        RS = get3d(R_SHOULDER)
        LH = get3d(L_HIP)
        RH = get3d(R_HIP)

        S  = get3d(R_SHOULDER)
        E  = get3d(R_ELBOW)
        Wp = get3d(R_WRIST)

        if any(p is None for p in [LS, RS, LH, RH, S, E, Wp]):
            cv2.imshow("feed", bgr)
            if cv2.waitKey(1) == 27:
                break
            continue

        # Torso frame:
        # x: left->right shoulder axis
        # y: hips->shoulders (up)
        # z: forward (approx)
        x_axis = norm(RS - LS)
        y_axis = norm(((LS + RS) * 0.5) - ((LH + RH) * 0.5))
        if x_axis is None or y_axis is None:
            continue
        z_axis = norm(np.cross(x_axis, y_axis))
        if z_axis is None:
            continue
        # re-orthogonalize y
        y_axis = norm(np.cross(z_axis, x_axis))
        if y_axis is None:
            continue

        hum = norm(E - S)     # humerus direction
        fore = norm(Wp - E)   # forearm direction
        if hum is None or fore is None:
            continue

        hum_sag = hum - float(np.dot(hum, x_axis)) * x_axis
        hum_sag = norm(hum_sag)
        if hum_sag is not None:
            flex = angle_deg(hum_sag, -y_axis)
            if flex is not None:
                flexion_max = max(flexion_max, flex)

        # Gate: elbow around 90 deg, abduction around 90 deg
        elbow = angle_deg(S - E, Wp - E)   # angle at elbow
        abd   = angle_deg(hum, y_axis)     # humerus vs "up" (rough abduction proxy)

        if elbow is not None and abd is not None and (70.0 < elbow < 110.0) and (60.0 < abd < 120.0):
            fore_perp = fore - float(np.dot(fore, hum)) * hum
            fore_perp = norm(fore_perp)

            ref = z_axis - float(np.dot(z_axis, hum)) * hum
            ref = norm(ref)

            if fore_perp is not None and ref is not None:
                # Signed rotation: ref -> fore_perp around humerus axis
                rot = signed_angle_deg(hum, ref, fore_perp)
                if rot is not None:
                    rot_min = min(rot_min, rot)
                    rot_max = max(rot_max, rot)

        cv2.putText(bgr, f"Flex max: {flexion_max:.1f} deg", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if rot_min < 900 and rot_max > -900:
            cv2.putText(bgr, f"IR/ER (min/max): {rot_min:.1f} / {rot_max:.1f} deg", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(bgr, f"Total arc: {(rot_max-rot_min):.1f} deg", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(bgr, "IR/ER: (hold ~90/90 position to measure)", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("feed", bgr)
        if cv2.waitKey(1) == 27:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

print("\nFINAL RESULTS")
print(f"Flexion max: {flexion_max:.1f} deg")
if rot_min < 900 and rot_max > -900:
    print(f"IR/ER min/max: {rot_min:.1f} / {rot_max:.1f} deg")
    print(f"Total arc: {(rot_max-rot_min):.1f} deg")
else:
    print("IR/ER: not captured (didn't hold ~90/90 position long enough)")
