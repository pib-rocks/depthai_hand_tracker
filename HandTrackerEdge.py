import numpy as np
from collections import namedtuple
import threading
from numpy.lib.arraysetops import isin
import mediapipe_utils as mpu
import depthai as dai
import cv2
from pathlib import Path
from FPS import FPS, now
import time
import sys
from string import Template
import marshal
import math
import json
from tinkerforge.ip_connection import IPConnection
from tinkerforge.brick_hat import BrickHAT
from tinkerforge.bricklet_servo_v2 import BrickletServoV2
from tinkerforge.bricklet_solid_state_relay_v2 import BrickletSolidStateRelayV2
from datetime import datetime
from tinkerforge.bricklet_base import Bricklet

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# --- Servo Configuration Constants ---
HOST = "localhost"
PORT = 4223
UID_SERVO_BRICK_1 = '29Fy' # Right Arm Joints (Elbow, Lower Arm, Shoulder Horizontal)
UID_SERVO_BRICK_2 = '29F5' # Shoulder Vertical Joints (Left, Right)
UID_SERVO_BRICK_3 = '29F3' # Left Arm Joints (Elbow, Lower Arm, Shoulder Horizontal) + Left Fingers

# Servo Indices - Based on observed usage
# Brick 1 (Right Arm)
SERVO_IDX_ELBOW_R = 8
SERVO_IDX_LOWER_ARM_R = 7
SERVO_IDX_SHOULDER_HORIZONTAL_R = 9
# Brick 2 (Shoulder Vertical)
SERVO_IDX_SHOULDER_VERTICAL_L = 9
SERVO_IDX_SHOULDER_VERTICAL_R = 1
# Brick 3 (Left Arm + Fingers)
SERVO_IDX_SHOULDER_HORIZONTAL_L = 9 # Shoulder Horizontal Left
SERVO_IDX_THUMB_L_OPPOSITION = 0
SERVO_IDX_THUMB_L_PROXIMAL = 1
SERVO_IDX_INDEX_L_PROXIMAL = 2
SERVO_IDX_MIDDLE_L_PROXIMAL = 3
SERVO_IDX_RING_L_PROXIMAL = 4
SERVO_IDX_PINKY_L_PROXIMAL = 5
SERVO_IDX_ELBOW_L = 8
SERVO_IDX_LOWER_ARM_L = 7

# Default Motion Parameters
DEFAULT_VELOCITY = 9000
DEFAULT_ACCELERATION = 9000
DEFAULT_DECELERATION = 9000
DEFAULT_PULSE_MIN = 700
DEFAULT_PULSE_MAX = 2500

# --- End Servo Configuration Constants ---

SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
LANDMARK_MODEL_SPARSE = str(SCRIPT_DIR / "models/hand_landmark_sparse_sh4.blob")
DETECTION_POSTPROCESSING_MODEL = str(SCRIPT_DIR / "custom_models/PDPostProcessing_top2_sh1.blob")
TEMPLATE_MANAGER_SCRIPT_SOLO = str(SCRIPT_DIR / "template_manager_script_solo.py")
TEMPLATE_MANAGER_SCRIPT_DUO = str(SCRIPT_DIR / "template_manager_script_duo.py")

ipcon = IPConnection()
ipcon.connect(HOST, PORT)
# Use constants for UIDs
servoBrick1 = BrickletServoV2(UID_SERVO_BRICK_1, ipcon) # Right Arm
servoBrick2 = BrickletServoV2(UID_SERVO_BRICK_2, ipcon) # Shoulder Vertical
servoBrick3 = BrickletServoV2(UID_SERVO_BRICK_3, ipcon) # Left Arm + Fingers

green_cube_center = 0.0

# --- Servo Helper Function ---

def set_servo(servo_brick: BrickletServoV2, servo_index: int, position: int,
              enable: bool = True,
              pulse_min: int = DEFAULT_PULSE_MIN, pulse_max: int = DEFAULT_PULSE_MAX,
              velocity: int = DEFAULT_VELOCITY, acceleration: int = DEFAULT_ACCELERATION,
              deceleration: int = DEFAULT_DECELERATION):
    """Configures motion profile, pulse width, sets target position, and enables a specific servo."""
    try:
        # Configure first
        servo_brick.set_pulse_width(servo_index, pulse_min, pulse_max)
        servo_brick.set_motion_configuration(servo_index, velocity, acceleration, deceleration)
        # Then set position
        servo_brick.set_position(servo_index, position)
        # Finally enable
        if enable:
            servo_brick.set_enable(servo_index, True)
    except Exception as e:
        print(f"Error setting/configuring servo {servo_index} on brick {servo_brick.uid}: {e}")

# --- End Servo Helper Function ---


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()

def to_radians(servo_value):
    # Convert from thousandths of a degree to radians.
    return (servo_value / 100) * (math.pi / 180.0)
    
def get_left_observation():
    obs = []
    try:
        # Use constants for servo indices
        obs.append(to_radians(servoBrick2.get_position(SERVO_IDX_SHOULDER_VERTICAL_L)))
        obs.append(to_radians(servoBrick3.get_position(SERVO_IDX_SHOULDER_HORIZONTAL_L))) # shoulder_horizontal_left
        obs.append(to_radians(servoBrick3.get_position(SERVO_IDX_THUMB_L_OPPOSITION)))
        obs.append(to_radians(servoBrick3.get_position(SERVO_IDX_THUMB_L_PROXIMAL)))
        obs.append(to_radians(servoBrick3.get_position(SERVO_IDX_INDEX_L_PROXIMAL)))
        obs.append(to_radians(servoBrick3.get_position(SERVO_IDX_MIDDLE_L_PROXIMAL)))
        obs.append(to_radians(servoBrick3.get_position(SERVO_IDX_RING_L_PROXIMAL)))
        obs.append(to_radians(servoBrick3.get_position(SERVO_IDX_PINKY_L_PROXIMAL)))
    except Exception as e: # Catch specific exceptions if possible
        print(f"Observation error occurred: {e}")
    return obs

def get_left_action():
    if left_target is None:
        return None

    acts = []
    try:
        # Map action names to servo indices (assuming the order matches get_left_observation)
        # Note: Elbow, forearm, wrist are not included here as per the comment
        acts.append(to_radians(left_target["shoulder_vertical"]))   # Corresponds to SERVO_IDX_SHOULDER_VERTICAL_L
        acts.append(to_radians(left_target["shoulder_horizontal"])) # Corresponds to SERVO_IDX_SHOULDER_HORIZONTAL_L (Renamed key)
        acts.append(to_radians(left_target["thumb_opposition"]))    # Corresponds to SERVO_IDX_THUMB_L_OPPOSITION
        acts.append(to_radians(left_target["thumb_proximal"]))      # Corresponds to SERVO_IDX_THUMB_L_PROXIMAL
        acts.append(to_radians(left_target["index"]))               # Corresponds to SERVO_IDX_INDEX_L_PROXIMAL
        acts.append(to_radians(left_target["middle"]))            # Corresponds to SERVO_IDX_MIDDLE_L_PROXIMAL
        acts.append(to_radians(left_target["ring"]))              # Corresponds to SERVO_IDX_RING_L_PROXIMAL
        acts.append(to_radians(left_target["pinky"]))             # Corresponds to SERVO_IDX_PINKY_L_PROXIMAL
    except Exception as e: # Catch specific exceptions if possible
        print(f"Acts exception occurred: {e}")

    return acts

def get_green_cube_height():
    if green_cube_center is None:
        return -1.0
    
    return green_cube_center

def detect_color_area(frame, lower_hsv, upper_hsv, draw_color, label):
    """
    Detects the largest area of a specified color in the frame, draws the contour,
    marks the center, and labels it.

    Args:
        frame (np.ndarray): Input image in BGR.
        lower_hsv (np.ndarray): Lower HSV bound.
        upper_hsv (np.ndarray): Upper HSV bound.
        draw_color (tuple): BGR color for drawing.
        label (str): Text label for the detected color.

    Returns:
        tuple or None: The (x, y) coordinates of the detected color's center, or None if not found.
    """
    # Convert frame to HSV color space.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the color.
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Clean the mask with morphological operations.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    center = None
    if contours:
        # Assume the largest contour corresponds to the area of interest.
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
            # Draw the contour and center point.
            cv2.drawContours(frame, [largest_contour], -1, draw_color, 2)
            cv2.circle(frame, center, 5, draw_color, -1)
            cv2.putText(frame, label, (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
    return center

def detect_colors(frame):
    """
    Processes the frame for pink, green, and orange colors.

    Returns:
        dict: A dictionary with keys 'pink', 'green', 'orange' and values being the center coordinates.
    """
    # Define HSV ranges for each color.
    # Green range.
    green_lower = np.array([36, 25, 25])
    green_upper = np.array([70, 255, 255])
    # For drawing, we choose colors that visually match the target (BGR format).
    center = detect_color_area(frame, green_lower, green_upper, (0, 255, 0), "Green")

    return center if center is not None else (-1,-1)

trajectory = {
    "obs": [],
    "acts": [],
    "infos": None,
    "terminal": True
}

recording = True
def record_trajectory():
    global recording
    while recording:
        obs = get_left_observation()
        acts = get_left_action()
        if acts is None:
            print("acts are None")
            continue

        cube_height = get_green_cube_height()
        obs_text=str(obs)
        acts_text=str(acts)
        obs.append(cube_height)
        if len(obs_text) > 0 and len(acts_text) > 0:
            trajectory["obs"].append(obs)
            trajectory["acts"].append(acts)
        else:
            print("length of obs or acts 0")
        time.sleep(0.050)  #50 ms interval

class HandTracker:
    """
    Mediapipe Hand Tracker for depthai
    Arguments:
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host (Edge mode only),
                    - a file path of an image or a video,
                    - an integer (eg 0) for a webcam id,
                    In edge mode, only "rgb" and "rgb_laconic" are possible
    - pd_model: palm detection model blob file,
    - pd_score: confidence score to determine whether a detection is reliable (a float between 0 and 1).
    - pd_nms_thresh: NMS threshold.
    - use_lm: boolean. When True, run landmark model. Otherwise, only palm detection model is run
    - lm_model: landmark model. Either:
                    - 'full' for LANDMARK_MODEL_FULL,
                    - 'lite' for LANDMARK_MODEL_LITE,
                    - 'sparse' for LANDMARK_MODEL_SPARSE,
                    - a path of a blob file.  
    - lm_score_thresh : confidence score to determine whether landmarks prediction is reliable (a float between 0 and 1).
    - use_world_landmarks: boolean. The landmarks model yields 2 types of 3D coordinates : 
                    - coordinates expressed in pixels in the image, always stored in hand.landmarks,
                    - coordinates expressed in meters in the world, stored in hand.world_landmarks 
                    only if use_world_landmarks is True.
    - pp_model: path to the detection post processing model,
    - solo: boolean, when True detect one hand max (much faster since we run the pose detection model only if no hand was detected in the previous frame)
                    On edge mode, always True
    - xyz : boolean, when True calculate the (x, y, z) coords of the detected palms.
    - crop : boolean which indicates if square cropping on source images is applied or not
    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - resolution : sensor resolution "full" (1920x1080) or "ultra" (3840x2160),
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
                    The width is calculated accordingly to height and depends on value of 'crop'
    - use_gesture : boolean, when True, recognize hand poses froma predefined set of poses
                    (ONE, TWO, THREE, FOUR, FIVE, OK, PEACE, FIST)
    - use_handedness_average : boolean, when True the handedness is the average of the last collected handednesses.
                    This brings robustness since the inferred robustness is not reliable on ambiguous hand poses.
                    When False, handedness is the last inferred handedness.
    - single_hand_tolerance_thresh (Duo mode only) : In Duo mode, if there is only one hand in a frame, 
                    in order to know when a second hand will appear you need to run the palm detection 
                    in the following frames. Because palm detection is slow, you may want to delay 
                    the next time you will run it. 'single_hand_tolerance_thresh' is the number of 
                    frames during only one hand is detected before palm detection is run again.   
    - lm_nb_threads : 1 or 2 (default=2), number of inference threads for the landmark model
    - use_same_image (Edge Duo mode only) : boolean, when True, use the same image when inferring the landmarks of the 2 hands
                    (setReusePreviousImage(True) in the ImageManip node before the landmark model). 
                    When True, the FPS is significantly higher but the skeleton may appear shifted on one of the 2 hands.
    - stats : boolean, when True, display some statistics when exiting.   
    - trace : int, 0 = no trace, otherwise print some debug messages or show output of ImageManip nodes
            if trace & 1, print application level info like number of palm detections,
            if trace & 2, print lower level info like when a message is sent or received by the manager script node,
            if trace & 4, show in cv2 windows outputs of ImageManip node,
            if trace & 8, save in file tmp_code.py the python code of the manager script node
            Ex: if trace==3, both application and low level info are displayed.
                      
    """
    def __init__(self, input_src=None,
                pd_model=PALM_DETECTION_MODEL, 
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                use_lm=True,
                lm_model="full",
                lm_score_thresh=0.5,
                use_world_landmarks=False,
                pp_model = DETECTION_POSTPROCESSING_MODEL,
                solo=True,
                xyz=False,
                crop=False,
                internal_fps=None,
                resolution="full",
                internal_frame_height=640,
                use_gesture=False,
                use_handedness_average=True,
                single_hand_tolerance_thresh=10,
                use_same_image=True,
                lm_nb_threads=2,
                stats=False,
                trace=0,
                angle_array_final=[]
                ):

        self.is_initialized = False
        self.use_lm = use_lm
        if not use_lm:
            print("use_lm=False is not supported in Edge mode.")
            sys.exit()
        self.pd_model = pd_model
        print(f"Palm detection blob     : {self.pd_model}")
        if lm_model == "full":
            self.lm_model = LANDMARK_MODEL_FULL
        elif lm_model == "lite":
            self.lm_model = LANDMARK_MODEL_LITE
        elif lm_model == "sparse":
                self.lm_model = LANDMARK_MODEL_SPARSE
        else:
            self.lm_model = lm_model
        print(f"Landmark blob           : {self.lm_model}")
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.lm_score_thresh = lm_score_thresh
        self.pp_model = pp_model
        print(f"PD post processing blob : {self.pp_model}")
        self.solo = solo
        if self.solo:
            print("In Solo mode, # of landmark model threads is forced to 1")
            self.lm_nb_threads = 1
        else:
            assert lm_nb_threads in [1, 2]
            self.lm_nb_threads = lm_nb_threads
        self.xyz = False
        self.crop = crop 
        self.use_world_landmarks = use_world_landmarks
           
        self.stats = stats
        self.trace = trace
        self.use_gesture = use_gesture
        self.use_handedness_average = use_handedness_average
        self.single_hand_tolerance_thresh = single_hand_tolerance_thresh
        self.use_same_image = use_same_image

        self.device = dai.Device()

        if input_src == None or input_src == "rgb" or input_src == "rgb_laconic":
            # Note that here (in Host mode), specifying "rgb_laconic" has no effect
            # Color camera frames are systematically transferred to the host
            self.input_type = "rgb" # OAK* internal color camera
            self.laconic = input_src == "rgb_laconic" # Camera frames are not sent to the host
            if resolution == "full":
                self.resolution = (1920, 1080)
            elif resolution == "ultra":
                self.resolution = (3840, 2160)
            else:
                print(f"Error: {resolution} is not a valid resolution !")
                sys.exit()
            print("Sensor resolution:", self.resolution)

            if xyz:
                # Check if the device supports stereo
                cameras = self.device.getConnectedCameras()
                if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras:
                    self.xyz = True
                else:
                    print("Warning: depth unavailable on this device, 'xyz' argument is ignored")

            if internal_fps is None:
                if lm_model == "full":
                    if self.xyz:
                        self.internal_fps = 22 
                    else:
                        self.internal_fps = 26 
                elif lm_model == "lite":
                    if self.xyz:
                        self.internal_fps = 29 
                    else:
                        self.internal_fps = 36 
                elif lm_model == "sparse":
                    if self.xyz:
                        self.internal_fps = 24 
                    else:
                        self.internal_fps = 29
                else:
                    self.internal_fps = 39
            else:
                self.internal_fps = internal_fps 
            print(f"Internal camera FPS set to: {self.internal_fps}") 

            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps

            if self.crop:
                self.frame_size, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height, self.resolution)
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
                self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2
            else:
                width, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height * self.resolution[0] / self.resolution[1], self.resolution, is_height=False)
                self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w
                self.crop_w = 0
        
            print(f"Internal camera image size: {self.img_w} x {self.img_h} - pad_h: {self.pad_h}")

        else:
            print("Invalid input source:", input_src)
            sys.exit()
        
        # Define and start pipeline
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # Define data queues 
        if not self.laconic:
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        self.q_manager_out = self.device.getOutputQueue(name="manager_out", maxSize=1, blocking=False)
        # For showing outputs of ImageManip nodes (debugging)
        if self.trace & 4:
            self.q_pre_pd_manip_out = self.device.getOutputQueue(name="pre_pd_manip_out", maxSize=1, blocking=False)
            self.q_pre_lm_manip_out = self.device.getOutputQueue(name="pre_lm_manip_out", maxSize=1, blocking=False)    

        self.fps = FPS()

        self.nb_frames_pd_inference = 0
        self.nb_frames_lm_inference = 0
        self.nb_lm_inferences = 0
        self.nb_failed_lm_inferences = 0
        self.nb_frames_lm_inference_after_landmarks_ROI = 0
        self.nb_frames_no_hand = 0
        

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_4)
        self.pd_input_length = 128

        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.createColorCamera()
        if self.resolution[0] == 1920:
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        else:
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setInterleaved(False)
        cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        cam.setFps(self.internal_fps)

        if self.crop:
            cam.setVideoSize(self.frame_size, self.frame_size)
            cam.setPreviewSize(self.frame_size, self.frame_size)
        else: 
            cam.setVideoSize(self.img_w, self.img_h)
            cam.setPreviewSize(self.img_w, self.img_h)

        if not self.laconic:
            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            cam_out.input.setQueueSize(1)
            cam_out.input.setBlocking(False)
            cam.video.link(cam_out.input)

        # Define manager script node
        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(self.build_manager_script())

        if self.xyz:
            print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes...")
            # For now, RGB needs fixed focus to properly align with depth.
            # The value used during calibration should be used here
            calib_data = self.device.readCalibration()
            calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.RGB)
            print(f"RGB calibration lens position: {calib_lens_pos}")
            cam.initialControl.setManualFocus(calib_lens_pos)

            mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
            left = pipeline.createMonoCamera()
            left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            left.setResolution(mono_resolution)
            left.setFps(self.internal_fps)

            right = pipeline.createMonoCamera()
            right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            right.setResolution(mono_resolution)
            right.setFps(self.internal_fps)

            stereo = pipeline.createStereoDepth()
            stereo.setConfidenceThreshold(230)
            # LR-check is required for depth alignment
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            stereo.setSubpixel(False)  # subpixel True brings latency
            # MEDIAN_OFF necessary in depthai 2.7.2. 
            # Otherwise : [critical] Fatal error. Please report to developers. Log: 'StereoSipp' '533'
            # stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)

            spatial_location_calculator = pipeline.createSpatialLocationCalculator()
            spatial_location_calculator.setWaitForConfigInput(True)
            spatial_location_calculator.inputDepth.setBlocking(False)
            spatial_location_calculator.inputDepth.setQueueSize(1)

            left.out.link(stereo.left)
            right.out.link(stereo.right)    

            stereo.depth.link(spatial_location_calculator.inputDepth)

            manager_script.outputs['spatial_location_config'].link(spatial_location_calculator.inputConfig)
            spatial_location_calculator.out.link(manager_script.inputs['spatial_data'])

        # Define palm detection pre processing: resize preview to (self.pd_input_length, self.pd_input_length)
        print("Creating Palm Detection pre processing image manip...")
        pre_pd_manip = pipeline.create(dai.node.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_pd_manip.inputImage)
        manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)

        # For debugging
        if self.trace & 4:
            pre_pd_manip_out = pipeline.createXLinkOut()
            pre_pd_manip_out.setStreamName("pre_pd_manip_out")
            pre_pd_manip.out.link(pre_pd_manip_out.input)

        # Define palm detection model
        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(self.pd_model)
        pre_pd_manip.out.link(pd_nn.input)

        # Define pose detection post processing "model"
        print("Creating Palm Detection post processing Neural Network...")
        post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
        post_pd_nn.setBlobPath(self.pp_model)
        pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs['from_post_pd_nn'])
        
        # Define link to send result to host 
        manager_out = pipeline.create(dai.node.XLinkOut)
        manager_out.setStreamName("manager_out")
        manager_script.outputs['host'].link(manager_out.input)

        # Define landmark pre processing image manip
        print("Creating Hand Landmark pre processing image manip...") 
        self.lm_input_length = 224
        pre_lm_manip = pipeline.create(dai.node.ImageManip)
        pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length*self.lm_input_length*3)
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_lm_manip.inputImage)

        # For debugging
        if self.trace & 4:
            pre_lm_manip_out = pipeline.createXLinkOut()
            pre_lm_manip_out.setStreamName("pre_lm_manip_out")
            pre_lm_manip.out.link(pre_lm_manip_out.input)

        manager_script.outputs['pre_lm_manip_cfg'].link(pre_lm_manip.inputConfig)

        # Define landmark model
        print(f"Creating Hand Landmark Neural Network ({'1 thread' if self.lm_nb_threads == 1 else '2 threads'})...")          
        lm_nn = pipeline.create(dai.node.NeuralNetwork)
        lm_nn.setBlobPath(self.lm_model)
        lm_nn.setNumInferenceThreads(self.lm_nb_threads)
        pre_lm_manip.out.link(lm_nn.input)
        lm_nn.out.link(manager_script.inputs['from_lm_nn'])
            
        print("Pipeline created.")
        return pipeline        
    
    def build_manager_script(self):
        '''
        The code of the scripting node 'manager_script' depends on :
            - the score threshold,
            - the video frame shape
        So we build this code from the content of the file template_manager_script_*.py which is a python template
        '''
        # Read the template
        with open(TEMPLATE_MANAGER_SCRIPT_SOLO if self.solo else TEMPLATE_MANAGER_SCRIPT_DUO, 'r') as file:
            template = Template(file.read())
        
        # Perform the substitution
        code = template.substitute(
                    _TRACE1 = "node.warn" if self.trace & 1 else "#",
                    _TRACE2 = "node.warn" if self.trace & 2 else "#",
                    _pd_score_thresh = self.pd_score_thresh,
                    _lm_score_thresh = self.lm_score_thresh,
                    _pad_h = self.pad_h,
                    _img_h = self.img_h,
                    _img_w = self.img_w,
                    _frame_size = self.frame_size,
                    _crop_w = self.crop_w,
                    _IF_XYZ = "" if self.xyz else '"""',
                    _IF_USE_HANDEDNESS_AVERAGE = "" if self.use_handedness_average else '"""',
                    _single_hand_tolerance_thresh= self.single_hand_tolerance_thresh,
                    _IF_USE_SAME_IMAGE = "" if self.use_same_image else '"""',
                    _IF_USE_WORLD_LANDMARKS = "" if self.use_world_landmarks else '"""',
        )
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'\n\s*\n', '\n', code) # Use raw string for regex
        # For debugging
        if self.trace & 8:
            with open("tmp_code.py", "w") as file:
                file.write(code)

        return code

    def extract_hand_data(self, res, hand_idx):
        hand = mpu.HandRegion()
        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * self.frame_size
        hand.rotation = res["rotation"][hand_idx] 
        hand.rect_points = mpu.rotated_rect_to_points(hand.rect_x_center_a, hand.rect_y_center_a, hand.rect_w_a, hand.rect_h_a, hand.rotation)
        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res['rrn_lms'][hand_idx]).reshape(-1,3)
        hand.landmarks = (np.array(res["sqn_lms"][hand_idx]) * self.frame_size).reshape(-1,2).astype(np.int32)
    
        #print(len(res.get("lm_score",[])))
        if self.xyz:
            hand.xyz = np.array(res["xyz"][hand_idx])
            hand.xyz_zone = res["xyz_zone"][hand_idx]
        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
        if self.pad_h > 0:
            hand.landmarks[:,1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            hand.landmarks[:,0] -= self.pad_w
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][0] -= self.pad_w

        # World landmarks
        if self.use_world_landmarks:
            hand.world_landmarks = np.array(res["world_lms"][hand_idx]).reshape(-1, 3)

        if self.use_gesture: mpu.recognize_gesture(hand)

        return hand

    def next_frame(self):
        global green_cube_center
        self.fps.update()

        if self.laconic:
            video_frame = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()
            green_cube_center = detect_colors(video_frame)[1]

        #video_frame = cv2.flip(video_frame,1)       

        # For debugging
        if self.trace & 4:
            pre_pd_manip = self.q_pre_pd_manip_out.tryGet()
            if pre_pd_manip:
                pre_pd_manip = pre_pd_manip.getCvFrame()
                cv2.imshow("pre_pd_manip", pre_pd_manip)
            pre_lm_manip = self.q_pre_lm_manip_out.tryGet()
            if pre_lm_manip:
                pre_lm_manip = pre_lm_manip.getCvFrame()
                cv2.imshow("pre_lm_manip", pre_lm_manip)

        # Get result from device
        res = marshal.loads(self.q_manager_out.get().getData())
        hands = []
        norm_landmarks = []
        orientation = 0
        
        index_hand_right = -1
        index_hand_left  = -1
        shoulder_horizontal_left = 0
        shoulder_horizontal_right = 0 # Renamed from horizontal_right
        
        
        if len(res.get("lm_score",[])) == 0: # No hands detected, move to default/initial position
            # Set positions using the combined helper function (config uses defaults)
            # Shoulder Vertical Left
            set_servo(servoBrick2, SERVO_IDX_SHOULDER_VERTICAL_L, 0)
            # Shoulder Vertical Right
            set_servo(servoBrick2, SERVO_IDX_SHOULDER_VERTICAL_R, -9000)
            # Elbow Left
            set_servo(servoBrick3, SERVO_IDX_ELBOW_L, -3500)
            # Elbow Right
            set_servo(servoBrick1, SERVO_IDX_ELBOW_R, 5000)
            # Lower Arm Rotation Left
            set_servo(servoBrick3, SERVO_IDX_LOWER_ARM_L, 0)
            # Lower Arm Rotation Right
            set_servo(servoBrick1, SERVO_IDX_LOWER_ARM_R, 0)
            # Shoulder Horizontal Left
            set_servo(servoBrick3, SERVO_IDX_SHOULDER_HORIZONTAL_L, 6000)
            # Shoulder Horizontal Right
            # Note: shoulder_horizontal_right is 0 here as it's calculated later based on hand landmarks
            set_servo(servoBrick1, SERVO_IDX_SHOULDER_HORIZONTAL_R, 0) # Set to 0 initially

        for i in range(len(res.get("lm_score",[]))):
            hand = self.extract_hand_data(res, i)
            # Calculate shoulder_horizontal_left based on left hand if present
            if hand.label == "left":
                 shoulder_horizontal_left = hand.landmarks[0][0] * 13 - 10000 # Keep calculation for now
            if hand.label=="left":
                index_hand_left = i
                shoulder_horizontal_left = hand.landmarks[0][0] * 13 - 10000
                shoulder_vertical_left = 4000 - hand.landmarks[0][1] * 13 - 2000
            if hand.label=="right":
                index_hand_right = i
                shoulder_horizontal_right = hand.landmarks[0][0] * 13 - 4000 # Renamed from horizontal_right
                shoulder_vertical_right = hand.landmarks[0][1] * 13 + 1500
            hands.append(hand)
            
            norm_landmarks.append(hand.norm_landmarks)      #ziehe normierte landmark koordinaten aus hand; in Schleife appenden um Anzahl der Haende zu beruecksichtigen
            orientation = hand.handedness
       

        angle_array = []
        if norm_landmarks:          #nur rechnen wenn Handkoordinaten verfuegbar
            #initialisiere Matritzen fuer Winkel
            vec_thumb_low = []
            vec_thumb_high = []
            vec_thumb_low2 = []
            vec_thumb_high2 = []
            angle_thumb = []
            angle_thumb2 = []
            vec_idx_low = []
            vec_idx_high = []
            angle_idx = []
            vec_mid_low = []
            vec_mid_high = []
            angle_mid = []
            vec_rng_low = []
            vec_rng_high = []
            angle_rng = []
            vec_ltl_low = []
            vec_ltl_high = []
            angle_ltl = []

            #beruecksichtige Anzahl der Haende; Nummerierung der Landmarks stimmt mit Zeichnung aus Github ueberein; erste Dimension der Matrix norm_landmarks ist index der Hand (0 wenn keine Hand, 1 wenn eine Hand, 2 wenn 2 Haende, ...)
            for i in range(0,len(norm_landmarks)):
                #Daumen Schliesswinkel
                vec_thumb_low.append(norm_landmarks[i][1][:] - norm_landmarks[i][2][:]) #Berechne oberen Vektor
                vec_thumb_high.append(norm_landmarks[i][4][:] - norm_landmarks[i][3][:])#Berechne unteren Vektor
                angle_thumb.append(math.acos(np.dot(vec_thumb_high[i], vec_thumb_low[i])/(np.linalg.norm(vec_thumb_low[i])*np.linalg.norm(vec_thumb_high[i])))) #Berechne Winkel in rad
                #Daumenwinkel in Handebene
                vec_thumb_low2.append(norm_landmarks[i][3][:] - norm_landmarks[i][2][:])
                vec_thumb_high2.append(norm_landmarks[i][9][:] - norm_landmarks[i][0][:])
                angle_thumb2.append(math.acos(np.dot(vec_thumb_high2[i], vec_thumb_low2[i])/(np.linalg.norm(vec_thumb_low2[i])*np.linalg.norm(vec_thumb_high2[i]))))

                #Zeigefinger                
                vec_idx_low.append(norm_landmarks[i][5][:] - norm_landmarks[i][6][:])
                vec_idx_high.append(norm_landmarks[i][8][:] - norm_landmarks[i][7][:])
                angle_idx.append(math.acos(np.dot(vec_idx_high[i], vec_idx_low[i])/(np.linalg.norm(vec_idx_low[i])*np.linalg.norm(vec_idx_high[i]))))


                #Mittelfinger                
                vec_mid_low.append(norm_landmarks[i][9][:] - norm_landmarks[i][10][:])
                vec_mid_high.append(norm_landmarks[i][12][:] - norm_landmarks[i][11][:])
                angle_mid.append(math.acos(np.dot(vec_mid_high[i], vec_mid_low[i])/(np.linalg.norm(vec_mid_low[i])*np.linalg.norm(vec_mid_high[i]))))


                #Ringfinger                
                vec_rng_low.append(norm_landmarks[i][13][:] - norm_landmarks[i][14][:])
                vec_rng_high.append(norm_landmarks[i][16][:] - norm_landmarks[i][15][:])
                angle_rng.append(math.acos(np.dot(vec_rng_high[i], vec_rng_low[i])/(np.linalg.norm(vec_rng_low[i])*np.linalg.norm(vec_rng_high[i]))))


                #Kleiner Finger                
                vec_ltl_low.append(norm_landmarks[i][17][:] - norm_landmarks[i][18][:])
                vec_ltl_high.append(norm_landmarks[i][20][:] - norm_landmarks[i][19][:])
                angle_ltl.append(math.acos(np.dot(vec_ltl_high[i], vec_ltl_low[i])/(np.linalg.norm(vec_ltl_low[i])*np.linalg.norm(vec_ltl_high[i]))))


            # Sammeln der Winkeldaten in einzelnem Array: angle_array[0,:] - thumb, angle_array[1,:] - index usw.
            # Note: angle_array will have shape (6, num_hands)
            angle_array.append(angle_thumb)
            angle_array.append(angle_thumb2)
            angle_array.append(angle_idx)
            angle_array.append(angle_mid)
            angle_array.append(angle_rng)
            angle_array.append(angle_ltl)
            angle_array = np.array(angle_array) # Shape (6, num_hands)

            # Convert radians to degrees
            if angle_array.size > 0:
                angle_array = angle_array * (180 / math.pi)

            # Correct orientation if needed (swap columns if left hand detected first but is actually right hand)
            # This part assumes 'orientation' reflects the handedness of the *first* detected hand (index 0)
            if len(norm_landmarks) == 2 and orientation < 0.5: # If 2 hands detected and first one is left (but might be mirrored)
                 # Swap columns so angle_array[:, 0] is always left, angle_array[:, 1] is always right
                 angle_array = angle_array[:, ::-1]


            # --- Servo Control based on detected hands ---
            global left_target
            # TODO: This whole section could be refactored further for clarity

            # Placeholder values for finger servo commands - will be updated if right hand detected
            value_thumb_stretch = 0
            value_thumb_stretch2 = 0
            value_idx = 0
            value_mid = 0
            value_rng = 0
            value_ltl = 0

            # Determine the column index for the right hand's angles in angle_array
            # It's 1 if two hands are detected, 0 if only the right hand is detected.
            right_hand_angle_col = -1
            if index_hand_right != -1:
                if len(norm_landmarks) == 2:
                    right_hand_angle_col = 1
                elif len(norm_landmarks) == 1:
                    right_hand_angle_col = 0

            # Map calculated right hand angles (degrees) to servo values if right hand is present
            if right_hand_angle_col != -1 and angle_array.size > 0 and angle_array.shape[1] > right_hand_angle_col:
                # These mappings are guesses based on original intent and need verification/tuning
                # Map angles (0-180 deg) to servo range (e.g., 0-9000 or specific pulse widths)
                # Example: Simple linear mapping (needs adjustment)
                # angle_thumb[right] -> value_thumb_stretch
                # angle_thumb2[right] -> value_thumb_stretch2
                # angle_idx[right] -> value_idx
                # ... etc ...
                # Placeholder scaling - replace with actual calibrated mapping
                value_thumb_stretch = int(angle_array[0, right_hand_angle_col] * 50)  # Map 180deg to 9000
                value_thumb_stretch2 = int(angle_array[1, right_hand_angle_col] * 50) # Map 180deg to 9000
                value_idx = int(angle_array[2, right_hand_angle_col] * 50)           # Map 180deg to 9000
                value_mid = int(angle_array[3, right_hand_angle_col] * 50)           # Map 180deg to 9000
                value_rng = int(angle_array[4, right_hand_angle_col] * 50)           # Map 180deg to 9000
                value_ltl = int(angle_array[5, right_hand_angle_col] * 50)           # Map 180deg to 9000
                # Clamp values to servo limits if necessary (e.g., 0 to 9000 or pulse min/max)


            # The calculation of value_... finger servo commands now happens *before* this block,
            # using the right hand's angles if available.

            if index_hand_right > -1: # If right hand is detected
                # Move left arm based on right hand's position (mirroring) using the combined helper
                # Shoulder Horizontal Left controlled by right hand horizontal pos
                set_servo(servoBrick3, SERVO_IDX_SHOULDER_HORIZONTAL_L, shoulder_horizontal_right) # Use renamed variable
                # Shoulder Vertical Left controlled by right hand vertical pos
                set_servo(servoBrick2, SERVO_IDX_SHOULDER_VERTICAL_L, shoulder_vertical_right)

                # Elbow Left - Set to a fixed position when right hand detected?
                set_servo(servoBrick3, SERVO_IDX_ELBOW_L, -3000)
                # Lower Arm Rotation Left - Set to a fixed position when right hand detected?
                set_servo(servoBrick3, SERVO_IDX_LOWER_ARM_L, 5500)

                # TODO: Control left fingers based on calculated right hand finger angles
                # set_servo(servoBrick3, SERVO_IDX_THUMB_L_OPPOSITION, value_thumb_stretch2)
                # set_servo_position(servoBrick3, SERVO_IDX_THUMB_L_OPPOSITION, value_thumb_stretch2)
                # TODO: Control left fingers based on calculated right hand finger angles (value_... variables)
                # set_servo(servoBrick3, SERVO_IDX_THUMB_L_OPPOSITION, value_thumb_stretch2)
                # set_servo(servoBrick3, SERVO_IDX_THUMB_L_PROXIMAL, value_thumb_stretch)
                # ... and so on for other fingers ...

                # Update left_target for recording (using the mirrored arm values and calculated finger values)
                # The value_... variables now hold the servo commands derived from the right hand's finger angles.
                left_target = {
                        "shoulder_vertical": shoulder_vertical_right, # Mirrored value
                        "shoulder_horizontal": shoulder_horizontal_right, # Renamed key and use renamed variable
                        "thumb_opposition": value_thumb_stretch2,  # Use calculated servo value from right hand angle
                        "thumb_proximal": value_thumb_stretch,     # Use calculated servo value from right hand angle
                        "index": value_idx,                        # Use calculated servo value from right hand angle
                        "middle": value_mid,                       # Use calculated servo value from right hand angle
                        "ring": value_rng,                         # Use calculated servo value from right hand angle
                        "pinky": value_ltl                         # Use calculated servo value from right hand angle
                        }
                if not self.is_initialized:
                    global record_thread
                    record_thread = threading.Thread(target=record_trajectory)
                    record_thread.start()
                    self.is_initialized = True

            # Angle calculation moved before this block

            #Aufbau der Matrix angle_array:
                #[Daumen Schliesswinkel links][Daumen Schliesswinkel rechts]
                #[Daumenwinkel Handebene links][Daumenwinkel Handebene rechts]
                #[Zeigefingerwinkel links][Zeigefingerwinkel rechts]
                #[Mittelfingerwinkel links][Mittelfingerwinkel rechts]
                #[Ringfingerwinkel links][Ringfingerwinkel rechts]
                #[Kleiner Fingerwinkel links][Kleiner Fingerwinkel rechts]
                    #2 Haende im Bild: Werte für linke hand in nullter Dimension eines Winkelvektors (self.angle_array_final[:,0]);Werte fuer rechte Hand in erster Dimension
                    #Eindimensional wenn nur eine Hand im Bild
                    #3 dimensional wenn 3 Haende im Bild aber nicht getestet

            for i in range(0, len(angle_array)):  #Umrechnung von rad in grad
                for j in range(0, len(angle_array[i])):
                    angle_array[i][j] = angle_array[i][j] * (180/math.pi)

            
            #print('Platzhalter', angle_array.ndim) #Print fuer Debugging

        #Speichere Array als Attribut von Tracker
        self.angle_array_final = angle_array  

        #Wenn Orientation kleiner/gleich 0.5 ist werden Positionen fuer linke und rechte Hand in matrix vertauscht, das wird im folgenden umgekehrt, funktioniert nur im Fall fuer 2 Haende
        if len(norm_landmarks) == 2:      
            if orientation < 0.5:
                self.angle_array_final[:,0] = angle_array[:,1]
                self.angle_array_final[:,1] = angle_array[:,0]
        
                


        # Statistics
        if self.stats:
            if res["pd_inf"]:
                self.nb_frames_pd_inference += 1
            else:
                if res["nb_lm_inf"] > 0:
                     self.nb_frames_lm_inference_after_landmarks_ROI += 1
            if res["nb_lm_inf"] == 0:
                self.nb_frames_no_hand += 1
            else:
                self.nb_frames_lm_inference += 1
                self.nb_lm_inferences += res["nb_lm_inf"]
                self.nb_failed_lm_inferences += res["nb_lm_inf"] - len(hands)

        return video_frame, hands, None


    def exit(self):
        # Move servos to a safe/final position before exiting
        # Using set_servo_position which also handles enabling
        # Note: Configuration (pulse, speed) is likely already set from next_frame
        print("Exiting: Moving servos to final positions...")
        # Shoulder vertical left (Example: move to 0)
        # set_servo(servoBrick2, SERVO_IDX_SHOULDER_VERTICAL_L, 0)
        # Elbow left
        set_servo(servoBrick3, SERVO_IDX_ELBOW_L, -3500)
        # Lower arm rotation left
        set_servo(servoBrick3, SERVO_IDX_LOWER_ARM_L, 5500)
        # Shoulder horizontal left
        set_servo(servoBrick3, SERVO_IDX_SHOULDER_HORIZONTAL_L, 6000)
        # TODO: Add final positions for other servos (right arm, fingers) if needed

        global recording
        recording = False
        global record_thread
        record_thread.join()
        with open(filename, "w") as f:
            json.dump([trajectory], f, indent=2)
        time.sleep(0.500)  #50 ms interval
        self.device.close()
        ipcon.disconnect()
        # Print some stats
        if self.stats:
            nb_frames = self.fps.nb_frames()
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {nb_frames})")
            print(f"# frames w/ no hand           : {self.nb_frames_no_hand} ({100*self.nb_frames_no_hand/nb_frames:.1f}%)")
            print(f"# frames w/ palm detection    : {self.nb_frames_pd_inference} ({100*self.nb_frames_pd_inference/nb_frames:.1f}%)")
            print(f"# frames w/ landmark inference : {self.nb_frames_lm_inference} ({100*self.nb_frames_lm_inference/nb_frames:.1f}%)- # after palm detection: {self.nb_frames_lm_inference - self.nb_frames_lm_inference_after_landmarks_ROI} - # after landmarks ROI prediction: {self.nb_frames_lm_inference_after_landmarks_ROI}")
            if not self.solo:
                print(f"On frames with at least one landmark inference, average number of landmarks inferences/frame: {self.nb_lm_inferences/self.nb_frames_lm_inference:.2f}")
            if self.nb_lm_inferences:
                print(f"# lm inferences: {self.nb_lm_inferences} - # failed lm inferences: {self.nb_failed_lm_inferences} ({100*self.nb_failed_lm_inferences/self.nb_lm_inferences:.1f}%)")
            
            print(f"trajectory obs: {len(trajectory['obs'])}")
            print(f"trajectory acts: {len(trajectory['acts'])}")
