"""
HandTrackerEdge - Optimized version for hand tracking with servo control
Refactored for better maintainability and configurability
"""

import numpy as np
import mediapipe_utils as mpu
import depthai as dai
import cv2
from pathlib import Path
from FPS import FPS
import sys
from string import Template
import marshal
from typing import Optional, List, Tuple

from servo_controller import ServoController
from hand_angle_calculator import HandAngleCalculator


SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
LANDMARK_MODEL_SPARSE = str(SCRIPT_DIR / "models/hand_landmark_sparse_sh4.blob")
DETECTION_POSTPROCESSING_MODEL = str(SCRIPT_DIR / "custom_models/PDPostProcessing_top2_sh1.blob")
TEMPLATE_MANAGER_SCRIPT_SOLO = str(SCRIPT_DIR / "template_manager_script_solo.py")
TEMPLATE_MANAGER_SCRIPT_DUO = str(SCRIPT_DIR / "template_manager_script_duo.py")


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    """Converts an array to planar format for DepthAI."""
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


class HandTracker:
    """
    Mediapipe Hand Tracker for DepthAI with servo control for robot hand imitation.
    
    This class has been refactored for:
    - Better maintainability through separation of hand tracking and servo control
    - Configurability via YAML file
    - Clearer code structure without magic numbers
    
    Args:
        input_src: Frame source ("rgb", "rgb_laconic" or None for internal camera)
        pd_model: Path to palm detection model
        pd_score_thresh: Confidence threshold for palm detection (0-1)
        pd_nms_thresh: NMS threshold
        use_lm: Whether to use landmark model
        lm_model: Landmark model ("full", "lite", "sparse" or path)
        lm_score_thresh: Confidence threshold for landmarks (0-1)
        use_world_landmarks: Whether to calculate world landmarks
        pp_model: Path to post-processing model
        solo: Whether to detect only one hand (Edge mode: always True)
        xyz: Whether to calculate depth information
        crop: Whether to apply square cropping
        internal_fps: Internal camera FPS (None = automatic)
        resolution: Sensor resolution ("full" or "ultra")
        internal_frame_height: Internal frame height
        use_gesture: Whether to recognize gestures
        use_handedness_average: Whether to average handedness
        single_hand_tolerance_thresh: Tolerance for single-hand mode
        use_same_image: Whether to use same image for both hands
        lm_nb_threads: Number of threads for landmark model
        stats: Whether to display statistics
        trace: Trace level (0-15)
        enable_servo_control: Whether to enable servo control
        config_path: Path to configuration file
    """
    
    def __init__(self, input_src=None,
                pd_model=PALM_DETECTION_MODEL, 
                pd_score_thresh=0.5, pd_nms_thresh=0.3,
                use_lm=True,
                lm_model="full",
                lm_score_thresh=0.5,
                use_world_landmarks=False,
                pp_model=DETECTION_POSTPROCESSING_MODEL,
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
                enable_servo_control=True,
                config_path=None):
        
        # Validation
        self.use_lm = use_lm
        if not use_lm:
            print("use_lm=False is not supported in Edge mode.")
            sys.exit(1)
        
        # Model paths
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
        
        # Parameters
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
        
        # Initialize servo control
        self.enable_servo_control = enable_servo_control
        if self.enable_servo_control:
            try:
                self.servo_controller = ServoController(config_path)
                self.angle_calculator = HandAngleCalculator()
                self.calibration = self.servo_controller.config['calibration']
            except Exception as e:
                print(f"Warning: Servo control could not be initialized: {e}")
                print("Continuing without servo control...")
                self.enable_servo_control = False
                self.servo_controller = None
                self.angle_calculator = None
        else:
            self.servo_controller = None
            self.angle_calculator = None
        
        # Initialize DepthAI device
        self.device = dai.Device()
        
        # Configure input source
        if input_src is None or input_src == "rgb" or input_src == "rgb_laconic":
            self.input_type = "rgb"
            self.laconic = input_src == "rgb_laconic"
            
            if resolution == "full":
                self.resolution = (1920, 1080)
            elif resolution == "ultra":
                self.resolution = (3840, 2160)
            else:
                print(f"Error: {resolution} is not a valid resolution!")
                sys.exit(1)
            print("Sensor resolution:", self.resolution)
            
            # Check XYZ (depth)
            if xyz:
                cameras = self.device.getConnectedCameras()
                if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras:
                    self.xyz = True
                else:
                    print("Warning: depth unavailable on this device, 'xyz' argument is ignored")
            
            # Determine FPS
            if internal_fps is None:
                fps_map = {
                    "full": {True: 22, False: 26},
                    "lite": {True: 29, False: 36},
                    "sparse": {True: 24, False: 29}
                }
                self.internal_fps = fps_map.get(lm_model, {}).get(self.xyz, 39)
            else:
                self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")
            
            self.video_fps = self.internal_fps
            
            # Calculate frame size
            if self.crop:
                self.frame_size, self.scale_nd = mpu.find_isp_scale_params(
                    internal_frame_height, self.resolution
                )
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
                self.crop_w = (int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2
            else:
                width, self.scale_nd = mpu.find_isp_scale_params(
                    internal_frame_height * self.resolution[0] / self.resolution[1],
                    self.resolution,
                    is_height=False
                )
                self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w
                self.crop_w = 0
            
            print(f"Internal camera image size: {self.img_w} x {self.img_h} - pad_h: {self.pad_h}")
        else:
            print("Invalid input source:", input_src)
            sys.exit(1)
        
        # Start pipeline
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")
        
        # Define queues
        if not self.laconic:
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        self.q_manager_out = self.device.getOutputQueue(name="manager_out", maxSize=1, blocking=False)
        
        if self.trace & 4:
            self.q_pre_pd_manip_out = self.device.getOutputQueue(name="pre_pd_manip_out", maxSize=1, blocking=False)
            self.q_pre_lm_manip_out = self.device.getOutputQueue(name="pre_lm_manip_out", maxSize=1, blocking=False)
        
        # Statistics
        self.fps = FPS()
        self.nb_frames_pd_inference = 0
        self.nb_frames_lm_inference = 0
        self.nb_lm_inferences = 0
        self.nb_failed_lm_inferences = 0
        self.nb_frames_lm_inference_after_landmarks_ROI = 0
        self.nb_frames_no_hand = 0
        
        self.angle_array_final = np.array([])
    
    def create_pipeline(self):
        """Creates the DepthAI pipeline."""
        print("Creating pipeline...")
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)
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
        
        # Manager Script Node
        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(self.build_manager_script())
        
        # XYZ (Stereo Depth) if enabled
        if self.xyz:
            print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes...")
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
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            stereo.setSubpixel(False)
            
            spatial_location_calculator = pipeline.createSpatialLocationCalculator()
            spatial_location_calculator.setWaitForConfigInput(True)
            spatial_location_calculator.inputDepth.setBlocking(False)
            spatial_location_calculator.inputDepth.setQueueSize(1)
            
            left.out.link(stereo.left)
            right.out.link(stereo.right)
            stereo.depth.link(spatial_location_calculator.inputDepth)
            manager_script.outputs['spatial_location_config'].link(spatial_location_calculator.inputConfig)
            spatial_location_calculator.out.link(manager_script.inputs['spatial_data'])
        
        # Palm Detection Pre-Processing
        print("Creating Palm Detection pre processing image manip...")
        pre_pd_manip = pipeline.create(dai.node.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length * self.pd_input_length * 3)
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_pd_manip.inputImage)
        manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)
        
        if self.trace & 4:
            pre_pd_manip_out = pipeline.createXLinkOut()
            pre_pd_manip_out.setStreamName("pre_pd_manip_out")
            pre_pd_manip.out.link(pre_pd_manip_out.input)
        
        # Palm Detection Model
        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(self.pd_model)
        pre_pd_manip.out.link(pd_nn.input)
        
        # Palm Detection Post-Processing
        print("Creating Palm Detection post processing Neural Network...")
        post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
        post_pd_nn.setBlobPath(self.pp_model)
        pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs['from_post_pd_nn'])
        
        # Manager Output
        manager_out = pipeline.create(dai.node.XLinkOut)
        manager_out.setStreamName("manager_out")
        manager_script.outputs['host'].link(manager_out.input)
        
        # Landmark Pre-Processing
        print("Creating Hand Landmark pre processing image manip...")
        self.lm_input_length = 224
        pre_lm_manip = pipeline.create(dai.node.ImageManip)
        pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length * self.lm_input_length * 3)
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_lm_manip.inputImage)
        
        if self.trace & 4:
            pre_lm_manip_out = pipeline.createXLinkOut()
            pre_lm_manip_out.setStreamName("pre_lm_manip_out")
            pre_lm_manip.out.link(pre_lm_manip_out.input)
        
        manager_script.outputs['pre_lm_manip_cfg'].link(pre_lm_manip.inputConfig)
        
        # Landmark Model
        print(f"Creating Hand Landmark Neural Network ({'1 thread' if self.lm_nb_threads == 1 else '2 threads'})...")
        lm_nn = pipeline.create(dai.node.NeuralNetwork)
        lm_nn.setBlobPath(self.lm_model)
        lm_nn.setNumInferenceThreads(self.lm_nb_threads)
        pre_lm_manip.out.link(lm_nn.input)
        lm_nn.out.link(manager_script.inputs['from_lm_nn'])
        
        print("Pipeline created.")
        return pipeline
    
    def build_manager_script(self):
        """Builds the manager script from template."""
        template_file = TEMPLATE_MANAGER_SCRIPT_SOLO if self.solo else TEMPLATE_MANAGER_SCRIPT_DUO
        
        with open(template_file, 'r') as file:
            template = Template(file.read())
        
        code = template.substitute(
            _TRACE1="node.warn" if self.trace & 1 else "#",
            _TRACE2="node.warn" if self.trace & 2 else "#",
            _pd_score_thresh=self.pd_score_thresh,
            _lm_score_thresh=self.lm_score_thresh,
            _pad_h=self.pad_h,
            _img_h=self.img_h,
            _img_w=self.img_w,
            _frame_size=self.frame_size,
            _crop_w=self.crop_w,
            _IF_XYZ="" if self.xyz else '"""',
            _IF_USE_HANDEDNESS_AVERAGE="" if self.use_handedness_average else '"""',
            _single_hand_tolerance_thresh=self.single_hand_tolerance_thresh,
            _IF_USE_SAME_IMAGE="" if self.use_same_image else '"""',
            _IF_USE_WORLD_LANDMARKS="" if self.use_world_landmarks else '"""',
        )
        
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)
        
        if self.trace & 8:
            with open("tmp_code.py", "w") as file:
                file.write(code)
        
        return code
    
    def extract_hand_data(self, res, hand_idx):
        """Extracts hand data from result."""
        hand = mpu.HandRegion()
        hand.rect_x_center_a = res["rect_center_x"][hand_idx] * self.frame_size
        hand.rect_y_center_a = res["rect_center_y"][hand_idx] * self.frame_size
        hand.rect_w_a = hand.rect_h_a = res["rect_size"][hand_idx] * self.frame_size
        hand.rotation = res["rotation"][hand_idx]
        hand.rect_points = mpu.rotated_rect_to_points(
            hand.rect_x_center_a, hand.rect_y_center_a, hand.rect_w_a, hand.rect_h_a, hand.rotation
        )
        hand.lm_score = res["lm_score"][hand_idx]
        hand.handedness = res["handedness"][hand_idx]
        hand.label = "right" if hand.handedness > 0.5 else "left"
        hand.norm_landmarks = np.array(res['rrn_lms'][hand_idx]).reshape(-1, 3)
        hand.landmarks = (np.array(res["sqn_lms"][hand_idx]) * self.frame_size).reshape(-1, 2).astype(np.int32)
        
        if self.xyz:
            hand.xyz = np.array(res["xyz"][hand_idx])
            hand.xyz_zone = res["xyz_zone"][hand_idx]
        
        # Remove padding
        if self.pad_h > 0:
            hand.landmarks[:, 1] -= self.pad_h
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][1] -= self.pad_h
        if self.pad_w > 0:
            hand.landmarks[:, 0] -= self.pad_w
            for i in range(len(hand.rect_points)):
                hand.rect_points[i][0] -= self.pad_w
        
        if self.use_world_landmarks:
            hand.world_landmarks = np.array(res["world_lms"][hand_idx]).reshape(-1, 3)
        
        if self.use_gesture:
            mpu.recognize_gesture(hand)
        
        return hand
    
    def _process_servo_control(self, hands: List):
        """Processes servo control based on detected hands."""
        if not self.enable_servo_control or not self.servo_controller:
            return
        
        if not hands:
            # No hand detected - set default positions
            self.servo_controller.set_default_positions()
            return
        
        # Sort hands by label
        left_hand = None
        right_hand = None
        
        for hand in hands:
            if hand.label == "left":
                left_hand = hand
            elif hand.label == "right":
                right_hand = hand
        
        # Calculate angles
        angle_dicts, angle_array = self.angle_calculator.calculate_angles_for_hands(
            hands, self.calibration
        )
        self.angle_array_final = angle_array
        
        # Control left hand
        if left_hand:
            left_idx = next(i for i, h in enumerate(hands) if h.label == "left")
            shoulder_h, shoulder_v = self.angle_calculator.calculate_shoulder_positions(
                left_hand.landmarks, "left", self.calibration
            )
            
            finger_angles = angle_dicts[left_idx]
            
            # Right arm positions
            right_arm_pos = {
                'elbow': self.servo_controller.config['default_positions']['no_hand']['right_arm_when_left_detected']['elbow'],
                'lower_arm_rotation': self.servo_controller.config['default_positions']['no_hand']['right_arm_when_left_detected']['lower_arm_rotation']
            }
            
            self.servo_controller.control_left_hand(
                shoulder_h, shoulder_v, finger_angles, right_arm_pos
            )
        
        # Control right hand
        if right_hand:
            right_idx = next(i for i, h in enumerate(hands) if h.label == "right")
            shoulder_h, shoulder_v = self.angle_calculator.calculate_shoulder_positions(
                right_hand.landmarks, "right", self.calibration
            )
            
            finger_angles = angle_dicts[right_idx]
            
            # Left arm positions
            left_arm_pos = {
                'elbow': self.servo_controller.config['default_positions']['no_hand']['left_arm_when_right_detected']['elbow'],
                'lower_arm_rotation': self.servo_controller.config['default_positions']['no_hand']['left_arm_when_right_detected']['lower_arm_rotation']
            }
            
            self.servo_controller.control_right_hand(
                shoulder_h, shoulder_v, finger_angles, left_arm_pos
            )
    
    def next_frame(self):
        """Processes the next frame."""
        self.fps.update()
        
        # Get video frame
        if self.laconic:
            video_frame = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()
        
        # Debug output
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
        
        # Extract hands
        for i in range(len(res.get("lm_score", []))):
            hand = self.extract_hand_data(res, i)
            hands.append(hand)
        
        # Servo control
        self._process_servo_control(hands)
        
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
        """Exits the tracker and disconnects connections."""
        self.device.close()
        
        if self.servo_controller:
            self.servo_controller.disconnect()
        
        # Print statistics
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
