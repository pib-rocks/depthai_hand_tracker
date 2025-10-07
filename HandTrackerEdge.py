import numpy as np
from collections import namedtuple
from pib_sdk.control import *
from pib_sdk.kinematics import *
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

#Start control client to send all data to ROS2 topics
w = Write()
#Set all motors to default settings, except hands with faster velocity
w.set(right_hand, left_hand, default, velocity=30000)
w.set(right_arm, left_arm, default, velocity=10000)

def safe_angle(u: np.ndarray, v: np.ndarray, default: float = 0.0) -> float:
    #Return angle between vectors u and v in radians, safely. to prevent shutdown
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0.0 or nv == 0.0:
        return default
    cosx = float(np.dot(u, v) / (nu * nv))
    # Clamp to valid domain for acos
    if cosx > 1.0:
        cosx = 1.0
    elif cosx < -1.0:
        cosx = -1.0
    return math.acos(cosx)

SCRIPT_DIR = Path(__file__).resolve().parent
PALM_DETECTION_MODEL = str(SCRIPT_DIR / "models/palm_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/hand_landmark_full_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/hand_landmark_lite_sh4.blob")
LANDMARK_MODEL_SPARSE = str(SCRIPT_DIR / "models/hand_landmark_sparse_sh4.blob")
DETECTION_POSTPROCESSING_MODEL = str(SCRIPT_DIR / "custom_models/PDPostProcessing_top2_sh1.blob")
TEMPLATE_MANAGER_SCRIPT_SOLO = str(SCRIPT_DIR / "template_manager_script_solo.py")
TEMPLATE_MANAGER_SCRIPT_DUO = str(SCRIPT_DIR / "template_manager_script_duo.py")


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()


class HandTracker:
    '''
    Mediapipe Hand Tracker for DepthAI

    Arguments:
    - input_src: Frame source
        - "rgb" / None ? OAK camera
        - "rgb_laconic" ? OAK camera without host frames (Edge only)
        - file path ? image/video
        - int ? webcam ID
        (Edge mode supports only "rgb" and "rgb_laconic")
    - pd_model: Palm detection model file
    - pd_score: Palm detection confidence (0?1)
    - pd_nms_thresh: Non-max suppression threshold
    - use_lm: Run landmark model if True
    - lm_model: Landmark model ("full", "lite", "sparse", or path)
    - lm_score_thresh: Landmark confidence (0?1)
    - use_world_landmarks: Also output 3D world coords if True
    - pp_model: Post-processing model file
    - solo: Detect max one hand (faster, required in Edge mode)
    - xyz: Compute (x,y,z) palm coords if True
    - crop: Apply square cropping on input images
    - internal_fps: FPS for internal camera
    - resolution: "full" (1920x1080) or "ultra" (3840x2160)
    - internal_frame_height: ISP scale height (width auto-set)
    - use_gesture: Recognize simple poses (ONE, TWO, OK, FIST, etc.)
    - use_handedness_average: Smooth handedness over frames
    - single_hand_tolerance_thresh: (#frames before re-running palm detection
                                    in Duo mode when only one hand visible)
    - lm_nb_threads: Inference threads for landmark model (1 or 2)
    - use_same_image: (Duo only) reuse same image for both hands ? faster but less accurate
    - stats: Print performance stats on exit
    - trace: Debug output level (bitmask)
             1 = app info
             2 = low-level
             4 = show cv2 windows
             8 = save manager code
             (combine values, e.g. 3 = 1+2)
    '''

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
        code = re.sub('\n\s*\n', '\n', code)
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

        self.fps.update()

        if self.laconic:
            video_frame = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
        else:
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()
            
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
        horizontal_right = 0
        horizontal_left = 0
        vertical_right = 0
        vertical_left = 0
        
        if len(res.get("lm_score",[])) == 0:
            w.move(All, zero_position)

        for i in range(len(res.get("lm_score",[]))):
            hand = self.extract_hand_data(res, i)
            if hand.label=="left":
                # variable inverted due to mirroring
                index_hand_left = i
                horizontal_right = hand.landmarks[0][0] * 0.1 - 100 #offseted more than left due to opposite direction
                vertical_right = hand.landmarks[0][1] * 0.1 - 35
            if hand.label=="right":
                index_hand_right = i
                horizontal_left = hand.landmarks[0][0] * 0.1 - 35
                vertical_left = hand.landmarks[0][1] * 0.1 - 35
            hands.append(hand)
            # extract normalized landmark coordinates from hand; append in loop to account for number of hands
            norm_landmarks.append(hand.norm_landmarks)      
            orientation = hand.handedness
       
        angle_array = []
        # only compute if hand landmarks are available
        if norm_landmarks:          
            # Angle containers (radians)
            angle_thumb  = []
            angle_thumb2 = []
            angle_idx    = []
            angle_mid    = []
            angle_rng    = []
            angle_ltl    = []

            # Consider number of hands; landmark indices follow the Mediapipe drawing
            for i in range(len(norm_landmarks)):
                lm = norm_landmarks[i]
                angle_thumb.append(safe_angle(lm[4] - lm[3], lm[1] - lm[2]))
                angle_thumb2.append(safe_angle(lm[9] - lm[0], lm[3] - lm[2]))
                angle_idx.append(safe_angle(lm[8] - lm[7], lm[5] - lm[6]))
                angle_mid.append(safe_angle(lm[12] - lm[11], lm[9] - lm[10]))
                angle_rng.append(safe_angle(lm[16] - lm[15], lm[13] - lm[14]))
                angle_ltl.append(safe_angle(lm[20] - lm[19], lm[17] - lm[18]))

            if index_hand_left > -1:
                value_thumb_stretch    = 50 * angle_thumb[index_hand_left]  - 90
                value_thumb_opposition = 50 * angle_thumb2[index_hand_left] - 90
                value_idx              = 50 * angle_idx[index_hand_left]    - 90
                value_mid              = 50 * angle_mid[index_hand_left]    - 90
                value_rng              = 50 * angle_rng[index_hand_left]    - 90
                value_ltl              = 50 * angle_ltl[index_hand_left]    - 90

                # Shoulder horizontal / upper arm rotation and visibility pose
                w.move(right_arm, vertical_right, 10, horizontal_right, -50, -70, 50,
                right_hand, value_thumb_opposition, value_thumb_stretch, value_idx, value_mid, value_rng, value_ltl)
                
            if index_hand_right > -1:
                # Hand calculations (keep radians here to preserve original control logic)
                value_thumb_stretch    = 50 * angle_thumb[index_hand_right]  - 90
                value_thumb_opposition = 50 * angle_thumb2[index_hand_right] - 90
                value_idx              = 50 * angle_idx[index_hand_right]    - 90
                value_mid              = 50 * angle_mid[index_hand_right]    - 90
                value_rng              = 50 * angle_rng[index_hand_right]    - 90
                value_ltl              = 50 * angle_ltl[index_hand_right]    - 90

                # Shoulder horizontal / upper arm rotation and visibility pose and left hand values
                w.move(left_arm, vertical_left, 10, horizontal_left, -50, -70, 50, 
                left_hand, value_thumb_opposition, value_thumb_stretch, value_idx, value_mid, value_rng, value_ltl)
                
            # Collect angle data into a single array
            angle_array = np.array([angle_thumb, angle_thumb2, angle_idx, angle_mid, angle_rng, angle_ltl], dtype=object)


            # Convert radians ? degrees for exported angle_array_final
            for i in range(len(angle_array)):
                for j in range(len(angle_array[i])):
                    angle_array[i][j] = angle_array[i][j] * (180 / math.pi)

        # Save array as an attribute of tracker
        self.angle_array_final = angle_array

        # If handedness orientation ? 0.5, swap left/right columns (only when two hands detected)
        if len(norm_landmarks) == 2:
            if orientation < 0.5:
                self.angle_array_final[:, 0] = angle_array[:, 1]
                self.angle_array_final[:, 1] = angle_array[:, 0]
                


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
        self.device.close()
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
