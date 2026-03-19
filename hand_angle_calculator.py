"""
HandAngleCalculator - Calculates finger angles from hand landmarks.
"""

import numpy as np
import math
from typing import List, Dict, Tuple


class HandAngleCalculator:
    """
    Calculates finger angles from normalized hand landmark coordinates.
    """
    
    # Landmark indices according to MediaPipe Hand Landmark model
    # See: https://google.github.io/mediapipe/solutions/hands.html
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20
    
    @staticmethod
    def calculate_angle_between_vectors(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates the angle between two vectors in radians.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Angle in radians
        """
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        # Clamp to [-1, 1] to avoid numerical errors
        cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
        return math.acos(cos_angle)
    
    @staticmethod
    def calculate_finger_angles(norm_landmarks: np.ndarray) -> Dict[str, float]:
        """
        Calculates all finger angles from normalized landmarks.
        
        Args:
            norm_landmarks: Array of shape (21, 3) with normalized landmark coordinates
            
        Returns:
            Dictionary with finger angles:
            - thumb_stretch: Thumb stretch angle
            - thumb_opposition: Thumb opposition (angle in hand plane)
            - index: Index finger angle
            - middle: Middle finger angle
            - ring: Ring finger angle
            - little: Little finger angle
        """
        angles = {}
        
        # Thumb stretch angle (closing angle)
        vec_thumb_low = norm_landmarks[HandAngleCalculator.THUMB_MCP] - norm_landmarks[HandAngleCalculator.THUMB_CMC]
        vec_thumb_high = norm_landmarks[HandAngleCalculator.THUMB_TIP] - norm_landmarks[HandAngleCalculator.THUMB_IP]
        angles['thumb_stretch'] = HandAngleCalculator.calculate_angle_between_vectors(
            vec_thumb_high, vec_thumb_low
        )
        
        # Thumb opposition (angle in hand plane)
        vec_thumb_low2 = norm_landmarks[HandAngleCalculator.THUMB_IP] - norm_landmarks[HandAngleCalculator.THUMB_MCP]
        vec_thumb_high2 = norm_landmarks[HandAngleCalculator.MIDDLE_MCP] - norm_landmarks[HandAngleCalculator.WRIST]
        angles['thumb_opposition'] = HandAngleCalculator.calculate_angle_between_vectors(
            vec_thumb_high2, vec_thumb_low2
        )
        
        # Index finger
        vec_idx_low = norm_landmarks[HandAngleCalculator.INDEX_PIP] - norm_landmarks[HandAngleCalculator.INDEX_MCP]
        vec_idx_high = norm_landmarks[HandAngleCalculator.INDEX_TIP] - norm_landmarks[HandAngleCalculator.INDEX_DIP]
        angles['index'] = HandAngleCalculator.calculate_angle_between_vectors(
            vec_idx_high, vec_idx_low
        )
        
        # Middle finger
        vec_mid_low = norm_landmarks[HandAngleCalculator.MIDDLE_PIP] - norm_landmarks[HandAngleCalculator.MIDDLE_MCP]
        vec_mid_high = norm_landmarks[HandAngleCalculator.MIDDLE_TIP] - norm_landmarks[HandAngleCalculator.MIDDLE_DIP]
        angles['middle'] = HandAngleCalculator.calculate_angle_between_vectors(
            vec_mid_high, vec_mid_low
        )
        
        # Ring finger
        vec_rng_low = norm_landmarks[HandAngleCalculator.RING_PIP] - norm_landmarks[HandAngleCalculator.RING_MCP]
        vec_rng_high = norm_landmarks[HandAngleCalculator.RING_TIP] - norm_landmarks[HandAngleCalculator.RING_DIP]
        angles['ring'] = HandAngleCalculator.calculate_angle_between_vectors(
            vec_rng_high, vec_rng_low
        )
        
        # Little finger
        vec_ltl_low = norm_landmarks[HandAngleCalculator.PINKY_PIP] - norm_landmarks[HandAngleCalculator.PINKY_MCP]
        vec_ltl_high = norm_landmarks[HandAngleCalculator.PINKY_TIP] - norm_landmarks[HandAngleCalculator.PINKY_DIP]
        angles['little'] = HandAngleCalculator.calculate_angle_between_vectors(
            vec_ltl_high, vec_ltl_low
        )
        
        return angles
    
    @staticmethod
    def calculate_shoulder_positions(landmarks: np.ndarray, hand_label: str, 
                                    calibration: Dict) -> Tuple[int, int]:
        """
        Calculates shoulder positions from hand landmarks.
        
        Args:
            landmarks: Landmark coordinates in pixels (array of shape (21, 2))
            hand_label: "left" or "right"
            calibration: Calibration configuration
            
        Returns:
            Tuple (shoulder_horizontal, shoulder_vertical)
        """
        wrist = landmarks[HandAngleCalculator.WRIST]
        cal_shoulder = calibration['shoulder']
        
        if hand_label == "left":
            horizontal = int(wrist[0] * cal_shoulder['horizontal']['left']['multiplier'] + 
                           cal_shoulder['horizontal']['left']['offset'])
            vertical = int(cal_shoulder['vertical']['left']['base'] - 
                          wrist[1] * cal_shoulder['vertical']['left']['multiplier'] + 
                          cal_shoulder['vertical']['left']['offset_y'])
        else:  # right
            horizontal = int(wrist[0] * cal_shoulder['horizontal']['right']['multiplier'] + 
                           cal_shoulder['horizontal']['right']['offset'])
            vertical = int(wrist[1] * cal_shoulder['vertical']['right']['multiplier'] + 
                          cal_shoulder['vertical']['right']['offset'])
        
        return horizontal, vertical
    
    @staticmethod
    def calculate_angles_for_hands(hands: List, calibration: Dict) -> Tuple[List[Dict[str, float]], np.ndarray]:
        """
        Calculates angles for all detected hands.
        
        Args:
            hands: List of HandRegion objects
            calibration: Calibration configuration
            
        Returns:
            Tuple (angle_dicts, angle_array)
            - angle_dicts: List of dictionaries with finger angles per hand
            - angle_array: NumPy array with angles in degrees (6 fingers x N hands)
        """
        angle_dicts = []
        
        for hand in hands:
            angles = HandAngleCalculator.calculate_finger_angles(hand.norm_landmarks)
            angle_dicts.append(angles)
        
        if not angle_dicts:
            return [], np.array([])
        
        # Convert to array format
        num_hands = len(angle_dicts)
        angle_array = np.zeros((6, num_hands))
        
        finger_order = ['thumb_stretch', 'thumb_opposition', 'index', 'middle', 'ring', 'little']
        
        for i, angles in enumerate(angle_dicts):
            for j, finger in enumerate(finger_order):
                # Convert from radians to degrees
                angle_array[j, i] = angles[finger] * (180.0 / math.pi)
        
        return angle_dicts, angle_array
