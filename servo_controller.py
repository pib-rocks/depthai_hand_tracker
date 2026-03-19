"""
ServoController - Manages the control of Tinkerforge servo motors
for robot hand imitation.
"""

from typing import Optional, Dict, Any
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_servo_v2 import BrickletServoV2
import yaml
from pathlib import Path


class ServoController:
    """
    Manages the connection and control of Tinkerforge servo motors.
    Loads configuration from YAML file.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initializes the ServoController.
        
        Args:
            config_path: Path to configuration file (default: config.yaml in current directory)
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config = self._load_config(config_path)
        self.ipcon = IPConnection()
        self.servo_bricks = {}
        self._connect()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Loads configuration from a YAML file."""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _connect(self):
        """Establishes connection to servo bricks."""
        tinkerforge_config = self.config['tinkerforge']
        self.ipcon.connect(tinkerforge_config['host'], tinkerforge_config['port'])
        
        for brick_config in tinkerforge_config['servo_bricks']:
            uid = brick_config['uid']
            self.servo_bricks[uid] = BrickletServoV2(uid, self.ipcon)
    
    def _get_servo_brick(self, brick_index: int) -> BrickletServoV2:
        """
        Returns the servo brick by index.
        
        Args:
            brick_index: 0 = Brick 1 (right hand), 1 = Brick 2 (shoulder), 2 = Brick 3 (left hand)
        """
        uids = [brick['uid'] for brick in self.config['tinkerforge']['servo_bricks']]
        if brick_index < 0 or brick_index >= len(uids):
            raise ValueError(f"Invalid brick index: {brick_index}")
        return self.servo_bricks[uids[brick_index]]
    
    def set_servo_position(self, brick_index: int, channel: int, position: int, 
                          pulse_width: Optional[tuple] = None):
        """
        Sets the position of a servo.
        
        Args:
            brick_index: Index of servo brick (0-2)
            channel: Servo channel (0-15)
            position: Position (-10000 to 10000)
            pulse_width: Optional (min, max) in microseconds, otherwise from config
        """
        brick = self._get_servo_brick(brick_index)
        servo_config = self.config['servo']
        
        if pulse_width is None:
            pw_min, pw_max = servo_config['pulse_width']['min'], servo_config['pulse_width']['max']
        else:
            pw_min, pw_max = pulse_width
        
        brick.set_pulse_width(channel, pw_min, pw_max)
        brick.set_position(channel, position)
        
        motion_config = servo_config['motion_config']
        brick.set_motion_configuration(
            channel,
            motion_config['velocity'],
            motion_config['acceleration'],
            motion_config['deceleration']
        )
        brick.set_enable(channel, True)
    
    def set_default_positions(self):
        """Sets all servos to default positions (no hand detected)."""
        defaults = self.config['default_positions']['no_hand']
        
        # Shoulder vertical
        self.set_servo_position(1, 9, defaults['shoulder_vertical']['left'])  # Brick 2, channel 9
        self.set_servo_position(1, 1, defaults['shoulder_vertical']['right'])  # Brick 2, channel 1
        
        # Elbow
        self.set_servo_position(2, 8, defaults['elbow']['left'])  # Brick 3, channel 8
        self.set_servo_position(0, 8, defaults['elbow']['right'])  # Brick 1, channel 8
        
        # Lower arm rotation
        self.set_servo_position(2, 7, defaults['lower_arm_rotation']['left'])  # Brick 3, channel 7
        self.set_servo_position(0, 7, defaults['lower_arm_rotation']['right'])  # Brick 1, channel 7
        
        # Shoulder horizontal
        self.set_servo_position(2, 9, defaults['shoulder_horizontal']['left'])  # Brick 3, channel 9
        self.set_servo_position(0, 9, defaults['shoulder_horizontal']['right'])  # Brick 1, channel 9
    
    def control_left_hand(self, shoulder_horizontal: int, shoulder_vertical: int,
                         finger_angles: Dict[str, float], 
                         right_arm_positions: Optional[Dict[str, int]] = None):
        """
        Controls the left hand and associated servos.
        
        Args:
            shoulder_horizontal: Horizontal shoulder position
            shoulder_vertical: Vertical shoulder position
            finger_angles: Dictionary with finger angles (thumb_stretch, thumb_opposition, index, middle, ring, little)
            right_arm_positions: Optional positions for right arm (elbow, lower_arm_rotation)
        """
        channels = self.config['servo_channels']['left_hand']
        cal = self.config['calibration']['finger_angles']
        
        # Shoulder horizontal
        self.set_servo_position(0, channels['shoulder_horizontal'], shoulder_horizontal)
        
        # Shoulder vertical
        shoulder_channels = self.config['servo_channels']['shoulder']
        self.set_servo_position(1, shoulder_channels['vertical_left'], shoulder_vertical)
        
        # Fingers
        self.set_servo_position(0, channels['thumb_stretch'], 
                               int(cal['thumb_stretch']['multiplier'] * finger_angles['thumb_stretch'] + 
                                   cal['thumb_stretch']['offset']))
        self.set_servo_position(0, channels['thumb_opposition'], 
                               int(cal['thumb_opposition']['multiplier'] * finger_angles['thumb_opposition'] + 
                                   cal['thumb_opposition']['offset_left']))
        self.set_servo_position(0, channels['index'], 
                               int(cal['index']['multiplier'] * finger_angles['index'] + 
                                   cal['index']['offset_left']))
        self.set_servo_position(0, channels['middle'], 
                               int(cal['middle']['multiplier'] * finger_angles['middle'] + 
                                   cal['middle']['offset']))
        self.set_servo_position(0, channels['ring'], 
                               int(cal['ring']['multiplier'] * finger_angles['ring'] + 
                                   cal['ring']['offset_left']))
        self.set_servo_position(0, channels['little'], 
                               int(cal['little']['multiplier'] * finger_angles['little'] + 
                                   cal['little']['offset_left']))
        
        # Right arm (when left hand is detected)
        if right_arm_positions:
            defaults = self.config['default_positions']['no_hand']['right_arm_when_left_detected']
            self.set_servo_position(0, channels['elbow'], 
                                   right_arm_positions.get('elbow', defaults['elbow']))
            self.set_servo_position(0, channels['lower_arm_rotation'], 
                                   right_arm_positions.get('lower_arm_rotation', defaults['lower_arm_rotation']))
    
    def control_right_hand(self, shoulder_horizontal: int, shoulder_vertical: int,
                          finger_angles: Dict[str, float],
                          left_arm_positions: Optional[Dict[str, int]] = None):
        """
        Controls the right hand and associated servos.
        
        Args:
            shoulder_horizontal: Horizontal shoulder position
            shoulder_vertical: Vertical shoulder position
            finger_angles: Dictionary with finger angles
            left_arm_positions: Optional positions for left arm
        """
        channels = self.config['servo_channels']['right_hand']
        cal = self.config['calibration']['finger_angles']
        
        # Shoulder horizontal
        self.set_servo_position(2, channels['shoulder_horizontal'], shoulder_horizontal)
        
        # Shoulder vertical
        shoulder_channels = self.config['servo_channels']['shoulder']
        self.set_servo_position(1, shoulder_channels['vertical_right'], shoulder_vertical)
        
        # Fingers
        self.set_servo_position(2, channels['thumb_stretch'], 
                               int(cal['thumb_stretch']['multiplier'] * finger_angles['thumb_stretch'] + 
                                   cal['thumb_stretch']['offset']))
        self.set_servo_position(2, channels['thumb_opposition'], 
                               int(cal['thumb_opposition']['multiplier'] * finger_angles['thumb_opposition'] + 
                                   cal['thumb_opposition']['offset_right']))
        self.set_servo_position(2, channels['index'], 
                               int(cal['index']['multiplier'] * finger_angles['index'] + 
                                   cal['index']['offset_right']))
        self.set_servo_position(2, channels['middle'], 
                               int(cal['middle']['multiplier'] * finger_angles['middle'] + 
                                   cal['middle']['offset']))
        self.set_servo_position(2, channels['ring'], 
                               int(cal['ring']['multiplier'] * finger_angles['ring'] + 
                                   cal['ring']['offset_right']))
        self.set_servo_position(2, channels['little'], 
                               int(cal['little']['multiplier'] * finger_angles['little'] + 
                                   cal['little']['offset_right']))
        
        # Left arm (when right hand is detected)
        if left_arm_positions:
            defaults = self.config['default_positions']['no_hand']['left_arm_when_right_detected']
            self.set_servo_position(2, channels['elbow'], 
                                   left_arm_positions.get('elbow', defaults['elbow']))
            self.set_servo_position(2, channels['lower_arm_rotation'], 
                                   left_arm_positions.get('lower_arm_rotation', defaults['lower_arm_rotation']))
    
    def disconnect(self):
        """Disconnects from servo bricks."""
        self.ipcon.disconnect()
