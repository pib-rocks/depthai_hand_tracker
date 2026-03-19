"""
Unit-Tests für ServoController
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import yaml
from pathlib import Path
from servo_controller import ServoController


class TestServoController(unittest.TestCase):
    """Test-Klasse für ServoController."""
    
    def setUp(self):
        """Setzt Test-Daten auf."""
        # Erstelle Mock-Konfiguration
        self.mock_config = {
            'tinkerforge': {
                'host': 'localhost',
                'port': 4223,
                'servo_bricks': [
                    {'uid': '29q6'},
                    {'uid': '29pF'},
                    {'uid': '29G8'}
                ]
            },
            'servo': {
                'pulse_width': {'min': 700, 'max': 2500},
                'motion_config': {
                    'velocity': 9000,
                    'acceleration': 9000,
                    'deceleration': 9000
                }
            },
            'calibration': {
                'finger_angles': {
                    'thumb_stretch': {'multiplier': 5000, 'offset': -9000},
                    'thumb_opposition': {'multiplier': 5000, 'offset_left': -9000, 'offset_right': -6000},
                    'index': {'multiplier': 5000, 'offset_left': -6000, 'offset_right': -9000},
                    'middle': {'multiplier': 5000, 'offset': -9000},
                    'ring': {'multiplier': 5000, 'offset_left': -9000, 'offset_right': -6000},
                    'little': {'multiplier': 5000, 'offset_left': -9000, 'offset_right': -6000}
                }
            },
            'default_positions': {
                'no_hand': {
                    'shoulder_vertical': {'left': 9000, 'right': -9000},
                    'elbow': {'left': 4500, 'right': 5000},
                    'lower_arm_rotation': {'left': 0, 'right': 0},
                    'shoulder_horizontal': {'left': 0, 'right': 0},
                    'right_arm_when_left_detected': {'elbow': -6000, 'lower_arm_rotation': -7500},
                    'left_arm_when_right_detected': {'elbow': -5000, 'lower_arm_rotation': 7000}
                }
            },
            'servo_channels': {
                'left_hand': {
                    'thumb_stretch': 1,
                    'thumb_opposition': 0,
                    'index': 2,
                    'middle': 3,
                    'ring': 4,
                    'little': 5,
                    'elbow': 8,
                    'lower_arm_rotation': 7,
                    'shoulder_horizontal': 9
                },
                'right_hand': {
                    'thumb_stretch': 1,
                    'thumb_opposition': 0,
                    'index': 2,
                    'middle': 3,
                    'ring': 4,
                    'little': 5,
                    'elbow': 8,
                    'lower_arm_rotation': 7,
                    'shoulder_horizontal': 9
                },
                'shoulder': {
                    'vertical_left': 1,
                    'vertical_right': 9
                }
            }
        }
    
    @patch('servo_controller.IPConnection')
    @patch('servo_controller.BrickletServoV2')
    @patch('builtins.open', create=True)
    def test_init(self, mock_open, mock_bricklet, mock_ipcon):
        """Testet die Initialisierung des ServoControllers."""
        # Mock YAML-Datei
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.read.return_value = yaml.dump(self.mock_config)
        mock_open.return_value = mock_file
        
        # Mock IPConnection und Bricklet
        mock_ipcon_instance = Mock()
        mock_ipcon.return_value = mock_ipcon_instance
        
        mock_bricklet_instance = Mock()
        mock_bricklet.return_value = mock_bricklet_instance
        
        controller = ServoController(config_path="test_config.yaml")
        
        # Prüfe, dass Verbindung hergestellt wurde
        mock_ipcon_instance.connect.assert_called_once_with('localhost', 4223)
        self.assertEqual(len(controller.servo_bricks), 3)
    
    @patch('servo_controller.IPConnection')
    @patch('servo_controller.BrickletServoV2')
    @patch('builtins.open', create=True)
    def test_set_servo_position(self, mock_open, mock_bricklet, mock_ipcon):
        """Testet das Setzen einer Servo-Position."""
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.read.return_value = yaml.dump(self.mock_config)
        mock_open.return_value = mock_file
        
        mock_ipcon_instance = Mock()
        mock_ipcon.return_value = mock_ipcon_instance
        
        mock_bricklet_instance = Mock()
        mock_bricklet.return_value = mock_bricklet_instance
        
        controller = ServoController(config_path="test_config.yaml")
        
        # Setze Position
        controller.set_servo_position(0, 1, 5000)
        
        # Prüfe, dass Methoden aufgerufen wurden
        mock_bricklet_instance.set_pulse_width.assert_called()
        mock_bricklet_instance.set_position.assert_called_with(1, 5000)
        mock_bricklet_instance.set_motion_configuration.assert_called()
        mock_bricklet_instance.set_enable.assert_called_with(1, True)
    
    @patch('servo_controller.IPConnection')
    @patch('servo_controller.BrickletServoV2')
    @patch('builtins.open', create=True)
    def test_set_default_positions(self, mock_open, mock_bricklet, mock_ipcon):
        """Testet das Setzen der Standard-Positionen."""
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.read.return_value = yaml.dump(self.mock_config)
        mock_open.return_value = mock_file
        
        mock_ipcon_instance = Mock()
        mock_ipcon.return_value = mock_ipcon_instance
        
        mock_bricklet_instance = Mock()
        mock_bricklet.return_value = mock_bricklet_instance
        
        controller = ServoController(config_path="test_config.yaml")
        
        # Setze Standard-Positionen
        controller.set_default_positions()
        
        # Prüfe, dass mehrere Positionen gesetzt wurden
        self.assertGreater(mock_bricklet_instance.set_position.call_count, 0)
    
    @patch('servo_controller.IPConnection')
    @patch('servo_controller.BrickletServoV2')
    @patch('builtins.open', create=True)
    def test_control_left_hand(self, mock_open, mock_bricklet, mock_ipcon):
        """Testet die Steuerung der linken Hand."""
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.read.return_value = yaml.dump(self.mock_config)
        mock_open.return_value = mock_file
        
        mock_ipcon_instance = Mock()
        mock_ipcon.return_value = mock_ipcon_instance
        
        mock_bricklet_instance = Mock()
        mock_bricklet.return_value = mock_bricklet_instance
        
        controller = ServoController(config_path="test_config.yaml")
        
        finger_angles = {
            'thumb_stretch': 1.0,
            'thumb_opposition': 1.0,
            'index': 1.0,
            'middle': 1.0,
            'ring': 1.0,
            'little': 1.0
        }
        
        controller.control_left_hand(1000, 2000, finger_angles)
        
        # Prüfe, dass mehrere Servos gesetzt wurden
        self.assertGreater(mock_bricklet_instance.set_position.call_count, 5)


if __name__ == '__main__':
    unittest.main()
