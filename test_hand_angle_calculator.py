"""
Unit-Tests für HandAngleCalculator
"""

import unittest
import numpy as np
import math
from hand_angle_calculator import HandAngleCalculator


class TestHandAngleCalculator(unittest.TestCase):
    """Test-Klasse für HandAngleCalculator."""
    
    def setUp(self):
        """Setzt Test-Daten auf."""
        # Erstelle Beispiel-Landmarks (21 Punkte, 3D)
        self.norm_landmarks = np.array([
            [0.5, 0.5, 0.0],  # Wrist
            [0.4, 0.5, 0.0],  # Thumb CMC
            [0.3, 0.5, 0.0],  # Thumb MCP
            [0.2, 0.5, 0.0],  # Thumb IP
            [0.1, 0.5, 0.0],  # Thumb TIP
            [0.6, 0.4, 0.0],  # Index MCP
            [0.6, 0.3, 0.0],  # Index PIP
            [0.6, 0.2, 0.0],  # Index DIP
            [0.6, 0.1, 0.0],  # Index TIP
            [0.7, 0.4, 0.0],  # Middle MCP
            [0.7, 0.3, 0.0],  # Middle PIP
            [0.7, 0.2, 0.0],  # Middle DIP
            [0.7, 0.1, 0.0],  # Middle TIP
            [0.8, 0.4, 0.0],  # Ring MCP
            [0.8, 0.3, 0.0],  # Ring PIP
            [0.8, 0.2, 0.0],  # Ring DIP
            [0.8, 0.1, 0.0],  # Ring TIP
            [0.9, 0.4, 0.0],  # Pinky MCP
            [0.9, 0.3, 0.0],  # Pinky PIP
            [0.9, 0.2, 0.0],  # Pinky DIP
            [0.9, 0.1, 0.0],  # Pinky TIP
        ])
    
    def test_calculate_angle_between_vectors(self):
        """Testet die Winkelberechnung zwischen Vektoren."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        angle = HandAngleCalculator.calculate_angle_between_vectors(vec1, vec2)
        self.assertAlmostEqual(angle, math.pi / 2, places=5)
        
        # Parallele Vektoren
        vec3 = np.array([2, 0, 0])
        angle2 = HandAngleCalculator.calculate_angle_between_vectors(vec1, vec3)
        self.assertAlmostEqual(angle2, 0.0, places=5)
        
        # Null-Vektor
        vec4 = np.array([0, 0, 0])
        angle3 = HandAngleCalculator.calculate_angle_between_vectors(vec1, vec4)
        self.assertEqual(angle3, 0.0)
    
    def test_calculate_finger_angles(self):
        """Testet die Berechnung aller Fingerwinkel."""
        angles = HandAngleCalculator.calculate_finger_angles(self.norm_landmarks)
        
        # Prüfe, dass alle Winkel vorhanden sind
        self.assertIn('thumb_stretch', angles)
        self.assertIn('thumb_opposition', angles)
        self.assertIn('index', angles)
        self.assertIn('middle', angles)
        self.assertIn('ring', angles)
        self.assertIn('little', angles)
        
        # Prüfe, dass Winkel im gültigen Bereich sind (0 bis π)
        for angle_name, angle_value in angles.items():
            self.assertGreaterEqual(angle_value, 0.0)
            self.assertLessEqual(angle_value, math.pi)
    
    def test_calculate_shoulder_positions(self):
        """Testet die Berechnung der Schulterpositionen."""
        landmarks_2d = self.norm_landmarks[:, :2] * 640  # Konvertiere zu Pixel-Koordinaten
        
        calibration = {
            'shoulder': {
                'horizontal': {
                    'left': {'multiplier': 13, 'offset': -10000},
                    'right': {'multiplier': 13, 'offset': -4000}
                },
                'vertical': {
                    'left': {'multiplier': 13, 'offset_x': 0, 'offset_y': -2000, 'base': 5000},
                    'right': {'multiplier': 13, 'offset': -4000}
                }
            }
        }
        
        # Linke Hand
        h_left, v_left = HandAngleCalculator.calculate_shoulder_positions(
            landmarks_2d, "left", calibration
        )
        self.assertIsInstance(h_left, int)
        self.assertIsInstance(v_left, int)
        
        # Rechte Hand
        h_right, v_right = HandAngleCalculator.calculate_shoulder_positions(
            landmarks_2d, "right", calibration
        )
        self.assertIsInstance(h_right, int)
        self.assertIsInstance(v_right, int)
    
    def test_calculate_angles_for_hands_empty(self):
        """Testet die Winkelberechnung mit leerer Handliste."""
        angle_dicts, angle_array = HandAngleCalculator.calculate_angles_for_hands([], {})
        self.assertEqual(angle_dicts, [])
        self.assertEqual(angle_array.size, 0)


if __name__ == '__main__':
    unittest.main()
