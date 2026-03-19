"""
Test-Suite für alle Module
Führt alle Unit-Tests aus
"""

import unittest
import sys
from test_hand_angle_calculator import TestHandAngleCalculator
from test_servo_controller import TestServoController


def run_tests():
    """Führt alle Tests aus."""
    # Erstelle Test-Suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Füge alle Test-Klassen hinzu
    suite.addTests(loader.loadTestsFromTestCase(TestHandAngleCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestServoController))
    
    # Führe Tests aus
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Gebe Ergebnis zurück
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
