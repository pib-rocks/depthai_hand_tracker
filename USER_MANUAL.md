# User Manual - Hand Tracker with Servo Control

## Overview

This system enables the imitation of hand and finger movements on a robot through hand tracking with a DepthAI camera and control of Tinkerforge servo motors.

## Prerequisites

### Hardware
- DepthAI camera (OAK-D, OAK-1, etc.)
- Tinkerforge Servo V2 Bricklets (at least 3)
- Tinkerforge HAT or Master Brick
- USB connection to camera
- Network connection to Tinkerforge (localhost or IP)

### Software
- Python 3.7 or higher
- DepthAI library
- Tinkerforge Python API
- PyYAML
- NumPy
- OpenCV

### Installing Dependencies

```bash
pip install depthai numpy opencv-python pyyaml tinkerforge
```

## Configuration

### Configuration File (config.yaml)

All configuration is done through the `config.yaml` file. This file contains:

1. **Tinkerforge Connection Settings**
   - Host and port of Tinkerforge connection
   - UIDs of servo bricklets

2. **Servo Configuration**
   - Pulse width settings (Min/Max)
   - Motion configuration (velocity, acceleration, deceleration)

3. **Calibration Values**
   - Multipliers and offsets for hand position to servo position conversion
   - Angle to servo position conversion

4. **Default Positions**
   - Positions when no hand is detected

5. **Servo Channel Assignment**
   - Which servo channel is assigned to which finger/arm part

### Getting Started

1. **Configure Tinkerforge Connection**

   Open `config.yaml` and adjust the settings:

   ```yaml
   tinkerforge:
     host: "localhost"  # Or IP address of Tinkerforge master
     port: 4223
     servo_bricks:
       - uid: "29q6"  # Enter your UIDs here
       - uid: "29pF"
       - uid: "29G8"
   ```

   **Important:** The UIDs must match the actual UIDs of your servo bricklets. You can find these in the Tinkerforge Brick Viewer.

2. **Check Servo Channel Assignment**

   Make sure the channel assignments in `config.yaml` match your hardware installation:

   ```yaml
   servo_channels:
     left_hand:
       thumb_stretch: 1
       thumb_opposition: 0
       # ... etc.
   ```

## Calibration

Calibration is the most important step for precise control. You need to adjust the values in `config.yaml` to match your specific hardware and mounting.

### Shoulder Calibration

Shoulder positions are calculated from hand landmark coordinates:

```yaml
calibration:
  shoulder:
    horizontal:
      left:
        multiplier: 13    # Adjust for horizontal movement
        offset: -10000   # Adjust for zero position
      right:
        multiplier: 13
        offset: -4000
    vertical:
      left:
        multiplier: 13
        offset_y: -2000
        base: 5000
      right:
        multiplier: 13
        offset: -4000
```

**Calibration Process:**

1. Start the program and hold your hand in a neutral position
2. Observe the servo positions
3. Adjust `multiplier` to change sensitivity:
   - Larger value = more sensitive
   - Smaller value = less sensitive
4. Adjust `offset` to shift the zero position
5. Repeat the process until movements are correct

### Finger Angle Calibration

Finger angles are converted to servo positions:

```yaml
calibration:
  finger_angles:
    thumb_stretch:
      multiplier: 5000   # Adjust for stretch
      offset: -9000      # Adjust for closed position
    index:
      multiplier: 5000
      offset_left: -6000
      offset_right: -9000
```

**Calibration Process:**

1. Open your hand completely
2. Observe the servo positions
3. Adjust `multiplier` to change movement range
4. Adjust `offset` to correct closed position
5. Test various hand poses

### Default Positions

When no hand is detected, servos go to default positions:

```yaml
default_positions:
  no_hand:
    shoulder_vertical:
      left: 9000
      right: -9000
    elbow:
      left: 4500
      right: 5000
```

Adjust these values to define the robot's "rest position".

## Usage

### Basic Usage

```python
from HandTrackerEdge import HandTracker

# Initialize tracker
tracker = HandTracker(
    enable_servo_control=True,  # Enable servo control
    config_path="config.yaml"    # Path to configuration
)

# Main loop
try:
    while True:
        frame, hands, _ = tracker.next_frame()
        
        # Display frame (optional)
        cv2.imshow("Hand Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    tracker.exit()
```

### Disable Servo Control

If you want to test hand tracking without servo control:

```python
tracker = HandTracker(
    enable_servo_control=False
)
```

### Advanced Options

```python
tracker = HandTracker(
    lm_model="lite",              # "full", "lite" or "sparse"
    resolution="full",             # "full" or "ultra"
    internal_frame_height=640,    # Frame height
    use_gesture=True,              # Enable gesture recognition
    stats=True,                    # Display statistics
    trace=1                        # Debug output (0-15)
)
```

## Troubleshooting

### Problem: Servo Bricklets Not Found

**Solution:**
1. Check the UIDs in `config.yaml`
2. Use the Tinkerforge Brick Viewer to find the UIDs
3. Make sure the Tinkerforge connection works:
   ```python
   from tinkerforge.ip_connection import IPConnection
   ipcon = IPConnection()
   ipcon.connect("localhost", 4223)
   ```

### Problem: Movements Too Sensitive/Insensitive

**Solution:**
- Adjust the `multiplier` values in calibration
- Larger multiplier = more sensitive
- Smaller multiplier = less sensitive

### Problem: Fingers Don't Move Correctly

**Solution:**
1. Check the servo channel assignment in `config.yaml`
2. Verify hardware wiring
3. Adjust `offset` values for each finger

### Problem: Hand Not Detected

**Solution:**
1. Make sure the camera is connected correctly
2. Check lighting
3. Hold your hand clearly visible in front of the camera
4. Enable debug output: `trace=1`

## Code Structure

### Modules

- **HandTrackerEdge.py**: Main class for hand tracking
- **servo_controller.py**: Servo motor management
- **hand_angle_calculator.py**: Finger angle calculation
- **config.yaml**: Configuration file

### Data Flow

1. **Hand Tracking**: DepthAI camera detects hand landmarks
2. **Angle Calculation**: `HandAngleCalculator` calculates finger angles
3. **Servo Control**: `ServoController` converts angles to servo positions
4. **Execution**: Servos move accordingly

## Advanced Configuration

### Adjusting Servo Speed

```yaml
servo:
  motion_config:
    velocity: 9000      # Higher = faster
    acceleration: 9000  # Higher = faster acceleration
    deceleration: 9000  # Higher = faster deceleration
```

### Adjusting Pulse Width

```yaml
servo:
  pulse_width:
    min: 700   # Minimum in microseconds
    max: 2500  # Maximum in microseconds
```

**Important:** These values depend on your specific servo motors. Consult the servo documentation.

## Tips for Best Results

1. **Good Lighting**: Make sure your hands are well lit
2. **Stable Camera**: The camera should be firmly mounted
3. **Contrast**: Wear clothing that contrasts with your skin color
4. **Calibration**: Take time for calibration - it's worth it!
5. **Testing**: Test various hand poses and adjust configuration

## Support

For problems:
1. Check the troubleshooting section above
2. Enable debug output (`trace=1` or higher)
3. Check the configuration file for syntax errors
4. Make sure all dependencies are installed

## License

See LICENSE.txt for details.
