import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3
import pib_DH
from roboticstoolbox.backends.PyPlot import PyPlot

Arm_R = pib_DH.pib_right()
Arm_L = pib_DH.pib_left()
print(Arm_R)
print(Arm_L)
env=PyPlot()
env.launch()
env.add(Arm_R)
env.add(Arm_L)

def get_joint_angles():
    n = Arm_R.n 
    print("\nType L for left arm and R for right arm")
    flag = input()
    print("\nEnter joint angles in degrees:")
    joint_angles = []
    for i in range(n):
        angle = float(input(f"Joint {i+1} angle (degrees): "))
        joint_angles.append(np.radians(angle))
    q = np.array(joint_angles)
    return q, flag
q, flag = get_joint_angles()

if flag == 'R':
    T = Arm_R.fkine(q)  
    print("\nR-End-effector pose calculated from forward kinematics (Transformation Matrix):")
    print(T)
    rpy = T.rpy(order='xyz', unit='deg')
    print("\nEnd-effector orientation in Roll-Pitch-Yaw angles (degrees):")
    print(f"Roll: {rpy[0]:.2f}°, Pitch: {rpy[1]:.2f}°, Yaw: {rpy[2]:.2f}°")
    Arm_R.q = q
    while True:    
        env.step()
if flag == 'L':
    T = Arm_L.fkine(q)  
    print("\nL-End-effector pose calculated from forward kinematics (Transformation Matrix):")
    print(T)
    rpy = T.rpy(order='xyz', unit='deg')
    print("\nEnd-effector orientation in Roll-Pitch-Yaw angles (degrees):")
    print(f"Roll: {rpy[0]:.2f}°, Pitch: {rpy[1]:.2f}°, Yaw: {rpy[2]:.2f}°")
    Arm_L.q = q
    while True:
        env.step()
