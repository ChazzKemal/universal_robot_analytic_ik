#!usr/bin/env python
import numpy as np
import moveit_commander
from geometry_msgs.msg import Pose

# Function to compute the Modified Denavit-Hartenberg (MDH) transformation matrix
def MDHMatrix(DHparams):
    c1 = np.cos(np.deg2rad(DHparams[3]))
    s1 = np.sin(np.deg2rad(DHparams[3]))
    c2 = np.cos(np.deg2rad(DHparams[0]))
    s2 = np.sin(np.deg2rad(DHparams[0]))
    T = np.array([[c1, -s1, 0, DHparams[1]],
                  [s1*c2, c1*c2, -s2, -s2*DHparams[2]],
                  [s1*s2, c1*s2, c2, c2*DHparams[2]],
                  [0, 0, 0, 1]])
    return T

# Function to compute the joint 4 angle using inverse trigonometric functions
def joint4(d, a, theta2, theta3, T_06, T_01, T_64):
    T_12 = MDHMatrix([90, 0, d[1], np.rad2deg(theta2)])
    T_23 = MDHMatrix([0, a[2], d[3], np.rad2deg(theta3)])
    T_03 = T_01 @ T_12 @ T_23
    T_36 = np.linalg.inv(T_03) @ T_06
    T_34 = T_36 @ T_64
    x_34 = T_34[:, 0]
    joint4 = np.arctan2(x_34[1], x_34[0])
    return joint4

# Function to compute the forward kinematics
def fwdKin(DHMatrix):
    numFrames = DHMatrix.shape[0]
    iT = np.zeros((numFrames, 4, 4))  # Initialize the transformation matrices
    for i in range(numFrames):
        iT[i] = MDHMatrix(DHMatrix[i])  # Compute the MDH transformation matrices
    T0 = np.zeros((numFrames, 4, 4))
    T0[0] = iT[0]
    for i in range(1, numFrames):
        T0[i] = T0[i-1] @ iT[i]  # Compute the cumulative transformation matrices
    return [iT, T0]

# Function to compute the inverse kinematics with 8 solutions
def invKin8sol(d, a, eePosOri):
    ikSol = 8
    numJoints = 6
    joint = np.zeros((ikSol, numJoints))  # Initialize the joint angles array

    P_05 = eePosOri @ np.array([0, 0, -d[5], 1])
    phi1 = np.arctan2(P_05[1], P_05[0])
    phi2 = np.arccos(d[3] / np.sqrt(P_05[1]**2 + P_05[0]**2))

    for i in range(4, 8):
        joint[i, 0] = (np.pi/2 + phi1 - phi2)
    for i in range(4):
        joint[i, 0] = (np.pi/2 + phi1 + phi2)

    T_06 = eePosOri
    P_06 = T_06[:, 3]

    for j in range(8):
        T_01 = MDHMatrix([0, 0, d[0], np.rad2deg(joint[j, 0])])
        T_16 = np.linalg.inv(T_01) @ T_06
        P_16 = T_16[:, 3]
        
        if j in [0, 1, 4, 5]:
            joint[j, 4] = np.arccos((P_06[0]*np.sin(joint[j, 0]) - P_06[1]*np.cos(joint[j, 0]) - d[3]) / d[5])
        else:
            joint[j, 4] = -np.arccos((P_06[0]*np.sin(joint[j, 0]) - P_06[1]*np.cos(joint[j, 0]) - d[3]) / d[5])

        T_61 = np.linalg.inv(T_16)
        Y_16 = T_61[:, 1]
        X_60 = np.linalg.inv(T_06)[:, 0]
        Y_60 = np.linalg.inv(T_06)[:, 1]

        if int(np.rad2deg(np.real(joint[j, 4]))) == 0 or int(np.rad2deg(np.real(joint[j, 4]))) == 2*np.pi:
            joint[j, 5] = np.deg2rad(0)
        else:
            joint[j, 5] = np.arctan2((-X_60[1]*np.sin(joint[j, 0]) + Y_60[1]*np.cos(joint[j, 0])) / np.sin(joint[j, 4]),
                                     (X_60[0]*np.sin(joint[j, 0]) - Y_60[0]*np.cos(joint[j, 0])) / np.sin(joint[j, 4]))

        T_45 = MDHMatrix([90, a[4], d[4], np.rad2deg(joint[j, 4])])
        T_56 = MDHMatrix([-90, a[5], d[5], np.rad2deg(joint[j, 5])])
        T_46 = T_45 @ T_56
        T_64 = np.linalg.inv(T_46)
        T_14 = T_16 @ T_64

        P_14 = T_14[:, 3]
        P_14_xz = np.sqrt(P_14[0]**2 + P_14[2]**2)

        if j % 2 == 0:
            phi3 = np.arccos((P_14_xz**2 - a[3]**2 - a[2]**2) / (2*a[2]*a[3]))
            joint[j, 2] = np.arccos((P_14_xz**2 - a[3]**2 - a[2]**2) / (2*a[2]*a[3]))
            if joint[j, 2] > np.pi:
                joint[j, 2] = joint[j, 2] - np.pi*2
            joint[j, 1] = round((np.arctan2(-P_14[2], -P_14[0])) - np.arcsin((-a[3]*np.sin(joint[j, 2])) / P_14_xz), 2)
            joint[j, 3] = joint4(d, a, joint[j, 1], joint[j, 2], T_06, T_01, T_64)
        else:
            phi3 = np.arccos((P_14_xz**2 - a[3]**2 - a[2]**2) / (-2*a[2]*a[3]))
            joint[j, 2] = -np.arccos((P_14_xz**2 - a[3]**2 - a[2]**2) / (2*a[2]*a[3]))
            if joint[j, 2] > np.pi:
                joint[j, 2] = joint[j, 2] - np.pi*2
            joint[j, 1] = round((np.arctan2(-P_14[2], -P_14[0])) - np.arcsin((-a[3]*np.sin(joint[j, 2])) / P_14_xz), 2)
            joint[j, 3] = joint4(d, a, joint[j, 1], joint[j, 2], T_06, T_01, T_64)

    return joint

if __name__ == "__main__":
    dof = 6
    d = np.zeros(dof)  # Array to store the link offsets (d values)
    a = np.zeros(dof)  # Array to store the link lengths (a values)
    theta = np.zeros(dof)  # Array to store the joint angles (theta values)
    totalIKsol = 8

    alpha = np.array([0, 90, 0, 0, 90, -90])
    a[2] = -0.612
    a[3] = -0.5723
    d[0] = 0.1273
    d[4] = 0.1157
    d[5] = 0.0922

    theta[0] = 35
    theta[1] = 0
    theta[2] = 90
    theta[3] = 50
    theta[4] = 30
    theta[5] = 20

    DHMatrix = np.column_stack((alpha, a, d, theta))
    M = fwdKin(DHMatrix)

    target_points = [(-0.5, -0.3, 0.5), (-0.3, -0.2, 0.5), (0.5, 0.3, 0.2)]

    moveit_commander.roscpp_initialize(sys.argv)
    robot = moveit_commander.RobotCommander()
    move_group = moveit_commander.MoveGroupCommander("manipulator")

    for target_point in target_points:
        target_pos = np.identity(4)
        target_pos[:, 3] = [target_point[0], target_point[1], target_point[2], 1]
        joints = invKin8sol(d, a, target_pos)
        selected_angle = 2
        joint_goal = move_group.get_current_joint_values()

        # Set the joint angles for the selected solution
        joint_goal[:6] = joints[selected_angle]

        move_group.go(joint_goal, wait=True)

    move_group.stop()
    moveit_commander.roscpp_shutdown()








    



