import casadi
import meshcat.geometry as mg
import math
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
from tf_transformations import quaternion_from_euler, euler_from_quaternion, quaternion_from_matrix
import os
import sys
import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import argparse
import threading
from ament_index_python.packages import get_package_share_directory


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


def matrix_to_xyzrpy(matrix):
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]
    roll = math.atan2(matrix[2, 1], matrix[2, 2])
    pitch = math.asin(-matrix[2, 0])
    yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return [x, y, z, roll, pitch, yaw]


def create_transformation_matrix(x, y, z, roll, pitch, yaw):
    transformation_matrix = np.eye(4)
    A = np.cos(yaw)
    B = np.sin(yaw)
    C = np.cos(pitch)
    D = np.sin(pitch)
    E = np.cos(roll)
    F = np.sin(roll)
    DE = D * E
    DF = D * F
    transformation_matrix[0, 0] = A * C
    transformation_matrix[0, 1] = A * DF - B * E
    transformation_matrix[0, 2] = B * F + A * DE
    transformation_matrix[0, 3] = x
    transformation_matrix[1, 0] = B * C
    transformation_matrix[1, 1] = A * E + B * DF
    transformation_matrix[1, 2] = B * DE - A * F
    transformation_matrix[1, 3] = y
    transformation_matrix[2, 0] = -D
    transformation_matrix[2, 1] = C * F
    transformation_matrix[2, 2] = C * E
    transformation_matrix[2, 3] = z
    transformation_matrix[3, 0] = 0
    transformation_matrix[3, 1] = 0
    transformation_matrix[3, 2] = 0
    transformation_matrix[3, 3] = 1
    return transformation_matrix


class Arm_FK:
    def __init__(self, args):
        self.args = args

        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        package_path = get_package_share_directory('piper_description')
        urdf_path = os.path.join(package_path, 'urdf', 'piper_description_old' + ('-lift.urdf' if args.lift else '.urdf'))

        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path,
            package_dirs=package_path
        )

        # self.robot = pin.RobotWrapper.BuildFromURDF(urdf_path)

        self.mixed_jointsToLockIDs = ["joint7",
                                      "joint8"
                                      ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0] * self.robot.model.nq),
        )
        # self.first_matrix = create_transformation_matrix(0, 0, 0, 0, -1.57, -1.57)
        self.first_matrix = create_transformation_matrix(0, 0, 0, 0, -1.57, 0)
        self.second_matrix = create_transformation_matrix(self.args.gripper_xyzrpy[0], self.args.gripper_xyzrpy[1], self.args.gripper_xyzrpy[2],
                                                          self.args.gripper_xyzrpy[3], self.args.gripper_xyzrpy[4], self.args.gripper_xyzrpy[5])
        self.last_matrix = np.dot(self.first_matrix, self.second_matrix)
        q = quaternion_from_matrix(self.last_matrix)
        self.reduced_robot.model.addFrame(
            pin.Frame('ee',
                      self.reduced_robot.model.getJointId('joint6'),
                      pin.SE3(
                          pin.Quaternion(q[3], q[0], q[1], q[2]),
                          np.array([self.last_matrix[0, 3], self.last_matrix[1, 3], self.last_matrix[2, 3]]),
                      ),
                      pin.FrameType.OP_FRAME)
        )

    def get_pose(self, q):
        index = 6 + (1 if self.args.lift else 0)
        pin.forwardKinematics(self.reduced_robot.model, self.reduced_robot.data, np.concatenate([q], axis=0))
        end_pose = create_transformation_matrix(self.reduced_robot.data.oMi[index].translation[0], self.reduced_robot.data.oMi[index].translation[1], self.reduced_robot.data.oMi[index].translation[2],
                                                math.atan2(self.reduced_robot.data.oMi[index].rotation[2, 1], self.reduced_robot.data.oMi[index].rotation[2, 2]),
                                                math.asin(-self.reduced_robot.data.oMi[index].rotation[2, 0]),
                                                math.atan2(self.reduced_robot.data.oMi[index].rotation[1, 0], self.reduced_robot.data.oMi[index].rotation[0, 0]))
        end_pose = np.dot(end_pose, self.last_matrix)
        return matrix_to_xyzrpy(end_pose)


class RosOperator(Node):
    def __init__(self, args):
        super().__init__(f'piper_FK{args.index_name}')
        self.args = args
        self.lift_subscriber = None
        self.arm_end_pose_publisher = None
        self.arm_end_pose_orient_publisher = None
        self.arm_joint_state_subscriber = None
        self.arm_msg = None
        self.lift_msg = None
        self.calc_thread = None
        self.arm_fk = Arm_FK(args)
        self.init_ros()

    def lift_callback(self, msg):
        self.lift_msg = msg

    def arm_joint_state_callback(self, msg):
        self.arm_msg = msg

    def start(self):
        self.calc_thread = threading.Thread(target=self.calc)
        self.calc_thread.start()

    def calc(self):
        rate = self.create_rate(200)
        while rclpy.ok():
            if (self.args.lift and self.lift_msg is None) or self.arm_msg is None:
                rate.sleep()
                continue
            xyzrpy = self.arm_fk.get_pose((self.lift_msg.position if self.args.lift else []) + list(self.arm_msg.position[:6]))
            end_pose_msg = PoseStamped()
            end_pose_msg.header = Header()
            end_pose_msg.header.stamp = self.get_clock().now().to_msg()
            end_pose_msg.header.frame_id = "map"
            end_pose_msg.pose.position.x = xyzrpy[0]
            end_pose_msg.pose.position.y = xyzrpy[1]
            end_pose_msg.pose.position.z = xyzrpy[2]
            end_pose_msg.pose.orientation.x = xyzrpy[3]
            end_pose_msg.pose.orientation.y = xyzrpy[4]
            end_pose_msg.pose.orientation.z = xyzrpy[5]
            end_pose_msg.pose.orientation.w = self.arm_msg.position[6] if len(self.arm_msg.position) > 6 else 0.0
            self.arm_end_pose_publisher.publish(end_pose_msg)
            x, y, z, w = quaternion_from_euler(end_pose_msg.pose.orientation.x, end_pose_msg.pose.orientation.y, end_pose_msg.pose.orientation.z)
            end_pose_msg.pose.orientation.x = x
            end_pose_msg.pose.orientation.y = y
            end_pose_msg.pose.orientation.z = z
            end_pose_msg.pose.orientation.w = w
            self.arm_end_pose_orient_publisher.publish(end_pose_msg)
            # print("end_pose:", xyzrpy)
            rate.sleep()

    def init_ros(self):
        if self.args.lift:
            self.lift_subscriber = self.create_subscription(JointState, f'/joint_states_single_lift', self.lift_callback, 1)
        self.arm_joint_state_subscriber = self.create_subscription(JointState, f'/joint_states_single_gripper{self.args.index_name}', self.arm_joint_state_callback, 1)
        self.arm_end_pose_publisher = self.create_publisher(PoseStamped, f'/piper_FK{self.args.index_name}/urdf_end_pose', 1)
        self.arm_end_pose_orient_publisher = self.create_publisher(PoseStamped, f'/piper_FK{self.args.index_name}/urdf_end_pose_orient', 1)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lift', action='store', type=bool, help='lift',
                        default=False, required=False)
    parser.add_argument('--index_name', action='store', type=str, help='index_name',
                        default="", required=False)
    parser.add_argument('--gripper_xyzrpy', action='store', nargs='+', type=float, help='gripper_xyzrpy',
                        default=[0.19, 0, 0, 0, 0, 0], required=False)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args


def main():
    args = get_arguments()
    rclpy.init()
    ros_operator = RosOperator(args)
    ros_operator.start()
    rclpy.spin(ros_operator)


if __name__ == "__main__":
    main()
