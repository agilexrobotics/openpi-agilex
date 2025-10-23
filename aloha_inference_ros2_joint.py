# -- coding: UTF-8
import numpy as np
import argparse
from einops import rearrange
from collections import deque
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header, String
from sensor_msgs.msg import JointState, Image, CameraInfo
from cv_bridge import CvBridge
import threading
import cv2
from functools import partial
import sys
import math

sys.path.append("./packages/openpi-client/src")
from openpi_client import websocket_client_policy, image_tools
import tree
import os
from datetime import datetime 
import pickle

sys.path.append("./")

inference_thread = None
inference_lock = threading.Lock()
inference_actions = None
inference_timestep = None
inference_log_package = None
pre_instruction = ""
pre_instruction_attention_vector = None
pre_instruction_input_ids = None
pre_instruction_attention_mask = None


class BlockingDeque:
    def __init__(self):
        self.deque = deque()
        self.not_empty = threading.Condition()

    def append(self, item):
        with self.not_empty:
            self.deque.append(item)
            self.not_empty.notify()

    def popleft(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque.popleft()
        return item

    def left(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque[0]
            return item

    def right(self):
        with self.not_empty:
            while len(self.deque) == 0:
                self.not_empty.wait()
            item = self.deque[-1]
            return item

    def size(self):
        with self.not_empty:
            return len(self.deque)


def get_camera_color(camera_names, history_num, camera_color):
    colors = []
    for cam_name in camera_names:
        for i in range(history_num):
            color = camera_color[cam_name][i]
            color = cv2.imencode('.jpg', color)[1].tobytes()
            color = cv2.imdecode(np.frombuffer(color, np.uint8), cv2.IMREAD_COLOR)
            default_width = 640
            default_height = 480
            camera_width = color.shape[1]
            camera_height = color.shape[0]
            width_diff = default_width - camera_width
            height_diff = default_height - camera_height
            if width_diff < 0:
                clip_width = abs(width_diff) // 2
                color = color[:, clip_width:clip_width + default_width]
            elif width_diff > 0:
                add_width = width_diff // 2
                top, bottom, left, right = 0, 0, add_width, add_width
                color = cv2.copyMakeBorder(color, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            if height_diff < 0:
                clip_height = abs(height_diff) // 2
                color = color[clip_height:clip_height + default_height, :]
            elif height_diff > 0:
                add_height = height_diff // 2
                top, bottom, left, right = add_height, add_height, 0, 0
                color = cv2.copyMakeBorder(color, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
            color = rearrange(color, 'h w c -> c h w')
            colors.append(color)
    colors = np.stack(colors, axis=0)
    colors = np.expand_dims(colors, axis=0)  # shape: [1, num_camera, ...]
    return colors  # b, num_camera, num_history, c, h, w


def normalize_gripper_dims(
    data_array, 
    dims=None,
    original_min=0.0, 
    original_max=1.0, 
    new_min=0.0, 
    new_max=0.08,
    flip=False
):
    """Normalizes specified dimensions (defaults to gripper) to a new interval.
    If flip=True, the normalized interval is reversed (open and closed are swapped).
    By default, Agilex gripper values use 0 for fully closed and 0.08 for fully open.
    And pi0 expects 0 for fully open and 1 for fully closed gripper values.
    """
    if dims is None:
        dims = [6, 13]
        
    normalized_array = data_array.copy()
    dim_data = normalized_array[..., dims]

    scale = (new_max - new_min) / (original_max - original_min)
    normalized_dims = new_min + (dim_data - original_min) * scale

    if flip:
        normalized_dims = new_max - (normalized_dims - new_min)

    normalized_array[..., dims] = np.clip(normalized_dims, new_min, new_max)
    return normalized_array


def inference_process(args, t, ros_operator, policy):
    global inference_lock
    global inference_actions
    global inference_timestep
    global inference_log_package
    global pre_instruction
    global pre_instruction_attention_vector
    global pre_instruction_input_ids
    global pre_instruction_attention_mask
    bridge = CvBridge()

    (instruction, camera_colors, arm_joint_states) = ros_operator.get_frame()

    observation = {}

    camera_color_dict = {}
    camera_color = None
    for j in range(len(args.camera_color_names)):
        camera_color_dict[args.camera_color_names[j]] = [bridge.imgmsg_to_cv2(camera_colors[i][j], 'bgr8') for i in range(args.obs_history_num)]
    if args.use_camera_color:
        camera_color = get_camera_color(args.camera_color_names, args.obs_history_num, camera_color_dict)  # b, num_camera*history, c, h, w

    # observation["observation.state"] = []
    robot_state = []
    if args.use_arm_joint_state % 2 == 1:
        qpos_joint_state = [np.concatenate([np.array(arm_joint_states[i][j].position) for j in range(len(args.arm_joint_state_names))], axis=0)[np.newaxis, :] for i in range(args.obs_history_num)]
        qpos_joint_state = np.concatenate(qpos_joint_state, axis=0)
        robot_state.append(qpos_joint_state.astype(np.float32))  # obs_history_num, num_arm * 7
    robot_state = np.concatenate(robot_state, axis=-1)  # shape: (obs_history_num, n)

    RENDER_H, RENDER_W = 224, 224
    images = {}
    name_map = {
        "left" : "cam_left_wrist",
        "front": "cam_high",
        "right": "cam_right_wrist",
    }
    for cam_idx, cam_name in enumerate(args.camera_color_names):
        img_chw = camera_color[0, cam_idx].astype(np.uint8)
        img_hwc  = rearrange(img_chw, "c h w -> h w c")
        img_hwc  = cv2.cvtColor(img_hwc, cv2.COLOR_BGR2RGB)
        img_hwc  = image_tools.resize_with_pad(img_hwc, RENDER_H, RENDER_W)
        img_chw  = rearrange(img_hwc, "h w c -> c h w")
        images[name_map[cam_name]] = img_chw

    observation = {
        "state" : robot_state[0],   # (14,)
        "images": images,
    }
    if instruction != 'null':
        observation['prompt'] = instruction

    # Normalize gripper dims 6 and 13 in state to [0, 1] (1=closed, 0=open, flip=True)
    action_pred = policy.infer(observation)
    action = tree.map_structure(lambda x: x[:args.chunk_size, ...], action_pred['actions'])
    # NOTE: The effective range of gripper is [0, 0.08]
    action = normalize_gripper_dims(action, dims=[6, 13], original_min=0.0, original_max=1.0, new_min=0.0, new_max=0.08, flip=True)

    if args.debug_mode:
        # Store results and the log package in global variables
        # --- package the raw data for logging ---
        log_package = {
            'observation': observation,
            'action': action_pred['actions']
        }
        if instruction != 'null':
            log_package['prompt'] = instruction
    inference_lock.acquire()
    inference_actions = action[np.newaxis, :]  # 1, chunk_size, 14
    inference_timestep = t
    if args.debug_mode:
        inference_log_package = log_package
    inference_lock.release()


class RosOperator(Node):
    def __init__(self, args):
        super().__init__('inference')
        self.args = args
        self.bridge = CvBridge()

        self.instruction = args.instruction
        self.camera_color_deques = [BlockingDeque() for _ in range(len(args.camera_color_names))]
        self.arm_joint_state_deques = [BlockingDeque() for _ in range(len(args.arm_joint_state_names))]

        self.camera_color_history_list = []
        self.arm_joint_state_history_list = []

        self.instruction_subscriber = self.create_subscription(String, self.args.instruction_topic, self.instruction_callback, 1)
        self.camera_color_subscriber = [self.create_subscription(Image, self.args.camera_color_topics[i], partial(self.camera_color_callback, i), 1) for i in range(len(self.args.camera_color_names))]

        self.arm_joint_state_subscriber = [self.create_subscription(JointState, self.args.arm_joint_state_topics[i], partial(self.arm_joint_state_callback, i), 1) for i in range(len(self.args.arm_joint_state_names))]

        self.arm_joint_state_ctrl_publisher = [self.create_publisher(JointState, self.args.arm_joint_state_ctrl_topics[i], 10) for i in range(len(self.args.arm_joint_state_ctrl_topics))]

        self.inference_status = -1

        self.arm_joint_state_ctrl_thread = None
        self.arm_joint_state_ctrl_thread_return_lock = threading.Lock()
        self.arm_joint_state_ctrl_thread_return_lock.acquire()

        self.last_ctrl_arm_joint_state = None

        self.k = 3
        self.times = np.array([i for i in range(self.k)])
        self.arm_joint_state_ctrl_history_list = []

        self.model_inference_thread = threading.Thread(target=self.model_inference)
        self.model_inference_thread.start()

    def model_inference(self):
        global inference_lock
        global inference_actions
        global inference_timestep
        global inference_thread
        global inference_log_package

        # Logger Setup
        if self.args.debug_mode:
            log_dir = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(log_dir, exist_ok=True)
            print(f"Logging observation/action pickles to: {log_dir}")
        policy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8010)
        joint_state0 = [[0, 0, 0, 0, 0, 0, 0.08] for _ in range(len(self.args.arm_joint_state_ctrl_topics))]
        joint_state1 = [[0, 0, 0, 0, 0, 0, 0.0] for _ in range(len(self.args.arm_joint_state_ctrl_topics))]

        print("publishing:")
        if self.args.use_arm_joint_state > 1:
            self.arm_joint_state_ctrl_linear_interpolation(joint_state0, True)
        pre_inference_status = -1
        ctrl_rate = self.create_rate(30)
        all_actions = None
        print("done")
        while rclpy.ok():
            t = 0
            max_t = 0
            if inference_thread is not None:
                inference_thread.join()
                inference_thread = None
            while t < self.args.max_publish_step and rclpy.ok():
                inference_status = self.get_inference_status()
                print("inference_status:", inference_status)
                if inference_status == -1:
                    input("Please press any key to start inference:")
                    self.set_inference_status(0)
                is_new_action = False
                if self.args.asynchronous_inference:
                    if inference_thread is None and (self.args.pos_lookahead_step == 0 or t % self.args.pos_lookahead_step == 0 or t >= max_t) and self.check_frame():
                        inference_thread = threading.Thread(target=inference_process,
                                                            args=(self.args, t, self, policy))
                        inference_thread.start()

                    if inference_thread is not None and (not inference_thread.is_alive() or t >= max_t):
                        inference_thread.join()
                        inference_lock.acquire()
                        inference_thread = None
                        all_actions = inference_actions
                        t_start = inference_timestep
                        max_t = t_start + self.args.chunk_size
                        is_new_action = True
                        if self.args.debug_mode:
                            # Retrieve the log package and clear the global buffer under the lock
                            package_to_log = inference_log_package
                            inference_log_package = None
                        inference_lock.release()
                        if self.args.debug_mode:
                            if package_to_log:
                                log_path = os.path.join(log_dir, f"t_{t_start}.pkl")
                                print(f"Saving log to: {log_path}")
                                with open(log_path, 'wb') as f:
                                    pickle.dump(package_to_log, f)

                            input(f"New action chunk received. Ready to execute from t={t_start}. Press Enter to proceed...")

                if t >= max_t:
                    print("inference time error")
                    continue
                raw_action = all_actions[:, t - t_start]
                raw_action = raw_action[0]
                if self.args.use_arm_joint_state > 1:
                    action_joint_state = raw_action[:self.args.arm_joint_state_dim * len(self.args.arm_joint_state_names)]
                    raw_action = raw_action[self.args.arm_joint_state_dim * len(self.args.arm_joint_state_names):]
                    action_joint_state = [action_joint_state[i*self.args.arm_joint_state_dim:(i+1)*self.args.arm_joint_state_dim] for i in range(len(self.args.arm_joint_state_names))]
                    self.arm_joint_state_ctrl(action_joint_state)
                print("t:", t)
                ctrl_rate.sleep()
                t += 1

    def instruction_callback(self, msg):
        self.instruction = msg.data

    def camera_color_callback(self, index, msg):
        if self.camera_color_deques[index].size() >= 200:
            self.camera_color_deques[index].popleft()
        self.camera_color_deques[index].append(msg)

    def arm_joint_state_callback(self, index, msg):
        if self.arm_joint_state_deques[index].size() >= 200:
            self.arm_joint_state_deques[index].popleft()
        self.arm_joint_state_deques[index].append(msg)

    def arm_joint_state_ctrl(self, joint_states):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = [f'joint{i+1}' for i in range(self.args.arm_joint_state_dim)]
        self.last_ctrl_arm_joint_state = joint_states
        for i in range(len(joint_states)):
            joint_state_msg.position = [float(v) for v in joint_states[i]]
            self.arm_joint_state_ctrl_publisher[i].publish(joint_state_msg)

    def arm_joint_state_ctrl_linear_interpolation(self, joint_states, calc_step):
        if self.last_ctrl_arm_joint_state is None:
            last_ctrl_joint_state = np.concatenate(
                [np.array(self.arm_joint_state_deques[i].right().position) for i in range(len(self.args.arm_joint_state_names))], axis=0)
        else:
            last_ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in self.last_ctrl_arm_joint_state], axis=0)

        ctrl_joint_state = np.concatenate(
                [np.array(joint_state) for joint_state in joint_states], axis=0)
        joint_state_diff = ctrl_joint_state - last_ctrl_joint_state

        hz = 500
        if calc_step:
            step = int(max([max(abs(joint_state_diff[i*self.args.arm_joint_state_dim: (i+1)*self.args.arm_joint_state_dim-1]) / np.array(self.args.arm_steps_length[:self.args.arm_joint_state_dim-1])) for i in range(len(self.args.arm_joint_state_names))]))
            step = 1 if step == 0 else step
        else:
            step = 10
        rate = self.create_rate(hz)
        append_to_history_list_step = 10
        joint_state_list = np.linspace(last_ctrl_joint_state, ctrl_joint_state, step + 1)
        for i in range(1, len(joint_state_list)):
            if self.arm_joint_state_ctrl_thread_return_lock.acquire(False):
                return
            ctrl_joint_state = [joint_state_list[i][j * self.args.arm_joint_state_dim: (j + 1) * self.args.arm_joint_state_dim] for j in range(len(self.args.arm_joint_state_names))]
            self.arm_joint_state_ctrl(ctrl_joint_state)
            if i % append_to_history_list_step == 0 or i + 1 == len(joint_state_list):
                self.arm_joint_state_ctrl_history_list.append(ctrl_joint_state)
            rate.sleep()
        self.arm_joint_state_ctrl_history_list = self.arm_joint_state_ctrl_history_list[-self.k:]
        return

    def get_frame(self):
        camera_colors = [self.camera_color_deques[i].right() for i in range(len(self.args.camera_color_names))]
        arm_joint_states = [self.arm_joint_state_deques[i].right() for i in range(len(self.args.arm_joint_state_names))]
        frame_time = max([rclpy.time.Time.from_msg(msg.header.stamp).nanoseconds / 1e9 for msg in
                          (camera_colors + arm_joint_states)])

        for i in range(len(self.args.camera_color_names)):
            closer_time_diff = math.inf
            while (self.camera_color_deques[i].size() > 0 and
                   abs(rclpy.time.Time.from_msg(self.camera_color_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time) < closer_time_diff):
                closer_time_diff = abs(rclpy.time.Time.from_msg(self.camera_color_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time)
                camera_colors[i] = self.camera_color_deques[i].popleft()
        for i in range(len(self.args.arm_joint_state_names)):
            closer_time_diff = math.inf
            while (self.arm_joint_state_deques[i].size() > 0 and
                   abs(rclpy.time.Time.from_msg(self.arm_joint_state_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time) < closer_time_diff):
                closer_time_diff = abs(rclpy.time.Time.from_msg(self.arm_joint_state_deques[i].left().header.stamp).nanoseconds / 1e9 - frame_time)
                arm_joint_states[i] = self.arm_joint_state_deques[i].popleft()

        #for i in range(len(self.args.camera_color_names)):
        #    while rclpy.time.Time.from_msg(self.camera_color_deques[i].left().header.stamp).nanoseconds / 1e9 < frame_time:
        #        self.camera_color_deques[i].popleft()
        #    camera_colors[i] = self.camera_color_deques[i].popleft()
        #for i in range(len(self.args.arm_joint_state_names)):
        #    while rclpy.time.Time.from_msg(self.arm_joint_state_deques[i].left().header.stamp).nanoseconds / 1e9 < frame_time:
        #        self.arm_joint_state_deques[i].popleft()
        #    arm_joint_states[i] = self.arm_joint_state_deques[i].popleft()

        if len(self.camera_color_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.camera_color_history_list.append(camera_colors)
        self.camera_color_history_list.append(camera_colors)
        self.camera_color_history_list = self.camera_color_history_list[-self.args.obs_history_num:]

        if len(self.arm_joint_state_history_list) == 0:
            for i in range(self.args.obs_history_num):
                self.arm_joint_state_history_list.append(arm_joint_states)
        self.arm_joint_state_history_list.append(arm_joint_states)
        self.arm_joint_state_history_list = self.arm_joint_state_history_list[-self.args.obs_history_num:]

        return (self.instruction, self.camera_color_history_list, self.arm_joint_state_history_list)

    def check_frame(self):
        for i in range(len(self.args.camera_color_names)):
            if self.camera_color_deques[i].size() == 0:
                print(self.args.camera_color_topics[i], "has no data")
                return False
        for i in range(len(self.args.arm_joint_state_names)):
            if self.arm_joint_state_deques[i].size() == 0:
                print(self.args.arm_joint_state_topics[i], "has no data")
                return False
        return True

    def change_inference_status(self, request, response):
        self.inference_status = request.status
        return response

    def get_inference_status(self):
        return self.inference_status

    def set_inference_status(self, status):
        self.inference_status = status


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, help='max_publish_step', default=10000,
                        required=False)

    parser.add_argument('--instruction_topic', action='store', type=str, help='instruction_topic',
                        default="/instruction",
                        required=False)

    parser.add_argument('--camera_color_names', action='store', type=str, help='camera_color_names',
                        default=['left', 'front', 'right'],
                        required=False)
    parser.add_argument('--camera_color_parent_frame_ids', action='store', type=str, help='camera_color_parent_frame_ids',
                        default=['camera_l_link', 'camera_f_link', 'camera_r_link'],
                        required=False)
    parser.add_argument('--camera_color_topics', action='store', type=str, help='camera_color_topics',
                        default=['/camera_l/color/image_raw', '/camera_f/color/image_raw', '/camera_r/color/image_raw'],
                        required=False)
    parser.add_argument('--arm_joint_state_names', action='store', type=str, help='arm_joint_state_names',
                        default=['left', 'right'],
                        required=False)
    parser.add_argument('--arm_joint_state_topics', action='store', type=str, help='arm_joint_state_topics',
                        default=['/puppet/joint_left', '/puppet/joint_right'],
                        required=False)
    parser.add_argument('--arm_joint_state_ctrl_topics', action='store', type=str, help='arm_joint_state_ctrl_topics', nargs='+',
                        default=['/joint_left_states', '/joint_right_states'],
                        required=False)

    parser.add_argument('--use_camera_color', action='store', type=bool, help='use_camera_color', default=True, required=False)
    parser.add_argument('--use_arm_joint_state', action='store', type=int, help='use_arm_joint_state', default=3, required=False)
    parser.add_argument('--arm_joint_state_dim', action='store', type=int, help='arm_joint_state_dim', default=7, required=False)

    parser.add_argument('--obs_history_num', action='store', type=int, help='obs_history_num', default=1, required=False)
    parser.add_argument('--use_instruction', action='store', type=bool, help='use_instruction', default=False, required=False)
    parser.add_argument('--instruction', action='store', type=str, help='instruction',
                        default='null', required=False)

    parser.add_argument('--aloha_inference_status_service', action='store', type=str,
                        help='aloha_inference_status_service',
                        default='/aloha/inference_status_service', required=False)
    parser.add_argument('--pos_lookahead_step', action='store', type=int, help='pos_lookahead_step',
                        default=25, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size',
                        default=25, required=False)  # NOTE: the pos_lookahead_step and chunk_size should be aligned.
    parser.add_argument('--arm_steps_length', action='store', type=float, help='arm_steps_length',
                        default=[0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.0015, 0.03], required=False)
    parser.add_argument('--robot_base_steps_length', action='store', type=float, help='robot_base_steps_length',
                        default=[0.1, 0.1], required=False)
    parser.add_argument('--asynchronous_inference', action='store', type=bool, help='asynchronous_inference',
                        default=True, required=False)
    parser.add_argument('--preemptive_publishing', action='store', type=bool, help='preemptive_publishing',
                        default=False, required=False)
    parser.add_argument('--blocking_publish', action='store', type=bool, help='blocking_publish',
                        default=True, required=False)
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode')

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    rclpy.init()
    ros_operator = RosOperator(args)
    rclpy.spin(ros_operator)


if __name__ == '__main__':
    main()
