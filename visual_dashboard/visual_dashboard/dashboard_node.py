#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import CompressedImage, BatteryState, Image
from std_msgs.msg import Int32, String, Bool
from irobot_create_msgs.msg import DockStatus
from vehicle_alert.msg import VehicleAlert
from cv_bridge import CvBridge
import numpy as np
import cv2
import threading
from datetime import datetime
import time

FONT = cv2.FONT_HERSHEY_SIMPLEX
PAD = 4
CAM_SIZE = 360
SIDE_W = 480
SUB_CNT = 3
LOG_H = 120
LOG_LINES = 5

def draw_button(img, x, y, w, h, text, base_color, text_color=(255,255,255), pressed=False, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.rectangle(img, (x, y), (x + w, y + h), base_color, -1)
    # 입체감 색상
    if pressed:
        light = tuple(min(c + 40, 255) for c in base_color)
        dark  = tuple(max(c - 40, 0) for c in base_color)
        cv2.line(img, (x, y + h), (x + w, y + h), light, 1)
        cv2.line(img, (x + w, y), (x + w, y + h), light, 1)
        cv2.line(img, (x, y), (x + w, y), dark, 1)
        cv2.line(img, (x, y), (x, y + h), dark, 1)
    else:
        light = tuple(min(c + 40, 255) for c in base_color)
        dark  = tuple(max(c - 40, 0) for c in base_color)
        cv2.line(img, (x, y), (x + w, y), light, 1)
        cv2.line(img, (x, y), (x, y + h), light, 1)
        cv2.line(img, (x, y + h), (x + w, y + h), dark, 1)
        cv2.line(img, (x + w, y), (x + w, y + h), dark, 1)
    # 텍스트 중앙 정렬
    text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
    tx = x + (w - text_size[0]) // 2
    ty = y + (h + text_size[1]) // 2
    cv2.putText(img, text, (tx, ty), font, 0.8, text_color, 2, cv2.LINE_AA)

class MultiCamDashboard(Node):
    def __init__(self):
        super().__init__('multi_cam_dashboard')

        self.battery_0_last = 0
        self.battery_1_last = 0
        self.dock_0_last = 0
        self.dock_1_last = 0
        self.led_0_last = [0]*5
        self.led_1_last = [0]*5

        self.timeout_sec = 5.0

        self.detected_vehicles = []
        self.create_subscription(VehicleAlert, '/vehicle_alert', self.cb_vehicle_alert, 10)
        self.br = CvBridge()
        self.lock = threading.Lock()

        self.logs_robot0 = ['']
        self.logs_robot1 = ['']

        # 패널: CCTV1, CCTV2, Robot0, Robot1
        self.imgs = [
            np.zeros((CAM_SIZE, CAM_SIZE, 3), np.uint8),  # CCTV1
            np.zeros((CAM_SIZE, CAM_SIZE, 3), np.uint8),  # CCTV2
            np.zeros((CAM_SIZE, CAM_SIZE, 3), np.uint8),  # Robot0
            np.zeros((CAM_SIZE, CAM_SIZE, 3), np.uint8),  # Robot1
        ]

        self.battery_percentage_0 = None
        self.battery_percentage_1 = None
        self.dock_state_0 = None
        self.dock_state_1 = None

        self.sub1_img_0 = None
        self.sub1_img_1 = None

        self.create_subscription(Image, '/robot0/map_dashboard', self.cb_sub1_img_0, 10)
        self.create_subscription(Image, '/robot1/map_dashboard', self.cb_sub1_img_1, 10)

        self.leds_0 = [False] * 5
        self.create_subscription(Int32, '/robot0/hmi/led/_motors',  self.cb_led_0_motors,  10)
        self.create_subscription(Int32, '/robot0/hmi/led/_comms',   self.cb_led_0_comms,   10)
        self.create_subscription(Int32, '/robot0/hmi/led/_wifi',    self.cb_led_0_wifi,    10)
        self.create_subscription(Int32, '/robot0/hmi/led/_battery', self.cb_led_0_battery, 10)
        self.create_subscription(Int32, '/robot0/hmi/led/_power',   self.cb_led_0_power,   10)

        self.leds_1 = [False] * 5
        self.create_subscription(Int32, '/robot1/hmi/led/_motors',  self.cb_led_1_motors,  10)
        self.create_subscription(Int32, '/robot1/hmi/led/_comms',   self.cb_led_1_comms,   10)
        self.create_subscription(Int32, '/robot1/hmi/led/_wifi',    self.cb_led_1_wifi,    10)
        self.create_subscription(Int32, '/robot1/hmi/led/_battery', self.cb_led_1_battery, 10)
        self.create_subscription(Int32, '/robot1/hmi/led/_power',   self.cb_led_1_power,   10)

        img_path1 = '/home/ethica/rokey_ws/maps/b01.pgm'
        img_path2 = '/home/ethica/rokey_ws/maps/b02.pgm'
        img1 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
        img2 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
        if img1 is None:
            img1 = np.full((CAM_SIZE, CAM_SIZE, 3), 100, np.uint8)
        if img2 is None:
            img2 = np.full((CAM_SIZE, CAM_SIZE, 3), 100, np.uint8)
        self.map_imgs = [
            cv2.resize(img1, (SIDE_W // 2 - 6, CAM_SIZE)),
            cv2.resize(img2, (SIDE_W // 2 - 6, CAM_SIZE))
        ]

        self.qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.create_subscription(CompressedImage, 'robot0/cam/image/compressed', lambda msg: self.cb_img(msg, 0), self.qos_best_effort)
        self.create_subscription(CompressedImage, 'robot1/cam/image/compressed', lambda msg: self.cb_img(msg, 1), self.qos_best_effort)

        self.create_subscription(CompressedImage, '/robot0/oakd/rgb/image_fast/compressed', lambda msg: self.cb_img(msg, 2), self.qos_best_effort)
        self.create_subscription(CompressedImage, '/robot1/oakd/rgb/image_fast/compressed', lambda msg: self.cb_img(msg, 3), self.qos_best_effort)
        
        self.create_subscription(BatteryState, '/robot0/battery_state', self.cb_battery_0, 10)
        self.create_subscription(BatteryState, '/robot1/battery_state', self.cb_battery_1, 10)
        self.create_subscription(DockStatus, '/robot0/dock_status', self.cb_dock_0, 10)
        self.create_subscription(DockStatus, '/robot1/dock_status', self.cb_dock_1, 10)
        self.create_subscription(String, '/robot0/log', self.cb_log_robot0, 10)
        self.create_subscription(String, '/robot1/log', self.cb_log_robot1, 10)

        # 버튼 override 상태 확인용 
        self.override_state_0 = False
        self.override_state_1 = False

        self.create_timer(0.1, self.override_loop)

        self.create_subscription(Bool, '/robot0/action_override', self.cb_override_0, self.qos)
        self.create_subscription(Bool, '/robot1/action_override', self.cb_override_1, self.qos)

        self.pub_robot0_override = self.create_publisher(Bool, '/robot0/action_override', self.qos)
        self.pub_robot1_override = self.create_publisher(Bool, '/robot1/action_override', self.qos)

        self.pub_robot0_action = self.create_publisher(Int32, '/robot0/action', self.qos)
        self.pub_robot0_alert  = self.create_publisher(Int32, '/robot0/system_alert', self.qos)
        self.pub_robot1_action = self.create_publisher(Int32, '/robot1/action', self.qos)
        self.pub_robot1_alert  = self.create_publisher(Int32, '/robot1/system_alert', self.qos)
        

        # 버튼 영역(패널 내부 좌표) 저장용
        self.button_regions_robot0 = {}  
        self.button_regions_robot1 = {}

        # 오른쪽 패널(3분할) 위치 계산용
        self.left_panel_w = CAM_SIZE*2
        self.right_panel_x = self.left_panel_w
        self.right_panel_y = 0  # 항상 0부터 시작
        self.status_panel_idx = 1  # 오른쪽 패널에서 "Status"는 두 번째 분할(i==1)
        self.status_panel_offset_y = int((CAM_SIZE*2) // SUB_CNT * self.status_panel_idx)

        cv2.namedWindow('Dashboard')
        cv2.setMouseCallback('Dashboard', self.mouse_callback)
        self.create_timer(1/20, self.render)

    # --- LED 콜백 ---
    def cb_led_0_motors(self, msg):  self.leds_0[0] = (msg.data == 1); self.led_0_last[0] = time.time()
    def cb_led_0_comms(self, msg):   self.leds_0[1] = (msg.data == 1); self.led_0_last[1] = time.time()
    def cb_led_0_wifi(self, msg):    self.leds_0[2] = (msg.data == 1); self.led_0_last[2] = time.time()
    def cb_led_0_battery(self, msg): self.leds_0[3] = msg.data       ; self.led_0_last[3] = time.time()
    def cb_led_0_power(self, msg):   self.leds_0[4] = (msg.data == 1); self.led_0_last[4] = time.time()
    def cb_led_1_motors(self, msg):  self.leds_1[0] = (msg.data == 1); self.led_1_last[0] = time.time()
    def cb_led_1_comms(self, msg):   self.leds_1[1] = (msg.data == 1); self.led_1_last[1] = time.time()
    def cb_led_1_wifi(self, msg):    self.leds_1[2] = (msg.data == 1); self.led_1_last[2] = time.time()
    def cb_led_1_battery(self, msg): self.leds_1[3] = msg.data       ; self.led_1_last[3] = time.time()
    def cb_led_1_power(self, msg):   self.leds_1[4] = (msg.data == 1); self.led_1_last[4] = time.time()

    def cb_img(self, msg, idx):
        if isinstance(msg, CompressedImage):
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif isinstance(msg, Image):
            img = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        else:
            self.get_logger().warn(f"Unknown image type received at index {idx}")
            return
        if img is not None:
            img = cv2.resize(img, (CAM_SIZE, CAM_SIZE))
            with self.lock:
                self.imgs[idx] = img

    def cb_battery_0(self, msg):
        if msg.percentage is not None:
            self.battery_percentage_0 = round(msg.percentage * 100, 1)
            self.battery_0_last = time.time()
    def cb_battery_1(self, msg):
        if msg.percentage is not None:
            self.battery_percentage_1 = round(msg.percentage * 100, 1)
            self.battery_1_last = time.time()
    def cb_dock_0(self, msg):
        self.dock_state_0 = msg.is_docked
        self.dock_0_last = time.time()
    def cb_dock_1(self, msg):
        self.dock_state_1 = msg.is_docked
        self.dock_1_last = time.time()
    def cb_vehicle_alert(self, msg):
        if len(str(msg.id)) == 4:
            item = (msg.id, msg.location)
            if item not in self.detected_vehicles:
                self.detected_vehicles.insert(0, item)
                if len(self.detected_vehicles) > 5:
                    self.detected_vehicles.pop()
        else:
            return
    def cb_log_robot0(self, msg):
        log = str(msg.data)
        now = datetime.now().strftime('%H:%M:%S')
        if 'Waiting for robot0 log...' in self.logs_robot0:
            self.logs_robot0.remove('Waiting for robot0 log...')
        self.logs_robot0.insert(0, f"[{now}] {log}")
        self.logs_robot0 = self.logs_robot0[:LOG_LINES]
    def cb_log_robot1(self, msg):
        log = str(msg.data)
        now = datetime.now().strftime('%H:%M:%S')
        if 'Waiting for robot0 log...' in self.logs_robot1:
            self.logs_robot1.remove('Waiting for robot1 log...')
        self.logs_robot1.insert(0, f"[{now}] {log}")
        self.logs_robot1 = self.logs_robot1[:LOG_LINES]
    def cb_sub1_img_0(self, msg):
        try:
            img = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.sub1_img_0 = img
        except Exception as e:
            self.get_logger().warn(f"Failed to convert /robot0/map_dashboard: {e}")
    def cb_sub1_img_1(self, msg):
        try:
            img = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.sub1_img_1 = img
        except Exception as e:
            self.get_logger().warn(f"Failed to convert /robot1/map_dashboard: {e}")

    def get_battery_color(self, percent):
        if percent is None:
            return (150, 150, 150)
        if percent > 50:
            return (50, 220, 50)
        elif percent > 30:
            return (0, 220, 220)
        else:
            return (0, 0, 220)
        
    def override_loop(self):
        if self.override_state_0:
            msg = Int32()
            msg.data = 1
            self.pub_robot0_action.publish(msg)
        if self.override_state_1:
            msg = Int32()
            msg.data = 1
            self.pub_robot1_action.publish(msg)

    def boxed(self, img, title='', extra_text=None, extra_colors=None, title_thickness=1, content_thickness=1):
        h, w = img.shape[:2]
        out = np.full((h + 2*PAD, w + 2*PAD, 3), 255, np.uint8)
        out[PAD:PAD+h, PAD:PAD+w] = img
        cv2.rectangle(out, (0, 0), (out.shape[1]-1, out.shape[0]-1), (230, 230, 230), 2)
        cv2.rectangle(out, (2, 2), (out.shape[1]-3, out.shape[0]-3), (100, 100, 100), 1)
        y_text = 28
        if title:
            cv2.putText(out, title, (10, y_text), FONT, 0.8, (60, 60, 60), title_thickness, cv2.LINE_AA)
            y_text += 32
        if extra_text is not None:
            if isinstance(extra_text, list) and isinstance(extra_colors, list):
                x0 = 10
                x1 = 240
                y = y_text
                for row in range(2):
                    idx0 = row * 2
                    idx1 = row * 2 + 1
                    th0 = th1 = content_thickness
                    if idx0 < len(extra_text):
                        cv2.putText(out, extra_text[idx0], (x0, y), FONT, 0.7, extra_colors[idx0], th0, cv2.LINE_AA)
                    if idx1 < len(extra_text):
                        cv2.putText(out, extra_text[idx1], (x1, y), FONT, 0.7, extra_colors[idx1], th1, cv2.LINE_AA)
                    y += 32
            else:
                c = extra_colors if extra_colors else (40, 80, 120)
                cv2.putText(out, extra_text, (10, y_text), FONT, 0.7, c, content_thickness, cv2.LINE_AA)
        return out

    def boxed_vertical_list(self, img, title='', lines=None, colors=None):
        h, w = img.shape[:2]
        out = np.full((h + 2*PAD, w + 2*PAD, 3), 255, np.uint8)
        out[PAD:PAD+h, PAD:PAD+w] = img
        cv2.rectangle(out, (0, 0), (out.shape[1]-1, out.shape[0]-1), (230, 230, 230), 2)
        cv2.rectangle(out, (2, 2), (out.shape[1]-3, out.shape[0]-3), (100, 100, 100), 1)
        y = 28
        if title:
            cv2.putText(out, title, (10, y), FONT, 0.8, (60, 60, 60), 1, cv2.LINE_AA)
            y += 32
        if lines:
            for i, text in enumerate(lines[:6]):
                color = colors[i] if colors and i < len(colors) else (60, 60, 60)
                cv2.putText(out, text, (10, y), FONT, 0.7, color, 2, cv2.LINE_AA)
                y += 28
        return out

    def draw_led_row_rect(self, states, size=32, margin=8):
        w = len(states) * (size + margin) - margin + 2 * margin
        h = size + 2 * margin
        img = np.zeros((h, w, 3), np.uint8)
        for i, state in enumerate(states):
            x = margin + i * (size + margin)
            if state == 1:
                color = (40, 220, 40)
            elif state == 2:
                color = (40, 40, 220)
            elif state == 3:
                color = (0, 220, 220)
            else:
                color = (180, 180, 180)
            cv2.rectangle(img, (x, margin), (x + size, margin + size), color, -1)
            cv2.rectangle(img, (x, margin), (x + size, margin + size), (80, 80, 80), 1)
        return img
    
    # 주행 override 여부
    def cb_override_0(self, msg): self.override_state_0 = msg.data
    def cb_override_1(self, msg): self.override_state_1 = msg.data

    def render(self):
        now = time.time()
        if now - self.battery_0_last > self.timeout_sec:
            self.battery_percentage_0 = None
        if now - self.battery_1_last > self.timeout_sec:
            self.battery_percentage_1 = None
        if now - self.dock_0_last > self.timeout_sec:
            self.dock_state_0 = None
        if now - self.dock_1_last > self.timeout_sec:
            self.dock_state_1 = None
        for i in range(5):
            if now - self.led_0_last[i] > self.timeout_sec:
                self.leds_0[i] = False
            if now - self.led_1_last[i] > self.timeout_sec:
                self.leds_1[i] = False
        with self.lock:
            imgs = [img.copy() for img in self.imgs]
        if self.override_state_0:
            msg = Int32()
            msg.data = 1
            self.pub_robot0_action.publish(msg)

        cctv1 = self.boxed(imgs[0], 'CCTV0')
        cctv2 = self.boxed(imgs[1], 'CCTV1')
        tb0 = self.boxed(imgs[2], 'Robot0')
        tb1 = self.boxed(imgs[3], 'Robot1')
        row1 = np.hstack([cctv1, cctv2])
        row2 = np.hstack([tb0, tb1])
        cam_panel = np.vstack([row1, row2])
        logs_panel = np.full((LOG_H, cam_panel.shape[1], 3), 245, np.uint8)
        half_w = cam_panel.shape[1] // 2
        for i, line in enumerate(self.logs_robot0):
            cv2.putText(logs_panel, line, (8, 24 + i * 20), FONT, 0.54, (60, 60, 60), 1, cv2.LINE_AA)
        for i, line in enumerate(self.logs_robot1):
            cv2.putText(logs_panel, line, (half_w + 8, 24 + i * 20), FONT, 0.54, (60, 60, 60), 1, cv2.LINE_AA)
        left_panel = np.vstack([cam_panel, logs_panel])

        # 오른쪽 패널(3분할)
        total_h = cam_panel.shape[0]
        sub_h = total_h // SUB_CNT
        right_panels = []
        # 버튼 좌표 딕셔너리 초기화
        self.button_regions_robot0 = {}
        self.button_regions_robot1 = {}

        for i in range(SUB_CNT):
            img = np.zeros((CAM_SIZE, SIDE_W, 3), np.uint8)
            extra = None
            extra_colors = None
            if i == 0:
                h = CAM_SIZE
                w = SIDE_W // 2
                left_img = self.sub1_img_0 if self.sub1_img_0 is not None else self.map_imgs[0]
                right_img = self.sub1_img_1 if self.sub1_img_1 is not None else self.map_imgs[1]
                left_img = cv2.resize(left_img, (w, h))
                right_img = cv2.resize(right_img, (w, h))
                img = np.hstack([left_img, right_img])
                boxed_img = self.boxed(
                    cv2.resize(img, (SIDE_W - 2*PAD, sub_h - 2*PAD)),
                    'Map', extra, extra_colors, title_thickness=1, content_thickness=2
                )
                right_panels.append(boxed_img)
            elif i == 1:
                texts = []
                colors = []
                if self.battery_percentage_0 is not None:
                    texts.append(f"Robot0: {self.battery_percentage_0:.1f}%")
                    colors.append(self.get_battery_color(self.battery_percentage_0))
                else:
                    texts.append("Robot0: -")
                    colors.append((150, 150, 150))
                if self.battery_percentage_1 is not None:
                    texts.append(f"Robot1: {self.battery_percentage_1:.1f}%")
                    colors.append(self.get_battery_color(self.battery_percentage_1))
                else:
                    texts.append("Robot1: -")
                    colors.append((150, 150, 150))
                dock0_str = "DOCKED" if self.dock_state_0 else "UNDOCKED" if self.dock_state_0 is not None else "-"
                dock1_str = "DOCKED" if self.dock_state_1 else "UNDOCKED" if self.dock_state_1 is not None else "-"
                dock0_color = (100, 100, 100) if self.dock_state_0 else (0, 0, 220) if self.dock_state_0 is not None else (150, 150, 150)
                dock1_color = (100, 100, 100) if self.dock_state_1 else (0, 0, 220) if self.dock_state_1 is not None else (150, 150, 150)
                texts.append(f"Dock: {dock0_str}")
                colors.append(dock0_color)
                texts.append(f"Dock: {dock1_str}")
                colors.append(dock1_color)
                extra = texts
                extra_colors = colors

                boxed_img = self.boxed(
                    cv2.resize(img, (SIDE_W - 2*PAD, sub_h - 2*PAD)),
                    'Status', extra, extra_colors, title_thickness=1, content_thickness=2
                )
                led_img_0 = self.draw_led_row_rect(self.leds_0)
                led_img_1 = self.draw_led_row_rect(self.leds_1)
                bx_h, bx_w = boxed_img.shape[:2]
                led_h, led_w = led_img_0.shape[:2]
                y0, x0 = 105, 5
                y1, x1 = 105, 233
                boxed_img[y0:y0+led_h, x0:x0+led_w] = led_img_0
                boxed_img[y1:y1+led_h, x1:x1+led_w] = led_img_1

                # 버튼(패널 내부 상대좌표 기준)
                btn_w, btn_h = 90, 32
                btn_gap = 10
                btn1_y = y0 + led_h + 4
                btn2_y = btn1_y + btn_h + btn_gap

                # Robot0 (왼쪽)
                x0_btn = 15
                draw_button(boxed_img, x0_btn, btn1_y, btn_w, btn_h, 'Start', (60,200,60))
                self.button_regions_robot0['start']  = (x0_btn, btn1_y, btn_w, btn_h)
                draw_button(boxed_img, x0_btn+btn_w+btn_gap, btn1_y, btn_w, btn_h, 'Stop', (60,60,220))
                self.button_regions_robot0['stop']   = (x0_btn + btn_w + btn_gap, btn1_y, btn_w, btn_h)
                draw_button(boxed_img, x0_btn, btn2_y, 2*btn_w+btn_gap, btn_h, 'Search', (255,210,120))
                self.button_regions_robot0['search'] = (x0_btn, btn2_y, 2*btn_w+btn_gap, btn_h)

                # Robot1 (오른쪽)
                x1_btn = 240
                draw_button(boxed_img, x1_btn, btn1_y, btn_w, btn_h, 'Start', (60,200,60))
                self.button_regions_robot1['start']  = (x1_btn, btn1_y, btn_w, btn_h)
                draw_button(boxed_img, x1_btn+btn_w+btn_gap, btn1_y, btn_w, btn_h, 'Stop', (60,60,220))
                self.button_regions_robot1['stop']   = (x1_btn + btn_w + btn_gap, btn1_y, btn_w, btn_h)
                draw_button(boxed_img, x1_btn, btn2_y, 2*btn_w+btn_gap, btn_h, 'Search', (255,210,120))
                self.button_regions_robot1['search'] = (x1_btn, btn2_y, 2*btn_w+btn_gap, btn_h)

                right_panels.append(boxed_img)
            elif i == 2:
                extra = []
                extra_colors = []
                for id, location in self.detected_vehicles:
                    extra.append(f"Car ID: {id}, Locate: {location}")
                    extra_colors.append((233, 219, 227))
                boxed_img = self.boxed_vertical_list(
                    cv2.resize(img, (SIDE_W - 2*PAD, sub_h - 2*PAD)),
                    title='Detected', lines=extra, colors=extra_colors
                )
                right_panels.append(boxed_img)

        right_panel = np.vstack(right_panels)
        pad_needed = total_h - right_panel.shape[0]
        if pad_needed > 0:
            pad = np.full((pad_needed, SIDE_W, 3), 255, np.uint8)
            right_panel = np.vstack([right_panel, pad])
        elif pad_needed < 0:
            right_panel = right_panel[:total_h, :]

        lh = left_panel.shape[0]
        rh = right_panel.shape[0]
        if lh > rh:
            pad = np.full((lh - rh, SIDE_W, 3), 245, np.uint8)
            right_panel = np.vstack([right_panel, pad])
        elif lh < rh:
            pad = np.full((rh - lh, left_panel.shape[1], 3), 245, np.uint8)
            left_panel = np.vstack([left_panel, pad])

        dashboard = np.hstack([left_panel, right_panel])

        if self.override_state_0:
            cv2.putText(dashboard, 'ROBOT0: AUTORIDE MODE OFF', (5, dashboard.shape[0] - 130), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(dashboard, 'ROBOT0: AUTORIDE MODE ON', (5, dashboard.shape[0] - 130), FONT, 0.5, (80, 200, 60), 1, cv2.LINE_AA)
        if self.override_state_1:
            cv2.putText(dashboard, 'ROBOT1: AUTORIDE MODE OFF', (375, dashboard.shape[0] - 130), FONT, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(dashboard, 'ROBOT1: AUTORIDE MODE ON', (375, dashboard.shape[0] - 130), FONT, 0.5, (80, 200, 60), 1, cv2.LINE_AA)

        time_str = datetime.now().strftime('%H:%M:%S')
        cv2.putText(dashboard, f"Time: {time_str}", (dashboard.shape[1]-140, dashboard.shape[0]-10), FONT, 0.5, (80,80,80), 1, cv2.LINE_AA)
        cv2.imshow('Dashboard', dashboard)
        cv2.waitKey(40)

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Status 패널의 위치는 항상 오른쪽 패널 내 두 번째 1/3 영역
        status_panel_x = self.right_panel_x
        status_panel_y = self.status_panel_offset_y
        # Robot0 버튼
        for name, (bx, by, bw, bh) in self.button_regions_robot0.items():
            gx = status_panel_x + bx
            gy = status_panel_y + by
            if gx <= x <= gx + bw and gy <= y <= gy + bh:
                msg = Int32()
                if name == 'start':
                    msg.data = 0
                    self.pub_robot0_action.publish(msg)
                    self.pub_robot0_override.publish(Bool(data=False))
                    self.get_logger().info("[Robot0] Start(override OFF) → /robot0/action = auto")
                elif name == 'stop':
                    msg.data = 1
                    self.pub_robot0_action.publish(msg)
                    self.pub_robot0_override.publish(Bool(data=True))
                    self.get_logger().info("[Robot0] Stop(override ON) → /robot0/action = 1(stop)")
                elif name == 'search':
                    msg.data = 1
                    self.pub_robot0_alert.publish(msg)
                    self.get_logger().info("[Robot0] Search → /robot0/system_alert = 1")
        # Robot1 버튼
        for name, (bx, by, bw, bh) in self.button_regions_robot1.items():
            gx = status_panel_x + bx
            gy = status_panel_y + by
            if gx <= x <= gx + bw and gy <= y <= gy + bh:
                msg = Int32()
                if name == 'start':
                    msg.data = 0
                    self.pub_robot1_action.publish(msg)
                    self.pub_robot1_override.publish(Bool(data=False))
                    self.get_logger().info("[Robot1] Start(override OFF) → /robot1/action = 0")
                elif name == 'stop':
                    msg.data = 1
                    self.pub_robot1_action.publish(msg)
                    self.pub_robot1_override.publish(Bool(data=True))
                    self.get_logger().info("[Robot1] Stop(override ON) → /robot1/action = 1")
                elif name == 'search':
                    msg.data = 1
                    self.pub_robot1_alert.publish(msg)
                    self.get_logger().info("[Robot1] Search → /robot1/system_alert = 1")


def main(args=None):
    rclpy.init(args=args)
    node = MultiCamDashboard()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
