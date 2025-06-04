#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import yaml
import os
from ament_index_python.packages import get_package_share_directory


class WaypointMarkerPublisher(Node):
    def __init__(self):
        super().__init__('path_marker0')

        self.pub = self.create_publisher(MarkerArray, '/robot0/waypoint_markers', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1Hz

        self.frame_id = 'map'

        # YAML 로딩
        try:
            yaml_path = os.path.join(
                get_package_share_directory('rokey_pjt'),
                'config',
                'waypoints_0.yaml'
            )
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            self.waypoints = data.get('waypoints', [])
            self.wp_dict = {wp['id']: wp for wp in self.waypoints}
            self.get_logger().info(f'[WaypointMarkerPublisher] Loaded {len(self.wp_dict)} waypoints.')
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoints: {e}')
            self.waypoints = []
            self.wp_dict = {}

    def timer_callback(self):
        marker_array = MarkerArray()

        # ── (1) 점 마커 및 텍스트 마커 ─────────────────────
        for idx, wp in enumerate(self.waypoints):
            x = float(wp['x'])
            y = float(wp['y'])

            # 점(SPHERE)
            sphere = Marker()
            sphere.header.frame_id = self.frame_id
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = 'waypoints'
            sphere.id = idx
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = 0.1
            sphere.scale.x = 0.2
            sphere.scale.y = 0.2
            sphere.scale.z = 0.2
            sphere.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            marker_array.markers.append(sphere)

            # 텍스트(TEXT_VIEW_FACING)
            text = Marker()
            text.header.frame_id = self.frame_id
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = 'labels'
            text.id = 1000 + idx
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = 0.4
            text.scale.z = 0.2
            text.text = wp['id']
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            marker_array.markers.append(text)

        # ── (2) 연결선(LINE_LIST) 마커 ─────────────────────
        line_marker = Marker()
        line_marker.header.frame_id = self.frame_id
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = 'connections'
        line_marker.id = 9999
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05
        line_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

        added = set()
        for wp in self.waypoints:
            wp_id = wp['id']
            x1, y1 = float(wp['x']), float(wp['y'])
            for nbr_id in wp.get('neighbors', []):
                key = tuple(sorted([wp_id, nbr_id]))
                if key in added or nbr_id not in self.wp_dict:
                    continue
                added.add(key)

                nbr = self.wp_dict[nbr_id]
                x2, y2 = float(nbr['x']), float(nbr['y'])

                line_marker.points.append(Point(x=x1, y=y1, z=0.15))
                line_marker.points.append(Point(x=x2, y=y2, z=0.15))

        marker_array.markers.append(line_marker)

        # Publish all
        self.pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = WaypointMarkerPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
