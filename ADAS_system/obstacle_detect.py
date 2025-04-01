import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')

        # Đăng ký subscriber nhận dữ liệu từ LiDAR
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        self.subscription  

        # Ngưỡng khoảng cách để phát hiện vật cản (m)
        self.obstacle_distance_threshold = 0.3  

        self.get_logger().info("Obstacle Detector Initialized.")

    def lidar_callback(self, scan_data):
        """ Xử lý dữ liệu LiDAR để lọc góc và phát hiện vật cản """
        ranges = np.array(scan_data.ranges)  # Dữ liệu khoảng cách (m)
        angle_min = scan_data.angle_min      # Góc bắt đầu (rad)
        angle_increment = scan_data.angle_increment  # Bước nhảy (rad)

        # Tính toán góc tương ứng (độ)
        angles = np.arange(len(ranges)) * angle_increment + angle_min
        angles_deg = np.degrees(angles)  

        # Lọc góc 135° → 180° và -180° → -135°
        valid_indices = (angles_deg >= 135) | (angles_deg <= -135)
        filtered_ranges = ranges[valid_indices]
        filtered_angles_deg = angles_deg[valid_indices]

        # Phát hiện vật cản trong phạm vi
        obstacles = [
            (angle, distance)
            for angle, distance in zip(filtered_angles_deg, filtered_ranges)
            if 0.1 < distance < self.obstacle_distance_threshold
        ]

        # Tìm vật cản gần nhất
        if obstacles:
            closest_obstacle = min(obstacles, key=lambda x: x[1])
            self.get_logger().warn(f"Obstacle detected at {closest_obstacle[0]:.1f}° - Distance: {closest_obstacle[1]:.2f}m")
        else:
            self.get_logger().info("No obstacles detected in 135° → 180° and -180° → -135° region.")

def main(args=None):
    rclpy.init(args=args)
    obstacle_detector = ObstacleDetector()

    try:
        rclpy.spin(obstacle_detector)
    except KeyboardInterrupt:
        pass
    finally:
        obstacle_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

