import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # Thay bằng topic của camera
            self.image_callback,
            10)
        self.publisher = self.create_publisher(Image, '/lane_detected', 10)
        self.lane_status_publisher = self.create_publisher(Image, '/lane_status', 10)
        self.roi_publisher = self.create_publisher(Image, '/ROI', 10)

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 80, 220)
        return edges

    def find_lane_histogram(self, image):
        height, width = image.shape
        histogram = np.sum(image[int(height / 2):, :], axis=0)
        midpoint = width // 2
        left_base = np.argmax(histogram[:midpoint])  
        right_base = np.argmax(histogram[midpoint:]) + midpoint  
        return left_base, right_base

    def region_of_interest(self, image):
        height, width = image.shape
        mask = np.zeros_like(image)
        polygon = np.array([[
            (0, height),
            (width, height),
            (width * 0.75, height * 0.5),
            (width * 0.25, height * 0.5),
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        cropped = cv2.bitwise_and(image, mask)
        return cropped

    def detect_lines(self, image):
        lines = cv2.HoughLinesP(image, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=150)
        return lines

    def average_slope_intercept(self, image, lines, min_length=100):
        left_lines, right_lines = [], []
        left_weights, right_weights = [], []
        if lines is None:
            return None, None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if length < min_length:
                continue
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            intercept = y1 - slope * x1
            if slope < -.3:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            elif slope > .3:
                right_lines.append((slope, intercept))
                right_weights.append(length)
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
        return left_lane, right_lane

    def make_line_points(self, y1, y2, line, max_length=200):
        if line is None:
            return None
        slope, intercept = line
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        if abs(x2 - x1) > max_length:
            x2 = x1 + np.sign(x2 - x1) * max_length
        return np.array([x1, y1, x2, y2])

    def draw_lines(self, image, lines):
        line_image = np.zeros_like(image)
        if lines is not None:
            left_lane, right_lane = self.average_slope_intercept(image, lines)
            y1 = image.shape[0]
            y2 = int(y1 * 0.7)
            left_line = self.make_line_points(y1, y2, left_lane)
            right_line = self.make_line_points(y1, y2, right_lane)
            if left_line is not None:
                cv2.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 5)
            if right_line is not None:
                cv2.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 5)
        return line_image

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        edges = self.preprocess_image(cv_image)
        left_x, right_x = self.find_lane_histogram(edges)
        roi = self.region_of_interest(edges)
        lines = self.detect_lines(roi)
        if lines is not None:
            line_image = self.draw_lines(cv_image, lines)
        else:
            line_image = np.zeros_like(cv_image)  # Tránh lỗi
        final_output = cv2.addWeighted(cv_image, 0.8, line_image, 1, 1)
        output_msg = self.bridge.cv2_to_imgmsg(final_output, "bgr8")
        output_roi = self.bridge.cv2_to_imgmsg(roi, "mono8")
        self.roi_publisher.publish(output_roi)
        self.publisher.publish(output_msg)

        width = edges.shape[1]
        center_x = width // 2
        lane_center = (left_x + right_x) // 2
        if lane_center < center_x - 100:
            status_msg = "Lệch làn trái"
        elif lane_center > center_x + 100:
            status_msg = "Lệch làn phải"
        else:
            status_msg = "Xe ở giữa làn"
        self.get_logger().info(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
