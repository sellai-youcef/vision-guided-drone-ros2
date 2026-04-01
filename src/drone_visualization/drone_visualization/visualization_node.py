import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np

class VisualizationNode(Node):
    def __init__(self):
        # initialize ROS node, subscribe topic, set state, start periodic update
        super().__init__('visualization_node')
        self.subscription = self.create_subscription(
            String,
            'drone_pos',
            self.position_callback,
            10
        )
        
        self.drone_x = None
        self.drone_y = None
        self.drone_depth = None
        
        self.map_width = 600
        self.map_height = 600
        self.map_image = None
        
        self.camera_y = 20
        self.max_depth = 8.0
        
        self.timer = self.create_timer(0.033, self.update_visualization)
        
        # Kalman filter state
        self.kalman_depth = None
        self.kalman_velocity = 0.0
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]])  # initial covariance
        self.Q = np.array([[0.001, 0.0], [0.0, 0.001]])  # process noise
        self.R = 10.0  # measurement noise
        self.dt = 0.033  # time step
        
        self.get_logger().info('Visualization node started')

    def position_callback(self, msg):
        # parse incoming position string, store x/y/depth, and log
        try:
            x, y, depth = msg.data.split(',')
            self.drone_x = float(x)
            self.drone_y = float(y)
            self.drone_depth = float(depth)
            self.get_logger().info(f'Received: x={self.drone_x} y={self.drone_y} depth={self.drone_depth}')
            
            # Kalman filter update
            self.kalman_update(self.drone_depth)
        except Exception as e:
            self.get_logger().error(f'Error parsing position: {e}')

    def kalman_update(self, z):
        # update Kalman filter with new measurement
        if self.kalman_depth is None:
            self.kalman_depth = z
            self.kalman_velocity = 0.0
        else:
            H = np.array([1.0, 0.0])
            y = z - (H @ np.array([self.kalman_depth, self.kalman_velocity]))
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T / S
            state = np.array([self.kalman_depth, self.kalman_velocity]) + K * y
            self.kalman_depth, self.kalman_velocity = state
            self.P = (np.eye(2) - np.outer(K, H)) @ self.P

    def update_visualization(self):
        # draw map, camera marker, and depth-based drone marker; show window
        self.map_image = np.ones((self.map_height, self.map_width, 3), dtype=np.uint8) * 255
        
        cam_x = self.map_width // 2
        cv2.rectangle(self.map_image, (cam_x - 10, self.camera_y - 10), (cam_x + 10, self.camera_y + 10), (255, 0, 0), -1)
        
        # Kalman predict
        if self.kalman_depth is not None:
            self.kalman_depth += self.kalman_velocity * self.dt
            F = np.array([[1.0, self.dt], [0.0, 1.0]])
            self.P = F @ self.P @ F.T + self.Q
        
        if self.kalman_depth is not None:
            depth_pixel = int((self.kalman_depth / self.max_depth) * (self.map_height - self.camera_y - 20))
            drone_map_y = self.camera_y + depth_pixel
            
            drone_map_y = max(self.camera_y, min(self.map_height - 1, drone_map_y))
            
            cv2.rectangle(self.map_image, (cam_x - 15, drone_map_y - 15), (cam_x + 15, drone_map_y + 15), (0, 0, 255), -1)
            
            cv2.putText(self.map_image, f'Depth: {self.kalman_depth:.2f}m', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.imshow("Room Map", self.map_image)
        cv2.waitKey(30)
    
def main(args=None):
    # start ROS, create node, spin, and shutdown cleanly
    rclpy.init(args=args)
    node = VisualizationNode()
    rclpy.spin(node)
    rclpy.shutdown()