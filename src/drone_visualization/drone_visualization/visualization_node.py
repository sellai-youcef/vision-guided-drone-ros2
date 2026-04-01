import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import numpy as np

class VisualizationNode(Node):
    def __init__(self):
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
        self.max_x = 10.0  # total range for x, from 0 to 10 meters, center at 5
        
        self.timer = self.create_timer(0.033, self.update_visualization)
        
        # Kalman filter state for depth
        self.kalman_depth = None
        self.kalman_velocity = 0.0
        self.P_depth = np.array([[1.0, 0.0], [0.0, 1.0]])  # initial covariance
        self.Q = np.array([[0.001, 0.0], [0.0, 0.001]])  # process noise
        self.R = 10.0  # measurement noise
        self.dt = 0.033  # time step
        
        # Kalman filter state for x
        self.kalman_x = None
        self.kalman_vx = 0.0
        self.P_x = np.array([[1.0, 0.0], [0.0, 1.0]])  # initial covariance
        
        self.camera_width = 640  # assumed camera image width in pixels
        
        self.get_logger().info('Visualization node started')

    def position_callback(self, msg):
        # parse incoming position string, convert pixels to meters, store x/y/depth, and log
        try:
            x_pixel, y_pixel, depth_m = msg.data.split(',')
            # Depth is already in meters
            self.drone_depth = float(depth_m)
            # Convert x from pixels to meters: far right (640) is 0m, far left (0) is 10m
            self.drone_x = 10.0 - (float(x_pixel) / self.camera_width) * 10.0
            self.drone_y = float(y_pixel)  # y not converted, assuming not used for visualization
            self.get_logger().info(f'Received: x={self.drone_x} y={self.drone_y} depth={self.drone_depth}')
            
            # Kalman filter update for depth
            self.kalman_update_depth(self.drone_depth)
            # Kalman filter update for x
            self.kalman_update_x(self.drone_x)
        except Exception as e:
            self.get_logger().error(f'Error parsing position: {e}')

    def kalman_update_depth(self, z):
        # update Kalman filter with new depth measurement
        if self.kalman_depth is None:
            self.kalman_depth = z
            self.kalman_velocity = 0.0
        else:
            H = np.array([1.0, 0.0])
            y = z - (H @ np.array([self.kalman_depth, self.kalman_velocity]))
            S = H @ self.P_depth @ H.T + self.R
            K = self.P_depth @ H.T / S
            state = np.array([self.kalman_depth, self.kalman_velocity]) + K * y
            self.kalman_depth, self.kalman_velocity = state
            self.P_depth = (np.eye(2) - np.outer(K, H)) @ self.P_depth

    def kalman_update_x(self, z):
        # update Kalman filter with new x measurement
        if self.kalman_x is None:
            self.kalman_x = z
            self.kalman_vx = 0.0
        else:
            H = np.array([1.0, 0.0])
            y = z - (H @ np.array([self.kalman_x, self.kalman_vx]))
            S = H @ self.P_x @ H.T + self.R
            K = self.P_x @ H.T / S
            state = np.array([self.kalman_x, self.kalman_vx]) + K * y
            self.kalman_x, self.kalman_vx = state
            self.P_x = (np.eye(2) - np.outer(K, H)) @ self.P_x

    def update_visualization(self):
        # draw map, camera marker, and depth/x-based drone marker; show window
        self.map_image = np.ones((self.map_height, self.map_width, 3), dtype=np.uint8) * 255
        
        cam_x = self.map_width // 2
        cv2.rectangle(self.map_image, (cam_x - 10, self.camera_y - 10), (cam_x + 10, self.camera_y + 10), (255, 0, 0), -1)
        
        # Kalman predict for depth
        if self.kalman_depth is not None:
            self.kalman_depth += self.kalman_velocity * self.dt
            F = np.array([[1.0, self.dt], [0.0, 1.0]])
            self.P_depth = F @ self.P_depth @ F.T + self.Q
        
        # Kalman predict for x
        if self.kalman_x is not None:
            self.kalman_x += self.kalman_vx * self.dt
            F = np.array([[1.0, self.dt], [0.0, 1.0]])
            self.P_x = F @ self.P_x @ F.T + self.Q
        
        if self.kalman_depth is not None and self.kalman_x is not None:
            depth_pixel = int((self.kalman_depth / self.max_depth) * (self.map_height - self.camera_y - 20))
            drone_map_y = self.camera_y + depth_pixel
            drone_map_y = max(self.camera_y, min(self.map_height - 1, drone_map_y))
            
            x_pixel = int(((self.kalman_x - 5.0) / 5.0) * (self.map_width / 2))
            drone_map_x = cam_x + x_pixel
            drone_map_x = max(0, min(self.map_width - 1, drone_map_x))
            
            cv2.rectangle(self.map_image, (drone_map_x - 15, drone_map_y - 15), (drone_map_x + 15, drone_map_y + 15), (0, 0, 255), -1)
            
            cv2.putText(self.map_image, f'Depth: {self.kalman_depth:.2f}m, X: {self.kalman_x:.2f}m', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.imshow("Room Map", self.map_image)
        cv2.waitKey(30)
    
def main(args=None):
    # start ROS, create node, spin, and shutdown cleanly
    rclpy.init(args=args)
    node = VisualizationNode()
    rclpy.spin(node)
    rclpy.shutdown()