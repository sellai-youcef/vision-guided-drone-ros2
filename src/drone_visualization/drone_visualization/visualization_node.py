import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')
        self.subscription = self.create_subscription(
            String,
            'drone_pos',
            self.position_callback,
            10
        )
        self.get_logger().info('Visualization node started')

    def position_callback(self, msg):
        x, y, depth = msg.data.split(',')
        x = float(x)
        y = float(y)
        depth = float(depth)
        self.get_logger().info(f'Received: x={x} y={y} depth={depth}')
def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    rclpy.spin(node)
    rclpy.shutdown()