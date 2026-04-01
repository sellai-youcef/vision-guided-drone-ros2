import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import depthai as dai
import cv2
import numpy as np
from ultralytics import YOLO

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        self.model = YOLO('/home/youcef-sellai/vision_guided_drone_ws/src/drone_detection/drone_detection/runs/detect/drone_detector3/weights/best.pt')
        self.publisher_=self.create_publisher(String, 'drone_pos', 10 )

        self.pipeline = dai.Pipeline()
        self.cam = self.pipeline.create(dai.node.Camera).build()
        self.color_output = self.cam.requestOutput((640,480), type=dai.ImgFrame.Type.BGR888p)

        self.stereo = self.pipeline.create(dai.node.StereoDepth)
        
        self.left_cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        self.right_cam = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

        left_out = self.left_cam.requestOutput((640,400), dai.ImgFrame.Type.GRAY8)
        right_out = self.right_cam.requestOutput((640,400), dai.ImgFrame.Type.GRAY8)

        left_out.link(self.stereo.left)
        right_out.link(self.stereo.right)

        self.stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setSubpixel(True)

        depth_out = self.stereo.depth.createOutputQueue()
        self.color_queue = self.color_output.createOutputQueue()
        self.depth_queue = depth_out

        self.pipeline.start()
        self.get_logger().info('Detection node started')

        self.timer = self.create_timer(0.1, self.detect_callback)

    def detect_callback(self):
        color_frame = self.color_queue.get().getCvFrame()
        depth_frame = self.depth_queue.get().getFrame()

        results = self.model(color_frame, verbose=False)

        for box in results[0].boxes:
            if box.conf[0] > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                region = depth_frame[max(0,cy-3):cy+3, max(0,cx-3):cx+3]
                valid = region[region > 0]
                if len(valid) == 0:
                    continue
                depth_meters = float(np.median(valid)) / 1000.0

                if depth_meters > 10 or depth_meters <= 0:
                    continue

                msg = String()
                msg.data = f"{cx},{cy},{depth_meters:.2f}"
                self.publisher_.publish(msg)
                self.get_logger().info(f'Drone at: x={cx} y={cy} depth={depth_meters:.2f}m')

        annotated = results[0].plot()
        cv2.imshow("Detection", annotated)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()