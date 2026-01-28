import cv2
import cv_bridge
import numpy
import rclpy.node
import rclpy.qos
import sam3
import sam3.model.sam3_image_processor
import sensor_msgs.msg


class Segmentation(rclpy.node.Node):
    def __init__(self) -> None:
        super().__init__("segmentation")

        use_compressed_image = (
            self.declare_parameter("use_compressed_image", True).get_parameter_value().bool_value
        )
        self.text_prompt = (
            self.declare_parameter("text_prompt", "").get_parameter_value().string_value
        )
        confidence_threshold = (
            self.declare_parameter("confidence_threshold", 0.5).get_parameter_value().double_value
        )

        self.model = sam3.build_sam3_image_model()
        self.processor = sam3.model.sam3_image_processor.Sam3Processor(
            self.model, confidence_threshold=confidence_threshold
        )

        self.cv_bridge = cv_bridge.CvBridge()

        if use_compressed_image:
            self.image_sub = self.create_subscription(
                sensor_msgs.msg.CompressedImage,
                "image/compressed",
                self._image_compressed_callback,
                rclpy.qos.qos_profile_sensor_data,
            )
        else:
            self.image_sub = self.create_subscription(
                sensor_msgs.msg.Image,
                "image",
                self._image_callback,
                rclpy.qos.qos_profile_sensor_data,
            )

    def _image_compressed_callback(self, msg: sensor_msgs.msg.CompressedImage) -> None:
        image = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
        self._run_inference(image)

    def _image_callback(self, msg: sensor_msgs.msg.Image) -> None:
        image = self.cv_bridge.imgmsg_to_cv2(msg)
        self._run_inference(image)

    def _run_inference(self, image: numpy.ndarray) -> None:
        inference_state = self.processor.set_image(image)
        output = self.processor.set_text_prompt(self.text_prompt, inference_state)

        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]


def main(args=None):
    rclpy.init(args=args)
    segmentation = Segmentation()
    rclpy.spin(segmentation)
    segmentation.destroy_node()
    rclpy.shutdown()
