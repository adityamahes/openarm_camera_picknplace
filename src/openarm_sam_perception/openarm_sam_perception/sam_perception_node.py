#!/usr/bin/env python3
"""
sam_perception_node.py
======================
ROS 2 Python node that provides the /segment_object service.

Pipeline (per service call):
  1. Grab the latest RGB + aligned-depth frame from the RealSense camera.
  2. Run GroundingDINO with the text prompt to get a pixel bounding box.
  3. Run MobileSAM with that box to get a binary segmentation mask.
  4. Back-project the masked depth pixels through the camera intrinsics
     to compute a 3D centroid in the camera optical frame.
  5. Return the centroid as a geometry_msgs/PointStamped.

The caller (openarm_perception_control) is responsible for transforming
the returned point into its own planning frame via TF.
"""
import numpy as np
import torch
import cv2
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

from openarm_perception_msgs.srv import SegmentObject

# GroundingDINO — open-vocabulary object detector (text → bounding box)
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict

# MobileSAM — lightweight Segment Anything model (box → pixel mask)
from mobile_sam import sam_model_registry, SamPredictor


class SamPerceptionNode(Node):
    """
    Provides the /segment_object ROS 2 service.

    Subscribers
    -----------
    <rgb_topic>         sensor_msgs/Image       Latest colour frame
    <depth_topic>       sensor_msgs/Image       Aligned depth frame (mm, uint16)
    <camera_info_topic> sensor_msgs/CameraInfo  Camera intrinsics (read once)

    Services
    --------
    /segment_object     openarm_perception_msgs/SegmentObject
        Request:  text_prompt  (e.g. "red tool")
        Response: success, message, target_position (PointStamped), bounding_box
    """

    def __init__(self):
        super().__init__('sam_perception_node')

        # ------------------------------------------------------------------
        # Parameters — override in launch file or via ros2 param set
        # ------------------------------------------------------------------
        self.declare_parameter('grounding_dino_config', '')       # path to .py config
        self.declare_parameter('grounding_dino_checkpoint', '')   # path to .pth weights
        self.declare_parameter('mobile_sam_checkpoint', '')       # path to .pt weights
        self.declare_parameter('box_threshold', 0.35)   # min GroundingDINO box confidence
        self.declare_parameter('text_threshold', 0.25)  # min GroundingDINO text confidence
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        # RealSense publishes depth in millimetres (uint16); scale to metres
        self.declare_parameter('depth_scale', 0.001)

        self.bridge = CvBridge()

        # Latest frames — updated every subscription callback
        self.latest_rgb: Image | None = None
        self.latest_depth: Image | None = None
        self.camera_info: CameraInfo | None = None   # stored once; intrinsics don't change

        # Use GPU if available, otherwise fall back to CPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Running on device: {self.device}')

        self._load_models()

        # ------------------------------------------------------------------
        # Subscriptions
        # ------------------------------------------------------------------
        rgb_topic   = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        info_topic  = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        self.create_subscription(Image,      rgb_topic,   self._rgb_cb,   10)
        self.create_subscription(Image,      depth_topic, self._depth_cb, 10)
        self.create_subscription(CameraInfo, info_topic,  self._info_cb,  10)

        # ------------------------------------------------------------------
        # Service server
        # ------------------------------------------------------------------
        self.create_service(SegmentObject, 'segment_object', self._handle_segment)
        self.get_logger().info('sam_perception_node ready')

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self):
        """Load GroundingDINO and MobileSAM from the configured checkpoint paths."""
        dino_cfg  = self.get_parameter('grounding_dino_config').get_parameter_value().string_value
        dino_ckpt = self.get_parameter('grounding_dino_checkpoint').get_parameter_value().string_value
        sam_ckpt  = self.get_parameter('mobile_sam_checkpoint').get_parameter_value().string_value

        self.get_logger().info('Loading GroundingDINO …')
        self.dino_model = load_model(dino_cfg, dino_ckpt, device=self.device)

        self.get_logger().info('Loading MobileSAM …')
        # 'vit_t' is the MobileSAM-specific tiny ViT backbone key
        sam = sam_model_registry['vit_t'](checkpoint=sam_ckpt)
        sam.to(self.device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)

        # Cache threshold params so they're not re-fetched on every service call
        self.box_threshold  = self.get_parameter('box_threshold').get_parameter_value().double_value
        self.text_threshold = self.get_parameter('text_threshold').get_parameter_value().double_value
        self.depth_scale    = self.get_parameter('depth_scale').get_parameter_value().double_value

    # ------------------------------------------------------------------
    # Subscription callbacks — just store the latest message
    # ------------------------------------------------------------------

    def _rgb_cb(self, msg: Image):
        """Cache the most recent colour frame for use by the next service call."""
        self.latest_rgb = msg

    def _depth_cb(self, msg: Image):
        """Cache the most recent aligned-depth frame."""
        self.latest_depth = msg

    def _info_cb(self, msg: CameraInfo):
        """Store camera intrinsics once; they don't change while the camera is running."""
        if self.camera_info is None:
            self.camera_info = msg

    # ------------------------------------------------------------------
    # Service handler — orchestrates the full perception pipeline
    # ------------------------------------------------------------------

    def _handle_segment(self, request, response):
        """
        Handle a /segment_object service call.

        Steps:
          1. Validate that camera data is available.
          2. Run GroundingDINO to find bounding box(es) for the text prompt.
          3. Run MobileSAM on the top-confidence box to get a pixel mask.
          4. Convert the mask + depth to a 3D point in the camera frame.
        """
        if self.latest_rgb is None or self.latest_depth is None or self.camera_info is None:
            response.success = False
            response.message = 'No camera data received yet'
            return response

        # Snapshot the current frames so the callbacks can keep updating
        rgb_msg   = self.latest_rgb
        depth_msg = self.latest_depth

        # Convert ROS image messages to OpenCV arrays
        bgr   = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='passthrough').astype(np.float32)

        # ---- Step 1: text → bounding boxes --------------------------------
        boxes_px = self._run_grounding_dino(bgr, request.text_prompt)
        if boxes_px is None or len(boxes_px) == 0:
            response.success = False
            response.message = f"No object matching '{request.text_prompt}' found"
            return response

        # ---- Step 2: bounding box → binary mask ---------------------------
        # boxes_px is sorted by confidence; use the highest-confidence detection
        mask = self._run_mobile_sam(bgr, boxes_px[0])
        if mask is None:
            response.success = False
            response.message = 'SAM segmentation failed'
            return response

        # ---- Step 3: mask + depth → 3D centroid --------------------------
        point_3d = self._mask_to_3d(mask, depth, self.camera_info)
        if point_3d is None:
            response.success = False
            response.message = 'Could not compute 3D position (invalid depth in mask)'
            return response

        # Build the PointStamped in the camera frame (same header as the image)
        pt = PointStamped()
        pt.header       = rgb_msg.header
        pt.point.x, pt.point.y, pt.point.z = (
            float(point_3d[0]), float(point_3d[1]), float(point_3d[2]))

        response.success          = True
        response.message          = 'OK'
        response.target_position  = pt
        response.bounding_box     = [float(v) for v in boxes_px[0]]

        self.get_logger().info(
            f"Segmented '{request.text_prompt}' → "
            f"3D ({pt.point.x:.3f}, {pt.point.y:.3f}, {pt.point.z:.3f}) m")
        return response

    # ------------------------------------------------------------------
    # GroundingDINO — text-prompted object detection
    # ------------------------------------------------------------------

    def _run_grounding_dino(self, bgr: np.ndarray, text_prompt: str):
        """
        Run GroundingDINO on the image to find pixel bounding boxes that match
        the text prompt.

        Returns a numpy array of shape (N, 4) in pixel xyxy order, sorted by
        descending confidence, or None if no detections pass the threshold.
        """
        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        # GroundingDINO expects normalised, pre-processed tensor input
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor, _ = transform(pil_img, None)
        img_tensor = img_tensor.to(self.device)

        # GroundingDINO performs better when the caption ends with a period
        caption = text_prompt.strip().rstrip('.') + '.'

        with torch.no_grad():
            # boxes: (N, 4) normalised cx,cy,w,h
            # logits: (N,) confidence scores
            boxes, logits, _ = predict(
                model=self.dino_model,
                image=img_tensor,
                caption=caption,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device,
            )

        if len(boxes) == 0:
            return None

        h, w = bgr.shape[:2]

        # Sort by confidence (highest first) before converting coordinates
        order  = torch.argsort(logits, descending=True)
        boxes  = boxes[order]

        # Convert normalised (cx, cy, w, h) → pixel (x1, y1, x2, y2)
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = ((cx - bw / 2) * w).clamp(0, w).int()
        y1 = ((cy - bh / 2) * h).clamp(0, h).int()
        x2 = ((cx + bw / 2) * w).clamp(0, w).int()
        y2 = ((cy + bh / 2) * h).clamp(0, h).int()

        return torch.stack([x1, y1, x2, y2], dim=1).cpu().numpy()

    # ------------------------------------------------------------------
    # MobileSAM — bounding-box-prompted segmentation
    # ------------------------------------------------------------------

    def _run_mobile_sam(self, bgr: np.ndarray, box_xyxy: np.ndarray):
        """
        Run MobileSAM with a single bounding-box prompt to produce a binary mask.

        Parameters
        ----------
        bgr      : HxWx3 uint8 BGR image
        box_xyxy : [x1, y1, x2, y2] in pixel coordinates

        Returns a boolean HxW numpy mask, or None on failure.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # set_image encodes the image once; predict can be called multiple times
        self.sam_predictor.set_image(rgb)

        x1, y1, x2, y2 = box_xyxy.tolist()
        # multimask_output=False → one mask instead of three candidates
        masks, _, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([x1, y1, x2, y2], dtype=float),
            multimask_output=False,
        )
        # masks shape: (1, H, W) bool array
        return masks[0]

    # ------------------------------------------------------------------
    # Depth back-projection — mask → 3D centroid
    # ------------------------------------------------------------------

    def _mask_to_3d(self, mask: np.ndarray, depth: np.ndarray, info: CameraInfo):
        """
        Convert a 2D binary mask + aligned depth image into a 3D point.

        Uses the pinhole camera model:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth (in metres)

        Returns the mean [X, Y, Z] over all valid masked pixels, or None if
        there are no pixels with valid depth readings.
        """
        # Unpack intrinsics from the flat row-major K matrix
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]

        # Pixel coordinates of every True pixel in the mask
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None

        # Convert raw depth values (mm) to metres
        depths_m = depth[ys, xs] * self.depth_scale

        # Filter out zero/invalid depth and readings outside a realistic workspace
        valid = (depths_m > 0.05) & (depths_m < 2.5)
        if valid.sum() == 0:
            return None

        xs_v = xs[valid].astype(np.float64)
        ys_v = ys[valid].astype(np.float64)
        d_v  = depths_m[valid].astype(np.float64)

        # Back-project each pixel to 3D and average for the centroid
        X = np.mean((xs_v - cx) * d_v / fx)
        Y = np.mean((ys_v - cy) * d_v / fy)
        Z = np.mean(d_v)

        return [X, Y, Z]


def main(args=None):
    rclpy.init(args=args)
    node = SamPerceptionNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
