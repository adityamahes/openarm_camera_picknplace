#!/usr/bin/env python3
# =============================================================================
# sam_perception_node.py
# =============================================================================
# ROS 2 Python node that converts a text prompt and a live RealSense camera
# stream into a 3D pick target point.
#
# Pipeline (triggered by a message on /segment_prompt):
#   1. GroundingDINO  — open-vocabulary object detector
#                       text prompt → one or more 2D bounding boxes in pixel space
#   2. MobileSAM      — promptable segmentation model (SAM variant, mobile-optimised)
#                       bounding box → binary pixel mask of the object
#   3. Depth back-projection
#                       mask + aligned depth image + camera intrinsics → 3D centroid
#   4. Publish        → /pick_target as geometry_msgs/PointStamped (camera frame)
#
# Topics consumed:
#   /camera/color/image_raw                  — RGB image (sensor_msgs/Image)
#   /camera/aligned_depth_to_color/image_raw — 16-bit depth image aligned to RGB
#   /camera/color/camera_info                — intrinsic matrix K
#   /segment_prompt                          — text prompt (std_msgs/String)
#
# Topics published:
#   /pick_target   — 3D centroid of the detected object (geometry_msgs/PointStamped)
# =============================================================================

import os
import numpy as np
import torch
import cv2

# =============================================================================
# Compatibility patch for transformers >= 5.x
# =============================================================================
# GroundingDINO depends on transformers.PreTrainedModel.get_head_mask, a method
# that was removed in transformers 5.0.  If the method is missing we inject a
# backport so GroundingDINO can still run without downgrading transformers.
try:
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, 'get_head_mask'):
        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            # head_mask can be None (no masking) or a 1-D / 2-D tensor.
            # We expand it to the 5-D shape expected by transformer attention layers:
            #   [num_layers, num_heads, 1, seq_len, 1]  (or with chunk dim if chunked).
            if head_mask is not None:
                if head_mask.dim() == 1:
                    # 1-D: one mask value per attention head, same across all layers.
                    # Expand to [num_layers, 1, num_heads, 1, 1].
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    # 2-D: per-layer per-head mask [num_layers, num_heads].
                    # Expand to [num_layers, 1, num_heads, 1, 1].
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                if is_attention_chunked:
                    # Attention chunking adds an extra sequence-chunk dimension.
                    head_mask = head_mask.unsqueeze(-1)
            else:
                # None mask → list of None, one per layer.  Transformer layers
                # check for None to skip masking rather than multiplying by 1.
                head_mask = [None] * num_hidden_layers
            return head_mask
        PreTrainedModel.get_head_mask = _get_head_mask
except ImportError:
    pass   # transformers not installed; GroundingDINO import will fail later anyway

from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge   # converts sensor_msgs/Image ↔ OpenCV numpy arrays

# GroundingDINO image pre-processing transforms (resize, tensor, normalise).
import groundingdino.datasets.transforms as T
# load_model: builds the GroundingDINO model from a config .py + checkpoint .pth.
# predict:    runs inference and returns boxes + logit scores + token phrases.
from groundingdino.util.inference import load_model, predict

# MobileSAM: lightweight SAM variant optimised for CPU/edge.
# vit_t ("tiny" ViT backbone) trades accuracy for speed.
from mobile_sam import sam_model_registry, SamPredictor


class SamPerceptionNode(Node):
    """
    Perceives objects in the scene given a plain-text prompt.

    State held between callbacks:
        latest_rgb   : most recent RGB image message (overwritten each frame)
        latest_depth : most recent aligned depth image message
        camera_info  : camera intrinsic matrix K (stored once on first message)
        dino_model   : loaded GroundingDINO model (None until checkpoint exists)
        sam_predictor: MobileSAM predictor wrapper (None until loaded)
    """

    def __init__(self):
        super().__init__('sam_perception_node')

        # -----------------------------------------------------------------
        # Parameter declarations
        # -----------------------------------------------------------------
        # Declaring parameters before reading them allows the launch file
        # and command-line --ros-args to override any of these values.
        self.declare_parameter('grounding_dino_config', '')
        self.declare_parameter('grounding_dino_checkpoint', '')
        self.declare_parameter('mobile_sam_checkpoint', '')
        self.declare_parameter('box_threshold', 0.35)    # detection confidence cutoff
        self.declare_parameter('text_threshold', 0.25)   # text-to-box match cutoff
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_scale', 0.001)     # RealSense: raw uint16 mm → metres

        # CvBridge translates between ROS Image messages and OpenCV/numpy arrays.
        self.bridge = CvBridge()

        # Image caches — updated by subscription callbacks, read by _prompt_cb.
        # Type annotations (Image | None) document what is stored; they are
        # not enforced at runtime in Python.
        self.latest_rgb:   Image      | None = None
        self.latest_depth: Image      | None = None
        self.camera_info:  CameraInfo | None = None

        # Pick CUDA if available; fall back to CPU otherwise.
        # GroundingDINO and MobileSAM both support both devices.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Running on device: {self.device}')

        # Attempt to load models immediately.  If checkpoint files are missing
        # or paths are not set, _load_models sets both to None and retries
        # on the first prompt.
        self._load_models()

        # -----------------------------------------------------------------
        # Subscriptions and publisher
        # -----------------------------------------------------------------
        rgb_topic   = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        info_topic  = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        # Queue depth 10: keep up to 10 pending messages.  For images arriving
        # at 30 Hz, this gives ~333 ms of buffer before messages are dropped.
        self.create_subscription(Image,      rgb_topic,   self._rgb_cb,   10)
        self.create_subscription(Image,      depth_topic, self._depth_cb, 10)
        self.create_subscription(CameraInfo, info_topic,  self._info_cb,  10)
        # /segment_prompt is published by control_node when the arm reaches
        # the scan pose and is ready for the perception result.
        self.create_subscription(String, '/segment_prompt', self._prompt_cb, 10)

        # /pick_target is consumed by control_node's on_pick_target callback.
        self.target_pub = self.create_publisher(PointStamped, '/pick_target', 10)
        self.get_logger().info('sam_perception_node ready')

    # =========================================================================
    # Model loading
    # =========================================================================

    def _load_models(self):
        """
        Loads GroundingDINO and MobileSAM from checkpoint files.

        Sets self.dino_model and self.sam_predictor to None if any path is
        missing or the files do not exist yet (allows the node to start
        before checkpoints are downloaded, and retry on first prompt).
        """
        dino_cfg  = self.get_parameter('grounding_dino_config').get_parameter_value().string_value
        dino_ckpt = self.get_parameter('grounding_dino_checkpoint').get_parameter_value().string_value
        sam_ckpt  = self.get_parameter('mobile_sam_checkpoint').get_parameter_value().string_value

        # Bail out early if any path is the empty default.
        if not dino_cfg or not dino_ckpt or not sam_ckpt:
            self.get_logger().warn(
                'Model checkpoint paths not set; models will be loaded on first prompt.')
            self.dino_model    = None
            self.sam_predictor = None
            return

        # Verify all files exist before trying to load (avoids cryptic errors).
        missing = [p for p in (dino_cfg, dino_ckpt, sam_ckpt) if not os.path.isfile(p)]
        if missing:
            self.get_logger().warn(
                f'Checkpoint file(s) not found: {missing}. '
                'Models will be loaded on first prompt once files exist.')
            self.dino_model    = None
            self.sam_predictor = None
            return

        # --- GroundingDINO ---
        # load_model reads the Python config file (which defines the model architecture)
        # and loads the .pth weight file.  Returns the model on self.device.
        self.get_logger().info('Loading GroundingDINO …')
        self.dino_model = load_model(dino_cfg, dino_ckpt, device=self.device)

        # --- MobileSAM ---
        # sam_model_registry['vit_t'] builds a MobileSAM with a "tiny" ViT image encoder.
        # sam.eval() disables dropout and batch-norm training behaviour.
        # SamPredictor wraps the model and handles image preprocessing/postprocessing.
        self.get_logger().info('Loading MobileSAM …')
        sam = sam_model_registry['vit_t'](checkpoint=sam_ckpt)
        sam.to(self.device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)

        # Cache inference parameters so they are not re-read on every prompt.
        self.box_threshold  = self.get_parameter('box_threshold').get_parameter_value().double_value
        self.text_threshold = self.get_parameter('text_threshold').get_parameter_value().double_value
        self.depth_scale    = self.get_parameter('depth_scale').get_parameter_value().double_value

    # =========================================================================
    # Image cache callbacks
    # =========================================================================

    def _rgb_cb(self, msg: Image):
        # Cache the latest RGB frame.  The message is a shared reference;
        # we hold a reference to the most recent one and discard the previous.
        self.latest_rgb = msg

    def _depth_cb(self, msg: Image):
        # Cache the latest aligned depth frame (same pixel grid as RGB).
        self.latest_depth = msg

    def _info_cb(self, msg: CameraInfo):
        # Camera intrinsics are static for a given camera resolution mode.
        # Store them once and ignore subsequent messages to save memory.
        if self.camera_info is None:
            self.camera_info = msg

    # =========================================================================
    # Main perception callback
    # =========================================================================

    def _prompt_cb(self, msg: String):
        """
        Triggered when control_node publishes the text prompt on /segment_prompt.
        Runs the full perception pipeline and publishes the 3D target.
        """
        # Retry model loading in case checkpoints arrived after node startup.
        if self.dino_model is None:
            self._load_models()
        if self.dino_model is None:
            self.get_logger().error('Models not loaded')
            return

        # Require at least one frame from all three camera topics.
        if self.latest_rgb is None or self.latest_depth is None or self.camera_info is None:
            self.get_logger().error('No camera data received yet')
            return

        text_prompt = msg.data    # e.g. "red tool"

        # Snapshot the current frame pair.  Storing local references prevents
        # the cache from being overwritten by new frames mid-pipeline.
        rgb_msg   = self.latest_rgb
        depth_msg = self.latest_depth

        # Convert ROS Image messages to OpenCV numpy arrays.
        # 'bgr8'        → uint8 H×W×3 array in BGR order (OpenCV default).
        # 'passthrough' → preserves the raw encoding (uint16 for 16-bit depth).
        bgr   = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='passthrough').astype(np.float32)

        # Step 1: Detect the object with GroundingDINO.
        # Returns an array of bounding boxes in pixel coordinates [x1, y1, x2, y2],
        # sorted by detection confidence (highest first).
        boxes_px = self._run_grounding_dino(bgr, text_prompt)
        if boxes_px is None or len(boxes_px) == 0:
            self.get_logger().error(f"No object matching '{text_prompt}' found")
            return

        # Step 2: Segment the highest-confidence detection with MobileSAM.
        # boxes_px[0] is the best bounding box (most confident GroundingDINO hit).
        # Returns a boolean H×W mask where True = object pixel.
        mask = self._run_mobile_sam(bgr, boxes_px[0])
        if mask is None:
            self.get_logger().error('SAM segmentation failed')
            return

        # Step 3: Back-project the mask to a 3D centroid.
        # Returns [X, Y, Z] in the camera optical frame, in metres.
        point_3d = self._mask_to_3d(mask, depth, self.camera_info)
        if point_3d is None:
            self.get_logger().error('Could not compute 3D position (invalid depth in mask)')
            return

        # Build and publish the PointStamped.
        # header.frame_id comes from the RGB image (e.g. "camera_color_optical_frame").
        # control_node's TF2 transform step converts this to the planning frame.
        pt = PointStamped()
        pt.header       = rgb_msg.header   # inherit frame_id and stamp from the source image
        pt.point.x, pt.point.y, pt.point.z = (
            float(point_3d[0]), float(point_3d[1]), float(point_3d[2]))

        self.target_pub.publish(pt)
        self.get_logger().info(
            f"Segmented '{text_prompt}' → "
            f"3D ({pt.point.x:.3f}, {pt.point.y:.3f}, {pt.point.z:.3f}) m")

    # =========================================================================
    # GroundingDINO inference
    # =========================================================================

    def _run_grounding_dino(self, bgr: np.ndarray, text_prompt: str):
        """
        Runs GroundingDINO on the given BGR image with the text prompt.

        Returns an (N, 4) int32 numpy array of bounding boxes [x1, y1, x2, y2]
        in pixel coordinates, sorted by confidence (descending).
        Returns None if no boxes pass the thresholds.
        """
        # GroundingDINO expects RGB; OpenCV imread/CvBridge gives BGR.
        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Convert to a PIL Image because the GroundingDINO transform pipeline
        # (T.Compose) works with PIL images.
        pil_img = PILImage.fromarray(rgb)

        # Standard GroundingDINO pre-processing:
        #   RandomResize: resize so the shortest side is 800 px (max 1333 px long side).
        #   ToTensor:     H×W×C uint8 PIL → C×H×W float32 tensor in [0, 1].
        #   Normalize:    subtract ImageNet mean, divide by ImageNet std.
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # transform returns (tensor, target_dict); we only need the tensor.
        img_tensor, _ = transform(pil_img, None)
        img_tensor = img_tensor.to(self.device)   # move to GPU if available

        # GroundingDINO requires the caption to end with a period.
        caption = text_prompt.strip().rstrip('.') + '.'

        with torch.no_grad():   # disable gradient computation (inference only)
            # predict returns:
            #   boxes  — (N, 4) tensor, normalised [cx, cy, w, h] in [0, 1]
            #   logits — (N,)   tensor, confidence scores
            #   _      — (N,)   list of matched text phrases (unused here)
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

        h, w = bgr.shape[:2]   # original image dimensions (before resize)

        # Sort detections by confidence, highest first, so boxes_px[0] is best.
        order = torch.argsort(logits, descending=True)
        boxes = boxes[order]   # re-order rows of the (N, 4) tensor

        # Convert from normalised [cx, cy, bw, bh] to pixel [x1, y1, x2, y2].
        # cx, cy are the box centre in [0, 1]; bw, bh are the box width/height in [0, 1].
        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = ((cx - bw / 2) * w).clamp(0, w).int()
        y1 = ((cy - bh / 2) * h).clamp(0, h).int()
        x2 = ((cx + bw / 2) * w).clamp(0, w).int()
        y2 = ((cy + bh / 2) * h).clamp(0, h).int()

        # Stack into (N, 4) and move to CPU numpy for MobileSAM (which uses numpy).
        return torch.stack([x1, y1, x2, y2], dim=1).cpu().numpy()

    # =========================================================================
    # MobileSAM inference
    # =========================================================================

    def _run_mobile_sam(self, bgr: np.ndarray, box_xyxy: np.ndarray):
        """
        Segments the object inside box_xyxy using MobileSAM.

        box_xyxy: 1-D int array [x1, y1, x2, y2] in pixel coordinates.
        Returns a boolean H×W numpy array (True = object pixel), or None on failure.
        """
        # MobileSAM's image encoder expects RGB.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # set_image encodes the image with the ViT backbone and caches the
        # feature map.  This is the expensive step (~200 ms on CPU for vit_t).
        self.sam_predictor.set_image(rgb)

        x1, y1, x2, y2 = box_xyxy.tolist()

        # predict runs the SAM mask decoder given the bounding-box prompt.
        # point_coords/point_labels are optional additional prompts; we don't use them.
        # box: float64 [x1, y1, x2, y2] prompt in the ORIGINAL image pixel space.
        # multimask_output=False → return the single best mask instead of 3 candidates.
        # Returns:
        #   masks  — (1, H, W) bool array when multimask_output=False
        #   _      — (1,) confidence scores (unused)
        #   _      — (1, 4) low-res logit masks (unused)
        masks, _, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([x1, y1, x2, y2], dtype=float),
            multimask_output=False,
        )
        # masks[0] is the (H, W) boolean mask for the first (only) output.
        return masks[0]

    # =========================================================================
    # Depth back-projection
    # =========================================================================

    def _mask_to_3d(self, mask: np.ndarray, depth: np.ndarray, info: CameraInfo):
        """
        Converts a pixel mask + aligned depth image into a 3D centroid.

        The camera intrinsic matrix K is stored in CameraInfo.k as a flat
        row-major 9-element array:
            K = [ fx  0  cx ]
                [  0 fy  cy ]
                [  0  0   1 ]
        so:
            fx = info.k[0],  cx = info.k[2]
            fy = info.k[4],  cy = info.k[5]

        Back-projection formula (pinhole camera model):
            Z = depth_metres                      (mean over valid masked pixels)
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

        where (u, v) are pixel column and row, and (X, Y, Z) are in the
        camera optical frame (Z forward, X right, Y down).

        Returns [X, Y, Z] in metres, or None if no valid depth pixels exist.
        """
        # Extract focal lengths and principal point from the intrinsic matrix.
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]

        # np.where(mask) returns (row_indices, col_indices) for True pixels.
        # ys = row coordinates, xs = column coordinates.
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None    # empty mask (SAM found nothing inside the box)

        # Index the depth image at the masked pixel locations.
        # depth[ys, xs] is an array of raw uint16 values (millimetres for RealSense).
        # Multiply by depth_scale (0.001) to convert mm → metres.
        depths_m = depth[ys, xs] * self.depth_scale

        # Filter out invalid depth readings:
        #   < 0.05 m: too close (sensor noise floor / arm self-occlusion)
        #   > 2.5 m: beyond typical table workspace depth
        valid = (depths_m > 0.05) & (depths_m < 2.5)
        if valid.sum() == 0:
            return None    # all depth pixels are invalid (sensor failure or out of range)

        # Keep only the valid-depth pixels.  Cast to float64 for numerical precision.
        xs_v = xs[valid].astype(np.float64)
        ys_v = ys[valid].astype(np.float64)
        d_v  = depths_m[valid].astype(np.float64)

        # Back-project each valid pixel to 3D, then average.
        # This gives the mean 3D position of the object's visible surface.
        X = np.mean((xs_v - cx) * d_v / fx)
        Y = np.mean((ys_v - cy) * d_v / fy)
        Z = np.mean(d_v)

        return [X, Y, Z]


def main(args=None):
    rclpy.init(args=args)
    node = SamPerceptionNode()
    # spin() blocks here, delivering callbacks (image frames, prompts) until Ctrl-C.
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
