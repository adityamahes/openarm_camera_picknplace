#!/usr/bin/env python3
import os
import numpy as np
import torch
import cv2

try:
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, 'get_head_mask'):
        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                if is_attention_chunked:
                    head_mask = head_mask.unsqueeze(-1)
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask
        PreTrainedModel.get_head_mask = _get_head_mask
except ImportError:
    pass
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from cv_bridge import CvBridge

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, predict

from mobile_sam import sam_model_registry, SamPredictor


class SamPerceptionNode(Node):

    def __init__(self):
        super().__init__('sam_perception_node')

        self.declare_parameter('grounding_dino_config', '')
        self.declare_parameter('grounding_dino_checkpoint', '')
        self.declare_parameter('mobile_sam_checkpoint', '')
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        self.declare_parameter('rgb_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('depth_scale', 0.001)

        self.bridge = CvBridge()

        self.latest_rgb: Image | None = None
        self.latest_depth: Image | None = None
        self.camera_info: CameraInfo | None = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Running on device: {self.device}')

        self._load_models()

        rgb_topic   = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        info_topic  = self.get_parameter('camera_info_topic').get_parameter_value().string_value

        self.create_subscription(Image,      rgb_topic,   self._rgb_cb,   10)
        self.create_subscription(Image,      depth_topic, self._depth_cb, 10)
        self.create_subscription(CameraInfo, info_topic,  self._info_cb,  10)
        self.create_subscription(String, '/segment_prompt', self._prompt_cb, 10)

        self.target_pub = self.create_publisher(PointStamped, '/pick_target', 10)
        self.get_logger().info('sam_perception_node ready')

    def _load_models(self):
        dino_cfg  = self.get_parameter('grounding_dino_config').get_parameter_value().string_value
        dino_ckpt = self.get_parameter('grounding_dino_checkpoint').get_parameter_value().string_value
        sam_ckpt  = self.get_parameter('mobile_sam_checkpoint').get_parameter_value().string_value

        if not dino_cfg or not dino_ckpt or not sam_ckpt:
            self.get_logger().warn(
                'Model checkpoint paths not set; models will be loaded on first prompt.')
            self.dino_model = None
            self.sam_predictor = None
            return

        missing = [p for p in (dino_cfg, dino_ckpt, sam_ckpt) if not os.path.isfile(p)]
        if missing:
            self.get_logger().warn(
                f'Checkpoint file(s) not found: {missing}. '
                'Models will be loaded on first prompt once files exist.')
            self.dino_model = None
            self.sam_predictor = None
            return

        self.get_logger().info('Loading GroundingDINO …')
        self.dino_model = load_model(dino_cfg, dino_ckpt, device=self.device)

        self.get_logger().info('Loading MobileSAM …')
        sam = sam_model_registry['vit_t'](checkpoint=sam_ckpt)
        sam.to(self.device)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)

        self.box_threshold  = self.get_parameter('box_threshold').get_parameter_value().double_value
        self.text_threshold = self.get_parameter('text_threshold').get_parameter_value().double_value
        self.depth_scale    = self.get_parameter('depth_scale').get_parameter_value().double_value

    def _rgb_cb(self, msg: Image):
        self.latest_rgb = msg

    def _depth_cb(self, msg: Image):
        self.latest_depth = msg

    def _info_cb(self, msg: CameraInfo):
        if self.camera_info is None:
            self.camera_info = msg

    def _prompt_cb(self, msg: String):
        if self.dino_model is None:
            self._load_models()
        if self.dino_model is None:
            self.get_logger().error('Models not loaded')
            return

        if self.latest_rgb is None or self.latest_depth is None or self.camera_info is None:
            self.get_logger().error('No camera data received yet')
            return

        text_prompt = msg.data
        rgb_msg   = self.latest_rgb
        depth_msg = self.latest_depth

        bgr   = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        depth = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='passthrough').astype(np.float32)

        boxes_px = self._run_grounding_dino(bgr, text_prompt)
        if boxes_px is None or len(boxes_px) == 0:
            self.get_logger().error(f"No object matching '{text_prompt}' found")
            return

        mask = self._run_mobile_sam(bgr, boxes_px[0])
        if mask is None:
            self.get_logger().error('SAM segmentation failed')
            return

        point_3d = self._mask_to_3d(mask, depth, self.camera_info)
        if point_3d is None:
            self.get_logger().error('Could not compute 3D position (invalid depth in mask)')
            return

        pt = PointStamped()
        pt.header       = rgb_msg.header
        pt.point.x, pt.point.y, pt.point.z = (
            float(point_3d[0]), float(point_3d[1]), float(point_3d[2]))

        self.target_pub.publish(pt)
        self.get_logger().info(
            f"Segmented '{text_prompt}' → "
            f"3D ({pt.point.x:.3f}, {pt.point.y:.3f}, {pt.point.z:.3f}) m")

    def _run_grounding_dino(self, bgr: np.ndarray, text_prompt: str):
        rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor, _ = transform(pil_img, None)
        img_tensor = img_tensor.to(self.device)

        caption = text_prompt.strip().rstrip('.') + '.'

        with torch.no_grad():
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

        order  = torch.argsort(logits, descending=True)
        boxes  = boxes[order]

        cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = ((cx - bw / 2) * w).clamp(0, w).int()
        y1 = ((cy - bh / 2) * h).clamp(0, h).int()
        x2 = ((cx + bw / 2) * w).clamp(0, w).int()
        y2 = ((cy + bh / 2) * h).clamp(0, h).int()

        return torch.stack([x1, y1, x2, y2], dim=1).cpu().numpy()

    def _run_mobile_sam(self, bgr: np.ndarray, box_xyxy: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        self.sam_predictor.set_image(rgb)

        x1, y1, x2, y2 = box_xyxy.tolist()
        masks, _, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array([x1, y1, x2, y2], dtype=float),
            multimask_output=False,
        )
        return masks[0]

    def _mask_to_3d(self, mask: np.ndarray, depth: np.ndarray, info: CameraInfo):
        fx, fy = info.k[0], info.k[4]
        cx, cy = info.k[2], info.k[5]

        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None

        depths_m = depth[ys, xs] * self.depth_scale

        valid = (depths_m > 0.05) & (depths_m < 2.5)
        if valid.sum() == 0:
            return None

        xs_v = xs[valid].astype(np.float64)
        ys_v = ys[valid].astype(np.float64)
        d_v  = depths_m[valid].astype(np.float64)

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
