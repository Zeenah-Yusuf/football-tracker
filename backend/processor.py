import cv2
import numpy as np
import os
import tempfile
import supervision as sv
from collections import deque, defaultdict
from sklearn.cluster import KMeans
from typing import Dict, Tuple, List, Optional

from sports.common.team import TeamClassifier
from sports.configs.soccer import SoccerPitchConfiguration
from roboflow import Roboflow


# ============================================
# PITCH DIMENSIONS (Standard football pitch)
# ============================================

# Standard pitch in meters: 105m x 68m
PITCH_LENGTH = 105.0  # meters (touchline to touchline)
PITCH_WIDTH = 68.0    # meters (goal line to goal line)

# Define the 4 corners of the pitch in pitch coordinates
PITCH_CORNERS = np.array([
    [0, 0],           # Top-left
    [PITCH_LENGTH, 0], # Top-right
    [PITCH_LENGTH, PITCH_WIDTH], # Bottom-right
    [0, PITCH_WIDTH],  # Bottom-left
], dtype=np.float32)


# ============================================
# TRUE JERSEY COLOR EXTRACTOR
# ============================================

class TrueJerseyColorExtractor:
    """Extracts ACTUAL jersey colors from the video - no defaults"""
    
    def __init__(self):
        self.team_colors: Dict[int, Tuple[int, int, int]] = {}
        self.player_colors: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self.all_team_samples: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
        self.color_stability = 0.9
        
    def extract_jersey_color(self, frame, bbox):
        """Extract dominant jersey color from upper body"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(x2, w), min(y2, h)
            
            if x2 <= x1 + 15 or y2 <= y1 + 15:
                return None
            
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            
            crop_h = crop.shape[0]
            upper = crop[:max(1, int(crop_h * 0.55)), :]
            
            if upper.size == 0:
                return None
            
            upper_rgb = cv2.cvtColor(upper, cv2.COLOR_BGR2RGB)
            pixels = upper_rgb.reshape(-1, 3).astype(np.float32)
            
            brightness = np.mean(pixels, axis=1)
            r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
            
            mask = (brightness > 40) & (brightness < 230)
            grass_mask = (g > r * 1.2) & (g > b * 1.2)
            mask = mask & (~grass_mask)
            
            max_rgb = np.max(pixels, axis=1)
            min_rgb = np.min(pixels, axis=1)
            saturation = max_rgb - min_rgb
            mask = mask & (saturation > 15)
            
            filtered = pixels[mask]
            
            if len(filtered) < 30:
                return None
            
            n_clusters = min(2, max(1, len(filtered) // 30))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5)
            kmeans.fit(filtered)
            
            labels, counts = np.unique(kmeans.labels_, return_counts=True)
            dominant_idx = labels[np.argmax(counts)]
            dominant_rgb = kmeans.cluster_centers_[dominant_idx]
            
            r_val, g_val, b_val = map(int, dominant_rgb)
            return (b_val, g_val, r_val)
            
        except Exception:
            return None
    
    def update_colors_from_frame(self, frame, detections, tracker_ids, team_ids):
        """Update team colors with exponential moving average"""
        team_color_samples = defaultdict(list)
        
        for bbox, tracker_id, team_id in zip(detections.xyxy, tracker_ids, team_ids):
            color = self.extract_jersey_color(frame, bbox)
            if color is not None:
                b, g, r = color
                if not (g > r * 1.3 and g > b * 1.3):
                    if not (abs(r - g) < 15 and abs(g - b) < 15 and r < 60):
                        tid = int(team_id)
                        team_color_samples[tid].append(color)
                        self.player_colors[tracker_id].append(color)
                        if len(self.player_colors[tracker_id]) > 10:
                            self.player_colors[tracker_id].pop(0)
        
        for team_id, colors in team_color_samples.items():
            if len(colors) >= 2:
                median_color = np.median(colors, axis=0)
                median_tuple = tuple(int(x) for x in median_color)
                
                if team_id in self.team_colors:
                    prev = np.array(self.team_colors[team_id], dtype=np.float64)
                    new = np.array(median_tuple, dtype=np.float64)
                    smoothed = prev * self.color_stability + new * (1.0 - self.color_stability)
                    self.team_colors[team_id] = (int(smoothed[0]), int(smoothed[1]), int(smoothed[2]))
                else:
                    self.team_colors[team_id] = median_tuple
        
        return self.team_colors


# ============================================
# PITCH VISUALIZER
# ============================================

class PitchVisualizer:
    """Video game-style visualizer with true jersey colors"""
    
    def __init__(self):
        self.config = SoccerPitchConfiguration()
        self.scale = 0.1
        self.padding = 50
        self._cached_pitch = None
        
    def _get_pitch(self):
        if self._cached_pitch is None:
            self._cached_pitch = self._draw_full_pitch()
        return self._cached_pitch.copy()
    
    def _draw_full_pitch(self):
        """Draw the complete pitch using supervision"""
        from sports.annotators.soccer import draw_pitch
        return draw_pitch(config=self.config, padding=self.padding, scale=self.scale)
    
    def annotate_frame(self, frame, detections, team_ids, tracker_ids, 
                       team_colors, ball_detections=None):
        """Draw video game-style annotations with TRUE jersey colors"""
        annotated = frame.copy()
        
        if len(detections) > 0:
            for bbox, team_id, tracker_id in zip(detections.xyxy, team_ids, tracker_ids):
                x1, y1, x2, y2 = bbox.astype(int)
                
                raw = team_colors.get(int(team_id), (128, 128, 128))
                c_b, c_g, c_r = int(raw[0]), int(raw[1]), int(raw[2])
                
                cx, cy = int((x1 + x2) / 2), int(y2)
                ax, ay = int((x2 - x1) / 2), max(3, int((y2 - y1) / 4))
                
                # Shadow
                cv2.ellipse(annotated, (cx, cy + 2), (ax + 2, ay + 2), 0, 0, 360,
                          (0, 0, 0), -1, cv2.LINE_AA)
                # Main ellipse
                cv2.ellipse(annotated, (cx, cy), (ax, ay), 0, 0, 360,
                          (c_b, c_g, c_r), -1, cv2.LINE_AA)
                # Outline
                cv2.ellipse(annotated, (cx, cy), (ax, ay), 0, 0, 360,
                          (0, 0, 0), 2, cv2.LINE_AA)
                
                # ID badge
                text = f"#{tracker_id}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 2)
                tx, ty = cx - tw // 2, cy - 14
                cv2.rectangle(annotated, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3),
                            (0, 0, 0), -1)
                cv2.rectangle(annotated, (tx - 3, ty - th - 3), (tx + tw + 3, ty + 3),
                            (c_b, c_g, c_r), 1)
                cv2.putText(annotated, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                          0.4, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Ball
        if ball_detections is not None and len(ball_detections) > 0:
            for bbox in ball_detections.xyxy:
                x1, y1, x2, y2 = bbox.astype(int)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(annotated, (cx, cy), 10, (0, 180, 255), -1)
                cv2.drawMarker(annotated, (cx, cy), (0, 255, 255),
                             cv2.MARKER_STAR, 12, 2, cv2.LINE_AA)
        
        return annotated
    
    def create_birdseye(self, player_positions, team_ids, team_colors, ball_position=None):
        """Create bird's eye view with TRUE jersey colors - CONSTRAINED TO PITCH"""
        from sports.annotators.soccer import draw_points_on_pitch
        
        pitch = self._get_pitch()
        
        if len(player_positions) > 0:
            for pos, team_id in zip(player_positions, team_ids):
                # CLAMP positions to stay inside the pitch
                px = np.clip(pos[0], 0, PITCH_LENGTH)
                py = np.clip(pos[1], 0, PITCH_WIDTH)
                
                raw = team_colors.get(int(team_id), (128, 128, 128))
                c_r, c_g, c_b = int(raw[2]), int(raw[1]), int(raw[0])
                sv_color = sv.Color(r=c_r, g=c_g, b=c_b)
                pitch = draw_points_on_pitch(
                    self.config, np.array([[px, py]]), sv_color,
                    sv.Color.BLACK, 10, pitch
                )
        
        if ball_position is not None and len(ball_position) > 0:
            bx = np.clip(ball_position[0][0], 0, PITCH_LENGTH)
            by = np.clip(ball_position[0][1], 0, PITCH_WIDTH)
            pitch = draw_points_on_pitch(
                self.config, np.array([[bx, by]]), sv.Color.WHITE,
                sv.Color.BLACK, 6, pitch
            )
        
        return pitch
    
    def create_blank_pitch(self):
        return self._get_pitch()
    
    def get_pitch_size(self):
        pitch = self._get_pitch()
        return pitch.shape[1], pitch.shape[0]


# ============================================
# ROBUST HOMOGRAPHY HANDLER
# ============================================

class RobustHomography:
    """
    Handles perspective transformation with:
    - RANSAC outlier rejection
    - Temporal smoothing
    - Confidence-based point filtering
    - Pitch boundary clamping
    - Fallback to last valid homography
    """
    
    def __init__(self, pitch_vertices):
        self.pitch_vertices = np.array(pitch_vertices, dtype=np.float32)
        self.history = deque(maxlen=5)  # Smooth over 5 frames
        self.last_valid_H = None
        self.min_inliers = 6
        self.confidence_threshold = 0.4
        
    def compute(self, camera_points, confidences):
        """
        Compute homography matrix from camera points to pitch points.
        
        Args:
            camera_points: Nx2 array of detected keypoints in camera view
            confidences: N array of confidence scores (0-1)
        
        Returns:
            3x3 homography matrix or None
        """
        if len(camera_points) < 4:
            return self.last_valid_H
        
        # Filter by confidence
        high_conf_mask = confidences > self.confidence_threshold
        cam_filtered = camera_points[high_conf_mask]
        
        if len(cam_filtered) < 4:
            return self.last_valid_H
        
        # Get corresponding pitch points (first N vertices)
        pitch_filtered = self.pitch_vertices[:len(cam_filtered)]
        
        # Make sure we have matching point counts
        n = min(len(cam_filtered), len(pitch_filtered))
        cam_filtered = cam_filtered[:n]
        pitch_filtered = pitch_filtered[:n]
        
        if n < 4:
            return self.last_valid_H
        
        try:
            # Compute homography with RANSAC
            H, mask = cv2.findHomography(
                srcPoints=cam_filtered,
                dstPoints=pitch_filtered,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0
            )
            
            if H is None:
                return self.last_valid_H
            
            # Check number of inliers
            inliers = mask.sum() if mask is not None else 0
            if inliers < self.min_inliers:
                return self.last_valid_H
            
            # Normalize
            H = H / H[2, 2]
            
            # Add to history for smoothing
            self.history.append(H)
            
            # Average last N homographies for temporal stability
            H_smooth = np.mean(np.array(self.history), axis=0)
            H_smooth = H_smooth / H_smooth[2, 2]
            
            # Validate homography (check it maps corners reasonably)
            if self._validate_homography(H_smooth):
                self.last_valid_H = H_smooth.copy()
                return H_smooth
            else:
                return self.last_valid_H
                
        except Exception:
            return self.last_valid_H
    
    def _validate_homography(self, H):
        """
        Validate that the homography produces reasonable results.
        Checks that pitch corners map to finite, reasonable positions.
        """
        # Test with 4 pitch corners
        test_points = np.array([
            [0, 0],
            [PITCH_LENGTH, 0],
            [PITCH_LENGTH, PITCH_WIDTH],
            [0, PITCH_WIDTH],
        ], dtype=np.float32)
        
        transformed = self.transform_points(H, test_points)
        
        # Check that all points are finite
        if not np.all(np.isfinite(transformed)):
            return False
        
        # Check that points are within reasonable range
        if np.any(transformed < -PITCH_LENGTH) or np.any(transformed > PITCH_LENGTH * 2):
            return False
        
        return True
    
    def transform_points(self, H, points):
        """
        Apply homography to transform points from camera to pitch coordinates.
        Clamps output to stay within pitch boundaries.
        """
        if H is None or len(points) == 0:
            return np.array([])
        
        # Convert to homogeneous coordinates
        ones = np.ones((len(points), 1), dtype=np.float32)
        homogeneous = np.hstack([points, ones])
        
        # Apply transformation
        transformed = H @ homogeneous.T
        transformed = transformed / (transformed[2, :] + 1e-10)  # Avoid division by zero
        
        # Extract x, y
        result = transformed[:2, :].T
        
        # CLAMP to pitch boundaries
        result[:, 0] = np.clip(result[:, 0], 0, PITCH_LENGTH)
        result[:, 1] = np.clip(result[:, 1], 0, PITCH_WIDTH)
        
        return result
    
    def reset(self):
        self.history.clear()
        self.last_valid_H = None


# ============================================
# FAST DEMO PROCESSOR
# ============================================

class FastDemoProcessor:
    """Processor with robust homography and true jersey colors"""
    
    BALL_ID = 0
    PLAYER_ID = 2
    
    def __init__(self, api_key: str):
        self.config = SoccerPitchConfiguration()
        self.colors = TrueJerseyColorExtractor()
        self.viz = PitchVisualizer()
        self.tracker = sv.ByteTrack()
        self.team_classifier = None
        
        # Robust homography with pitch vertices
        self.homography = RobustHomography(self.config.vertices[:27])
        
        self.rf = Roboflow(api_key=api_key)
        self.p_model = self.rf.workspace().project(
            "football-players-detection-3zvbc").version(11).model
        self.f_model = self.rf.workspace().project(
            "football-field-detection-f07vi").version(14).model
        print("Models loaded!")
    
    def _predict(self, model, frame, confidence=20):
        """Fast prediction with downscaling"""
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        
        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(tmp.name, frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        try:
            result = model.predict(tmp.name, confidence=confidence).json()
            os.unlink(tmp.name)
            return result
        except:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            return None
    
    def _to_dets(self, preds, original_shape):
        """Convert predictions with scaling back"""
        if not preds or 'predictions' not in preds or not preds['predictions']:
            return sv.Detections.empty()
        
        h, w = original_shape[:2]
        scale_x = w / 640 if w > 640 else 1.0
        scale_y = h / (640 * h / w) if w > 640 else 1.0
        
        xyxy, cid, conf = [], [], []
        for p in preds['predictions']:
            x = p['x'] * scale_x
            y = p['y'] * scale_y
            bw = p['width'] * scale_x
            bh = p['height'] * scale_y
            xyxy.append([x - bw/2, y - bh/2, x + bw/2, y + bh/2])
            cid.append(p['class_id'])
            conf.append(p['confidence'])
        
        return sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            class_id=np.array(cid, dtype=np.int32),
            confidence=np.array(conf, dtype=np.float32)
        )
    
    def compute_homography(self, frame):
        """Compute homography using robust handler"""
        preds = self._predict(self.f_model, frame, confidence=15)
        if not preds or len(preds.get('predictions', [])) < 4:
            return self.homography.last_valid_H
        
        cam_pts = np.array([[p['x'], p['y']] for p in preds['predictions']], dtype=np.float32)
        confs = np.array([p['confidence'] for p in preds['predictions']], dtype=np.float32)
        
        return self.homography.compute(cam_pts, confs)
    
    def reset(self):
        self.tracker = sv.ByteTrack()
        self.colors = TrueJerseyColorExtractor()
        self.homography.reset()
    
    def train_teams(self, video_path, num_frames=8):
        """Fast team training"""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, total // num_frames)
        
        crops = []
        for i in range(0, total, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            preds = self._predict(self.p_model, frame)
            dets = self._to_dets(preds, frame.shape)
            if len(dets) == 0:
                continue
            for bbox in dets.xyxy[dets.class_id == self.PLAYER_ID]:
                crop = sv.crop_image(frame, bbox)
                if crop is not None and crop.size > 0:
                    crops.append(crop)
            if len(crops) >= 80:
                break
        cap.release()
        
        if len(crops) >= 6:
            self.team_classifier = TeamClassifier(device="cpu")
            self.team_classifier.fit(crops[:150])
            print(f"Trained with {len(crops)} samples")
            return True
        return False
    
    def process_frame(self, frame, frame_count):
        """Process single frame"""
        if self.tracker is None:
            self.reset()
        
        preds = self._predict(self.p_model, frame)
        dets = self._to_dets(preds, frame.shape)
        
        if len(dets) == 0:
            return frame, None
        
        ball = dets[dets.class_id == self.BALL_ID]
        if len(ball) > 0:
            ball.xyxy = sv.pad_boxes(ball.xyxy, px=8)
        
        players = dets[dets.class_id != self.BALL_ID]
        if len(players) == 0:
            return frame, None
        
        players = players.with_nms(threshold=0.5, class_agnostic=True)
        players = self.tracker.update_with_detections(detections=players)
        if len(players) == 0:
            return frame, None
        
        tids = players.tracker_id if hasattr(players, 'tracker_id') else np.arange(len(players))
        
        if self.team_classifier is not None:
            try:
                pmask = players.class_id == self.PLAYER_ID
                if pmask.sum() > 0:
                    crops = [sv.crop_image(frame, b) for b in players.xyxy[pmask]]
                    if crops:
                        players.class_id[pmask] = self.team_classifier.predict(crops)
            except:
                pass
        
        # True jersey colors
        team_colors = self.colors.update_colors_from_frame(frame, players, tids, players.class_id)
        
        # Annotate
        annotated = self.viz.annotate_frame(frame, players, players.class_id, tids, team_colors, ball)
        
        # ROBUST bird's eye with homography
        H = self.compute_homography(frame)
        birdseye = None
        if H is not None:
            try:
                pp = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                # Transform points (automatically clamped to pitch)
                tp = self.homography.transform_points(H, pp)
                
                pb = None
                if len(ball) > 0:
                    bp = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                    pb = self.homography.transform_points(H, bp)
                
                birdseye = self.viz.create_birdseye(tp, players.class_id, team_colors, pb)
            except:
                pass
        
        return annotated, birdseye
    
    def process_video(self, video_path, max_seconds=15, progress_callback=None):
        """Process video"""
        self.reset()
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        target_frames = min(total_frames, int(max_seconds * fps))
        start_frame = max(0, (total_frames - target_frames) // 2)
        
        self.train_teams(video_path, num_frames=8)
        
        pitch_w, pitch_h = self.viz.get_pitch_size()
        
        out_a = tempfile.NamedTemporaryFile(suffix='_tracked.mp4', delete=False)
        out_b = tempfile.NamedTemporaryFile(suffix='_birdseye.mp4', delete=False)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        wa = cv2.VideoWriter(out_a.name, fourcc, fps, (width, height))
        wb = cv2.VideoWriter(out_b.name, fourcc, fps, (pitch_w, pitch_h))
        
        blank = self.viz.create_blank_pitch()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        fc = 0
        while fc < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if fc % 2 != 0:
                fc += 1
                continue
            
            annotated, birdseye = self.process_frame(frame, fc)
            
            if annotated.shape[0] != height or annotated.shape[1] != width:
                annotated = cv2.resize(annotated, (width, height))
            wa.write(annotated)
            
            if birdseye is not None:
                if birdseye.shape[0] != pitch_h or birdseye.shape[1] != pitch_w:
                    birdseye = cv2.resize(birdseye, (pitch_w, pitch_h))
                wb.write(birdseye)
            else:
                wb.write(blank)
            
            fc += 1
            
            if progress_callback and fc % 3 == 0:
                progress_callback(fc / target_frames)
        
        cap.release()
        wa.release()
        wb.release()
        
        return out_a.name, out_b.name, fc
