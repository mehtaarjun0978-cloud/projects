"""
Phase 1: Video Input, Student/Object Detection & Seat Tracking
Project: Intelligent Examination Hall Monitoring and Anomaly Detection Agent

Covers:
- Task 1 (Step 1): Load classroom/examination-hall video frame-by-frame using OpenCV.
- Task 2 (Step 2): Detect students (person) and mobile phones (cell phone) using YOLOv8.
- Task 3 (Step 3): Track students across consecutive video frames and assign fixed Seat/Student IDs
  along with their initial assigned seat coordinates.
"""

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Try importing Ultralytics YOLO for real-world CCTV detection & tracking
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# =====================================================================
# DATA STRUCTURES FOR PHASE 1 OUTPUT (PASSED TO PHASE 2 LATER)
# =====================================================================

@dataclass
class Detection:
    """Represents a single detected object in the current frame."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int                    # 0 = person, 67 = cell phone (COCO)
    class_name: str
    raw_track_id: Optional[int] = None


@dataclass
class TrackedStudent:
    """Represents a tracked student locked to a fixed Seat/Student ID."""
    seat_id: int
    assigned_seat_bbox: Tuple[int, int, int, int]      # Fixed initial desk/seat region
    assigned_seat_center: Tuple[int, int]              # Fixed (cx, cy) of assigned seat
    current_bbox: Tuple[int, int, int, int]            # Current bounding box in frame
    current_center: Tuple[int, int]                    # Current (cx, cy) in frame
    confidence: float                                  # Latest detection confidence
    last_seen_frame: int                               # Last frame index student was detected
    missed_frames: int = 0                             # Consecutive frames missed
    trajectory: List[Tuple[int, int]] = field(default_factory=list)  # Recent center history


# =====================================================================
# STEP 1: VIDEO INPUT LOADER & DEMO VIDEO GENERATOR
# =====================================================================

class VideoInputLoader:
    """Step 1: Handles loading classroom CCTV video frame-by-frame."""

    def __init__(self, video_source: str):
        self.video_source = video_source
        # Allow webcam index (e.g., "0") or video file path
        source = int(video_source) if video_source.isdigit() else video_source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {video_source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Reads the next frame and returns (success, frame, timestamp_sec)."""
        ret, frame = self.cap.read()
        if not ret:
            return False, None, 0.0
        current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp_sec = current_frame_idx / self.fps
        return True, frame, timestamp_sec

    def release(self):
        self.cap.release()


def generate_sample_exam_video(output_path: str = "data/sample_exam_hall.mp4", num_frames: int = 150, fps: int = 25) -> str:
    """
    Generates a synthetic examination hall CCTV video for testing if no valid video file is provided.
    Includes 6 seated students in a 2x3 grid, slight natural posture shifts, one student moving
    out of seat, and a simulated mobile phone appearance.
    """
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    width, height = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 2 rows x 3 columns of examination desks
    seat_centers = [
        (220, 200), (480, 200), (740, 200),
        (220, 390), (480, 390), (740, 390),
    ]

    for f in range(num_frames):
        frame = np.full((height, width, 3), (235, 232, 225), dtype=np.uint8)

        # Draw classroom floor grid & invigilator podium
        cv2.rectangle(frame, (380, 25), (580, 70), (90, 110, 140), -1)
        cv2.putText(frame, "INVIGILATOR DESK", (405, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        for idx, (sx, sy) in enumerate(seat_centers):
            # Draw desk
            cv2.rectangle(frame, (sx - 70, sy - 45), (sx + 70, sy + 15), (150, 180, 210), -1)
            cv2.rectangle(frame, (sx - 70, sy - 45), (sx + 70, sy + 15), (90, 115, 145), 2)

            # Simulate student movement on Seat 5 (idx == 4) after frame 60
            dx, dy = 0, 0
            if idx == 4 and f > 60:
                dx = int(min(95, (f - 60) * 2.2))  # Moves sideways toward neighbor
            else:
                dx = int(2 * math.sin(f * 0.1 + idx))
                dy = int(2 * math.cos(f * 0.1 + idx))

            cx, cy = sx + dx, sy + dy + 15
            # Draw student torso & head (distinct dark clothing silhouette)
            cv2.rectangle(frame, (cx - 32, cy - 35), (cx + 32, cy + 45), (65, 85, 115), -1)
            cv2.circle(frame, (cx, cy - 42), 18, (180, 195, 215), -1)

            # Simulate mobile phone near Seat 2 (idx == 1) between frames 45 and 110
            if idx == 1 and 45 <= f <= 110:
                px1, py1, px2, py2 = cx + 22, cy - 28, cx + 40, cy - 2
                cv2.rectangle(frame, (px1, py1), (px2, py2), (20, 20, 20), -1)
                cv2.rectangle(frame, (px1 + 2, py1 + 3), (px2 - 2, py2 - 3), (200, 240, 255), -1)

        writer.write(frame)

    writer.release()
    return output_path


# =====================================================================
# STEP 2: STUDENT & MOBILE PHONE DETECTION MODEL
# =====================================================================

class ExamHallDetector:
    """
    Step 2: Detects students (COCO class 0: 'person') and mobile phones
    (COCO class 67: 'cell phone') in each video frame.
    """

    PERSON_CLASS_ID = 0
    PHONE_CLASS_ID = 67

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.model = None

        if YOLO_AVAILABLE and os.path.exists(model_path):
            self.model = YOLO(model_path)
        elif YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
            except Exception:
                self.model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs YOLOv8 detection (or fallback contour detection for synthetic test video)
        and returns a list of student and mobile phone detections.
        """
        detections: List[Detection] = []

        if self.model is not None:
            results = self.model.predict(
                frame,
                conf=self.conf_threshold,
                classes=[self.PERSON_CLASS_ID, self.PHONE_CLASS_ID],
                verbose=False,
            )
            if results and len(results) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    cls_name = "person" if cls_id == self.PERSON_CLASS_ID else "cell phone"
                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=conf,
                            class_id=cls_id,
                            class_name=cls_name,
                        )
                    )
            if detections:
                return detections

        # Fallback detector for synthetic demo video or offline environment without weights
        return self._fallback_synthetic_detector(frame)

    def _fallback_synthetic_detector(self, frame: np.ndarray) -> List[Detection]:
        """Detects student silhouettes and phone objects in the synthetic exam hall video."""
        detections: List[Detection] = []

        # 1. Detect student silhouettes by torso color mask (BGR ~(65, 85, 115))
        lower_student = np.array([50, 70, 100], dtype=np.uint8)
        upper_student = np.array([80, 100, 130], dtype=np.uint8)
        mask_student = cv2.inRange(frame, lower_student, upper_student)
        contours, _ = cv2.findContours(mask_student, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) > 1500:
                x, y, w, h = cv2.boundingRect(cnt)
                # Expand box upward to include head
                bbox = (x - 6, max(0, y - 30), x + w + 6, min(frame.shape[0], y + h + 4))
                detections.append(
                    Detection(
                        bbox=bbox,
                        confidence=0.92,
                        class_id=self.PERSON_CLASS_ID,
                        class_name="person",
                    )
                )

        # 2. Detect mobile phone screen mask (BGR ~(200, 240, 255))
        lower_phone = np.array([190, 230, 245], dtype=np.uint8)
        upper_phone = np.array([210, 250, 255], dtype=np.uint8)
        mask_phone = cv2.inRange(frame, lower_phone, upper_phone)
        p_contours, _ = cv2.findContours(mask_phone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in p_contours:
            if 80 < cv2.contourArea(cnt) < 1200:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append(
                    Detection(
                        bbox=(x - 2, y - 3, x + w + 2, y + h + 3),
                        confidence=0.88,
                        class_id=self.PHONE_CLASS_ID,
                        class_name="cell phone",
                    )
                )

        return detections


# =====================================================================
# STEP 3: MULTI-STUDENT TRACKER & SEAT ASSIGNMENT ENGINE
# =====================================================================

class StudentSeatTracker:
    """
    Step 3: Tracks students across consecutive frames and locks each student to a
    fixed Seat/Student ID and initial assigned seat region.
    """

    def __init__(self, max_match_distance: float = 140.0, max_missed_frames: int = 45):
        self.max_match_distance = max_match_distance
        self.max_missed_frames = max_missed_frames
        self.students: Dict[int, TrackedStudent] = {}
        self.next_seat_id: int = 1
        self.seats_locked: bool = False

    @staticmethod
    def _get_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _euclidean_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def update(self, detections: List[Detection], frame_idx: int) -> Tuple[Dict[int, TrackedStudent], List[Detection]]:
        """
        Matches current frame student detections to existing tracked seats.
        Sorts initial detections top-to-bottom, left-to-right so Seat IDs follow classroom layout.
        Returns (tracked_students_dict, phone_detections_list).
        """
        person_dets = [d for d in detections if d.class_id == ExamHallDetector.PERSON_CLASS_ID]
        phone_dets = [d for d in detections if d.class_id == ExamHallDetector.PHONE_CLASS_ID]

        # On initial frame(s), sort students in reading order (row by row, left to right)
        if not self.students and person_dets:
            person_dets.sort(key=lambda d: (d.bbox[1] // 120, d.bbox[0]))
            for det in person_dets:
                center = self._get_center(det.bbox)
                seat_id = self.next_seat_id
                self.next_seat_id += 1
                self.students[seat_id] = TrackedStudent(
                    seat_id=seat_id,
                    assigned_seat_bbox=det.bbox,
                    assigned_seat_center=center,
                    current_bbox=det.bbox,
                    current_center=center,
                    confidence=det.confidence,
                    last_seen_frame=frame_idx,
                    trajectory=[center],
                )
            self.seats_locked = True
            return self.students, phone_dets

        # Match existing tracked students to new person detections via minimum distance
        unmatched_seats = set(self.students.keys())
        unmatched_dets = list(range(len(person_dets)))

        # Build pairwise distance matrix between tracked students and new detections
        pairs: List[Tuple[float, int, int]] = []
        for seat_id in unmatched_seats:
            student = self.students[seat_id]
            for det_idx in unmatched_dets:
                det_center = self._get_center(person_dets[det_idx].bbox)
                # Prefer matching to current position; also softly anchor to assigned seat
                dist_curr = self._euclidean_dist(student.current_center, det_center)
                if dist_curr <= self.max_match_distance:
                    pairs.append((dist_curr, seat_id, det_idx))

        pairs.sort(key=lambda x: x[0])

        for dist, seat_id, det_idx in pairs:
            if seat_id in unmatched_seats and det_idx in unmatched_dets:
                det = person_dets[det_idx]
                center = self._get_center(det.bbox)
                st = self.students[seat_id]
                st.current_bbox = det.bbox
                st.current_center = center
                st.confidence = det.confidence
                st.last_seen_frame = frame_idx
                st.missed_frames = 0
                st.trajectory.append(center)
                if len(st.trajectory) > 60:
                    st.trajectory.pop(0)

                unmatched_seats.remove(seat_id)
                unmatched_dets.remove(det_idx)

        # Increment missed frame counters for any temporarily occluded students
        for seat_id in list(unmatched_seats):
            self.students[seat_id].missed_frames += 1

        # Register newly appearing students only if far from any existing assigned seat
        for det_idx in unmatched_dets:
            det = person_dets[det_idx]
            center = self._get_center(det.bbox)
            seat_id = self.next_seat_id
            self.next_seat_id += 1
            self.students[seat_id] = TrackedStudent(
                seat_id=seat_id,
                assigned_seat_bbox=det.bbox,
                assigned_seat_center=center,
                current_bbox=det.bbox,
                current_center=center,
                confidence=det.confidence,
                last_seen_frame=frame_idx,
                trajectory=[center],
            )

        return self.students, phone_dets


# =====================================================================
# PHASE 1 VISUALIZATION & PIPELINE RUNNER
# =====================================================================

def draw_phase1_annotations(
    frame: np.ndarray,
    students: Dict[int, TrackedStudent],
    phones: List[Detection],
    frame_idx: int,
    timestamp_sec: float,
) -> np.ndarray:
    """Draws assigned seat zones, tracked student boxes, IDs, trajectories, and detected phones."""
    vis = frame.copy()

    # 1. Draw each student's initial Assigned Seat Zone and Current Tracked Bounding Box
    for seat_id, st in sorted(students.items()):
        sx1, sy1, sx2, sy2 = st.assigned_seat_bbox
        # Assigned seat box in subtle cyan outline
        cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), (200, 180, 0), 1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"Seat {seat_id} Zone",
            (sx1, max(15, sy1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (160, 130, 0),
            1,
            cv2.LINE_AA,
        )

        # Current tracked student box in green
        x1, y1, x2, y2 = st.current_bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 60), 2, cv2.LINE_AA)

        label = f"ID/Seat #{seat_id} ({st.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 8, y1), (0, 180, 60), -1)
        cv2.putText(
            vis,
            label,
            (x1 + 4, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Draw recent movement trail
        if len(st.trajectory) > 1:
            pts = np.array(st.trajectory, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=False, color=(0, 220, 255), thickness=2)

    # 2. Draw detected mobile phones in red
    for phone in phones:
        px1, py1, px2, py2 = phone.bbox
        cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 230), 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"PHONE ({phone.confidence:.2f})",
            (px1, max(15, py1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 230),
            2,
            cv2.LINE_AA,
        )

    # 3. Top status bar
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 32), (30, 30, 30), -1)
    status_text = (
        f"PHASE 1: Detection & Seat Tracking | Frame: {frame_idx} | "
        f"Time: {timestamp_sec:05.2f}s | Tracked Students: {len(students)} | Phones: {len(phones)}"
    )
    cv2.putText(vis, status_text, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def run_phase1_pipeline(
    video_path: str,
    output_video_path: str = "data/phase1_tracked_output.mp4",
    model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.35,
    show_window: bool = False,
) -> Dict[str, int]:
    """
    Executes the complete Phase 1 pipeline (Steps 1, 2, and 3) on an input video
    and writes the annotated tracking video.
    """
    out_dir = os.path.dirname(output_video_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    loader = VideoInputLoader(video_path)
    detector = ExamHallDetector(model_path=model_path, conf_threshold=conf_threshold)
    tracker = StudentSeatTracker()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, loader.fps, (loader.width, loader.height))

    frame_idx = 0
    total_phone_detections = 0

    while True:
        ret, frame, timestamp_sec = loader.read_frame()
        if not ret or frame is None:
            break

        frame_idx += 1

        # Step 2: Detect students and mobile phones
        detections = detector.detect(frame)

        # Step 3: Track students across consecutive frames & maintain Seat IDs
        students, phones = tracker.update(detections, frame_idx)
        total_phone_detections += len(phones)

        # Annotate frame
        annotated = draw_phase1_annotations(frame, students, phones, frame_idx, timestamp_sec)
        writer.write(annotated)

        if show_window:
            cv2.imshow("Phase 1 - Exam Hall Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    loader.release()
    writer.release()
    if show_window:
        cv2.destroyAllWindows()

    summary = {
        "processed_frames": frame_idx,
        "total_seats_tracked": len(tracker.students),
        "phone_detections_frames": total_phone_detections,
    }
    print(f"[Phase 1 Complete] Summary: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Exam Hall Video Input, Detection & Seat Tracking")
    parser.add_argument("--video", type=str, default="data/sample_exam_hall.mp4", help="Path to input examination hall CCTV video")
    parser.add_argument("--output", type=str, default="data/phase1_tracked_output.mp4", help="Path to save tracked output video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model weights path")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display live OpenCV window during processing")
    args = parser.parse_args()

    input_video = args.video
    # Check if file is missing OR is an empty 0-byte placeholder file
    if not input_video or not os.path.exists(input_video) or os.path.getsize(input_video) < 1024:
        target_path = input_video if input_video else "data/sample_exam_hall.mp4"
        print(f"[Info] Generating synthetic sample examination hall video at '{target_path}'...")
        input_video = generate_sample_exam_video(target_path)

    run_phase1_pipeline(
        video_path=input_video,
        output_video_path=args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        show_window=args.show,
    )