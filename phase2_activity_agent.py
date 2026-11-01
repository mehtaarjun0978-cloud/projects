"""
Phase 2: Suspicious Activity Detection & Model-Based Agent Memory (Hybrid Upgrade)
Project: Intelligent Examination Hall Monitoring and Anomaly Detection Agent

Covers:
- Task 4 (Step 4): Detect the 5 examination-related events with Camera-Angle Relative Scaling:
    1. Student leaving the assigned seat (LEAVING_SEAT)
    2. Repeated turning toward another student (REPEATED_TURNING)
    3. Mobile phone detection (MOBILE_PHONE)
    4. Multiple students gathering near one location (STUDENTS_GATHERING)
    5. Unusual movement during examination (UNUSUAL_MOVEMENT)
- Task 5 (Step 5): Represent each student using an internal state:
    NORMAL -> OBSERVE -> SUSPICIOUS -> ALERT (plus INVIGILATOR role filter for teachers)
- Task 6 (Step 6): Implement a Model-Based Agent that remembers previous observations
    across frames so a single noisy frame or walking teacher does not trigger false alerts.
"""

import argparse
import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phase1_video_detection_tracking import (
    Detection,
    ExamHallDetector,
    StudentSeatTracker,
    TrackedStudent,
    VideoInputLoader,
)


# =====================================================================
# TASK 5: STUDENT INTERNAL STATES & EVENT TYPES
# =====================================================================

class StudentState(str, Enum):
    """Internal state progression for each student (plus Invigilator role)."""
    NORMAL = "NORMAL"
    OBSERVE = "OBSERVE"
    SUSPICIOUS = "SUSPICIOUS"
    ALERT = "ALERT"
    INVIGILATOR = "INVIGILATOR"


class EventType(str, Enum):
    """The 5 required examination anomaly event types."""
    LEAVING_SEAT = "Leaving Assigned Seat"
    REPEATED_TURNING = "Repeated Turning Toward Student"
    MOBILE_PHONE = "Mobile Phone Detected"
    STUDENTS_GATHERING = "Multiple Students Gathering"
    UNUSUAL_MOVEMENT = "Unusual Movement"


# Color map (BGR) for each state
STATE_COLORS: Dict[StudentState, Tuple[int, int, int]] = {
    StudentState.NORMAL: (0, 180, 60),        # Green
    StudentState.OBSERVE: (0, 215, 255),      # Yellow
    StudentState.SUSPICIOUS: (0, 140, 255),   # Orange
    StudentState.ALERT: (0, 0, 235),          # Red
    StudentState.INVIGILATOR: (210, 120, 0),  # Blue (Teacher / Invigilator)
}


@dataclass
class FrameEventObservation:
    """Single-frame observation recorded for a student."""
    frame_idx: int
    timestamp_sec: float
    event_type: EventType
    confidence: float
    detail: str = ""


@dataclass
class StudentAgentMemory:
    """
    Task 6: Internal memory of the Model-Based Agent for a single tracked person.
    Remembers past frame observations so single-frame noise is filtered out,
    and tracks whether this person is a standing/roaming Invigilator.
    """
    seat_id: int
    state: StudentState = StudentState.NORMAL
    is_invigilator: bool = False
    recent_abnormal_window: Deque[bool] = field(default_factory=lambda: deque(maxlen=50))
    event_durations_sec: Dict[EventType, float] = field(default_factory=dict)
    event_counts: Dict[EventType, int] = field(default_factory=dict)
    head_turn_frames: int = 0
    head_turn_count: int = 0
    last_turn_direction: int = 0
    last_turn_frame: int = -100
    active_events: List[FrameEventObservation] = field(default_factory=list)
    observation_history: List[FrameEventObservation] = field(default_factory=list)
    normal_streak_frames: int = 0
    standing_votes: int = 0


# =====================================================================
# TASK 4 & TASK 6: HYBRID MODEL-BASED ANOMALY DETECTION AGENT
# =====================================================================

class ExamMonitoringAgent:
    """
    Combines Task 4 (5-Event Detection), Task 5 (State Representation),
    and Task 6 (Model-Based Agent with Temporal Memory), upgraded with:
    1. Camera-Angle Relative Thresholds (scaled to each student's bounding box size)
    2. Invigilator / Standing Teacher Filter (ignores walking teachers for student alerts)
    """

    def __init__(
        self,
        fps: float = 25.0,
        seat_leave_dist_px: float = 55.0,
        gathering_dist_px: float = 135.0,
        turning_offset_px: float = 11.0,
        unusual_speed_px_per_frame: float = 9.0,
        phone_assoc_dist_px: float = 110.0,
        enable_invigilator_filter: bool = True,
    ):
        self.fps = max(1.0, fps)
        self.dt = 1.0 / self.fps
        self.seat_leave_dist_px = seat_leave_dist_px
        self.gathering_dist_px = gathering_dist_px
        self.turning_offset_px = turning_offset_px
        self.unusual_speed_px_per_frame = unusual_speed_px_per_frame
        self.phone_assoc_dist_px = phone_assoc_dist_px
        self.enable_invigilator_filter = enable_invigilator_filter

        self.memories: Dict[int, StudentAgentMemory] = {}

    @staticmethod
    def _dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def _box_width(bbox: Tuple[int, int, int, int]) -> float:
        return float(max(40, bbox[2] - bbox[0]))

    @staticmethod
    def _box_height(bbox: Tuple[int, int, int, int]) -> float:
        return float(max(40, bbox[3] - bbox[1]))

    def _detect_invigilators(self, frame: np.ndarray, students: Dict[int, TrackedStudent]) -> None:
        """
        Identifies standing/roaming Invigilators (teachers) so they are not flagged as cheating students:
        1. In synthetic videos, anyone patrolling the top invigilator zone (y_center < 115) is an Invigilator.
        2. In real videos, a standing person whose bounding-box height and aspect ratio (h/w) are
           significantly taller than the median seated student (or who walks upright across the room)
           accumulates standing votes and is locked as INVIGILATOR.
        """
        if not self.enable_invigilator_filter or not students:
            return

        heights = [self._box_height(st.current_bbox) for st in students.values()]
        median_h = float(np.median(heights)) if heights else 100.0

        for seat_id, st in students.items():
            mem = self.memories[seat_id]
            if mem.is_invigilator:
                mem.state = StudentState.INVIGILATOR
                continue

            w = self._box_width(st.current_bbox)
            h = self._box_height(st.current_bbox)
            aspect = h / max(1.0, w)

            # Check 1: Top podium/aisle patrol zone in synthetic hall (y_center < 115)
            if st.assigned_seat_center[1] < 115:
                mem.is_invigilator = True
                mem.state = StudentState.INVIGILATOR
                continue

            # Check 2: Standing teacher posture in real classroom CCTV (tall box compared to seated peers)
            if len(students) >= 3 and h > 1.28 * median_h and aspect > 1.42:
                mem.standing_votes += 2
            elif aspect > 1.85 and h > 160:
                mem.standing_votes += 1

            if mem.standing_votes >= 6:
                mem.is_invigilator = True
                mem.state = StudentState.INVIGILATOR
                mem.active_events.clear()

    def _get_head_turn_offset(self, frame: np.ndarray, student: TrackedStudent) -> float:
        """
        Estimates lateral head/upper-body turn offset relative to torso/seat center.
        """
        x1, y1, x2, y2 = student.current_bbox
        h = max(1, y2 - y1)
        head_y2 = y1 + int(h * 0.38)
        roi = frame[max(0, y1):min(frame.shape[0], head_y2), max(0, x1):min(frame.shape[1], x2)]

        head_offset = 0.0
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
            moments = cv2.moments(thresh)
            if moments["m00"] > 80:
                head_cx = moments["m10"] / moments["m00"]
                box_mid_x = (x2 - x1) / 2.0
                head_offset = head_cx - box_mid_x

        lean_offset = float(student.current_center[0] - student.assigned_seat_center[0])
        return head_offset + 0.75 * lean_offset

    def update(
        self,
        frame: np.ndarray,
        students: Dict[int, TrackedStudent],
        phones: List[Detection],
        frame_idx: int,
        timestamp_sec: float,
    ) -> Dict[int, StudentAgentMemory]:
        """
        Evaluates all 5 events for each tracked student using camera-angle adaptive thresholds,
        filters out standing Invigilators, and updates each student's state.
        """
        for seat_id in students:
            if seat_id not in self.memories:
                self.memories[seat_id] = StudentAgentMemory(seat_id=seat_id)

        # Step A: Identify and filter out standing/walking Invigilator(s)
        self._detect_invigilators(frame, students)

        # Step B: Pre-check Event 4 (Gathering) ONLY among non-invigilator students,
        # requiring at least one student to have displaced from their assigned seat
        gathering_map: Dict[int, List[int]] = {sid: [] for sid in students}
        student_ids = [sid for sid in sorted(students.keys()) if not self.memories[sid].is_invigilator]

        for i in range(len(student_ids)):
            for j in range(i + 1, len(student_ids)):
                s1, s2 = students[student_ids[i]], students[student_ids[j]]
                w1 = self._box_width(s1.assigned_seat_bbox)
                w2 = self._box_width(s2.assigned_seat_bbox)
                avg_w = 0.5 * (w1 + w2)

                # Adaptive gathering proximity + check that at least one student moved from their seat
                adaptive_gather_dist = max(self.gathering_dist_px, 1.55 * avg_w)
                dist = self._dist(s1.current_center, s2.current_center)
                disp1 = self._dist(s1.current_center, s1.assigned_seat_center)
                disp2 = self._dist(s2.current_center, s2.assigned_seat_center)

                if dist <= adaptive_gather_dist and (disp1 > 0.45 * w1 or disp2 > 0.45 * w2):
                    gathering_map[s1.seat_id].append(s2.seat_id)
                    gathering_map[s2.seat_id].append(s1.seat_id)

        # Step C: Associate detected mobile phones with nearest non-invigilator student
        phone_by_seat: Dict[int, Detection] = {}
        for phone in phones:
            px = (phone.bbox[0] + phone.bbox[2]) // 2
            py = (phone.bbox[1] + phone.bbox[3]) // 2
            best_seat, best_dist = None, float("inf")
            for sid in student_ids:
                st = students[sid]
                w = self._box_width(st.current_bbox)
                d = self._dist((px, py), st.current_center)
                if d < best_dist and d <= max(self.phone_assoc_dist_px, 1.3 * w):
                    best_dist = d
                    best_seat = sid
            if best_seat is not None:
                phone_by_seat[best_seat] = phone

        # Step D: Evaluate all 5 events per student using perspective-adaptive thresholds
        for seat_id, st in students.items():
            mem = self.memories[seat_id]
            if mem.is_invigilator:
                mem.state = StudentState.INVIGILATOR
                mem.active_events = []
                continue

            current_frame_events: List[FrameEventObservation] = []
            box_w = self._box_width(st.assigned_seat_bbox)

            # Scale thresholds to student's apparent size in the camera perspective
            adaptive_leave_px = max(self.seat_leave_dist_px, 0.72 * box_w)
            adaptive_turn_px = max(self.turning_offset_px, 0.14 * box_w)
            adaptive_speed_px = max(self.unusual_speed_px_per_frame, 0.11 * box_w)

            # -----------------------------------------------------------------
            # EVENT 1: Student leaving the assigned seat
            # -----------------------------------------------------------------
            seat_dist = self._dist(st.current_center, st.assigned_seat_center)
            if seat_dist > adaptive_leave_px:
                current_frame_events.append(
                    FrameEventObservation(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        event_type=EventType.LEAVING_SEAT,
                        confidence=st.confidence,
                        detail=f"Displaced {seat_dist:.0f}px from Seat #{seat_id}",
                    )
                )

            # -----------------------------------------------------------------
            # EVENT 2: Repeated turning toward another student
            # -----------------------------------------------------------------
            turn_offset = self._get_head_turn_offset(frame, st)
            if abs(turn_offset) >= adaptive_turn_px and seat_dist <= adaptive_leave_px:
                direction = 1 if turn_offset > 0 else -1
                mem.head_turn_frames += 1
                # Debounce direction reversals by at least 4 frames so camera noise doesn't inflate turns
                if direction != mem.last_turn_direction and (frame_idx - mem.last_turn_frame) >= 4:
                    mem.head_turn_count += 1
                    mem.last_turn_direction = direction
                    mem.last_turn_frame = frame_idx
                current_frame_events.append(
                    FrameEventObservation(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        event_type=EventType.REPEATED_TURNING,
                        confidence=st.confidence,
                        detail=f"Turn count: {mem.head_turn_count} (offset {turn_offset:+.1f}px)",
                    )
                )
            else:
                mem.last_turn_direction = 0

            # -----------------------------------------------------------------
            # EVENT 3: Mobile phone detection
            # -----------------------------------------------------------------
            if seat_id in phone_by_seat:
                ph = phone_by_seat[seat_id]
                current_frame_events.append(
                    FrameEventObservation(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        event_type=EventType.MOBILE_PHONE,
                        confidence=ph.confidence,
                        detail="Mobile phone near desk",
                    )
                )

            # -----------------------------------------------------------------
            # EVENT 4: Multiple students gathering near one location
            # -----------------------------------------------------------------
            if gathering_map[seat_id]:
                others = ", ".join(f"#{oid}" for oid in gathering_map[seat_id])
                current_frame_events.append(
                    FrameEventObservation(
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec,
                        event_type=EventType.STUDENTS_GATHERING,
                        confidence=st.confidence,
                        detail=f"Close proximity with Seat {others}",
                    )
                )

            # -----------------------------------------------------------------
            # EVENT 5: Unusual movement during examination
            # -----------------------------------------------------------------
            if len(st.trajectory) >= 4:
                recent_pts = st.trajectory[-4:]
                step_dists = [
                    self._dist(recent_pts[k], recent_pts[k + 1])
                    for k in range(len(recent_pts) - 1)
                ]
                avg_speed = sum(step_dists) / len(step_dists)
                if avg_speed >= adaptive_speed_px and seat_dist <= adaptive_leave_px:
                    current_frame_events.append(
                        FrameEventObservation(
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp_sec,
                            event_type=EventType.UNUSUAL_MOVEMENT,
                            confidence=st.confidence,
                            detail=f"Rapid motion ({avg_speed:.1f} px/frame)",
                        )
                    )

            # -----------------------------------------------------------------
            # TASK 6: UPDATE AGENT MEMORY & STATE TRANSITION
            # (NORMAL -> OBSERVE -> SUSPICIOUS -> ALERT)
            # -----------------------------------------------------------------
            self._update_student_memory_and_state(mem, current_frame_events)

        return self.memories

    def _update_student_memory_and_state(
        self,
        mem: StudentAgentMemory,
        current_frame_events: List[FrameEventObservation],
    ) -> None:
        has_event = len(current_frame_events) > 0
        mem.recent_abnormal_window.append(has_event)
        mem.active_events = current_frame_events

        active_types = {ev.event_type for ev in current_frame_events}

        for ev_type in EventType:
            prev_dur = mem.event_durations_sec.get(ev_type, 0.0)
            if ev_type in active_types:
                new_dur = prev_dur + self.dt
                mem.event_durations_sec[ev_type] = new_dur
                if prev_dur < 0.25 <= new_dur:
                    mem.event_counts[ev_type] = mem.event_counts.get(ev_type, 0) + 1
            else:
                mem.event_durations_sec[ev_type] = max(0.0, prev_dur - self.dt * 0.35)

        if has_event:
            mem.normal_streak_frames = 0
            mem.observation_history.extend(current_frame_events)
            if len(mem.observation_history) > 200:
                mem.observation_history = mem.observation_history[-200:]
        else:
            mem.normal_streak_frames += 1

        abnormal_frames_in_window = sum(1 for x in mem.recent_abnormal_window if x)
        max_duration = max(mem.event_durations_sec.values(), default=0.0)
        phone_dur = mem.event_durations_sec.get(EventType.MOBILE_PHONE, 0.0)
        gather_dur = mem.event_durations_sec.get(EventType.STUDENTS_GATHERING, 0.0)
        leave_dur = mem.event_durations_sec.get(EventType.LEAVING_SEAT, 0.0)

        if (
            phone_dur >= 1.2
            or gather_dur >= 1.4
            or leave_dur >= 1.8
            or max_duration >= 2.2
            or mem.head_turn_count >= 4
        ):
            mem.state = StudentState.ALERT
        elif (
            phone_dur >= 0.5
            or gather_dur >= 0.6
            or max_duration >= 1.0
            or mem.head_turn_count >= 2
            or abnormal_frames_in_window >= 14
        ):
            if mem.state != StudentState.ALERT or mem.normal_streak_frames > 35:
                mem.state = StudentState.SUSPICIOUS
        elif abnormal_frames_in_window >= 4 or max_duration >= 0.25:
            if mem.state not in (StudentState.ALERT, StudentState.SUSPICIOUS) or mem.normal_streak_frames > 25:
                mem.state = StudentState.OBSERVE
        else:
            if mem.normal_streak_frames > 40 or abnormal_frames_in_window <= 1:
                mem.state = StudentState.NORMAL


# =====================================================================
# MULTI-SCENARIO SYNTHETIC VIDEO GENERATOR (3 DISTINCT SCENARIOS)
# =====================================================================

def generate_scenario_exam_video(
    scenario_name: str = "Scenario 3: All 5 Anomalies + Patrolling Invigilator",
    output_path: str = "data/sample_exam_hall_phase2.mp4",
    num_frames: int = 200,
    fps: int = 25,
) -> str:
    """
    Generates 3 distinct synthetic examination hall scenarios:
    - Scenario 1: Calm Hall + Patrolling Invigilator + Single-Frame Glitch (0 False Alerts)
    - Scenario 2: Phone Usage (Seat 2) + Repeated Peeking (Seat 1) + Unusual Motion (Seat 3)
    - Scenario 3: Full Demo (All 5 Anomalies + Patrolling Invigilator)
    """
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    width, height = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    seat_centers = [
        (220, 220), (480, 220), (740, 220),
        (220, 405), (480, 405), (740, 405),
    ]

    is_s1 = "Scenario 1" in scenario_name
    is_s2 = "Scenario 2" in scenario_name
    # Default is Scenario 3 (full 5-event + invigilator)

    for f in range(num_frames):
        frame = np.full((height, width, 3), (235, 232, 225), dtype=np.uint8)

        # Invigilator Podium at top
        cv2.rectangle(frame, (380, 15), (580, 52), (90, 110, 140), -1)
        cv2.putText(frame, "INVIGILATOR ZONE", (402, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        # Draw walking Invigilator patrolling the front aisle (y=85) in Scenario 1 and Scenario 3
        if is_s1 or not is_s2:
            inv_x = int(480 + 240 * math.sin(f * 0.04))
            inv_y = 85
            cv2.rectangle(frame, (inv_x - 28, inv_y - 30), (inv_x + 28, inv_y + 38), (65, 85, 115), -1)
            cv2.circle(frame, (inv_x, inv_y - 36), 16, (180, 195, 215), -1)

        # Draw the 6 student desks and students
        for idx, (sx, sy) in enumerate(seat_centers):
            cv2.rectangle(frame, (sx - 70, sy - 45), (sx + 70, sy + 15), (150, 180, 210), -1)
            cv2.rectangle(frame, (sx - 70, sy - 45), (sx + 70, sy + 15), (90, 115, 145), 2)

            dx, dy = 0, 0
            head_shift_x = 0

            if is_s1:
                # Scenario 1: Calm students; Seat 4 has a 1-frame glitch at frame 50 (tests noise immunity)
                if idx == 3 and f == 50:
                    dx = 16
                    head_shift_x = 18
            elif is_s2:
                # Scenario 2: Seat 1 repeated turning, Seat 2 mobile phone, Seat 3 unusual movement
                if idx == 0 and 25 <= f <= 165:
                    turn_wave = math.sin((f - 25) * 0.25)
                    dx = int(20 * turn_wave)
                    head_shift_x = int(18 * turn_wave)
                elif idx == 2 and 70 <= f <= 145:
                    dx = int(18 * (1 if (f // 2) % 2 == 0 else -1))
                    dy = int(10 * (1 if f % 2 == 0 else -1))
            else:
                # Scenario 3: All 5 events + 1-frame noise test on Seat 4
                if idx == 0 and 25 <= f <= 165:
                    turn_wave = math.sin((f - 25) * 0.25)
                    dx = int(20 * turn_wave)
                    head_shift_x = int(18 * turn_wave)
                elif idx == 2 and 70 <= f <= 145:
                    dx = int(18 * (1 if (f // 2) % 2 == 0 else -1))
                    dy = int(10 * (1 if f % 2 == 0 else -1))
                elif idx == 3 and f == 50:
                    dx = 16
                    head_shift_x = 18
                elif idx == 4 and f > 55:
                    dx = int(min(155, (f - 55) * 2.4))

            cx, cy = sx + dx, sy + dy + 15
            cv2.rectangle(frame, (cx - 32, cy - 35), (cx + 32, cy + 45), (65, 85, 115), -1)
            cv2.circle(frame, (cx + head_shift_x, cy - 42), 18, (180, 195, 215), -1)

            # Mobile phone on Seat 2 in Scenario 2 & Scenario 3
            if not is_s1 and idx == 1 and 40 <= f <= 130:
                px1, py1, px2, py2 = cx + 22, cy - 28, cx + 40, cy - 2
                cv2.rectangle(frame, (px1, py1), (px2, py2), (20, 20, 20), -1)
                cv2.rectangle(frame, (px1 + 2, py1 + 3), (px2 - 2, py2 - 3), (200, 240, 255), -1)

        writer.write(frame)

    writer.release()
    return output_path


def generate_phase2_exam_video(
    output_path: str = "data/sample_exam_hall_phase2.mp4",
    num_frames: int = 200,
    fps: int = 25,
) -> str:
    """Backward-compatible wrapper for Scenario 3 default video."""
    return generate_scenario_exam_video(
        scenario_name="Scenario 3: All 5 Anomalies + Patrolling Invigilator",
        output_path=output_path,
        num_frames=num_frames,
        fps=fps,
    )