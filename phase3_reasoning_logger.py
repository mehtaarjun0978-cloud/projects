"""
Phase 3: Rule-Based Reasoning, Risk Scoring, Explainable Alerts & Evidence Logging
Project: Intelligent Examination Hall Monitoring and Anomaly Detection Agent

source venv/bin/activate
streamlit run app_dashboard.py

Covers:
- Task 7 (Step 6): Rule-based reasoning system (IF count > th AND duration > T THEN Suspicion_Level).
- Task 8 (Step 7): Risk/certainty score based on confidence, duration, frequency, and activity type.
- Task 9 (Step 8): Temporal reasoning (repeated suspicious actions progressively escalate risk level).
- Task 10 (Step 9): Explainable alert generator (e.g. Student/Seat 18 -> Repeated side movement detected for 14 sec -> Risk: High).
- Task 11 (Step 10): Store detected events to CSV with Time, Seat ID, Event Type, Confidence, Evidence Frame path, and Risk Level.
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure current script folder is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import Phase 1 & Phase 2 components
try:
    from phase1_video_detection_tracking import (
        Detection,
        ExamHallDetector,
        StudentSeatTracker,
        TrackedStudent,
        VideoInputLoader,
    )
    from phase2_activity_agent import (
        EventType,
        ExamMonitoringAgent,
        StudentAgentMemory,
        StudentState,
        generate_phase2_exam_video,
    )
except ImportError as e:
    print(f"[Warning] Importing from local directory failed: {e}. Please ensure phase1 and phase2 files are in the same folder.")
    raise


# =====================================================================
# TASK 8: RISK LEVEL ENUM & SEVERITY WEIGHTS
# =====================================================================

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# Base severity weight per activity type (scale 1.0 to 3.0)
ACTIVITY_BASE_WEIGHT: Dict[EventType, float] = {
    EventType.UNUSUAL_MOVEMENT: 1.1,
    EventType.REPEATED_TURNING: 1.5,
    EventType.LEAVING_SEAT: 2.1,
    EventType.STUDENTS_GATHERING: 2.4,
    EventType.MOBILE_PHONE: 2.8,
}

RISK_COLORS: Dict[RiskLevel, Tuple[int, int, int]] = {
    RiskLevel.LOW: (0, 200, 100),       # Greenish
    RiskLevel.MEDIUM: (0, 215, 255),    # Yellow
    RiskLevel.HIGH: (0, 140, 255),      # Orange
    RiskLevel.CRITICAL: (0, 0, 235),    # Red
}


# =====================================================================
# TASK 10 & 11: EXPLAINABLE ALERT & EVENT RECORD
# =====================================================================

@dataclass
class AlertRecord:
    """Task 10 & 11: A structured, explainable alert record."""
    timestamp_sec: float
    time_str: str
    seat_id: int
    event_type: str
    confidence: float
    duration_sec: float
    frequency_count: int
    risk_score_numeric: float
    risk_level: RiskLevel
    explanation: str
    evidence_frame_path: str


# =====================================================================
# TASK 7, 8, 9: RULE-BASED & TEMPORAL REASONING ENGINE
# =====================================================================

class ReasoningEngine:
    """
    Evaluates agent memory using:
    - Task 7: IF-THEN rules on event duration and count
    - Task 8: Risk score formula combining confidence, duration, frequency, and activity type
    - Task 9: Temporal escalation for repeated or concurrent infractions
    """

    def __init__(self, alert_cooldown_sec: float = 3.0):
        # Cooldown timer per seat to avoid logging duplicate alerts on consecutive frames
        self.last_alert_time: Dict[Tuple[int, EventType], float] = {}
        self.alert_cooldown_sec: float = alert_cooldown_sec

    def evaluate_student(
        self,
        seat_id: int,
        mem: StudentAgentMemory,
        current_time: float,
    ) -> List[Tuple[EventType, RiskLevel, float, float, int, str]]:
        """
        Evaluates rules against student's memory and active events.
        Returns a list of triggered alerts:
        (event_type, risk_level, risk_score_numeric, duration_sec, frequency, explanation)
        """
        triggered_alerts: List[Tuple[EventType, RiskLevel, float, float, int, str]] = []

        for ev in mem.active_events:
            ev_type = ev.event_type
            dur = mem.event_durations_sec.get(ev_type, 0.0)
            count = mem.event_counts.get(ev_type, 1)

            # Check per-event cooldown
            key = (seat_id, ev_type)
            last_time = self.last_alert_time.get(key, -999.0)
            if (current_time - last_time) < self.alert_cooldown_sec:
                continue

            # -------------------------------------------------------------
            # TASK 7: RULE-BASED REASONING (IF ... AND ... THEN ...)
            # -------------------------------------------------------------
            rule_matched = False
            assigned_risk = RiskLevel.LOW
            rule_reason = ""

            # Rule 1: Mobile Phone
            if ev_type == EventType.MOBILE_PHONE:
                if dur >= 1.5 or count >= 2:
                    rule_matched = True
                    assigned_risk = RiskLevel.CRITICAL
                    rule_reason = f"Mobile phone active for {dur:.1f}s (detected {count}x)"
                elif dur >= 0.4:
                    rule_matched = True
                    assigned_risk = RiskLevel.HIGH
                    rule_reason = f"Mobile phone visible near desk ({dur:.1f}s)"

            # Rule 2: Multiple Students Gathering
            elif ev_type == EventType.STUDENTS_GATHERING:
                if dur >= 2.0 or count >= 2:
                    rule_matched = True
                    assigned_risk = RiskLevel.CRITICAL
                    rule_reason = f"Sustained gathering with adjacent student for {dur:.1f}s"
                elif dur >= 0.8:
                    rule_matched = True
                    assigned_risk = RiskLevel.HIGH
                    rule_reason = f"Students converged within close proximity for {dur:.1f}s"

            # Rule 3: Leaving Assigned Seat
            elif ev_type == EventType.LEAVING_SEAT:
                if dur >= 2.5:
                    rule_matched = True
                    assigned_risk = RiskLevel.CRITICAL
                    rule_reason = f"Student completely vacated assigned desk for {dur:.1f}s"
                elif dur >= 1.0:
                    rule_matched = True
                    assigned_risk = RiskLevel.HIGH
                    rule_reason = f"Significant displacement from seat area for {dur:.1f}s"

            # Rule 4: Repeated Turning toward Neighbor
            elif ev_type == EventType.REPEATED_TURNING:
                turns = mem.head_turn_count
                if turns >= 5 and dur >= 2.0:
                    rule_matched = True
                    assigned_risk = RiskLevel.HIGH
                    rule_reason = f"Repeated side movement detected for {dur:.0f} sec ({turns} turns)"
                elif turns >= 3 or dur >= 1.2:
                    rule_matched = True
                    assigned_risk = RiskLevel.MEDIUM
                    rule_reason = f"Turning toward adjacent student ({turns} turns, {dur:.1f}s)"

            # Rule 5: Unusual Movement
            elif ev_type == EventType.UNUSUAL_MOVEMENT:
                if dur >= 2.0 or count >= 3:
                    rule_matched = True
                    assigned_risk = RiskLevel.MEDIUM
                    rule_reason = f"Persistent abnormal movement in seat for {dur:.1f}s"
                elif dur >= 0.8:
                    rule_matched = True
                    assigned_risk = RiskLevel.LOW
                    rule_reason = f"Brief burst of unusual movement for {dur:.1f}s"

            if not rule_matched:
                continue

            # -------------------------------------------------------------
            # TASK 8 & 9: RISK SCORE CALCULATION & TEMPORAL ESCALATION
            # Score = Confidence * [BaseSeverity * DurationFactor * FrequencyFactor]
            # -------------------------------------------------------------
            confidence = ev.confidence
            base_sev = ACTIVITY_BASE_WEIGHT.get(ev_type, 1.0)
            dur_factor = min(2.5, 0.6 + (dur * 0.45))
            freq_factor = min(2.0, 1.0 + ((count - 1) * 0.35))

            # Temporal escalation: If student is already in SUSPICIOUS/ALERT or has concurrent infractions
            active_violations_count = len(mem.active_events)
            temporal_multiplier = 1.0
            if mem.state == StudentState.ALERT:
                temporal_multiplier += 0.35
            elif mem.state == StudentState.SUSPICIOUS:
                temporal_multiplier += 0.18
            if active_violations_count > 1:
                temporal_multiplier += 0.25 * (active_violations_count - 1)

            raw_score = confidence * base_sev * dur_factor * freq_factor * temporal_multiplier
            normalized_score = min(100.0, raw_score * 16.5)

            # Map score to final RiskLevel
            if normalized_score >= 80.0:
                final_level = RiskLevel.CRITICAL
            elif normalized_score >= 55.0:
                final_level = RiskLevel.HIGH
            elif normalized_score >= 32.0:
                final_level = RiskLevel.MEDIUM
            else:
                final_level = RiskLevel.LOW

            # Task 10: Generate explainable alert
            explanation = f"Student/Seat {seat_id} -> {rule_reason} -> Risk: {final_level.value}"

            self.last_alert_time[key] = current_time
            triggered_alerts.append(
                (ev_type, final_level, normalized_score, dur, count, explanation)
            )

        return triggered_alerts


# =====================================================================
# TASK 11: EVIDENCE CAPTURE & STRUCTURED LOGGING
# =====================================================================

class EventEvidenceLogger:
    """
    Task 11: Stores detected alerts with timestamp, Seat ID, Event Type,
    Confidence, Evidence Frame, and Risk Level into a CSV file and folder.
    """

    def __init__(
        self,
        log_csv_path: str = "data/alerts_log.csv",
        evidence_dir: str = "data/evidence_frames",
    ):
        self.log_csv_path = log_csv_path
        self.evidence_dir = evidence_dir
        os.makedirs(self.evidence_dir, exist_ok=True)
        csv_parent = os.path.dirname(self.log_csv_path)
        if csv_parent:
            os.makedirs(csv_parent, exist_ok=True)

        self._init_csv()
        self.alert_history: List[AlertRecord] = []

    def _init_csv(self):
        """Initializes CSV header if file doesn't exist."""
        if not os.path.exists(self.log_csv_path) or os.path.getsize(self.log_csv_path) == 0:
            with open(self.log_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp (s)",
                    "Wall Time",
                    "Seat/Student ID",
                    "Event Type",
                    "Confidence",
                    "Duration (s)",
                    "Frequency",
                    "Risk Score",
                    "Risk Level",
                    "Explainable Alert",
                    "Evidence Frame Path",
                ])

    def log_alert(
        self,
        frame: np.ndarray,
        seat_id: int,
        student: TrackedStudent,
        event_type: EventType,
        confidence: float,
        duration: float,
        frequency: int,
        risk_score: float,
        risk_level: RiskLevel,
        explanation: str,
        timestamp_sec: float,
    ) -> AlertRecord:
        """Saves an annotated evidence frame to disk and writes a row in the CSV."""
        wall_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        evidence_frame = frame.copy()
        x1, y1, x2, y2 = student.current_bbox
        box_color = RISK_COLORS[risk_level]

        # Draw highlight box and alert label
        cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), box_color, 3, cv2.LINE_AA)
        badge = f"Seat #{seat_id} | {risk_level.value} Risk: {event_type.value}"
        cv2.rectangle(evidence_frame, (x1, max(0, y1 - 26)), (x1 + 360, y1), box_color, -1)
        cv2.putText(
            evidence_frame,
            badge,
            (x1 + 6, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        filename = f"evidence_seat_{seat_id}_{event_type.name}_{int(timestamp_sec * 100):05d}.jpg"
        evidence_path = os.path.join(self.evidence_dir, filename)
        cv2.imwrite(evidence_path, evidence_frame)

        record = AlertRecord(
            timestamp_sec=round(timestamp_sec, 2),
            time_str=wall_time,
            seat_id=seat_id,
            event_type=event_type.value,
            confidence=round(confidence, 2),
            duration_sec=round(duration, 1),
            frequency_count=frequency,
            risk_score_numeric=round(risk_score, 1),
            risk_level=risk_level,
            explanation=explanation,
            evidence_frame_path=evidence_path,
        )
        self.alert_history.append(record)

        with open(self.log_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record.timestamp_sec,
                record.time_str,
                record.seat_id,
                record.event_type,
                record.confidence,
                record.duration_sec,
                record.frequency_count,
                record.risk_score_numeric,
                record.risk_level.value,
                record.explanation,
                record.evidence_frame_path,
            ])

        return record


# =====================================================================
# PHASE 3 VISUALIZATION & PIPELINE RUNNER
# =====================================================================

def draw_phase3_annotations(
    frame: np.ndarray,
    students: Dict[int, TrackedStudent],
    phones: List[Detection],
    memories: Dict[int, StudentAgentMemory],
    recent_alerts: List[AlertRecord],
    frame_idx: int,
    timestamp_sec: float,
) -> np.ndarray:
    """Draws live video overlay with risk badges, active alerts, and alert ticker."""
    vis = frame.copy()

    for seat_id, st in sorted(students.items()):
        mem = memories.get(seat_id, StudentAgentMemory(seat_id=seat_id))
        x1, y1, x2, y2 = st.current_bbox

        if mem.state == StudentState.ALERT:
            color = (0, 0, 235)  # Red
        elif mem.state == StudentState.SUSPICIOUS:
            color = (0, 140, 255)  # Orange
        elif mem.state == StudentState.OBSERVE:
            color = (0, 215, 255)  # Yellow
        else:
            color = (0, 180, 60)   # Green

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"Seat #{seat_id} [{mem.state.value}]"
        cv2.putText(vis, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

    for ph in phones:
        px1, py1, px2, py2 = ph.bbox
        cv2.rectangle(vis, (px1, py1), (px2, py2), (0, 0, 235), 2, cv2.LINE_AA)
        cv2.putText(vis, "PHONE", (px1, max(15, py1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 235), 1)

    # Bottom alert ticker
    cv2.rectangle(vis, (0, vis.shape[0] - 38), (vis.shape[1], vis.shape[0]), (25, 25, 25), -1)
    if recent_alerts:
        latest = recent_alerts[-1]
        cv2.putText(
            vis,
            f"[LATEST ALERT] {latest.explanation}",
            (12, vis.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            RISK_COLORS[latest.risk_level],
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            vis,
            "[STATUS] All seats normal. Temporal reasoning active.",
            (12, vis.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (180, 220, 180),
            1,
            cv2.LINE_AA,
        )

    # Top header bar
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (30, 30, 30), -1)
    header = (
        f"PHASE 3: Reasoning & Logging | Time: {timestamp_sec:05.2f}s | "
        f"Total Logged Alerts: {len(recent_alerts)}"
    )
    cv2.putText(vis, header, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


def run_phase3_pipeline(
    video_path: str,
    output_video_path: str = "data/phase3_reasoning_output.mp4",
    log_csv_path: str = "data/alerts_log.csv",
    evidence_dir: str = "data/evidence_frames",
    model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.35,
    frame_stride: int = 1,
    show_window: bool = False,
) -> List[AlertRecord]:
    """
    Executes Phase 1 + Phase 2 + Phase 3 in a unified, efficient loop.
    Generates explainable alerts and saves structured evidence to CSV and disk.
    """
    out_dir = os.path.dirname(output_video_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    loader = VideoInputLoader(video_path)
    detector = ExamHallDetector(model_path=model_path, conf_threshold=conf_threshold)
    tracker = StudentSeatTracker()
    agent = ExamMonitoringAgent(fps=loader.fps)
    reasoning_engine = ReasoningEngine()
    logger = EventEvidenceLogger(log_csv_path=log_csv_path, evidence_dir=evidence_dir)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, loader.fps, (loader.width, loader.height))

    frame_idx = 0
    cached_detections: List[Detection] = []

    print("\n[Phase 3 Pipeline Started] Monitoring Examination Hall...")

    while True:
        ret, frame, timestamp_sec = loader.read_frame()
        if not ret or frame is None:
            break

        frame_idx += 1

        if frame_idx == 1 or frame_idx % max(1, frame_stride) == 0:
            cached_detections = detector.detect(frame)

        students, phones = tracker.update(cached_detections, frame_idx)
        memories = agent.update(frame, students, phones, frame_idx, timestamp_sec)

        # Phase 3: Evaluate rules and log alerts
        for seat_id, mem in memories.items():
            if seat_id not in students:
                continue
            triggered = reasoning_engine.evaluate_student(seat_id, mem, timestamp_sec)
            for ev_type, risk_lvl, score, dur, freq, explanation in triggered:
                record = logger.log_alert(
                    frame=frame,
                    seat_id=seat_id,
                    student=students[seat_id],
                    event_type=ev_type,
                    confidence=students[seat_id].confidence,
                    duration=dur,
                    frequency=freq,
                    risk_score=score,
                    risk_level=risk_lvl,
                    explanation=explanation,
                    timestamp_sec=timestamp_sec,
                )
                print(f"  [ALERT LOGGED] {record.explanation}")

        annotated = draw_phase3_annotations(
            frame, students, phones, memories, logger.alert_history, frame_idx, timestamp_sec
        )
        writer.write(annotated)

        if show_window:
            cv2.imshow("Phase 3 - Reasoning & Alert System", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    loader.release()
    writer.release()
    if show_window:
        cv2.destroyAllWindows()

    print(f"\n================ PHASE 3 SUMMARY ================")
    print(f"Total Frames Processed: {frame_idx}")
    print(f"Total Alerts Logged: {len(logger.alert_history)}")
    print(f"Alerts CSV Saved to: {logger.log_csv_path}")
    print(f"Evidence Frames Saved in: {logger.evidence_dir}/")
    print("=================================================\n")
    return logger.alert_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Reasoning, Risk Scoring, Alerts & Logging")
    parser.add_argument("--video", type=str, default="data/sample_exam_hall_phase2.mp4", help="Input CCTV video path")
    parser.add_argument("--output", type=str, default="data/phase3_reasoning_output.mp4", help="Output video path")
    parser.add_argument("--csv", type=str, default="data/alerts_log.csv", help="Alerts log CSV output path")
    parser.add_argument("--evidence", type=str, default="data/evidence_frames", help="Evidence images directory")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 weights path")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for processing speed")
    parser.add_argument("--show", action="store_true", help="Display live OpenCV window")
    args = parser.parse_args()

    input_video = args.video
    if not input_video or not os.path.exists(input_video) or os.path.getsize(input_video) < 1024:
        target_path = input_video if input_video else "data/sample_exam_hall_phase2.mp4"
        print(f"[Info] Generating Phase 2/3 test video at '{target_path}'...")
        input_video = generate_phase2_exam_video(target_path)

    run_phase3_pipeline(
        video_path=input_video,
        output_video_path=args.output,
        log_csv_path=args.csv,
        evidence_dir=args.evidence,
        model_path=args.model,
        conf_threshold=args.conf,
        frame_stride=args.stride,
        show_window=args.show,
    )