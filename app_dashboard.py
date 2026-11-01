"""
Phase 4: Invigilator Dashboard UI (Task 12 - Hybrid Multi-Scenario + Real CCTV Edition)
Project: Intelligent Examination Hall Monitoring and Anomaly Detection Agent

source venv/bin/activate
streamlit run app_dashboard.py

Displays:
- Examination Hall live CCTV view with highlighted anomaly bounding boxes & Blue Invigilator tag
- Highlighted suspicious locations and seat status cards (NORMAL / OBSERVE / SUSPICIOUS / ALERT / INVIGILATOR)
- 3 Built-in Synthetic Scenarios + Real CCTV Video Uploader with Perspective-Adaptive Scaling
- Real-time Alert History log table with filters
- Supporting Evidence snapshot viewer showing captured infraction images
"""

import glob
import os
import sys
import pandas as pd
import streamlit as st
import cv2
import numpy as np

# Ensure current directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phase1_video_detection_tracking import (
    Detection,
    ExamHallDetector,
    StudentSeatTracker,
    VideoInputLoader,
)
from phase2_activity_agent import (
    EventType,
    ExamMonitoringAgent,
    StudentState,
    generate_scenario_exam_video,
)
from phase3_reasoning_logger import (
    AlertRecord,
    EventEvidenceLogger,
    ReasoningEngine,
    RiskLevel,
    RISK_COLORS,
    draw_phase3_annotations,
)

# Page Configuration
st.set_page_config(
    page_title="AI Exam Hall Invigilator Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for clean cards and status badges
st.markdown(
    "<style>"
    ".main-title { font-size: 26px; font-weight: 700; margin-bottom: 2px; }"
    ".sub-title { color: #6c757d; font-size: 14px; margin-bottom: 18px; }"
    ".seat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(155px, 1fr)); gap: 10px; margin-bottom: 14px; }"
    ".seat-card { background: #f8f9fa; border: 1px solid #e9ecef; padding: 10px 12px; border-radius: 8px; }"
    ".seat-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }"
    ".seat-meta { font-size: 12px; color: #495057; }"
    ".badge-alert { background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }"
    ".badge-suspicious { background-color: #fd7e14; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }"
    ".badge-observe { background-color: #ffc107; color: #212529; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }"
    ".badge-normal { background-color: #198754; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }"
    ".badge-invigilator { background-color: #0d6efd; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }"
    ".alert-item { background: #ffffff; padding: 8px 12px; border-radius: 6px; border: 1px solid #e9ecef; margin-bottom: 6px; font-size: 13px; }"
    "</style>",
    unsafe_allow_html=True,
)

csv_log_path = "data/alerts_log.csv"
evidence_dir = "data/evidence_frames"
os.makedirs(evidence_dir, exist_ok=True)


def clear_previous_session_data() -> None:
    """Deletes all saved evidence frames and resets alerts_log.csv so previous video data never mixes."""
    for file_path in glob.glob(os.path.join(evidence_dir, "*")):
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except OSError:
            pass
    if os.path.exists(csv_log_path):
        try:
            os.remove(csv_log_path)
        except OSError:
            pass


# =====================================================================
# SIDEBAR: CONTROLS, SCENARIO SELECTOR & REAL VIDEO UPLOADER
# =====================================================================

st.sidebar.title("⚙️ Monitoring Controls")

video_source_type = st.sidebar.radio(
    "Select Video Source",
    options=[
        "Synthetic Exam Hall Scenarios",
        "Upload Real CCTV Video",
        "Enter Local Video Path",
    ],
)

selected_scenario = "Scenario 3: All 5 Anomalies + Patrolling Invigilator"
if video_source_type == "Synthetic Exam Hall Scenarios":
    selected_scenario = st.sidebar.selectbox(
        "Choose Synthetic Scenario",
        options=[
            "Scenario 1: Calm Hall + Walking Invigilator (0 False Alerts)",
            "Scenario 2: Mobile Phone + Repeated Head Turning",
            "Scenario 3: All 5 Anomalies + Patrolling Invigilator",
        ],
        index=2,
    )

# Automatically clear old frames & logs whenever the user switches the source or scenario
session_key = f"{video_source_type}::{selected_scenario}"
if "last_session_key" not in st.session_state:
    st.session_state.last_session_key = session_key
elif st.session_state.last_session_key != session_key:
    clear_previous_session_data()
    st.session_state.last_session_key = session_key

video_path_to_process = ""
temp_video_path = "data/uploaded_exam_cctv.mp4"

if video_source_type == "Synthetic Exam Hall Scenarios":
    scenario_slug = selected_scenario.split(":")[0].lower().replace(" ", "_")
    demo_path = f"data/sample_{scenario_slug}.mp4"
    if not os.path.exists(demo_path) or os.path.getsize(demo_path) < 1024:
        st.sidebar.info(f"Generating {selected_scenario.split(':')[0]}...")
        demo_path = generate_scenario_exam_video(selected_scenario, demo_path)
    video_path_to_process = demo_path

elif video_source_type == "Upload Real CCTV Video":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Examination CCTV Video (.mp4, .avi, .mov)",
        type=["mp4", "avi", "mov"],
    )
    if uploaded_file is not None:
        os.makedirs("data", exist_ok=True)
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.read())
        video_path_to_process = temp_video_path
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    else:
        st.sidebar.warning("Please upload a video file to begin.")

else:
    custom_path = st.sidebar.text_input("Local File Path", value="data/sample_scenario_3.mp4")
    if os.path.exists(custom_path):
        video_path_to_process = custom_path
    else:
        st.sidebar.error("File not found at specified path.")

st.sidebar.markdown("---")
st.sidebar.subheader("Detection & Hybrid Filters")
enable_invig_filter = st.sidebar.checkbox("Enable Standing Invigilator Filter", value=True, help="Automatically tags walking/standing teachers as INVIGILATOR and ignores them for cheating alerts")
conf_threshold = st.sidebar.slider("YOLO Confidence Threshold", 0.15, 0.85, 0.45, 0.05)
frame_stride = st.sidebar.slider("Frame Stride (Speedup)", 1, 5, 2, 1, help="Process every Nth frame for real-time speed")
seat_leave_dist = st.sidebar.slider("Base Seat Leave Threshold (px)", 30, 150, 65, 5)

run_monitoring = st.sidebar.button("▶️ Start Monitoring", type="primary", use_container_width=True)

if st.sidebar.button("🗑️ Clear Saved Frames & Logs", use_container_width=True):
    clear_previous_session_data()
    st.sidebar.success("Cleared all previous frames and alert logs!")


# =====================================================================
# MAIN DASHBOARD INTERFACE
# =====================================================================

st.markdown('<div class="main-title">AI Examination Hall Monitoring & Invigilator Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid Perspective-Adaptive Anomaly Detection, Invigilator Filtering & Evidence Logging</div>', unsafe_allow_html=True)

# Top KPI Summary Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
metric_students = kpi1.empty()
metric_normal = kpi2.empty()
metric_observe = kpi3.empty()
metric_suspicious = kpi4.empty()
metric_alert = kpi5.empty()

# Layout: Left = Live Hall View, Right = Real-Time Seat Status & Alerts
col_video, col_status = st.columns([1.45, 1.0])

with col_video:
    st.markdown("##### 📹 Live Examination Hall CCTV Feed")
    video_placeholder = st.empty()
    status_bar_placeholder = st.empty()

with col_status:
    st.markdown("##### 🪑 Examination Hall Seat Map & Roles")
    seats_container = st.empty()
    st.markdown("##### 🚨 Live Alert Feed")
    live_alert_placeholder = st.empty()

st.markdown("---")

# Bottom Tabs: Alert History and Evidence Gallery
tab_history, tab_evidence = st.tabs(["📋 Alert History Log", "📸 Supporting Evidence Snapshots"])


# =====================================================================
# LIVE PROCESSING PIPELINE RUNNER
# =====================================================================

TARGET_WIDTH = 960
TARGET_HEIGHT = 540

if run_monitoring and video_path_to_process:
    if not os.path.exists(video_path_to_process):
        st.error(f"Cannot start: Video file '{video_path_to_process}' not found.")
    else:
        # Automatically wipe previous video's saved frames and CSV logs before starting new run
        clear_previous_session_data()

        loader = VideoInputLoader(video_path_to_process)
        detector = ExamHallDetector(conf_threshold=conf_threshold)
        tracker = StudentSeatTracker(max_match_distance=160.0)
        agent = ExamMonitoringAgent(
            fps=loader.fps,
            seat_leave_dist_px=seat_leave_dist,
            enable_invigilator_filter=enable_invig_filter,
        )
        reasoning = ReasoningEngine()
        logger = EventEvidenceLogger(log_csv_path=csv_log_path, evidence_dir=evidence_dir)

        frame_idx = 0
        cached_detections = []
        progress_bar = st.progress(0.0)

        while True:
            ret, raw_frame, timestamp_sec = loader.read_frame()
            if not ret or raw_frame is None:
                break

            frame_idx += 1

            # Normalize high-res (1080p / 4K) videos to 960x540 for fast processing & consistent scaling
            if raw_frame.shape[1] != TARGET_WIDTH or raw_frame.shape[0] != TARGET_HEIGHT:
                frame = cv2.resize(raw_frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            else:
                frame = raw_frame

            # 1. Detection with stride
            if frame_idx == 1 or frame_idx % frame_stride == 0:
                cached_detections = detector.detect(frame)

            # 2. Tracking & Seat Mapping
            students, phones = tracker.update(cached_detections, frame_idx)

            # 3. Model-Based Agent & Memory (with Invigilator Filter + Perspective Scaling)
            memories = agent.update(frame, students, phones, frame_idx, timestamp_sec)

            # 4. Rule-Based Reasoning & Logging (skips Invigilators automatically)
            for seat_id, mem in memories.items():
                if seat_id not in students or mem.is_invigilator:
                    continue
                triggered = reasoning.evaluate_student(seat_id, mem, timestamp_sec)
                for ev_type, risk_lvl, score, dur, freq, explanation in triggered:
                    logger.log_alert(
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

            # Update KPI Cards (count only seated students, not the invigilator)
            counts = {s: 0 for s in StudentState}
            for m in memories.values():
                counts[m.state] += 1

            total_students_only = max(0, len(students) - counts[StudentState.INVIGILATOR])
            metric_students.metric("Students (+Invig.)", f"{total_students_only} ({counts[StudentState.INVIGILATOR]})")
            metric_normal.metric("Normal", counts[StudentState.NORMAL])
            metric_observe.metric("Observe", counts[StudentState.OBSERVE])
            metric_suspicious.metric("Suspicious", counts[StudentState.SUSPICIOUS])
            metric_alert.metric("Critical Alerts", counts[StudentState.ALERT])

            # Annotate frame (including blue INVIGILATOR box)
            annotated = draw_phase3_annotations(
                frame, students, phones, memories, logger.alert_history, frame_idx, timestamp_sec
            )
            for sid, st_obj in students.items():
                mem = memories.get(sid)
                if mem and mem.is_invigilator:
                    x1, y1, x2, y2 = st_obj.current_bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (210, 120, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + 125, y1), (210, 120, 0), -1)
                    cv2.putText(annotated, "INVIGILATOR", (x1 + 5, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)

            status_bar_placeholder.info(
                f"⏱️ Video Time: **{timestamp_sec:.1f}s** | Frame: **{frame_idx}/{loader.total_frames or 'Live'}** | Logged Alerts: **{len(logger.alert_history)}**"
            )

            # Update Seat Grid Cards
            cards_list = []
            for sid in sorted(students.keys()):
                mem = memories.get(sid)
                state_val = mem.state.value if mem else "NORMAL"
                badge_class = f"badge-{state_val.lower()}"
                if mem and mem.is_invigilator:
                    cards_list.append(
                        f"<div class='seat-card'>"
                        f"<div class='seat-header'><b>ID #{sid}</b><span class='{badge_class}'>INVIGILATOR</span></div>"
                        f"<div class='seat-meta'>Role: Hall Patrol (Ignored)</div>"
                        f"</div>"
                    )
                else:
                    turns = mem.head_turn_count if mem else 0
                    max_dur = max(mem.event_durations_sec.values(), default=0.0) if mem else 0.0
                    cards_list.append(
                        f"<div class='seat-card'>"
                        f"<div class='seat-header'><b>Seat #{sid}</b><span class='{badge_class}'>{state_val}</span></div>"
                        f"<div class='seat-meta'>Turns: {turns} | Active: {max_dur:.1f}s</div>"
                        f"</div>"
                    )
            seats_container.markdown(
                f"<div class='seat-grid'>{''.join(cards_list)}</div>",
                unsafe_allow_html=True,
            )

            # Update Latest Alerts
            if logger.alert_history:
                alert_items = []
                for rec in reversed(logger.alert_history[-4:]):
                    border_color = "#dc3545" if rec.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL) else "#fd7e14"
                    alert_items.append(
                        f"<div class='alert-item' style='border-left: 4px solid {border_color};'>"
                        f"<b>{rec.explanation}</b><br>"
                        f"<span style='color: #6c757d; font-size: 11px;'>⏱️ {rec.time_str} ({rec.timestamp_sec}s)</span>"
                        f"</div>"
                    )
                live_alert_placeholder.markdown("".join(alert_items), unsafe_allow_html=True)
            else:
                live_alert_placeholder.caption("No alerts triggered. Hall is calm.")

            if loader.total_frames > 0:
                progress_bar.progress(min(1.0, frame_idx / loader.total_frames))

        loader.release()
        st.success("✅ Examination monitoring completed!")


# =====================================================================
# BOTTOM TABS: ALERT HISTORY & EVIDENCE GALLERY
# =====================================================================

with tab_history:
    if os.path.exists(csv_log_path) and os.path.getsize(csv_log_path) > 50:
        df = pd.read_csv(csv_log_path)
        col_f1, _ = st.columns([1, 2])
        selected_risk = col_f1.multiselect(
            "Filter by Risk Level",
            options=["Low", "Medium", "High", "Critical"],
            default=["Low", "Medium", "High", "Critical"],
        )
        if selected_risk:
            df = df[df["Risk Level"].isin(selected_risk)]

        st.dataframe(
            df[[
                "Timestamp (s)", "Wall Time", "Seat/Student ID", "Event Type",
                "Duration (s)", "Frequency", "Risk Score", "Risk Level", "Explainable Alert"
            ]],
            use_container_width=True,
            hide_index=True,
        )
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Filtered Log CSV", data=csv_data, file_name="exam_hall_alerts_log.csv", mime="text/csv")
    else:
        st.info("No alert records found yet. Click '▶️ Start Monitoring' to run detection and populate the log.")

with tab_evidence:
    if os.path.exists(evidence_dir):
        image_files = [
            f for f in os.listdir(evidence_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ]
        image_files.sort(key=lambda f: os.path.getmtime(os.path.join(evidence_dir, f)), reverse=True)

        if image_files:
            st.caption(f"Showing **{len(image_files)}** captured evidence snapshots from the current run:")
            cols = st.columns(3)
            for idx, img_name in enumerate(image_files[:12]):
                img_path = os.path.join(evidence_dir, img_name)
                col = cols[idx % 3]
                with col:
                    st.image(img_path, caption=img_name, use_container_width=True)
        else:
            st.info("No evidence images captured yet. Click '▶️ Start Monitoring' to begin.")
    else:
        st.info("Evidence directory not created yet.")