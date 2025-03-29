import logging
from ultralytics import solutions
from tracker import Tracker
import streamlit as st
import cv2
import numpy as np
import subprocess
from datetime import datetime, timedelta
from supabase import create_client, Client
from zoneinfo import ZoneInfo

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Supabase configuration
SUPABASE_URL = "https://fktheptskxttqdcksztd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZrdGhlcHRza3h0dHFkY2tzenRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI5MjcwOTYsImV4cCI6MjA1ODUwMzA5Nn0.NiWRw28MXA0h6_5a8SboKYy3SBUmZlKFJr1II7SvZZI"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Timezone configuration
CET_TZ = ZoneInfo("Europe/Brussels")

CLASSES = {
    0: "CYC", 1: "B", 2: "B", 3: "P", 4: "C"
}

# ROI Lines
ROI_LINE_1 = 250
ROI_LINE_2 = 800
CROSSING_LINE = 540

STREAM_URL = "https://livecam.brucity.be/LiveBrusselsWebcams/streams/fDdnnEmqOn6Kyy3E1701416388577.m3u8"

def update_supabase(interval_counts):
    """Updates only interval_tracking table"""
    try:
        supabase.table("interval_tracking").upsert({
            "the_date": interval_counts["CET_date"].isoformat(),
            "time_interval": interval_counts["CET_interval"],
            "cyc": int(interval_counts["CYC"]),
            "b": int(interval_counts["B"]),
            "p": int(interval_counts["P"]),
            "c": int(interval_counts["C"])
        }, on_conflict="the_date,time_interval").execute()
    except Exception as e:
        logging.error(f"Supabase error: {str(e)}")

def main():
    st.set_page_config(
        page_title = 'Live Detection',
        page_icon = "🔴",
    )

    st.title("Altnova Live Detection")

    stframe = st.empty()
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-re",
            "-i",
            STREAM_URL,
            "-f",
            "image2pipe",
            "-pix_fmt",
            "bgr24",
            "-vcodec",
            "rawvideo",
            "-bufsize",
            "512k",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=10**8,
    )

    live_counts = {c: 0 for c in CLASSES.values()}
    interval_counts = {"CET_date": None, "CET_interval": None, "CYC": 0, "B": 0, "P": 0, "C": 0}
    counted_ids = set()
    trackzone = solutions.TrackZone(show=False, region=None, model="best100.pt")
    last_interval = datetime.now(CET_TZ)
    tracker = Tracker()

    while process:
        try:
            raw_frame = process.stdout.read(1920 * 1080 * 3)
            if not raw_frame:
                logging.warning("No frame received. Restarting stream.")
                process.kill()
                process = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-re",
                        "-i",
                        STREAM_URL,
                        "-f",
                        "image2pipe",
                        "-pix_fmt",
                        "bgr24",
                        "-vcodec",
                        "rawvideo",
                        "-bufsize",
                        "512k",
                        "-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=10**8,
                )
                continue

            frame = np.frombuffer(raw_frame, np.uint8).reshape((1080, 1920, 3)).copy()
            timestamp = datetime.now(CET_TZ)
            region_points = [
                (0, ROI_LINE_1),
                (frame.shape[1], ROI_LINE_1),
                (frame.shape[1], ROI_LINE_2),
                (0, ROI_LINE_2),
            ]
            trackzone.region = np.array(region_points, dtype=np.int32)
            processed_frame = trackzone.trackzone(frame)

            tracked_objects = tracker.update(
                [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in trackzone.boxes]
            )

            for box, track_id, cls in zip(
                trackzone.boxes, trackzone.track_ids, trackzone.clss
            ):
                cls = int(cls)
                if cls not in CLASSES:
                    continue
                cls_name = CLASSES[cls]

                x, y, w, h = box
                cx, cy = (x + w) // 2, (y + h) // 2

                if (
                    track_id not in counted_ids
                    and CROSSING_LINE - 7 <= cy <= CROSSING_LINE + 7
                ):
                    counted_ids.add(track_id)
                    live_counts[cls_name] += 1
                    interval_counts[cls_name] += 1

            # Handle 15-minute intervals
            current_interval = timestamp - timedelta(
                minutes=timestamp.minute % 15,
                seconds=timestamp.second,
                microseconds=timestamp.microsecond,
            )
            
            if current_interval != last_interval:
                interval_counts.update({
                    "CET_date": current_interval.date(),
                    "CET_interval": (
                        f"{current_interval.strftime('%H:%M')} - "
                        f"{(current_interval + timedelta(minutes=15)).strftime('%H:%M')}"
                    )
                })
                update_supabase(interval_counts)
                interval_counts.update({"CYC": 0, "B": 0, "P": 0, "C": 0})
                last_interval = current_interval
                live_counts = {c: 0 for c in CLASSES.values()}  # Reset live counter

            # Visualization remains unchanged
            for i, (cls_name, count) in enumerate(live_counts.items()):
                cv2.putText(
                    processed_frame,
                    f"{cls_name}: {count}",
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )
            cv2.line(
                processed_frame,
                (0, CROSSING_LINE),
                (frame.shape[1], CROSSING_LINE),
                (0, 0, 255),
                3,
            )
            cv2.putText(
                processed_frame,
                "Crossing Line",
                (10, CROSSING_LINE - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            stframe.image(processed_frame, channels="RGB", use_container_width=True)

        except Exception as e:
            logging.error(f"Critical error: {e}")
            process.kill()
            process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-re",
                    "-i",
                    STREAM_URL,
                    "-f",
                    "image2pipe",
                    "-pix_fmt",
                    "bgr24",
                    "-vcodec",
                    "rawvideo",
                    "-bufsize",
                    "512k",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8,
            )

if __name__ == "__main__":
    main()