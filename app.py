import logging
import time
from ultralytics import solutions
from tracker import Tracker
import streamlit as st
import cv2
import numpy as np
import subprocess
from datetime import datetime, timedelta
from supabase import create_client, Client
from zoneinfo import ZoneInfo

# Configuration
TARGET_FPS = 8  # Reduced from default 25-30
SKIP_FRAMES = 1  # Process every other frame
RESIZE_FACTOR = 0.75  # Reduce resolution if needed
BUFFER_SIZE = 1920 * 1080 * 3  # Frame buffer size

st.set_page_config(
    page_title = 'Live Detection',
    page_icon = "🔴",
)

st.title("Altnova Live Detection")

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Supabase setup
SUPABASE_URL = "https://fktheptskxttqdcksztd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZrdGhlcHRza3h0dHFkY2tzenRkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI5MjcwOTYsImV4cCI6MjA1ODUwMzA5Nn0.NiWRw28MXA0h6_5a8SboKYy3SBUmZlKFJr1II7SvZZI"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Timezone setup
CET_TZ = ZoneInfo("Europe/Brussels")

CLASSES = {
    0: "CYC", 1: "B", 2: "B", 3: "P", 4: "C"
}

# ROI configuration
ROI_LINE_1, ROI_LINE_2, CROSSING_LINE = 250, 800, 540
STREAM_URL = "https://livecam.brucity.be/LiveBrusselsWebcams/streams/fDdnnEmqOn6Kyy3E1701416388577.m3u8"

def update_supabase(interval_counts):
    """Update interval_tracking table with throttled writes"""
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
    stframe = st.empty()
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-re",
            "-i", STREAM_URL,
            "-r", str(TARGET_FPS),  # Request source FPS
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-bufsize", "512k",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=BUFFER_SIZE,
    )

    # Tracking initialization
    live_counts = {c: 0 for c in CLASSES.values()}
    interval_counts = {"CET_date": None, "CET_interval": None, "CYC": 0, "B": 0, "P": 0, "C": 0}
    counted_ids = set()
    trackzone = solutions.TrackZone(show=False, region=None, model="best100.pt")
    last_interval = datetime.now(CET_TZ)
    tracker = Tracker()
    
    # FPS control
    last_frame_time = time.time()
    frame_count = 0

    while process:
        try:
            # Throttle frame processing
            current_time = time.time()
            if (current_time - last_frame_time) < (1 / TARGET_FPS):
                continue
            last_frame_time = current_time

            # Skip frames to reduce processing load
            for _ in range(SKIP_FRAMES):
                process.stdout.read(BUFFER_SIZE)
                
            # Read and process frame
            raw_frame = process.stdout.read(BUFFER_SIZE)
            if not raw_frame:
                logging.warning("No frame received. Restarting stream.")
                process.kill()
                process = subprocess.Popen([...])  # Keep previous restart logic
                continue

            # Decode and resize frame
            frame = np.frombuffer(raw_frame, np.uint8).reshape((1080, 1920, 3))
            if RESIZE_FACTOR != 1:
                frame = cv2.resize(frame, (0,0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR)

            # Object tracking (rest of your processing logic)
            timestamp = datetime.now(CET_TZ)
            region_points = [(0, ROI_LINE_1), (frame.shape[1], ROI_LINE_1),
                            (frame.shape[1], ROI_LINE_2), (0, ROI_LINE_2)]
            trackzone.region = np.array(region_points, dtype=np.int32)
            processed_frame = trackzone.trackzone(frame)

            # ... (keep your existing tracking logic here) ...

            # Update display
            stframe.image(processed_frame, channels="RGB", use_container_width=True)

            # Interval handling (keep existing logic)
            # ... (your interval update code) ...

        except Exception as e:
            logging.error(f"Critical error: {e}")
            process.kill()
            process = subprocess.Popen([...])  # Keep previous restart logic

if __name__ == "__main__":
    main()