import os
import shutil
for dir_path in [r"C:\Users\Jimmy\Documents\GitHub\NudeNet\Screenshots", r"C:\Users\Jimmy\Documents\GitHub\NudeNet\detected_frames"]:
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)
print("Cleaned up temporary directories")

# DELETE FROM detections WHERE video_id IN (SELECT video_name FROM video_processing WHERE processed = 0);
# DELETE FROM video_processing WHERE processed = 0;