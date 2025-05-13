import os
import shutil
temp_dir = r"C:\Users\Jimmy\Documents\GitHub\NudeNet\Screenshots"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
print(f"Cleaned up {temp_dir}")

# DELETE FROM detections WHERE video_id IN (SELECT video_name FROM video_processing WHERE processed = 0);
# DELETE FROM video_processing WHERE processed = 0;