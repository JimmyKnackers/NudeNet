import os
for root, _, files in os.walk(r"E:\Prol"):
    for f in files:
        if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm', '.mpg', '.mpeg', '.wmv')):
            print(os.path.relpath(os.path.join(root, f), r"E:\Prol"))