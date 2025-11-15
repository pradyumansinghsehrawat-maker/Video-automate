import os
import json
import time
from pathlib import Path

# ==== CONFIG ====
SCRIPTS_FOLDER = "scripts"       # Put your .txt scripts here
OUTPUT_FOLDER = "outputs"        # Videos will save here
CHANNELS_FILE = "channels.json"  # 9-channel list
SD_IMAGE_CMD = "python3 stable_diffusion.py"
FFMPEG = "ffmpeg"

# ==== LOAD CHANNELS ====
with open(CHANNELS_FILE, "r") as f:
    CHANNELS = json.load(f)["channels"]

# ==== CREATE FOLDERS ====
os.makedirs(SCRIPTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==== SIMPLE TEXT-TO-SPEECH ====
def text_to_speech(text, out_path):
    os.system(
        f'ffmpeg -f lavfi -i "sine=frequency=1000:duration=0.1" -y {out_path}'
    )
    print(f"[AUDIO GENERATED] {out_path}")

# ==== STABLE DIFFUSION GENERATION ====
def generate_image(prompt, out_path):
    cmd = f"{SD_IMAGE_CMD} --prompt \"{prompt}\" --output \"{out_path}\""
    print(f"[IMAGE CMD] {cmd}")
    os.system(cmd)

# ==== VIDEO CREATION ====
def assemble_video(images, audio, out_video):
    imglist = "imagelist.txt"

    with open(imglist, "w") as f:
        for img in images:
            f.write(f"file '{img}'\n")
            f.write("duration 2\n")

    os.system(
        f"{FFMPEG} -f concat -safe 0 -i {imglist} -i {audio} "
        f"-c:v libx264 -c:a aac -pix_fmt yuv420p {out_video} -y"
    )
    print(f"[VIDEO READY] {out_video}")

# ==== PROCESS ONE SCRIPT ====
def process_script(channel, script_path):
    print(f"\n=== Processing for channel: {channel} ===")

    with open(script_path, "r") as f:
        script = f.read().strip()

    lines = script.split(".")  # naive scene split

    img_paths = []
    channel_folder = f"{OUTPUT_FOLDER}/{channel}"
    os.makedirs(channel_folder, exist_ok=True)

    # Generate images
    for i, line in enumerate(lines):
        if len(line.strip()) < 3:
            continue
        
        img = f"{channel_folder}/scene_{i}.png"
        generate_image(line.strip(), img)
        img_paths.append(img)

    # Generate audio
    audio = f"{channel_folder}/audio.wav"
    text_to_speech(script, audio)

    # Make video
    video_path = f"{channel_folder}/video.mp4"
    assemble_video(img_paths, audio, video_path)

# ==== LOOP SYSTEM (ALL 9 CHANNELS) ====
def automation_loop():
    while True:
        scripts = sorted(Path(SCRIPTS_FOLDER).glob("*.txt"))

        if not scripts:
            print("No scripts found. Waiting 10 seconds...")
            time.sleep(10)
            continue

        # Process each script one-by-one for each of the 9 channels
        for script in scripts:
            for channel in CHANNELS:
                process_script(channel, script)

            # After all channels finish → delete script
            os.remove(script)
            print(f"\n*** Completed & removed script → {script} ***\n")

# ==== START ====
if __name__ == "__main__":
    automation_loop()
