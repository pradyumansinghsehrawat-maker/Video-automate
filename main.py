#!/usr/bin/env python3
"""
FutureChannels Automation System
---------------------------------
Runs 8 channel pipelines automatically:
- Alternate Pasts
- Med Insiders
- Future Mechanisms
- EconDepth Insiders
- Stock Basics Insiders
- Axela247
- Parallel Hidden Patterns
- Nomad Insiders

Generates:
• Scenes
• Prompts
• AI Images (Stable Diffusion)
• Ken-Burns cinematic videos
• Auto Variants for long narration
• macOS TTS audio
• Final assembled video
"""

import os
import json
import time
from moviepy.editor import *
from PIL import Image
from pathlib import Path

# -----------------------------------------------------------
# LOAD CHANNEL LIST
# -----------------------------------------------------------
with open("channels/channels.json", "r") as f:
    CHANNELS = json.load(f)["channels"]

# -----------------------------------------------------------
# LOAD GLOBAL CONFIG
# -----------------------------------------------------------
with open("automation/config.json", "r") as f:
    CONFIG = json.load(f)

NEG = CONFIG["universal_negative_prompt"]


# -----------------------------------------------------------
# TEXT-TO-SPEECH (macOS built-in — FREE)
# -----------------------------------------------------------
def generate_tts(text, out_file):
    os.system(f'say -v {CONFIG["audio"]["voice"]} -r {int(CONFIG["audio"]["speed"]*200)} "{text}" -o "{out_file}"')
    return out_file


# -----------------------------------------------------------
# MAKE OUTPUT DIRECTORIES FOR EACH CHANNEL
# -----------------------------------------------------------
def ensure_dirs(channel):
    base = f"channels/{channel}"
    dirs = ["scripts", "scenes", "prompts", "images", "clips", "voice", "final"]
    for d in dirs:
        Path(f"{base}/{d}").mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# READ SCRIPT
# -----------------------------------------------------------
def read_script(script_path):
    with open(script_path, "r") as f:
        return f.read()


# -----------------------------------------------------------
# SPLIT SCRIPT INTO SCENES (14 scenes fixed)
# -----------------------------------------------------------
def split_into_scenes(full_script):
    lines = full_script.split(".")
    chunks = []
    current = ""
    for line in lines:
        if len(current.split()) > 40:
            chunks.append(current.strip())
            current = ""
        current += line + ". "
    if current.strip():
        chunks.append(current.strip())

    # force 14 scenes by slicing or padding
    while len(chunks) < 14:
        chunks.append(chunks[-1])
    return chunks[:14]


# -----------------------------------------------------------
# WRITE PROMPTS FROM SCENES
# -----------------------------------------------------------
def write_prompts(channel, scenes):
    base = f"channels/{channel}/prompts"
    paths = []

    for i, s in enumerate(scenes):
        prompt = f"{channel} documentary cinematic style. Scene: {s} --negative {NEG}"
        fp = f"{base}/scene_{i+1}.txt"
        with open(fp, "w") as f:
            f.write(prompt)
        paths.append(fp)

    return paths


# -----------------------------------------------------------
# GENERATE IMAGE USING STABLE DIFFUSION (LOCAL)
# -----------------------------------------------------------
def generate_image(prompt_path, out_path):
    # You will run this using Diffusers + a local SDXL model
    # Here we just put a placeholder, real code runs via your Mac's model
    os.system(f'python3 automation/sd_generate.py "{prompt_path}" "{out_path}"')
    return out_path


# -----------------------------------------------------------
# KEN BURNS CLIP CREATION
# -----------------------------------------------------------
def ken_burns(image_path, duration):
    img = ImageClip(image_path)

    zoom_start = CONFIG["ken_burns"]["zoom_start"]
    zoom_end = CONFIG["ken_burns"]["zoom_end"]

    clip = img.fx(vfx.zoom_in, zoom=zoom_start).fx(vfx.zoom_out, zoom=zoom_end)
    return clip.set_duration(duration)


# -----------------------------------------------------------
# VARIANTS (duplicate image → slight crop/pan)
# -----------------------------------------------------------
def make_variants(img_path, count):
    variants = []
    img = Image.open(img_path)

    for i in range(count):
        w, h = img.size
        crop = img.crop((20*i, 20*i, w - 20*i, h - 20*i))
        new_path = img_path.replace(".png", f"_v{i+1}.png")
        crop.save(new_path)
        variants.append(new_path)

    return variants


# -----------------------------------------------------------
# GENERATE FULL VIDEO FOR ONE SCRIPT
# -----------------------------------------------------------
def process_script(channel, script_path):
    print(f"\n=== Processing {channel} / {os.path.basename(script_path)} ===")

    ensure_dirs(channel)

    text = read_script(script_path)
    scenes = split_into_scenes(text)
    prompts = write_prompts(channel, scenes)

    base = f"channels/{channel}"
    image_dir = f"{base}/images"
    clip_dir = f"{base}/clips"
    voice_dir = f"{base}/voice"

    Path(image_dir).mkdir(exist_ok=True)
    Path(clip_dir).mkdir(exist_ok=True)
    Path(voice_dir).mkdir(exist_ok=True)

    clips = []

    for i, (scene, prompt) in enumerate(zip(scenes, prompts), 1):
        print(f" → Scene {i}")

        # TTS
        voice_path = f"{voice_dir}/scene_{i}.aiff"
        generate_tts(scene, voice_path)
        audio = AudioFileClip(voice_path)
        duration = audio.duration

        # Image
        img_path = f"{image_dir}/scene_{i}.png"
        generate_image(prompt, img_path)

        # Variants
        if i <= CONFIG["variants"]["highlight_scenes"]:
            v = make_variants(img_path, CONFIG["variants"]["variant_per_scene"])
            img_path = v[-1]

        # Ken Burns
        clip = ken_burns(img_path, duration)
        clip = clip.set_audio(audio)
        clip_path = f"{clip_dir}/scene_{i}.mp4"
        clip.write_videofile(clip_path, fps=CONFIG["video"]["fps"])
        clips.append(clip)

    final = concatenate_videoclips(clips)
    out = f"{base}/final/{Path(script_path).stem}_FINAL.mp4"
    final.write_videofile(out, fps=CONFIG["video"]["fps"])

    print("DONE:", out)


# -----------------------------------------------------------
# MAIN LOOP – PROCESS ALL CHANNELS SCRIPTS
# -----------------------------------------------------------
def main():
    print("\nStarting full 8-channel automation...\n")

    for channel in CHANNELS:
        ensure_dirs(channel)
        script_folder = f"channels/{channel}/scripts"

        for script in os.listdir(script_folder):
            if script.endswith(".txt"):
                process_script(channel, f"{script_folder}/{script}")

    print("\n🎉 ALL VIDEOS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
 
