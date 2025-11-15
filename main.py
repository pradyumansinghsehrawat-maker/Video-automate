#!/usr/bin/env python3
"""
Production-grade FutureChannels automation engine (Stable Diffusion 1.5, macOS TTS)

Features:
- Config-driven channels & styles
- Scans channel script folders for new .txt scripts
- Splits script into scenes, generates prompts
- Generates images via SD1.5 (MPS if available)
- macOS 'say' TTS (free) to create audio per scene
- Ken-Burns panning zoom on each scene (duration matches audio)
- Auto-variant images for long scenes
- Skip already-rendered videos (idempotent)
- Logging to automation/logs/
- Safe error handling & retries
"""

import os
import sys
import json
import time
import math
import logging
import traceback
from pathlib import Path
from typing import List

# media libs
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from pydub import AudioSegment
from PIL import Image

# diffusers (SD)
try:
    from diffusers import StableDiffusionPipeline
    import torch
except Exception:
    StableDiffusionPipeline = None
    torch = None

# ------------------ CONFIG / PATHS ------------------
ROOT = Path(__file__).resolve().parents[1]  # repo root /Video-automate
AUTO = ROOT / "automation"                 # inner automation folder
CHANNELS_DIR = ROOT / "channels"
OUTPUT_ROOT = ROOT / "outputs"
LOG_DIR = AUTO / "logs"
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[
                        logging.FileHandler(LOG_DIR / "automation.log"),
                        logging.StreamHandler(sys.stdout),
                    ])

# ------------------ LOAD CONFIGS ------------------
with open(AUTO / "config.json", "r") as f:
    CONFIG = json.load(f)

with open(AUTO / "channel_styles.json", "r") as f:
    STYLES = json.load(f)

UNIVERSAL_NEG = CONFIG.get("universal_negative_prompt", "")

# SD device
DEVICE = "mps" if (torch is not None and torch.backends.mps.is_available()) else "cpu"

# ------------------ SD PIPELINE (lazy init) ------------------
SD_PIPE = None
def init_sd_pipeline():
    global SD_PIPE
    if SD_PIPE is not None:
        return SD_PIPE
    if StableDiffusionPipeline is None:
        logging.warning("diffusers not available — SD image generation disabled.")
        return None
    logging.info(f"Loading SD model {SD_MODEL_ID} on {DEVICE} (this may take a moment)...")
    try:
        dtype = torch.float16 if DEVICE == "mps" else torch.float32
        SD_PIPE = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=dtype)
        SD_PIPE.to(DEVICE)
        # optional: disable nsfw checker if used locally (careful)
        # SD_PIPE.safety_checker = lambda images, **kwargs: (images, False)
        logging.info("SD pipeline ready.")
        return SD_PIPE
    except Exception as e:
        logging.error("Failed to load SD pipeline: " + str(e))
        return None

# ------------------ UTILITIES ------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def write_text_file(p: Path, txt: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")

# ------------------ SCENE SPLITTING & PROMPTS ------------------
def split_into_scenes(script_text: str, max_words_per_scene=60) -> List[str]:
    # naive but robust: split on sentences and group to ~max_words_per_scene
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script_text) if s.strip()]
    scenes = []
    current = ""
    for s in sentences:
        if len((current + " " + s).split()) > max_words_per_scene:
            if current:
                scenes.append(current.strip())
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        scenes.append(current.strip())
    return scenes

def build_prompt(channel_key: str, scene_text: str) -> str:
    style = STYLES.get(channel_key, {})
    template = style.get("scene_template", "{description}. Show: {elements}.")
    master = style.get("master_style", "")
    neg = style.get("neg", "")
    prompt = template.format(description=scene_text, elements=scene_text) + ", " + master
    return prompt, neg

# ------------------ IMAGE GENERATION ------------------
def generate_image_for_prompt(prompt_txt_path: Path, out_image: Path, negative_prompt: str, retries=2):
    """
    Call SD pipeline using the prompt file path to allow easier caching & reproducibility.
    """
    if out_image.exists():
        logging.info(f"Image exists, skipping: {out_image}")
        return out_image
    sd = init_sd_pipeline()
    if sd is None:
        # fallback: placeholder gray image
        logging.warning("SD not available — creating placeholder image.")
        img = Image.new("RGB", (1280, 720), color=(40, 40, 40))
        d = Image.new("RGB", (1280, 720))
        img.save(out_image)
        return out_image

    prompt = prompt_txt_path.read_text(encoding="utf-8")
    attempt = 0
    while attempt <= retries:
        try:
            logging.info(f"SD generate: attempt {attempt+1} for {out_image.name}")
            out = sd(prompt, negative_prompt=negative_prompt, guidance_scale=7.5, num_inference_steps=30)
            image = out.images[0]
            image.save(out_image)
            return out_image
        except Exception as e:
            attempt += 1
            logging.error(f"SD generation failed (attempt {attempt}): {e}")
            logging.debug(traceback.format_exc())
            time.sleep(1 + attempt*2)
    # final fallback
    logging.error("SD generation failed all attempts — creating placeholder.")
    placeholder = Image.new("RGB", (1280,720), (50,50,50))
    placeholder.save(out_image)
    return out_image

# ------------------ TTS (macOS say) ------------------
def tts_say_to_wav(text: str, out_wav: Path):
    if out_wav.exists():
        logging.info(f"Audio exists, skipping TTS: {out_wav}")
        return out_wav
    # 'say' writes AIFF; we'll convert to wav with pydub
    temp_aiff = out_wav.with_suffix(".aiff")
    cmd = f'say -v {CONFIG["audio"]["voice"]} -r {int(CONFIG["audio"]["speed"]*200)} "{text.replace("\"","\'")}" -o "{temp_aiff}"'
    logging.debug("Running TTS: " + cmd)
    rc = os.system(cmd)
    if rc != 0:
        logging.error("macOS say failed (rc=%s)" % rc)
        # fallback: silent WAV 1s
        silent = AudioSegment.silent(duration=1000)
        silent.export(out_wav, format="wav")
        return out_wav
    # convert to WAV
    try:
        seg = AudioSegment.from_file(temp_aiff)
        seg.export(out_wav, format="wav")
        temp_aiff.unlink(missing_ok=True)
        return out_wav
    except Exception as e:
        logging.error("Failed convert aiff->wav: " + str(e))
        temp_aiff.unlink(missing_ok=True)
        silent = AudioSegment.silent(duration=1000)
        silent.export(out_wav, format="wav")
        return out_wav

# ------------------ KEN-BURNS & CLIP CREATION ------------------
def ken_burns_clip(image_path: Path, duration: float):
    # subtle zoom in/out implemented using moviepy transforms
    clip = ImageClip(str(image_path)).set_duration(duration)
    # subtle zoom: scale from 1.0 to zoom_end
    zoom_start = CONFIG["ken_burns"].get("zoom_start", 1.03)
    zoom_end = CONFIG["ken_burns"].get("zoom_end", 1.08)
    # use moviepy's resize with a function of t
    def resize_func(t):
        return zoom_start + (zoom_end-zoom_start)*(t/duration)
    clip = clip.resize(lambda t: resize_func(t))
    return clip

# ------------------ VARIANTS ------------------
def create_variants(image_path: Path, count: int) -> List[Path]:
    img = Image.open(image_path)
    w,h = img.size
    variants = []
    for i in range(count):
        # slight crop & save
        margin = 6 * (i+1)
        left = margin
        upper = margin
        right = max(w - margin, left + 10)
        lower = max(h - margin, upper + 10)
        crop = img.crop((left, upper, right, lower)).resize((w,h))
        outp = image_path.with_name(image_path.stem + f"_var{i+1}.png")
        crop.save(outp)
        variants.append(outp)
    return variants

# ------------------ ASSEMBLE VIDEO ------------------
def assemble_video(clips: List[ImageClip], audio_path: Path, out_video: Path):
    if out_video.exists():
        logging.info("Final video exists, skipping: %s", out_video)
        return out_video
    # ensure audio exists
    audio_clip = None
    if audio_path.exists():
        audio_clip = AudioFileClip(str(audio_path))
    if not clips:
        logging.error("No clips to assemble.")
        return None
    final = concatenate_videoclips(clips, method="compose")
    if audio_clip is not None:
        final = final.set_audio(audio_clip)
    safe_mkdir(out_video.parent)
    final.write_videofile(str(out_video), fps=CONFIG["video"].get("fps",30), codec="libx264", audio_codec="aac")
    return out_video

# ------------------ PROCESS SINGLE SCRIPT ------------------
def process_one_script(channel_folder: Path, script_path: Path, style_key: str):
    logging.info(f"Start processing script {script_path} for channel {style_key}")
    # output paths
    channel_name = channel_folder.name
    base_out = OUTPUT_ROOT / channel_name
    safe_mkdir(base_out)
    script_stem = script_path.stem
    final_video = base_out / f"{script_stem}_FINAL.mp4"
    if final_video.exists():
        logging.info("Final video already exists, skipping script: %s", final_video)
        return

    text = read_text_file(script_path)
    scenes = split_into_scenes(text, max_words_per_scene=CONFIG.get("split_max_words", 60))
    # force desired number of scenes if config requires (optional)
    desired_scenes = CONFIG.get("target_scenes", None)
    if desired_scenes:
        # pad or trim
        if len(scenes) < desired_scenes:
            while len(scenes) < desired_scenes:
                scenes.append(scenes[-1])
        else:
            scenes = scenes[:desired_scenes]

    # prepare working folders
    imgs_dir = channel_folder / "images"
    clips_dir = channel_folder / "clips"
    voice_dir = channel_folder / "voice"
    safe_mkdir(imgs_dir); safe_mkdir(clips_dir); safe_mkdir(voice_dir)

    clips = []
    audio_segments = []
    for i, scene_text in enumerate(scenes, start=1):
        logging.info(f"Scene {i}: {scene_text[:80]}...")
        prompt, neg = build_prompt(style_key, scene_text)
        prompt_file = channel_folder / "prompts" / f"{script_stem}_scene_{i}.txt"
        write_text_file(prompt_file, prompt)

        img_out = imgs_dir / f"{script_stem}_scene_{i}.png"
        generate_image_for_prompt(prompt_file, img_out, negative_prompt=UNIVERSAL_NEG + ", " + neg)

        # create tts audio per scene
        wav_out = voice_dir / f"{script_stem}_scene_{i}.wav"
        tts_say_to_wav(scene_text, wav_out)
        # determine duration
        try:
            audio = AudioSegment.from_wav(wav_out)
            dur_s = audio.duration_seconds
        except Exception:
            dur_s = max(8, len(scene_text)//8)
        # variants if long
        if dur_s > CONFIG["variants"].get("min_scene_length_for_variant", 12):
            variants = create_variants(img_out, CONFIG["variants"].get("variant_per_scene", 1))
            # first use main, then variants
            all_images = [img_out] + variants
        else:
            all_images = [img_out]

        # create clips for each image (duration split across images)
        per_img_dur = max(6, math.ceil(dur_s / len(all_images)))
        for imgp in all_images:
            clip = ken_burns_clip(imgp, per_img_dur)
            # attach audio only to the first clip of the scene
            if not clips or True:  # we will set audio on full concatenated track later
                clips.append(clip)
        audio_segments.append(wav_out)

    # Merge all audio segments into one narration track
    merged_audio = base_out / f"{script_stem}_narration.wav"
    combined = AudioSegment.silent(duration=0)
    for aw in audio_segments:
        try:
            combined += AudioSegment.from_wav(aw)
        except Exception:
            combined += AudioSegment.silent(duration=1000)
    combined.export(merged_audio, format="wav")

    # assemble final video, set merged_audio as audio
    final_video = base_out / f"{script_stem}_FINAL.mp4"
    assemble_video(clips, merged_audio, final_video)

    logging.info(f"Completed video: {final_video}")
    # Optionally move script to processed/ or delete
    processed_dir = channel_folder / "processed"
    safe_mkdir(processed_dir)
    script_path.rename(processed_dir / script_path.name)

# ------------------ AUTO SCAN & PROCESS ------------------
def auto_scan_and_process():
    # channel folder names expected to match the display names mapping
    for ch in (CHANNELS_DIR.iterdir()):
        if not ch.is_dir():
            continue
        # channel key needs to match keys in STYLES; mapping by simple normalization
        # try exact match first
        possible_keys = [k for k in STYLES.keys() if k.lower().replace(" ","_") in ch.name.lower().replace(" ","_") or ch.name.lower() in k.lower()]
        if not possible_keys:
            logging.warning(f"No style match found for channel folder '{ch.name}'. Skipping.")
            continue
        style_key = possible_keys[0]
        scripts_folder = ch / "scripts"
        if not scripts_folder.exists():
            logging.info(f"No scripts folder for channel {ch.name} — create {scripts_folder} and drop .txt files")
            continue
        txts = sorted(scripts_folder.glob("*.txt"))
        if not txts:
            logging.info(f"No scripts to process for {ch.name}")
            continue
        for s in txts:
            try:
                process_one_script(ch, s, style_key)
            except Exception as e:
                logging.error(f"Unhandled error processing {s}: {e}")
                logging.debug(traceback.format_exc())

# ------------------ ENTRYPOINT ------------------
def main():
    logging.info("Starting FutureChannels automation main loop")
    init_sd_pipeline()
    auto_scan_and_process()
    logging.info("Automation run complete")

if __name__ == "__main__":
    main()
