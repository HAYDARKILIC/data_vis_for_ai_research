"""Generate or fetch a small set of ImageNet-style sample images for the week-6 lab.

To keep the lab fully offline-runnable, we synthesise four simple but believable
"animal" images from procedural patterns at 224×224. They are not realistic; the point
is that we can demonstrate Grad-CAM and attention-rollout pipelines end-to-end without
a network download.

If you have ImageNet images of your own, drop them in this folder as
{cat,dog,bird,car}.jpg and the lab will pick them up automatically.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).parent


def synth_cat():
    img = Image.new("RGB", (224, 224), (210, 200, 180))
    d = ImageDraw.Draw(img)
    # body
    d.ellipse((40, 90, 200, 200), fill=(150, 130, 110))
    # head
    d.ellipse((70, 50, 150, 130), fill=(170, 145, 120))
    # ears
    d.polygon([(80, 70), (95, 35), (105, 70)], fill=(130, 110, 90))
    d.polygon([(125, 70), (135, 35), (145, 70)], fill=(130, 110, 90))
    # eyes
    d.ellipse((90, 80, 100, 90), fill=(20, 20, 20))
    d.ellipse((120, 80, 130, 90), fill=(20, 20, 20))
    # nose / whiskers omitted on purpose
    return img


def synth_dog():
    img = Image.new("RGB", (224, 224), (200, 210, 220))
    d = ImageDraw.Draw(img)
    d.ellipse((40, 80, 200, 210), fill=(180, 140, 100))
    d.ellipse((65, 45, 160, 140), fill=(200, 160, 120))
    d.ellipse((50, 60, 90, 130), fill=(160, 120, 80))   # ear
    d.ellipse((140, 60, 180, 130), fill=(160, 120, 80))
    d.ellipse((90, 90, 102, 102), fill=(20, 20, 20))
    d.ellipse((125, 90, 137, 102), fill=(20, 20, 20))
    d.ellipse((105, 115, 130, 135), fill=(40, 40, 40))   # nose
    return img


def synth_bird():
    img = Image.new("RGB", (224, 224), (150, 200, 220))
    d = ImageDraw.Draw(img)
    # body
    d.ellipse((70, 90, 180, 180), fill=(80, 120, 60))
    # head
    d.ellipse((140, 60, 200, 120), fill=(80, 120, 60))
    # beak
    d.polygon([(195, 85), (220, 92), (195, 100)], fill=(230, 160, 50))
    # eye
    d.ellipse((175, 78, 185, 88), fill=(20, 20, 20))
    # wing detail
    d.ellipse((90, 110, 160, 160), fill=(60, 100, 50))
    return img


def synth_car():
    img = Image.new("RGB", (224, 224), (200, 200, 200))
    d = ImageDraw.Draw(img)
    # body
    d.rectangle((30, 110, 200, 165), fill=(180, 50, 50))
    # cabin
    d.polygon([(70, 110), (90, 70), (160, 70), (180, 110)], fill=(180, 50, 50))
    # windows
    d.polygon([(80, 105), (95, 78), (125, 78), (125, 105)], fill=(160, 200, 230))
    d.polygon([(130, 78), (155, 78), (170, 105), (130, 105)], fill=(160, 200, 230))
    # wheels
    d.ellipse((55, 150, 90, 185), fill=(30, 30, 30))
    d.ellipse((140, 150, 175, 185), fill=(30, 30, 30))
    d.ellipse((63, 158, 82, 177), fill=(120, 120, 120))
    d.ellipse((148, 158, 167, 177), fill=(120, 120, 120))
    return img


GENERATORS = {"cat": synth_cat, "dog": synth_dog, "bird": synth_bird, "car": synth_car}

for name, gen in GENERATORS.items():
    p = HERE / f"{name}.jpg"
    if p.exists():
        print(f"  {p.name} already exists, skipping")
        continue
    gen().save(p, "JPEG", quality=92)
    print(f"  wrote {p.name}")
