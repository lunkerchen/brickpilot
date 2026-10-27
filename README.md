# BrickPilot (Stage I prototype)

**Demo video (2.5 min):** https://youtu.be/iHUNZwdZVLE

Real-time build guidance for LEGO®-compatible interlocking bricks. A top-down camera watches a baseplate, tracks which instruction step the builder is on, highlights where the next brick goes, and flags a wrong brick the moment it is placed.

Entry for the ASUS UGen AI League hackathon, Battlefield Lightning (ASUS UGen300 / Hailo-10H). LEGO® is a trademark of the LEGO Group, which does not sponsor or endorse this project.

## How it works

1. **Rectify**: the four baseplate corners give a perspective transform to a top-down stud grid.
2. **Observe**: each stud cell is classified by its median color in Lab space. Cells far from every palette color (hands, shadows) are marked unknown and ignored.
3. **Locate**: the instruction file is expanded into the expected top view after every step. The tracker picks the step with the fewest mismatches; ties (caused by occlusion) stay at the current step.
4. **Guide**: a green box shows the next brick and red cells show a wrong brick, projected back onto the camera image. Optional speech on macOS.

A frame is only accepted after it is stable for 5 frames, so a moving hand never triggers a false step.

## Run

```bash
uv venv .venv && uv pip install --python .venv/bin/python opencv-python numpy
.venv/bin/python test_brickpilot.py                       # self-check on synthetic frames
.venv/bin/python brickpilot.py demo --out demo.mp4        # synthetic demo video
.venv/bin/python brickpilot.py run --source 0 --voice     # webcam: click corners TL, TR, BR, BL
```

`--max-dist` sets how far a color can be from the palette before it counts as unknown. Tune it for your lighting. Press `r` to recalibrate and `q` to quit.

To match your camera's colors, photograph a finished build and measure the palette, then pass it with `--palette`:

```bash
.venv/bin/python brickpilot.py measure --source done.png --step 14 --out palette.json
```

## Simulated footage (Blender)

`sim/render_scene.py` renders a photorealistic build of `house.json` with an overhead camera, a loose brick pile, a hand that picks and places each brick, and one deliberate mistake. It also writes the baseplate corners to `sim/calib.json`.

The hand is the realistic hand from Blender Studio's [Human Base Meshes](https://www.blender.org/download/demo-files/#assets) bundle (CC0). The script rigs it automatically and poses it as a thumb-index pinch. Download and unzip the bundle first:

```bash
curl -fLO https://mirror.blender.org/demo/asset-bundles/human-base-meshes/human-base-meshes-bundle-v1.4.1.zip && unzip human-base-meshes-bundle-v1.4.1.zip
blender -b -P sim/render_scene.py -- --model house.json --hand human-base-meshes-bundle-v1.4.1/human_base_meshes_bundle.blend --out sim/frames
.venv/bin/python brickpilot.py measure --source sim/frames/1025.png --step 14 --calib sim/calib.json --out sim/palette.json
.venv/bin/python brickpilot.py run --source sim/frames/%04d.png --calib sim/calib.json --palette sim/palette.json --out sim/annotated.mp4
```

The setup assumes an overhead camera and a top light (ring light), which keeps shadows and stacking parallax small. Skin is ignored by its a* (redness) range, so a hand over the board is never read as a white or yellow brick.

## Demo video

`video/make_demo.py` cuts the annotated footage, title cards and an English voiceover (macOS `say`) into a demo of about 2.5 minutes, with burned-in captions and a separate `.srt` for YouTube.

```bash
python3 video/make_demo.py --clip sim/annotated.mp4 --out video/brickpilot_demo.mp4
```

## Instruction format

`house.json` lists the grid size, the palette (BGR) and the steps. Each step is a brick color and its `[x, y, width, height]` in studs. Every step must change the top view.

## Limits of this prototype and Stage II plan

| Stage I (this repo) | Stage II on UGen300 |
| --- | --- |
| Color classification per stud with OpenCV | YOLO brick detection compiled to Hailo HEF |
| Top view only; a same-color brick stacked on itself is invisible | Second side camera for height |
| Hand-written JSON instructions | Import LDraw `.ldr` / `.mpd` models |
| Guidance on screen | Mini projector highlights the next brick in the loose pile |
| Synthetic test footage | Real builds with children, seniors and classrooms |

On a laptop CPU, one observation takes about 6 ms on a 16x16 baseplate.
