# BrickPilot (Stage I prototype)

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
