"""把模擬畫面、字卡與英文旁白剪成約 3 分鐘的比賽 demo 影片（macOS say + ffmpeg）。

python3 video/make_demo.py --clip sim/annotated.mp4 --out video/brickpilot_demo.mp4 [--voice en-US-AndrewNeural | --tts say --voice Samantha]
旁白預設用 edge-tts 神經網路語音；每段的長度跟著旁白走；影片段落會加速或停格去配合旁白長度。
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

FPS = 25
SIZE = "1280x720"
FONT = "/System/Library/Fonts/Avenir Next.ttc"
WRONG_FRAME = 242    # 模擬畫面裡追蹤器亮紅框的那一格（brickpilot.py run 的輸出）

SEGMENTS = [
    {"card": ["BrickPilot", "Real-time brick-building guidance on the edge", "ASUS x Intel x Hailo  |  Lightning track  |  Everyday AI"],
     "say": "This is BrickPilot. It turns any desk into a guided brick-building station, designed to run fully on the ASUS UGen300 edge AI accelerator."},
    {"card": ["The problem", "Paper instructions are 2D pictures of a 3D build",
              "One wrong brick is often found ten steps later", "One teacher cannot watch fifteen desks at once"],
     "say": "Paper instructions are flat pictures of a 3D build. A child places one brick wrong, notices it ten steps later, "
            "and has to tear the model apart. Seniors use bricks for cognitive training, but struggle with tiny diagrams. "
            "And in a brick classroom, one teacher cannot see which of fifteen students is stuck."},
    {"card": ["The idea", "A camera looks down at the baseplate", "Every frame is compared with the instructions",
              "Next brick highlighted, mistakes flagged instantly"],
     "say": "BrickPilot puts a camera above the baseplate. On every frame it compares the board with the instructions, "
            "highlights where the next brick goes, and flags a mistake the moment it happens. No app to tap through, and no cloud."},
    {"clip": (0, 182),
     "say": "Here is our Stage One prototype. This footage is rendered in Blender and labelled as simulated. "
            "The overlay is the real output of our tracking program. The green box shows where the next brick goes. "
            "When the hand places it, the step advances by itself."},
    {"clip": (182, 300), "sfx": (WRONG_FRAME, "Wrong brick. Remove it."),
     "say": "Now a deliberate mistake: a blue brick in the wrong place. A red box appears at once, with a spoken prompt. "
            "Take it away, and the guidance continues."},
    {"clip": (300, 1026),
     "say": "The hand often covers the board. BrickPilot recognises skin and ignores those cells, so it never skips ahead "
            "and never raises a false alarm. Across all 1,026 frames, the only alert was the deliberate one, "
            "and the build ends complete."},
    {"card": ["How it works", "1. Rectify the baseplate into a stud grid", "2. Read the colour of every cell, mask the hand",
              "3. Match the board against every step", "About 6 ms per frame on a laptop CPU"],
     "say": "Each frame goes through three steps. Rectify the baseplate into a stud grid. Read the colour of every cell, and mask out the hand. "
            "Then match the board against every step of the instructions. On a laptop CPU this takes about six milliseconds per frame."},
    {"card": ["Stage Two on the UGen300", "YOLO part detection in the loose pile", "On-device voice Q&A for children",
              "Side camera for tall builds"],
     "say": "In Stage Two we move to the UGen300. A YOLO model on the Hailo-10H points to the next part in the loose pile, "
            "and a small on-device language model answers a child's questions by voice."},
    {"card": ["Why the edge", "Children's hands on camera: video stays on the desk", "Feedback in under 0.3 seconds",
              "Twenty desks, no cloud streaming"],
     "say": "Edge AI is essential here. The camera sees children's hands, so video must never leave the desk. "
            "Feedback has to arrive in under a third of a second. And a classroom of twenty desks cannot stream video to the cloud."},
    {"card": ["Who pays", "Brick and STEM classrooms: kit per desk + teacher dashboard",
              "Senior day-care centres: cognitive training", "Toy retail: in-store demo stations"],
     "say": "We start with brick and STEM classrooms: one kit per desk, plus a teacher dashboard by subscription. "
            "Next come senior day-care centres and in-store demo stations."},
    {"card": ["BrickPilot", "See the next brick. Catch every mistake.", "github.com/lunkerchen/brickpilot"],
     "say": "BrickPilot. See the next brick, and catch every mistake. The code is on GitHub."},
]


# 字幕用正式寫法；念的時候換成 say 念得對的拼法
SPOKEN = {"UGen300": "U Gen 300", "Hailo-10H": "Hailo ten H", "3D": "three dimensional", "1,026": "one thousand and twenty six",
          "day-care": "day care", "YOLO": "yolo"}


def spoken(text):
    for k, v in SPOKEN.items():
        text = text.replace(k, v)
    return text


def sh(*cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out)


def narrate(text, path, args):
    """預設用 edge-tts 的神經網路語音（需連網）；--tts say 改用 macOS 內建語音（離線）。"""
    if args.tts == "say":
        aiff = path.with_suffix(".aiff")
        sh("say", "-v", args.voice, "-r", str(args.rate), "-o", str(aiff), text)
        sh("ffmpeg", "-y", "-i", str(aiff), str(path))
        return
    for attempt in range(3):
        try:
            subprocess.run(["uvx", "--from", "edge-tts", "edge-tts", "--voice", args.voice, "--rate", args.edge_rate, "--text", text, "--write-media", str(path)],
                           check=True, capture_output=True, timeout=120)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 2:
                raise


def text_filters(lines, tmp, i):
    """第一行是標題，其餘是條列；每行寫成檔案再給 drawtext，避免跳脫字元問題。"""
    fs = []
    y = 200 if len(lines) > 3 else 250
    for k, line in enumerate(lines):
        f = tmp / f"t{i}_{k}.txt"
        f.write_text(line)
        size, color = (64, "white") if k == 0 else (36, "0xd0d6dc")
        fs.append(f"drawtext=fontfile='{FONT}':textfile='{f}':fontsize={size}:fontcolor={color}:x=120:y={y}")
        y += 110 if k == 0 else 62
    return ",".join(fs + ["drawbox=x=120:y=170:w=90:h=6:color=0x2ecc40:t=fill"])


def build_segment(i, seg, args, tmp):
    voice = tmp / f"v{i}.mp3"
    narrate(spoken(seg["say"]), voice, args)
    length = duration(voice) + 0.8
    out = tmp / f"s{i}.mp4"
    audio = ["-i", str(voice)]
    mix = "[1:a]apad,atrim=0:{L}[a]".format(L=length)
    if "card" in seg:
        video = ["-f", "lavfi", "-i", f"color=c=0x101418:s={SIZE}:r={FPS}:d={length}"]
        vf = text_filters(seg["card"], tmp, i) + f",fade=in:st=0:d=0.3,fade=out:st={length - 0.3}:d=0.3"
    else:
        f0, f1 = seg["clip"]
        natural = (f1 - f0) / FPS
        video = ["-i", args.clip]
        # 影片比旁白長就加速；比旁白短就照原速播完再停格
        scale = min(1.0, length / natural)
        vf = (f"trim=start_frame={f0}:end_frame={f1},setpts=(PTS-STARTPTS)*{scale},fps={FPS},"
              f"tpad=stop_mode=clone:stop_duration={length},trim=duration={length}")
        if "sfx" in seg:   # 追蹤器的語音提示（brickpilot.py --voice 用的同一個 say）
            frame, words = seg["sfx"]
            sfx = tmp / f"x{i}.aiff"
            sh("say", "-v", args.sfx_voice, "-o", str(sfx), words)
            delay = int((frame - f0) / FPS * scale * 1000)
            audio += ["-i", str(sfx)]
            mix = (f"[1:a]apad,atrim=0:{length},volume=0.9[n];[2:a]adelay={delay}|{delay},apad,atrim=0:{length}[s];"
                   f"[n][s]amix=inputs=2:normalize=0[a]")
    sh("ffmpeg", "-y", *video, *audio, "-filter_complex", f"[0:v]{vf}[v];{mix}", "-map", "[v]", "-map", "[a]",
       "-t", str(length), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-ar", "48000", "-ac", "2", str(out))
    return out, length


def srt_time(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def captions(segments, lengths):
    """每句一段字幕，時間依字數比例分配到該段旁白。"""
    rows, t0 = [], 0.0
    for seg, length in zip(segments, lengths):
        sentences = [s.strip() + "." for s in seg["say"].replace("?", ".").split(".") if s.strip()]
        total = sum(len(s) for s in sentences)
        t = t0 + 0.2
        for s in sentences:
            d = (length - 0.8) * len(s) / total
            rows.append((t, t + d, s))
            t += d
        t0 += length
    return "\n".join(f"{k + 1}\n{srt_time(a)} --> {srt_time(b)}\n{s}\n" for k, (a, b, s) in enumerate(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="sim/annotated.mp4")
    ap.add_argument("--out", default="video/brickpilot_demo.mp4")
    ap.add_argument("--tts", choices=["edge", "say"], default="edge")
    ap.add_argument("--voice", default="en-US-AndrewNeural", help="edge-tts voice, or a macOS voice with --tts say")
    ap.add_argument("--edge-rate", default="+0%")
    ap.add_argument("--sfx-voice", default="Daniel")
    ap.add_argument("--rate", type=int, default=172, help="words per minute")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        parts = [build_segment(i, s, args, tmp) for i, s in enumerate(SEGMENTS)]
        (tmp / "list.txt").write_text("".join(f"file '{p}'\n" for p, _ in parts))
        srt = Path(args.out).with_suffix(".srt")
        srt.write_text(captions(SEGMENTS, [l for _, l in parts]))
        style = "FontName=Avenir Next,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,BorderStyle=3,Outline=1,MarginV=24"
        sh("ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
           "-vf", f"subtitles='{srt}':force_style='{style}'", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
           "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-ar", "48000", "-movflags", "+faststart", args.out)
    print(json.dumps({"out": args.out, "seconds": round(duration(args.out), 1), "captions": str(srt)}))


if __name__ == "__main__":
    main()
