import subprocess
import os

def create_srt(captions_data, srt_path):
    def ms_to_srt_time(ms):
        total_sec = ms / 1000
        h = int(total_sec // 3600)
        m = int((total_sec % 3600) // 60)
        s = int(total_sec % 60)
        ms_remainder = int((total_sec - int(total_sec)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms_remainder:03}"
    with open(srt_path, "w", encoding="utf-8") as f:
        for c in captions_data:
            f.write(f"{c['index']}\n")
            f.write(f"{ms_to_srt_time(c['start'])} --> {ms_to_srt_time(c['end'])}\n")
            f.write(f"{c['text']}\n\n")

def burn_subtitles(video_path, captions_data, output_path):
    srt_file = "temp_captions.srt"
    create_srt(captions_data, srt_file)
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"subtitles={srt_file}:force_style='FontName=Arial,FontSize=22,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,Outline=2,Shadow=1,Alignment=2'",
        "-c:a", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True)
    if os.path.exists(srt_file):
        os.remove(srt_file)
