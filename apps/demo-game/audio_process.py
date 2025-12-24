from pydub import AudioSegment
import librosa
import soundfile as sf
import numpy as np
import os

# ========= 参数 =========
INPUT_MP3 = "sample.mp3"   # ← 改成你的 MP3 路径
OUTPUT_WAV = "background_music_0.5x.wav"

START_TIME = 43           # 秒
END_TIME = 114            # 1分54秒
SPEED = 0.5               # 0.5x 变慢

# ========= Step 1: MP3 → WAV 并裁剪 =========
audio = AudioSegment.from_mp3(INPUT_MP3)
segment = audio[START_TIME * 1000 : END_TIME * 1000]
segment.export("trimmed.wav", format="wav")

# ========= Step 2: 高质量变速（不变调） =========
y, sr = librosa.load("trimmed.wav", sr=None, mono=True)

# time-stretch (rate < 1 → slower)
y_slow = librosa.effects.time_stretch(y, rate=SPEED)

# ========= Step 3: 保存结果 =========
sf.write(OUTPUT_WAV, y_slow, sr)

print(f"Done! Output saved as {OUTPUT_WAV}")
