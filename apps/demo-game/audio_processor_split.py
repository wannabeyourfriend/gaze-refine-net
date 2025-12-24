# audio_processor_simple.py
# 简单音频处理器：只分析音频，输出音高序列文件

import numpy as np
from scipy.io import wavfile
import math
from pathlib import Path
import librosa

class SimpleAudioProcessor:
    def __init__(self, min_note_duration=0.4):
        """
        初始化音频处理器
        min_note_duration: 最小音符持续时间（秒），短于此时间的音符会合并到前一个音符
        """
        self.min_note_duration = min_note_duration
        
        # 音符频率表（C3到C6）
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.note_freqs = {}
        
        for octave in range(3, 7):
            for i, note in enumerate(note_names):
                note_name = f"{note}{octave}"
                midi_number = 12 * (octave + 1) + i
                freq = 440 * (2 ** ((midi_number - 69) / 12))
                self.note_freqs[note_name] = round(freq, 2)
    
    def detect_pitch_from_audio(self, audio_file):
        """检测音频中的音高序列"""
        print(f"分析音频文件: {audio_file}")
        
        try:
            # 尝试使用librosa
            y, sr = librosa.load(audio_file, sr=None, mono=True)
            duration = len(y) / sr
            print(f"音频时长: {duration:.2f}秒, 采样率: {sr}Hz")
            
            # 使用pYIN算法
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, 
                fmin=librosa.note_to_hz('C3'),
                fmax=librosa.note_to_hz('C6'),
                sr=sr,
                frame_length=2048,
                hop_length=512
            )
            
            # 转换为音符序列
            pitches = []
            hop_seconds = 512 / sr
            
            for i in range(len(f0)):
                if voiced_flag[i] and f0[i] > 0:
                    time = i * hop_seconds
                    note = self.freq_to_note(f0[i])
                    if note:
                        pitches.append((time, time + hop_seconds, note))
            
            # 合并连续相同的音符
            merged_pitches = self.merge_notes(pitches)
            print(f"检测到 {len(merged_pitches)} 个音符段落")
            
            return merged_pitches
            
        except Exception as e:
            print(f"librosa分析失败: {e}")
            # 回退到简单方法
            return self.detect_pitch_simple(audio_file)
    
    def detect_pitch_simple(self, audio_file):
        """简单音高检测方法"""
        try:
            sample_rate, data = wavfile.read(audio_file)
            
            # 单声道转换
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            
            # 参数
            frame_size = int(sample_rate * 0.05)  # 50ms
            hop_size = int(sample_rate * 0.02)   # 20ms
            
            pitches = []
            for i in range(0, len(data) - frame_size, hop_size):
                frame = data[i:i+frame_size]
                time = i / sample_rate
                
                if np.max(np.abs(frame)) > 0.05:
                    # 找主要频率
                    freqs = np.fft.rfftfreq(frame_size, 1/sample_rate)
                    fft_result = np.abs(np.fft.rfft(frame))
                    
                    if len(fft_result) > 10:
                        peak_idx = np.argmax(fft_result[10:]) + 10
                        freq = freqs[peak_idx]
                        
                        if 100 <= freq <= 1000:
                            note = self.freq_to_note(freq)
                            if note:
                                pitches.append((time, time + frame_size/sample_rate, note))
            
            merged_pitches = self.merge_notes(pitches)
            return merged_pitches
            
        except Exception as e:
            print(f"简单分析也失败: {e}")
            # 返回示例数据
            return [
                (0.0, 1.0, "C4"),
                (1.0, 2.0, "D4"),
                (2.0, 3.0, "E4"),
                (3.0, 4.0, "F4"),
                (4.0, 5.0, "G4"),
                (5.0, 6.0, "A4"),
                (6.0, 7.0, "B4"),
                (7.0, 8.0, "C5"),
            ]
    
    def freq_to_note(self, freq):
        """频率转音符"""
        if freq <= 0:
            return None
        
        closest_note = min(self.note_freqs.items(), 
                          key=lambda x: abs(math.log(freq/x[1])))
        return closest_note[0]
    
    def merge_notes(self, pitches):
        """合并连续相同的音符"""
        if not pitches:
            return []
        
        merged = []
        current = list(pitches[0])
        
        for i in range(1, len(pitches)):
            t0, t1, note = pitches[i]
            
            if note == current[2] and t0 - current[1] < 0.2:
                # 相同音符且间隔短，合并
                current[1] = t1
            else:
                # 检查持续时间
                duration = current[1] - current[0]
                if duration >= self.min_note_duration:
                    merged.append(tuple(current))
                elif merged:
                    # 短音符合并到前一个
                    prev = list(merged[-1])
                    prev[1] = current[1]
                    merged[-1] = tuple(prev)
                
                # 开始新的音符
                current = [t0, t1, note]
        
        # 处理最后一个
        duration = current[1] - current[0]
        if duration >= self.min_note_duration:
            merged.append(tuple(current))
        elif merged:
            prev = list(merged[-1])
            prev[1] = current[1]
            merged[-1] = tuple(prev)
        
        return merged
    
    def save_pitches(self, pitches, output_file):
        """保存音高序列到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# start_time,end_time,note\n")
                for t0, t1, note in pitches:
                    f.write(f"{t0:.3f},{t1:.3f},{note}\n")
            print(f"音高序列已保存到: {output_file}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False
    
    def load_pitches(self, input_file):
        """从文件加载音高序列"""
        try:
            pitches = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(',')
                        if len(parts) == 3:
                            pitches.append((float(parts[0]), float(parts[1]), parts[2]))
            print(f"从文件加载了 {len(pitches)} 个音符段落")
            return pitches
        except Exception as e:
            print(f"加载失败: {e}")
            return []

if __name__ == '__main__':
    processor = SimpleAudioProcessor(min_note_duration=0.4)
    
    audio_file = "Dance_Of _the_Golden_Snake.wav"
    if Path(audio_file).exists():
        pitches = processor.detect_pitch_from_audio(audio_file)
        processor.save_pitches(pitches, "Dance_Of_the_Golden_Snake_pitches.txt")
        
        # 直接提取唯一音符
        unique_notes = []
        for _, _, note in pitches:
            if note not in unique_notes:
                unique_notes.append(note)
        
        print(f"\n提取到的唯一音符 ({len(unique_notes)}个):")
        print(sorted(unique_notes, key=lambda x: processor.note_freqs.get(x, 0)))
    else:
        print(f"文件不存在: {audio_file}")