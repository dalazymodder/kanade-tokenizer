"""
Kanade Real-Time Voice Conversion Desktop GUI - STREAMING OPTIMIZED
Uses proper overlap-add with context buffering for natural, low-latency voice conversion

Key improvements:
- Streaming engine with context awareness
- Overlap-add for smooth transitions
- Cached reference features
- Optimized defaults: 1000ms chunk, 2000ms context
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import deque
from pathlib import Path

import numpy as np
import torch
import sounddevice as sd
import torchaudio
import soundfile as sf

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    from kanade_tokenizer.model import KanadeModel
    from kanade_tokenizer.util import load_vocoder
    from kanade_streaming import StreamingKanadeEngine, AdaptiveStreamingEngine
    import webrtcvad
    HAS_WEBRTC_VAD = True
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("   If missing webrtcvad: pip install webrtcvad")
    HAS_WEBRTC_VAD = False

# --- Configuration ---
class Config:
    def __init__(self):
        # Audio Stream Settings
        self.stream_rate = 48000     # Rate for Mic/Speakers
        self.model_rate = 24000      # Rate Kanade expects
        self.chunk_time = 1.0        # 1000ms chunks (new audio per iteration)
        self.context_time = 2.0      # 2000ms context window (total audio for model)
        self.overlap_time = 0.05     # 50ms overlap for crossfading
        self.input_channels = 1
        self.output_channels = 2
        
        self.reference_path = ""
        self.repo_id = "frothywater/kanade-25hz-clean"
        self.vocoder_name = "hift"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # VAD settings
        self.vad_enabled = True
        self.vad_aggressiveness = 2  # Internal WebRTC level (0-3)
        self.vad_aggressiveness_display = 30  # Display value (0-100)
        self.vad_threshold = 0.3
        
        # Performance settings
        self.use_torch_compile = True
        self.use_adaptive_streaming = False  # Use adaptive buffer adjustment

# --- Optimized Voice Activity Detection ---
class VoiceActivityDetector:
    def __init__(self, config):
        self.config = config
        if HAS_WEBRTC_VAD:
            self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        else:
            self.vad = None
        
        self.vad_target_rate = 16000
        self.last_sample_rate = None
        self.cached_resampler = None

    def update_aggressiveness(self, level):
        """Update VAD aggressiveness level (0-3)"""
        if self.vad is not None:
            self.vad.set_mode(int(level))
            self.config.vad_aggressiveness = int(level)

    def _get_resampler(self, source_rate):
        """Get or create cached resampler for VAD"""
        if source_rate == self.vad_target_rate:
            return None
        
        if self.last_sample_rate != source_rate:
            import librosa
            self.cached_resampler = lambda x: librosa.resample(
                x, orig_sr=source_rate, target_sr=self.vad_target_rate
            )
            self.last_sample_rate = source_rate
        
        return self.cached_resampler

    def is_speech(self, audio_chunk, sample_rate):
        if not self.config.vad_enabled or not self.vad:
            return True
            
        try:
            resampler = self._get_resampler(sample_rate)
            if resampler:
                audio_16k = resampler(audio_chunk)
            else:
                audio_16k = audio_chunk

            audio_int16 = np.clip(audio_16k * 32767, -32768, 32767).astype(np.int16)
            
            frame_size = 480  # 30ms at 16kHz
            if len(audio_int16) < frame_size:
                return False
            
            speech_frames = 0
            total_frames = 0
            step = frame_size * 2
            
            for i in range(0, len(audio_int16) - frame_size, step):
                frame = audio_int16[i:i+frame_size].tobytes()
                if self.vad.is_speech(frame, self.vad_target_rate):
                    speech_frames += 1
                total_frames += 1
                
            if total_frames == 0:
                return False
            
            return (speech_frames / total_frames) > self.config.vad_threshold

        except Exception as e:
            print(f"VAD Error: {e}")
            return True

# --- Streaming Kanade Real-Time Engine ---
class KanadeRealtime:
    def __init__(self, config):
        self.config = config
        self.processing_times = deque(maxlen=30)
        self.speech_count = 0
        self.total_chunks = 0
        
        print(f"🚀 Initializing Kanade on {config.device}...")
        
        # 1. Load Model
        self.model = KanadeModel.from_pretrained(repo_id=config.repo_id)
        self.model = self.model.to(config.device).eval()
        
        # 2. Load Vocoder
        self.vocoder = load_vocoder(name=config.vocoder_name)
        self.vocoder = self.vocoder.to(config.device).eval()
        
        # 3. OPTIMIZATION: Use torch.compile() for faster inference
        if config.use_torch_compile and hasattr(torch, 'compile'):
            print("⚡ Applying torch.compile() optimization...")
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("✅ torch.compile() enabled - expect 20-40% speedup after warmup")
            except Exception as e:
                print(f"⚠️ torch.compile() failed: {e}")
        
        # 4. Create Streaming Engine
        engine_class = AdaptiveStreamingEngine if config.use_adaptive_streaming else StreamingKanadeEngine
        self.streaming_engine = engine_class(
            model=self.model,
            vocoder=self.vocoder,
            sample_rate=config.model_rate,
            chunk_size_ms=int(config.chunk_time * 1000),
            context_size_ms=int(config.context_time * 1000),
            overlap_size_ms=int(config.overlap_time * 1000),
            device=config.device
        )
        
        # 5. Setup VAD
        self.vad = VoiceActivityDetector(config)
        
        # 6. CPU-based resamplers for I/O
        print("🔧 Setting up CPU-based resamplers...")
        self.resample_in = torchaudio.transforms.Resample(
            config.stream_rate, config.model_rate
        )
        self.resample_out = torchaudio.transforms.Resample(
            config.model_rate, config.stream_rate
        )
        
        # Preallocate silence buffer
        self._silence_buffer = None
        self._silence_buffer_size = 0

        print("✅ Kanade Streaming Engine Ready!")

    def load_reference(self, path):
        """Loads and pre-processes reference audio"""
        try:
            print(f"📥 Loading reference: {path}")
            
            wav_np, sr = sf.read(path)
            wav = torch.from_numpy(wav_np).float()
            
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.t()
            
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
                
            if sr != self.config.model_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.model_rate)
                wav = resampler(wav)
            
            # Load into streaming engine (this caches the reference features)
            self.streaming_engine.load_reference(wav.squeeze(0))
            
            print(f"✅ Reference loaded and cached")
            return True
        except Exception as e:
            print(f"❌ Error loading ref: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_silence_buffer(self, size):
        """Get or create preallocated silence buffer"""
        if self._silence_buffer is None or self._silence_buffer_size != size:
            self._silence_buffer = np.zeros(
                (size, self.config.output_channels), 
                dtype=np.float32
            )
            self._silence_buffer_size = size
        return self._silence_buffer

    def process_chunk(self, audio_chunk_np):
        """
        STREAMING OPTIMIZED Main Real-Time Loop
        
        Key improvements:
        - Uses streaming engine with context buffering
        - Proper overlap-add for smooth transitions
        - Cached reference features
        """
        self.total_chunks += 1
        
        # Early exit with silence if no speech detected
        if not self.vad.is_speech(audio_chunk_np, self.config.stream_rate):
            return self._get_silence_buffer(len(audio_chunk_np))
        
        self.speech_count += 1
        
        start_t = time.time()
        
        try:
            # 1. Resample input: 48k -> 24k (CPU)
            wav_tensor = torch.from_numpy(audio_chunk_np).float()
            wav_24k = self.resample_in(wav_tensor)
            
            # 2. Process through streaming engine (handles context + overlap-add)
            output_24k = self.streaming_engine.process_chunk(wav_24k)
            
            # 3. Resample output: 24k -> 48k (CPU)
            output_48k = self.resample_out(output_24k)
            
            # 4. Convert to numpy
            output_np = output_48k.numpy()
            
            # 5. Ensure correct output shape (samples, channels)
            if output_np.ndim == 1:
                output_np = np.column_stack([output_np, output_np])
            
            # 6. Match input length
            target_len = len(audio_chunk_np)
            if len(output_np) < target_len:
                padding = np.zeros(
                    (target_len - len(output_np), self.config.output_channels),
                    dtype=np.float32
                )
                output_np = np.vstack([output_np, padding])
            elif len(output_np) > target_len:
                output_np = output_np[:target_len]
            
            # Track processing time
            elapsed = time.time() - start_t
            self.processing_times.append(elapsed)
            
            return output_np.astype(np.float32)
            
        except Exception as e:
            print(f"❌ Processing Error: {e}")
            import traceback
            traceback.print_exc()
            return self._get_silence_buffer(len(audio_chunk_np))

    def get_stats(self):
        """Returns average processing time, RTF, and speech percentage"""
        if not self.processing_times:
            return 0, 0.0, 0.0
        
        avg_ms = int(np.mean(self.processing_times) * 1000)
        rtf = np.mean(self.processing_times) / self.config.chunk_time
        speech_pct = (self.speech_count / max(1, self.total_chunks)) * 100
        
        # Also get streaming engine stats
        stream_stats = self.streaming_engine.get_stats()
        
        return avg_ms, rtf, speech_pct

# --- Audio Manager (SoundDevice) ---
class AudioManager:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        self.stream = None
        self.running = False
        self.devices_in = []
        self.devices_out = []
        self.update_device_list()

    def update_device_list(self):
        sd._terminate()
        sd._initialize()
        devs = sd.query_devices()
        self.devices_in = [d['name'] for d in devs if d['max_input_channels'] > 0]
        self.devices_out = [d['name'] for d in devs if d['max_output_channels'] > 0]

    def callback(self, indata, outdata, frames, time_info, status):
        if status and not hasattr(self, '_last_status_print'):
            self._last_status_print = time.time()
            print(f"Audio Status: {status}")
        elif status and (time.time() - getattr(self, '_last_status_print', 0)) > 5.0:
            self._last_status_print = time.time()
            print(f"Audio Status: {status}")
        
        try:
            mono_in = indata[:, 0] if indata.ndim > 1 else indata
            result = self.engine.process_chunk(mono_in)
            outdata[:] = result
        except Exception as e:
            print(f"Callback error: {e}")
            outdata[:] = np.zeros((frames, self.config.output_channels), dtype=np.float32)

    def start(self, in_dev, out_dev):
        if self.running:
            return
        try:
            print(f"🎤 Starting stream: {in_dev} -> {out_dev}")
            print(f"   Chunk size: {self.config.chunk_time * 1000:.0f}ms")
            print(f"   Context window: {self.config.context_time * 1000:.0f}ms")
            
            # Reset streaming engine state
            self.engine.streaming_engine.reset()
            
            in_id = [i for i, d in enumerate(sd.query_devices()) if d['name'] == in_dev][0]
            out_id = [i for i, d in enumerate(sd.query_devices()) if d['name'] == out_dev][0]
            
            self.stream = sd.Stream(
                device=(in_id, out_id),
                samplerate=self.config.stream_rate,
                blocksize=int(self.config.stream_rate * self.config.chunk_time),
                channels=(1, self.config.output_channels),
                callback=self.callback,
                latency='low'
            )
            self.stream.start()
            self.running = True
            return True
        except Exception as e:
            print(f"Stream Error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.running = False

# --- GUI ---
class KanadeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kanade Real-Time Voice Changer")
        self.config = Config()
        self.engine = None
        self.audio = None
        
        self.setup_ui()
        threading.Thread(target=self.init_engine, daemon=True).start()
        self.update_monitor()

    def init_engine(self):
        self.status_var.set("Loading Models (Please Wait)...")
        self.engine = KanadeRealtime(self.config)
        self.audio = AudioManager(self.engine, self.config)
        self.status_var.set("Ready")
        self.refresh_devices()

    def setup_ui(self):
        pad = {'padx': 5, 'pady': 5}
        
        # 1. Reference Audio
        ref_frame = ttk.LabelFrame(self.root, text="Reference Voice (Target)", padding=10)
        ref_frame.pack(fill='x', **pad)
        
        self.ref_path = tk.StringVar()
        ttk.Entry(ref_frame, textvariable=self.ref_path, width=40).pack(side='left', **pad)
        ttk.Button(ref_frame, text="Browse", command=self.browse_ref).pack(side='left', **pad)
        ttk.Button(ref_frame, text="Load", command=self.load_ref).pack(side='left', **pad)
        self.ref_lbl = ttk.Label(ref_frame, text="Not Loaded", foreground='red')
        self.ref_lbl.pack(side='left', **pad)

        # 2. Audio Devices
        dev_frame = ttk.LabelFrame(self.root, text="I/O Devices", padding=10)
        dev_frame.pack(fill='x', **pad)
        
        self.in_dev_var = tk.StringVar()
        self.out_dev_var = tk.StringVar()
        
        ttk.Label(dev_frame, text="Input (Mic):").grid(row=0, column=0, sticky='w')
        self.cb_in = ttk.Combobox(dev_frame, textvariable=self.in_dev_var, width=40, state='readonly')
        self.cb_in.grid(row=0, column=1, **pad)
        
        ttk.Label(dev_frame, text="Output (Speaker):").grid(row=1, column=0, sticky='w')
        self.cb_out = ttk.Combobox(dev_frame, textvariable=self.out_dev_var, width=40, state='readonly')
        self.cb_out.grid(row=1, column=1, **pad)
        
        ttk.Button(dev_frame, text="Refresh", command=self.refresh_devices).grid(row=0, column=2, rowspan=2, **pad)

        # 3. Settings
        set_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        set_frame.pack(fill='x', **pad)
        
        # Chunk size slider
        ttk.Label(set_frame, text="Chunk Size (Latency):").grid(row=0, column=0, sticky='w', **pad)
        self.chunk_var = tk.DoubleVar(value=1.0)
        scl_chunk = ttk.Scale(set_frame, from_=0.05, to=2.0, variable=self.chunk_var, orient='horizontal')
        scl_chunk.grid(row=0, column=1, sticky='ew', **pad)
        self.lbl_chunk = ttk.Label(set_frame, text="1000ms")
        self.lbl_chunk.grid(row=0, column=2, **pad)
        
        # Context size slider
        ttk.Label(set_frame, text="Context Window:").grid(row=1, column=0, sticky='w', **pad)
        self.context_var = tk.DoubleVar(value=2.0)
        scl_context = ttk.Scale(set_frame, from_=0.2, to=4.0, variable=self.context_var, orient='horizontal')
        scl_context.grid(row=1, column=1, sticky='ew', **pad)
        self.lbl_context = ttk.Label(set_frame, text="2000ms")
        self.lbl_context.grid(row=1, column=2, **pad)
        
        # Overlap size slider
        ttk.Label(set_frame, text="Overlap Window:").grid(row=2, column=0, sticky='w', **pad)
        self.overlap_var = tk.DoubleVar(value=0.05)
        scl_overlap = ttk.Scale(set_frame, from_=0.01, to=0.2, variable=self.overlap_var, orient='horizontal')
        scl_overlap.grid(row=2, column=1, sticky='ew', **pad)
        self.lbl_overlap = ttk.Label(set_frame, text="50ms")
        self.lbl_overlap.grid(row=2, column=2, **pad)
        
        # VAD Enable checkbox
        self.vad_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(set_frame, text="Enable VAD", variable=self.vad_var, 
                       command=self.on_vad_toggle).grid(row=3, column=0, sticky='w', **pad)
        
        # VAD Aggressiveness slider (0-100 in steps of 10)
        ttk.Label(set_frame, text="VAD Aggressiveness:").grid(row=4, column=0, sticky='w', **pad)
        self.vad_aggr_var = tk.IntVar(value=30)
        scl_vad_aggr = ttk.Scale(set_frame, from_=0, to=100, variable=self.vad_aggr_var, 
                                orient='horizontal', command=self.on_vad_aggr_change)
        scl_vad_aggr.grid(row=4, column=1, sticky='ew', **pad)
        self.lbl_vad_aggr = ttk.Label(set_frame, text="30")
        self.lbl_vad_aggr.grid(row=4, column=2, **pad)
        
        set_frame.columnconfigure(1, weight=1)

        # 4. Controls & Stats
        ctrl_frame = ttk.Frame(self.root, padding=10)
        ctrl_frame.pack(fill='both', expand=True)
        
        self.btn_start = ttk.Button(ctrl_frame, text="START", command=self.toggle_stream)
        self.btn_start.pack(fill='x', pady=10)
        
        self.status_var = tk.StringVar(value="Initializing...")
        ttk.Label(ctrl_frame, textvariable=self.status_var, font=('Arial', 10, 'bold')).pack()
        
        stats_frame = ttk.LabelFrame(ctrl_frame, text="Performance Stats")
        stats_frame.pack(fill='x', pady=10)
        self.lbl_stats = ttk.Label(stats_frame, text="RTF: 0.00 | Latency: 0ms | Speech: 0%")
        self.lbl_stats.pack(**pad)
        
        # OPTIMIZATION INFO
        info_frame = ttk.LabelFrame(ctrl_frame, text="Streaming Optimizations", padding=5)
        info_frame.pack(fill='x', pady=5)
        
        opt_info = [
            "✓ Context Buffering",
            "✓ Overlap-Add",
            "✓ Cached Reference",
            "✓ CPU Resampling"
        ]
        if self.config.use_torch_compile and hasattr(torch, 'compile'):
            opt_info.append("✓ torch.compile")
        
        ttk.Label(info_frame, text=" | ".join(opt_info), font=('Arial', 8)).pack()

    def on_vad_toggle(self):
        """Handle VAD enable/disable"""
        if self.engine:
            self.config.vad_enabled = self.vad_var.get()

    def on_vad_aggr_change(self, value):
        """Handle VAD aggressiveness slider change (0-100 in steps of 10)"""
        # Snap to nearest 10
        val = int(float(value))
        snapped = round(val / 10) * 10
        
        if snapped != self.vad_aggr_var.get():
            self.vad_aggr_var.set(snapped)
        
        # Map 0-100 to 0-3 for WebRTC VAD
        # 0-24 -> 0, 25-49 -> 1, 50-74 -> 2, 75-100 -> 3
        vad_level = min(3, snapped // 25)
        
        # Update VAD in real-time if engine exists
        if self.engine:
            self.engine.vad.update_aggressiveness(vad_level)

    def browse_ref(self):
        f = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
        if f:
            self.ref_path.set(f)

    def load_ref(self):
        if not self.engine:
            return
        path = self.ref_path.get()
        if not os.path.exists(path):
            return
        
        self.status_var.set("Loading Reference...")
        self.root.update()
        if self.engine.load_reference(path):
            self.ref_lbl.config(text="Loaded", foreground='green')
            self.status_var.set("Ready")
        else:
            self.ref_lbl.config(text="Error", foreground='red')

    def refresh_devices(self):
        if not self.audio:
            return
        self.audio.update_device_list()
        self.cb_in['values'] = self.audio.devices_in
        self.cb_out['values'] = self.audio.devices_out
        if self.audio.devices_in:
            self.cb_in.current(0)
        if self.audio.devices_out:
            self.cb_out.current(0)

    def toggle_stream(self):
        if not self.audio:
            return
        
        if self.audio.running:
            self.audio.stop()
            self.btn_start.config(text="START")
            self.status_var.set("Stopped")
        else:
            # Update settings
            self.config.chunk_time = self.chunk_var.get()
            self.config.context_time = self.context_var.get()
            self.config.overlap_time = self.overlap_var.get()
            self.config.vad_enabled = self.vad_var.get()
            
            # Convert 0-100 VAD display value to 0-3 internal value
            vad_display = self.vad_aggr_var.get()
            vad_level = min(3, vad_display // 25)
            self.config.vad_aggressiveness = vad_level
            
            # Recreate streaming engine with new settings
            self.engine.streaming_engine = StreamingKanadeEngine(
                model=self.engine.model,
                vocoder=self.engine.vocoder,
                sample_rate=self.config.model_rate,
                chunk_size_ms=int(self.config.chunk_time * 1000),
                context_size_ms=int(self.config.context_time * 1000),
                overlap_size_ms=int(self.config.overlap_time * 1000),
                device=self.config.device
            )
            
            # Update VAD settings
            self.engine.vad.update_aggressiveness(self.config.vad_aggressiveness)
            
            # Reload reference into new engine
            if hasattr(self.engine.streaming_engine, 'reference_global'):
                path = self.ref_path.get()
                if os.path.exists(path):
                    self.engine.load_reference(path)
            
            in_d = self.in_dev_var.get()
            out_d = self.out_dev_var.get()
            if self.engine.streaming_engine.reference_global is not None:
                if self.audio.start(in_d, out_d):
                    self.btn_start.config(text="STOP")
                    self.status_var.set("Running (Speak now)")
                else:
                    messagebox.showerror("Error", "Could not start audio stream")
            else:
                messagebox.showerror("Error", "Load reference audio first!")

    def update_monitor(self):
        # Update chunk label
        chunk_ms = self.chunk_var.get() * 1000
        self.lbl_chunk.config(text=f"{chunk_ms:.0f}ms")
        
        # Update context label
        context_ms = self.context_var.get() * 1000
        self.lbl_context.config(text=f"{context_ms:.0f}ms")
        
        # Update overlap label
        overlap_ms = self.overlap_var.get() * 1000
        self.lbl_overlap.config(text=f"{overlap_ms:.0f}ms")
        
        # Update VAD aggressiveness label (already 0-100)
        vad_aggr_display = self.vad_aggr_var.get()
        self.lbl_vad_aggr.config(text=f"{vad_aggr_display}")
        
        # Update Stats
        if self.engine:
            ms, rtf, speech = self.engine.get_stats()
            
            rtf_color = 'green' if rtf < 0.8 else 'orange' if rtf < 1.0 else 'red'
            
            self.lbl_stats.config(
                text=f"RTF: {rtf:.3f} | Latency: {ms}ms | Speech: {speech:.1f}%",
                foreground=rtf_color
            )
            
        self.root.after(100, self.update_monitor)

if __name__ == "__main__":
    root = tk.Tk()
    gui = KanadeGUI(root)
    root.mainloop()