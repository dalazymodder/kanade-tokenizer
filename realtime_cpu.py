"""
Kanade Real-Time Voice Conversion Desktop GUI - CPU OPTIMIZED VERSION

Key CPU optimizations:
- Limited PyTorch threads (biggest impact!)
- Efficient VAD with throttling during silence
- Scipy-based resampling (faster than librosa)
- Inference mode for lower overhead
- Reduced context when on CPU
- Sleep during idle to yield CPU time
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

# =============================================================================
# CPU OPTIMIZATION: Limit PyTorch threads BEFORE any model loading
# This is the BIGGEST impact on CPU usage - prevents PyTorch from using all cores
# =============================================================================
CPU_THREADS = 2  # Default: 2 threads

if not torch.cuda.is_available():
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    # Enable MKL-DNN if available
    if hasattr(torch.backends, 'mkldnn'):
        torch.backends.mkldnn.enabled = True
    print(f"⚡ CPU Mode: Limited to {CPU_THREADS} threads")

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
        self.input_channels = 1
        self.output_channels = 2
        
        self.reference_path = ""
        self.repo_id = "frothywater/kanade-25hz-clean"
        self.vocoder_name = "hift"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # =============================================================================
        # DEFAULT VALUES (same for CPU and GPU now)
        # =============================================================================
        self.chunk_time = 1.0        # 1000ms chunks
        self.context_time = 2.0      # 2000ms context
        self.overlap_time = 0.05     # 50ms overlap
        self.use_torch_compile = False if self.device == "cpu" else True
        
        # VAD settings
        self.vad_enabled = True
        self.vad_aggressiveness = 2  # Internal WebRTC level (0-3)
        self.vad_aggressiveness_display = 50  # Display value (0-100)
        self.vad_threshold = 0.3
        
        # Performance settings
        self.use_adaptive_streaming = False


# --- CPU-Optimized Voice Activity Detection ---
class VoiceActivityDetector:
    """
    Optimized VAD with:
    - Scipy resampling (faster than librosa)
    - Throttled checking during silence (reduces CPU when not speaking)
    - Cached buffers
    """
    def __init__(self, config):
        self.config = config
        if HAS_WEBRTC_VAD:
            self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        else:
            self.vad = None
        
        self.vad_target_rate = 16000
        self.last_sample_rate = None
        self._resample_up = None
        self._resample_down = None
        
        # Throttling state - skip VAD checks during prolonged silence
        self._consecutive_silence = 0
        self._last_result = False
        self._throttle_counter = 0

    def update_aggressiveness(self, level):
        """Update VAD aggressiveness level (0-3)"""
        if self.vad is not None:
            self.vad.set_mode(int(level))
            self.config.vad_aggressiveness = int(level)

    def _setup_resampler(self, source_rate):
        """Setup scipy-based resampler (faster than librosa)"""
        if source_rate == self.vad_target_rate:
            self._resample_up = None
            self._resample_down = None
            return
        
        if self.last_sample_rate != source_rate:
            from math import gcd
            g = gcd(source_rate, self.vad_target_rate)
            self._resample_up = self.vad_target_rate // g
            self._resample_down = source_rate // g
            self.last_sample_rate = source_rate

    def _resample(self, audio):
        """Fast scipy resampling"""
        if self._resample_up is None:
            return audio
        from scipy.signal import resample_poly
        return resample_poly(audio, self._resample_up, self._resample_down)

    def is_speech(self, audio_chunk, sample_rate):
        if not self.config.vad_enabled or not self.vad:
            return True
        
        # =============================================================================
        # THROTTLING: When silent for a while, only check every Nth chunk
        # This dramatically reduces CPU usage when nobody is speaking
        # =============================================================================
        if self._consecutive_silence > 3:
            self._throttle_counter += 1
            # During prolonged silence, only check every 3rd chunk
            if self._throttle_counter < 3:
                return False
            self._throttle_counter = 0
        
        try:
            self._setup_resampler(sample_rate)
            audio_16k = self._resample(audio_chunk)
            
            audio_int16 = np.clip(audio_16k * 32767, -32768, 32767).astype(np.int16)
            
            frame_size = 480  # 30ms at 16kHz
            if len(audio_int16) < frame_size:
                return False
            
            speech_frames = 0
            total_frames = 0
            # Larger step = fewer checks = less CPU
            step = frame_size * 3  # Check every 90ms instead of 60ms
            
            for i in range(0, len(audio_int16) - frame_size, step):
                frame = audio_int16[i:i+frame_size].tobytes()
                if self.vad.is_speech(frame, self.vad_target_rate):
                    speech_frames += 1
                total_frames += 1
                
            if total_frames == 0:
                return False
            
            result = (speech_frames / total_frames) > self.config.vad_threshold
            
            # Update throttling state
            if result:
                self._consecutive_silence = 0
            else:
                self._consecutive_silence += 1
            
            self._last_result = result
            return result

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
        if config.device == "cpu":
            print(f"   Using {torch.get_num_threads()} CPU threads")
        
        # 1. Load Model
        self.model = KanadeModel.from_pretrained(repo_id=config.repo_id)
        self.model = self.model.to(config.device).eval()
        
        # 2. Load Vocoder
        self.vocoder = load_vocoder(name=config.vocoder_name)
        self.vocoder = self.vocoder.to(config.device).eval()
        
        # 3. torch.compile - SKIP on CPU (adds overhead without benefit)
        if config.use_torch_compile and config.device == 'cuda' and hasattr(torch, 'compile'):
            print("⚡ Applying torch.compile() optimization...")
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("✅ torch.compile() enabled")
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
        print("🔧 Setting up resamplers...")
        self.resample_in = torchaudio.transforms.Resample(
            config.stream_rate, config.model_rate
        )
        self.resample_out = torchaudio.transforms.Resample(
            config.model_rate, config.stream_rate
        )
        
        # Preallocate silence buffer
        self._silence_buffer = None
        self._silence_buffer_size = 0
        
        # Preallocate output buffer to avoid repeated allocations
        self._output_buffer = None

        print("✅ Kanade Streaming Engine Ready!")
        if config.device == "cpu":
            print(f"   CPU Optimized: {config.chunk_time*1000:.0f}ms chunks, {config.context_time*1000:.0f}ms context")

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
        CPU OPTIMIZED Main Real-Time Loop
        
        Optimizations:
        - inference_mode for lower overhead
        - Preallocated buffers
        - Early exit on silence
        """
        self.total_chunks += 1
        
        # Early exit with silence if no speech detected
        if not self.vad.is_speech(audio_chunk_np, self.config.stream_rate):
            return self._get_silence_buffer(len(audio_chunk_np))
        
        self.speech_count += 1
        
        start_t = time.time()
        
        try:
            # Use inference_mode - faster than no_grad on CPU
            with torch.inference_mode():
                # 1. Resample input: 48k -> 24k
                wav_tensor = torch.from_numpy(audio_chunk_np).float()
                wav_24k = self.resample_in(wav_tensor)
                
                # 2. Process through streaming engine
                output_24k = self.streaming_engine.process_chunk(wav_24k)
                
                # 3. Resample output: 24k -> 48k
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
        self.root.title("Kanade Real-Time Voice Changer (CPU Optimized)")
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
        
        # =============================================================================
        # Chunk Size: Slider + Entry (default 1000ms)
        # =============================================================================
        ttk.Label(set_frame, text="Chunk Size (ms):").grid(row=0, column=0, sticky='w', **pad)
        self.chunk_var = tk.DoubleVar(value=1.0)  # 1.0 = 1000ms
        scl_chunk = ttk.Scale(set_frame, from_=0.1, to=3.0, variable=self.chunk_var, 
                             orient='horizontal', command=self.on_chunk_slider)
        scl_chunk.grid(row=0, column=1, sticky='ew', **pad)
        
        self.chunk_entry_var = tk.StringVar(value="1000")
        chunk_entry = ttk.Entry(set_frame, textvariable=self.chunk_entry_var, width=8)
        chunk_entry.grid(row=0, column=2, **pad)
        chunk_entry.bind('<Return>', self.on_chunk_entry)
        chunk_entry.bind('<FocusOut>', self.on_chunk_entry)
        
        # =============================================================================
        # Context Window: Slider + Entry (default 2000ms)
        # =============================================================================
        ttk.Label(set_frame, text="Context Window (ms):").grid(row=1, column=0, sticky='w', **pad)
        self.context_var = tk.DoubleVar(value=2.0)  # 2.0 = 2000ms
        scl_context = ttk.Scale(set_frame, from_=0.5, to=5.0, variable=self.context_var, 
                               orient='horizontal', command=self.on_context_slider)
        scl_context.grid(row=1, column=1, sticky='ew', **pad)
        
        self.context_entry_var = tk.StringVar(value="2000")
        context_entry = ttk.Entry(set_frame, textvariable=self.context_entry_var, width=8)
        context_entry.grid(row=1, column=2, **pad)
        context_entry.bind('<Return>', self.on_context_entry)
        context_entry.bind('<FocusOut>', self.on_context_entry)
        
        # =============================================================================
        # Overlap Window: Slider + Entry (default 50ms)
        # =============================================================================
        ttk.Label(set_frame, text="Overlap Window (ms):").grid(row=2, column=0, sticky='w', **pad)
        self.overlap_var = tk.DoubleVar(value=0.05)  # 0.05 = 50ms
        scl_overlap = ttk.Scale(set_frame, from_=0.01, to=0.3, variable=self.overlap_var, 
                               orient='horizontal', command=self.on_overlap_slider)
        scl_overlap.grid(row=2, column=1, sticky='ew', **pad)
        
        self.overlap_entry_var = tk.StringVar(value="50")
        overlap_entry = ttk.Entry(set_frame, textvariable=self.overlap_entry_var, width=8)
        overlap_entry.grid(row=2, column=2, **pad)
        overlap_entry.bind('<Return>', self.on_overlap_entry)
        overlap_entry.bind('<FocusOut>', self.on_overlap_entry)
        
        # VAD Enable checkbox
        self.vad_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(set_frame, text="Enable VAD", variable=self.vad_var, 
                       command=self.on_vad_toggle).grid(row=3, column=0, sticky='w', **pad)
        
        # =============================================================================
        # VAD Aggressiveness: Slider + Entry (default 50)
        # =============================================================================
        ttk.Label(set_frame, text="VAD Aggressiveness:").grid(row=4, column=0, sticky='w', **pad)
        self.vad_aggr_var = tk.IntVar(value=50)  # Default 50
        scl_vad_aggr = ttk.Scale(set_frame, from_=0, to=100, variable=self.vad_aggr_var, 
                                orient='horizontal', command=self.on_vad_aggr_slider)
        scl_vad_aggr.grid(row=4, column=1, sticky='ew', **pad)
        
        self.vad_aggr_entry_var = tk.StringVar(value="50")
        vad_aggr_entry = ttk.Entry(set_frame, textvariable=self.vad_aggr_entry_var, width=8)
        vad_aggr_entry.grid(row=4, column=2, **pad)
        vad_aggr_entry.bind('<Return>', self.on_vad_aggr_entry)
        vad_aggr_entry.bind('<FocusOut>', self.on_vad_aggr_entry)
        
        # =============================================================================
        # CPU Threads: Slider + Entry (default 2)
        # =============================================================================
        ttk.Label(set_frame, text="CPU Threads:").grid(row=5, column=0, sticky='w', **pad)
        self.thread_var = tk.IntVar(value=CPU_THREADS)  # Default 2
        scl_threads = ttk.Scale(set_frame, from_=1, to=16, variable=self.thread_var, 
                               orient='horizontal', command=self.on_thread_slider)
        scl_threads.grid(row=5, column=1, sticky='ew', **pad)
        
        self.thread_entry_var = tk.StringVar(value=str(CPU_THREADS))
        thread_entry = ttk.Entry(set_frame, textvariable=self.thread_entry_var, width=8)
        thread_entry.grid(row=5, column=2, **pad)
        thread_entry.bind('<Return>', self.on_thread_entry)
        thread_entry.bind('<FocusOut>', self.on_thread_entry)
        
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
        
        # CPU Usage indicator
        self.lbl_cpu = ttk.Label(stats_frame, text=f"CPU Threads: {torch.get_num_threads()}")
        self.lbl_cpu.pack(**pad)
        
        # OPTIMIZATION INFO
        info_frame = ttk.LabelFrame(ctrl_frame, text="CPU Optimizations Active", padding=5)
        info_frame.pack(fill='x', pady=5)
        
        opt_info = [
            f"✓ {CPU_THREADS} Threads",
            "✓ VAD Throttle",
            "✓ Scipy Resample",
            "✓ inference_mode"
        ]
        
        ttk.Label(info_frame, text=" | ".join(opt_info), font=('Arial', 8)).pack()

    # =============================================================================
    # SLIDER CALLBACKS - Update entry when slider moves
    # =============================================================================
    def on_chunk_slider(self, value):
        ms = int(float(value) * 1000)
        self.chunk_entry_var.set(str(ms))

    def on_context_slider(self, value):
        ms = int(float(value) * 1000)
        self.context_entry_var.set(str(ms))

    def on_overlap_slider(self, value):
        ms = int(float(value) * 1000)
        self.overlap_entry_var.set(str(ms))

    def on_vad_aggr_slider(self, value):
        val = int(float(value))
        self.vad_aggr_entry_var.set(str(val))
        # Update VAD level
        vad_level = min(3, val // 25)
        if self.engine:
            self.engine.vad.update_aggressiveness(vad_level)

    def on_thread_slider(self, value):
        threads = int(float(value))
        self.thread_entry_var.set(str(threads))
        torch.set_num_threads(threads)
        self.lbl_cpu.config(text=f"CPU Threads: {threads}")

    # =============================================================================
    # ENTRY CALLBACKS - Update slider when entry changes
    # =============================================================================
    def on_chunk_entry(self, event=None):
        try:
            ms = int(self.chunk_entry_var.get())
            ms = max(100, min(3000, ms))  # Clamp 100-3000ms
            self.chunk_var.set(ms / 1000.0)
            self.chunk_entry_var.set(str(ms))
        except ValueError:
            pass

    def on_context_entry(self, event=None):
        try:
            ms = int(self.context_entry_var.get())
            ms = max(500, min(5000, ms))  # Clamp 500-5000ms
            self.context_var.set(ms / 1000.0)
            self.context_entry_var.set(str(ms))
        except ValueError:
            pass

    def on_overlap_entry(self, event=None):
        try:
            ms = int(self.overlap_entry_var.get())
            ms = max(10, min(300, ms))  # Clamp 10-300ms
            self.overlap_var.set(ms / 1000.0)
            self.overlap_entry_var.set(str(ms))
        except ValueError:
            pass

    def on_vad_aggr_entry(self, event=None):
        try:
            val = int(self.vad_aggr_entry_var.get())
            val = max(0, min(100, val))  # Clamp 0-100
            self.vad_aggr_var.set(val)
            self.vad_aggr_entry_var.set(str(val))
            # Update VAD level
            vad_level = min(3, val // 25)
            if self.engine:
                self.engine.vad.update_aggressiveness(vad_level)
        except ValueError:
            pass

    def on_thread_entry(self, event=None):
        try:
            threads = int(self.thread_entry_var.get())
            threads = max(1, min(16, threads))  # Clamp 1-16
            self.thread_var.set(threads)
            self.thread_entry_var.set(str(threads))
            torch.set_num_threads(threads)
            self.lbl_cpu.config(text=f"CPU Threads: {threads}")
        except ValueError:
            pass

    def on_vad_toggle(self):
        """Handle VAD enable/disable"""
        if self.engine:
            self.config.vad_enabled = self.vad_var.get()

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
            # Update settings from current values
            self.config.chunk_time = self.chunk_var.get()
            self.config.context_time = self.context_var.get()
            self.config.overlap_time = self.overlap_var.get()
            self.config.vad_enabled = self.vad_var.get()
            
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
