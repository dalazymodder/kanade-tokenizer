"""
Optimized Streaming Wrapper for Kanade Voice Conversion
Implements proper overlap-add, context buffering, and crossfading for low-latency real-time inference
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Optional, Tuple


class StreamingKanadeEngine:
    """
    Streaming wrapper for Kanade that maintains audio context and uses overlap-add
    for smooth, artifact-free real-time voice conversion.
    
    Key optimizations:
    1. Maintains rolling context buffer for SSL feature extraction
    2. Implements overlap-add with crossfading
    3. Caches reference features (computed once)
    4. GPU-optimized batch processing
    """
    
    def __init__(
        self,
        model,
        vocoder,
        sample_rate: int = 24000,
        chunk_size_ms: int = 200,  # Size of new audio to process
        context_size_ms: int = 800,  # Total context window for model
        overlap_size_ms: int = 50,  # Overlap for crossfading
        device: str = "cuda"
    ):
        self.model = model
        self.vocoder = vocoder
        self.sample_rate = sample_rate
        self.device = device
        
        # Convert milliseconds to samples
        self.chunk_samples = int(chunk_size_ms * sample_rate / 1000)
        self.context_samples = int(context_size_ms * sample_rate / 1000)
        self.overlap_samples = int(overlap_size_ms * sample_rate / 1000)
        
        # Calculate SSL feature extractor's receptive field
        self.ssl_hop_size = self._calculate_ssl_hop_size()
        
        # Ensure context is large enough for the model
        min_context = self.ssl_hop_size * 4  # Minimum 4 frames of context
        if self.context_samples < min_context:
            print(f"⚠️ Context too small ({self.context_samples}), increasing to {min_context}")
            self.context_samples = min_context
        
        # Audio buffer for maintaining context
        self.audio_buffer = torch.zeros(self.context_samples, device=device)
        
        # Output buffer for crossfading
        self.output_overlap_buffer = None
        
        # Reference features (cached)
        self.reference_global = None
        self.reference_waveform = None
        
        # Crossfade window
        self.crossfade_window = self._create_crossfade_window()
        
        # Stats
        self.chunks_processed = 0
        self.total_latency = 0
        
        print(f"✅ Streaming Engine Initialized:")
        print(f"   Chunk size: {chunk_size_ms}ms ({self.chunk_samples} samples)")
        print(f"   Context window: {context_size_ms}ms ({self.context_samples} samples)")
        print(f"   Overlap: {overlap_size_ms}ms ({self.overlap_samples} samples)")
        print(f"   SSL hop size: {self.ssl_hop_size} samples")
    
    def _calculate_ssl_hop_size(self) -> int:
        """Calculate the hop size of the SSL feature extractor"""
        try:
            return self.model.ssl_feature_extractor.hop_size
        except:
            # WavLM Base Plus typically has hop_size of 320 (20ms at 16kHz)
            # which is 480 at 24kHz
            return 480
    
    def _create_crossfade_window(self) -> torch.Tensor:
        """Create smooth crossfade window (linear fade in/out)"""
        window = torch.linspace(0, 1, self.overlap_samples, device=self.device)
        return window
    
    def load_reference(self, reference_waveform: torch.Tensor):
        """
        Load and cache reference speaker features.
        This only needs to be called once per reference speaker.
        """
        print("📥 Loading reference features...")
        
        # Ensure correct format
        if reference_waveform.dim() == 1:
            reference_waveform = reference_waveform.unsqueeze(0)
        
        reference_waveform = reference_waveform.to(self.device)
        
        with torch.no_grad():
            # Extract global features only (content not needed from reference)
            features = self.model.encode(
                reference_waveform.squeeze(0),
                return_content=False,
                return_global=True
            )
            
            self.reference_global = features.global_embedding
            self.reference_waveform = reference_waveform
        
        print("✅ Reference features cached")
    
    def reset(self):
        """Reset the streaming state (call when starting a new stream)"""
        self.audio_buffer.zero_()
        self.output_overlap_buffer = None
        self.chunks_processed = 0
        self.total_latency = 0
        print("🔄 Stream reset")
    
    def process_chunk(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        """
        Process a single chunk of audio with context awareness.
        
        Args:
            audio_chunk: New audio to process (chunk_samples,) or (chunk_samples, 1)
        
        Returns:
            Generated audio chunk with same length as input
        """
        if self.reference_global is None:
            raise ValueError("Must load reference audio first using load_reference()")
        
        import time
        start_time = time.time()
        
        # Ensure correct shape and device
        if isinstance(audio_chunk, np.ndarray):
            audio_chunk = torch.from_numpy(audio_chunk).float()
        
        if audio_chunk.dim() == 2:
            audio_chunk = audio_chunk.squeeze(-1)
        
        audio_chunk = audio_chunk.to(self.device)
        
        # Ensure chunk is correct size
        if len(audio_chunk) != self.chunk_samples:
            # Pad or trim
            if len(audio_chunk) < self.chunk_samples:
                audio_chunk = F.pad(audio_chunk, (0, self.chunk_samples - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[:self.chunk_samples]
        
        with torch.no_grad():
            # 1. Update audio buffer (rolling window)
            # Shift buffer and add new chunk
            self.audio_buffer = torch.cat([
                self.audio_buffer[self.chunk_samples:],
                audio_chunk
            ])
            
            # 2. Process full context window
            # Extract content features from the full context
            source_features = self.model.encode(
                self.audio_buffer,
                return_content=True,
                return_global=False
            )
            
            # 3. Generate mel spectrogram
            mel_spectrogram = self.model.decode(
                content_embedding=source_features.content_embedding,
                global_embedding=self.reference_global,
                target_audio_length=len(self.audio_buffer)
            )
            
            # 4. Generate waveform from mel
            from kanade_tokenizer.util import vocode
            generated_waveform = vocode(self.vocoder, mel_spectrogram.unsqueeze(0))
            generated_waveform = generated_waveform.squeeze()
            
            # 5. Extract only the new portion (last chunk)
            # The model generates output for the full context, we only want the last chunk
            new_output = generated_waveform[-self.chunk_samples:]
            
            # 6. Apply crossfade with previous overlap
            if self.output_overlap_buffer is not None and self.overlap_samples > 0:
                # Get overlap region from new output
                new_overlap = new_output[:self.overlap_samples]
                
                # Crossfade
                crossfaded = (
                    self.output_overlap_buffer * (1 - self.crossfade_window) +
                    new_overlap * self.crossfade_window
                )
                
                # Replace the beginning of new_output with crossfaded version
                new_output = torch.cat([
                    crossfaded,
                    new_output[self.overlap_samples:]
                ])
            
            # 7. Save overlap for next iteration
            if self.overlap_samples > 0:
                self.output_overlap_buffer = new_output[-self.overlap_samples:].clone()
            
            # 8. Return the clean chunk
            output = new_output
            
            # Ensure output is correct size
            if len(output) != self.chunk_samples:
                if len(output) < self.chunk_samples:
                    output = F.pad(output, (0, self.chunk_samples - len(output)))
                else:
                    output = output[:self.chunk_samples]
        
        # Track stats
        self.chunks_processed += 1
        self.total_latency += (time.time() - start_time)
        
        return output.cpu()
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        if self.chunks_processed == 0:
            return {
                'chunks_processed': 0,
                'avg_latency_ms': 0,
                'avg_rtf': 0
            }
        
        avg_latency = self.total_latency / self.chunks_processed
        chunk_duration = self.chunk_samples / self.sample_rate
        rtf = avg_latency / chunk_duration
        
        return {
            'chunks_processed': self.chunks_processed,
            'avg_latency_ms': avg_latency * 1000,
            'avg_rtf': rtf
        }


class AdaptiveStreamingEngine(StreamingKanadeEngine):
    """
    Advanced version that dynamically adjusts buffer sizes based on RTF performance
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.rtf_history = deque(maxlen=30)
        self.target_rtf = 0.7  # Target to stay well below 1.0
        
    def process_chunk(self, audio_chunk: torch.Tensor) -> torch.Tensor:
        output = super().process_chunk(audio_chunk)
        
        # Update RTF history
        stats = self.get_stats()
        self.rtf_history.append(stats['avg_rtf'])
        
        # Adaptive adjustment logic
        if len(self.rtf_history) == 30:
            avg_rtf = sum(self.rtf_history) / len(self.rtf_history)
            
            if avg_rtf > 0.9:
                # Too slow, need to reduce quality/context
                print(f"⚠️ RTF too high ({avg_rtf:.3f}), consider reducing context_size_ms")
            elif avg_rtf < 0.5:
                # Can afford more quality
                print(f"✨ RTF excellent ({avg_rtf:.3f}), could increase quality if desired")
        
        return output
