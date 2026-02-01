import sys
import os
import time
import torch
import soundfile as sf
import gradio as gr
import numpy as np

# --- 1. PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.append(src_path)

# --- 2. Imports ---
try:
    from kanade_tokenizer.model import KanadeModel
    from kanade_tokenizer.util import load_vocoder, vocode, load_audio
    # Import Kokoro for local TTS
    from kokoro import KPipeline
except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("💡 FIX: Ensure 'kokoro' is installed (pip install kokoro) and 'app.py' is in the root folder.")
    raise e

# --- Configuration ---
KANADE_REPO = "frothywater/kanade-25hz-clean"
KANADE_VOCODER = "hift"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000

print(f"🚀 Initializing on {DEVICE}...")

# --- 3. Load Models (Kanade + Kokoro) ---
try:
    # A. Kanade (Voice Cloner)
    print(f"📥 Loading Kanade...")
    kanade_model = KanadeModel.from_pretrained(repo_id=KANADE_REPO).to(DEVICE).eval()
    
    print(f"🔊 Loading HiFT Vocoder...")
    kanade_vocoder = load_vocoder(name=KANADE_VOCODER).to(DEVICE).eval()
    
    # B. Kokoro (Text-to-Speech)
    print(f"🦜 Loading Kokoro TTS...")
    # 'a' is for American English. You can change to 'b' (British) if preferred.
    kokoro_pipeline = KPipeline(lang_code='a', device=DEVICE)
    print("✅ All Models Loaded.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise e

# --- 4. Core Logic ---

def run_kanade_inference(source_wav, ref_wav):
    """Shared inference logic: Takes source audio (content) + reference (style) -> Cloned Audio"""
    with torch.inference_mode():
        # Kanade extracts content from source, style from reference
        mel_output = kanade_model.voice_conversion(source_wav, ref_wav)
        # Vocoder turns Mel -> Audio
        generated_wav = vocode(kanade_vocoder, mel_output.unsqueeze(0))
    return generated_wav

# --- 5. Handlers ---

# Tab 1: Audio-to-Audio
def voice_conversion(source_path, reference_path):
    if not source_path or not reference_path: return None, "⚠️ Missing audio."
    try:
        source_wav = load_audio(source_path, sample_rate=SAMPLE_RATE).to(DEVICE)
        ref_wav = load_audio(reference_path, sample_rate=SAMPLE_RATE).to(DEVICE)
        
        start = time.time()
        final_wav = run_kanade_inference(source_wav, ref_wav)
        proc_time = time.time() - start
        
        output_np = final_wav.squeeze().cpu().float().numpy()
        return (SAMPLE_RATE, output_np), f"✅ Converted in {proc_time:.2f}s"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Tab 2: Text-to-Speech (Kokoro -> Audio -> Kanade)
def tts_conversion(text, reference_path, voice_style, speed):
    if not text or not reference_path: return None, "⚠️ Missing text or reference."
    try:
        temp_tts_file = "temp_kokoro_out.wav"
        
        # Step A: Generate Base Audio with Kokoro (The "Actor")
        # Generate yields segments; we concat them into one audio array
        generator = kokoro_pipeline(text, voice=voice_style, speed=speed, split_pattern=r'\n+')
        
        # Collect all audio segments
        all_audio = []
        for _, _, audio_segment in generator:
            all_audio.append(audio_segment)
        
        if not all_audio:
            return None, "❌ Kokoro produced no audio."
            
        full_audio = np.concatenate(all_audio)
        
        # Save to temp file (Kokoro usually outputs 24kHz, which matches Kanade perfectly!)
        sf.write(temp_tts_file, full_audio, 24000)
        
        # Step B: Load that new file as the Source for Kanade
        source_wav = load_audio(temp_tts_file, sample_rate=SAMPLE_RATE).to(DEVICE)
        ref_wav = load_audio(reference_path, sample_rate=SAMPLE_RATE).to(DEVICE)
        
        # Step C: Clone it!
        start = time.time()
        final_wav = run_kanade_inference(source_wav, ref_wav)
        proc_time = time.time() - start
        
        output_np = final_wav.squeeze().cpu().float().numpy()
        return (SAMPLE_RATE, output_np), f"✅ Generated (Kokoro) & Cloned (Kanade) in {proc_time:.2f}s"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"

# --- 6. Interface ---
with gr.Blocks(title="Kanade x Kokoro Local") as demo:
    gr.Markdown(f"# 🗣️ Kanade Local Cloning\n**Cloner:** `{KANADE_REPO}` | **TTS Base:** `Kokoro (Local)`")

    with gr.Tabs():
        # --- TAB 1: VOICE CONVERSION ---
        with gr.Tab("🎤 Audio Conversion"):
            gr.Markdown("Convert an existing audio file into the target voice.")
            with gr.Row():
                vc_src = gr.Audio(label="Source (Content)", type="filepath")
                vc_ref = gr.Audio(label="Reference (Target Voice)", type="filepath")
            vc_btn = gr.Button("Convert Voice", variant="primary")
            vc_out = gr.Audio(label="Result")
            vc_stat = gr.Textbox(label="Status")
            vc_btn.click(voice_conversion, [vc_src, vc_ref], [vc_out, vc_stat])

        # --- TAB 2: TEXT TO SPEECH ---
        with gr.Tab("📝 Text to Speech"):
            gr.Markdown("Type text, generate it locally with Kokoro, and clone it into the target voice.")
            with gr.Row():
                with gr.Column():
                    tts_text = gr.Textbox(label="Text to Speak", lines=3, placeholder="Type something here...")
                    
                    with gr.Row():
                        # Standard Kokoro voices (American English)
                        tts_voice = gr.Dropdown(
                            label="Base Actor Voice (Prosody)", 
                            choices=["af_heart", "af_bella", "af_nicole", "am_michael", "am_adam"], 
                            value="af_heart",
                            info="Select a base voice that matches the GENDER and SPEED of your target."
                        )
                        tts_speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")

                    tts_ref = gr.Audio(label="Reference (Target Voice)", type="filepath")
                    tts_btn = gr.Button("Generate & Clone", variant="primary")
                with gr.Column():
                    tts_out = gr.Audio(label="Result")
                    tts_stat = gr.Textbox(label="Status")
            
            tts_btn.click(tts_conversion, [tts_text, tts_ref, tts_voice, tts_speed], [tts_out, tts_stat])

if __name__ == "__main__":
    demo.launch()