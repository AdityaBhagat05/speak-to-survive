import os
import tempfile
import threading
import time
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from TTS.api import TTS
import re

# Global playback control
play_lock = threading.Lock()
is_speaking = threading.Event()

# Load models
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
whisper_model = whisper.load_model(WHISPER_MODEL, device="cuda")

TTS_MODEL = os.getenv("TTS_MODEL", "tts_models/en/ljspeech/tacotron2-DDC_ph")
tts_engine = TTS(TTS_MODEL)
tts_engine.to("cuda")



def record_audio(silence_threshold=0.1, silence_duration=3.0, samplerate=16000) -> str:

    while is_speaking.is_set():
        time.sleep(0.1)

    q = queue.Queue()
    audio_data = []
    silent_chunks = 0
    silence_limit = int(silence_duration * samplerate / 1024) 
    max_chunks = int(30 * samplerate / 1024)  # 30 sec max

    def callback(indata, frames, time, status):
        if status:
            print("InputStream status:", status)
        q.put(indata.copy())

    print("Speak now... (auto-stop after silence)")
    
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback, blocksize=1024):
        chunk_count = 0
        while chunk_count < max_chunks:
            chunk = q.get()
            audio_data.append(chunk)
            rms = np.sqrt(np.mean(chunk**2))
            
            if rms < silence_threshold:
                silent_chunks += 1
            else:
                silent_chunks = 0
                
            if silent_chunks >= silence_limit:
                break
                
            chunk_count += 1

    audio = np.concatenate(audio_data, axis=0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmp.name, audio, samplerate)
    return tmp.name

def speech_to_text(audio_path: str) -> str:
    print("Transcribing...")
    result = whisper_model.transcribe(audio_path)
    text = result.get("text", "").strip()
    print("Transcription:", text)
    return text

#tts
def text_to_speech(text: str, samplerate=22050) -> str:
    if not text:
        return ""
        
    fname = os.path.join(tempfile.gettempdir(), f"tts_{int(time.time()*1000)}.wav")
    print("Synthesizing speech...")
    tts_engine.tts_to_file(text=text, file_path=fname)

    def play_audio():
        try:
            with play_lock:  # 🔒 Only one playback at a time
                is_speaking.set()  # 🚨 Mark speaking
                data, sr = sf.read(fname, dtype="float32")
                sd.play(data, sr)
                sd.wait()
                is_speaking.clear()  # ✅ Done speaking
            try:
                os.remove(fname)
            except:
                pass
        except Exception as e:
            print("Playback error:", e)

    threading.Thread(target=play_audio).start()
    return fname

def clarity_score(audio_path: str, transcript: str):
    """
    Analyze clarity of the user's speech using audio + transcript.
    Returns a dict with a numeric score and feedback.
    """

    # ----------------------
    # 1. AUDIO ANALYSIS
    # ----------------------
    try:
        audio, samplerate = sf.read(audio_path)
    except Exception as e:
        return {"score": 0, "feedback": f"Audio error: {e}"}

    # Loudness (RMS)
    rms = np.sqrt(np.mean(audio**2))
    loudness_db = 20 * np.log10(rms + 1e-6)  # avoid log(0)

    # Duration in seconds
    duration_sec = len(audio) / samplerate

    # ----------------------
    # 2. TRANSCRIPT ANALYSIS
    # ----------------------
    words = transcript.strip().split()
    num_words = len(words)
    wpm = (num_words / duration_sec) * 60 if duration_sec > 0 else 0

    # Detect filler words
    fillers = re.findall(r"\b(um+|uh+|like|you know)\b", transcript.lower())
    filler_count = len(fillers)

    # ----------------------
    # 3. SCORING
    # ----------------------
    score = 100

    # Loudness penalty (too soft or too loud)
    if loudness_db < -30:
        score -= 20
        loudness_feedback = "Your voice is too soft."
    elif loudness_db > -5:
        score -= 10
        loudness_feedback = "Your voice is a bit too loud."
    else:
        loudness_feedback = "Good voice volume."

    # WPM penalty (ideal: 110–180)
    if wpm < 100:
        score -= 15
        speed_feedback = "You spoke too slowly."
    elif wpm > 190:
        score -= 15
        speed_feedback = "You spoke too quickly."
    else:
        speed_feedback = "Good speaking pace."

    # Filler penalty
    if filler_count > 3:
        score -= filler_count * 2
        filler_feedback = f"You used filler words {filler_count} times."
    else:
        filler_feedback = "Minimal filler words used."

    score = max(0, score)  # never below 0

    feedback = f"{loudness_feedback} {speed_feedback} {filler_feedback}"

    return {"score": score, "feedback": feedback}