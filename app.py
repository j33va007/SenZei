import sounddevice as sd
import scipy.io.wavfile as wav
from transformers import pipeline
import pyttsx3

# ---------- Settings ----------
DURATION = 5           # seconds to record
FS = 16000             # sample rate (Hz)
INPUT_WAV = "input.wav"
OUTPUT_WAV = "response.wav"
# ------------------------------

def record_audio(filename=INPUT_WAV, duration=DURATION, fs=FS):
    """Record audio from the default microphone and save to a WAV file."""
    print(f"🎤 Recording for {duration} seconds... Speak now")
    # sd.rec returns an array shaped (samples, channels)
    data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # wait until recording is finished
    # Write WAV file (scipy expects integer PCM for int16)
    wav.write(filename, fs, data)
    print(f"✅ Saved recording as {filename}")

def transcribe_audio(filename=INPUT_WAV):
    """Transcribe the WAV file using a local Whisper model (first run will download)."""
    print("⏳ Loading ASR model (this may download model weights on first run)...")
    # Use English-optimized small model; change to "openai/whisper-tiny.en" if low memory
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small.en")
    print("⏳ Transcribing...")
    result = asr(filename)
    text = result.get("text", "").strip() if isinstance(result, dict) else ""
    print("📝 Transcription:", text)
    return text

def speak_and_save(text, out_file=OUTPUT_WAV):
    """Use local Windows TTS (pyttsx3) to speak text and save to a WAV file."""
    print("🔊 Speaking and saving response...")
    engine = pyttsx3.init()
    engine.save_to_file(text, out_file)  # save to file
    engine.say(text)                    # speak aloud
    engine.runAndWait()                 # block until done
    print(f"✅ Saved spoken response as {out_file}")

def main():
    record_audio()
    text = transcribe_audio()
    if text:
        speak_and_save("I understood: " + text)
    else:
        speak_and_save("Sorry, I could not understand the audio.")

if __name__ == "__main__":
    main()
