import sounddevice as sd
import scipy.io.wavfile as wav
from transformers import pipeline
import pyttsx3

DURATION = 5           
FS = 16000            
INPUT_WAV = "input.wav"
OUTPUT_WAV = "response.wav"

def record_audio(filename=INPUT_WAV, duration=DURATION, fs=FS):
    """Record audio from the default microphone and save to a WAV file."""
    print(f"🎤 Recording for {duration} seconds... Speak now")
    data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, data)
    print(f"✅ Saved recording as {filename}")

def transcribe_audio(filename=INPUT_WAV):
    """Transcribe the WAV file using a local Whisper model (first run will download)."""
    print("⏳ Loading ASR model (this may download model weights on first run)...")
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
    engine.save_to_file(text, out_file)  
    engine.say(text)                    
    engine.runAndWait()            
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
