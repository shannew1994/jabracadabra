#!/usr/bin/env python3
"""Wake word detection with regex pattern matching."""

import speech_recognition as sr
import numpy as np
import re
import sounddevice as sd
import soundfile as sf
from queue import Queue
from time import sleep
from faster_whisper import WhisperModel

class WakeWordDetector:
    def __init__(
        self, 
        wake_pattern=r"\b(hey|hi|hello)\s+(jabra|assistant)\b",
        model_size="base.en",
        sound_file="sound/blow.aiff"
    ):
        self.wake_pattern = re.compile(wake_pattern, re.IGNORECASE)
        self.sound_file = sound_file
        
        print(f"Loading Whisper model: {model_size}")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0
        self.audio_queue = Queue()
        self.is_running = False
        
    def transcribe(self, audio_data):
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(audio_np, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments)
    
    def check_wake_word(self, text):
        return self.wake_pattern.search(text) is not None
    
    def play_sound(self):
        try:
            data, samplerate = sf.read(self.sound_file)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Could not play sound: {e}")
    
    def _audio_callback(self, _, audio):
        self.audio_queue.put(audio.get_raw_data())
    
    def start(self):
        self.is_running = True
        
        mic = sr.Microphone(sample_rate=16000)
        with mic as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        self.stop_listening = self.recognizer.listen_in_background(
            mic, 
            self._audio_callback,
            phrase_time_limit=3
        )
        
        print("\nListening for wake word...")
        print("Say: 'Hey Assistant' or 'Hello Jarvis'\n")
    
    def process_audio(self):
        while self.is_running:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                
                try:
                    text = self.transcribe(audio_data)
                    
                    if text.strip():
                        print(f"Heard: {text}")
                        
                        if self.check_wake_word(text):
                            print("\n✓ Wake word detected!\n")
                            self.play_sound()
                            
                except Exception as e:
                    print(f"Error: {e}")
            else:
                sleep(0.1)
    
    def stop(self):
        self.is_running = False
        if hasattr(self, 'stop_listening'):
            self.stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    detector = WakeWordDetector(
        wake_pattern=r"\b(hey|hi|hello)\s+(jabra|assistant)\b",
        model_size="base.en",
        sound_file="sound/blow.aiff"
    )
    
    try:
        detector.start()
        detector.process_audio()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        detector.stop()