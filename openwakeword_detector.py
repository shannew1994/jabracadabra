#!/usr/bin/env python3
"""Wake word detection using speech_recognition for audio capture."""

import speech_recognition as sr
import numpy as np
import sounddevice as sd
import soundfile as sf
from openwakeword.model import Model
import openwakeword
import time
from queue import Queue

class OpenWakeWordDetector:
    def __init__(
        self,
        model_path="models/hey_dja_bra.tflite",
        sound_file="sound/blow.aiff",
        threshold=0.5,
        cooldown=2.0,
        sample_rate=16000
    ):
        self.sound_file = sound_file
        self.threshold = threshold
        self.cooldown = cooldown
        self.sample_rate = sample_rate
        self.chunk_size = 1280  # openwakeword requirement
        
        self.last_detection = 0
        self.is_running = False
        
        # Queue for audio chunks
        self.audio_queue = Queue()
        
        # Download preprocessing models
        print("Downloading required preprocessing models...")
        openwakeword.utils.download_models()
        
        print(f"Loading openwakeword model: {model_path}")
        self.model = Model(wakeword_models=[model_path])
        
        self.wake_word_name = list(self.model.models.keys())[0]
        
        print(f"Model loaded: {self.wake_word_name}")
        print(f"Detection threshold: {self.threshold}")
        print(f"Cooldown period: {self.cooldown}s")
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
    def play_sound(self):
        try:
            data, samplerate = sf.read(self.sound_file)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Could not play sound: {e}")
    
    def audio_callback(self, recognizer, audio):
        """Callback for background audio listener."""
        self.audio_queue.put(audio.get_raw_data())
    
    def start(self):
        self.is_running = True
        
        # Initialize microphone
        mic = sr.Microphone(sample_rate=self.sample_rate)
        
        with mic as source:
            print("\nAdjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        print("\n" + "="*50)
        print("Listening for wake word...")
        print("="*50 + "\n")
        
        # Start background listener
        self.stop_listening = self.recognizer.listen_in_background(
            mic,
            self.audio_callback,
            phrase_time_limit=2  # Capture audio every 2 seconds
        )
        
        try:
            while self.is_running:
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()
                    
                    # Convert to numpy array
                    audio_np = np.frombuffer(audio_data, dtype=np.int16)
                    
                    # Process in chunks of 1280 samples (openwakeword requirement)
                    for i in range(0, len(audio_np), self.chunk_size):
                        chunk = audio_np[i:i + self.chunk_size]
                        
                        # Pad if chunk is too small
                        if len(chunk) < self.chunk_size:
                            chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
                        
                        # Predict
                        prediction = self.model.predict(chunk)
                        
                        for wake_word, score in prediction.items():
                            current_time = time.time()
                            
                            if score >= self.threshold:
                                if current_time - self.last_detection >= self.cooldown:
                                    print(f"✓ Wake word detected! (confidence: {score:.2f})")
                                    self.play_sound()
                                    self.last_detection = current_time
                                    print(f"Cooldown active for {self.cooldown}s...\n")
                else:
                    time.sleep(0.05)
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop_listening(wait_for_stop=False)
    
    def stop(self):
        self.is_running = False

if __name__ == "__main__":
    detector = OpenWakeWordDetector(
        model_path="models/hey_dja_bra.tflite",
        sound_file="sound/blow.aiff",
        threshold=0.5,
        cooldown=2.0
    )
    
    try:
        detector.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        detector.stop()