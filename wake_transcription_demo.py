#!/usr/bin/env python3
"""
Wake word + transcription using:
- speech_recognition for mic capture 
- faster_whisper for transcription 
- openwakeword for wake word detection
"""

import speech_recognition as sr
import numpy as np
import sounddevice as sd
import soundfile as sf
from openwakeword.model import Model
from faster_whisper import WhisperModel
import openwakeword
import time
import threading
from queue import Queue

class VoiceAssistant:
    def __init__(
        self,
        wake_model_path="models/hey_dja_bra.tflite",
        whisper_model="base.en",
        sound_file="sound/blow.aiff",
        wake_threshold=0.5,
        cooldown=2.0
    ):
        self.sound_file = sound_file
        self.wake_threshold = wake_threshold
        self.cooldown = cooldown
        
        self.last_detection = 0
        self.is_running = False
        self.is_awake = False
        
        # Queues
        self.wake_queue = Queue()
        self.transcription_queue = Queue()
        
        # Initialize wake word model
        print("Downloading required preprocessing models...")
        openwakeword.utils.download_models()
        
        print(f"Loading wake word model: {wake_model_path}")
        self.wake_model = Model(wakeword_models=[wake_model_path])
        self.wake_word_name = list(self.wake_model.models.keys())[0]
        
        # Initialize Whisper model
        print(f"Loading Whisper model: {whisper_model}")
        self.whisper = WhisperModel(whisper_model, device="cpu", compute_type="int8")
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0
        
        print(f"\nWake word: {self.wake_word_name}")
        print("Ready!\n")
        
    def play_sound(self):
        try:
            data, samplerate = sf.read(self.sound_file)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Sound error: {e}")
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using faster_whisper."""
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.whisper.transcribe(audio_np, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments)
    
    def wake_word_worker(self):
        """Worker thread for wake word detection."""
        print("[Wake Word] Detection active\n")
        
        while self.is_running:
            if not self.wake_queue.empty():
                audio_data = self.wake_queue.get()
                
                # Convert to numpy array for openwakeword
                audio_np = np.frombuffer(audio_data, dtype=np.int16)
                
                # Process in chunks of 1280 samples (openwakeword requirement)
                chunk_size = 1280
                for i in range(0, len(audio_np), chunk_size):
                    chunk = audio_np[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                    
                    prediction = self.wake_model.predict(chunk)
                    
                    for wake_word, score in prediction.items():
                        current_time = time.time()
                        
                        if score >= self.wake_threshold:
                            if current_time - self.last_detection >= self.cooldown:
                                print(f"\n✓ Wake word detected! (confidence: {score:.2f})")
                                print("Listening for command...\n")
                                self.play_sound()
                                self.is_awake = True
                                self.last_detection = current_time
                                break
            else:
                time.sleep(0.05)
        
        print("[Wake Word] Detection stopped")
    
    def transcription_worker(self):
        """Worker thread for transcription."""
        print("[Transcription] Worker active\n")
        
        while self.is_running:
            if not self.transcription_queue.empty():
                audio_data = self.transcription_queue.get()
                
                try:
                    text = self.transcribe_audio(audio_data)
                    
                    if text.strip():
                        if self.is_awake:
                            print(f"[Command] {text}\n")
                            
                            # Process command here
                            # e.g., send to Ollama, execute action, etc.
                            
                            self.is_awake = False
                            print("Ready for wake word...\n")
                        else:
                            print(f"[Monitoring] {text}")
                            
                except Exception as e:
                    print(f"[Transcription] Error: {e}")
            else:
                time.sleep(0.05)
        
        print("[Transcription] Worker stopped")
    
    def audio_callback(self, recognizer, audio):
        """Callback from speech_recognition background listener."""
        audio_data = audio.get_raw_data()
        
        # Send to wake word detection (always)
        self.wake_queue.put(audio_data)
        
        # Send to transcription queue
        self.transcription_queue.put(audio_data)
    
    def start(self):
        self.is_running = True
        
        # Start worker threads
        wake_thread = threading.Thread(target=self.wake_word_worker, daemon=True)
        transcription_thread = threading.Thread(target=self.transcription_worker, daemon=True)
        
        wake_thread.start()
        transcription_thread.start()
        
        # Initialize microphone
        mic = sr.Microphone(sample_rate=16000)
        
        with mic as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Start background listener
        print("\n" + "="*60)
        print("VOICE ASSISTANT ACTIVE")
        print("="*60)
        print("Wake word detection: Active")
        print("Continuous transcription: Active")
        print("="*60 + "\n")
        
        self.stop_listening = self.recognizer.listen_in_background(
            mic,
            self.audio_callback,
            phrase_time_limit=5
        )
        
        try:
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.stop_listening(wait_for_stop=False)
            wake_thread.join(timeout=2)
            transcription_thread.join(timeout=2)
    
    def stop(self):
        self.is_running = False

if __name__ == "__main__":
    assistant = VoiceAssistant(
        wake_model_path="models/hey_dja_bra.tflite",
        whisper_model="base.en",
        sound_file="sound/blow.aiff",
        wake_threshold=0.5,
        cooldown=2.0
    )
    
    try:
        assistant.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        assistant.stop()