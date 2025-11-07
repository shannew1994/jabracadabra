#!/usr/bin/env python3
"""Complete voice assistant with wake word and Ollama integration."""

import speech_recognition as sr
import numpy as np
import re
import sounddevice as sd
import soundfile as sf
from queue import Queue
from time import sleep
from faster_whisper import WhisperModel
import ollama

class VoiceAssistant:
    def __init__(
        self,
        wake_pattern=r"\b(hey|hi|hello)\s+(assistant|jarvis)\b",
        model_size="base.en",
        ollama_model="llama2",
        sound_file="sound/blow.aiff"
    ):
        self.wake_pattern = re.compile(wake_pattern, re.IGNORECASE)
        self.sound_file = sound_file
        self.ollama_model = ollama_model
        
        print(f"Loading Whisper model: {model_size}")
        self.whisper = WhisperModel(model_size, device="cpu", compute_type="int8")
        
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0
        
        self.audio_queue = Queue()
        self.is_running = False
        self.is_awake = False
        self.is_processing = False
        
    def transcribe(self, audio_data):
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.whisper.transcribe(audio_np, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments)
    
    def check_wake_word(self, text):
        return self.wake_pattern.search(text) is not None
    
    def play_sound(self):
        try:
            data, samplerate = sf.read(self.sound_file)
            sd.play(data, samplerate)
            sd.wait()
        except Exception as e:
            print(f"Sound error: {e}")
    
    def ask_ollama(self, question):
        try:
            print(f"\nSending to Ollama: {question}")
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': question}]
            )
            
            answer = response['message']['content']
            print(f"Response: {answer}\n")
            
            return answer
        except Exception as e:
            print(f"Ollama error: {e}")
            return "Error processing request."
    
    def _audio_callback(self, _, audio):
        if not self.is_processing:
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
            phrase_time_limit=5
        )
        
        print("\n" + "="*50)
        print("Voice Assistant Active")
        print("="*50)
        print("\n1. Say wake word: 'Hey Assistant'")
        print("2. Wait for beep")
        print("3. Ask your question\n")
    
    def process_audio(self):
        while self.is_running:
            if not self.audio_queue.empty() and not self.is_processing:
                audio_data = self.audio_queue.get()
                
                try:
                    text = self.transcribe(audio_data)
                    
                    if not text.strip():
                        continue
                    
                    if not self.is_awake:
                        print(f"[Monitoring] {text}")
                        
                        if self.check_wake_word(text):
                            print("\n✓ Wake word detected!")
                            print("Listening for command...\n")
                            self.play_sound()
                            self.is_awake = True
                    else:
                        print(f"[Command] {text}")
                        
                        self.is_processing = True
                        response = self.ask_ollama(text)
                        self.is_processing = False
                        
                        self.is_awake = False
                        print("Ready for wake word...\n")
                        
                except Exception as e:
                    print(f"Error: {e}")
                    self.is_awake = False
            else:
                sleep(0.1)
    
    def stop(self):
        self.is_running = False
        if hasattr(self, 'stop_listening'):
            self.stop_listening(wait_for_stop=False)

if __name__ == "__main__":
    assistant = VoiceAssistant(
        wake_pattern=r"\b(hey|hi|hello)\s+(assistant|jarvis)\b",
        model_size="base.en",
        ollama_model="qwen2.5:0.5b",
        sound_file="sound/blow.aiff"
    )
    
    try:
        assistant.start()
        assistant.process_audio()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        assistant.stop()