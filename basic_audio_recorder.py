#!/usr/bin/env python3
"""Record audio to WAV file with guaranteed duration."""

import speech_recognition as sr
import wave
from queue import Queue
from datetime import datetime
from time import sleep

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.5
        self.recognizer.energy_threshold = 300
        self.audio_queue = Queue()
        
    def _audio_callback(self, _, audio: sr.AudioData):
        self.audio_queue.put(audio.get_raw_data())
        
    def record(self, duration_seconds, chunk_size=1):
        source = sr.Microphone(sample_rate=self.sample_rate)
        
        with source as src:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(src, duration=1)
        
        print(f"Recording for {duration_seconds} seconds...")
        
        stop_listening = self.recognizer.listen_in_background(
            source, 
            self._audio_callback, 
            phrase_time_limit=chunk_size
        )
        
        all_audio = b''
        start_time = datetime.now()
        expected_bytes = self.sample_rate * 2 * duration_seconds
        
        try:
            while True:
                elapsed = (datetime.now() - start_time).total_seconds()
                
                while not self.audio_queue.empty():
                    all_audio += self.audio_queue.get()
                    current = len(all_audio) / (self.sample_rate * 2)
                    print(f"Captured: {current:.1f}s / {duration_seconds}s")
                
                if len(all_audio) >= expected_bytes:
                    break
                
                if elapsed > duration_seconds + 3:
                    print(f"Warning: Timeout at {elapsed:.1f}s")
                    break
                    
                sleep(0.05)
        finally:
            stop_listening(wait_for_stop=False)
        
        return all_audio
    
    def save_wav(self, audio_data, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
        
        duration = len(audio_data) / (self.sample_rate * 2)
        print(f"\nSaved: {filename} ({duration:.2f}s)")

if __name__ == "__main__":
    recorder = AudioRecorder(sample_rate=16000)
    audio = recorder.record(duration_seconds=5, chunk_size=1)
    recorder.save_wav(audio, "recording.wav")