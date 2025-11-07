# Voice Assistant with Wake Word Detection

A collection of Python scripts for building voice assistants with speech recognition, wake word detection, and AI chat integration using Ollama.

## 📋 Overview

This repository contains progressive examples of voice assistant implementations, from basic audio recording to a full-featured assistant with wake word detection and AI responses.

## 🗂️ Files

### Core Scripts

#### `voice_assistant.py` ⭐ **RECOMMENDED**
Complete voice assistant combining all features:
- **Continuous transcription** using Whisper
- **Wake word detection** using OpenWakeWord
- **AI responses** via Ollama integration

**Configuration:**
- `wake_model_path`: Path to custom wake word model (default: `models/hey_dja_bra.tflite`)
- `whisper_model`: Whisper model size (default: `base.en`)
- `ollama_model`: Ollama model name (default: `qwen2.5:0.5b`)
- `wake_threshold`: Detection sensitivity 0.0-1.0 (default: `0.5`)
- `cooldown`: Seconds between detections (default: `2.0`)
- `pause_threshold`: Silence duration before phrase ends (default: `1.0`)

**Usage:**
```bash
python voice_assistant.py
```

---

### Building Block Scripts

#### `basic_audio_recorder.py`
Simple audio recorder that saves to WAV file.

**Configuration:**
- `sample_rate`: Audio sample rate (default: `16000`)
- `pause_threshold`: Silence before stopping (default: `0.5`)
- `energy_threshold`: Minimum audio energy (default: `300`)

**Usage:**
```bash
python basic_audio_recorder.py
```

---

#### `whisper_transcriber.py`
Records audio and transcribes using Faster Whisper.

**Configuration:**
- `sample_rate`: Audio sample rate (default: `16000`)
- `pause_threshold`: Silence before stopping (default: `0.5`)
- `energy_threshold`: Minimum audio energy (default: `300`)
- `model_size`: Whisper model (default: `base.en`)

**Usage:**
```bash
python whisper_transcriber.py
```

---

#### `regex_wake_detector.py`
Wake word detection using regex pattern matching on transcribed text.

**Configuration:**
- `wake_pattern`: Regex pattern for wake word (e.g., `r"\b(hey|hi)\s+(assistant)\b"`)
- `model_size`: Whisper model (default: `base.en`)
- `pause_threshold`: Silence before processing (default: `1.0`)

**Usage:**
```bash
python regex_wake_detector.py
```

---

#### `openwakeword_detector.py`
Standalone wake word detector using OpenWakeWord models.

**Configuration:**
- `model_path`: Path to .tflite/.onnx model (default: `models/hey_dja_bra.tflite`)
- `threshold`: Detection confidence 0.0-1.0 (default: `0.5`)
- `cooldown`: Seconds between detections (default: `2.0`)
- `sample_rate`: Audio sample rate (default: `16000`)

**Usage:**
```bash
python openwakeword_detector.py
```

---

#### `ollama_voice_assistant.py`
Voice assistant with regex wake word and Ollama chat.

**Configuration:**
- `wake_pattern`: Regex for wake word (default: `r"\b(hey|hi|hello)\s+(assistant|jarvis)\b"`)
- `model_size`: Whisper model (default: `base.en`)
- `ollama_model`: Ollama model (default: `qwen2.5:0.5b`)
- `pause_threshold`: Silence duration (default: `1.0`)

**Requirements:** Ollama must be installed and running

**Usage:**
```bash
python ollama_voice_assistant.py
```

---

#### `wake_transcription_demo.py`
Demo of OpenWakeWord + continuous transcription (no Ollama).

**Configuration:**
- `wake_model_path`: Path to wake word model (default: `models/hey_dja_bra.tflite`)
- `whisper_model`: Whisper model (default: `base.en`)
- `wake_threshold`: Detection sensitivity (default: `0.5`)
- `cooldown`: Seconds between detections (default: `2.0`)
- `pause_threshold`: Silence duration (default: `1.0`)

**Usage:**
```bash
python wake_transcription_demo.py
```

---

### Test Files

- `faster_whisper_test.py`: Test script for Whisper functionality
- `microphone_test.wav`: Sample audio file
- `recording.wav`: Test recording file

---

## 📦 Dependencies

### Required Packages
```bash
pip install speech-recognition
pip install faster-whisper
pip install sounddevice
pip install soundfile
pip install openwakeword
pip install numpy
```

### For Ollama Integration
```bash
pip install ollama
```

---

## 🤖 Ollama Setup

Scripts that use Ollama: `voice_assistant.py`, `ollama_voice_assistant.py`

### Installation

1. **Install Ollama:**
   ```bash
   # macOS
   brew install ollama
   
   # Or download from: https://ollama.ai
   ```

2. **Start Ollama service:**
   ```bash
   ollama serve
   ```

3. **Download a model:**
   ```bash
   # Small fast model (recommended)
   ollama pull qwen2.5:0.5b
   
   # Or other models
   ollama pull modelname
   ```

4. **Verify it works:**
   ```bash
   ollama list
   ```

### Troubleshooting

- **Connection refused**: Make sure `ollama serve` is running
- **Model not found**: Run `ollama pull <model-name>` first

---

## 🎤 Wake Word Models

Custom wake word models are in the `models/` directory:
- `hey_dja_bra.tflite` / `hey_dja_bra.onnx`
- `hello_dja_bra.tflite` / `hello_dja_bra.onnx`

### Creating Custom Wake Words

See the included Jupyter notebook for a complete guide on training your own custom wake word models using OpenWakeWord.

---

## ⚙️ Key Configuration Parameters

### Threshold Values

| Parameter | Range | Purpose | Recommendation |
|-----------|-------|---------|----------------|
| `pause_threshold` | 0.1-3.0s | Silence before phrase ends | 1.0s for natural speech |
| `energy_threshold` | 100-1000 | Minimum audio volume | 300 for normal rooms |
| `wake_threshold` | 0.0-1.0 | Wake word confidence | 0.5 (higher = less false positives) |
| `cooldown` | 0.5-5.0s | Time between wake detections | 2.0s to prevent re-triggers |
| `phrase_time_limit` | 1-10s | Max phrase duration | 5s for commands |

### Model Sizes (Whisper)

- `tiny.en`: Fastest, least accurate (~75MB)
- `base.en`: Good balance ⭐ (~150MB)
- `small.en`: Better accuracy (~500MB)
- `medium.en`: High accuracy, slower (~1.5GB)

---

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install speech-recognition faster-whisper sounddevice soundfile openwakeword numpy ollama
   ```

2. **Setup Ollama (if using AI features):**
   ```bash
   brew install ollama
   ollama serve
   ollama pull qwen2.5:0.5b
   ```

3. **Run the main assistant:**
   ```bash
   python voice_assistant.py
   ```

4. **Say your wake word** (e.g., "Hey Dja Bra")

5. **Ask a question** after the beep

6. **Get AI response** from Ollama

---

## 📁 Directory Structure

```
ML_Insights/
├── voice_assistant.py              # Main assistant (recommended)
├── ollama_voice_assistant.py       # Regex-based wake + Ollama
├── wake_transcription_demo.py      # Wake + transcription demo
├── openwakeword_detector.py        # Pure wake word detection
├── regex_wake_detector.py          # Regex wake detection
├── whisper_transcriber.py          # Audio transcription
├── basic_audio_recorder.py         # Basic audio recorder
├── models/                         # Wake word models (.tflite/.onnx)
├── sound/                          # Notification sounds
│   └── blow.aiff
├── *.wav                          # Test audio files
└── README.md
```

---

## 🔧 Customization Tips

### Change Wake Word
Edit `wake_model_path` to point to your custom model:
```python
assistant = VoiceAssistant(
    wake_model_path="models/your_custom_wake.tflite"
)
```

### Use Different Ollama Model
```python
assistant = VoiceAssistant(
    ollama_model="llama2"  
)
```

### Adjust Sensitivity
```python
assistant = VoiceAssistant(
    wake_threshold=0.7,      # Higher = fewer false triggers
    pause_threshold=1.5      # Wait longer for pauses
)
```

---

## 📝 License

MIT License - Feel free to use and modify for your projects.

---

## 🐛 Troubleshooting

**Microphone not working:**
- Check system permissions for microphone access
- Try `python -m speech_recognition` to test

**Wake word not detected:**
- Lower `wake_threshold` (try 0.3)
- Speak clearly and closer to mic
- Check model file exists

**Ollama errors:**
- Ensure `ollama serve` is running
- Verify model is downloaded: `ollama list`
- Check model name matches exactly

**Audio quality issues:**
- Adjust `energy_threshold` based on room noise
- Increase `pause_threshold` if cut off mid-speech
- Try different Whisper model sizes
