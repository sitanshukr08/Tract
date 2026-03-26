# ⚡ Tract | Media Intelligence

**Next-Generation Media Ingestion & Neural Analysis Suite**

Tract is a powerful Streamlit-based application that combines high-fidelity media extraction with lightning-fast AI analysis. Download video/audio from YouTube or Instagram, transcribe content natively in any language using **Groq Whisper**, and leverage state-of-the-art LLMs (**Mixtral 8x7B & LLaMA 3**) for semantic chapter detection, summarization, and key-entity extraction.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Groq](https://img.shields.io/badge/Powered_by-Groq-f55036.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Architecture & Models](#-architecture--models)
- [💻 Installation](#-installation)
- [📖 Usage](#-usage)
- [⚙️ Configuration](#-configuration)
- [🔧 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

### 🎬 Media Asset Extractor
- **High-Fidelity Video**: Download videos in multiple resolutions (up to 4K) directly to your local drive.
- **Audio Extraction**: Seamlessly rip audio streams into MP3, WAV, or FLAC formats.
- **Safe Extraction**: Uses robust `yt-dlp` backend to bypass standard throttles.

### 🧠 Neural Analysis (Powered by Groq API)
- **Lightning-Fast Transcription**: Uses Groq's `whisper-large-v3` infrastructure. Transcribe hours of audio in seconds.
- **Native Multilingual Support**: Flawlessly understands, transcribes, and analyzes content in Hindi, Japanese, Spanish, English, and more *in their native scripts*.
- **Semantic Chapter Segments**: Uses `sentence-transformers` and cosine similarity to map out dynamic, logical chapters based on topic shifts, rather than arbitrary time codes.
- **LLM-Powered Summarization**: Summarizes massive transcripts using a 32k context window via **Mixtral 8x7B**.
- **Smart Classification & Entities**: Uses **LLaMA 3** for zero-shot topic classification and keyword extraction.

### 💎 Premium UI/UX & Exports
- **Modern Interface**: Designed with a sleek Dark Zinc/Indigo theme, glassmorphism cards, and interactive data pills.
- **Structured Data Exports**: Export your generated intelligence as structured JSON, raw Text transcripts, or timestamped Subtitles (.srt).

---

## 🚀 Architecture & Models

Tract has moved away from heavy local processing to an optimized API/Local hybrid approach:

| Pipeline Step | Technology / Model | Execution |
|---------------|--------------------|-----------|
| **Ingestion** | `yt-dlp` + `ffmpeg` | Local |
| **Speech-to-Text** | Whisper Large V3 / Turbo | Cloud (Groq) |
| **Semantic Routing** | `all-MiniLM-L6-v2` | Local |
| **Summarization** | Mixtral 8x7B (32k context) | Cloud (Groq) |
| **Topic & Keywords** | LLaMA 3 (8B) | Cloud (Groq) |

---

## 💻 Installation

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.8+ | Core requirement |
| FFmpeg | **Required** for audio/video stream merging and mp3 extraction |
| Groq API Key | **Required** for AI analysis (Get one free at [console.groq.com](https://console.groq.com)) |

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/tract.git
cd tract
```

#### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

#### 3. Install Core Dependencies
```bash
pip install streamlit yt-dlp groq python-dotenv torch sentence-transformers scikit-learn numpy yake
```

#### 4. Install FFmpeg
**Windows (Using Chocolatey):**
```bash
choco install ffmpeg
```
**macOS:**
```bash
brew install ffmpeg
```
**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install ffmpeg
```

#### 5. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API Key:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

---

## 📖 Usage

### Start the Application
Launch Tract from your terminal:
```bash
streamlit run tract_app.py
```
The application will open in your default browser at `http://localhost:8501`.

### Workflow
1. **Target**: Paste a YouTube/Media URL into the central search bar.
2. **Configure**: Use the sidebar to set your local Output Directory and choose your Whisper model (`large-v3-turbo` for speed, `large-v3` for complex translations).
3. **Download Assets**: Switch to the **⬇️ Asset Extractor** tab to grab raw MP4 or MP3 files.
4. **Generate Intelligence**: Switch to the **🧠 Neural Analysis** tab and click **Generate Intelligence Report**. Wait a few seconds for Groq to process the media.
5. **Export**: Go to the **📦 Export** tab to download your Subtitles (.srt), Transcripts, or JSON metadata.

---

## ⚙️ Configuration (Sidebar)

| Setting | Description | Default |
|---------|-------------|---------|
| **Directory** | Absolute path where video/audio files will be saved. | `~/Downloads` |
| **Whisper Model** | `large-v3-turbo` (Ultra-fast) vs `large-v3` (Highest accuracy). | `large-v3-turbo` |
| **Processing Task** | `Transcribe Native` (keeps original language) vs `Translate` (forces output to English). | `Transcribe Native` |

---

## 🔧 Troubleshooting

### "FFmpeg Extract Audio Error"
If your audio downloads fail or get stuck on `.webm`/`.m4a` files:
- Ensure `ffmpeg` is installed and added to your system's `PATH`.
- Check installation by running `ffmpeg -version` in your terminal.

### "Groq API Key Missing"
- Ensure your `.env` file is in the exact same folder as `tract_app.py`.
- Ensure there are no spaces around the `=` sign in your `.env` file.

### "Audio file is X MB — Groq limit is 25 MB"
- Groq has a hard limit of 25MB per audio file. Tract attempts to compress audio to `32kbps mp3` automatically. If a video is extremely long (e.g., a 4-hour podcast), it may still exceed 25MB. 

---

## 🤝 Contributing

Contributions are highly welcome! To contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
