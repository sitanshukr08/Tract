# ⚡ Tract | AI Media Analysis Suite

Tract is a Streamlit application that processes long-form media (YouTube videos, Twitter links, audio files) to generate transcripts, structured summaries, and semantic chapters. Powered by the Groq API, it allows you to extract notes and interact with the transcript via a chat interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Groq](https://img.shields.io/badge/Powered_by-Groq-f55036.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## ✨ Features

### 💬 Chat with Video (RAG)
The application holds the generated transcript in memory, allowing you to ask specific questions and extract information directly from the video context using a chat interface.

### 🧠 Summarization for Long Media
To handle multi-hour transcripts without exceeding API token limits, Tract extracts brief introductions from detected chapters to create a condensed outline. This outline is then passed to the LLM to generate the final Markdown study notes.

### 🧮 Topic Classification
Instead of relying solely on an LLM prompt for topic detection, the app uses local `sentence-transformers` (`all-MiniLM-L6-v2`) to compute the cosine similarity between the transcript and predefined category vectors, applying softmax to generate a confidence score.

### 🌍 Multi-Lingual Handling (Hinglish)
The audio transcription layer uses vocabulary seeding to better handle code-switching (e.g., mixing Hindi and English). The LLM is then prompted to translate and synthesize the final summary and chapters in English.

### 🗂️ UI & Exports
- **Embedded Player:** Watch the source media alongside the generated notes.
- **Exports:** Download study notes as Markdown (`.md`), or extract raw Transcripts (`.txt`) and Subtitles (`.srt`).
- **Semantic Chapters:** Video is chunked by topic shifts based on vector distance, rather than fixed time intervals.

---

## 🚀 Architecture & Models

Tract uses a hybrid pipeline combining local processing and cloud inference:

| Pipeline Step | Technology / Model | Execution |
|---------------|--------------------|-----------|
| **Audio Extraction** | `yt-dlp` + `ffmpeg` | Local |
| **Speech-to-Text** | `whisper-large-v3-turbo` | Cloud (Groq) |
| **Vector Embeddings** | `all-MiniLM-L6-v2` | Local |
| **Data Formatting** | `llama-3.1-8b-instant` | Cloud (Groq) |
| **Summary & Chat** | `llama-3.3-70b-versatile` | Cloud (Groq) |

---

## 💻 Installation

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.9+ | Core requirement |
| FFmpeg | **Required** for audio extraction and compression |
| Groq API Key | **Required** for AI analysis (Get one at [console.groq.com](https://console.groq.com)) |

### Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/tract.git
cd tract
```

#### 2. Create a Virtual Environment
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install streamlit yt-dlp groq python-dotenv torch sentence-transformers scikit-learn numpy requests
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
Run the following command in your terminal:
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

### Workflow
1. **Input**: Paste a supported media URL into the main input bar.
2. **Process**: Click "Process Media". The app will download the audio, compress it, and run the transcription/analysis pipeline.
3. **Review**: Check the dashboard tabs:
   - **Summary:** Markdown study notes.
   - **Interactive Q&A:** Chat interface to query the transcript.
   - **Timeline:** Expandable semantic chapters.
4. **Export**: Use the download buttons to save your files.

---

## 🔧 Troubleshooting

### FFmpeg Errors
If audio downloads fail or get stuck:
- Ensure `ffmpeg` is installed and properly added to your system's `PATH`.
- Verify by running `ffmpeg -version` in your terminal.

### Missing API Key
- Check that your `.env` file is in the same directory as `app.py`.
- Ensure there are no spaces around the `=` sign in the `.env` file.

### Rate Limits
If using the free tier of the Groq API, you may hit Requests Per Minute (RPM) limits. The app includes basic retry logic (wait 3 seconds and retry) to help mitigate this.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
