# Tract — AI Media Analysis Suite

Tract is a Streamlit application that processes long-form media from YouTube, Twitter/X, and other yt-dlp compatible sources. It produces transcripts, structured study notes, and semantic chapter timelines from audio content, and exposes a retrieval-augmented chat interface for querying the transcript directly.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Groq](https://img.shields.io/badge/Powered_by-Groq-f55036.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

### Transcript Q&A (RAG)

After transcription, the full transcript is chunked and embedded locally using `all-MiniLM-L6-v2`. When a question is submitted, cosine similarity is computed between the query embedding and all chunks, and the top four most relevant segments are retrieved and passed to the LLM as context. This keeps token usage bounded and improves answer relevance on long transcripts.

### Summarization

The transcript is passed directly to LLaMA 3.3 70B (up to 25,000 characters) with a structured prompt that produces an executive bullet-point summary. The model is explicitly instructed to respond in the same language as the source content, so non-English media yields summaries in the original language without requiring a separate translation step.

### Semantic Chapter Detection

Rather than splitting by fixed time intervals, Tract segments transcripts by topic shift. Consecutive text blocks are embedded and pairwise cosine similarities are computed. Boundaries are detected where similarity drops below one standard deviation below the mean. Each detected chapter is titled using LLaMA 3.1 8B.

### Topic Classification

The transcript is classified into one of nine topic categories (Technology, Politics, Entertainment, Education, Finance, Gaming, Health, Science, Other) using a zero-shot LLM prompt against LLaMA 3.1 8B. This approach is language-agnostic and requires no local classifier or labelled training data.

### Multilingual and Hinglish Support

Whisper natively handles multilingual transcription. For the Q&A layer, the application detects Hindi and Hinglish queries — via Devanagari script character range and a stop-word overlap check — and bypasses the English-trained semantic router when detected. This prevents the embedding-based routing and cache system from degrading on non-English input while keeping the full LLM response path intact.

### Exports

| Export | Format | Contents |
|--------|--------|----------|
| Study Notes | `.md` | Structured executive summary |
| Transcript | `.txt` | Full verbatim transcript |
| Subtitles | `.srt` | Timestamped subtitle file |

---

## Architecture

Tract uses a hybrid pipeline: local embedding inference for privacy and latency, Groq Cloud for transcription and LLM tasks.

| Pipeline Step | Technology | Execution |
|---------------|------------|-----------|
| Audio Extraction | `yt-dlp` + `ffmpeg` (32 kbps MP3) | Local |
| Speech-to-Text | Whisper (`large-v3-turbo` / `large-v3` / `distil-large-v3`) | Cloud (Groq) |
| Vector Embeddings | `all-MiniLM-L6-v2` via `sentence-transformers` | Local |
| Topic, Keywords, Chapter Titles | `llama-3.1-8b-instant` | Cloud (Groq) |
| Summary and Q&A | `llama-3.3-70b-versatile` | Cloud (Groq) |

Audio is downloaded to a temporary directory, compressed to 32 kbps MP3, and deleted immediately after transcription. The 25 MB Groq API file limit is enforced before upload; files exceeding the limit are rejected with a clear error.

---

## Installation

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.9+ | Core runtime |
| FFmpeg | Required for audio extraction and MP3 compression |
| Groq API Key | Required for transcription and LLM inference. Free tier available at [console.groq.com](https://console.groq.com) |

### Setup

**1. Clone the repository**

```bash
git clone https://github.com/sitanshukr08/Tract.git
cd Tract
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install streamlit yt-dlp groq python-dotenv torch sentence-transformers scikit-learn numpy
```

**4. Install FFmpeg**

```bash
# Windows (Chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg
```

**5. Configure environment variables**

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_api_key_here
```

---

## Usage

**Start the application**

```bash
streamlit run app.py
```

The interface opens at `http://localhost:8501`.

**Workflow**

1. Paste a supported media URL into the input bar and click **Process Media**.
2. The pipeline runs automatically: audio download, compression, transcription, embedding, summarization, topic classification, and chapter detection.
3. Results are displayed across four tabs:
   - **Summary** — Structured study notes, downloadable as Markdown.
   - **Interactive Q&A** — Chat interface for querying the transcript directly.
   - **Timeline** — Expandable semantic chapter list with timestamps.
   - **Transcript** — Full verbatim transcript and SRT subtitle export.

---

## Troubleshooting

### FFmpeg not found

Verify that FFmpeg is installed and on your system PATH:

```bash
ffmpeg -version
```

If the command is not found, reinstall FFmpeg and ensure the installation directory is included in your PATH environment variable.

### Missing or invalid API key

- Confirm the `.env` file is in the same directory as `app.py`.
- Ensure there are no spaces around the `=` sign: `GROQ_API_KEY=gsk_...` not `GROQ_API_KEY = gsk_...`.
- The sidebar shows a **Compute Engine: Online / Offline** indicator reflecting the key's status at startup.

### Groq API rate limits

The free tier of the Groq API enforces Requests Per Minute (RPM) limits. Processing a single video makes several sequential API calls (transcription, summary, topic classification, keyword extraction, and chapter titles). If you hit rate limits, wait 30–60 seconds before retrying, or upgrade to a paid Groq tier.

### Audio file exceeds 25 MB

Groq's Whisper endpoint enforces a 25 MB upload limit. Tract compresses audio to 32 kbps MP3 to minimize file size, but very long recordings (typically over 90 minutes) may still exceed the limit. In this case, trim the source media before processing.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
