"""
Tract | Media Intelligence

This tool allows users to:
1. Download high-fidelity video/audio assets from YouTube/Instagram.
2. Transcribe speech-to-text using Groq Whisper API (Auto-detects Hindi, Japanese, etc.).
3. Analyze content semantics (Summarization, Topic Classification, Keyword Extraction via LLM).
4. Automatically segment videos into semantic chapters using embedding analysis.
"""

import streamlit as st
import yt_dlp
import os
import uuid
import json
import torch
import numpy as np
from datetime import timedelta
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv
import traceback
import logging
import glob

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("Tract")

load_dotenv()

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Tract | Media Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- PREMIUM MODERN CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #09090B;
        color: #FAFAFA;
    }
    
    /* Sleek Inputs */
    .stTextInput>div>div>input {
        background-color: #18181B;
        color: #FAFAFA;
        border: 1px solid #27272A;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    .stTextInput>div>div>input:focus {
        border-color: #6366F1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }
    
    /* Modern Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        height: 2.8rem;
        transition: all 0.2s ease;
        border: 1px solid #27272A;
        background-color: #18181B;
        color: #E4E4E7;
    }
    .stButton>button:hover {
        background-color: #27272A;
        color: #FFF;
        border-color: #3F3F46;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.2);
    }
    .stButton>button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.3);
        opacity: 0.95;
    }

    /* Cards & Components */
    .media-card {
        background: #121214;
        border: 1px solid #27272A;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .chapter-card {
        background: #121214;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #27272A;
        margin-bottom: 12px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .chapter-card:hover {
        border-color: #6366F1;
        transform: translateX(4px);
    }
    .timestamp-badge {
        background: rgba(99, 102, 241, 0.1);
        color: #818CF8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85em;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        display: inline-block;
        margin-bottom: 8px;
    }
    .tag-pill {
        background-color: transparent;
        border: 1px solid #3F3F46;
        color: #A1A1AA;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 500;
        margin-right: 8px;
        display: inline-block;
        margin-bottom: 8px;
        transition: all 0.2s;
    }
    .tag-pill:hover {
        background-color: #27272A;
        color: #FAFAFA;
        border-color: #52525B;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] { color: #FAFAFA; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #A1A1AA; font-weight: 500; }
    
    hr { border-color: #27272A; margin: 2rem 0; }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'work_dir' not in st.session_state: st.session_state.work_dir = str(Path.home() / "Downloads")
if 'ai_results' not in st.session_state: st.session_state.ai_results = {}
if 'media_info' not in st.session_state: st.session_state.media_info = None
if 'active_url' not in st.session_state: st.session_state.active_url = ""

# --- HELPER UTILITIES ---
def format_timestamp(seconds):
    millis = int(round((seconds % 1) * 1000))
    td = timedelta(seconds=int(seconds))
    return f"{str(td).zfill(8)},{millis:03d}"

def generate_srt(segments):
    srt = ""
    for i, seg in enumerate(segments):
        start = format_timestamp(seg['start'])
        end = format_timestamp(seg['end'])
        srt += f"{i+1}\n{start} --> {end}\n{seg['text'].strip()}\n\n"
    return srt

def download_audio_temp(url, work_dir):
    uid = uuid.uuid4().hex
    output_dir = Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path_base = str(output_dir / f"temp_ai_audio_{uid}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': path_base + '.%(ext)s',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '32'}],
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {'youtube': {'player_client': ['default']}}
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        downloaded_files = glob.glob(path_base + ".*")
        if not downloaded_files:
            raise FileNotFoundError("Download succeeded but no audio file was generated.")
        
        final_path = downloaded_files[0]
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        if size_mb > 24.5:
            os.remove(final_path)
            raise ValueError(f"Audio file is {size_mb:.1f} MB — Groq limit is 25 MB. The video is too long.")
        return final_path
    except Exception as e:
        for f in glob.glob(path_base + ".*"):
            try: os.remove(f)
            except: pass
        raise

def _coerce_segment(seg):
    if isinstance(seg, dict): return {"start": float(seg["start"]), "end": float(seg["end"]), "text": str(seg.get("text", ""))}
    return {"start": float(seg.start), "end": float(seg.end), "text": str(seg.text)}

# --- AI ENGINE ---
class AIEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self.GROQ_WHISPER_MODELS = {"large-v3-turbo": "whisper-large-v3-turbo", "large-v3": "whisper-large-v3"}
        self.embedder = None

    def transcribe_audio(self, audio_path, task="transcribe", model_size="large-v3-turbo"):
        if not self.groq_client: raise EnvironmentError("GROQ_API_KEY is missing.")
        groq_model = self.GROQ_WHISPER_MODELS.get(model_size, "whisper-large-v3-turbo")
        groq_task  = "translate" if task == "translate" else "transcribe"

        with open(audio_path, "rb") as audio_file:
            if groq_task == "translate":
                response = self.groq_client.audio.translations.create(model=groq_model, file=audio_file, response_format="verbose_json")
            else:
                response = self.groq_client.audio.transcriptions.create(model=groq_model, file=audio_file, response_format="verbose_json")

        raw_segments = getattr(response, "segments", None) or []
        segments = [_coerce_segment(s) for s in raw_segments if s]

        return {
            "text": getattr(response, "text", ""),
            "language": getattr(response, "language", "en") or "en",
            "segments": segments,
        }

    def load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            if self.embedder is None: self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            return self.embedder
        except ImportError: return None

    def detect_chapters(self, segments):
        if not segments: return []
        try: from sklearn.metrics.pairwise import cosine_similarity
        except ImportError: return []

        embedder = self.load_embedder()
        if not embedder: return []

        blocks = []
        current_block = {"text": "", "start": segments[0]['start'], "end": 0}
        for seg in segments:
            current_block["text"] += " " + seg["text"]
            current_block["end"]   = seg["end"]
            if (seg["end"] - current_block["start"] > 30) or len(current_block["text"]) > 500:
                blocks.append(current_block)
                current_block = {"text": "", "start": seg["end"], "end": 0}

        if current_block["text"]: blocks.append(current_block)
        if len(blocks) < 2: return []

        texts = [b["text"] for b in blocks]
        embeddings = embedder.encode(texts)
        sims = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] for i in range(len(embeddings) - 1)]

        threshold = float(np.mean(sims) - 0.5 * np.std(sims))
        chapters = []
        current_chap_start_idx = 0

        for i in range(len(sims)):
            if sims[i] < threshold and (blocks[i]['end'] - blocks[current_chap_start_idx]['start']) > 60:
                chap_text = " ".join([b['text'] for b in blocks[current_chap_start_idx:i+1]])
                chapters.append({"start": blocks[current_chap_start_idx]['start'], "end": blocks[i]['end'], "title": self.generate_smart_title(chap_text)})
                current_chap_start_idx = i + 1

        final_text  = " ".join([b['text'] for b in blocks[current_chap_start_idx:]])
        chapters.append({"start": blocks[current_chap_start_idx]['start'], "end": blocks[-1]['end'], "title": self.generate_smart_title(final_text)})

        if self.device == "cuda": torch.cuda.empty_cache()
        return chapters

    def generate_smart_title(self, text):
        if not text or len(text.strip()) < 10 or not self.groq_client: return "Untitled Segment"
        try:
            prompt = f"Provide a short chapter title (max 5 words) for the following text. Respond in the exact same language as the text. Do not include quotes.\n\nText: {text[:2000]}"
            completion = self.groq_client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=15)
            return completion.choices[0].message.content.strip().replace('"', '')
        except: return " ".join(text.strip().split()[:5]).title() + "..."

    def classify_topic(self, text):
        if not text or len(text.strip()) < 10 or not self.groq_client: return "Unknown", 0.0
        labels = ["Technology", "Politics", "Entertainment", "Education", "Finance", "Gaming", "Health", "Science", "Other"]
        try:
            prompt = f"Classify this text into exactly ONE of these categories: {', '.join(labels)}. Respond ONLY with the English category name.\n\nText: {text[:3000]}"
            completion = self.groq_client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}], temperature=0.1, max_tokens=10)
            cat = completion.choices[0].message.content.strip().title()
            return cat if cat in labels else "Unknown", 0.95
        except: return "Unknown", 0.0

    def generate_summary(self, text):
        if not text or len(text.strip()) < 50 or not self.groq_client: return "Summary unavailable."
        try:
            prompt = f"Provide a comprehensive, highly accurate summary of the following text. You MUST respond in the exact same language as the text provided below:\n\n{text[:25000]}"
            completion = self.groq_client.chat.completions.create(model="mixtral-8x7b-32768", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=600)
            return completion.choices[0].message.content.strip()
        except: return "Unable to generate summary."

    def extract_keywords(self, text, lang="en"):
        if not text or len(text.strip()) < 20 or not self.groq_client: return []
        try:
            prompt = f"Extract exactly 5 to 8 important keywords from the following text. Return ONLY a comma-separated list. Respond in the exact same language as the text.\n\nText: {text[:5000]}"
            completion = self.groq_client.chat.completions.create(model="llama3-8b-8192", messages=[{"role": "user", "content": prompt}], temperature=0.2, max_tokens=50)
            raw = completion.choices[0].message.content.strip()
            return [k.strip().replace('"', '') for k in raw.split(',') if k.strip()][:8]
        except: return []

@st.cache_resource
def get_engine(): return AIEngine()

engine = get_engine()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("<p style='color: #A1A1AA; font-size: 0.9rem; margin-bottom: 0;'>Output Directory</p>", unsafe_allow_html=True)
    new_dir = st.text_input("Directory", value=st.session_state.work_dir, label_visibility="collapsed")
    if new_dir != st.session_state.work_dir:
        if os.path.exists(new_dir) and os.path.isdir(new_dir): st.session_state.work_dir = new_dir
        else: st.error("Invalid directory path.")

    st.markdown("<br><p style='color: #A1A1AA; font-size: 0.9rem; margin-bottom: 0;'>AI Model Settings</p>", unsafe_allow_html=True)
    model_size = st.selectbox("Whisper Model", ["large-v3-turbo", "large-v3"], index=0, label_visibility="collapsed")
    task_type = st.radio("Processing Task", ["Transcribe Native", "Translate to English"])
    task_code = "translate" if "Translate" in task_type else "transcribe"

    if task_code == "translate" and model_size == "large-v3-turbo":
        st.warning("⚠️ Translation requires the **large-v3** model.")

    st.divider()
    key_color = "#34D399" if os.getenv("GROQ_API_KEY") else "#F87171"
    key_text = "Active" if os.getenv("GROQ_API_KEY") else "Missing"
    st.markdown(f"**Groq API:** <span style='color: {key_color};'>{key_text}</span>", unsafe_allow_html=True)

# --- MAIN APP ---
# Hero Section
st.markdown("""
<div style="text-align: center; margin-top: 2rem; margin-bottom: 3rem;">
    <h1 style="font-size: 4rem; font-weight: 800; letter-spacing: -0.05em; margin-bottom: 0.5rem; background: linear-gradient(135deg, #818CF8 0%, #C084FC 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Tract</h1>
    <p style="color: #A1A1AA; font-size: 1.2rem; font-weight: 400;">Next-generation media intelligence & extraction</p>
</div>
""", unsafe_allow_html=True)

c_left, c_mid, c_right = st.columns([1, 4, 1])
with c_mid:
    url_in = st.text_input("Media URL", placeholder="Paste YouTube or Media URL here...", label_visibility="collapsed")

if url_in:
    if st.session_state.active_url != url_in:
        st.session_state.media_info, st.session_state.ai_results = None, {}
        st.session_state.active_url = url_in
        with st.spinner("Fetching media metadata..."):
            try:
                with yt_dlp.YoutubeDL({'quiet': True, 'extractor_args': {'youtube': {'player_client': ['default']}}}) as ydl:
                    st.session_state.media_info = ydl.extract_info(url_in, download=False)
            except Exception as e:
                st.error(f"Failed to process URL: {e}")

if st.session_state.media_info:
    data = st.session_state.media_info
    
    st.markdown("<div class='media-card'>", unsafe_allow_html=True)
    m1, m2 = st.columns([1.5, 4])
    with m1:
        if data.get('thumbnail'): st.image(data.get('thumbnail'), use_container_width=True, width=300)
    with m2:
        st.markdown(f"<h3 style='margin-top:0; font-weight: 600;'>{data.get('title', 'Untitled Media')}</h3>", unsafe_allow_html=True)
        dur = str(timedelta(seconds=data.get('duration', 0)))
        st.markdown(f"<p style='color: #A1A1AA; font-size: 1.05rem;'>👤 {data.get('uploader', 'Unknown')} &nbsp;•&nbsp; ⏱️ {dur} &nbsp;•&nbsp; 👁️ {data.get('view_count', 0):,} views</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    tab_ai, tab_down, tab_export = st.tabs(["🧠 Neural Analysis", "⬇️ Asset Extractor", "📦 Export"])

    # --- TAB: AI ANALYSIS ---
    with tab_ai:
        if not os.getenv("GROQ_API_KEY"):
            st.error("⚠️ GROQ_API_KEY is missing. Add it to your .env file to run analysis.")
        else:
            cache_key = f"{url_in}_{model_size}_{task_code}"
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn, _ = st.columns([2, 5])
            with col_btn:
                if not st.session_state.ai_results.get(cache_key):
                    if st.button("✨ Generate Intelligence Report", type="primary", use_container_width=True):
                        status = st.status("Initializing AI Pipeline...", expanded=True)
                        audio_path = None
                        try:
                            status.write("📡 Extracting audio streams...")
                            audio_path = download_audio_temp(url_in, st.session_state.work_dir)
                            
                            status.write(f"🎙️ Whisper ({model_size}): Transcribing audio...")
                            res = engine.transcribe_audio(audio_path, task=task_code, model_size=model_size)
                            lang = "en" if task_code == "translate" else res.get("language", "en")
                            
                            status.write("📑 Embedding space: Mapping semantic chapters...")
                            chapters = engine.detect_chapters(res.get("segments", []))
                            
                            status.write(f"📝 LLaMA/Mixtral: Generating insights ({lang.upper()})...")
                            summary = engine.generate_summary(res.get("text", ""))
                            topic, conf = engine.classify_topic(summary)
                            keywords = engine.extract_keywords(res.get("text", ""), lang)

                            st.session_state.ai_results[cache_key] = {
                                "chapters": chapters, "summary": summary, "topic": topic, "conf": conf,
                                "keywords": keywords, "text": res.get("text", ""), 
                                "srt": generate_srt(res.get("segments", [])), "lang": lang
                            }
                            status.update(label="✅ Analysis Complete", state="complete")
                            st.rerun()
                        except Exception as e:
                            status.update(label="❌ Pipeline Error", state="error")
                            st.error(f"Error: {str(e)}")
                        finally:
                            if audio_path and os.path.exists(audio_path):
                                try: os.remove(audio_path)
                                except: pass

            if st.session_state.ai_results.get(cache_key):
                r = st.session_state.ai_results[cache_key]
                st.markdown("<hr style='margin: 1rem 0 2rem 0;'>", unsafe_allow_html=True)
                
                c_chaps, c_insights = st.columns([2.5, 1.5], gap="large")
                
                with c_chaps:
                    st.markdown("<h3 style='margin-bottom: 1.5rem;'>📑 Semantic Chapters</h3>", unsafe_allow_html=True)
                    if r.get('chapters'):
                        for ch in r['chapters']:
                            st.markdown(f"""
                            <div class="chapter-card">
                                <span class="timestamp-badge">{format_timestamp(ch['start'])} - {format_timestamp(ch['end'])}</span>
                                <div style="font-size: 1.1rem; font-weight: 500; margin-top: 4px; color: #FAFAFA;">{ch['title']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    else: st.info("Content too brief or varied to generate distinct chapters.")

                with c_insights:
                    st.markdown("<h3 style='margin-bottom: 1.5rem;'>📊 Analytics</h3>", unsafe_allow_html=True)
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric("Primary Topic", r.get('topic', 'Unknown'))
                    m_col2.metric("Language", r.get('lang', 'en').upper())
                    
                    st.markdown("<br><b>Key Entities & Topics</b><br>", unsafe_allow_html=True)
                    if r.get('keywords'):
                        st.markdown("".join([f"<span class='tag-pill'>{k}</span>" for k in r['keywords']]), unsafe_allow_html=True)
                    else: st.caption("No entities detected.")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.expander("Executive Summary", expanded=True):
                        st.write(r.get('summary', 'No summary available'))

    # --- TAB: DOWNLOADER ---
    with tab_down:
        st.markdown("<br>", unsafe_allow_html=True)
        d1, d2 = st.columns(2, gap="large")

        with d1:
            st.markdown("#### 🎬 Video Extraction")
            formats = data.get('formats', [])
            res_set = {f.get('height') for f in formats if isinstance(f.get('height'), int)}
            if res_set:
                res_choice = st.selectbox("Resolution", sorted(list(res_set), reverse=True), format_func=lambda x: f"{x}p")
                if st.button("Download Video", type="primary", use_container_width=True):
                    pb = st.progress(0)
                    def v_hook(d):
                        if d['status'] == 'downloading':
                            try: pb.progress(min(float(d.get('_percent_str', '0%').replace('%', '')) / 100, 1.0))
                            except: pass
                    try:
                        with yt_dlp.YoutubeDL({'outtmpl': f'{st.session_state.work_dir}/%(title)s.%(ext)s', 'format': f'bestvideo[height<={res_choice}]+bestaudio/best[height<={res_choice}]', 'progress_hooks': [v_hook], 'merge_output_format': 'mp4'}) as ydl:
                            ydl.download([url_in])
                        st.success(f"Saved to {st.session_state.work_dir}")
                    except Exception as e: st.error(str(e))
            else: st.info("No video formats available.")

        with d2:
            st.markdown("#### 🎵 Audio Extraction")
            aud_fmt = st.selectbox("Format", ["mp3", "wav", "flac"])
            if st.button("Download Audio", type="primary", use_container_width=True):
                pb = st.progress(0)
                def a_hook(d):
                    if d['status'] == 'downloading':
                        try: pb.progress(min(float(d.get('_percent_str', '0%').replace('%', '')) / 100, 1.0))
                        except: pass
                try:
                    with yt_dlp.YoutubeDL({'outtmpl': f'{st.session_state.work_dir}/%(title)s.%(ext)s', 'format': 'bestaudio/best', 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': aud_fmt}], 'progress_hooks': [a_hook]}) as ydl:
                        ydl.download([url_in])
                    st.success(f"Saved to {st.session_state.work_dir}")
                except Exception as e: st.error(str(e))

    # --- TAB: EXPORTS ---
    with tab_export:
        st.markdown("<br>", unsafe_allow_html=True)
        r = st.session_state.ai_results.get(f"{url_in}_{model_size}_{task_code}")
        if r:
            e1, e2, e3 = st.columns(3)
            e1.download_button("📦 JSON Metadata", json.dumps(r, indent=4, default=str), "tract_data.json", mime="application/json", use_container_width=True)
            e2.download_button("⏱️ Subtitles (.srt)", r.get('srt', ''), "tract_subs.srt", mime="text/plain", use_container_width=True)
            e3.download_button("📜 Raw Transcript", r.get('text', ''), "tract_transcript.txt", mime="text/plain", use_container_width=True)
        else:
            st.info("Run the AI Analysis to unlock data exports.")
