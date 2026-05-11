import os
import streamlit as st
from dotenv import load_dotenv

from config import APP_CSS
from modules.ai_engine import AIEngine
from modules.media_handler import download_audio_temp
from modules.utils import format_timestamp, generate_srt

load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tract | Media Analysis", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "processed" not in st.session_state:
    st.session_state.processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "source_url" not in st.session_state:
    st.session_state.source_url = ""

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Tract")
    
    # Show Settings ONLY if not processed yet
    if not st.session_state.processed:
        with st.container(border=True):
            st.subheader("Configuration")
            model_choice = st.selectbox("Audio Model", ["large-v3-turbo", "large-v3", "distil-large-v3"], index=0)
            task_choice = st.radio("Processing Task", ["transcribe", "translate"], horizontal=True)

        with st.container(border=True):
            st.subheader("System Status")
            groq_active = bool(os.getenv("GROQ_API_KEY"))
            st.markdown(f"**Compute Engine:** {'Online' if groq_active else 'Offline'}")
    
    # Show Video Player & Reset Button if processed
    if st.session_state.processed:
        st.subheader("Source Media")
        try:
            st.video(st.session_state.source_url)
        except Exception:
            st.info("Video preview not available.")
            
        st.write("---")
        st.button("Analyze New Media", on_click=reset_app, use_container_width=True)

# --- MAIN UI HEADER ---
if not st.session_state.processed:
    st.markdown("<h1 style='text-align: center; color: #7c6af7; font-size: 3.5rem; margin-bottom: 0;'>Tract</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem; margin-bottom: 2rem;'>Transcribe, analyze, and query long-form media.</p>", unsafe_allow_html=True)

engine = AIEngine()

# --- INPUT SECTION (Only show if not processed) ---
if not st.session_state.processed:
    col_spacer1, col_main, col_spacer2 = st.columns([1, 3, 1])
    with col_main:
        with st.container(border=True):
            url_in = st.text_input("Media URL", placeholder="Paste a YouTube, Twitter, or Audio link here...", label_visibility="collapsed")
            process_btn = st.button("Process Media", type="primary", use_container_width=True)

    # --- HERO EMPTY STATE ---
    if not process_btn:
        st.write("---")
        st.write("")
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            st.info("**High-Speed Transcription**\n\nProcess hours of audio in seconds with near-perfect accuracy using optimized Whisper models.")
        with h_col2:
            st.success("**Contextual Summaries**\n\nAutomatically generate structured executive notes, core concepts, and timelines bypassing standard token limits.")
        with h_col3:
            st.warning("**Interactive Querying**\n\nQuery the transcript directly. Ask specific questions and extract exact technical references instantly.")

    # --- PROCESSING PIPELINE ---
    if process_btn:
        if not url_in:
            st.toast("Please enter a valid URL.")
            st.stop()
        if not groq_active:
            st.error("API Key is missing. Please check your configuration.")
            st.stop()
            
        with col_main:
            with st.status("Initializing analysis pipeline...", expanded=True) as status:
                try:
                    st.write("Downloading media assets...")
                    audio_path = download_audio_temp(url_in)
                    
                    st.write("Running transcription model...")
                    ai_result = engine.transcribe_audio(audio_path, task=task_choice, model_size=model_choice)
                    
                    st.session_state.source_url = url_in
                    st.session_state.text = ai_result["text"]
                    st.session_state.segments = ai_result["segments"]
                    st.session_state.language = ai_result.get("language", "en")
                    st.session_state.word_count = len(st.session_state.text.split())
                    
                    st.write("Extracting timeline structures...")
                    st.session_state.chapters = engine.detect_chapters(st.session_state.segments)
                    
                    st.write("Synthesizing executive summary...")
                    st.session_state.summary = engine.generate_refined_summary(st.session_state.text)
                    
                    st.write("Classifying metadata...")
                    st.session_state.topic, st.session_state.conf = engine.classify_topic(st.session_state.text)
                    st.session_state.keywords = engine.extract_keywords(st.session_state.text)
                    
                    st.session_state.chat_history = []
                    st.session_state.processed = True
                    status.update(label="Analysis complete.", state="complete", expanded=False)
                    
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                        
                    st.rerun()
                    
                except Exception as e:
                    status.update(label="Pipeline Failed", state="error", expanded=True)
                    st.error(f"System Error: {str(e)}")
                    st.stop()

# --- UPGRADED RESULTS DASHBOARD ---
if st.session_state.processed:
    st.markdown("<h2 style='color: #7c6af7; margin-bottom: 0;'>Media Insights</h2>", unsafe_allow_html=True)
    
    # Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with st.container(border=True): m_col1.metric(label="Primary Topic", value=st.session_state.topic)
    with st.container(border=True): m_col2.metric(label="Confidence Score", value=f"{st.session_state.conf*100:.1f}%")
    with st.container(border=True): m_col3.metric(label="Word Count", value=f"{st.session_state.word_count:,}")
    with st.container(border=True): m_col4.metric(label="Detected Language", value=st.session_state.language.upper())
    
    # Custom Keywords UI
    if st.session_state.keywords:
        html_keywords = "".join([f"<span style='background-color:#1e293b; color:#e2e8f0; padding:6px 14px; border-radius:20px; margin-right:8px; font-size:0.85rem; border: 1px solid #334155; font-weight: 500;'>{k}</span>" for k in st.session_state.keywords])
        st.markdown(f"<div style='margin-top: 15px; margin-bottom: 25px;'>{html_keywords}</div>", unsafe_allow_html=True)
        
    # Content Tabs - Clean Professional Names
    tab_sum, tab_chat, tab_chap, tab_tx, tab_srt = st.tabs(["Summary", "Interactive Q&A", "Timeline", "Transcript", "Subtitles (SRT)"])
    
    # --- SUMMARY ---
    with tab_sum:
        st.download_button("Export to Markdown (.md)", st.session_state.summary, file_name="tract_summary.md")
        with st.container(border=True):
            st.markdown(st.session_state.summary)
            
    # --- INTERACTIVE Q&A ---
    with tab_chat:
        c_col1, c_col2 = st.columns([5, 1])
        with c_col1: 
            st.markdown("### Document Interface")
            st.caption("Ask specific questions about the transcript, concepts, or references mentioned in the media.")
        with c_col2: 
            if st.button("Clear Conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
                
        st.divider()
        
        # Proper empty state for chat
        if len(st.session_state.chat_history) == 0:
            with st.chat_message("assistant"):
                st.markdown("I have finished analyzing the media context. What specific information would you like to extract?")
                
        # Display chat history natively (Streamlit's default icons are the most professional)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Ask a question regarding this media..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("Scanning context..."):
                    response = engine.answer_question(
                        context_text=st.session_state.text,
                        chat_history=st.session_state.chat_history,
                        question=prompt
                    )
                    st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()

    # --- TIMELINE (CHAPTERS) ---
    with tab_chap:
        if st.session_state.chapters:
            for ch in st.session_state.chapters:
                start_str = format_timestamp(ch['start']).split(',')[0]
                end_str = format_timestamp(ch['end']).split(',')[0]
                with st.expander(f"{start_str} - {end_str}  |  {ch['title']}"):
                    st.write(f"Segment spans from **{start_str}** to **{end_str}**.")
        else:
            st.info("Content insufficient to generate a structured timeline.")
            
    # --- TRANSCRIPT & SRT ---
    with tab_tx:
        st.text_area("Full Transcript", st.session_state.text, height=400, label_visibility="collapsed")
        st.download_button("Download Transcript (.txt)", st.session_state.text, file_name="transcript.txt", use_container_width=True)
        
    with tab_srt:
        srt_data = generate_srt(st.session_state.segments)
        st.text_area("SRT Format", srt_data, height=400, label_visibility="collapsed")
        st.download_button("Download Subtitles (.srt)", srt_data, file_name="subtitles.srt", use_container_width=True)