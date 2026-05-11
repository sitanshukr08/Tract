"""
metrics.py — Benchmarking & metrics script for Tract (Modular + Dual-LLM)
-----------------------------------------------------------
Run: python metrics.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
"""

import os
import sys
import json
import time
import argparse
import warnings
import tempfile
from pathlib import Path
from datetime import datetime
import importlib.util

warnings.filterwarnings("ignore")

# ── dependency check ─────────────────────────────────────────────────────────
REQUIRED = {
    "groq": "groq",
    "yt_dlp": "yt-dlp",
    "sentence_transformers": "sentence-transformers",
    "sklearn": "scikit-learn",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
}

missing = []
for mod, pkg in REQUIRED.items():
    if importlib.util.find_spec(mod) is None:
        missing.append(pkg)

if missing:
    print(f"[!] Missing packages: {', '.join(missing)}\n    pip install {' '.join(missing)}")
    sys.exit(1)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import from the modular Tract architecture
try:
    from modules.ai_engine import AIEngine
    from modules.media_handler import download_audio_temp
    from modules.utils import coerce_segment
except ImportError:
    print("[!] Could not import Tract modules. Ensure this script is in the project root.")
    sys.exit(1)

# ── helpers ───────────────────────────────────────────────────────────────────
def fmt_time(seconds: float) -> str:
    if seconds < 60: return f"{seconds:.2f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"

def extract_sim_series_for_chart(segments: list, embed_model) -> tuple:
    """Helper purely for generating the visual chart data (AIEngine handles the real logic)."""
    blocks = []
    current_block = {"text": "", "start": segments[0]['start'], "end": 0}
    for seg in segments:
        current_block["text"] += " " + seg["text"]
        current_block["end"] = seg["end"]
        if (seg["end"] - current_block["start"] > 30) or len(current_block["text"]) > 500:
            blocks.append(current_block)
            current_block = {"text": "", "start": seg["end"], "end": 0}
    if current_block["text"]: blocks.append(current_block)
    
    if len(blocks) < 2: return np.array([]), 0.0, []
    
    texts = [b["text"] for b in blocks]
    embeddings = embed_model.encode(texts)
    sims = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] for i in range(len(embeddings) - 1)]
    threshold = float(np.mean(sims) - 0.5 * np.std(sims))
    boundaries = [i for i, s in enumerate(sims) if s < threshold]
    
    return np.array(sims), threshold, boundaries

# ── pipeline stages ───────────────────────────────────────────────────────────
def stage_download(url: str) -> dict:
    print("\n[1/4] Downloading audio (via media_handler)...")
    t0 = time.perf_counter()
    audio_path = download_audio_temp(url)
    elapsed = time.perf_counter() - t0
    
    file_size_mb = os.path.getsize(audio_path) / 1_048_576
    print(f"    ✓ {file_size_mb:.1f} MB downloaded in {fmt_time(elapsed)}")
    
    return {"audio_path": audio_path, "file_size_mb": round(file_size_mb, 2), "download_time_sec": round(elapsed, 2)}

def stage_transcribe(audio_path: str, engine: AIEngine, model: str) -> dict:
    print(f"\n[2/4] Transcribing with {model}...")
    t0 = time.perf_counter()
    res = engine.transcribe_audio(audio_path, model_size=model)
    elapsed = time.perf_counter() - t0
    
    word_count = len(res["text"].split())
    print(f"    ✓ {word_count:,} words in {fmt_time(elapsed)} (Lang: {res['language']})")
    
    return {
        "text": res["text"], "segments": res["segments"], "language": res["language"],
        "word_count": word_count, "transcription_time_sec": round(elapsed, 2)
    }

def stage_chapters(segments: list, engine: AIEngine) -> dict:
    print("\n[3/4] Semantic Chapters (Embeddings + Dual-LLM Title Gen)...")
    t0 = time.perf_counter()
    chapters = engine.detect_chapters(segments)
    elapsed = time.perf_counter() - t0
    
    print(f"    ✓ {len(chapters)} chapters mapped & titled in {fmt_time(elapsed)}")
    return {"chapters_count": len(chapters), "chapters": chapters, "chapter_time_sec": round(elapsed, 2)}

def stage_intelligence(text: str, engine: AIEngine) -> dict:
    print("\n[4/4] Advanced NLP (Dual-LLM Summary, Classification, Extraction)...")
    t0 = time.perf_counter()
    
    summary = engine.generate_refined_summary(text)
    topic, conf = engine.classify_topic(summary)
    keywords = engine.extract_keywords(text)
    
    elapsed = time.perf_counter() - t0
    print(f"    ✓ NLP complete in {fmt_time(elapsed)} | Topic: {topic}")
    return {"topic": topic, "keywords": keywords, "nlp_time_sec": round(elapsed, 2)}

# ── charts ────────────────────────────────────────────────────────────────────
def generate_charts(summary: dict, sim_data: tuple, output_path: str):
    sim, threshold, boundaries = sim_data
    pipeline = summary["pipeline_breakdown"]

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ACCENT, ACCENT2, WARN, TEXT, MUTED, BG, GRID = "#7c6af7", "#38d9a9", "#ff6b6b", "#e2e8f0", "#94a3b8", "#1a1f2e", "#2d3748"

    def style_ax(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors=MUTED, labelcolor=MUTED)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    ax1 = fig.add_subplot(gs[0, :])
    style_ax(ax1)
    if len(sim) > 0:
        x = np.arange(len(sim))
        ax1.plot(x, sim, color=ACCENT, linewidth=1.4, alpha=0.9, label="Cosine similarity")
        ax1.fill_between(x, sim, alpha=0.15, color=ACCENT)
        ax1.axhline(threshold, color=WARN, linewidth=1.2, linestyle="--", label=f"Chapter threshold ({threshold:.3f})")
        for b in boundaries: ax1.axvline(b, color=ACCENT2, linewidth=0.8, alpha=0.6)
        ax1.set_xlabel("Segment Index", color=MUTED, fontsize=10)
        ax1.set_title(f"Embedding Similarity Timeline ({summary['chapters_count']} Chapters)", fontsize=12, fontweight="bold", color=TEXT)
        ax1.legend(framealpha=0, labelcolor=MUTED)

    ax2 = fig.add_subplot(gs[1, 0])
    style_ax(ax2)
    if len(sim) > 0:
        ax2.hist(sim, bins=20, color=ACCENT, alpha=0.75)
        ax2.axvline(np.mean(sim), color=ACCENT2, linewidth=1.5, label=f"Mean: {np.mean(sim):.3f}")
        ax2.axvline(threshold, color=WARN, linewidth=1.5, linestyle="--", label=f"Threshold: {threshold:.3f}")
        ax2.set_xlabel("Cosine Similarity", color=MUTED, fontsize=10)
        ax2.set_title("Semantic Distance Distribution", fontsize=11, fontweight="bold", color=TEXT)
        ax2.legend(framealpha=0, labelcolor=MUTED)

    ax3 = fig.add_subplot(gs[1, 1])
    style_ax(ax3)
    stages = ["Download", "Transcribe", "Semantic Chapters\n(Dual-LLM)", "Summary & NLP\n(Dual-LLM)"]
    times = [pipeline["download_sec"], pipeline["transcribe_sec"], pipeline["chapters_sec"], pipeline["nlp_sec"]]
    bars = ax3.barh(stages, times, color=[ACCENT, ACCENT2, "#f6a623", "#ec4899"], alpha=0.85, height=0.45)
    
    for bar, t in zip(bars, times):
        ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, fmt_time(t), va="center", color=TEXT, fontsize=9)
    ax3.set_xlabel("Seconds", color=MUTED, fontsize=10)
    ax3.set_title("Architecture Processing Latency", fontsize=11, fontweight="bold", color=TEXT)
    ax3.set_xlim(0, max(times) * 1.3)

    fig.suptitle(f"Tract Dual-LLM Pipeline Benchmarks  ·  {summary['word_count']:,} Words  ·  RTF: {summary['rtf']}x", fontsize=13, fontweight="bold", color=TEXT, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

# ── resume bullet generator ───────────────────────────────────────────────────
def print_resume_bullets(s: dict):
    print("\n" + "═" * 65 + "\n  RESUME-READY BULLET POINTS (Dual-LLM Focus)\n" + "═" * 65)
    print(f"""
• Architected a modular media intelligence pipeline processing
  {s['word_count']:,}-word transcripts in {s['total_time']} — achieving an
  RTF of {s['rtf']}x via Groq Whisper Large V3.

• Engineered an Agentic Dual-LLM (Drafter + Critic) workflow
  using Llama-3 and Mixtral-8x7b to eliminate hallucinations and 
  enforce strict UI formatting for summarization and chapter titling.

• Built a semantic routing algorithm using Sentence-Transformers
  and cosine similarity dip detection, autonomously structuring raw 
  transcripts into {s['chapters_count']} accurate, intelligently-titled chapters.
""")
    print("═" * 65)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Tract modular metrics benchmarker")
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    os.environ["GROQ_API_KEY"] = os.environ.get("GROQ_API_KEY") or next((line.split("=", 1)[1].strip().strip('"\'') for line in Path(".env").read_text().splitlines() if line.startswith("GROQ_API_KEY=")), "")
    if not os.environ["GROQ_API_KEY"]: sys.exit("[!] GROQ_API_KEY not found. Set it in .env.")

    engine = AIEngine()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'═'*65}\n  TRACT METRICS BENCHMARK (Modular Architecture)\n  URL   : {args.url}\n  Model : {args.model}\n{'═'*65}")

    t_start = time.perf_counter()
    
    # Run the modular pipeline
    try:
        import yt_dlp
        duration_sec = yt_dlp.YoutubeDL({'quiet': True}).extract_info(args.url, download=False).get('duration', 1)
        
        dl_data = stage_download(args.url)
        tr_data = stage_transcribe(dl_data["audio_path"], engine, args.model)
        ch_data = stage_chapters(tr_data["segments"], engine)
        nlp_data = stage_intelligence(tr_data["text"], engine)
    finally:
        if 'dl_data' in locals() and os.path.exists(dl_data["audio_path"]): os.remove(dl_data["audio_path"])

    total_time = time.perf_counter() - t_start

    # Generate Chart Data
    print("    * Generating chart visuals...")
    sim_data = extract_sim_series_for_chart(tr_data["segments"], engine._load_embedder())

    summary = {
        "word_count": tr_data["word_count"],
        "chapters_count": ch_data["chapters_count"],
        "rtf": round(duration_sec / max(tr_data["transcription_time_sec"], 0.1), 1),
        "total_time": fmt_time(total_time),
        "pipeline_breakdown": {
            "download_sec": dl_data["download_time_sec"],
            "transcribe_sec": tr_data["transcription_time_sec"],
            "chapters_sec": ch_data["chapter_time_sec"],
            "nlp_sec": nlp_data["nlp_time_sec"]
        }
    }

    json_path = os.path.join(args.out_dir, "tract_metrics_report.json")
    with open(json_path, "w") as f: json.dump({"summary": summary}, f, indent=2)
    generate_charts(summary, sim_data, os.path.join(args.out_dir, "tract_metrics_charts.png"))

    print(f"\n[✓] Reports saved to {args.out_dir}")
    print_resume_bullets(summary)

if __name__ == "__main__":
    main()