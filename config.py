APP_CSS = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #09090B; color: #FAFAFA; }
    
    .stTextInput>div>div>input {
        background-color: #18181B; color: #FAFAFA; border: 1px solid #27272A;
        border-radius: 8px; padding: 12px 16px; font-size: 1rem; transition: all 0.2s ease;
    }
    .stTextInput>div>div>input:focus { border-color: #6366F1; box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2); }
    
    .stButton>button {
        border-radius: 8px; font-weight: 600; height: 2.8rem; transition: all 0.2s ease;
        border: 1px solid #27272A; background-color: #18181B; color: #E4E4E7;
    }
    .stButton>button:hover { background-color: #27272A; color: #FFF; border-color: #3F3F46; }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white; border: none; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.2);
    }
    .stButton>button[kind="primary"]:hover { transform: translateY(-1px); opacity: 0.95; }

    .media-card { background: #121214; border: 1px solid #27272A; border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    .chapter-card {
        background: #121214; padding: 18px; border-radius: 10px; border: 1px solid #27272A;
        margin-bottom: 12px; transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .chapter-card:hover { border-color: #6366F1; transform: translateX(4px); }
    .timestamp-badge {
        background: rgba(99, 102, 241, 0.1); color: #818CF8; padding: 4px 10px;
        border-radius: 6px; font-size: 0.85em; font-weight: 600; font-family: 'JetBrains Mono', monospace;
        display: inline-block; margin-bottom: 8px;
    }
    .tag-pill {
        background-color: transparent; border: 1px solid #3F3F46; color: #A1A1AA;
        padding: 6px 14px; border-radius: 20px; font-size: 0.85em; font-weight: 500;
        margin-right: 8px; display: inline-block; margin-bottom: 8px; transition: all 0.2s;
    }
    .tag-pill:hover { background-color: #27272A; color: #FAFAFA; border-color: #52525B; }
    
    [data-testid="stMetricValue"] { color: #FAFAFA; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #A1A1AA; font-weight: 500; }
    hr { border-color: #27272A; margin: 2rem 0; }
    </style>
"""

GROQ_WHISPER_MODELS = {
    "large-v3-turbo": "whisper-large-v3-turbo",
    "large-v3": "whisper-large-v3"
}   