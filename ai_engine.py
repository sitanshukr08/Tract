import os
import json
import time
import torch
import numpy as np
from groq import Groq
from .utils import coerce_segment
from config import GROQ_WHISPER_MODELS

class AIEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        groq_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_key) if groq_key else None
        self.embedder = None
        
        self.fast_model = "llama-3.1-8b-instant"      
        self.smart_model = "llama-3.3-70b-versatile"  
        self.latest_outline = None 

    def _safe_groq_call(self, model, messages, temperature=0.1, response_format=None, max_tokens=None, retries=2):
        for attempt in range(retries):
            try:
                kwargs = {"model": model, "messages": messages, "temperature": temperature}
                if response_format: kwargs["response_format"] = response_format
                if max_tokens: kwargs["max_tokens"] = max_tokens
                
                res = self.groq_client.chat.completions.create(**kwargs)
                return res.choices[0].message.content
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < retries - 1:
                    time.sleep(3)
                    continue
                raise e

    def transcribe_audio(self, audio_path: str, task="transcribe", model_size="large-v3-turbo") -> dict:
        if not self.groq_client: raise EnvironmentError("GROQ_API_KEY is missing.")
        groq_model = GROQ_WHISPER_MODELS.get(model_size, "whisper-large-v3-turbo")
        
        # HINGLISH & TECH SEEDING:
        # By mixing Hindi and English in the prompt, Whisper is primed to understand code-switching (Hinglish).
        tech_prompt = "Namaste, welcome to the video. Aaj hum RAG, Retrieval-Augmented Generation, LLM, Pydantic, Python, API, OpenAI, Groq, aur Agentic architecture ke baare mein baat karenge. Let's begin."
        
        with open(audio_path, "rb") as audio_file:
            if task == "translate":
                response = self.groq_client.audio.translations.create(
                    model=groq_model, file=audio_file, response_format="verbose_json", prompt=tech_prompt
                )
            else:
                response = self.groq_client.audio.transcriptions.create(
                    model=groq_model, file=audio_file, response_format="verbose_json", prompt=tech_prompt
                )

        text = getattr(response, "text", "").strip()
        if not text: raise ValueError("No speech detected in the audio file.")

        raw_segments = getattr(response, "segments", None) or []
        return {
            "text": text,
            "language": getattr(response, "language", "en") or "en",
            "segments": [coerce_segment(s) for s in raw_segments if s]
        }

    def _load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            if self.embedder is None: 
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            return self.embedder
        except ImportError: return None

    def detect_chapters(self, segments: list) -> list:
        if not segments or not self.groq_client: return []
        embedder = self._load_embedder()
        if not embedder: return []

        blocks = []
        current_block = {"text": "", "start": segments[0]['start'], "end": 0}
        for seg in segments:
            current_block["text"] += " " + seg["text"]
            current_block["end"] = seg["end"]
            if (seg["end"] - current_block["start"] > 30) or len(current_block["text"]) > 500:
                blocks.append(current_block)
                current_block = {"text": "", "start": seg["end"], "end": 0}
        if current_block["text"]: blocks.append(current_block)
        if len(blocks) < 2: return []

        from sklearn.metrics.pairwise import cosine_similarity
        texts = [b["text"] for b in blocks]
        embeddings = embedder.encode(texts)
        sims = [cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0] for i in range(len(embeddings) - 1)]

        threshold = float(np.mean(sims) - 0.5 * np.std(sims))
        chapter_segments = []
        current_idx = 0

        for i, sim in enumerate(sims):
            if sim < threshold and (blocks[i]['end'] - blocks[current_idx]['start']) > 60:
                chap_text = " ".join([b['text'] for b in blocks[current_idx:i+1]])
                chapter_segments.append({"start": blocks[current_idx]['start'], "end": blocks[i]['end'], "text": chap_text})
                current_idx = i + 1

        final_text = " ".join([b['text'] for b in blocks[current_idx:]])
        chapter_segments.append({"start": blocks[current_idx]['start'], "end": blocks[-1]['end'], "text": final_text})

        payload = {}
        for i, ch in enumerate(chapter_segments):
            payload[str(i)] = " ".join(ch["text"].split()[:150])
        
        prompt = f"""You are an elite Technical Architect. I am providing a JSON dictionary of transcript introductions. 
        The transcript may be in Hinglish (Hindi + English).
        
        TASK: Extract a strict, noun-based Title (max 5 words) AND 2 key bullet points for each segment.
        
        CRITICAL RULES:
        1. ALL OUTPUT MUST BE IN STRICT, PROFESSIONAL ENGLISH. Translate any Hindi/Hinglish concepts to English.
        2. NO conversational filler ('In This Video', 'Next We Will').
        
        Return ONLY a valid JSON object with a single key "chapters" containing a list of objects.
        Example:
        {{
            "chapters": [
                {{"id": "0", "title": "If-Else Control Flow", "bullets": ["point 1", "point 2"]}},
                {{"id": "1", "title": "Database Architecture", "bullets": ["point 1", "point 2"]}}
            ]
        }}
        
        Input Data:
        {json.dumps(payload)}
        """

        try:
            res_content = self._safe_groq_call(
                model=self.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            clean_json = res_content.strip()
            if clean_json.startswith("```json"): clean_json = clean_json[7:]
            if clean_json.endswith("```"): clean_json = clean_json[:-3]
            clean_json = clean_json.strip()
            
            generated_data = json.loads(clean_json)
            chapter_list = generated_data.get("chapters", [])
            ch_dict = {str(item.get("id", "")): item for item in chapter_list}
            
            final_chapters = []
            structured_outline = []
            
            for i, ch in enumerate(chapter_segments):
                ch_data = ch_dict.get(str(i), {})
                title = ch_data.get("title", "Core Concept").title()
                bullets = ch_data.get("bullets", ["Key point discussed."])
                
                if any(bad in title.lower() for bad in ["in this", "we will", "going to", "so here"]):
                    title = "Key Topic"
                    
                final_chapters.append({"start": ch["start"], "end": ch["end"], "title": title})
                structured_outline.append(f"Chapter: {title}\n- " + "\n- ".join(bullets))
            
            self.latest_outline = "\n\n".join(structured_outline)
            return final_chapters

        except Exception as e:
            return [{"start": ch["start"], "end": ch["end"], "title": "Section " + str(i+1)} for i, ch in enumerate(chapter_segments)]

    def generate_refined_summary(self, text: str) -> str:
        if not self.groq_client: return "Summary unavailable."
        
        if self.latest_outline:
            analysis_text = f"Here is the detailed structural outline of the video:\n\n{self.latest_outline}"
        else:
            analysis_text = f"Transcript:\n{text[:20000]}"
            
        prompt = f"""You are an elite Technical Writer. Transform the following video outline/transcript into highly structured, professional study notes.
        
        CRITICAL RULES:
        1. The source material may contain Hinglish (Hindi + English mix). You MUST translate and write the final summary entirely in fluent, professional English.
        2. Auto-correct obvious phonetic transcription errors in the text (e.g., change "RAC" to "RAG" if discussing AI).
        
        Format your response exactly like this Markdown template:

        ### 📖 Executive Overview
        (A clear 2-sentence summary of the entire content in pure English)

        ### 🔑 Core Concepts
        - **[Specific Noun/Concept 1]**: [Direct, technical English definition]
        - **[Specific Noun/Concept 2]**: [Direct, technical English definition]

        ### 🚀 Actionable Takeaways
        - (Practical applications, step-by-step instructions, or conclusions in English)

        Input Data:
        {analysis_text}
        """

        try:
            return self._safe_groq_call(
                model=self.smart_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            ).strip()
        except Exception as e:
            return f"Unable to generate structured notes. Details: {str(e)}"

    def classify_topic(self, text: str) -> tuple:
        """Uses Cosine Similarity + Softmax Probability to classify the topic."""
        if not text or len(text.strip()) < 10: return "Unknown", 0.0
        
        embedder = self._load_embedder()
        if not embedder: return "Unknown", 0.0

        label_definitions = {
            "Software Development": "Software engineering, coding, programming, web development, and computer science.",
            "Artificial Intelligence": "Artificial intelligence, machine learning, deep learning, neural networks, and LLMs.",
            "Business & Finance": "Business, finance, stock market, economics, startups, and entrepreneurship.",
            "Education": "Education, teaching, studying, academia, courses, and learning methodologies.",
            "Entertainment": "Entertainment, movies, gaming, pop culture, comedy, and media.",
            "Science": "Science, physics, space, biology, chemistry, and research.",
            "Health": "Health, fitness, medicine, nutrition, wellness, and biology."
        }
        
        category_names = list(label_definitions.keys())
        category_descriptions = list(label_definitions.values())

        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            label_embeddings = embedder.encode(category_descriptions)
            text_chunk = " ".join(text.split()[:1000])
            text_embedding = embedder.encode([text_chunk])
            
            # 1. Get raw mathematical distance
            sims = cosine_similarity(text_embedding, label_embeddings)[0]
            
            # 2. THE FIX: Apply mathematical Softmax with Temperature Scaling
            # This converts raw vector distance into a standard 0-100% Probability Distribution
            temperature = 15  # Sharpens the contrast between the winner and losers
            exp_sims = np.exp(sims * temperature)
            probabilities = exp_sims / np.sum(exp_sims)
            
            # 3. Find the highest probability category
            max_idx = int(np.argmax(probabilities))
            confidence_score = float(probabilities[max_idx])
            
            # If the probability is still too low/even across the board, it's "Other"
            if confidence_score < 0.30:
                return "Other", confidence_score
                
            return category_names[max_idx], confidence_score

        except Exception as e:
            print(f"Mathematical classification error: {e}")
            return "Unknown", 0.0

    def extract_keywords(self, text: str, lang="en") -> list:
        if not text or len(text.strip()) < 20 or not self.groq_client: return []
        
        prompt = f"""Extract 5 to 8 highly specific, technical keywords from this text.
        The text may be in Hinglish, but the extracted keywords MUST be in English.
        Return a valid JSON object with a single key 'keywords' containing a list of strings.
        Text: {text[:4000]}"""
        
        try:
            res_content = self._safe_groq_call(
                model=self.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            data = json.loads(res_content)
            return data.get("keywords", [])[:8]
        except Exception: 
            return []

    def answer_question(self, context_text: str, chat_history: list, question: str) -> str:
        """Interactive RAG: Answers questions based on the video context."""
        if not self.groq_client: return "API missing."
        
        safe_context = context_text[:60000] 
        
        system_prompt = f"""You are an expert AI assistant answering questions based on the provided video transcript.
        CRITICAL: The transcript may contain Hinglish (Hindi + English mix). You MUST translate your thoughts and provide the final answer entirely in clear, professional English.
        Be concise, accurate, and directly address the question. 
        If the answer is not in the transcript, say "I cannot find this in the media context."
        
        TRANSCRIPT/CONTEXT:
        {safe_context}
        """
        
        messages = [{"role": "system", "content": system_prompt}]
        for msg in chat_history[-4:]: messages.append(msg)
        messages.append({"role": "user", "content": question})
        
        try:
            return self._safe_groq_call(
                model=self.smart_model,
                messages=messages,
                temperature=0.2,
                max_tokens=1000
            )
        except Exception as e:
            return f"Error connecting to chat: {str(e)}"