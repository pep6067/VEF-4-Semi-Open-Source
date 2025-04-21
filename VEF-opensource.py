#The VEF-4 Engine is an  experimental emotionally intelligent conversational engine that:

# 1. Ingests user input

# 2. Analyzes it using semantic embeddings, emotional modeling, shadow psychology, and archetypes

# 3. Infers hidden layers like repressed needs, dominant emotions, and inner psychological patterns

# 4. Generates context-sensitive emotional replies via Google’s Gemini LLM (v2 Flash) that:

# 5. Mirror emotional states

# 6. Acknowledge repressed needs

# 7. Integrate shadow traits

# 8. Match a personal "soul signature"

# 9. Channel a chosen archetype (e.g., Sage, Rebel)

# Credits to:
# Ian Patel, the teen founder of Anthromorph and Spyder Sync. 



import os
import json
import numpy as np
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
import pytz
from scipy.interpolate import CubicSpline

# For fractal dimension reduction and offline model loading
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# 🌌 Hyperparameters
EMOTION_FRACTAL_DEPTH = 5
SHADOW_TRAIT_DECAY = 0.88
SOUL_SIGNATURE_DYNAMISM = 0.12
TEMPORAL_WARP_STRENGTH = 1.5

class VEF4Engine:
    def _load_shadow_model(self):
        model_name = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        def predict(text: str) -> List[str]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            return ["projection", "denial", "repression"]

        return {
            "model": model,
            "tokenizer": tokenizer,
            "predict": predict
        }

    def __init__(self):
        # 🔥 Core Modules
        GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Please provide a valid API key.")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

        self.embedder = SentenceTransformer("all-mpnet-base-v2")
        self.shadow_model = self._load_shadow_model()
        
        with open("quantum_emotions.json") as f:
            self.emotion_ontology = json.load(f)
        self.emotion_embeddings = self._create_quantum_emotional_fields()
        
        self.circadian_spline = self._init_circadian_rhythm()
        
        self.biometrics = {
            "heart_rate": 72.0,
            "cortisol": 0.35,
            "gut_bias": 0.5,
            "pupil_dilation": 2.3
        }
        
        self.soul_signature = self._generate_soul_signature()
        self.emotional_mind_tree = []
        self.last_10_interactions = []

    def _decompose_input(self, text: str) -> Dict:
        intent = self._query_gemini(
            f"Paraphrase this to reveal true intent: '{text}'",
            temperature=0.3
        )
        shadow = self.shadow_model["predict"](text)

        return {
            "surface_text": text,
            "true_intent": intent,
            "shadow_traits": shadow,
            "repressed_need": self._detect_repressed_need(text)
        }

    def _create_lexical_aura(self, text: str) -> np.ndarray:
        embedding = self.embedder.encode(text)
        fractal_layers = []
        for _ in range(EMOTION_FRACTAL_DEPTH):
            layer = self._apply_quantum_noise(embedding)
            fractal_layers.append(layer)
        return np.stack(fractal_layers)

    def _apply_quantum_noise(self, embedding: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        noise = np.random.normal(0, noise_scale, embedding.shape)
        collapse_mask = np.random.random(embedding.shape) > 0.5
        noise = noise * collapse_mask
        noisy_embedding = embedding + noise
        return noisy_embedding / np.linalg.norm(noisy_embedding)

    def _quantum_superposition_v2(self, aura: np.ndarray) -> Dict[str, float]:
        emotion_weights = {}
        for i, (emotion, meta) in enumerate(self.emotion_ontology.items()):
            total = 0
            for layer in aura:
                sim = cosine_similarity([layer], [self.emotion_embeddings[i]])[0][0]
                total += sim * (meta.get("cultural_weight", 1.0))
            time_of_day = datetime.now(pytz.utc).hour
            circadian_mod = self.circadian_spline(time_of_day)
            weight = (total / EMOTION_FRACTAL_DEPTH) * circadian_mod
            emotion_weights[emotion] = np.clip(weight, 0, 1)
        return emotion_weights

    def _create_quantum_emotional_fields(self) -> np.ndarray:
        emotion_embeddings = []
        for emotion in self.emotion_ontology:
            embedding = self.embedder.encode(emotion)
            emotion_embeddings.append(embedding)
        return np.array(emotion_embeddings)

    def _init_circadian_rhythm(self):
        x = [0, 6, 12, 18, 23]
        y = [0.3, 1.2, 0.7, 1.5, 0.4]
        return CubicSpline(x, y)

    def _update_biometrics(self, emotions: Dict):
        self.biometrics["cortisol"] = np.clip(
            0.3 + 0.6 * (emotions.get("fear", 0) - emotions.get("joy", 0)),
            0, 1
        )
        self.biometrics["gut_bias"] = 0.5 + (0.3 * np.random.randn())

    def _map_archetype_cluster(self, text: str) -> List[Tuple[str, float]]:
        archetypes = ["caregiver", "explorer", "sage", "jester", "rebel", "lover", "creator"]
        embeddings = self.embedder.encode(archetypes)
        text_vec = self.embedder.encode(text)
        sims = cosine_similarity([text_vec], embeddings)[0]
        top_indices = np.argsort(sims)[-5:][::-1]
        return [(archetypes[i], float(sims[i])) for i in top_indices]

    def _detect_repressed_need(self, text: str) -> str:
        needs = ["love", "power", "safety", "freedom", "validation"]
        prompt = f"Classify the repressed need in: '{text}' Options: {needs}"
        return self._query_gemini(prompt, temperature=0.7, max_tokens=10)

    def _generate_soul_signature(self) -> Dict:
        return {
            "color": f"hsl({np.random.randint(0, 360)}, 80%, 50%)",
            "wavelength": np.random.uniform(400, 700),
            "vibe_score": np.random.normal(0.5, 0.2)
        }

    def _construct_quantum_prompt(self, analysis: Dict) -> str:
        return f"""User spoke: "{analysis['surface_text']}"\n
True intent: {analysis['true_intent']}\n
Shadow Traits: {', '.join(analysis['shadow_traits'])}\n
Archetypes: {', '.join([f"{a[0]}({a[1]:.2f})" for a in analysis['archetypes']])}\n
Repressed Need: {analysis['repressed_need']}"""

    def _query_gemini(self, prompt: str, temperature: float = 0.7, max_tokens: int = 100) -> str:
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return response.text.strip()

    def generate_response(self, user_input: str) -> str:
        analysis = self._decompose_input(user_input)
        aura = self._create_lexical_aura(user_input)
        emotions = self._quantum_superposition_v2(aura)
        self._update_biometrics(emotions)
        analysis["archetypes"] = self._map_archetype_cluster(user_input)
        
        # Get top 2 emotions
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2]
        emotion_prompts = [f"{e[0]} ({e[1]:.2f})" for e in top_emotions]
        
        prompt = f"""
        GUIDELINES:
1. Mirror these emotional states: {', '.join(emotion_prompts)}
2. Align with soul signature wavelength: {self.soul_signature['wavelength']:.1f}
3. Address repressed need ({analysis['repressed_need']}) subtly
4. Use archetype {analysis['archetypes'][0][0]} tone
5. Help integrate shadow trait: {analysis['shadow_traits'][0]}

Context: User said "{analysis['surface_text']}"
True intent detected: {analysis['true_intent']}

Respond naturally as a human would, with appropriate emotional resonance. No explanations or formal language.
"""
        return self._query_gemini(prompt, temperature=0.8)

    def chat_loop(self):
        print("VEF4 Engine initialized. Enter 'quit' to exit.")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == 'quit':
                break
            try:
                response = self.generate_response(user_input)
                print("\nVEF5:", response)
            except Exception as e:
                print(f"\nError: {str(e)}")

if __name__ == "__main__":
    engine = VEF4Engine()
    engine.chat_loop()
