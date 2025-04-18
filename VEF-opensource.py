import os
import json
import numpy as np
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime, timedelta
import hashlib
from collections import deque
import random
import re

# Hyperparameters
EMOTION_DECAY_RATE = 0.93
TRAUMA_HALFLIFE = 24 * 3600  # 24 hours in seconds
NEUROPLASTICITY_FACTOR = 0.15
EMOTIONAL_MEMORY_SIZE = 100
PHYSIO_UPDATE_INTERVAL = 5

class HumanLikeEmotionEngine:
    def __init__(self):
        # Initialize Gemini API
        gemini_api_key = "Your_GEMINI_Api"  # Replace with your Gemini API key
        genai.configure(api_key=gemini_api_key)
        
        # Set up the Gemini model
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        
        # Human-like traits
        self.name = self._generate_human_name()
        self.age = random.randint(18, 45)
        self.personal_backstory = self._generate_backstory()
        self.idiosyncrasies = self._generate_idiosyncrasies()
        
        # Emotional systems
        self.emotional_memory = deque(maxlen=EMOTIONAL_MEMORY_SIZE)
        self.personality = self._init_personality()
        self.biometrics = self._init_biometrics()
        self.subconscious = {}
        self.conversation_style = "neutral"
        
        # Load human-like emotion mappings
        with open('quantum_emotions.json') as f:
            self.emotion_ontology = json.load(f)
            
        self.emotion_embeddings = self._create_emotion_embeddings()
        self.conversation_styles = self._load_conversation_styles()
        
        # Human imperfections
        self.typos_prob = 0.02
        self.hesitation_prob = 0.15
        self.memory_failures = 0.05
        self.current_mood = "neutral"
    
    def _generate_human_name(self) -> str:
        first_names = ["Alex", "Jamie", "Taylor", "Morgan", "Casey", "Riley", "Jordan"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones"]
        return f"{random.choice(first_names)} {random.choice(last_names)}"
    
    def _generate_backstory(self) -> Dict:
        backgrounds = [
            "grew up in a small town", "was raised in a big city", 
            "comes from a family of artists", "has lived abroad for several years"
        ]
        interests = [
            "photography and hiking", "reading sci-fi novels", 
            "playing musical instruments", "experimental cooking"
        ]
        return {
            "background": random.choice(backgrounds),
            "interests": random.choice(interests),
            "quirks": ["always forgets where they put their keys", "hums when thinking"][random.randint(0,1)]
        }
    
    def _generate_idiosyncrasies(self) -> List[str]:
        return random.sample([
            "uses 'like' occasionally in speech",
            "sometimes trails off mid-thought",
            "has unique laugh patterns",
            "prefers certain words over others",
            "occasionally corrects themselves"
        ], 3)
    
    def _init_personality(self) -> Dict[str, float]:
        """Create a realistically varied personality profile"""
        traits = {
            'openness': random.uniform(0.3, 0.9),
            'conscientiousness': random.uniform(0.2, 0.8),
            'extraversion': random.uniform(0.1, 0.85),
            'agreeableness': random.uniform(0.4, 0.95),
            'neuroticism': random.uniform(0.1, 0.7),
            'curiosity': random.uniform(0.5, 0.9),
            'humor': random.uniform(0.3, 0.8)
        }
        # Ensure some traits correlate like real humans
        traits['empathy'] = (traits['agreeableness'] * 0.8 + random.uniform(0.1, 0.3))
        return traits
    
    def _init_biometrics(self) -> Dict[str, float]:
        """Realistic human physiological ranges"""
        return {
            'heart_rate': random.randint(65, 75),
            'stress': random.uniform(0.1, 0.4),
            'energy': random.uniform(0.6, 0.9),
            'attention': random.uniform(0.7, 1.0)
        }
    
    def _load_conversation_styles(self) -> Dict[str, str]:
        """Different human communication styles"""
        return {
            "warm": "uses more emotive language, shares personal anecdotes",
            "professional": "more structured, avoids slang, precise",
            "friendly": "casual, uses contractions, occasional humor",
            "thoughtful": "slower paced, more reflective, asks questions",
            "enthusiastic": "exclamation points! faster responses! energy!"
        }
    
    def _create_emotion_embeddings(self):
        """Human-like emotional understanding"""
        emotion_texts = []
        for name, meta in self.emotion_ontology.items():
            text = f"{name}: {meta['description']} "
            text += f"Physical sensations: {meta.get('physical', 'varies')} "
            text += f"Common human response: {meta.get('human_response', 'varies')}"
            emotion_texts.append(text)
        return self.embedder.encode(emotion_texts)
    
    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """Human-like emotional interpretation"""
        text_vec = self.embedder.encode(text)
        similarities = cosine_similarity([text_vec], self.emotion_embeddings)[0]
        
        weights = {}
        for i, (emotion, meta) in enumerate(self.emotion_ontology.items()):
            base = 1 / (1 + np.exp(-similarities[i] * 10))
            
            # Personality influences
            if meta['polarity'] == 'positive':
                base *= self.personality['agreeableness']
            elif meta['polarity'] == 'negative':
                base *= self.personality['neuroticism']
            
            # Add some randomness like human perception
            base *= random.uniform(0.9, 1.1)
            weights[emotion] = np.clip(base, 0, 1)
        
        return weights
    
    def _update_conversation_style(self, emotions: Dict[str, float]):
        """Adapt communication style organically"""
        joy_score = emotions.get('joy', 0)
        anger_score = emotions.get('anger', 0)
        
        if joy_score > 0.7:
            self.conversation_style = "enthusiastic"
        elif anger_score > 0.6:
            self.conversation_style = "professional"
        elif self.personality['extraversion'] > 0.7:
            self.conversation_style = random.choice(["warm", "friendly"])
        else:
            self.conversation_style = "thoughtful"
        
        # Occasionally switch styles like humans do
        if random.random() < 0.1:
            self.conversation_style = random.choice(list(self.conversation_styles.keys()))
    
    def _add_human_imperfections(self, text: str) -> str:
        """Make text more human-like with imperfections"""
        words = text.split()
        
        # Occasionally add filler words
        if random.random() < self.hesitation_prob:
            fillers = ["um", "ah", "like", "you know", "well"]
            pos = random.randint(0, len(words)//2)
            words.insert(pos, random.choice(fillers))
        
        # Sometimes repeat words for emphasis
        if random.random() < 0.05 and len(words) > 3:
            pos = random.randint(0, len(words)-1)
            words.insert(pos, words[pos])
        
        # Maybe add a typo
        if random.random() < self.typos_prob and len(words) > 0:
            pos = random.randint(0, len(words)-1)
            if len(words[pos]) > 3:
                typo_pos = random.randint(1, len(words[pos])-1)
                words[pos] = words[pos][:typo_pos] + words[pos][typo_pos+1:]
        
        return ' '.join(words)
    
    def _select_response_style(self) -> str:
        """Choose how to phrase responses based on style"""
        styles = {
            "warm": [
                "I really appreciate you sharing that...",
                "That reminds me of when I...",
                "I can imagine how that must feel..."
            ],
            "professional": [
                "Based on what you've shared...",
                "From my understanding...",
                "That's an important perspective because..."
            ],
            "friendly": [
                "Oh wow, that's so...",
                "Haha, yeah I get what you mean...",
                "No way! That's crazy because..."
            ],
            "thoughtful": [
                "Let me think about that for a second...",
                "There's a few ways to look at this...",
                "What's interesting is..."
            ],
            "enthusiastic": [
                "That's amazing! I love that because...",
                "Wow!! That's so cool!...",
                "Oh my gosh, yes!..."
            ]
        }
        return random.choice(styles[self.conversation_style])
    
    def _generate_human_response(self, prompt: str) -> str:
        """Get response from Gemini with human-like qualities"""
        try:
            # Set up generation config for Gemini
            generation_config = {
                "temperature": random.uniform(0.7, 0.9),
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": random.randint(500, 1500),
            }
            
            # Generate content with Gemini
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Extract the text content from the response
            response_text = response.text
            
            return self._add_human_imperfections(response_text)
        
        except Exception as e:
            # Human-like error recovery
            recovery_phrases = [
                "Sorry, I got distracted for a sec - what was I saying? Oh right...",
                "Wait, let me rephrase that...",
                "I think I lost my train of thought there..."
            ]
            print(f"API error: {str(e)}")  # Add error logging
            return random.choice(recovery_phrases)
    
    def _remember_past_interactions(self, current_emotions: Dict) -> str:
        """Human-like memory recall with imperfections"""
        if not self.emotional_memory or random.random() < self.memory_failures:
            return ""
        
        # Find emotionally similar past interactions
        current_vec = np.array(list(current_emotions.values()))
        similarities = []
        
        for memory in self.emotional_memory:
            past_vec = np.array(list(memory['emotions'].values()))
            if len(past_vec) == len(current_vec):
                sim = cosine_similarity([current_vec], [past_vec])[0][0]
                similarities.append((sim, memory))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Sometimes misremember details like humans do
        if similarities and random.random() > 0.3:
            memory = similarities[0][1]
            recall_accuracy = random.uniform(0.7, 0.95)
            
            if recall_accuracy > 0.8:
                return f" This reminds me of when you mentioned {memory['input'][:50]}..."
            else:
                return " I feel like we've talked about something similar before..."
        return ""
    
    def respond_to(self, user_input: str) -> str:
        """Generate a human-like response to user input"""
        # Analyze emotions with human-like variability
        emotions = self._analyze_emotions(user_input)
        self._update_conversation_style(emotions)
        
        # Create human-like prompt
        memory_context = self._remember_past_interactions(emotions)
        style_guide = self.conversation_styles[self.conversation_style]
        
        prompt = f"""You are {self.name}, a {self.age}-year-old human with these traits:
Personality: {json.dumps(self.personality, indent=2)}
Backstory: {self.personal_backstory}
Idiosyncrasies: {self.idiosyncrasies}

Current conversation style: {style_guide}

User's message appears to contain these emotional tones:
{json.dumps({k: f"{v:.0%}" for k, v in emotions.items() if v > 0.3}, indent=2)}
{memory_context}

Respond naturally as a human would to:
"{user_input}"
"""
        # Generate and format response
        response = self._generate_human_response(prompt)
        
        # Store interaction
        self.emotional_memory.append({
            'timestamp': datetime.now().isoformat(),
            'input': user_input,
            'response': response,
            'emotions': emotions
        })
        
        return response

# Example usage
if __name__ == "__main__":
    print("Creating human-like AI persona...")
    
    try:
        # Add error handling for the JSON file
        if not os.path.exists('quantum_emotions.json'):
            print("ERROR: quantum_emotions.json file not found!")
            print("Creating a basic emotions file for testing...")
            
            # Create a basic emotions file if it doesn't exist
            basic_emotions = {
                "joy": {"description": "Feeling of happiness", "polarity": "positive"},
                "anger": {"description": "Feeling of frustration", "polarity": "negative"},
                "sadness": {"description": "Feeling of sorrow", "polarity": "negative"},
                "fear": {"description": "Feeling of anxiety", "polarity": "negative"},
                "surprise": {"description": "Feeling of astonishment", "polarity": "neutral"}
            }
            
            with open('quantum_emotions.json', 'w') as f:
                json.dump(basic_emotions, f, indent=2)
                
        human_ai = HumanLikeEmotionEngine()
        
        print(f"\nYou're talking to {human_ai.name}, a {human_ai.age}-year-old who {human_ai.personal_backstory['background']}.")
        print(f"They enjoy {human_ai.personal_backstory['interests']} and {human_ai.personal_backstory['quirks']}.\n")
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                    
                print(f"\n{human_ai.name}: ", end='')
                response = human_ai.respond_to(user_input)
                print(response + "\n")
                
            except KeyboardInterrupt:
                print("\n[Ending conversation naturally...]")
                print(f"{human_ai.name}: Anyway, I should get going. It was nice talking with you!")
                break
                
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        import traceback
        traceback.print_exc()
