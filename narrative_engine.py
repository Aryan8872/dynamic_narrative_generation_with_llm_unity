import json
import os
import re
import requests
import time
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NarrativeEngine:
    def __init__(self, lm_studio_url: str = "http://127.0.0.1:1234/v1/chat/completions", model_name: str | None = None):
        """
        Initialize the narrative engine with LM Studio integration.
        
        Args:
            lm_studio_url: URL for LM Studio API (default: localhost:1234)
        """
        self.lm_studio_url = lm_studio_url
        # Model name is required by LM Studio's OpenAI-compatible API; default to env or 'local'
        self.model_name = model_name or os.getenv("LMSTUDIO_MODEL", "local")
        self.story_context = ""
        self.player_metrics = {
            "kindness": 0,
            "aggression": 0,
            "honesty": 0
        }
        self.npc_states = {
            "Ellie": {"trust": 0.8, "anger": 0.1, "fear": 0.2},
            "Tess": {"trust": 0.6, "anger": 0.2, "fear": 0.1},
            "Marlene": {"trust": 0.3, "anger": 0.4, "fear": 0.0}
        }
        self.memory_summary = ""
        self.story_history = []
        self.interaction_count = 0
        # Narrative control (sequential or LLM)
        self.story_mode = "llm"  # 'llm' or 'tlou'
        self.tlou_part = 1       # 1 or 2
        self.story_beats: List[str] = []
        self.current_beat_index: int = 0
        
        # Content filtering for ethical responses
        self.inappropriate_keywords = [
            'vagina', 'penis', 'sex', 'sexual', 'nude', 'naked', 'lick', 'suck', 'fuck', 'porn',
            'breast', 'ass', 'butt', 'dick', 'cock', 'pussy', 'clit', 'orgasm', 'ejaculate',
            'masturbate', 'foreplay', 'intimate', 'erotic', 'seduce', 'flirt', 'kiss', 'touch',
            'caress', 'fondle', 'grope', 'penetrate', 'thrust', 'moan', 'scream', 'pleasure'
        ]
        # Game objective and NPC role remapping
        self.objective_text = "Find the castle password from NPCs and get past the final gate guard."
        self.npc_remap: Dict[str, str] = {
            "rock guard": "Checkpoint Sentry",
            "castle guard": "Gate Sentry",
            "shopkeeper": "Scavenger Trader",
            "depressed villager": "Worn Survivor",
            "knight in training farmer jitt": "Rookie Runner"
        }
        
    def call_lm_studio(self, messages: List[Dict[str, str]], max_tokens: int = 500) -> str:
        """
        Call LM Studio API with the DeepSeek model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "stream": False,
                "stop": ["</think>"]
            }
            
            response = requests.post(
                self.lm_studio_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"LM Studio API error: {response.status_code} - {response.text}")
                return "The world seems to pause for a moment..."
                
        except Exception as e:
            logger.error(f"Error calling LM Studio: {e}")
            return "The world seems to pause for a moment..."
    
    def update_player_metrics(self, choice: str):
        """Update player metrics based on their choice."""
        choice_lower = choice.lower()
        
        # Kindness metrics
        if any(word in choice_lower for word in ["help", "save", "protect", "kind", "gentle", "care"]):
            self.player_metrics["kindness"] += 1
        elif any(word in choice_lower for word in ["ignore", "abandon", "hurt", "cruel"]):
            self.player_metrics["kindness"] -= 1
            
        # Aggression metrics
        if any(word in choice_lower for word in ["fight", "attack", "kill", "destroy", "violent", "force"]):
            self.player_metrics["aggression"] += 1
        elif any(word in choice_lower for word in ["peace", "calm", "avoid", "retreat"]):
            self.player_metrics["aggression"] -= 1
            
        # Honesty metrics
        if any(word in choice_lower for word in ["truth", "honest", "confess", "admit"]):
            self.player_metrics["honesty"] += 1
        elif any(word in choice_lower for word in ["lie", "deceive", "hide", "bribe", "cheat"]):
            self.player_metrics["honesty"] -= 1
            
        # Keep metrics in reasonable bounds
        for metric in self.player_metrics:
            self.player_metrics[metric] = max(-5, min(5, self.player_metrics[metric]))
    
    def update_npc_states(self, choice: str, story_segment: str):
        """Update NPC states based on player choice and story outcome."""
        choice_lower = choice.lower()
        story_lower = story_segment.lower()
        
        # Update Ellie's states
        if "ellie" in choice_lower or "ellie" in story_lower:
            if any(word in choice_lower for word in ["protect", "save", "help", "care"]):
                self.npc_states["Ellie"]["trust"] += 0.1
                self.npc_states["Ellie"]["fear"] -= 0.05
            elif any(word in choice_lower for word in ["abandon", "ignore", "hurt"]):
                self.npc_states["Ellie"]["trust"] -= 0.2
                self.npc_states["Ellie"]["fear"] += 0.2
                self.npc_states["Ellie"]["anger"] += 0.1
                
        # Update Tess's states
        if "tess" in choice_lower or "tess" in story_lower:
            if any(word in choice_lower for word in ["cooperate", "help", "trust"]):
                self.npc_states["Tess"]["trust"] += 0.1
                self.npc_states["Tess"]["anger"] -= 0.05
            elif any(word in choice_lower for word in ["betray", "attack", "lie"]):
                self.npc_states["Tess"]["trust"] -= 0.2
                self.npc_states["Tess"]["anger"] += 0.2
                
        # Update Marlene's states
        if "marlene" in choice_lower or "marlene" in story_lower:
            if any(word in choice_lower for word in ["cooperate", "hand over", "trust"]):
                self.npc_states["Marlene"]["trust"] += 0.1
                self.npc_states["Marlene"]["anger"] -= 0.1
            elif any(word in choice_lower for word in ["resist", "fight", "refuse"]):
                self.npc_states["Marlene"]["trust"] -= 0.3
                self.npc_states["Marlene"]["anger"] += 0.2
                
        # Ensure all values stay in [0, 1] range
        for npc in self.npc_states:
            for emotion in self.npc_states[npc]:
                self.npc_states[npc][emotion] = max(0.0, min(1.0, self.npc_states[npc][emotion]))
    
    # Implement short-term vs long-term memory management
    def summarize_memory(self):
        """Summarize the story to maintain long-term context."""
        if len(self.story_history) > 3:
            self.memory_summary = " ".join(self.story_history[-3:])
            self.story_history = self.story_history[-3:]
        else:
            self.memory_summary = " ".join(self.story_history)

    def construct_prompt(self, choice: str, npc: str = "") -> Dict[str, Any]:
        """Construct the prompt for the LLM based on the current game state."""
        system_prompt = (
            "You are a story generator for The Last of Us universe. You are Joel, a hardened survivor in a post-apocalyptic world overrun by infected. "
            "You're escorting Ellie, a 14-year-old girl who may hold the key to a cure. The story takes place in the ruins of Boston and beyond. "
            "IMPORTANT: Keep all content family-friendly and appropriate. Avoid any sexual, vulgar, or inappropriate content. "
            "Return ONLY a single JSON object with no extra text, no explanations, no code fences, and no markdown formatting. "
            "The JSON schema is: {\n"
            "  \"story_segment\": string,\n"
            "  \"npc_dialogues\": object mapping NPC name to a single line string,\n"
            "  \"choices\": array of 3-4 strings\n"
            "}."
        )
        # Resolve NPC role if provided
        npc_key = (npc or "").strip().lower()
        npc_role = self.npc_remap.get(npc_key, npc.title() if npc else "")
        user_prompt = f"Current story: {self.story_context}\n"
        user_prompt += f"Player metrics: kindness={self.player_metrics['kindness']}, aggression={self.player_metrics['aggression']}, honesty={self.player_metrics['honesty']}\n"
        user_prompt += "NPC states: " + ", ".join([f"{npc}: trust={state['trust']}, anger={state['anger']}, fear={state['fear']}" for npc, state in self.npc_states.items()]) + "\n"
        user_prompt += f"Player chose: \"{choice}\"\n"
        # Inject current objective and NPC interaction context
        if npc_role:
            user_prompt += f"Interacting with: {npc_role}.\n"
            user_prompt += f"Objective: {self.objective_text}.\n"
            user_prompt += (
                "When producing npc_dialogues, keep lines short and in character: \n"
                "- 'Worn Survivor': subdued, weary tone.\n"
                "- 'Scavenger Trader': pragmatic, transactional.\n"
                "- 'Checkpoint Sentry': brusque, by-the-book.\n"
                "- 'Gate Sentry': firm, suspicious.\n"
                "- 'Rookie Runner': eager, unsure.\n"
            )
            user_prompt += (
                "Choices must be 3-4 actionable steps that progress the objective (e.g., barter for hints, eavesdrop, search records, persuade a sentry)."
            )
            user_prompt += f"Generate the NPC's response to the player's request and next choices in JSON format."
        else:
            user_prompt += f"Objective: {self.objective_text}.\n"
            user_prompt += "Generate the next story segment in The Last of Us setting, NPC dialogues (Ellie, Tess, Marlene, or other survivors), and next choices in JSON format."
        return {"system": system_prompt, "user": user_prompt}
    
    def _is_inappropriate_content(self, text: str) -> bool:
        """Check if the text contains inappropriate content."""
        if not text:
            return False
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.inappropriate_keywords)
    
    def _filter_inappropriate_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out inappropriate content from the response."""
        if not response_data:
            return response_data
            
        # Check story segment
        if 'story_segment' in response_data and self._is_inappropriate_content(response_data['story_segment']):
            response_data['story_segment'] = "The NPC looks uncomfortable and refuses to engage with inappropriate behavior."
            
        # Check NPC dialogues
        if 'npc_dialogues' in response_data and isinstance(response_data['npc_dialogues'], dict):
            filtered_dialogues = {}
            for npc, dialogue in response_data['npc_dialogues'].items():
                if self._is_inappropriate_content(dialogue):
                    filtered_dialogues[npc] = "I don't appreciate that kind of talk. Please be respectful."
                else:
                    filtered_dialogues[npc] = dialogue
            response_data['npc_dialogues'] = filtered_dialogues
            
        # Check choices
        if 'choices' in response_data and isinstance(response_data['choices'], list):
            filtered_choices = []
            for choice in response_data['choices']:
                if self._is_inappropriate_content(choice):
                    filtered_choices.append("Apologize for inappropriate behavior")
                else:
                    filtered_choices.append(choice)
            response_data['choices'] = filtered_choices
            
        return response_data

    def generate_story_response(self, choice: str, npc: str = "") -> Dict[str, Any]:
        """Generate the story response based on player choice."""
        # Sequential mode: return next beat and no dialogues/choices
        if self.story_mode == "tlou":
            if not self.story_beats:
                self._load_tlou_beats()
            if self.current_beat_index < len(self.story_beats):
                segment = self.story_beats[self.current_beat_index]
                self.current_beat_index += 1
            else:
                segment = "The journey continues beyond this chapter..."
            self.story_context += (segment + "\n")
            self.story_history.append(segment)
            self.interaction_count += 1
            if self.interaction_count % 4 == 0:
                self.summarize_memory()
            return {"story_segment": segment, "npc_dialogues": {}, "choices": []}

        # LLM mode
        # Check if player input is inappropriate
        if self._is_inappropriate_content(choice):
            logger.warning(f"Inappropriate content detected in player input: {choice}")
            return {
                "story_segment": "The NPC looks uncomfortable and refuses to engage with inappropriate behavior.",
                "npc_dialogues": {"NPC": "I don't appreciate that kind of talk. Please be respectful."},
                "choices": ["Apologize for inappropriate behavior", "Leave the area", "Change the subject"]
            }
            
        prompt = self.construct_prompt(choice, npc)
        raw_response = self.call_lm_studio([
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": prompt['user']}
        ], max_tokens=200)
        response = self._strip_think_blocks(raw_response)
        # Parse the JSON response more robustly (extract first JSON object if extra text)
        try:
            try:
                response_data = json.loads(response)
            except Exception:
                start = response.find('{')
                response_data = None
                if start != -1:
                    depth = 0
                    end = -1
                    for i, ch in enumerate(response[start:], start):
                        if ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    if end != -1:
                        try:
                            response_data = json.loads(response[start:end])
                        except Exception:
                            response_data = None
                if response_data is None:
                    # Fallback: wrap cleaned raw text into expected shape
                    cleaned = (response or "The world seems to pause for a moment...").strip()
                    response_data = {
                        "story_segment": cleaned,
                        "npc_dialogues": {},
                        "choices": []
                    }
            # Filter inappropriate content from response
            response_data = self._filter_inappropriate_response(response_data)
            
            self.story_context += response_data['story_segment']
            self.story_history.append(response_data['story_segment'])
            for npc, dialogue in response_data['npc_dialogues'].items():
                # Update NPC states based on dialogue
                if "angry" in dialogue:
                    self.npc_states[npc]['anger'] += 0.1
                # Add more rules as needed
            # Summarize memory periodically
            self.interaction_count += 1
            if self.interaction_count % 4 == 0:
                self.summarize_memory()
            return response_data
        except Exception:
            logger.error("Failed to parse JSON response from LLM.")
            return {"story_segment": "The world seems to pause for a moment...", "npc_dialogues": {}, "choices": []}

    def _strip_think_blocks(self, text: str) -> str:
        """Remove <think>...</think> blocks or leading chain-of-thought from the model output."""
        if not isinstance(text, str):
            return text
        # Remove any complete <think>...</think> blocks
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
        # Remove markdown code blocks
        cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.IGNORECASE)
        # If output still starts with <think> without closing, drop until first '{' or double newline
        if cleaned.lstrip().lower().startswith("<think>"):
            # Try to cut to first JSON object start
            brace_pos = cleaned.find('{')
            if brace_pos != -1:
                cleaned = cleaned[brace_pos:]
            else:
                # Fallback: cut after the first blank line
                parts = cleaned.split("\n\n", 1)
                cleaned = parts[1] if len(parts) > 1 else ""
        return cleaned
    
    def start_new_story(self, mode: str = "tlou", part: int = 1) -> Dict[str, Any]:
        """Start a new story session.
        mode: 'tlou' for sequential beats, 'llm' for generative
        part: 1 or 2 when mode == 'tlou'
        """
        self.story_mode = mode or "tlou"
        self.tlou_part = part if part in (1, 2) else 1
        self.story_context = ""
        self.player_metrics = {"kindness": 0, "aggression": 0, "honesty": 0}
        # Default NPCs for The Last of Us universe
        self.npc_states = {
            "Ellie": {"trust": 0.8, "anger": 0.1, "fear": 0.2},
            "Tess": {"trust": 0.6, "anger": 0.2, "fear": 0.1},
            "Marlene": {"trust": 0.3, "anger": 0.4, "fear": 0.0}
        }
        self.memory_summary = ""
        self.story_history = []
        self.interaction_count = 0
        self.story_beats = []
        self.current_beat_index = 0
        
        if self.story_mode == "tlou":
            self._load_tlou_beats()
            first_segment = self.story_beats[0] if self.story_beats else "The story begins..."
            self.story_context = first_segment + "\n"
            self.story_history.append(first_segment)
            return {"story_segment": first_segment, "npc_dialogues": {}, "choices": []}
        else:
            # LLM mode: start with a brief setup
            setup = (
                "You are Joel, a survivor escorting Ellie through a hostile world. Supplies are scarce and dangers are near."
            )
            self.story_context = setup
            self.story_history.append(setup)
            return {"story_segment": setup, "npc_dialogues": {}, "choices": []}

    def _load_tlou_beats(self) -> None:
        """Load a concise sequence of story beats for The Last of Us parts 1 or 2.
        These are brief summaries intended to drive sequential narration, not verbatim text.
        """
        if self.tlou_part == 1:
            self.story_beats = [
                "Outbreak day shatters normal life; in the chaos, Joel's world is irreversibly changed.",
                "Years later, quarantine zones and smuggling work define survival in a broken world.",
                "Joel is tasked to escort Ellie, a girl with a rare immunity, to distant allies.",
                "Leaving the city, they navigate ruins, evading patrols and the infected alike.",
                "Across towns and wastelands, bonds form under constant threat and hard choices.",
                "Reaching a supposed safe haven, grim truths test purpose and loyalty.",
                "A final act of defiance sets a quiet future on an uncertain path."
            ]
        else:
            self.story_beats = [
                "A fragile peace is broken, and the cost of past choices returns to demand payment.",
                "Driven by resolve and grief, a journey begins toward a distant, hostile city.",
                "New allies and enemies blur lines as survival and justice collide.",
                "Retribution spirals; every step forward asks what it truly means to be right.",
                "At the water's edge, mercy and memory weigh heavier than any blade.",
                "What remains is not victory, but the chance to live with what was lost and learned."
            ]
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current game state for debugging or UI display."""
        return {
            "story_context": self.story_context,
            "player_metrics": self.player_metrics,
            "npc_states": self.npc_states,
            "memory_summary": self.memory_summary,
            "interaction_count": self.interaction_count
        } 