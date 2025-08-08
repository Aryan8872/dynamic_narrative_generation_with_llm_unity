from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import logging
import requests
from narrative_engine import NarrativeEngine
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Unity integration

# Initialize the narrative engine
narrative_engine = NarrativeEngine()

LM_HOST = "http://localhost:1234"  # LM Studio host

# Game state storage (in production, use a database)
game_sessions = {}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "Narrative Engine API is running"})

@app.route('/start', methods=['POST'])
def start_story():
    """Start a new story session."""
    try:
        payload = request.get_json(silent=True) or {}
        mode = payload.get('mode', 'tlou')
        part = payload.get('part', 1)
        result = narrative_engine.start_new_story(mode=mode, part=part)
        # Convert npc_dialogues dict to array format for Unity
        if isinstance(result.get("npc_dialogues"), dict):
            npc_array = []
            for npc_name, dialogue in result["npc_dialogues"].items():
                npc_array.append({"npc_name": npc_name, "dialogue": dialogue})
            result["npc_dialogues"] = npc_array
        logger.info("New story started")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error starting story: {e}")
        return jsonify({"error": "Failed to start story"}), 500

@app.route('/choice', methods=['POST'])
def make_choice():
    """Process player choice and return story response."""
    try:
        data = request.get_json()
        if not data or 'choice' not in data:
            return jsonify({"error": "Choice is required"}), 400
            
        choice = data['choice']
        npc = data.get('npc', '')
        logger.info(f"Processing choice: {choice}")
        
        result = narrative_engine.generate_story_response(choice, npc)
        # Convert npc_dialogues dict to array format for Unity
        if isinstance(result.get("npc_dialogues"), dict):
            npc_array = []
            for npc_name, dialogue in result["npc_dialogues"].items():
                npc_array.append({"npc_name": npc_name, "dialogue": dialogue})
            result["npc_dialogues"] = npc_array
        # Ensure required keys exist even on fallback
        for key, default in [("story_segment", ""), ("npc_dialogues", []), ("choices", [])]:
            result.setdefault(key, default)
        return jsonify(result)
        
    except ValueError as ve:
        logger.error(f"Value error: {ve}")
        return jsonify({"error": str(ve)}), 500
    except Exception as e:
        logger.error(f"Error processing choice: {e}")
        return jsonify({"error": "Failed to process choice"}), 500

@app.route('/state', methods=['GET'])
def get_state():
    """Get current game state for debugging or UI display."""
    try:
        state = narrative_engine.get_current_state()
        return jsonify(state)
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        return jsonify({"error": "Failed to get state"}), 500

@app.route('/reset', methods=['POST'])
def reset_game():
    """Reset the game to initial state."""
    try:
        result = narrative_engine.start_new_story()
        # Convert npc_dialogues dict to array format for Unity
        if isinstance(result.get("npc_dialogues"), dict):
            npc_array = []
            for npc_name, dialogue in result["npc_dialogues"].items():
                npc_array.append({"npc_name": npc_name, "dialogue": dialogue})
            result["npc_dialogues"] = npc_array
        logger.info("Game reset")
        return jsonify({"message": "Game reset successfully", "initial_state": result})
    except Exception as e:
        logger.error(f"Error resetting game: {e}")
        return jsonify({"error": "Failed to reset game"}), 500

@app.route('/test-lm', methods=['GET'])
def test_lm_connection():
    """Test LM Studio connection."""
    try:
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, LM Studio is working!'"}
        ]
        
        response = narrative_engine.call_lm_studio(test_messages, max_tokens=50)
        return jsonify({
            "status": "success",
            "response": response,
            "message": "LM Studio connection successful"
        })
    except Exception as e:
        logger.error(f"LM Studio test failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "LM Studio connection failed"
        }), 500

@app.route('/lm/models', methods=['GET'])
def lm_models():
    """Fetch available models from LM Studio for diagnostics."""
    try:
        resp = requests.get(f"{LM_HOST}/v1/models", timeout=5)
        return (resp.content, resp.status_code, dict(resp.headers))
    except Exception as e:
        logger.error(f"LM Studio models fetch failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/lm/config', methods=['GET'])
def lm_config():
    """Return current LM Studio configuration used by the server."""
    try:
        return jsonify({
            "lm_host": LM_HOST,
            "lm_studio_url": narrative_engine.lm_studio_url,
            "model_name": getattr(narrative_engine, 'model_name', None)
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def proxy():
    """Forward the Unity request to LM Studio with enhanced error handling."""
    try:
        # Get the request data
        request_data = request.get_json()
        
        if not request_data:
            return jsonify({"error": "No data provided"}), 400
            
        # Forward to LM Studio
        response = requests.post(
            f"{LM_HOST}/v1/chat/completions",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            # Return the response directly to Unity
            return response.content, response.status_code, dict(response.headers)
        else:
            logger.error(f"LM Studio error: {response.status_code} - {response.text}")
            return jsonify({"error": f"LM Studio error: {response.status_code}"}), 500
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error forwarding request to LM Studio: {e}")
        return jsonify({"error": "Failed to connect to LM Studio"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/quest', methods=['POST'])
def generate_quest():
    """Generate a dynamic quest based on current game state."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Extract game state from request
        game_state = data.get('game_state', {})
        quest_context = data.get('quest_context', '')
        
        # Build prompt for quest generation
        system_prompt = """You are a quest generation engine for a fantasy RPG. 
        Generate quests that are engaging and appropriate for the current game state.
        Always respond in valid JSON with keys: quest_title, quest_description, quest_goals, quest_rewards, quest_difficulty."""
        
        user_prompt = f"""Current game state: {json.dumps(game_state)}
        Quest context: {quest_context}
        
        Generate a new quest that fits the current situation. Consider the player's metrics and NPC states when creating the quest."""
        
        # Send to LM Studio
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = narrative_engine.call_lm_studio(messages, max_tokens=300)
        
        try:
            quest_data = json.loads(response)
            return jsonify(quest_data)
        except json.JSONDecodeError:
            logger.error("Failed to parse quest response as JSON")
            return jsonify({"error": "Invalid quest response format"}), 500
            
    except Exception as e:
        logger.error(f"Error generating quest: {e}")
        return jsonify({"error": "Failed to generate quest"}), 500

@app.route('/api/narration', methods=['POST'])
def generate_narration():
    """Generate dynamic narration based on current game state."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Extract game state from request
        game_state = data.get('game_state', {})
        narration_context = data.get('narration_context', '')
        
        # Build prompt for narration generation
        system_prompt = """You are a narration generation engine for a fantasy RPG.
        Generate atmospheric and engaging narration that fits the current game state.
        Always respond in valid JSON with keys: narration_text, mood, audio_event."""
        
        user_prompt = f"""Current game state: {json.dumps(game_state)}
        Narration context: {narration_context}
        
        Generate narration that fits the current situation and mood."""
        
        # Send to LM Studio
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = narrative_engine.call_lm_studio(messages, max_tokens=200)
        
        try:
            narration_data = json.loads(response)
            return jsonify(narration_data)
        except json.JSONDecodeError:
            logger.error("Failed to parse narration response as JSON")
            return jsonify({"error": "Invalid narration response format"}), 500
            
    except Exception as e:
        logger.error(f"Error generating narration: {e}")
        return jsonify({"error": "Failed to generate narration"}), 500

@app.route('/generate_narrative', methods=['POST'])
def generate_narrative_for_unity():
    """Compatibility endpoint for Unity FlaskChatConnector expecting { narrative: string }.
    Accepts either {"prompt": string} or {"choice": string} and returns only narrative text.
    In 'tlou' mode, advances to the next beat regardless of input.
    """
    try:
        data = request.get_json(silent=True) or {}
        user_text = data.get('choice') or data.get('prompt') or ''
        # Generate via engine
        result = narrative_engine.generate_story_response(user_text)
        # Prefer story_segment as narrative
        narrative_text = result.get('story_segment') or ""
        if not narrative_text:
            # Fallback: join npc dialogues as narrative if present
            npc = result.get('npc_dialogues')
            if isinstance(npc, dict):
                narrative_text = "\n".join(f"{k}: {v}" for k, v in npc.items())
            elif isinstance(npc, list):
                narrative_text = "\n".join(npc)
        return jsonify({"narrative": narrative_text})
    except Exception as e:
        logger.error(f"Error generating narrative: {e}")
        return jsonify({"narrative": "The world seems to pause for a moment..."}), 200

@app.route('/api/soulslike/boss', methods=['POST'])
def generate_boss_dialogue():
    """Generate Dark Souls-style boss dialogue based on game state."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Extract game state from request
        game_state = data.get('game_state', {})
        boss_name = data.get('boss_name', 'Unknown Boss')
        context = data.get('context', 'entrance')
        
        # Build prompt for boss dialogue generation
        system_prompt = """You are a Dark Souls-style boss dialogue generator. 
        Generate atmospheric, threatening boss dialogue that fits the dark fantasy theme.
        Always respond in valid JSON with keys: story_segment, boss_dialogue, choices.
        The dialogue should be menacing, atmospheric, and fit the Dark Souls tone."""
        
        user_prompt = f"""Current game state: {json.dumps(game_state)}
        Boss: {boss_name}
        Context: {context}
        
        Generate boss dialogue that fits the current situation and boss personality."""
        
        # Send to LM Studio
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = narrative_engine.call_lm_studio(messages, max_tokens=300)
        
        try:
            boss_data = json.loads(response)
            return jsonify(boss_data)
        except json.JSONDecodeError:
            logger.error("Failed to parse boss response as JSON")
            return jsonify({"error": "Invalid boss response format"}), 500
            
    except Exception as e:
        logger.error(f"Error generating boss dialogue: {e}")
        return jsonify({"error": "Failed to generate boss dialogue"}), 500

@app.route('/api/soulslike/combat', methods=['POST'])
def generate_combat_narrative():
    """Generate Dark Souls-style combat narrative based on game state."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Extract game state from request
        game_state = data.get('game_state', {})
        event_type = data.get('event_type', 'combat')
        result = data.get('result', 'victory')
        
        # Build prompt for combat narrative generation
        system_prompt = """You are a Dark Souls-style combat narrative generator. 
        Generate atmospheric descriptions of combat events and their consequences.
        Always respond in valid JSON with keys: story_segment, combat_description, choices.
        The narrative should be dark, atmospheric, and fit the Dark Souls tone."""
        
        user_prompt = f"""Current game state: {json.dumps(game_state)}
        Event: {event_type}
        Result: {result}
        
        Generate combat narrative that fits the current situation."""
        
        # Send to LM Studio
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = narrative_engine.call_lm_studio(messages, max_tokens=250)
        
        try:
            combat_data = json.loads(response)
            return jsonify(combat_data)
        except json.JSONDecodeError:
            logger.error("Failed to parse combat response as JSON")
            return jsonify({"error": "Invalid combat response format"}), 500
            
    except Exception as e:
        logger.error(f"Error generating combat narrative: {e}")
        return jsonify({"error": "Failed to generate combat narrative"}), 500

@app.route('/api/soulslike/exploration', methods=['POST'])
def generate_exploration_narrative():
    """Generate Dark Souls-style exploration narrative based on game state."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Extract game state from request
        game_state = data.get('game_state', {})
        location = data.get('location', 'Unknown Area')
        area_type = data.get('area_type', 'exploration')
        
        # Build prompt for exploration narrative generation
        system_prompt = """You are a Dark Souls-style environmental narrative generator. 
        Generate atmospheric descriptions of locations and events.
        Always respond in valid JSON with keys: story_segment, environmental_description, choices.
        The narrative should be dark, atmospheric, and fit the Dark Souls tone."""
        
        user_prompt = f"""Current game state: {json.dumps(game_state)}
        Location: {location}
        Area Type: {area_type}
        
        Generate environmental narrative that fits the current location."""
        
        # Send to LM Studio
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = narrative_engine.call_lm_studio(messages, max_tokens=250)
        
        try:
            exploration_data = json.loads(response)
            return jsonify(exploration_data)
        except json.JSONDecodeError:
            logger.error("Failed to parse exploration response as JSON")
            return jsonify({"error": "Invalid exploration response format"}), 500
            
    except Exception as e:
        logger.error(f"Error generating exploration narrative: {e}")
        return jsonify({"error": "Failed to generate exploration narrative"}), 500

if __name__ == '__main__':
    print("Starting Narrative Engine API...")
    print("Make sure LM Studio is running on http://127.0.0.1:1234")
    print("API will be available at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True) 