# Adaptive Story Arc & NPC Relationship Engine

A dynamic narrative engine that uses LLMs to generate branching stories with persistent NPC relationships and moral consequences. Built for easy integration with Unity and optimized for local model usage with LM Studio.

> Note: This repository is an active prototype for research/education. Interfaces and prompts may evolve.

## Features

- **Dynamic Story Generation**: LLM-powered branching narratives that adapt to player choices
- **NPC Relationship System**: Persistent emotional states (trust, anger, fear) that influence dialogue and story branches
- **Moral Metrics Tracking**: Player behavior affects story outcomes and NPC reactions
- **Memory Management**: Long-term story consistency through summarization
- **Unity Integration**: Ready-to-use C# scripts for game integration
- **Local Model Support**: Optimized for LM Studio with DeepSeek models
- **RESTful API**: Clean JSON-based communication

## System Architecture

```
┌──────────────────────┐      HTTP (JSON)       ┌──────────────────────────┐
│  Unity Game Client   │  ───────────────────▶  │   Flask API Server       │
│  (FlaskDialogueConn) │                         │   (app.py)               │
└──────────────────────┘                         └─────────────┬────────────┘
                                                             calls
                                              ┌────────────────┴────────────────┐
                                              │     NarrativeEngine (Python)    │
                                              │  state, memory, prompt builder  │
                                              └────────────────┬────────────────┘
                                                             HTTP (OpenAI compat)
                                              ┌────────────────┴────────────────┐
                                              │         LM Studio Server        │
                                              │   model router / host runtime   │
                                              └───────────┬───────────┬─────────┘
                                                          │           │
                                                (DeepSeek family)   (Google Gemma)
```

- The backend talks to LM Studio via the OpenAI-compatible `/v1/chat/completions` API.
- You can run multiple models in LM Studio (e.g., DeepSeek + Google Gemma) and select which one to use at runtime.

### Selecting the model (LM Studio)

- Set the environment variable before starting the API:

```powershell
# Windows PowerShell example
$env:LMSTUDIO_MODEL = "gemma-7b-it-q4"   # or the exact model name LM Studio shows
python app.py
```

```bash
# macOS/Linux example
export LMSTUDIO_MODEL="gemma-7b-it-q4"   # or deepseek-r1:latest, etc.
python app.py
```

- If not set, the backend defaults to `local` (LM Studio's default model). Ensure the name matches what LM Studio lists in the model dropdown.

## Quick Start

### Prerequisites

1. **Python 3.8+**
2. **LM Studio** with DeepSeek model loaded
3. **Unity 2021.3+** (for game integration)

### Installation

1. **Clone/Download** this repository
2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start LM Studio**:
   - Load your DeepSeek model
   - Start the local server (default: http://localhost:1234)

4. **Start the API server**:
   ```bash
   python app.py
   ```

5. **Test the system**:
   ```bash
   python test_client.py
   ```

## API Endpoints

### Core Endpoints

- `POST /start` - Start a new story session
- `POST /choice` - Process player choice and get story response
- `GET /state` - Get current game state
- `POST /reset` - Reset game to initial state

### Utility Endpoints

- `GET /health` - API health check
- `GET /test-lm` - Test LM Studio connection

### Request/Response Format

**Start Story:**
```json
POST /start
Response: {
  "story_segment": "You find yourself in a bustling medieval town...",
  "npc_dialogues": {
    "Alice": "A young woman approaches you...",
    "Bob": "A gruff-looking man glares...",
    "Guard": "A town guard stands watch..."
  },
  "choices": [
    "Approach Alice and ask what's wrong",
    "Confront Bob about his hostile attitude", 
    "Ask the guard about the town's safety"
  ]
}
```

**Make Choice:**
```json
POST /choice
{
  "choice": "Approach Alice and ask what's wrong"
}
Response: {
  "story_segment": "Alice's eyes light up as you approach...",
  "npc_dialogues": {...},
  "choices": [...]
}
```

## Unity Integration

### Setup

1. **Copy Unity scripts** from the `Unity/` folder to your Unity project
2. **Add Newtonsoft.Json** to your Unity project (via Package Manager)
3. **Create UI elements**:
   - Story Text (Text component)
   - NPC Dialogues Text (Text component)
   - Choice Panel with 3 buttons
   - Player Metrics Text (Text component, optional)
   - Game State Text (Text component, optional)

### Usage

1. **Add NarrativeManager** to a GameObject in your scene
2. **Assign UI references** in the inspector
3. **Set API URL** (default: http://localhost:5000)
4. **Run the game** - it will automatically start a new story

#### NPC Auto-Detect (no manual wiring)

- `Assets/FlaskDialogueConnector.cs` auto-detects the NPC you're looking at via camera raycast and a proximity fallback.
- If `playerCamera` isn't assigned, it uses `Camera.main` automatically.
- Ensure your NPCs have a Collider and `FlaskChatConnector` with `botName` set.

### Key Scripts

- `NarrativeManager.cs` - Main controller for story interaction
- `ChoiceButton.cs` - Individual choice button handler

## Configuration

### LM Studio Settings

The system expects LM Studio to be running on `http://localhost:1234`. To change this:

1. **In LM Studio**: Set your preferred port
2. **In narrative_engine.py**: Update `lm_studio_url` parameter
3. **In Unity**: Update `apiBaseUrl` in NarrativeManager

### Model Configuration

The system works with any chat-completion model exposed by LM Studio (e.g., DeepSeek, Google Gemma). Key settings:

- **Temperature**: 0.7 (balanced creativity/consistency)
- **Max Tokens**: 800 (for story responses)
- **Format**: JSON responses for structured output

Tips for Gemma and other models:
- Use instruction-tuned variants (e.g., `-it`) for best dialogue adherence.
- If JSON formatting drifts, lower temperature (e.g., 0.2–0.4) or add stronger JSON enforcement in prompts.
- Bigger models improve coherence but increase latency; adjust `max_tokens` accordingly.

## Game Mechanics

### Player Metrics

The system tracks three moral dimensions:
- **Kindness** (-5 to +5): Affects helpful/compassionate story branches
- **Aggression** (-5 to +5): Influences combat/conflict scenarios  
- **Honesty** (-5 to +5): Determines truth-based story paths

### NPC States

Each NPC has emotional states (0.0 to 1.0):
- **Trust**: Affects willingness to help/share information
- **Anger**: Influences hostile reactions and dialogue
- **Fear**: Determines defensive behavior

### Memory System

- **Short-term**: Last 2-3 story segments
- **Long-term**: Summarized key events every 4 interactions
- **Context Window**: Optimized for LLM token limits

## Immersive Techniques Used

- **Per-NPC Personas & Memory**: Backend `npc_profiles` and short `npc_memories` inject role, tone, and recent dialogue for stable character voices.
- **World-State Prompting**: Player metrics, NPC states, and objective are embedded into prompts for coherent, reactive narrative beats.
- **Diegetic Constraints**: Prompts instruct models to respond strictly as the addressed NPC, reducing cross-talk and preserving immersion.
- **Automatic NPC Identification**: Unity camera-based detection ties the player’s gaze/focus to the backend’s narrative persona selection.
- **Periodic Summarization**: Story memory is condensed every few turns to maintain continuity within model context windows.

## Ethical Considerations

- **Local-Only Inference**: Runs via LM Studio on-device to reduce data exposure and support privacy.
- **Player Agency & Transparency**: README and UI messaging clarify that dialogue is AI-generated and may be imperfect.
- **Content Controls (Configurable)**: While the prototype disables filtering for research, the backend is structured so filters can be re-enabled or customized.
- **Bias & Safety Awareness**: Prompts and personas avoid harmful stereotypes; users are encouraged to review/curate NPC profiles.
- **Logging & Debugging**: Minimal PII; logs focus on system state and errors for responsible iteration.

## Advantages

- **Plug-and-Play with Unity**: Minimal wiring; camera auto-detects NPCs.
- **Consistent Characterization**: Persona + memory yields stable voices and relationships.
- **Scalable Story Control**: Centralized backend manages objectives, state, and memory.
- **Runs Offline**: Works with local models; no external API dependency required.

## Fun Factor & Engagement

- **Reactive NPCs**: Characters remember what you said and react accordingly.
- **Emergent Play**: Open-ended prompts enable creative solutions beyond fixed dialogue trees.
- **Diegetic Discovery**: Finding the right NPC and persuading them becomes a gameplay loop.

## Future Improvements

- **Richer Long-Term Memory**: Per-NPC episodic memory with time decay and summarization by theme.
- **Toolformer Actions**: Let NPCs trigger in-game actions (unlock doors, modify quest flags) via validated action schemas.
- **Better JSON Robustness**: Stricter JSON schemas and validation feedback loops.
- **Dynamic Persona Tuning**: Adapt personalities based on arc outcomes and relationship trajectories.
- **Multimodal Cues**: Use gaze, proximity, and audio cues to modulate NPC response emotion.
- **Authoring Tools**: GUI to edit personas, objectives, and narrative rails without code.

## Development

### Project Structure

```
├── app.py                 # Flask API server
├── narrative_engine.py    # Core narrative logic
├── test_client.py        # Interactive test client
├── requirements.txt      # Python dependencies
├── Unity/
│   ├── NarrativeManager.cs  # Unity integration
│   └── ChoiceButton.cs      # Choice button handler
└── README.md
```

### Adding New Features

1. **New NPCs**: Add to `npc_states` in `NarrativeEngine.__init__()`
2. **New Metrics**: Extend `player_metrics` and `update_player_metrics()`
3. **Custom Prompts**: Modify prompt templates in `generate_story_response()`
4. **Additional Endpoints**: Add routes in `app.py`

### Testing

1. **API Testing**: Use `test_client.py` for interactive testing
2. **Unity Testing**: Use the built-in test functions in NarrativeManager
3. **LM Studio Testing**: Use `/test-lm` endpoint

### Performance Tips

1. **Local Models**: Use smaller models for faster responses
2. **Caching**: Implement response caching for repeated scenarios
3. **Batching**: Group multiple API calls when possible
4. **Token Limits**: Monitor and adjust max_tokens based on model performance

## License

This project is created for educational and research purposes. Feel free to modify and extend for your thesis or personal projects.

## Credits

Based on the research and concepts from:
- GENEVA: Branching narrative generation with GPT-4
- WHAT-IF: Meta-prompting for interactive fiction
- Cross-platform NPC systems with persistent memory
- Game design principles for meaningful decisions