# Adaptive Story Arc & NPC Relationship Engine

A dynamic narrative engine that uses LLMs to generate branching stories with persistent NPC relationships and moral consequences. Built for easy integration with Unity and optimized for local model usage with LM Studio.

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
[Unity Game Client] ←→ [Flask API Server] ←→ [LM Studio (DeepSeek)]
                              ↓
                    [Game State Management]
```

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

The system is optimized for DeepSeek models but works with any chat-completion model. Key settings:

- **Temperature**: 0.7 (balanced creativity/consistency)
- **Max Tokens**: 800 (for story responses)
- **Format**: JSON responses for structured output

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

## Troubleshooting

### Common Issues

1. **LM Studio Connection Failed**:
   - Ensure LM Studio is running on localhost:1234
   - Check if DeepSeek model is loaded
   - Verify API server is accessible

2. **Unity Connection Issues**:
   - Check firewall settings
   - Verify API server is running
   - Test with `test_client.py` first

3. **JSON Parse Errors**:
   - LM Studio may return malformed JSON
   - System includes fallback responses
   - Check LM Studio model quality

4. **Memory Issues**:
   - Long stories may exceed context limits
   - Summarization happens every 4 interactions
   - Consider reducing story complexity

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