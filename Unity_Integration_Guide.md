# Unity Integration Guide for AI Narrative Engine

This guide will help you integrate the AI narrative engine with your existing Unity project.

## Prerequisites

1. **LM Studio Running**: Make sure LM Studio is running with your DeepSeek model on `http://localhost:1234`
2. **Flask Server Running**: The AI narrative engine API should be running on `http://localhost:5000`
3. **Unity Project**: Your existing Unity project with dialogue system

## Step 1: Install Dependencies

### Add Newtonsoft.Json to Unity

1. Open Unity Package Manager (Window > Package Manager)
2. Click the "+" button in the top-left
3. Select "Add package from git URL"
4. Enter: `com.unity.nuget.newtonsoft-json`
5. Click "Add"

## Step 2: Copy Integration Scripts

Copy these files to your Unity project's Scripts folder:
- `Unity/NarrativeIntegration.cs` - Main integration script
- `Unity/NarrativeManager.cs` - Alternative UI-based manager
- `Unity/ChoiceButton.cs` - Choice button handler

## Step 3: Setup in Unity

### Option A: Simple Integration (Recommended)

1. **Create a GameObject** in your scene (e.g., "AI Narrative Engine")
2. **Add the NarrativeIntegration script** to this GameObject
3. **Configure the script** in the Inspector:
   - `API Base URL`: `http://localhost:5000`
   - `Enable AI Integration`: ✅ (checked)
   - `Auto Start Story`: ✅ (checked)
   - `Show Debug Info`: ✅ (checked)

### Option B: Integration with Existing Dialogue System

If you want to integrate with your existing dialogue system:

1. **Find your DialogueManager** in the scene
2. **Add NarrativeIntegration** to the same GameObject or create a new one
3. **Connect the event channels**:
   - Assign your `DialogueDataChannelSO` to the `Dialogue Data Channel` field
   - Assign your `DialogueChoicesChannelSO` to the `Dialogue Choices Channel` field

## Step 4: Test the Connection

1. **Start the Flask server** (if not already running):
   ```bash
   python app.py
   ```

2. **Start LM Studio** with your DeepSeek model

3. **Play your Unity scene** - you should see connection messages in the Console

## Step 5: Integration with Your Existing Systems

### Connecting to Your Dialogue System

Your Unity project already has a sophisticated dialogue system. Here's how to integrate:

```csharp
// In your DialogueManager or similar script
public class YourDialogueManager : MonoBehaviour
{
    private NarrativeIntegration aiNarrative;
    
    void Start()
    {
        aiNarrative = FindObjectOfType<NarrativeIntegration>();
        
        if (aiNarrative != null)
        {
            // Listen for AI story updates
            aiNarrative.OnStoryUpdated += HandleAIStoryUpdate;
            aiNarrative.OnStateUpdated += HandleAIStateUpdate;
        }
    }
    
    void HandleAIStoryUpdate(StoryResponse story)
    {
        // Convert AI story to your dialogue format
        // You can use your existing DialogueDataSO structure
        var dialogueData = new DialogueDataSO();
        dialogueData.DialogueLines = new List<DialogueLineSO>();
        
        // Add story segment as a dialogue line
        var storyLine = ScriptableObject.CreateInstance<DialogueLineSO>();
        storyLine.Text = story.story_segment;
        dialogueData.DialogueLines.Add(storyLine);
        
        // Add NPC dialogues
        foreach (var npcDialogue in story.npc_dialogues)
        {
            var npcLine = ScriptableObject.CreateInstance<DialogueLineSO>();
            npcLine.Text = $"{npcDialogue.Key}: {npcDialogue.Value}";
            dialogueData.DialogueLines.Add(npcLine);
        }
        
        // Trigger your existing dialogue system
        // dialogueDataChannel.RaiseEvent(dialogueData);
    }
    
    void HandleAIStateUpdate(GameState state)
    {
        // Handle AI state updates (player metrics, NPC states, etc.)
        Debug.Log($"Player Kindness: {state.player_metrics["kindness"]}");
    }
}
```

### Connecting to Your Choice System

```csharp
// In your choice handling script
public class YourChoiceHandler : MonoBehaviour
{
    private NarrativeIntegration aiNarrative;
    
    void Start()
    {
        aiNarrative = FindObjectOfType<NarrativeIntegration>();
    }
    
    public void OnChoiceSelected(string choice)
    {
        if (aiNarrative != null && aiNarrative.IsConnected())
        {
            // Send choice to AI engine
            aiNarrative.MakeChoicePublic(choice);
        }
    }
}
```

## Step 6: Advanced Integration

### Custom NPC Integration

You can connect the AI NPC states to your existing NPC system:

```csharp
public class NPCAIController : MonoBehaviour
{
    public string npcName = "Alice";
    private NarrativeIntegration aiNarrative;
    
    void Start()
    {
        aiNarrative = FindObjectOfType<NarrativeIntegration>();
        
        if (aiNarrative != null)
        {
            aiNarrative.OnStateUpdated += HandleNPCStateUpdate;
        }
    }
    
    void HandleNPCStateUpdate(GameState state)
    {
        if (state.npc_states.ContainsKey(npcName))
        {
            var npcState = state.npc_states[npcName];
            
            // Update your NPC's behavior based on AI state
            float trust = npcState["trust"];
            float anger = npcState["anger"];
            float fear = npcState["fear"];
            
            // Apply to your NPC system
            UpdateNPCBehavior(trust, anger, fear);
        }
    }
    
    void UpdateNPCBehavior(float trust, float anger, float fear)
    {
        // Integrate with your existing NPC system
        // This could affect dialogue, animations, AI behavior, etc.
    }
}
```

### Player Metrics Integration

```csharp
public class PlayerMetricsDisplay : MonoBehaviour
{
    public Text kindnessText;
    public Text aggressionText;
    public Text honestyText;
    
    private NarrativeIntegration aiNarrative;
    
    void Start()
    {
        aiNarrative = FindObjectOfType<NarrativeIntegration>();
        
        if (aiNarrative != null)
        {
            aiNarrative.OnStateUpdated += UpdateMetricsDisplay;
        }
    }
    
    void UpdateMetricsDisplay(GameState state)
    {
        if (kindnessText) kindnessText.text = $"Kindness: {state.player_metrics["kindness"]}";
        if (aggressionText) aggressionText.text = $"Aggression: {state.player_metrics["aggression"]}";
        if (honestyText) honestyText.text = $"Honesty: {state.player_metrics["honesty"]}";
    }
}
```

## Step 7: Testing

### Test the Complete System

1. **Start everything**:
   - LM Studio with DeepSeek model
   - Flask server (`python app.py`)
   - Unity project

2. **Check Console** for connection messages:
   - ✅ AI Narrative Engine connected successfully!
   - 🎮 AI Story started successfully!

3. **Test story generation** by making choices in your game

### Debug Information

The integration script provides extensive debug information:
- Connection status
- Story updates
- State changes
- Error messages

## Troubleshooting

### Common Issues

1. **Connection Failed**:
   - Check if Flask server is running on port 5000
   - Check if LM Studio is running on port 1234
   - Check firewall settings

2. **JSON Parse Errors**:
   - LM Studio might return malformed JSON
   - Check LM Studio model quality
   - System includes fallback responses

3. **Unity Integration Issues**:
   - Ensure Newtonsoft.Json is installed
   - Check script compilation errors
   - Verify GameObject setup

### Performance Tips

1. **Local Models**: Use smaller models for faster responses
2. **Caching**: Consider caching responses for repeated scenarios
3. **Async Operations**: All AI calls are async to prevent blocking

## Next Steps

Once integrated, you can:

1. **Customize prompts** in `narrative_engine.py`
2. **Add more NPCs** to the system
3. **Extend player metrics** for more complex behavior
4. **Implement save/load** for persistent stories
5. **Add voice synthesis** for spoken dialogue

## Support

If you encounter issues:
1. Check the Unity Console for error messages
2. Test the API directly with `test_client.py`
3. Verify LM Studio is working correctly
4. Check network connectivity between Unity and Flask server 