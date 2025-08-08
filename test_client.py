import requests
import json
import time

class NarrativeTestClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        
    def test_health(self):
        """Test if the API is running."""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ API is healthy!")
                return True
            else:
                print(f"❌ API health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            return False
    
    def test_lm_connection(self):
        """Test LM Studio connection."""
        try:
            response = requests.get(f"{self.base_url}/test-lm")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ LM Studio connection: {result['message']}")
                print(f"   Response: {result['response']}")
                return True
            else:
                print(f"❌ LM Studio test failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ LM Studio test error: {e}")
            return False
    
    def start_story(self):
        """Start a new story."""
        try:
            response = requests.post(f"{self.base_url}/start")
            if response.status_code == 200:
                result = response.json()
                print("🎮 Story started!")
                print(f"📖 Story: {result['story_segment']}")
                print("🗣️  NPC Dialogues:")
                for npc, dialogue in result['npc_dialogues'].items():
                    print(f"   {npc}: {dialogue}")
                print("🎯 Choices:")
                for i, choice in enumerate(result['choices'], 1):
                    print(f"   {i}. {choice}")
                return result
            else:
                print(f"❌ Failed to start story: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Start story error: {e}")
            return None
    
    def make_choice(self, choice):
        """Make a choice in the story."""
        try:
            data = {"choice": choice}
            response = requests.post(f"{self.base_url}/choice", json=data)
            if response.status_code == 200:
                result = response.json()
                # Validate JSON response format
                if not all(key in result for key in ["story_segment", "npc_dialogues", "choices"]):
                    print("❌ Invalid response format")
                    return None
                print(f"🎯 You chose: {choice}")
                print(f"📖 Story: {result['story_segment']}")
                print("🗣️  NPC Dialogues:")
                for npc, dialogue in result['npc_dialogues'].items():
                    print(f"   {npc}: {dialogue}")
                print("🎯 New Choices:")
                for i, choice in enumerate(result['choices'], 1):
                    print(f"   {i}. {choice}")
                return result
            else:
                print(f"❌ Failed to make choice: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            return None
        except Exception as e:
            print(f"❌ Make choice error: {e}")
            return None
    
    def get_state(self):
        """Get current game state."""
        try:
            response = requests.get(f"{self.base_url}/state")
            if response.status_code == 200:
                result = response.json()
                print("📊 Current Game State:")
                print(f"   Interactions: {result['interaction_count']}")
                print(f"   Memory: {result['memory_summary']}")
                print("   Player Metrics:")
                for metric, value in result['player_metrics'].items():
                    print(f"     {metric}: {value}")
                print("   NPC States:")
                for npc, states in result['npc_states'].items():
                    print(f"     {npc}: {states}")
                return result
            else:
                print(f"❌ Failed to get state: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Get state error: {e}")
            return None

def run_interactive_test():
    """Run an interactive test session."""
    client = NarrativeTestClient()
    
    print("🚀 Starting Narrative Engine Test Client")
    print("=" * 50)
    
    # Test API health
    if not client.test_health():
        print("❌ API is not running. Please start the Flask server first.")
        return
    
    # Test LM Studio connection
    if not client.test_lm_connection():
        print("❌ LM Studio is not running. Please start LM Studio with DeepSeek model.")
        return
    
    # Start story
    story = client.start_story()
    if not story:
        return
    
    print("\n" + "=" * 50)
    print("🎮 Interactive Test Mode")
    print("Commands:")
    print("  [1-3] - Make a choice")
    print("  state - Show current game state")
    print("  quit - Exit test")
    print("=" * 50)
    
    while True:
        try:
            command = input("\nEnter command: ").strip().lower()
            
            if command == "quit":
                print("👋 Goodbye!")
                break
            elif command == "state":
                client.get_state()
            elif command in ["1", "2", "3"]:
                choice_index = int(command) - 1
                if choice_index < len(story['choices']):
                    choice = story['choices'][choice_index]
                    result = client.make_choice(choice)
                    if result:
                        story = result
                else:
                    print("❌ Invalid choice number")
            else:
                print("❌ Unknown command. Use 1-3, 'state', or 'quit'")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_interactive_test() 