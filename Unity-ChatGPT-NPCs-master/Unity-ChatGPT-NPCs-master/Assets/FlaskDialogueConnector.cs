using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.Networking;
using System.Collections;
using System.Text;

[System.Serializable]
public class ChoiceRequest
{
    public string choice;
    public string npc;
}

[System.Serializable]
public class ChoiceResponse
{
    public string story_segment;
    public NpcDialogue[] npc_dialogues;
    public string[] choices;
}

[System.Serializable]
public class NpcDialogue
{
    public string npc_name;
    public string dialogue;
}

public class FlaskDialogueConnector : MonoBehaviour
{
    [Header("UI References")]
    public TMP_InputField inputField; // Player input box
    public Button sendButton;         // Send button
    public TextMeshProUGUI dialogueText; // To show story + NPC dialogues
    // Choices UI disabled: keep fields for compatibility but unused
    public Transform choicesContainer;   // (unused)
    public GameObject choiceButtonPrefab; // (unused)

    [Header("Backend Settings")]
    public string flaskChoiceUrl = "http://127.0.0.1:5000/choice";

    [Header("NPC Settings")]
    public string currentNpc = ""; // Set this when interacting with specific NPCs

    void Start()
    {
        sendButton.onClick.AddListener(OnSendClicked);
        dialogueText.text = "Welcome! Type and press Send.";
    }

    void OnSendClicked()
    {
        string playerText = inputField.text.Trim();

        if (string.IsNullOrEmpty(playerText))
        {
            dialogueText.text = "Please enter something.";
            return;
        }

        inputField.text = ""; // Clear input for next typing

        StartCoroutine(SendChoice(playerText));
    }

    IEnumerator SendChoice(string choiceText)
    {
        ChoiceRequest request = new ChoiceRequest { choice = choiceText, npc = currentNpc };
        string json = JsonUtility.ToJson(request);

        Debug.Log($"Sending request to: {flaskChoiceUrl}");
        Debug.Log($"Request data: {json}");

        using (UnityWebRequest www = new UnityWebRequest(flaskChoiceUrl, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            www.uploadHandler = new UploadHandlerRaw(bodyRaw);
            www.downloadHandler = new DownloadHandlerBuffer();
            www.SetRequestHeader("Content-Type", "application/json");
            www.timeout = 30; // Set 30 second timeout

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                try
                {
                    Debug.Log($"Response received: {www.downloadHandler.text}");
                    ChoiceResponse response = JsonUtility.FromJson<ChoiceResponse>(www.downloadHandler.text);
                    UpdateDialogueUI(response);
                }
                catch (System.Exception e)
                {
                    Debug.LogError($"Error parsing server response: {e.Message}");
                    dialogueText.text = $"Error parsing server response: {e.Message}";
                }
            }
            else
            {
                Debug.LogError($"Network error: {www.error} - Response: {www.downloadHandler.text}");
                dialogueText.text = $"Network error: {www.error}\nResponse: {www.downloadHandler.text}";
            }
        }
    }

    void UpdateDialogueUI(ChoiceResponse response)
    {
        dialogueText.text = response.story_segment + "\n\n";

        if (response.npc_dialogues != null && response.npc_dialogues.Length > 0)
        {
            foreach (var npcDialogue in response.npc_dialogues)
            {
                dialogueText.text += $"{npcDialogue.npc_name}: {npcDialogue.dialogue}\n";
            }
        }
        // Choices UI disabled: do not spawn any buttons
    }

    void ClearChoices()
    {
        // No-op since choices UI is disabled
    }

    // Call this method when starting interaction with a specific NPC
    public void SetCurrentNpc(string npcName)
    {
        currentNpc = npcName;
        Debug.Log($"Now interacting with: {currentNpc}");
    }

    // Call this method when ending interaction
    public void ClearCurrentNpc()
    {
        currentNpc = "";
        Debug.Log("Ended NPC interaction");
    }

    // Test backend connectivity
    public void TestBackendConnection()
    {
        StartCoroutine(HealthCheck());
    }

    IEnumerator HealthCheck()
    {
        string healthUrl = flaskChoiceUrl.Replace("/choice", "/health");
        Debug.Log($"Testing backend connection to: {healthUrl}");

        using (UnityWebRequest www = new UnityWebRequest(healthUrl, "GET"))
        {
            www.downloadHandler = new DownloadHandlerBuffer();
            www.timeout = 10;

            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Debug.Log($"Backend is healthy: {www.downloadHandler.text}");
                dialogueText.text = "Backend connection successful!";
            }
            else
            {
                Debug.LogError($"Backend health check failed: {www.error}");
                dialogueText.text = $"Backend connection failed: {www.error}";
            }
        }
    }
}


