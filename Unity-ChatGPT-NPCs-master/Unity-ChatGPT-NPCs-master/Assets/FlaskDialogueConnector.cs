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
    [Tooltip("Camera used to auto-detect NPC when sending a message if currentNpc is empty.")]
    public Camera playerCamera;      // assign in Inspector
    [Tooltip("Max distance for NPC auto-detect raycast.")]
    public float detectDistance = 3f;
    [Tooltip("Optional layer mask for NPC detection. Leave default to detect all layers.")]
    public LayerMask npcLayerMask;

    void Start()
    {
        // Auto-resolve camera if not assigned
        if (playerCamera == null)
        {
            playerCamera = Camera.main;
            if (playerCamera != null)
            {
                Debug.Log("[AutoDetect] playerCamera not assigned; using Camera.main.");
            }
            else
            {
                Debug.LogWarning("[AutoDetect] No camera assigned and Camera.main is null. Please assign a camera on FlaskDialogueConnector.");
            }
        }
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

        // Auto-set NPC if not set yet by detecting who the player is looking at
        if (string.IsNullOrWhiteSpace(currentNpc))
        {
            string detected = AutoDetectNpc();
            if (!string.IsNullOrWhiteSpace(detected))
            {
                SetCurrentNpc(detected);
                Debug.Log($"[AutoDetect] NPC set to: {detected}");
            }
            else
            {
                Debug.LogWarning("[AutoDetect] No NPC detected. Consider looking at an NPC or adjusting detectDistance/layer mask.");
            }
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

    // Attempts to detect the NPC the player is looking at by raycasting from the playerCamera.
    // Uses trigger-colliding raycast and falls back to an overlap sphere scan in front of the camera.
    string AutoDetectNpc()
    {
        if (playerCamera == null)
        {
            Debug.LogWarning("[AutoDetect] playerCamera is not assigned on FlaskDialogueConnector.");
            return "";
        }

        Ray r = new Ray(playerCamera.transform.position, playerCamera.transform.forward);
        int mask = npcLayerMask.value == 0 ? ~0 : npcLayerMask.value; // if unset, detect all layers
        if (Physics.Raycast(r, out RaycastHit hit, detectDistance, mask, QueryTriggerInteraction.Collide))
        {
            // Expect NPCs to have a FlaskChatConnector to carry botName
            var npcConn = hit.collider.GetComponentInParent<FlaskChatConnector>();
            if (npcConn != null)
            {
                // Prefer explicit botName; fallback to GameObject name
                string name = string.IsNullOrWhiteSpace(npcConn.botName) ? npcConn.gameObject.name : npcConn.botName;
                return name;
            }
        }
        // Fallback: search a small sphere in front of the camera for the nearest NPC
        Vector3 center = playerCamera.transform.position + playerCamera.transform.forward * Mathf.Max(1f, detectDistance * 0.5f);
        float radius = Mathf.Max(0.75f, detectDistance * 0.35f);
        var hits = Physics.OverlapSphere(center, radius, mask, QueryTriggerInteraction.Collide);
        float bestScore = -1f;
        FlaskChatConnector best = null;
        foreach (var col in hits)
        {
            var npcConn = col.GetComponentInParent<FlaskChatConnector>();
            if (npcConn == null) continue;
            // Prefer objects more in front and closer
            Vector3 dir = (npcConn.transform.position - playerCamera.transform.position).normalized;
            float dot = Vector3.Dot(playerCamera.transform.forward, dir); // -1..1, higher is more centered
            float dist = Vector3.Distance(playerCamera.transform.position, npcConn.transform.position);
            float score = dot - (dist / Mathf.Max(0.001f, detectDistance));
            if (score > bestScore)
            {
                bestScore = score;
                best = npcConn;
            }
        }
        if (best != null)
        {
            string name = string.IsNullOrWhiteSpace(best.botName) ? best.gameObject.name : best.botName;
            return name;
        }
        return "";
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


