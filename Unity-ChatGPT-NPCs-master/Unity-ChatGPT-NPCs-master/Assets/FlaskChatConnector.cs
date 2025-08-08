using System.Collections;
using System.Collections.Generic;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;

[System.Serializable]
public class PromptPayload
{
    public string prompt;
    public string botName;
}

[System.Serializable]
public class FlaskResponse
{
    public string narrative;
}

public class FlaskChatConnector : MonoBehaviour
{
    public TextMeshProUGUI dialogueText;  // <-- This creates the slot in Inspector
    public string botName;
    public AudioClip soundStart;    // when starting interaction
    public AudioClip soundStop;     // when stopping interaction (e.g., user cancels)
    public AudioClip soundSend;     // when sending the request to Flask
    public AudioClip soundReceive;  // when receiving response from Flask
    private AudioSource audioSource;
    public string flaskUrl = "http://YOUR_SERVER_IP_OR_DOMAIN:PORT/generate_narrative";


    private void Awake()
    {
        audioSource = GetComponent<AudioSource>();
        if (audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
        }
    }

    public void StartInteraction()
    {
        PlaySound(soundStart);
    }

    public void StopInteraction()
    {
        PlaySound(soundStop);
    }

    public void RequestNarrative(string playerPrompt)
    {
        StartCoroutine(RequestNarrativeCoroutine(playerPrompt));
    }

    IEnumerator RequestNarrativeCoroutine(string playerPrompt)
    {
        PlaySound(soundSend);

        var payload = new PromptPayload { prompt = playerPrompt, botName = botName };
        string json = JsonUtility.ToJson(payload);

        using (var uwr = new UnityWebRequest(flaskUrl, "POST"))
        {
            byte[] bodyRaw = Encoding.UTF8.GetBytes(json);
            uwr.uploadHandler = new UploadHandlerRaw(bodyRaw);
            uwr.downloadHandler = new DownloadHandlerBuffer();
            uwr.SetRequestHeader("Content-Type", "application/json");

            yield return uwr.SendWebRequest();

            if (uwr.result == UnityWebRequest.Result.Success)
            {
                PlaySound(soundReceive);

                try
                {
                    var resp = JsonUtility.FromJson<FlaskResponse>(uwr.downloadHandler.text);
                    dialogueText.text = resp.narrative;
                }
                catch
                {
                    dialogueText.text = uwr.downloadHandler.text;
                }
            }
            else
            {
                dialogueText.text = "Error: " + uwr.error;
                Debug.LogError($"Flask request failed: {uwr.error} | {uwr.downloadHandler.text}");
            }
        }
    }

    private void PlaySound(AudioClip clip)
    {
        if (clip != null && audioSource != null)
        {
            audioSource.PlayOneShot(clip);
        }
    }
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
