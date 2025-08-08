using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.Collections.Generic;

public class DialogueChoiceManager : MonoBehaviour
{
    [Header("UI References")]
    public Transform choiceContainer; // Drag your ChoiceContainer here
    public GameObject choiceButtonPrefab; // Drag your ChoiceButton prefab here

    public void DisplayChoices(List<string> choices)
    {
        // Clear old buttons
        foreach (Transform child in choiceContainer)
        {
            Destroy(child.gameObject);
        }

        // Create a button for each choice
        foreach (string choiceText in choices)
        {
            GameObject buttonObj = Instantiate(choiceButtonPrefab, choiceContainer);
            TMP_Text buttonLabel = buttonObj.GetComponentInChildren<TMP_Text>();
            buttonLabel.text = choiceText;

            Button btn = buttonObj.GetComponent<Button>();
            btn.onClick.AddListener(() => OnChoiceSelected(choiceText));
        }
    }

    void OnChoiceSelected(string choice)
    {
        Debug.Log("User selected: " + choice);
        // TODO: Send this choice back to your Flask backend
        // Call your dialogue update function here
    }
}
