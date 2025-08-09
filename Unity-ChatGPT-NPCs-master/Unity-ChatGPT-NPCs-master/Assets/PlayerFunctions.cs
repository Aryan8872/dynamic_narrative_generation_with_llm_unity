using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class PlayerFunctions : MonoBehaviour
{
    private GameObject playerCamera;
    private GameObject mainUI;
    void Start()
    {
        playerCamera = GameObject.FindGameObjectWithTag("MainCamera");    
        mainUI = GameObject.FindGameObjectWithTag("MainUI");
    }

    public void DisablePlayer()
    {
        // Safely disable movement
        var movement = gameObject.GetComponent<PlayerMovement>();
        if (movement != null)
        {
            movement.enabled = false;
        }

        // Safely disable look (search inactive children too)
        var mouseLook = gameObject.GetComponentInChildren<MouseLook>(true);
        if (mouseLook != null)
        {
            mouseLook.enabled = false;
        }

        Cursor.lockState = CursorLockMode.None;

        // Safely disable UI text
        if (mainUI != null)
        {
            var uiText = mainUI.GetComponentInChildren<TMP_Text>(true);
            if (uiText != null)
            {
                uiText.enabled = false;
            }
        }
        //mainUI.SetActive(false);

        // Safely disable player camera
        if (playerCamera != null)
        {
            playerCamera.SetActive(false);
        }
    }

    public void EnablePlayer()
    {
        if (playerCamera != null)
        {
            playerCamera.SetActive(true);
        }
        //mainUI.SetActive(true);
        if (mainUI != null)
        {
            var uiText = mainUI.GetComponentInChildren<TMP_Text>(true);
            if (uiText != null)
            {
                uiText.enabled = true;
            }
        }


        var movement = gameObject.GetComponent<PlayerMovement>();
        if (movement != null)
        {
            movement.enabled = true;
        }

        var mouseLook = gameObject.GetComponentInChildren<MouseLook>(true);
        if (mouseLook != null)
        {
            mouseLook.enabled = true;
        }

        Cursor.lockState = CursorLockMode.Locked;        
    }

}
