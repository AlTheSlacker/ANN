using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ANN_VisGeneralGUI : MonoBehaviour
{
    private Text annUIText;

    public void Awake()
    {
        Canvas annUICanvas = gameObject.AddComponent<Canvas>();
        annUICanvas.renderMode = RenderMode.ScreenSpaceOverlay;

        CanvasScaler annUICanvasScaler = gameObject.AddComponent<CanvasScaler>();

        CanvasRenderer annUICanvasRenderer = gameObject.AddComponent<CanvasRenderer>();

        annUIText = gameObject.AddComponent<Text>();
        annUIText.alignment = TextAnchor.UpperLeft;
        annUIText.font = (Font)Resources.GetBuiltinResource(typeof(Font), "Arial.ttf");
        annUIText.fontSize = 17;
        annUIText.color = Color.black;
    }


    public void SetText(string text)
    {
        annUIText.text = text;
    }

}