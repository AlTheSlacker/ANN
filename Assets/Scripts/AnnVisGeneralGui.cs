using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace ANN
{

    public class AnnVisGeneralGui : MonoBehaviour
    {
        private Text annUIText;

        public void Awake()
        {
            Canvas annUICanvas = gameObject.AddComponent<Canvas>();
            annUICanvas.renderMode = RenderMode.ScreenSpaceOverlay;

            gameObject.AddComponent<CanvasScaler>();
            gameObject.AddComponent<CanvasRenderer>();

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
}