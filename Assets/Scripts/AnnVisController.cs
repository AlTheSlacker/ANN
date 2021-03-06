﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace ANN
{

    public class AnnVisController : MonoBehaviour
    {

        [SerializeField] private float horizontalSpacing = 6;
        [SerializeField] private float verticalSpacing = 3;
        [SerializeField] private float weightPos = 0.3f;

        private GameObject[][] neuronObjects;
        private GameObject[] synapseObjects;
        private AnnNetwork net = null;
        private AnnVisGeneralGui generalGUI;


        public void Awake()
        {
            GameObject go_GUI = new GameObject("ANN_GUI");
            generalGUI = go_GUI.AddComponent<AnnVisGeneralGui>();
        }


        public void VisArchitecture(AnnNetwork newNet)
        {
            net = newNet;

            CreateNeuronGO();
            CreateSynapseGO();

            PositionMainCamera();
        }


        private void CreateNeuronGO()
        {
            int maxNeuronsInSingleLayer = MaxNeuronsInSingleLayer();
            float initialOffset;
            float zPos = 0;

            neuronObjects = new GameObject[net.NetLayers.Length][];

            for (int layerID = 0; layerID < net.NetLayers.Length; layerID++)
            {
                neuronObjects[layerID] = new GameObject[net.NetLayers[layerID].Length];
                for (int neuronID = 0; neuronID < net.NetLayers[layerID].Length; neuronID++)
                {
                    string name = layerID + "_" + neuronID;
                    Color color = SetColor(net.NetLayers[layerID][neuronID].ActivationFunction.ActivationFunctionType);
                    GameObject neuronSphere = CreateVisualSphere(name, color);

                    neuronObjects[layerID][neuronID] = neuronSphere;

                    initialOffset = (maxNeuronsInSingleLayer - net.NetLayers[layerID].Length) * verticalSpacing / 2;
                    float yPos = (maxNeuronsInSingleLayer * verticalSpacing) - (initialOffset + verticalSpacing * neuronID);
                    neuronSphere.transform.position = new Vector3(0, yPos, zPos);
                }
                zPos += horizontalSpacing;
            }
        }


        private void CreateSynapseGO()
        {
            int synapseCount = net.SynapsesAll.Count;
            synapseObjects = new GameObject[synapseCount];
            for (int synapseID = 0; synapseID < synapseCount; synapseID++)
            {
                synapseObjects[synapseID] = CreateSynapseObject(net.SynapsesAll[synapseID], synapseID);
            }
        }


        private GameObject CreateVisualSphere(string nameRef, Color color)
        {
            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = nameRef;
            go.GetComponent<MeshRenderer>().material.color = color;
            MakeClickable(go);
            AddTextObject(go);

            return go;
        }


        private GameObject CreateSynapseObject(AnnSynapse aNN_Synapse, int synapseID)
        {
            int inputLayerID = aNN_Synapse.InputNeuron.LayerID;
            int inputPositionID = aNN_Synapse.InputNeuron.PositionID;
            int outputLayerID = aNN_Synapse.OutputNeuron.LayerID;
            int outputPositionID = aNN_Synapse.OutputNeuron.PositionID;

            GameObject synapseGO = new GameObject
            {
                name = "Synapse_" + synapseID
            };

            AddTextObject(synapseGO);

            Vector3 dir = (neuronObjects[outputLayerID][outputPositionID].transform.position - neuronObjects[inputLayerID][inputPositionID].transform.position).normalized;
            synapseGO.transform.position = neuronObjects[inputLayerID][inputPositionID].transform.position + dir * horizontalSpacing * weightPos;

            GameObject go = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            go.name = "-2_" + synapseID;
            float diameter = 0.05f;
            float length = Vector3.Distance(neuronObjects[inputLayerID][inputPositionID].transform.position, neuronObjects[outputLayerID][outputPositionID].transform.position) / 2;
            go.transform.localScale = new Vector3(diameter, length, diameter);
            go.transform.position = neuronObjects[inputLayerID][inputPositionID].transform.position;
            go.transform.LookAt(neuronObjects[outputLayerID][outputPositionID].transform);
            go.transform.rotation *= Quaternion.Euler(90, 0, 0);
            go.transform.position = (neuronObjects[outputLayerID][outputPositionID].transform.position + neuronObjects[inputLayerID][inputPositionID].transform.position) / 2.0f;
            go.transform.SetParent(synapseGO.transform);
            MakeClickable(go);

            return synapseGO;
        }


        private void MakeClickable(GameObject go)
        {
            AnnVisClick visClick = go.AddComponent<AnnVisClick>();
            visClick.Net = net;
            visClick.GeneralGUI = generalGUI;
        }


        private int MaxNeuronsInSingleLayer()
        {
            int maxNeuronsInSingleLayer = 0;
            for (int currentLayer = 0; currentLayer < net.NetLayers.Length - 1; currentLayer++)
            {
                int neuronsInCurrentLayer = net.NetLayers[currentLayer].Length;
                if (neuronsInCurrentLayer > maxNeuronsInSingleLayer) maxNeuronsInSingleLayer = neuronsInCurrentLayer;
            }
            return maxNeuronsInSingleLayer;
        }


        private Color SetColor(AFType aF)
        {
            return ((int)aF) switch
            {
                // LeakyReLu
                1 => Color.green,
                // Sigmoid
                2 => Color.blue,
                // Tanh
                3 => Color.magenta,
                // Input
                200 => Color.cyan,
                // Output
                300 => Color.black,
                // Bias
                400 => Color.yellow,
                _ => Color.grey,
            };
        }


        private void PositionMainCamera()
        {
            int maxNeuronsInSingleLayer = MaxNeuronsInSingleLayer();
            float yCentreHeight = maxNeuronsInSingleLayer * verticalSpacing / 2 + verticalSpacing / 2;
            float zCentreHeight = net.NetLayers.Length * horizontalSpacing / 2 - horizontalSpacing / 2;
            float cameraFOV = Camera.main.fieldOfView * 0.0174533f;
            float cameraXDistance = ((maxNeuronsInSingleLayer) * verticalSpacing) / Mathf.Atan(cameraFOV);
            Camera.main.transform.position = new Vector3(cameraXDistance, yCentreHeight, zCentreHeight);
        }


        public void NeuronTextOutput()
        {
            TMP_Text annUIText;
            for (int currentLayer = 0; currentLayer < neuronObjects.Length; currentLayer++)
            {
                for (int neuronID = 0; neuronID < neuronObjects[currentLayer].Length; neuronID++)
                {
                    annUIText = neuronObjects[currentLayer][neuronID].GetComponentInChildren<TMPro.TextMeshPro>();
                    if (net.NetLayers[currentLayer][neuronID].ActivationFunction.ActivationFunctionType == AFType.Input)
                    {
                        annUIText.text = net.NetLayers[currentLayer][neuronID].Input.ToString("f3");
                    }
                    else
                    {
                        annUIText.text = net.NetLayers[currentLayer][neuronID].Output.ToString("f3");
                    }
                }
            }
        }


        public void SynapseTextOutput()
        {
            TMP_Text annUIText;
            for (int synapseID = 0; synapseID < synapseObjects.Length; synapseID++)
            {
                annUIText = synapseObjects[synapseID].GetComponentInChildren<TMPro.TextMeshPro>();
                annUIText.color = Color.black;
                annUIText.text = net.SynapsesAll[synapseID].Weight.ToString("f3");
            }
        }


        private void AddTextObject(GameObject go)
        {
            GameObject textDisplay;
            TMPro.TextMeshPro tmp;
            textDisplay = new GameObject
            {
                name = "TextObject"
            };

            textDisplay.transform.rotation *= Quaternion.Euler(0, -90, 0);
            textDisplay.AddComponent<TMPro.TextMeshPro>();
            textDisplay.transform.SetParent(go.transform);
            textDisplay.transform.localPosition = new Vector3(0.5f, 0, 0);

            tmp = textDisplay.GetComponent<TMPro.TextMeshPro>();
            tmp.alignment = TextAlignmentOptions.Center;
            tmp.fontSize = 2;
            tmp.margin = new Vector4(9.5f, 2.3f, 9.5f, 2.3f);
        }

    }
}