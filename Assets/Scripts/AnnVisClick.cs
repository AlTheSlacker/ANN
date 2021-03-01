using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace ANN
{

    public class AnnVisClick : MonoBehaviour
    {
        public AnnNetwork Net { get; set; }
        public AnnVisGeneralGui GeneralGUI { get; set; }

        void OnMouseDown()
        {
            int layerID = GetLayerID();
            int positionID = GetPositionID();
            string message = layerID switch
            {
                -2 => GetSynapseInfo(positionID),
                -1 => GetTrainingInfo(positionID),
                _ => GetNeuronInfo(layerID, positionID),
            };
            GeneralGUI.SetText(message);
        }


        private int GetLayerID()
        {
            int index = name.IndexOf("_");
            if (index > 0) return int.Parse(name.Substring(0, index));
            return int.Parse(name);
        }


        private int GetPositionID()
        {
            int index = name.IndexOf("_");
            if (index > 0) return int.Parse(name.Substring(index + 1));
            return int.Parse(name);
        }


        private string GetSynapseInfo(int positionID)
        {
            AnnSynapse synapse = Net.SynapsesAll[positionID];

            string message = "Synapse" + "\n";
            message += "ID: " + positionID + "\n";
            message += "Input Neuron: " + synapse.InputNeuron.LayerID + " / " + synapse.InputNeuron.PositionID + "\n";
            message += "Input Neuron [Output]: " + synapse.InputNeuron.Output + "\n";
            message += "Output Neuron: " + synapse.OutputNeuron.LayerID + " / " + synapse.OutputNeuron.PositionID + "\n";
            message += "Output Neuron [Total Input]: " + synapse.OutputNeuron.Input + "\n";
            message += "Weight: " + synapse.Weight + "\n";
            message += "Synapse output: " + synapse.Weight * synapse.InputNeuron.Output + "\n";

            return message;
        }


        private string GetTrainingInfo(int positionID)
        {
            string message = "Training" + "\n";
            message += "ID: " + positionID + "\n";

            return message;
        }


        private string GetNeuronInfo(int layerID, int positionID)
        {
            AnnNeuron neuron = Net.NetLayers[layerID][positionID];

            string message = "Neuron" + "\n";
            message += "Layer / NeuronID: " + layerID + " / " + positionID + "\n";
            message += "Input: " + neuron.Input + "\n";
            message += "Output: " + neuron.Output + "\n";
            message += "Activation Function: " + neuron.ActivationFunction.ActivationFunctionType + "\n";

            return message;
        }

    }
}