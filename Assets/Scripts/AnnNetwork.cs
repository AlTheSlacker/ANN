using System;
using System.Collections;
using System.Collections.Generic;

namespace ANN
{

    public class AnnNetwork
    {
        private readonly List<AnnSynapse> synapsesAll = new List<AnnSynapse>();

        public List<AnnSynapse> SynapsesAll { get => synapsesAll; }
        public double LearningRate { get; set; }
        public AnnNeuron[][] NetLayers { get; set; }

        public AnnNetwork(int inputs, int[] hiddenLayers, AFType[] afFunctions, int outputs, bool[] addBias)
        {
            NetLayers = new AnnNeuron[hiddenLayers.Length + 2][];

            NetLayers[0] = new AnnNeuron[inputs];

            for (int layerID = 0; layerID < hiddenLayers.Length; layerID++)
            {
                NetLayers[layerID + 1] = new AnnNeuron[hiddenLayers[layerID]];
            }

            NetLayers[hiddenLayers.Length + 1] = new AnnNeuron[outputs];

            CreateNeurons(afFunctions);
            InsertBiasNeurons(addBias);
            CreateSynapses();
            SetWeights();
        }


        private void CreateNeurons(AFType[] afFunctions)
        {
            for (int layerID = 0; layerID < NetLayers.Length; layerID++)
            {
                for (int neuronID = 0; neuronID < NetLayers[layerID].Length; neuronID++)
                {
                    NetLayers[layerID][neuronID] = AddNeuron(layerID, neuronID, afFunctions[layerID]);
                }
            }
        }


        private void InsertBiasNeurons(bool[] addBias)
        {
            for (int layerID = 1; layerID < NetLayers.Length; layerID++)
            {
                int previousLayer = layerID - 1;
                if (addBias[layerID])
                {
                    Array.Resize(ref NetLayers[previousLayer], NetLayers[previousLayer].Length + 1);
                    NetLayers[previousLayer][NetLayers[previousLayer].Length - 1] = AddNeuron(previousLayer, NetLayers[previousLayer].Length - 1, AFType.Bias);
                }
            }
        }


        private AnnNeuron AddNeuron(int layerID, int neuronID, AFType activationFunction)
        {
            AnnNeuron neuron = new AnnNeuron();
            neuron.ActivationFunction.ActivationFunctionType = activationFunction;
            neuron.LayerID = layerID;
            neuron.PositionID = neuronID;
            return neuron;
        }


        private void CreateSynapses()
        {
            for (int layerID = 0; layerID < NetLayers.Length - 1; layerID++)
            {
                for (int inpNeuronID = 0; inpNeuronID < NetLayers[layerID].Length; inpNeuronID++)
                {
                    for (int outNeuronID = 0; outNeuronID < NetLayers[layerID + 1].Length; outNeuronID++)
                    {
                        AnnNeuron outputNeuron = NetLayers[layerID + 1][outNeuronID];
                        if (outputNeuron.ActivationFunction.ActivationFunctionType != AFType.Bias)
                        {
                            AnnNeuron inputNeuron = NetLayers[layerID][inpNeuronID];
                            AnnSynapse newSynapse = new AnnSynapse(inputNeuron, outputNeuron);
                            synapsesAll.Add(newSynapse);
                            inputNeuron.AddOutputSynapse(newSynapse);
                            outputNeuron.AddInputSynapse(newSynapse);
                        }
                    }
                }
            }
        }


        private void SetWeights()
        {
            Random rand = new Random(0);
            for (int i = 0; i < SynapsesAll.Count; i++)
            {
                SynapsesAll[i].Weight = 0.5 - rand.NextDouble() * 1;
            }
        }

    }
}