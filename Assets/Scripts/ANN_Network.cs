using System;
using System.Collections;
using System.Collections.Generic;

public class ANN_Network
{
    public ANN_Neuron[][] netLayers;
    private List<ANN_Synapse> synapsesAll = new List<ANN_Synapse>();

    public List<ANN_Synapse> SynapsesAll { get => synapsesAll; }

    private double learningRate;
    public double LearningRate { get => learningRate; set => learningRate = value; }


    public ANN_Network(int inputs, int[] hiddenLayers, AFType[] afFunctions, int outputs, bool[] addBias)
    {
        netLayers = new ANN_Neuron[hiddenLayers.Length + 2][];

        netLayers[0] = new ANN_Neuron[inputs];

        for (int layerID = 0; layerID < hiddenLayers.Length; layerID++)
        {
            netLayers[layerID + 1] = new ANN_Neuron[hiddenLayers[layerID]];
        }

        netLayers[hiddenLayers.Length + 1] = new ANN_Neuron[outputs];
                
        CreateNeurons(afFunctions);
        InsertBiasNeurons(addBias);
        CreateSynapses();
        SetWeights();
    }


    private void CreateNeurons(AFType[] afFunctions)
    {
        for (int layerID = 0; layerID < netLayers.Length; layerID++)
        {
            for (int neuronID = 0; neuronID < netLayers[layerID].Length; neuronID++)
            {
                netLayers[layerID][neuronID] = AddNeuron(layerID, neuronID, afFunctions[layerID]);
            }
        }
    }


    private void InsertBiasNeurons(bool[] addBias)
    {
        for (int layerID = 1; layerID < netLayers.Length; layerID++)
        {
            int previousLayer = layerID - 1;
            if (addBias[layerID])
            {
                Array.Resize(ref netLayers[previousLayer], netLayers[previousLayer].Length + 1);
                netLayers[previousLayer][netLayers[previousLayer].Length - 1] = AddNeuron(previousLayer, netLayers[previousLayer].Length - 1, AFType.Bias);
            }
        }
    }


    private ANN_Neuron AddNeuron(int layerID, int neuronID, AFType activationFunction)
    {
        ANN_Neuron neuron = new ANN_Neuron();
        neuron.ActivationFunction.ActivationFunctionType = activationFunction;
        neuron.LayerID = layerID;
        neuron.PositionID = neuronID;
        return neuron;
    }


    private void CreateSynapses()
    {
        for (int layerID = 0; layerID < netLayers.Length - 1; layerID++)
        {
            for (int inpNeuronID = 0; inpNeuronID < netLayers[layerID].Length; inpNeuronID++)
            {
                for (int outNeuronID = 0; outNeuronID < netLayers[layerID + 1].Length; outNeuronID++)
                {
                    ANN_Neuron outputNeuron = netLayers[layerID + 1][outNeuronID];
                    if (outputNeuron.ActivationFunction.ActivationFunctionType != AFType.Bias)
                    {
                        ANN_Neuron inputNeuron = netLayers[layerID][inpNeuronID];
                        ANN_Synapse newSynapse = new ANN_Synapse(inputNeuron, outputNeuron);
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