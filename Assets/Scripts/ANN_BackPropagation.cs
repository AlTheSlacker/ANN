using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN_BackPropagation
{
    private ANN_Network net;


    public ANN_BackPropagation(ANN_Network targetNetwork)
    {
        net = targetNetwork;
    }


    public void BackPropagate(double[][] sampleData, double[][] targetData, int epochs, double maxError)
    {
        double[] sampleDataSet;
        double[] targetDataSet;
        double[] epochError = new double[targetData.Length];
        int epoch;

        for (epoch = 0; epoch < epochs; epoch++)
        {
            ClearSynapseDeltas();

            for (int dataSetID = 0; dataSetID < sampleData.Length; dataSetID++)
            {
                sampleDataSet = sampleData[dataSetID];
                targetDataSet = targetData[dataSetID];
                epochError[dataSetID] = BackPropSingleDataSet(sampleDataSet, targetDataSet);
            }
            if (MaxPercentageError(epochError) < maxError) break;
            UpdateSynapseWeights();
        }
        Debug.Log("Max Error: " + MaxPercentageError(epochError) + " @ epoch: " + epoch);
        
    }


    private double BackPropSingleDataSet(double[] sampleDataSet, double[] targetDataSet)
    {
        ANN_FeedForward feedForwardModel = new ANN_FeedForward(net);
        feedForwardModel.FeedForward(sampleDataSet);
        int outputLayerID = net.netLayers.Length - 1;
        double[] errors = new double[net.netLayers[outputLayerID].Length];

        // calculate output deltas
        for (int outNeuronID = 0; outNeuronID < net.netLayers[outputLayerID].Length; outNeuronID++)
        {
            ANN_Neuron neuron = net.netLayers[outputLayerID][outNeuronID];
            double derivAFOutput = neuron.ActivationFunction.GetAFDerivValue(neuron.Input);
            double outputScalar = neuron.ActivationFunction.scalar;
            double derivError = -(targetDataSet[outNeuronID] / outputScalar - neuron.Output / outputScalar);
            if (targetDataSet[outNeuronID] != 0) errors[outNeuronID] += (targetDataSet[outNeuronID] - neuron.Output) / targetDataSet[outNeuronID];
            else errors[outNeuronID] += 0;
            neuron.BackPropDelta = derivAFOutput * derivError;
        }

        // calculate remaining deltas
        for (int layerID = outputLayerID - 1; layerID > 0; layerID--)
        {
            for (int neuronID = 0; neuronID < net.netLayers[layerID].Length; neuronID++)
            {
                net.netLayers[layerID][neuronID].BackPropDeltaUpdate();
            }
        }

        // add delta * input neuron output to the synapse deltainput list
        for (int synapseID = 0; synapseID < net.SynapsesAll.Count; synapseID++)
        {
            ANN_Synapse synapse = net.SynapsesAll[synapseID];
            synapse.DeltaInputAdd(synapse.OutputNeuron.BackPropDelta * synapse.InputNeuron.Output);
        }

        return MaxPercentageError(errors);

    }


    private double MaxPercentageError(double[] errors)
    {
        double error = 0;
        for (int i = 0; i < errors.Length; i++)
        {
            if (Math.Abs(errors[i]) > error) error = Math.Abs(errors[i]);
        }
        return error;
    }

    private void UpdateSynapseWeights()
    {
        for (int synapseID = 0; synapseID < net.SynapsesAll.Count; synapseID++)
        {
            net.SynapsesAll[synapseID].EpochWeightsProcess(net.LearningRate);
        }
    }


    private void ClearSynapseDeltas()
    {
        for (int synapseID = 0; synapseID < net.SynapsesAll.Count; synapseID++)
        {
            net.SynapsesAll[synapseID].DeltaInputClear();
        }
    }

}
