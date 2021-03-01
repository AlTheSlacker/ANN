using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace ANN
{

    public class AnnBackPropagation
    {
        private readonly AnnNetwork net;


        public AnnBackPropagation(AnnNetwork targetNetwork)
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
            AnnFeedForward feedForwardModel = new AnnFeedForward(net);
            feedForwardModel.FeedForward(sampleDataSet);
            int outputLayerID = net.NetLayers.Length - 1;
            double[] errors = new double[net.NetLayers[outputLayerID].Length];

            // calculate output deltas
            for (int outNeuronID = 0; outNeuronID < net.NetLayers[outputLayerID].Length; outNeuronID++)
            {
                AnnNeuron neuron = net.NetLayers[outputLayerID][outNeuronID];
                double derivAFOutput = neuron.ActivationFunction.GetAFDerivValue(neuron.Input);
                double outputScalar = neuron.ActivationFunction.Scalar;
                double derivError = -(targetDataSet[outNeuronID] / outputScalar - neuron.Output / outputScalar);
                if (targetDataSet[outNeuronID] != 0) errors[outNeuronID] += (targetDataSet[outNeuronID] - neuron.Output) / targetDataSet[outNeuronID];
                else errors[outNeuronID] += 0;
                neuron.BackPropDelta = derivAFOutput * derivError;
            }

            // calculate remaining deltas
            for (int layerID = outputLayerID - 1; layerID > 0; layerID--)
            {
                for (int neuronID = 0; neuronID < net.NetLayers[layerID].Length; neuronID++)
                {
                    net.NetLayers[layerID][neuronID].BackPropDeltaUpdate();
                }
            }

            // add delta * input neuron output to the synapse deltainput list
            for (int synapseID = 0; synapseID < net.SynapsesAll.Count; synapseID++)
            {
                AnnSynapse synapse = net.SynapsesAll[synapseID];
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
}