using System.Collections;
using System.Collections.Generic;

namespace ANN
{

    public class AnnFeedForward
    {
        private readonly AnnNetwork net;


        public AnnFeedForward(AnnNetwork targetNetwork)
        {
            net = targetNetwork;
        }


        public void FeedForward(double[] inputs)
        {
            SetInputs(inputs);

            for (int layerID = 0; layerID < net.NetLayers.Length; layerID++)
            {
                NeuronsUpdateByLayer(layerID);
                LayerInputReset(layerID + 1);
                for (int neuronID = 0; neuronID < net.NetLayers[layerID].Length; neuronID++)
                {
                    ActivateOutputSynapses(net.NetLayers[layerID][neuronID].OutputSynapses);
                }
            }
        }


        public double[] GetOutputs()
        {
            int outputCount = net.NetLayers[net.NetLayers.Length - 1].Length;
            double[] outputs = new double[outputCount];

            for (int neuronID = 0; neuronID < outputCount; neuronID++)
            {
                outputs[neuronID] = net.NetLayers[net.NetLayers.Length - 1][neuronID].Output;
            }
            return outputs;
        }


        private void SetInputs(double[] inputs)
        {
            for (int neuronID = 0; neuronID < inputs.Length; neuronID++)
            {
                net.NetLayers[0][neuronID].Input = inputs[neuronID];
            }
        }


        private void ActivateOutputSynapses(List<AnnSynapse> outputSynapses)
        {
            for (int synapseIndex = 0; synapseIndex < outputSynapses.Count; synapseIndex++)
            {
                outputSynapses[synapseIndex].FireSynapse();
            }
        }


        private void NeuronsUpdateByLayer(int layerID)
        {
            for (int neuronID = 0; neuronID < net.NetLayers[layerID].Length; neuronID++)
            {
                net.NetLayers[layerID][neuronID].UpdateOutput();
            }
        }


        private void LayerInputReset(int layerID)
        {
            if (layerID > net.NetLayers.Length - 1) return;
            for (int neuronID = 0; neuronID < net.NetLayers[layerID].Length; neuronID++)
            {
                net.NetLayers[layerID][neuronID].Input = 0;
            }
        }

    }
}