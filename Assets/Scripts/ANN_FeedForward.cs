using System.Collections;
using System.Collections.Generic;

public class ANN_FeedForward
{
    private ANN_Network net;


    public ANN_FeedForward(ANN_Network targetNetwork)
    {
        net = targetNetwork;
    }


    public void FeedForward(double[] inputs)
    {
        SetInputs(inputs);

        for (int layerID = 0; layerID < net.netLayers.Length; layerID++)
        {
            NeuronsUpdateByLayer(layerID);
            LayerInputReset(layerID + 1);
            for (int neuronID = 0; neuronID < net.netLayers[layerID].Length; neuronID++)
            {
                ActivateOutputSynapses(net.netLayers[layerID][neuronID].OutputSynapses);
            }
        }
    }


    public double[] GetOutputs()
    {
        int outputCount = net.netLayers[net.netLayers.Length - 1].Length;
        double[] outputs = new double[outputCount];

        for (int neuronID = 0; neuronID < outputCount; neuronID++)
        {
            outputs[neuronID] = net.netLayers[net.netLayers.Length - 1][neuronID].Output;
        }
        return outputs;
    }


    private void SetInputs(double[] inputs)
    {
        for (int neuronID = 0; neuronID < inputs.Length; neuronID++)
        {
            net.netLayers[0][neuronID].Input = inputs[neuronID];
        }
    }


    private void ActivateOutputSynapses(List<ANN_Synapse> outputSynapses)
    {
        for (int synapseIndex = 0; synapseIndex < outputSynapses.Count; synapseIndex++)
        {
            outputSynapses[synapseIndex].FireSynapse();
        }
    }

    
    private void NeuronsUpdateByLayer(int layerID)
    {
        for (int neuronID = 0; neuronID < net.netLayers[layerID].Length; neuronID++)
        {
            net.netLayers[layerID][neuronID].UpdateOutput();
        }
    }


    private void LayerInputReset(int layerID)
    {
        if (layerID > net.netLayers.Length - 1) return;
        for (int neuronID = 0; neuronID < net.netLayers[layerID].Length; neuronID++)
        {
            net.netLayers[layerID][neuronID].Input = 0;
        }
    }

}
