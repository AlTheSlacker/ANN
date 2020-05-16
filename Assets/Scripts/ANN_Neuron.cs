using System.Collections;
using System.Collections.Generic;


public class ANN_Neuron 
{
    private double input = 0;
    private double output = 0;
    private ANN_ActivationFunction activationFunction;
    private int layerID;
    private int positionID;
    private List<ANN_Synapse> outputSynapses;
    private List<ANN_Synapse> inputSynapses;

    public double Input { get => input; set => input = value; }
    public double Output { get => output; }
    public ANN_ActivationFunction ActivationFunction { get => activationFunction; set => activationFunction = value; }
    public int LayerID { get => layerID; set => layerID = value; }
    public int PositionID { get => positionID; set => positionID = value; }
    public List<ANN_Synapse> OutputSynapses { get => outputSynapses; }
    public List<ANN_Synapse> InputSynapses { get => inputSynapses; }
    public double BackPropDelta;

    public ANN_Neuron()
    {
        outputSynapses = new List<ANN_Synapse>();
        inputSynapses = new List<ANN_Synapse>();
        activationFunction = new ANN_ActivationFunction();
    }


    public void AddInput(double additionalInput)
    {
        input += additionalInput;
    }


    public void UpdateOutput()
    {
        output = activationFunction.GetAFValue(input);
    }


    public void AddOutputSynapse(ANN_Synapse synapse)
    {
        outputSynapses.Add(synapse);
    }


    public void AddInputSynapse(ANN_Synapse synapse)
    {
        inputSynapses.Add(synapse);
    }


    public void BackPropDeltaUpdate()
    {
        double delta = 0;
        double derivAF = ActivationFunction.GetAFDerivValue(Input);
        for (int synapseID = 0; synapseID < outputSynapses.Count; synapseID++)
        {
            delta += derivAF * outputSynapses[synapseID].Weight * outputSynapses[synapseID].OutputNeuron.BackPropDelta;
        }
        BackPropDelta = delta;
    }

}
