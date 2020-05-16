using System.Collections;
using System.Collections.Generic;

public class ANN_Synapse
{
    private double weight;
    private ANN_Neuron inputNeuron;
    private ANN_Neuron outputNeuron;
    private List<double> deltaInput;

    public double Weight { get => weight; set => weight = value; }
    public ANN_Neuron InputNeuron { get => inputNeuron; }
    public ANN_Neuron OutputNeuron { get => outputNeuron; }


    public ANN_Synapse(ANN_Neuron requestedInputNeuron, ANN_Neuron requestedOutputNeuron)
    {
        deltaInput = new List<double>();
        inputNeuron = requestedInputNeuron;
        outputNeuron = requestedOutputNeuron;
        Weight = 1;
    }


    public void EpochWeightsProcess(double LearningRate)
    {
        Weight = Weight - LearningRate * ListAverage();
    }


    public double ListAverage()
    {
        if (deltaInput.Count == 0) return 0;
        double total = 0;

        for (int i = 0; i < deltaInput.Count; i++)
        {
            total += deltaInput[i];
        }
        return total / deltaInput.Count;
    }


    public void DeltaInputAdd(double additionalDelta)
    {
        deltaInput.Add(additionalDelta);
    }


    public void DeltaInputClear()
    {
        deltaInput.Clear();
    }


    public void FireSynapse()
    {
        outputNeuron.AddInput(InputNeuron.Output * Weight);
    }

}
