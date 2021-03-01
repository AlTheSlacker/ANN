using System.Collections;
using System.Collections.Generic;
namespace ANN
{


    public class AnnNeuron
    {
        private double output = 0;
        private readonly List<AnnSynapse> inputSynapses;
        public double Input { get; set; } = 0;
        public double Output { get => output; }
        public AnnActivationFunction ActivationFunction { get; set; }
        public int LayerID { get; set; }
        public int PositionID { get; set; }
        public List<AnnSynapse> OutputSynapses { get; }
        public List<AnnSynapse> InputSynapses { get => inputSynapses; }
        public double BackPropDelta { get; set; }

        public AnnNeuron()
        {
            OutputSynapses = new List<AnnSynapse>();
            inputSynapses = new List<AnnSynapse>();
            ActivationFunction = new AnnActivationFunction();
        }


        public void AddInput(double additionalInput)
        {
            Input += additionalInput;
        }


        public void UpdateOutput()
        {
            output = ActivationFunction.GetAFValue(Input);
        }


        public void AddOutputSynapse(AnnSynapse synapse)
        {
            OutputSynapses.Add(synapse);
        }


        public void AddInputSynapse(AnnSynapse synapse)
        {
            inputSynapses.Add(synapse);
        }


        public void BackPropDeltaUpdate()
        {
            double delta = 0;
            double derivAF = ActivationFunction.GetAFDerivValue(Input);
            for (int synapseID = 0; synapseID < OutputSynapses.Count; synapseID++)
            {
                delta += derivAF * OutputSynapses[synapseID].Weight * OutputSynapses[synapseID].OutputNeuron.BackPropDelta;
            }
            BackPropDelta = delta;
        }

    }
}