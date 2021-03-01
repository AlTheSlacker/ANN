using System.Collections;
using System.Collections.Generic;

namespace ANN
{

    public class AnnSynapse
    {
        private readonly AnnNeuron inputNeuron;
        private readonly AnnNeuron outputNeuron;
        private readonly List<double> deltaInput;

        public double Weight { get; set; }
        public AnnNeuron InputNeuron { get => inputNeuron; }
        public AnnNeuron OutputNeuron { get => outputNeuron; }


        public AnnSynapse(AnnNeuron requestedInputNeuron, AnnNeuron requestedOutputNeuron)
        {
            deltaInput = new List<double>();
            inputNeuron = requestedInputNeuron;
            outputNeuron = requestedOutputNeuron;
            Weight = 1;
        }


        public void EpochWeightsProcess(double LearningRate)
        {
            Weight -= LearningRate * ListAverage();
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
}