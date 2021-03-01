using System;
using UnityEngine;

namespace ANN
{

    public class AnnController : MonoBehaviour
    {
        [SerializeField] private bool train = false;
        [SerializeField] private bool loadWeightAF = false;
        [SerializeField] private string weightAFFile = ".\\weights.txt";
        [SerializeField] private bool saveWeightAF = false;
        [SerializeField] private int inputs = 2;
        [SerializeField] private int[] hiddenLayers = new int[] { 3, 3 };
        [SerializeField] private bool[] addBias = new bool[] { false, true, true, false };
        [SerializeField] private AFType[] afFunctions = new AFType[] { AFType.Input, AFType.Sigmoid, AFType.Sigmoid, AFType.Output };
        [SerializeField] private int outputs = 3;
        [SerializeField] private double learningRate = 0.05;
        [SerializeField] private int epochs = 20000;
        [SerializeField] private double maxError = 0.03;

        private AnnNetwork net;
        private double[][] sampledata;
        private double[][] targetdata;

        void Awake()
        {
            net = new AnnNetwork(inputs, hiddenLayers, afFunctions, outputs, addBias)
            {
                LearningRate = learningRate
            };

        }

        void Start()
        {

            AnnVisController ann_VisController = GetComponent<AnnVisController>();
            ann_VisController.VisArchitecture(net);

            if (loadWeightAF) LoadWeightAF();
            if (train) TrainNetwork();
            if (saveWeightAF) SaveWeightAF();

            // test stuff, manually setting input values
            double[] input_data = new double[] { 20 };

            // DEBUG remove later
            AnnFeedForward feedForwardModel = new AnnFeedForward(net);
            feedForwardModel.FeedForward(input_data);

            ann_VisController.NeuronTextOutput();
            ann_VisController.SynapseTextOutput();

        }


        private void CreateData()
        {
            // create training data

            // x2 test
            //sampledata = new double[10][] { new double[] { 0.0 }, new double[] { 10.0 }, new double[] { 20.0 }, new double[] { 30.0 }, new double[] { 40.0 }, new double[] { 50.0 }, new double[] { 60.0 }, new double[] { 70.0 }, new double[] { 80.0 }, new double[] { 90.0 } };
            //targetdata = new double[10][] { new double[] { 0.0 }, new double[] { 20.0 }, new double[] { 40.0 }, new double[] { 60.0 }, new double[] { 80.0 }, new double[] { 100.0 }, new double[] { 120.0 }, new double[] { 140.0 }, new double[] { 160.0 }, new double[] { 180.0 } };


            // sine test
            sampledata = new double[10][] { new double[] { 0.0 }, new double[] { 10.0 }, new double[] { 20.0 }, new double[] { 30.0 }, new double[] { 40.0 }, new double[] { 50.0 }, new double[] { 60.0 }, new double[] { 70.0 }, new double[] { 80.0 }, new double[] { 90.0 } };
            targetdata = new double[10][] { new double[] { 0.0 }, new double[] { 0.173648 }, new double[] { 0.34202 }, new double[] { 0.5 }, new double[] { 0.642788 }, new double[] { 0.766044 }, new double[] { 0.866025 }, new double[] { 0.93969 }, new double[] { 0.9848 }, new double[] { 1.0 } };


            NormalizeWithSignInputNodes();
            NormalizeWithSignOutputNodes();

        }


        private void NormalizeWithSignInputNodes()
        {
            double maxABSVal = 0;
            double dataVal;

            for (int inputNeuronID = 0; inputNeuronID < sampledata[0].Length; inputNeuronID++)
            {
                for (int dataSet = 0; dataSet < sampledata.Length; dataSet++)
                {
                    dataVal = sampledata[dataSet][inputNeuronID];
                    if (Math.Abs(dataVal) > Math.Abs(maxABSVal)) maxABSVal = Math.Abs(dataVal);
                }
                net.NetLayers[0][inputNeuronID].ActivationFunction.Scalar = 1 / maxABSVal;
            }
        }


        private void NormalizeWithSignOutputNodes()
        {
            double maxABSVal = 0;
            double dataVal;

            for (int outputNeuronID = 0; outputNeuronID < targetdata[0].Length; outputNeuronID++)
            {
                for (int dataSet = 0; dataSet < targetdata.Length; dataSet++)
                {
                    dataVal = targetdata[dataSet][outputNeuronID];
                    if (Math.Abs(dataVal) > Math.Abs(maxABSVal)) maxABSVal = Math.Abs(dataVal);
                }
                net.NetLayers[net.NetLayers.Length - 1][outputNeuronID].ActivationFunction.Scalar = maxABSVal;
            }
        }


        private void TrainNetwork()
        {
            CreateData();

            AnnBackPropagation backPropModel = new AnnBackPropagation(net);
            backPropModel.BackPropagate(sampledata, targetdata, epochs, maxError);

            print("Training complete");
        }


        private void SaveWeightAF()
        {
            System.IO.StreamWriter file = new System.IO.StreamWriter(weightAFFile);

            for (int i = 0; i < net.SynapsesAll.Count; i++)
            {
                file.WriteLine(net.SynapsesAll[i].Weight);
            }

            for (int layerID = 0; layerID < net.NetLayers.Length; layerID++)
            {
                for (int neuronID = 0; neuronID < net.NetLayers[layerID].Length; neuronID++)
                {
                    file.WriteLine(net.NetLayers[layerID][neuronID].ActivationFunction.Scalar);
                    file.WriteLine(net.NetLayers[layerID][neuronID].ActivationFunction.ActivationFunctionType);
                }
            }

            file.Close();
        }


        private void LoadWeightAF()
        {
            if (!System.IO.File.Exists(weightAFFile)) return;

            System.IO.StreamReader file = new System.IO.StreamReader(weightAFFile);

            for (int i = 0; i < net.SynapsesAll.Count; i++)
            {
                net.SynapsesAll[i].Weight = Convert.ToDouble(file.ReadLine());
            }

            for (int layerID = 0; layerID < net.NetLayers.Length; layerID++)
            {
                for (int neuronID = 0; neuronID < net.NetLayers[layerID].Length; neuronID++)
                {
                    net.NetLayers[layerID][neuronID].ActivationFunction.Scalar = Convert.ToDouble(file.ReadLine());
                    net.NetLayers[layerID][neuronID].ActivationFunction.ActivationFunctionType = (AFType)Enum.Parse(typeof(AFType), file.ReadLine());
                }
            }

            file.Close();
        }

    }
}