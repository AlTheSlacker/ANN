using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace ANN
{
    public enum AFType
    {
        LeakyReLu = 1,
        Sigmoid = 2,
        Tanh = 3,
        Input = 200,
        Output = 300,
        Bias = 400
    }

    public class AnnActivationFunction
    {
        public AFType ActivationFunctionType { get; set; }
        public double Scalar { get; set; } = 1;

        public double GetAFValue(double x)
        {
            return ((int)ActivationFunctionType) switch
            {
                1 => AFLeakyReLu(x),
                2 => AFSigmoid(x),
                3 => AFTanh(x),
                200 => AFInput(x),
                300 => AFOutput(x),
                400 => 1,// bias node
                _ => 0,
            };
        }


        public double GetAFDerivValue(double x)
        {
            switch ((int)ActivationFunctionType)
            {
                case 1:
                    return DerivAFLeakyReLu(x);
                case 2:
                    return DerivAFSigmoid(x);
                case 3:
                    return DerivAFTanh(x);
                case 200:
                    Debug.Log("The derivative of the input activation function should never be called.");
                    Debug.Log("The input activation function can only be used in LayerID 0.");
                    break;
                case 300:
                    return DerivAFOutput(x);
                case 400:
                    return 1; // bias node
            }
            return 0;
        }


        private double AFLeakyReLu(double x)
        {
            return Math.Max(0, x);
        }


        private double AFSigmoid(double x)
        {
            if (x < -45.0) return 0.0;
            else if (x > 45.0) return 1.0;
            else return 1.0 / (1.0 + Math.Exp(-x));
        }


        private double AFTanh(double x)
        {
            return (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
        }


        private double AFInput(double x)
        {
            return x * Scalar;
        }


        private double AFOutput(double x)
        {
            return x * Scalar;
        }


        private double DerivAFLeakyReLu(double x)
        {
            // if function has a non-zero gradient for negative values then return that... more code needed
            if (x > 0) return 1;
            else return 0;
        }


        private double DerivAFSigmoid(double x)
        {
            double aFSig = AFSigmoid(x);
            return aFSig * (1 - aFSig);
        }


        private double DerivAFTanh(double x)
        {
            return 1 - Math.Pow(AFTanh(x), 2);
        }


        private double DerivAFOutput(double x)
        {
            // NOT the scalar as this would break the back propagation
            // ANN_BackPropagation.cs
            // private void BackPropSingleDataSet(double[] sampleDataSet, double[] targetDataSet)
            // includes the effect of the scalar on output being scaled out
            return 1;
        }
    }
}