using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public enum AFType
{
    LeakyReLu = 1,
    Sigmoid = 2,
    Tanh = 3,
    Input = 200,
    Output = 300,
    Bias = 400
}

public class ANN_ActivationFunction
{

    private AFType activationFunctionType;

    public AFType ActivationFunctionType { get => activationFunctionType; set => activationFunctionType = value; }
    public double scalar = 1;

    public double GetAFValue(double x)
    {
        switch ((int)ActivationFunctionType)
        {
            case 1:
                return AFLeakyReLu(x);
            case 2:
                return AFSigmoid(x);
            case 3:
                return AFTanh(x);
            case 200:
                return AFInput(x);
            case 300:
                return AFOutput(x);
            case 400:
                return 1;  // bias node
        }
        return 0;
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
        return x * scalar;
    }


    private double AFOutput(double x)
    {
        return x * scalar;
    }


    private double DerivAFLeakyReLu(double x)
    {
        return Math.Max(0, 1);
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