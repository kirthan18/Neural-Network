package com.kirthanaa.hw3.entities;

/**
 * Created by kirthanaaraghuraman on 10/21/15.
 */
public class NeuralNetOutput {

    private int mInstanceOrdinal = -1;

    private int mFold = -1;

    private String mPredictedClass = "";

    private String mActualClass = "";

    private double mConfidence = 0.0;

    public NeuralNetOutput(int ordinal, int fold, String predictedClass, String actualClass, double confidence){
        this.mInstanceOrdinal = ordinal;
        this.mFold = fold;
        this.mPredictedClass = predictedClass;
        this.mActualClass = actualClass;
        this.mConfidence = confidence;
    }
}
