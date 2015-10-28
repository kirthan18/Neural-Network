package com.kirthanaa.hw2.entities;

/**
 * Created by kirthanaaraghuraman on 10/21/15.
 */
public class NeuralNetOutput implements Comparable<NeuralNetOutput>{

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

    @Override
    public int compareTo(NeuralNetOutput output) {
        if (this.mInstanceOrdinal == output.mInstanceOrdinal) {
            return 0;
        } else if (this.mInstanceOrdinal > output.mInstanceOrdinal) {
            return 1;
        } else {
            return -1;
        }
    }

    public String getActualClass() {
        return this.mActualClass;
    }

    public String getPredictedClass() {
        return this.mPredictedClass;
    }

    public void printOutput(){
        System.out.println(String.format("%2d",(this.mFold + 1)) + " " + this.mPredictedClass + " " + this.mActualClass
                + " " + mConfidence);
    }
}
