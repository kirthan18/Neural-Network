package com.kirthanaa.hw2.entities;

/**
 * Created by kirthanaaraghuraman on 10/21/15.
 */
public class NeuralNetOutput implements Comparable<NeuralNetOutput> {

    /**
     * Index of instance in original data set
     */
    private int mInstanceOrdinal = -1;

    /**
     * Fold to which the instance is assigned to
     */
    private int mFold = -1;

    /**
     * Predicted class of the instance
     */
    private String mPredictedClass = "";

    /**
     * Actual class of the instance
     */
    private String mActualClass = "";

    /**
     * Confidence measure of the prediction for the instance
     */
    private double mConfidence = 0.0;

    /**
     * Initializes NeuralNetOutput object
     *
     * @param ordinal        Index of instance
     * @param fold           Fold of instance
     * @param predictedClass Predicted class of instance
     * @param actualClass    Actual class of instance
     * @param confidence     Confidence of prediction for instance
     */
    public NeuralNetOutput(int ordinal, int fold, String predictedClass, String actualClass, double confidence) {
        this.mInstanceOrdinal = ordinal;
        this.mFold = fold;
        this.mPredictedClass = predictedClass;
        this.mActualClass = actualClass;
        this.mConfidence = confidence;
    }

    /**
     * Returns actual class of instance
     *
     * @return Actual class of instance
     */
    public String getActualClass() {
        return this.mActualClass;
    }

    /**
     * Returns predicted class of instance
     *
     * @return Predicted class of instance
     */
    public String getPredictedClass() {
        return this.mPredictedClass;
    }

    /**
     * Prints output in the desired format of fold, predicted class,
     * actual class and confidence
     */
    public void printOutput() {
        System.out.println(String.format("%2d", (this.mFold + 1)) + " " + this.mPredictedClass + " " + this.mActualClass
                + " " + mConfidence);
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

        /*
            Use  this measure to sort based on confidence for obtaining ROC Values
         */
        /*if (this.mConfidence == output.mConfidence) {
            return 0;
        } else if (this.mConfidence < output.mConfidence) {
            return 1;
        } else {
            return -1;
        }*/
    }

}
