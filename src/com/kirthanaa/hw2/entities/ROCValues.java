package com.kirthanaa.hw2.entities;

/**
 * Created by kirthanaaraghuraman on 10/28/15.
 */
public class ROCValues {

    /**
     * True positive rate value
     */
    private double mTruePositiveRate;

    /**
     * False positive rate value
     */
    private double mFalsePositiveRate;

    /**
     * Initializes ROCValues object
     *
     * @param truePositiveRate  True positive rate value
     * @param falsePositiveRate False positive rate value
     */
    public ROCValues(double truePositiveRate, double falsePositiveRate) {
        this.mTruePositiveRate = truePositiveRate;
        this.mFalsePositiveRate = falsePositiveRate;
    }

    /**
     * Returns False positive rate value
     *
     * @return False positive rate value
     */
    public double getFalsePositiveRate() {
        return this.mFalsePositiveRate;
    }

    /**
     * Returns True positive rate value
     *
     * @return True positive rate value
     */
    public double getTruePositiveRate() {
        return this.mTruePositiveRate;
    }
}
