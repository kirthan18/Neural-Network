package com.kirthanaa.hw2.entities;

/**
 * Created by kirthanaaraghuraman on 10/28/15.
 */
public class ROCValues {

    private double mTruePositiveRate;

    private double mFalsePositiveRate;

    public ROCValues(double truePositiveRate, double falsePositiveRate){
        this.mTruePositiveRate = truePositiveRate;
        this.mFalsePositiveRate = falsePositiveRate;
    }

    public double getFalsePositiveRate() {
        return this.mFalsePositiveRate;
    }

    public double getTruePositiveRate() {
        return this.mTruePositiveRate;
    }
}
