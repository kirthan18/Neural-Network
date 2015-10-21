package com.kirthanaa.hw3.neuralnet;

import com.kirthanaa.hw3.arffreader.ARFFReader;
import com.kirthanaa.hw3.entities.Fractions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by kirthanaaraghuraman on 10/20/15.
 */
public class NeuralNetwork {

    private static final int NUM_CROSS_VALIDATION = 10;

    private static int mPositiveInstances = 0;

    private static int mNegativeInstances = 0;

    private static double mClassRatio = 0.0;

    private ArrayList<ArrayList<HashMap<String, Double>>> mStratifiedSamples = null;

    private static int mTestSampleIndex = -1;

    /**
     * Returns the output after applying sigmoidal function to the value given by argument
     *
     * @param value Value for which sigmoidal function is to be calculated
     * @return Value after applying sigmoidal function
     */
    private static double getSigmoidalOutput(double value) {
        return (1.0 / (1.0 + Math.pow(Math.E, -value)));
    }

    /**
     * Returns a random number in the range [min, max]
     *
     * @param min Lower bound of range
     * @param max Upper bound of range
     * @return Random number in the range [min, max]
     */
    private static int getRandomNumber(int min, int max) {
        Random random = new Random();
        return (random.nextInt((max - min) + 1) + min);
    }

    /**
     * Implementing Fisher–Yates shuffle to shuffle the contents of the array
     * @param ar Array to be shuffled
     */
    private static void shuffleArray(int[] ar, int length)
    {
        Random rnd = ThreadLocalRandom.current();
        for (int i = length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            int a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }


    private static void createStratifiedSamples(ARFFReader arffReader) {
        mTestSampleIndex = getRandomNumber(0, NUM_CROSS_VALIDATION - 1);

        int[] positiveInstanceIndex = new int[arffReader.getNumberOfDataInstances()];
        int[] negativeInstanceIndex = new int[arffReader.getNumberOfDataInstances()];

        int negIndex = 0;
        int posIndex = 0;

        for (int i = 0; i < arffReader.getNumberOfDataInstances(); i++) {
            if (arffReader.mClassLabelList.get(i).equalsIgnoreCase(arffReader.mClassLabels[0])) {
                mNegativeInstances++;
                negativeInstanceIndex[negIndex++] = i;
            } else if (arffReader.mClassLabelList.get(i).equalsIgnoreCase(arffReader.mClassLabels[1])) {
                mPositiveInstances++;
                positiveInstanceIndex[posIndex++] = i;
            }
        }
        mClassRatio = (double)mPositiveInstances/(double)mNegativeInstances;

        shuffleArray(positiveInstanceIndex, posIndex);
        shuffleArray(negativeInstanceIndex, negIndex);

        Fractions posToNeg = new Fractions(mPositiveInstances, mNegativeInstances);

        int numInstancesPerFold = arffReader.getNumberOfDataInstances() / 10;
        int numPosInstancesPerFold = 0;
        int numNegInstancesPerFold = 0;

        for(int i = 0; i < NUM_CROSS_VALIDATION; i++){
            for(int j = 0; j < numInstancesPerFold; j++){

            }
        }
    }


    private static void trainNeuralNet(){

    }

    private static void testNeuralNet(){

    }


    public static void main(String args[]) {
        ARFFReader arffReader = ARFFReader.getInstance("/Users/kirthanaaraghuraman/Documents/CS760/HW#2/src/com/kirthanaa/hw3/trainingset/sonar.arff");
        arffReader.parseARFFFile();

        createStratifiedSamples(arffReader);

        trainNeuralNet();

        testNeuralNet();

    }
}
