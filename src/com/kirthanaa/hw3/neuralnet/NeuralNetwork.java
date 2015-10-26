package com.kirthanaa.hw3.neuralnet;

import com.kirthanaa.hw3.arffreader.ARFFReader;
import com.kirthanaa.hw3.entities.NeuralNetOutput;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by kirthanaaraghuraman on 10/20/15.
 */
public class NeuralNetwork {

    private static int mFolds = 12;

    private static int mEpochs = 1000;

    private static double mLearningRate = 0.1;

    private static int mPositiveInstances = 0;

    private static int mNegativeInstances = 0;

    private static int foldList[] = null;

    private static ArrayList<ArrayList<HashMap<String, Double>>> mStratifiedSamples = null;

    private static double[] mWeights = null;

    private static double mBias = 0.0;

    private static ArrayList<ArrayList<Integer>> mIndexList = null;

    private static ArrayList<NeuralNetOutput> mClassificationOutput = null;

    private static int mCorrectlyClassified = 0;


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
     *
     * @param ar Array to be shuffled
     */
    private static void shuffleArray(int[] ar, int length) {
        Random rnd = ThreadLocalRandom.current();
        for (int i = length - 1; i > 0; i--) {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            int a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
        }
    }


    /**
     * Creates a 10 fold stratified sample set for cross validation
     *
     * @param arffReader ARFF Reader instance containing details about the learning set
     */
    private static void createStratifiedSamples(ARFFReader arffReader) {
        mStratifiedSamples = new ArrayList<>(mFolds);
        foldList = new int[arffReader.getNumberOfDataInstances()];

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
        //mClassRatio = (double) mPositiveInstances / (double) mNegativeInstances;

        shuffleArray(positiveInstanceIndex, posIndex);
        shuffleArray(negativeInstanceIndex, negIndex);

        double fractionPosInstance = (double) posIndex / (double) arffReader.getNumberOfDataInstances();
        double fractionNegInstance = (double) negIndex / (double) arffReader.getNumberOfDataInstances();

        int numInstancesPerFold = arffReader.getNumberOfDataInstances() / mFolds;
        int numPosInstancesPerFold = 0;
        int numNegInstancesPerFold = 0;

        if (fractionPosInstance > fractionNegInstance) {
            numPosInstancesPerFold = (int) (fractionPosInstance * numInstancesPerFold) + 1;
            numNegInstancesPerFold = (int) (fractionNegInstance * numInstancesPerFold);
        } else {
            numPosInstancesPerFold = (int) (fractionPosInstance * numInstancesPerFold);
            numNegInstancesPerFold = (int) (fractionNegInstance * numInstancesPerFold) + 1;
        }

        int posInstanceIndex = 0;
        int negInstanceIndex = 0;

        mIndexList = new ArrayList<>(mFolds);
        for (int i = 0; i < mFolds; i++) {

            int l = 0;
            ArrayList<HashMap<String, Double>> instancesListPerFold = new ArrayList<>();
            ArrayList<Integer> instancesIndexPerFold = new ArrayList<>();
            while (l < numInstancesPerFold) {

                for (int k = 0; k < numPosInstancesPerFold; k++) {
                    if (posInstanceIndex < posIndex) {
                        instancesIndexPerFold.add(positiveInstanceIndex[posInstanceIndex]);
                        foldList[positiveInstanceIndex[posInstanceIndex]] = i;
                        instancesListPerFold.add(arffReader.getDataInstanceList().get(positiveInstanceIndex[posInstanceIndex++]));
                        l++;
                    }
                }

                for (int n = 0; n < numNegInstancesPerFold; n++) {
                    if (negInstanceIndex < negIndex) {
                        foldList[negativeInstanceIndex[negInstanceIndex]] = i;
                        instancesIndexPerFold.add(negativeInstanceIndex[negInstanceIndex]);
                        instancesListPerFold.add(arffReader.getDataInstanceList().get(negativeInstanceIndex[negInstanceIndex++]));
                        l++;
                    }
                }
            }
            mStratifiedSamples.add(i, instancesListPerFold);
            mIndexList.add(i, instancesIndexPerFold);
        }
        int k = 0;
        for (int i = posInstanceIndex; i < posIndex; i++) {
            mIndexList.get(k).add(positiveInstanceIndex[i]);
            foldList[i] = k;
            mStratifiedSamples.get(k).add(arffReader.getDataInstanceList().get(positiveInstanceIndex[i]));
            k = (k + 1) % mFolds;
        }

        for (int i = negInstanceIndex; i < negIndex; i++) {
            foldList[i] = k;
            mIndexList.get(k).add(negativeInstanceIndex[i]);
            mStratifiedSamples.get(k).add(arffReader.getDataInstanceList().get(negativeInstanceIndex[i]));
            k = (k + 1) % mFolds;
        }
        /*for(int i = 0; i < mStratifiedSamples.size(); i++){
            for (int j = 0; j < mStratifiedSamples.get(i).size(); j++){
                shuffleList(mStratifiedSamples.get(i).get(j));
            }
        }*/

    }

    private static void crossValidate(ARFFReader arffReader) {
        mClassificationOutput = new ArrayList<>(arffReader.getNumberOfDataInstances());
        for (int i = 0; i < mFolds; i++) {
            mBias = 0.1;
            mWeights = new double[arffReader.getNumberOfAttributes()];
            Arrays.fill(mWeights, 0.1);
            int testFold = i;

            for (int j = 0; j < mEpochs; j++) {
                trainNeuralNet(arffReader, testFold);
            }
            /*System.out.println("\nFinal Weights: ");
            for(double weight : mWeights){
                System.out.println(weight);
            }*/
            testNeuralNet(testFold, arffReader);
        }
    }

    private static void updateWeightAndBias(double delta, ARFFReader arffReader, int instanceIndex) {

        for (int i = 0; i < arffReader.getNumberOfAttributes(); i++) {
            double delWeight = delta * mLearningRate * arffReader.getDataInstanceList().get(instanceIndex)
                    .get(arffReader.getAttributeList().get(i).getAttributeName());
            mWeights[i] = mWeights[i] + delWeight;
        }
        mBias = mBias + delta;
    }

    private static void trainNeuralNet(ARFFReader arffReader, int testFold) {
        /*for (int m = 0; m < mStratifiedSamples.size(); m++) {
            if (m == testFold) {
                continue;
            } else {
                for (int n = 0; n < mStratifiedSamples.get(m).size(); n++) {
                    double predictedOutput = -1;
                    double actualOutput = -1;
                    double delta = 0.0;

                    if (arffReader.mClassLabelList.get(mIndexList.get(m).get(n)).equalsIgnoreCase("Rock")) {
                        predictedOutput = 0.0;
                    } else if (arffReader.mClassLabelList.get(mIndexList.get(m).get(n)).equalsIgnoreCase("Mine")) {
                        predictedOutput = 1.0;
                    }

                    ArrayList<HashMap<String, Double>> currentFoldSampleList = mStratifiedSamples.get(m);
                    for (int k = 0; k < currentFoldSampleList.size(); k++) {
                        double weightSum = 0.0;

                        for (int l = 0; l < currentFoldSampleList.get(k).size(); l++) {
                            double attributeValue = currentFoldSampleList.get(k).get(arffReader.getAttributeList().get(l).getAttributeName());
                            weightSum = weightSum + (attributeValue * mWeights[l]);
                        }
                        //System.out.println("Weight sum : " + weightSum);
                        weightSum = weightSum + (1 * mBias);
                        actualOutput = getSigmoidalOutput(weightSum);

                        if(actualOutput > 0.5){
                            actualOutput = 1.0;
                        }else{
                            actualOutput = 0.0;
                        }

                        delta = (actualOutput) * (1 - actualOutput) * (predictedOutput - actualOutput);
                        //System.out.println("Delta : " + delta);
                        updateWeightAndBias(delta, arffReader, mIndexList.get(m).get(k));

                        //System.out.println("Sum of weights for instance " + k + " is : " + weightSum);
                    }

                }
            }
        }*/

        for (int i = 0; i < arffReader.getNumberOfDataInstances(); i++) {
            if (foldList[i] == testFold) {
                continue;
            } else {
                double weightSum = 0.0;
                double predictedOutput = -1;

                if (arffReader.mClassLabelList.get(i).equalsIgnoreCase("Rock")) {
                    predictedOutput = 0.0;
                } else if (arffReader.mClassLabelList.get(i).equalsIgnoreCase("Mine")) {
                    predictedOutput = 1.0;
                }
                for (int j = 0; j < arffReader.getDataInstanceList().get(i).size(); j++) {
                    weightSum = weightSum + (arffReader.getDataInstanceList().get(i).get(arffReader.getAttributeList().get(j).getAttributeName()) * mWeights[j]);
                }
                weightSum = weightSum + (1 * mBias);
                double actualOutput = getSigmoidalOutput(weightSum);
                double delta = (actualOutput) * (1 - actualOutput) * (predictedOutput - actualOutput);
                updateWeightAndBias(delta, arffReader, i);
            }
        }
    }

    private static void testNeuralNet(int testFold, ARFFReader arffReader) {
        /*ArrayList<HashMap<String, Double>> testFoldInstanceList = mStratifiedSamples.get(testFold);
        ArrayList<Integer> testFoldInstanceIndexList = mIndexList.get(testFold);

        for (int i = 0; i < testFoldInstanceList.size(); i++) {
            double weightSum = 0.0;
            for (int j = 0; j < arffReader.getNumberOfAttributes(); j++) {
                weightSum = weightSum + (mWeights[j] * testFoldInstanceList.get(i).
                        get(arffReader.getAttributeList().get(j).getAttributeName()));
            }*/

        for (int i = 0; i < arffReader.getNumberOfDataInstances(); i++) {
            if (foldList[i] == testFold) {
                double weightSum = 0.0;
                for (int j = 0; j < arffReader.getNumberOfAttributes(); j++) {
                    weightSum = weightSum + (mWeights[j] * arffReader.getDataInstanceList().get(i)
                            .get(arffReader.getAttributeList().get(j).getAttributeName()));
                }
                double outputWeight = weightSum + mBias;
                double predictedOutput = getSigmoidalOutput(outputWeight);
                String predictedClass = "";
                if (predictedOutput > 0.5) {
                    predictedClass = "Mine";
                } else {
                    predictedClass = "Rock";
                }
                //String actualClass = arffReader.mClassLabelList.get(testFoldInstanceIndexList.get(i));
                String actualClass = arffReader.mClassLabelList.get(i);

                if(actualClass.equalsIgnoreCase(predictedClass)){
                    mCorrectlyClassified++;
                }
            /*NeuralNetOutput neuralNetOutput = new NeuralNetOutput(testFoldInstanceIndexList.get(i), testFold, predictedClass,
                    actualClass, predictedOutput);*/
                NeuralNetOutput neuralNetOutput = new NeuralNetOutput(i, testFold, predictedClass, actualClass, predictedOutput);
                mClassificationOutput.add(neuralNetOutput);
            }
        }
    }


    private static void printOutput(ARFFReader arffReader) {
        Collections.sort(mClassificationOutput);
        for (int i = 0; i < arffReader.getNumberOfDataInstances(); i++) {
            mClassificationOutput.get(i).printOutput();
        }
    }

    public static void main(String args[]) {
        ARFFReader arffReader = ARFFReader.getInstance("/Users/kirthanaaraghuraman/Documents/CS760/HW#2/src/com/kirthanaa/hw3/trainingset/sonar.arff");
        arffReader.parseARFFFile();

        createStratifiedSamples(arffReader);

        crossValidate(arffReader);

        printOutput(arffReader);

        System.out.println("No of correctly classified instances : " + mCorrectlyClassified);
        System.out.println("Accuracy : " + (double)mCorrectlyClassified/(double)arffReader.getNumberOfDataInstances());
    }
}
