package com.kirthanaa.hw2.neuralnet;

import com.kirthanaa.hw2.arffreader.ARFFReader;
import com.kirthanaa.hw2.entities.NeuralNetOutput;
import com.kirthanaa.hw2.entities.ROCValues;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by kirthanaaraghuraman on 10/20/15.
 */
public class NeuralNetwork {

    private static int mFolds = 0;

    private static int mEpochs = 0;

    private static double mLearningRate = 0.0;

    private static int mPositiveInstances = 0;

    private static int mNegativeInstances = 0;

    private static int foldList[] = null;

    private static ArrayList<ArrayList<HashMap<String, Double>>> mStratifiedSamples = null;

    private static double[] mWeights = null;

    private static double mBias = 0.0;

    private static ArrayList<ArrayList<Integer>> mIndexList = null;

    private static ArrayList<NeuralNetOutput> mClassificationOutput = null;

    private static int mCorrectlyClassifiedTestSet = 0;

    private static int mCorrectlyClassifiedTrainSet = 0;


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

        int negInstanceSize = 0;
        int posInstanceSize = 0;

        for (int i = 0; i < arffReader.getNumberOfDataInstances(); i++) {
            if (arffReader.mClassLabelList.get(i).equalsIgnoreCase(arffReader.mClassLabels[0])) {
                mNegativeInstances++;
                negativeInstanceIndex[negInstanceSize++] = i;
            } else if (arffReader.mClassLabelList.get(i).equalsIgnoreCase(arffReader.mClassLabels[1])) {
                mPositiveInstances++;
                positiveInstanceIndex[posInstanceSize++] = i;
            }
        }
        //mClassRatio = (double) mPositiveInstances / (double) mNegativeInstances;

        shuffleArray(positiveInstanceIndex, posInstanceSize);
        shuffleArray(negativeInstanceIndex, negInstanceSize);

        double fractionPosInstance = (double) posInstanceSize / (double) arffReader.getNumberOfDataInstances();
        double fractionNegInstance = (double) negInstanceSize / (double) arffReader.getNumberOfDataInstances();

        int numInstancesPerFold = arffReader.getNumberOfDataInstances() / mFolds;
        int numPosInstancesPerFold = 0;
        int numNegInstancesPerFold = 0;

        numPosInstancesPerFold = (int) (fractionPosInstance * numInstancesPerFold);
        numNegInstancesPerFold = (int) (numInstancesPerFold - numPosInstancesPerFold);

        /*if (fractionPosInstance > fractionNegInstance) {
            numPosInstancesPerFold = (int) (fractionPosInstance * numInstancesPerFold) + 1;
            numNegInstancesPerFold = (int) (fractionNegInstance * numInstancesPerFold);
        } else {
            numPosInstancesPerFold = (int) (fractionPosInstance * numInstancesPerFold);
            numNegInstancesPerFold = (int) (fractionNegInstance * numInstancesPerFold) + 1;
        }*/

        int posInstanceIndex = 0;
        int negInstanceIndex = 0;

        mIndexList = new ArrayList<>(mFolds);
        for (int i = 0; i < mFolds; i++) {

            int l = 0;
            ArrayList<HashMap<String, Double>> instancesListPerFold = new ArrayList<>();
            ArrayList<Integer> instancesIndexPerFold = new ArrayList<>();
            /*while (l < numInstancesPerFold) {

                for (int k = 0; k < numPosInstancesPerFold; k++) {
                    if (posInstanceIndex < posInstanceSize) {
                        instancesIndexPerFold.add(positiveInstanceIndex[posInstanceIndex]);
                        foldList[positiveInstanceIndex[posInstanceIndex]] = i;
                        instancesListPerFold.add(arffReader.getDataInstanceList().get(positiveInstanceIndex[posInstanceIndex++]));
                        l++;
                    }
                }

                for (int n = 0; n < numNegInstancesPerFold; n++) {
                    if (negInstanceIndex < negInstanceSize) {
                        foldList[negativeInstanceIndex[negInstanceIndex]] = i;
                        instancesIndexPerFold.add(negativeInstanceIndex[negInstanceIndex]);
                        instancesListPerFold.add(arffReader.getDataInstanceList().get(negativeInstanceIndex[negInstanceIndex++]));
                        l++;
                    }
                }
            }*/


            for (int p = 0; p < numPosInstancesPerFold; p++) {
                if (posInstanceIndex < posInstanceSize) {
                    foldList[positiveInstanceIndex[posInstanceIndex]] = i;
                    instancesIndexPerFold.add(positiveInstanceIndex[posInstanceIndex]);
                    instancesListPerFold.add(arffReader.getDataInstanceList().get(positiveInstanceIndex[posInstanceIndex++]));
                }
            }

            for (int n = 0; n < numNegInstancesPerFold; n++) {
                if (negInstanceIndex < negInstanceSize) {
                    foldList[negativeInstanceIndex[negInstanceIndex]] = i;
                    instancesIndexPerFold.add(negativeInstanceIndex[negInstanceIndex]);
                    instancesListPerFold.add(arffReader.getDataInstanceList().get(negativeInstanceIndex[negInstanceIndex++]));
                }
            }

            mStratifiedSamples.add(i, instancesListPerFold);
            mIndexList.add(i, instancesIndexPerFold);
        }
        int k = mFolds - 1;
        for (int i = posInstanceIndex; i < posInstanceSize; i++) {
            mIndexList.get(k).add(positiveInstanceIndex[i]);
            foldList[i] = k;
            mStratifiedSamples.get(k).add(arffReader.getDataInstanceList().get(positiveInstanceIndex[i]));
            k = (k + 1) % mFolds;
        }

        for (int i = negInstanceIndex; i < negInstanceSize; i++) {
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

    private static double crossValidate(ARFFReader arffReader) {
        mClassificationOutput = new ArrayList<>(arffReader.getNumberOfDataInstances());
        double foldAccuracy[] = new double[mFolds];
        for (int i = 0; i < mFolds; i++) {
            double avg = 0.0;
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

            int correctTrainSetPrediction = 0;
            int totalTrainSetPrediction = 0;

            for (int t = 0; t < arffReader.getNumberOfDataInstances(); t++) {
                if (foldList[t] != testFold) {
                    totalTrainSetPrediction++;
                    double weightSum = 0.0;
                    for (int j = 0; j < arffReader.getNumberOfAttributes(); j++) {
                        weightSum = weightSum + (mWeights[j] * arffReader.getDataInstanceList().get(t)
                                .get(arffReader.getAttributeList().get(j).getAttributeName()));
                    }
                    double outputWeight = weightSum + mBias;
                    double predictedOutput = getSigmoidalOutput(outputWeight);
                    String predictedClass = "";
                    if (predictedOutput > 0.5) {
                        predictedClass = arffReader.mClassLabels[1];
                    } else {
                        predictedClass = arffReader.mClassLabels[0];
                    }
                    //String actualClass = arffReader.mClassLabelList.get(testFoldInstanceIndexList.get(i));
                    String actualClass = arffReader.mClassLabelList.get(t);

                    if (actualClass.equalsIgnoreCase(predictedClass)) {
                        correctTrainSetPrediction++;
                    }
                }
            }
            foldAccuracy[i] = (double) correctTrainSetPrediction / (double) totalTrainSetPrediction;
            testNeuralNet(testFold, arffReader);
        }
        double sum = 0.0;
        for (double avg : foldAccuracy) {
            sum += avg;
        }
        return sum / foldAccuracy.length;
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
        mCorrectlyClassifiedTrainSet = 0;

        for (int i = 0; i < arffReader.getNumberOfDataInstances(); i++) {
            if (foldList[i] == testFold) {
                continue;
            } else {
                double weightSum = 0.0;
                double actualOutput = -1;
                String predictedClass = "";

                if (arffReader.mClassLabelList.get(i).equalsIgnoreCase(arffReader.mClassLabels[0])) {
                    actualOutput = 0.0;
                } else if (arffReader.mClassLabelList.get(i).equalsIgnoreCase(arffReader.mClassLabels[1])) {
                    actualOutput = 1.0;
                }
                for (int j = 0; j < arffReader.getDataInstanceList().get(i).size(); j++) {
                    weightSum = weightSum + (arffReader.getDataInstanceList().get(i).get(arffReader.getAttributeList().get(j).getAttributeName()) * mWeights[j]);
                }
                weightSum = weightSum + (1 * mBias);
                double predOutput = getSigmoidalOutput(weightSum);
                if (predOutput > 0.5) {
                    predictedClass = arffReader.mClassLabels[1];
                } else {
                    predictedClass = arffReader.mClassLabels[0];
                }

                if (predictedClass.equalsIgnoreCase(arffReader.mClassLabelList.get(i))) {
                    mCorrectlyClassifiedTrainSet++;
                }
                double delta = (predOutput) * (1 - predOutput) * (actualOutput - predOutput);
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
            }
            double outputWeight = weightSum + mBias;
            double predictedOutput = getSigmoidalOutput(outputWeight);
            String predictedClass = "";
            if (predictedOutput > 0.5) {
                predictedClass = "Mine";
            } else {
                predictedClass = "Rock";
            }
            String actualClass = arffReader.mClassLabelList.get(testFoldInstanceIndexList.get(i));

            if(actualClass.equalsIgnoreCase(predictedClass)){
                mCorrectlyClassifiedTestSet++;
            }

            NeuralNetOutput neuralNetOutput = new NeuralNetOutput(testFoldInstanceIndexList.get(i), testFold, predictedClass,
                    actualClass, predictedOutput);
            mClassificationOutput.add(neuralNetOutput);*/

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
                    predictedClass = arffReader.mClassLabels[1];
                } else {
                    predictedClass = arffReader.mClassLabels[0];
                }
                //String actualClass = arffReader.mClassLabelList.get(testFoldInstanceIndexList.get(i));
                String actualClass = arffReader.mClassLabelList.get(i);

                if (actualClass.equalsIgnoreCase(predictedClass)) {
                    mCorrectlyClassifiedTestSet++;
                }

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

    private static void question5(ARFFReader arffReader) {
        int[] epochs = {1, 10, 100, 1000};
        double[] testAccuracyForDiffEpochs = new double[4];
        double[] trainAccuracyForDiffEpochs = new double[4];

        for (int i = 0; i < epochs.length; i++) {
            mCorrectlyClassifiedTestSet = 0;
            mEpochs = epochs[i];
            trainAccuracyForDiffEpochs[i] = crossValidate(arffReader);
            //printOutput(arffReader);

            System.out.println("\n********EPOCHS : " + mEpochs + "*********");
            testAccuracyForDiffEpochs[i] = (double) mCorrectlyClassifiedTestSet / (double) arffReader.getNumberOfDataInstances();
            System.out.println("Train set Accuracy : " + trainAccuracyForDiffEpochs[i]);
            System.out.println("No of correctly classified instances : " + mCorrectlyClassifiedTestSet);
            System.out.println("Test set Accuracy : " + testAccuracyForDiffEpochs[i] + "\n");
        }
    }


    private static void computeValuesForROC(ARFFReader arffReader) {
        mEpochs = 100;
        mLearningRate = 0.1;
        int totalTruePositive = 0;
        int totalTrueNegative = 0;
        int totalFalsePositive = 0;
        int totalFalseNegative = 0;

        ArrayList<ROCValues> rocValuesList = new ArrayList<>(arffReader.getNumberOfDataInstances());
        crossValidate(arffReader);
        for (NeuralNetOutput neuralNetOutput : mClassificationOutput) {
            if (neuralNetOutput.getActualClass().equalsIgnoreCase(neuralNetOutput.getPredictedClass())) {
                if (neuralNetOutput.getActualClass().equalsIgnoreCase("Rock")) {
                    totalTrueNegative++;
                } else if (neuralNetOutput.getActualClass().equalsIgnoreCase("Mine")) {
                    totalTruePositive++;
                }
            } else {
                if (neuralNetOutput.getActualClass().equalsIgnoreCase("Mine")) {
                    if (neuralNetOutput.getPredictedClass().equalsIgnoreCase("Rock")) {
                        totalFalseNegative++;
                    }
                } else if (neuralNetOutput.getActualClass().equalsIgnoreCase("Rock")) {
                    if (neuralNetOutput.getPredictedClass().equalsIgnoreCase("Mine")) {
                        totalFalsePositive++;
                    }
                }

            }
        }

        int truePositive = 0;
        int falsePositive = 0;

        Collections.sort(mClassificationOutput);
        for (NeuralNetOutput neuralNetOutput : mClassificationOutput) {
            if (neuralNetOutput.getActualClass().equalsIgnoreCase(neuralNetOutput.getPredictedClass())) {
                if (neuralNetOutput.getActualClass().equalsIgnoreCase("Mine")) {
                    truePositive++;
                }
            } else {
                if (neuralNetOutput.getActualClass().equalsIgnoreCase("Rock")) {
                    if (neuralNetOutput.getPredictedClass().equalsIgnoreCase("Mine")) {
                        falsePositive++;
                    }
                }
            }
            ROCValues rocValue = new ROCValues((double)truePositive/(double)mPositiveInstances,
                    (double)falsePositive/(double)mNegativeInstances);
            rocValuesList.add(rocValue);
        }

        for(ROCValues value : rocValuesList){
            System.out.println(value.getTruePositiveRate());
        }
        System.out.println("Roc values length : " + rocValuesList.size());

        for(ROCValues value : rocValuesList){
            System.out.println(value.getFalsePositiveRate());
        }

        System.out.println("Roc values length : " + rocValuesList.size());
    }

    public static void main(String args[]) {

        String inputFile = args[0];
        mFolds = Integer.parseInt(args[1]);
        mLearningRate = Double.parseDouble(args[2]);
        mEpochs = Integer.parseInt(args[3]);

        ARFFReader arffReader = ARFFReader.getInstance(inputFile);
        arffReader.parseARFFFile();

        createStratifiedSamples(arffReader);

        crossValidate(arffReader);

        printOutput(arffReader);

        //System.out.println("Accuracy : " + ((double) mCorrectlyClassifiedTestSet) / (double) arffReader.getNumberOfDataInstances());

        //question5(arffReader);

        //computeValuesForROC(arffReader);
    }
}
