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

    /**
     * Number of folds to create
     */
    private static int mFolds = 0;

    /**
     * Number of epochs to train
     */
    private static int mEpochs = 0;

    /**
     * Learning rate of neural network
     */
    private static double mLearningRate = 0.0;

    /**
     * Number of positive instances in data set
     */
    private static int mPositiveInstances = 0;

    /**
     * Number of negative instances in data set
     */
    private static int mNegativeInstances = 0;

    /**
     * List with details of which fold each instance is assigned to
     */
    private static int foldList[] = null;

    private static ArrayList<ArrayList<HashMap<String, Double>>> mStratifiedSamples = null;

    /**
     * Array of weights for connections between input and output units
     */
    private static double[] mWeights = null;

    /**
     * Bias value for the neural network
     */
    private static double mBias = 0.0;

    private static ArrayList<ArrayList<Integer>> mIndexList = null;

    /**
     * List of outputs containing various details
     */
    private static ArrayList<NeuralNetOutput> mClassificationOutput = null;

    /**
     * Number of correctly classified test instances
     */
    private static int mCorrectlyClassifiedTestSet = 0;


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
     * Creates a n fold stratified sample set for cross validation
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

        int posInstanceIndex = 0;
        int negInstanceIndex = 0;

        mIndexList = new ArrayList<>(mFolds);
        for (int i = 0; i < mFolds; i++) {

            int l = 0;
            ArrayList<HashMap<String, Double>> instancesListPerFold = new ArrayList<>();
            ArrayList<Integer> instancesIndexPerFold = new ArrayList<>();

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

            //mStratifiedSamples.add(i, instancesListPerFold);
            //mIndexList.add(i, instancesIndexPerFold);
        }
        int k = mFolds - 1;
        for (int i = posInstanceIndex; i < posInstanceSize; i++) {
            //mIndexList.get(k).add(positiveInstanceIndex[i]);
            foldList[i] = k;
            //mStratifiedSamples.get(k).add(arffReader.getDataInstanceList().get(positiveInstanceIndex[i]));
            k = (k + 1) % mFolds;
        }

        for (int i = negInstanceIndex; i < negInstanceSize; i++) {
            foldList[i] = k;
            //mIndexList.get(k).add(negativeInstanceIndex[i]);
            //mStratifiedSamples.get(k).add(arffReader.getDataInstanceList().get(negativeInstanceIndex[i]));
            k = (k + 1) % mFolds;
        }

        /*for(int i = 0; i < mStratifiedSamples.size(); i++){
            for (int j = 0; j < mStratifiedSamples.get(i).size(); j++){
                shuffleList(mStratifiedSamples.get(i).get(j));
            }
        }*/

    }

    /**
     * Trains and tests the neural network using stratified cross validation
     *
     * @param arffReader ARFF Reader instance containing details about the learning set
     * @return Average train set accuracy
     */
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

    /**
     * Updates weight and bias parameters for the neural network
     *
     * @param delta         Error between actual and predicted class
     * @param arffReader    ARFF Reader instance containing details about the learning set
     * @param instanceIndex Index of instance whose values should be used in weight updation
     */
    private static void updateWeightAndBias(double delta, ARFFReader arffReader, int instanceIndex) {

        for (int i = 0; i < arffReader.getNumberOfAttributes(); i++) {
            double delWeight = delta * mLearningRate * arffReader.getDataInstanceList().get(instanceIndex)
                    .get(arffReader.getAttributeList().get(i).getAttributeName());
            mWeights[i] = mWeights[i] + delWeight;
        }
        mBias = mBias + delta;
    }

    /**
     * Trains the neural network on all folds except the test fold
     * Trains for specified number of epochs
     *
     * @param arffReader ARFF Reader instance containing details about the learning set
     * @param testFold   Fold that is set aside for testing
     */
    private static void trainNeuralNet(ARFFReader arffReader, int testFold) {
        /**
         * Shuffling instances within each fold
         */
        for (int f = 0; f < mFolds; f++) {
            int[] foldInstanceIndex = new int[10000];
            int foldIndex = 0;

            for (int i = 0; i < arffReader.getNumberOfDataInstances(); i++) {
                if (foldList[i] == testFold) {
                    continue;
                } else {
                    if (foldList[i] == f) {
                        foldInstanceIndex[foldIndex++] = i;
                    }
                }
            }
            if (foldIndex != 0) {
                shuffleArray(foldInstanceIndex, foldIndex);

                for (int k = 0; k < foldIndex; k++) {
                    double weightSum = 0.0;
                    double actualOutput = -1;

                    if (arffReader.mClassLabelList.get(foldInstanceIndex[k]).equalsIgnoreCase(arffReader.mClassLabels[0])) {
                        actualOutput = 0.0;
                    } else if (arffReader.mClassLabelList.get(foldInstanceIndex[k]).equalsIgnoreCase(arffReader.mClassLabels[1])) {
                        actualOutput = 1.0;
                    }
                    for (int j = 0; j < arffReader.getDataInstanceList().get(foldInstanceIndex[k]).size(); j++) {
                        weightSum = weightSum + (arffReader.getDataInstanceList().get(foldInstanceIndex[k]).get(arffReader.getAttributeList().get(j).getAttributeName()) * mWeights[j]);
                    }
                    weightSum = weightSum + (1 * mBias);
                    double predOutput = getSigmoidalOutput(weightSum);
                    double delta = (predOutput) * (1 - predOutput) * (actualOutput - predOutput);
                    updateWeightAndBias(delta, arffReader, foldInstanceIndex[k]);
                }
            }
        }

        /**
         * Without shuffling instances within each fold
         */
        /*for (int i = 0; i < arffReader.getNumberOfDataInstances(); i++) {
            if (foldList[i] == testFold) {
                continue;
            } else {
                double weightSum = 0.0;
                double actualOutput = -1;

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

                double delta = (predOutput) * (1 - predOutput) * (actualOutput - predOutput);
                updateWeightAndBias(delta, arffReader, i);
            }
        }*/
    }


    /**
     * Tests the trained neural network on the test fold instances
     *
     * @param testFold   Fold that is kept for testing
     * @param arffReader ARFF Reader instance containing details about the learning set
     */
    private static void testNeuralNet(int testFold, ARFFReader arffReader) {

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

                String actualClass = arffReader.mClassLabelList.get(i);

                if (actualClass.equalsIgnoreCase(predictedClass)) {
                    mCorrectlyClassifiedTestSet++;
                }

                NeuralNetOutput neuralNetOutput = new NeuralNetOutput(i, testFold, predictedClass, actualClass, predictedOutput);
                mClassificationOutput.add(neuralNetOutput);
            }
        }
    }


    /**
     * Prints the output in desired format
     *
     * @param arffReader ARFF Reader instance containing details about the learning set
     */
    private static void printOutput(ARFFReader arffReader) {
        Collections.sort(mClassificationOutput);
        for (int i = 0; i < arffReader.getNumberOfDataInstances(); i++) {
            mClassificationOutput.get(i).printOutput();
        }
    }

    /**
     * Computes train and test set accuracy for 1, 10, 100 and 1000 epochs
     *
     * @param arffReader ARFF Reader instance containing details about the learning set
     */
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


    /**
     * Computes true and false positive rate for plotting ROC Curve
     *
     * @param arffReader ARFF Reader instance containing details about the learning set
     */
    private static void computeValuesForROC(ARFFReader arffReader) {
        mEpochs = 100;
        mLearningRate = 0.1;
        int totalTruePositive = 0;
        int totalTrueNegative = 0;
        int totalFalsePositive = 0;
        int totalFalseNegative = 0;

        ArrayList<ROCValues> rocValuesList = new ArrayList<>(arffReader.getNumberOfDataInstances());
        crossValidate(arffReader);
        System.out.println("Accuracy : " + ((double) mCorrectlyClassifiedTestSet) / (double) arffReader.getNumberOfDataInstances());
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
            ROCValues rocValue = new ROCValues((double) truePositive / (double) mPositiveInstances,
                    (double) falsePositive / (double) mNegativeInstances);
            rocValuesList.add(rocValue);
        }

        for (ROCValues value : rocValuesList) {
            System.out.println(value.getTruePositiveRate());
        }
        System.out.println("Roc values length : " + rocValuesList.size());

        for (ROCValues value : rocValuesList) {
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
