package com.kirthanaa.hw3.arffreader;

import com.kirthanaa.hw3.entities.NeuralNetAttribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * Created by kirthanaaraghuraman on 10/20/15.
 */
public class ARFFReader {
    /**
     * File name/path that is to be parsed
     */
    private String mFileName = "";

    /**
     * List of hash map of data instances in ARFF file
     */
    private ArrayList<HashMap<String, Double>> mDataInstanceList = null;

    /**
     * List of class labels of the training instances in the file
     */
    public ArrayList<String> mClassLabelList = null;

    /**
     * Class labels for the parsed file
     */
    public String[] mClassLabels = null;

    /**
     * List of attributes in the parsed ARFF File
     */
    private ArrayList<NeuralNetAttribute> mNeuralNetAttributeList = null;

    /**
     * Gets an instance of ARFFReader class
     *
     * @param fileName Name/Path of file to be parsed
     * @return ARFFReader instance
     */
    public static ARFFReader getInstance(String fileName) {
        ARFFReader arffReader = new ARFFReader();
        /*if (fileName == "") {
            System.out.println("Filename is empty!");
        } else {
            System.out.println("Filename : " + fileName);
        }*/
        arffReader.mFileName = fileName;
        arffReader.mDataInstanceList = null;
        arffReader.mClassLabels = null;
        arffReader.mNeuralNetAttributeList = null;
        return arffReader;
    }

    /**
     * Given data of Instances format, parses them and stores in ID3Attribute format
     *
     * @param data Instance variable containing data
     */
    public void setNeuralNetAttributes(Instances data) {
        int attributeOrdinal = -1;
        String attributeName = "";

        if (mNeuralNetAttributeList == null) {
            mNeuralNetAttributeList = new ArrayList<NeuralNetAttribute>();
        }
        if (data.classIndex() == -1) {
            for (int i = 0; i < data.numAttributes() - 1; i++) {
                attributeOrdinal = i;
                attributeName = data.attribute(i).name();
                NeuralNetAttribute neuralNetAttribute = new NeuralNetAttribute(attributeOrdinal, attributeName);
                mNeuralNetAttributeList.add(neuralNetAttribute);
                //System.out.println("Added attribute " + id3Attribute.mAttributeName + " to attribute list!");
            }
        }
    }

    /**
     * Returns the attribute list in the currently parsed ARFF File
     *
     * @return List of ID3Attributes
     */
    public ArrayList<NeuralNetAttribute> getAttributeList() {
        if (mNeuralNetAttributeList == null) {
            //System.out.println("Attribute list is null!");
            return null;
        } else {
            return mNeuralNetAttributeList;
        }
    }

    /**
     * Identifies the class labels from the Instances variable and stores them as ID3Class variable
     *
     * @param data Instance variable containing data
     */
    public void setNeuralNetClass(Instances data) {
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
            mClassLabels = new String[data.attribute(data.numAttributes() - 1).numValues()];
            for (int j = 0; j < data.attribute(data.numAttributes() - 1).numValues(); j++) {
                mClassLabels[j] = data.attribute(data.numAttributes() - 1).value(j);
                //System.out.println("Added class label " + classLabels[j] + " to ID3 class ");
            }
        }
    }

    /**
     * Returns the ID3Class variable of the current file
     *
     * @return ID3Class variable
     */
    public String[] getNeuralNetClass() {
        if (mClassLabels == null) {
            //System.out.println("Class label is null!");
            return null;
        } else {
            return mClassLabels;
        }
    }

    /**
     * Returns the list of data instances
     *
     * @return List of Data Instances
     */
    public ArrayList<HashMap<String, Double>> getDataInstanceList() {
        if (mDataInstanceList == null) {
            //System.out.println("Data instance list is empty");
            return null;
        } else {
            return mDataInstanceList;
        }
    }

    /**
     * Parses the data instances in the Instance variable and stores them in an array list containing string array
     *
     * @param data Instance variable containing data
     */
    public void setDataInstanceList(Instances data) {
        if (mDataInstanceList == null) {
            mDataInstanceList = new ArrayList<HashMap<String, Double>>();
            mClassLabelList = new ArrayList<>();
        }
        //System.out.println("Total number of Data instances : " + data.numInstances());
        if (data.numInstances() != 0) {
            for (int i = 0; i < data.numInstances(); i++) {
                HashMap<String, Double> attributeHash = new HashMap<>();
                for (int j = 0; j < data.instance(i).numValues() - 1; j++) {
                    attributeHash.put(mNeuralNetAttributeList.get(j).getAttributeName(), data.instance(i).value(j));
                }
                mClassLabelList.add(data.instance(i).stringValue(data.instance(i).numValues() - 1));
                mDataInstanceList.add(attributeHash);
                //System.out.println("Instance value of attribute 1: " + data.instance(i).stringValue(1));
            }
            //System.out.println("Number of data instances added to list: " + mDataInstanceList.size());

            /*for(HashMap<String, Double> hashMap : mDataInstanceList){
                System.out.println(hashMap);
            }*/
        }
    }

    /**
     * Returns the number of data instances parsed in the ARFF input file
     *
     * @return Number of data instances
     */
    public int getNumberOfDataInstances() {
        if (mDataInstanceList == null) {
            //System.out.println("Data instance list is null!");
            return -1;
        } else {
            return mDataInstanceList.size();
        }
    }

    /**
     * Returns the number of attributes in the parsed ARFF File
     *
     * @return Number of attributes
     */
    public int getNumberOfAttributes() {
        if (mNeuralNetAttributeList == null) {
            //System.out.println("Attribute list is null!");
            return -1;
        } else {
            return mNeuralNetAttributeList.size();
        }
    }

    /**
     * Parses the ARFF File using the ARFFLoader class on WEKA
     */
    public void parseARFFFile() {

        ArffLoader arffLoader = new ArffLoader();
        File filedata = new File(mFileName);
        //File filedata = new File("/Users/kirthanaaraghuraman/Documents/CS760/Assignments/HW#1/src/com/kirthanaa/id3/trainingset/diabetes_train.arff");
        if (filedata.exists() && !filedata.isDirectory()) {
            try {
                arffLoader.setFile(filedata);
                Instances data = arffLoader.getDataSet();
                //System.out.println("Data instance set successfully! No instances : " + data.numInstances());
                setNeuralNetAttributes(data);

                setNeuralNetClass(data);

                setDataInstanceList(data);

            } catch (IOException e) {
                System.out.println(e.getMessage());
            }
        } else {
            System.out.println("File " + mFileName + " does not exist!");
        }
    }
}
