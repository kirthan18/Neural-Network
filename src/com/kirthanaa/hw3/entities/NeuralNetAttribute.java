package com.kirthanaa.hw3.entities;

/**
 * Created by kirthanaaraghuraman on 10/20/15.
 */
public class NeuralNetAttribute {

    /**
     * Order of the attribute in the ARFF File
     */
    private int mAttributeOrdinal = -1;

    /**
     * Name of the attribute
     */
    private String mAttributeName;

    /**
     * Constructor for initializing ID3Attribute instance
     *
     * @param attributeOrdinal Order in which attribute appears in the ARFF File
     * @param attributeName    Name of the attribute
     */
    public NeuralNetAttribute(int attributeOrdinal, String attributeName) {
        this.mAttributeOrdinal = attributeOrdinal;
        this.mAttributeName = attributeName;
    }

    /**
     * Function to get the ordinal of the attribute
     * @return Integer representing the attribute ordinal
     */
    public int getAttributeOrdinal() {
        return mAttributeOrdinal;
    }

    /**
     * Function to get the attribute name
     * @return String containing attribute name
     */
    public String getAttributeName() {
        return mAttributeName;
    }
}
