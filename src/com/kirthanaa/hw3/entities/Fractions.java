package com.kirthanaa.hw3.entities;

/**
 * Created by kirthanaaraghuraman on 10/21/15.
 */

public class Fractions extends Number {
    private int numerator;
    private int denominator;

    public Fractions(int numerator, int denominator) {
        if (denominator == 0) {
            throw new IllegalArgumentException("denominator is zero");
        }
        if (denominator < 0) {
            numerator *= -1;
            denominator *= -1;
        }
        this.numerator = numerator;
        this.denominator = denominator;
        simplify();
    }

    public Fractions(int numerator) {
        this.numerator = numerator;
        this.denominator = 1;
    }

    public int getNumerator() {
        return this.numerator;
    }

    public int getDenominator() {
        return this.denominator;
    }

    public byte byteValue() {
        return (byte) this.doubleValue();
    }

    public double doubleValue() {
        return ((double) numerator) / ((double) denominator);
    }

    public float floatValue() {
        return (float) this.doubleValue();
    }

    public int intValue() {
        return (int) this.doubleValue();
    }

    public long longValue() {
        return (long) this.doubleValue();
    }

    public short shortValue() {
        return (short) this.doubleValue();
    }

    /**
     * Returns the absolute value of the greatest common divisor of this
     * fraction's numerator and denominator. If the numerator or denominator is
     * zero, this method returns 0. This method always returns either a positive
     * integer, or zero.
     *
     * @return Returns the greatest common denominator
     */
    private int gcd() {
        int s;
        if (numerator > denominator)
            s = denominator;
        else
            s = numerator;
        for (int i = s; i > 0; i--) {
            if ((numerator % i == 0) && (denominator % i == 0))
                return i;
        }
        return -1;
    }

    /**
     * Changes this fraction's numerator and denominator to "lowest terms"
     * (sometimes referred to as a "common fraction"), by dividing the numerator
     * and denominator by their greatest common divisor. This includes fixing
     * the signs. For example, if a fraction is 24/-18, this method will change
     * it to -4/3. If the numerator or denominator of the fraction is zero, no
     * change is made.
     */
    public void simplify() {

        if (numerator != 0 && denominator != 0) {// Making sure num or den is not zero.

            int gcd = gcd();
            if (gcd > 1) {
                this.numerator = this.numerator / gcd;
                this.denominator = this.denominator / gcd;
            }
        }

    }
}
