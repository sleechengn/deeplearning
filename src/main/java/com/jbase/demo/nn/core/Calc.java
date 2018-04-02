package com.jbase.demo.nn.core;

/**
 * Created by lee on 2018/4/1.
 */
public class Calc {

    public static double mathSigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double mathTanh(double x) {
        return Math.tanh(x);
    }

}
