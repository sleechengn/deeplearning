package com.jbase.demo.demo;

import com.jbase.demo.nn.core.Toolkit;
import com.jbase.demo.nn.core.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NNModule {

    private Variable layer1weights;
    private Variable layer1bias;
    private Variable layer2weights;
    private Variable layer2bias;
    private Toolkit tool;

    public NNModule() {
        tool = new Toolkit();
        layer1weights = new Variable(Nd4j.randn(new int[]{4, 3}));
        layer1bias = new Variable(Nd4j.create(new double[]{0, 0, 0}));
        layer2weights = new Variable(Nd4j.randn(new int[]{3, 2}));
        layer2bias = new Variable(Nd4j.create(new double[]{0, 0}));
    }

    public void train(Variable x, Variable y) {
        Variable out = forward(x);
        out = out.sub(y);
        out = out.square();
        out = out.mean();
        tool.zeroGrad(out);
        tool.backward(out);
        ((INDArray) layer1weights.data).addi(((INDArray) layer1weights.grad).mul(-0.1));
        ((INDArray) layer1bias.data).addi(((INDArray) layer1bias.grad).mul(-0.1));
        ((INDArray) layer2weights.data).addi(((INDArray) layer2weights.grad).mul(-0.1));
        ((INDArray) layer2bias.data).addi(((INDArray) layer2bias.grad).mul(-0.1));
    }

    public Variable forward(Variable x) {
        Variable out = x.matMul(layer1weights);
        out = out.addVec(layer1bias);
        out = out.sigmoid();
        out = out.matMul(layer2weights);
        out = out.addVec(layer2bias);
        out = out.sigmoid();
        out = out.relu();
        return out;
    }
}
