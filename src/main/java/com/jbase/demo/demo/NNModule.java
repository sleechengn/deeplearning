package com.jbase.demo.demo;

import com.jbase.demo.nn.core.Toolkit;
import com.jbase.demo.nn.core.Variable;
import org.nd4j.linalg.factory.Nd4j;

public class NNModule {

    private Variable layer1weights;
    private Variable layer1bias;
    private Variable layer2weights;
    private Variable layer2bias;
    private Toolkit tool;
    private double learningSpeed;

    public NNModule() {
        tool = new Toolkit();
        layer1weights = new Variable(Nd4j.randn(new int[]{4, 3}));
        layer1bias = new Variable(Nd4j.create(new double[]{0, 0, 0}));
        layer2weights = new Variable(Nd4j.randn(new int[]{3, 2}));
        layer2bias = new Variable(Nd4j.create(new double[]{0, 0}));
        learningSpeed = 0.01D;
    }

    public void train(Variable x, Variable y) {
        Variable out = forward(x);
        out = out.sub(y);
        out = out.square();
        out = out.mean();
        tool.grad2zero(out);
        tool.backward(out);

        //权重自身调整，减梯度乘学习率
        layer1weights.data.tensor.subi(layer1weights.grad.tensor.mul(learningSpeed));
        layer1bias.data.tensor.subi(layer1bias.grad.tensor.mul(learningSpeed));
        layer2weights.data.tensor.subi(layer2weights.grad.tensor.mul(learningSpeed));
        layer2bias.data.tensor.subi(layer2bias.grad.tensor.mul(learningSpeed));
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
