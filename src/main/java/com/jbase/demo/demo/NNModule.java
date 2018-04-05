package com.jbase.demo.demo;

import com.jbase.demo.nn.core.Toolkit;
import com.jbase.demo.nn.core.Variable;
import org.nd4j.linalg.factory.Nd4j;

public class NNModule {

    private Variable layer1weights; //层1权重矩阵
    private Variable layer1bias;    //层1偏执向量
    private Variable layer2weights; //层2权重矩阵
    private Variable layer2bias;    //层2偏执向量
    private Toolkit tool;           //自动微分工具
    private double learningSpeed;   //学习速率

    public NNModule() {
        tool = new Toolkit();
        //初始化权重为高斯分布
        layer1weights = new Variable(Nd4j.randn(new int[]{4, 3}));
        layer1bias = new Variable(Nd4j.create(new double[]{0, 0, 0}));
        layer2weights = new Variable(Nd4j.randn(new int[]{3, 2}));
        layer2bias = new Variable(Nd4j.create(new double[]{0, 0}));
        learningSpeed = 0.01D;
    }

    public void train(Variable x, Variable y) {
        Variable out = forward(x);  //从输入层计算到输出层
        out = out.sub(y);           //减去目标Label
        out = out.square();         //计算平方
        out = out.mean();           //求平均
        tool.grad2zero(out);        //自动微分准备，导数清0
        tool.backward(out);         //自动微分

        //权重自身调整，减梯度乘学习率
        layer1weights.data.tensor.subi(layer1weights.grad.tensor.mul(learningSpeed));
        layer1bias.data.tensor.subi(layer1bias.grad.tensor.mul(learningSpeed));
        layer2weights.data.tensor.subi(layer2weights.grad.tensor.mul(learningSpeed));
        layer2bias.data.tensor.subi(layer2bias.grad.tensor.mul(learningSpeed));
    }

    public Variable forward(Variable x) {
        Variable out = x.matMul(layer1weights);     //输入矩阵乘权重矩阵
        out = out.addVec(layer1bias);               //加偏执向量
        out = out.sigmoid();                        //应用sigmoid函数
        out = out.matMul(layer2weights);            //乘权重矩阵
        out = out.addVec(layer2bias);               //加偏执向量
        out = out.sigmoid();                        //应用sigmoid
        out = out.relu();                           //应用relu函数
        return out;
    }
}
