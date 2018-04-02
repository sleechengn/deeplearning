package com.jbase.demo.demo;

import com.jbase.demo.nn.core.Toolkit;
import com.jbase.demo.nn.core.Variable;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by lee on 2018/3/27.
 */
@Slf4j
public class Nd4jTest {

    public static void main(String[] arguments) {

        //定义一个4*3*2的网络，激活函数用sigmoid
        Variable layer1weights = new Variable(Nd4j.randn(new int[]{4, 3}));
        Variable layer1bias = new Variable(Nd4j.create(new double[]{0, 0, 0}));
        Variable layer2weights = new Variable(Nd4j.randn(new int[]{3, 2}));
        Variable layer2bias = new Variable(Nd4j.create(new double[]{0, 0}));
        //定义训练数据
        Variable x = new Variable(Nd4j.create(new double[][]{
                {0.1, 0.2, 0.3, 0.4},
                {0.3, 0.4, 0.5, 0.7}
        }));
        Variable y = new Variable(Nd4j.create(new double[][]{
                {1, 0.5},
                {1, 1}
        }));

        Toolkit tool = new Toolkit();
        for (int i = 0; i < 100001; i++) {
            Variable out = x.matMul(layer1weights);  //输入数据乘以权重矩阵
            out = out.addVec(layer1bias);            //加上偏执项
            out = out.sigmoid();                     //应用sigmoid函数
            out = out.matMul(layer2weights);         //乘以第二层权重矩阵
            out = out.addVec(layer2bias);            //加上第二层偏执项
            out = out.sigmoid();                     //应用sigmoid函数
            out = out.relu();                        //应用relu

            if (i % 1000 == 0)
                System.out.println(out.data);

            out = out.sub(y);                        //减去label
            out = out.square();                      //求平方
            out = out.mean();                        //求平均
            tool.zeroGrad(out);                      //清空梯度
            tool.backward(out);                      //反向求导

            if (i % 1000 == 0)
                System.out.println(out.data.toString());

            //更新权重
            ((INDArray) layer1weights.data).addi(((INDArray) layer1weights.grad).mul(-0.1));
            ((INDArray) layer1bias.data).addi(((INDArray) layer1bias.grad).mul(-0.1));
            ((INDArray) layer2weights.data).addi(((INDArray) layer2weights.grad).mul(-0.1));
            ((INDArray) layer2bias.data).addi(((INDArray) layer2bias.grad).mul(-0.1));

        }

        //例子二模块化的
        NNModule nn = new NNModule();
        for (int i = 0; i < 100001; i++) {
            nn.train(x, y);
        }
        Variable predict = nn.forward(new Variable(Nd4j.create(new double[]{0.3, 0.4, 0.5, 0.69})));
        System.out.println(predict.data);

    }


}
