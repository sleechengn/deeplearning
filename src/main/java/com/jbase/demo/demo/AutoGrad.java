package com.jbase.demo.demo;

import com.jbase.demo.nn.core.Variable;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by lee on 2018/3/27.
 */
@Slf4j
public class AutoGrad {

    public static void main(String[] arguments) {

        //定义训练数据
        Variable x = new Variable(Nd4j.create(new double[][]{
                {0.1, 0.2, 0.3, 0.4},
                {0.3, 0.4, 0.5, 0.7}
        }));
        Variable y = new Variable(Nd4j.create(new double[][]{
                {1, 0},
                {0, 1}
        }));

        //例子二模块化的
        NNModule nn = new NNModule();
        for (int i = 0; i < 100001; i++) {
            if(i % 1000 == 0) {
                System.out.println("epoch:" + i /1000);
            }
            nn.train(x, y);
        }
        Variable predict = nn.forward(new Variable(Nd4j.create(new double[]{0.3, 0.4, 0.5, 0.69})));
        System.out.println(predict.data);

    }


}
