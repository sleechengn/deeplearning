package com.jbase.demo.demo;

import com.jbase.demo.AEModel;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Arrays;
import java.util.Random;

/**
 * Created by lee on 2018/3/17.
 */
public class AETest {

    public static void main(String[] args) throws Exception {


        Random rand = new Random();
        int epoch = 10000;
        int simple_count = 5;
        int input_dim = 18;
        int latent_dim = 1;

        AEModel ae = new AEModel();

        ae.setRawVectorDimension(input_dim);
        ae.setLatentVectorDimension(latent_dim);
        ae.setEncodeLayerDimensions(new int[]{15, 7, 3});
        ae.setDecodeLayerDimensions(new int[]{3, 7, 15});
        ae.setOutputLayerActivationFunction(Activation.SIGMOID);
        ae.setOutputLayerLoss(LossFunctions.LossFunction.SQUARED_LOSS);
        ae.init();

        //生成样本数据
        double[][] trainData = new double[simple_count][input_dim];
        for (int i = 0; i < simple_count; i++) {
            for (int j = 0; j < input_dim; j++) {
                trainData[i][j] = rand.nextDouble();
            }
        }
        INDArray features = Nd4j.create(trainData);
        INDArray labels = Nd4j.create(trainData);

        DataSet trainDataSet = new DataSet();
        trainDataSet.setFeatures(features);
        trainDataSet.setLabels(labels);

        DataSetIterator trainDataSetIt = new ListDataSetIterator<>(Arrays.asList(trainDataSet));

        for (int i = 0; i < epoch; i++) {
            ae.fit(trainDataSetIt);
            if (i % 100 == 0)
                System.out.println("epoch:" + i);
        }

        MultiLayerNetwork encoder = ae.getEncoder();
        MultiLayerNetwork decoder = ae.getDecoder();

        System.out.println("编码前的特征:");
        System.out.println(features);

        INDArray encoded = encoder.output(features);
        System.out.println("被编码的特征:");
        System.out.println(encoded);

        INDArray decoded = decoder.output(encoded);
        System.out.println("解码后的特征");
        System.out.println(decoded);

        INDArray subed = features.sub(decoded);
        System.out.println("差异");
        System.out.println(subed);

    }

}
