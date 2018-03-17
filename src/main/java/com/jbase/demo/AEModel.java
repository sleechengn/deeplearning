package com.jbase.demo;

import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Created by lee on 2018/3/17.
 */
@Data
public class AEModel {

    private double learningSpeed = 1e-2;        //学习速率
    private Updater updater = Updater.ADAM;     //优化器

    private int rawVectorDimension = 0;         //原始特征数定义
    private int latentVectorDimension = 0;      //被编码的特征数定义

    private int[] encodeLayerDimensions;        //编码层维度
    private Activation encodeLayerActivationFunction = Activation.TANH;     //编码层激活函数

    private int[] decodeLayerDimensions;                                    //解码层定义
    private Activation decodeLayerActivationFunction = Activation.TANH;     //解码层激活函数
    private Activation outputLayerActivationFunction = Activation.SIGMOID;      //输出层激活函数
    private LossFunctions.LossFunction outputLayerLoss = LossFunctions.LossFunction.MSE;    //输出层损失

    private MultiLayerNetwork trainNetwork;

    public void init() throws Exception {

        if (rawVectorDimension < 1) {
            throw new IllegalThreadStateException("输入向量维度：[rawVectorDimension]未设置或设置无效");
        }
        if (latentVectorDimension < 1) {
            throw new IllegalThreadStateException("latent向量维度：[latentVectorDimension]未设置或设置无效");
        }
        if (encodeLayerDimensions == null) {
            throw new IllegalThreadStateException("编码层堆叠：[encodeLayerDimensions]未设置或设置无效");
        }
        if (decodeLayerDimensions == null) {
            throw new IllegalThreadStateException("解码层堆叠：[decodeLayerDimensions]未设置或设置无效");
        }

        NeuralNetConfiguration.ListBuilder trainNetworkListBuilder = NeuralNetConfiguration.Builder.class.newInstance()
                .updater(updater)
                .learningRate(learningSpeed)
                .iterations(1)
                .list();

        trainNetworkListBuilder.backprop(true);
        trainNetworkListBuilder.pretrain(false);

        int layerIndex = 0;
        for (int i = 0; i < encodeLayerDimensions.length; i++) {
            if (i == 0) {
                trainNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(encodeLayerActivationFunction)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(rawVectorDimension)
                                .nOut(encodeLayerDimensions[i])
                                .build());

            } else {
                trainNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(encodeLayerActivationFunction)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(encodeLayerDimensions[i - 1])
                                .nOut(encodeLayerDimensions[i])
                                .build());

            }
        }
        trainNetworkListBuilder.layer(layerIndex++,
                new DenseLayer.Builder()
                        .activation(encodeLayerActivationFunction)
                        .weightInit(WeightInit.XAVIER)
                        .nIn(encodeLayerDimensions[encodeLayerDimensions.length - 1])
                        .nOut(latentVectorDimension)
                        .build());


        for (int i = 0; i < decodeLayerDimensions.length; i++) {
            if (i == 0) {
                trainNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(decodeLayerActivationFunction)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(latentVectorDimension)
                                .nOut(decodeLayerDimensions[i])
                                .build());

            } else {
                trainNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(decodeLayerActivationFunction)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(decodeLayerDimensions[i - 1])
                                .nOut(decodeLayerDimensions[i])
                                .build());

            }
        }

        trainNetworkListBuilder.layer(layerIndex++,
                new OutputLayer.Builder()
                        .activation(outputLayerActivationFunction)
                        .lossFunction(outputLayerLoss)
                        .weightInit(WeightInit.XAVIER)
                        .nIn(decodeLayerDimensions[decodeLayerDimensions.length - 1])
                        .nOut(rawVectorDimension)
                        .build());

        MultiLayerConfiguration conf = trainNetworkListBuilder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        this.trainNetwork = network;
        this.trainNetwork.init();
    }

    public void fit(INDArray indArray) {
        this.trainNetwork.fit(indArray);
    }

    public void fit(DataSet dataSet) {
        this.trainNetwork.fit(dataSet);
    }

    public void fit(DataSetIterator dataSetIterator) {
        trainNetwork.fit(dataSetIterator);
    }

    public MultiLayerNetwork getEncoder() throws Exception {

        NeuralNetConfiguration.ListBuilder encodeNetworkListBuilder = NeuralNetConfiguration.Builder.class.newInstance()
                .updater(updater)
                .learningRate(learningSpeed)
                .iterations(1)
                .list();

        encodeNetworkListBuilder.backprop(false);
        encodeNetworkListBuilder.pretrain(false);

        int layerIndex = 0;
        for (int i = 0; i < encodeLayerDimensions.length; i++) {
            if (i == 0) {
                encodeNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(encodeLayerActivationFunction)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(rawVectorDimension)
                                .nOut(encodeLayerDimensions[i])
                                .build());
            } else {
                encodeNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(encodeLayerActivationFunction)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(encodeLayerDimensions[i - 1])
                                .nOut(encodeLayerDimensions[i]).build());
            }
        }

        encodeNetworkListBuilder.layer(layerIndex++,
                new OutputLayer.Builder()
                        .activation(encodeLayerActivationFunction)
                        .weightInit(WeightInit.XAVIER)
                        .nIn(encodeLayerDimensions[encodeLayerDimensions.length - 1])
                        .nOut(latentVectorDimension)
                        .build());


        MultiLayerConfiguration encodeNetworkConf = encodeNetworkListBuilder.build();
        MultiLayerNetwork encodeNetwork = new MultiLayerNetwork(encodeNetworkConf);
        encodeNetwork.init();

        for (int i = 0; i < encodeLayerDimensions.length + 1; i++) {
            Layer trainedLayer = this.trainNetwork.getLayer(i);
            INDArray trainedParams = trainedLayer.params();
            Layer encodeLayer = encodeNetwork.getLayer(i);
            encodeLayer.setParams(trainedParams);
        }
        return encodeNetwork;
    }

    public MultiLayerNetwork getDecoder() throws Exception {
        NeuralNetConfiguration.ListBuilder decodeNetworkListBuilder = NeuralNetConfiguration.Builder.class.newInstance()
                .updater(updater)
                .learningRate(learningSpeed)
                .iterations(1)
                .list();

        decodeNetworkListBuilder.backprop(true);
        decodeNetworkListBuilder.pretrain(false);

        int layerIndex = 0;
        for (int i = 0; i < decodeLayerDimensions.length; i++) {
            if (i == 0) {
                decodeNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(decodeLayerActivationFunction)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(latentVectorDimension)
                                .nOut(decodeLayerDimensions[i])
                                .build());
            } else {
                decodeNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(decodeLayerActivationFunction)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(decodeLayerDimensions[i - 1])
                                .nOut(decodeLayerDimensions[i]).build());
            }
        }

        decodeNetworkListBuilder.layer(layerIndex++,
                new OutputLayer.Builder()
                        .activation(outputLayerActivationFunction)
                        .lossFunction(outputLayerLoss)
                        .weightInit(WeightInit.XAVIER)
                        .nIn(decodeLayerDimensions[decodeLayerDimensions.length - 1])
                        .nOut(rawVectorDimension)
                        .build());


        MultiLayerConfiguration decodeNetworkConf = decodeNetworkListBuilder.build();
        MultiLayerNetwork decodeNetwork = new MultiLayerNetwork(decodeNetworkConf);
        decodeNetwork.init();

        for (int i = 0; i < decodeLayerDimensions.length + 1; i++) {
            Layer trainedLayer = this.trainNetwork.getLayer(encodeLayerDimensions.length + 1 + i);
            INDArray trainedParams = trainedLayer.params();
            Layer decodeLayer = decodeNetwork.getLayer(i);
            decodeLayer.setParams(trainedParams);
        }
        return decodeNetwork;
    }
}
