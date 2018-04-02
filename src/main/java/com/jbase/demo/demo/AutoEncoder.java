package com.jbase.demo.demo;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
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
public class AutoEncoder {


    public static void main(String[] arguments) throws Exception {
        Random rand = new Random();
        int simple_count = 3;       //三个随机样本
        int epoch = 2000;
        int input_vector_dimension = 18, output_vector_dimension = 18;    //输入(输出)维度
        int encode_vector_dimension = 2;    //编码维度
        int[] encode_layer_dimensions = {16, 10, 8, 5}; //编码层维度序列
        int[] decode_layer_dimensions = {5, 8, 10, 16}; //解码层维度序列
        int encode_output_layer_index = -1;
        Activation encode_layer_activation_function = Activation.TANH; //编码层激活函数
        Activation decode_layer_activation_function = Activation.TANH;  //解码层激活函数
        Activation output_layer_activation_function = Activation.SIGMOID; //输出层激活函数
        LossFunctions.LossFunction output_layer_loss = LossFunctions.LossFunction.MSE; //输出层损失函数
        NeuralNetConfiguration.ListBuilder trainNetworkListBuilder = NeuralNetConfiguration.Builder.class.newInstance()
                .updater(Updater.ADADELTA)
                .learningRate(1e-2)
                .iterations(1)
                .list();
        trainNetworkListBuilder.backprop(true);
        trainNetworkListBuilder.pretrain(false);
        int layerIndex = 0;
        for (int i = 0; i < encode_layer_dimensions.length; i++) {
            if (i == 0) {
                trainNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(encode_layer_activation_function)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(input_vector_dimension)
                                .nOut(encode_layer_dimensions[i])
                                .build());

            } else {
                trainNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(encode_layer_activation_function)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(encode_layer_dimensions[i - 1])
                                .nOut(encode_layer_dimensions[i])
                                .build());

            }
        }
        encode_output_layer_index = layerIndex;
        trainNetworkListBuilder.layer(layerIndex++,
                new DenseLayer.Builder()
                        .activation(encode_layer_activation_function)
                        .weightInit(WeightInit.XAVIER)
                        .nIn(encode_layer_dimensions[encode_layer_dimensions.length - 1])
                        .nOut(encode_vector_dimension)
                        .build());

        for (int i = 0; i < decode_layer_dimensions.length; i++) {
            if (i == 0) {
                trainNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(decode_layer_activation_function)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(encode_vector_dimension)
                                .nOut(decode_layer_dimensions[i])
                                .build());

            } else {
                trainNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(decode_layer_activation_function)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(decode_layer_dimensions[i - 1])
                                .nOut(decode_layer_dimensions[i])
                                .build());

            }
        }
        //创建最后一层解码层
        trainNetworkListBuilder.layer(layerIndex++,
                new OutputLayer.Builder()
                        .activation(output_layer_activation_function)
                        .lossFunction(output_layer_loss)
                        .weightInit(WeightInit.XAVIER)
                        .nIn(decode_layer_dimensions[decode_layer_dimensions.length - 1])
                        .nOut(output_vector_dimension)
                        .build());
        MultiLayerConfiguration trainNetworkConf = trainNetworkListBuilder.build();
        MultiLayerNetwork trainNetwork = new MultiLayerNetwork(trainNetworkConf);
        trainNetwork.init();
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
        int listenerFrequency = 1;
        trainNetwork.setListeners(new StatsListener(statsStorage, listenerFrequency));
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        //开始构建解码网络
        NeuralNetConfiguration.ListBuilder decodeNetworkListBuilder = NeuralNetConfiguration.Builder.class.newInstance()
                .updater(Updater.ADADELTA)
                .learningRate(1e-2)
                .iterations(1)
                .list();
        decodeNetworkListBuilder.backprop(true);
        decodeNetworkListBuilder.pretrain(false);
        layerIndex = 0;
        for (int i = 0; i < decode_layer_dimensions.length; i++) {
            if (i == 0) {
                decodeNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(decode_layer_activation_function)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(encode_vector_dimension)
                                .nOut(decode_layer_dimensions[i])
                                .build());
            } else {
                decodeNetworkListBuilder.layer(layerIndex++,
                        new DenseLayer.Builder()
                                .activation(decode_layer_activation_function)
                                .weightInit(WeightInit.XAVIER)
                                .nIn(decode_layer_dimensions[i - 1])
                                .nOut(decode_layer_dimensions[i]).build());
            }
        }
        decodeNetworkListBuilder.layer(layerIndex++,
                new OutputLayer.Builder()
                        .activation(output_layer_activation_function)
                        .lossFunction(output_layer_loss)
                        .weightInit(WeightInit.XAVIER)
                        .nIn(decode_layer_dimensions[decode_layer_dimensions.length - 1])
                        .nOut(output_vector_dimension)
                        .build());
        MultiLayerConfiguration decodeNetworkConf = decodeNetworkListBuilder.build();
        MultiLayerNetwork decodeNetwork = new MultiLayerNetwork(decodeNetworkConf);
        decodeNetwork.init();
        //生成样本数据
        double[][] trainData = new double[simple_count][input_vector_dimension];
        for (int i = 0; i < simple_count; i++) {
            for (int j = 0; j < input_vector_dimension; j++) {
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
            trainNetwork.fit(trainDataSetIt);
            if (i % 100 == 0)
                System.out.println("epoch:" + i);
        }
        //开始拷贝参数，这个是DL4j扯淡的一面
        for (int i = 0; i < decode_layer_dimensions.length + 1; i++) {
            Layer trainedLayer = trainNetwork.getLayer(encode_layer_dimensions.length + 1 + i);
            INDArray trainedParams = trainedLayer.params();
            Layer decodeLayer = decodeNetwork.getLayer(i);
            decodeLayer.setParams(trainedParams);
        }
        Layer encode_layer = trainNetwork.getLayer(encode_output_layer_index);
        //激活数据
        INDArray output = trainNetwork.output(features);
        System.out.println("编码前的特征:");
        System.out.println(features);
        INDArray encoded = encode_layer.activate();
        System.out.println("被编码的特征:");
        System.out.println(encoded);
        INDArray decoded = decodeNetwork.output(encoded);
        System.out.println("解码后的特征");
        System.out.println(decoded);
    }


}
