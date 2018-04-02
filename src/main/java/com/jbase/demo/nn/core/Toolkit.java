package com.jbase.demo.nn.core;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static cn.redguest.jbase.ai.nn.core.Calc.*;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by lee on 2018/3/31.
 */
public class Toolkit {

    public void zeroGrad(Variable variable) {
        variable.applyTreeOperation(var -> {
            var.setBackward(0);
            if (INDArray.class.isInstance(var.data)) {
                var.setGrad(Nd4j.zeros(((INDArray) var.data).shape()));
            }
            if (Double.class.isInstance(var.data)) {
                var.setGrad(Double.valueOf(0));
            }
        });
    }

    public void backward(Variable variable) {
        if (Double.class.isInstance(variable.data)) {

            //变量参与运算计数
            LinkedList<Variable> leaveVars = new LinkedList<>();
            leaveVars.add(variable);
            while (leaveVars.size() > 0) {
                Variable current = leaveVars.removeFirst();
                if (current.backward > 0) {
                    current.backward++;
                    continue;
                } else {
                    current.backward++;
                    if (current.getDependencies() != null && current.getDependencies().length > 0) {
                        Variable[] dependencies = current.getDependencies();
                        for (Variable dep : dependencies) {
                            leaveVars.add(dep);
                        }
                    }
                }
            }

            variable.setGrad(1D);
            --variable.backward;
            LinkedList<Variable> variables = new LinkedList<>();    //这里一定是需要往后面计算导数的
            variables.add(variable);
            while (variables.size() > 0) {
                Variable current = variables.removeFirst();
                switch (current.getOperation()) {
                    case Add: {
                        Variable x = current.dependencies[0];
                        Variable y = current.dependencies[1];
                        if (INDArray.class.isInstance(current.data)) {
                            x.grad = ((INDArray) x.grad).add((INDArray) current.grad);
                            y.grad = ((INDArray) y.grad).add((INDArray) current.grad);
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        }
                        if (Double.class.isInstance(current.data)) {
                            x.grad = ((Double) x.grad) + ((Double) current.grad);
                            y.grad = ((Double) y.grad) + ((Double) current.grad);
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        }
                    }
                    break;
                    case Sub: {
                        Variable x = current.dependencies[0];
                        Variable y = current.dependencies[1];
                        if (Double.class.isInstance(current.data)) {
                            x.grad = (Double) x.grad + (Double) current.grad;
                            y.grad = (Double) y.grad - 1 * (Double) current.grad;
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        }
                        if (INDArray.class.isInstance(current.data)) {
                            x.grad = ((INDArray) x.grad).add((INDArray) current.grad);
                            y.grad = ((INDArray) y.grad).add(((INDArray) current.grad).mul(-1));
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        }
                    }
                    break;
                    case Mean: {
                        Variable x = current.dependencies[0];
                        if (Double.class.isInstance(current.data)) {
                            if (INDArray.class.isInstance(x.data)) {
                                INDArray xM = (INDArray) x.data;
                                List<Integer> shape = new LinkedList<>();
                                int[] shapeOfXM = xM.shape();
                                for (int i = 0; i < shapeOfXM.length; i++) {
                                    shape.add(shapeOfXM[i]);
                                }
                                int product = shape.stream().reduce((a, b) -> a * b).get();
                                x.grad = ((INDArray) x.grad).add(Nd4j.ones(shapeOfXM).div(product).mul(((Double) current.grad)));
                                if (--x.backward == 0) {
                                    variables.add(x);
                                }
                            }
                            if (Double.class.isInstance(x.data)) {
                                x.grad = (Double) x.grad + (Double) current.grad;
                                if (--x.backward == 0) {
                                    variables.add(x);
                                }
                            }
                        }
                    }
                    break;
                    case Sum: {
                        Variable x = current.dependencies[0];
                        if (Double.class.isInstance(current.data)) {
                            if (INDArray.class.isInstance(x.data)) {
                                INDArray xM = (INDArray) x.data;
                                int[] shapeOfXM = xM.shape();
                                x.grad = ((INDArray) x.grad).add(Nd4j.ones(shapeOfXM).mul(((Double) current.grad)));
                                if (--x.backward == 0) {
                                    variables.add(x);
                                }
                            }
                            if (Double.class.isInstance(x.data)) {
                                x.grad = (Double) x.grad + (Double) current.grad;
                                if (--x.backward == 0) {
                                    variables.add(x);
                                }
                            }
                        }
                    }
                    break;
                    case Square: {
                        Variable x = current.dependencies[0];
                        if (INDArray.class.isInstance(current.data)) {
                            INDArray x_data = (INDArray) x.data;
                            x.grad = ((INDArray) x.grad).add(x_data.mul(2).mul((INDArray) current.grad));
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        }
                        if (Double.class.isInstance(current.data)) {
                            x.grad = (Double) x.grad + ((Double) x.data) * 2 * ((Double) current.grad);
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        }
                    }
                    break;
                    case MatMul: {
                        Variable x = current.dependencies[0];
                        Variable y = current.dependencies[1];
                        y.grad = ((INDArray) y.grad).add(((INDArray) x.data).transpose().mmul((INDArray) current.grad));
                        x.grad = ((INDArray) x.grad).add(((INDArray) current.grad).mmul(((INDArray) y.data).transpose()));
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                    }
                    break;
                    case Sigmoid: {
                        if (INDArray.class.isInstance(current.data)) {
                            Variable x = current.dependencies[0];
                            INDArray x_data = (INDArray) current.dependencies[0].data;
                            INDArray x_data_c = Nd4j.create(x_data.shape());
                            Nd4j.copy(x_data, x_data_c);
                            x.grad = ((INDArray) x.grad).add(Activation.SIGMOID.getActivationFunction().backprop(x_data_c, (INDArray) current.grad).getFirst());
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        }
                        if (Double.class.isInstance(current.data)) {
                            Variable y = current.dependencies[0];
                            Double x = (Double) y.data;
                            y.grad = (Double) y.grad + mathSigmoid(x) * (1 - mathSigmoid(x));
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        }
                    }
                    break;
                    case TanH: {
                        if (INDArray.class.isInstance(current.data)) {
                            Variable x = current.dependencies[0];
                            INDArray x_data = (INDArray) x.data;
                            INDArray x_data_c = Nd4j.zeros(x_data.shape());
                            Nd4j.copy(x_data, x_data_c);
                            x.grad = ((INDArray) x.grad).add(Activation.TANH.getActivationFunction().backprop(x_data_c, (INDArray) current.grad).getFirst());
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        }
                        if (Double.class.isInstance(current.data)) {
                            Variable y = current.dependencies[0];
                            Double x = (Double) y.data;
                            y.grad = (Double) y.grad + (1 - Math.pow(mathTanh(x), 2));
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        }
                    }
                    break;
                    case RELU: {
                        if (INDArray.class.isInstance(current.data)) {
                            Variable x = current.dependencies[0];
                            INDArray x_data = (INDArray) x.data;
                            INDArray x_data_cpp = Nd4j.create(x_data.shape());
                            Nd4j.copy(x_data, x_data_cpp);
                            x.grad = ((INDArray) x.grad).add(Activation.RELU.getActivationFunction().backprop(x_data_cpp, (INDArray) current.grad).getFirst());
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        }
                        if (Double.class.isInstance(current.data)) {
                            Variable y = current.dependencies[0];
                            Double x = (Double) y.data;
                            y.grad = (Double) y.grad + (x >= 0 ? 1 : 0);
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        }
                    }
                    break;
                    case AddVec:{

                        Variable x = current.dependencies[0];
                        Variable y = current.dependencies[1];
                        INDArray c_grad = (INDArray) current.grad;
                        INDArray x_data = (INDArray) x.data;

                        INDArray oneColumn = Nd4j.ones(new int[]{1,x_data.shape()[0]});
                        INDArray vecGradMat = oneColumn.mmul(c_grad);
                        x.grad = ((INDArray)x.grad).add(c_grad);
                        y.grad = ((INDArray)y.grad).add(vecGradMat.getRow(0));

                        if (--y.backward == 0) {
                            variables.add(y);
                        }

                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    }
                }
            }
            return;
        }
        throw new IllegalStateException("不支持反向传播的类型[" + variable.data.getClass().getName() + "]");
    }

}
