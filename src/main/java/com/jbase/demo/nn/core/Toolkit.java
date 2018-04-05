package com.jbase.demo.nn.core;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static com.jbase.demo.nn.core.Calc.*;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by lee on 2018/3/31.
 */
public class Toolkit {

    public void grad2zero(Variable variable) {
        variable.backwardTreeOperation(var -> {
            var.setBackward(0);
            if (var.data.type > Tensor.SCALAR_TYPE) {
                var.setGrad(new Tensor(Nd4j.zeros(var.data.tensor.shape())));
            }
            if (var.data.type == Tensor.SCALAR_TYPE) {
                var.setGrad(new Tensor(Double.valueOf(0)));
            }
        });
    }

    public void backward(Variable variable) {
        if (variable.data.type == Tensor.SCALAR_TYPE) {
            //变量参与运算计数
            LinkedList<Variable> leaveVars = new LinkedList<>();
            leaveVars.add(variable);
            while (leaveVars.size() > 0) {
                Variable current = leaveVars.removeFirst();
                if (current.backward > 0) {
                    current.backward++;
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

            variable.setGrad(new Tensor(1));
            --variable.backward;
            LinkedList<Variable> variables = new LinkedList<>();    //这里一定是需要往后面计算导数的
            variables.add(variable);
            while (variables.size() > 0) {
                Variable current = variables.removeFirst();
                switch (current.getOperation()) {
                    case Add: {
                        Variable x = current.dependencies[0];
                        Variable y = current.dependencies[1];
                        if (current.data.type > Tensor.SCALAR_TYPE) {
                            x.grad = new Tensor(x.grad.tensor.add(current.grad.tensor));
                            y.grad = new Tensor(y.grad.tensor.add(current.grad.tensor));
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        } else if (current.data.type == Tensor.SCALAR_TYPE) {
                            x.grad = new Tensor(x.grad.scalar + current.grad.scalar);
                            y.grad = new Tensor(y.grad.scalar + current.grad.scalar);
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        } else {
                            throw new IllegalStateException("不支持");
                        }
                    }
                    break;
                    case Sub: {
                        Variable x = current.dependencies[0];
                        Variable y = current.dependencies[1];
                        if (current.data.type == Tensor.SCALAR_TYPE) {
                            x.grad = new Tensor(x.grad.scalar + current.grad.scalar);
                            y.grad = new Tensor(y.grad.scalar - 1 * current.grad.scalar);
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        } else if (current.data.type > Tensor.SCALAR_TYPE) {
                            x.grad = new Tensor(x.grad.tensor.add(current.grad.tensor));
                            y.grad = new Tensor(y.grad.tensor.add(current.grad.tensor.mul(-1)));
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        } else {
                            throw new IllegalStateException("不支持");
                        }
                    }
                    break;
                    case Mean: {
                        Variable x = current.dependencies[0];
                        if (current.data.type == Tensor.SCALAR_TYPE) {
                            if (x.data.type > Tensor.SCALAR_TYPE) {
                                INDArray x_data = x.data.tensor;
                                List<Integer> shape = new LinkedList<>();
                                int[] x_data_shape = x_data.shape();
                                for (int i = 0; i < x_data_shape.length; i++) {
                                    shape.add(x_data_shape[i]);
                                }
                                int product = shape.stream().reduce((a, b) -> a * b).get();
                                x.grad = new Tensor(x.grad.tensor.add(Nd4j.ones(x_data_shape).div(product).mul((current.grad.scalar))));
                                if (--x.backward == 0) {
                                    variables.add(x);
                                }
                            } else if (x.data.type == Tensor.SCALAR_TYPE) {
                                x.grad = new Tensor(x.grad.scalar + current.grad.scalar);
                                if (--x.backward == 0) {
                                    variables.add(x);
                                }
                            } else {
                                throw new IllegalStateException("不支持的类型");
                            }
                        } else {
                            throw new IllegalStateException("不支持的类型");
                        }
                    }
                    break;
                    case Sum: {
                        Variable x = current.dependencies[0];
                        if (current.data.type == Tensor.SCALAR_TYPE) {
                            if (x.data.type > Tensor.SCALAR_TYPE) {
                                INDArray x_data = x.data.tensor;
                                int[] x_data_shape = x_data.shape();
                                x.grad = new Tensor(x.grad.tensor.add(Nd4j.ones(x_data_shape).mul(current.grad.scalar)));
                                if (--x.backward == 0) {
                                    variables.add(x);
                                }
                            } else if (x.data.type == Tensor.SCALAR_TYPE) {
                                x.grad = new Tensor(x.grad.scalar + current.grad.scalar);
                                if (--x.backward == 0) {
                                    variables.add(x);
                                }
                            } else
                                throw new IllegalStateException("不支持");
                        } else {
                            throw new IllegalStateException("不支持");
                        }
                    }
                    break;
                    case Square: {
                        Variable x = current.dependencies[0];
                        if (current.data.type > Tensor.SCALAR_TYPE) {
                            INDArray x_data = x.data.tensor;
                            INDArray grad = x.grad.tensor.add(x_data.mul(2).mul(current.grad.tensor));
                            x.grad = new Tensor(grad);
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        } else if (current.data.type == Tensor.SCALAR_TYPE) {
                            x.grad = new Tensor(x.grad.scalar + x.data.scalar * 2 * current.grad.scalar);
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        } else {
                            throw new IllegalStateException("不支持");
                        }
                    }
                    break;
                    case MatMul: {
                        if (current.data.type > Tensor.SCALAR_TYPE) {
                            Variable x = current.dependencies[0];
                            Variable y = current.dependencies[1];
                            y.grad = new Tensor(y.grad.tensor.add((x.data.tensor.transpose().mmul(current.grad.tensor))));
                            x.grad = new Tensor(x.grad.tensor.add(current.grad.tensor.mmul((y.data.tensor.transpose()))));
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        } else {
                            throw new IllegalStateException("非法");
                        }
                    }
                    break;
                    case Sigmoid: {
                        if (current.data.type > Tensor.SCALAR_TYPE) {
                            Variable x = current.dependencies[0];
                            INDArray x_data = current.dependencies[0].data.tensor;
                            INDArray x_data_c = Nd4j.create(x_data.shape());
                            Nd4j.copy(x_data, x_data_c);
                            x.grad = new Tensor(x.grad.tensor.add(Activation.SIGMOID.getActivationFunction().backprop(x_data_c, current.grad.tensor).getFirst()));
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        } else if (current.data.type == Tensor.SCALAR_TYPE) {
                            Variable y = current.dependencies[0];
                            double x = y.data.scalar;
                            y.grad = new Tensor(y.grad.scalar + mathSigmoid(x) * (1 - mathSigmoid(x)));
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        } else {
                            throw new IllegalStateException("不支持");
                        }
                    }
                    break;
                    case TanH: {
                        if (current.data.type > Tensor.SCALAR_TYPE) {
                            Variable x = current.dependencies[0];
                            INDArray x_data = x.data.tensor;
                            INDArray x_data_c = Nd4j.zeros(x_data.shape());
                            Nd4j.copy(x_data, x_data_c);
                            x.grad = new Tensor
                                    (x.grad.tensor.add(Activation.TANH.getActivationFunction().backprop(x_data_c, current.grad.tensor).getFirst()));
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        }else
                        if (current.data.type == Tensor.SCALAR_TYPE) {
                            Variable y = current.dependencies[0];
                            double x = y.data.scalar;
                            y.grad = new Tensor(y.grad.scalar + (1 - Math.pow(mathTanh(x), 2)));
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        }else {
                            throw new IllegalStateException("不支持");
                        }
                    }
                    break;
                    case RELU: {
                        if (current.data.type > Tensor.SCALAR_TYPE) {
                            Variable x = current.dependencies[0];
                            INDArray x_data = x.data.tensor;
                            INDArray x_data_cpp = Nd4j.create(x_data.shape());
                            Nd4j.copy(x_data, x_data_cpp);
                            x.grad = new Tensor(x.grad.tensor.add(Activation.RELU.getActivationFunction().backprop(x_data_cpp, current.grad.tensor).getFirst()));
                            if (--x.backward == 0) {
                                variables.add(x);
                            }
                        }else
                        if (current.data.type == Tensor.SCALAR_TYPE) {
                            Variable y = current.dependencies[0];
                            double x = y.data.scalar;
                            y.grad = new Tensor(y.grad.scalar + (x >= 0 ? 1 : 0));
                            if (--y.backward == 0) {
                                variables.add(y);
                            }
                        }else {
                            throw new IllegalStateException("不支持");
                        }
                    }
                    break;
                    case AddVec: {

                        Variable x = current.dependencies[0];
                        Variable y = current.dependencies[1];
                        INDArray c_grad = current.grad.tensor;
                        INDArray x_data = x.data.tensor;

                        INDArray oneColumn = Nd4j.ones(new int[]{1, x_data.shape()[0]});
                        INDArray vecGradMat = oneColumn.mmul(c_grad);
                        x.grad = new Tensor(x.grad.tensor.add(c_grad));
                        y.grad = new Tensor(y.grad.tensor.add(vecGradMat.getRow(0)));

                        if (--y.backward == 0) {
                            variables.add(y);
                        }

                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    }
                    break;
                    case MulScalar: {
                        INDArray current_grad = current.grad.tensor;
                        Variable x = current.dependencies[0];
                        Variable y = current.dependencies[1];
                        INDArray x_data = x.data.tensor;
                        double y_data = y.data.scalar;
                        y.grad = new Tensor(y.grad.scalar + x_data.mul(current_grad).sumNumber().doubleValue());
                        x.grad = new Tensor(x.grad.tensor.add(current_grad.mul(y_data)));
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    }
                    break;
                    case Hadamard: {
                        INDArray current_gradient = current.grad.tensor;
                        Variable x = current.dependencies[0];
                        Variable y = current.dependencies[1];
                        INDArray x_data = x.data.tensor;
                        INDArray y_data = y.data.tensor;
                        x.grad = new Tensor(x.grad.tensor.add(current_gradient.mul(y_data)));
                        y.grad = new Tensor(y.grad.tensor.add(current_gradient.mul(x_data)));
                        if (--y.backward == 0) {
                            variables.add(y);
                        }
                        if (--x.backward == 0) {
                            variables.add(x);
                        }
                    }
                    case ASSIGN: {

                    }
                    break;
                    case Dot: {
                        throw new UnsupportedOperationException("不支持点积的反射传播！");
                    }
                    case Kronecker: {
                        throw new UnsupportedOperationException("还未实现！");
                    }
                }
            }
            return;
        }
        throw new IllegalStateException("不支持反向传播的类型[" + variable.data.type + "]");
    }

}
