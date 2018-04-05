package com.jbase.demo.nn.core;

import lombok.*;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static com.jbase.demo.nn.core.Calc.*;

import java.util.function.Consumer;

/**
 * Created by lee on 2018/3/27.
 */
@AllArgsConstructor
@NoArgsConstructor
@Setter
@Getter
@ToString
public class Variable {

    public long id = IDGenerator.nextId();

    public boolean isRequireGrad = false;

    public Tensor data;

    public Tensor grad;

    public Operation operation;

    public Variable[] dependencies;

    public int backward = 0;

    public void backwardTreeOperation(Consumer<Variable> operation) {
        operation.accept(this);
        if (dependencies != null) {
            for (Variable dependency : dependencies) {
                dependency.backwardTreeOperation(operation);
            }
        }
    }

    public Variable(INDArray data) {
        this.isRequireGrad = false;
        this.data = new Tensor(data);
        this.operation = Operation.ASSIGN;
    }

    public Variable(INDArray data, boolean isRequireGrad) {
        this.isRequireGrad = isRequireGrad;
        this.operation = Operation.ASSIGN;
        this.data = new Tensor(data);
    }

    public Variable(double data) {
        this.isRequireGrad = false;
        this.data = new Tensor(data);
        this.operation = Operation.ASSIGN;
    }

    public Variable(double data, boolean isRequireGrad) {
        this.isRequireGrad = isRequireGrad;
        this.operation = Operation.ASSIGN;
        this.data = new Tensor(data);
    }

    //矩阵乘法
    public Variable matMul(Variable variable) {
        if (this.data.type == Tensor.MATRIX_TYPE && variable.data.type == Tensor.MATRIX_TYPE) {
            INDArray x_data = this.data.tensor;
            INDArray y_data = variable.data.tensor;
            INDArray z_data = x_data.mmul(y_data);

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.MatMul);
            var.setData(new Tensor(z_data));
            return var;
        } else {
            throw new IllegalStateException("此运算符不支持的数据类型！");
        }
    }

    //矩阵（二阶张量）乘标量
    public Variable mulScalar(Variable variable) {
        if (this.data.type == Tensor.MATRIX_TYPE && variable.data.type == Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            double y = variable.data.scalar;
            INDArray z = x.mul(y);

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.MulScalar);
            var.setData(new Tensor(z));
            return var;
        } else {
            throw new IllegalStateException("此运算符不支持的数据类型！");
        }
    }

    //同类相加法
    public Variable add(Variable variable) {
        //张量加法
        if (this.data.type > Tensor.SCALAR_TYPE && variable.data.type == this.data.type) {
            INDArray x = this.data.tensor;
            INDArray y = variable.data.tensor;
            INDArray z = x.add(y);
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Add);
            var.setData(new Tensor(z));
            return var;
        } else if (this.data.type == Tensor.SCALAR_TYPE && this.data.type == variable.data.type) {
            double x = this.data.scalar;
            double y = variable.data.scalar;
            double z = x + y;

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Add);
            var.setData(new Tensor(z));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
    }

    //减法
    public Variable sub(Variable variable) {
        if (this.data.type > Tensor.SCALAR_TYPE && this.data.type == variable.data.type) {
            INDArray x = this.data.tensor;
            INDArray y = variable.data.tensor;
            INDArray z = x.sub(y);
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Sub);
            var.setData(new Tensor(z));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            double y = variable.data.scalar;
            double z = x - y;

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Sub);
            var.setData(new Tensor(z));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getType() + "]");
    }

    //哈達馬
    public Variable hadamard(Variable variable) {
        if (this.data.type > Tensor.SCALAR_TYPE && this.data.type == variable.data.type) {
            INDArray x = this.data.tensor;
            INDArray y = variable.data.tensor;
            INDArray z = x.mul(y);
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Hadamard);
            var.setData(new Tensor(z));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getType() + "]");
    }

    //激活函数
    public Variable sigmoid() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y = Nd4j.zeros(x.shape());
            Nd4j.copy(x, y);
            INDArray z = Activation.SIGMOID.getActivationFunction().getActivation(y, true);

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Sigmoid);
            var.setData(new Tensor(z));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            double z, y;
            y = mathSigmoid(x);
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Sigmoid);
            var.setData(new Tensor(z));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getType() + "]");
    }

    //双曲正切
    public Variable tanh() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y = Nd4j.zeros(x.shape());
            Nd4j.copy(x, y);
            INDArray z = Activation.TANH.getActivationFunction().getActivation(y, true);

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.TanH);
            var.setData(new Tensor(z));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            double z, y;
            y = mathTanh(x);
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.TanH);
            var.setData(new Tensor(z));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getClass().getName() + "]");
    }

    //平均值
    public Variable mean() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            double y, z;
            y = x.meanNumber().doubleValue();
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Mean);
            var.setData(new Tensor(z));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            return this;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getType() + "]");
    }

    //求和
    public Variable sum() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            double y, z;
            y = x.sumNumber().doubleValue();
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Sum);
            var.setData(new Tensor(z));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            return this;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
    }

    //平方
    public Variable square() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y, z;
            y = x.mul(x);
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Square);
            var.setData(new Tensor(z));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            double y, z;
            y = Math.pow(x, 2);
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Square);
            var.setData(new Tensor(z));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
    }

    //线性整流
    public Variable relu() {
        if (this.data.type > Tensor.SCALAR_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y = Nd4j.create(x.shape());
            Nd4j.copy(x, y);
            INDArray z = Activation.RELU.getActivationFunction().getActivation(y, true);
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.RELU);
            var.setData(new Tensor(z));
            return var;
        }
        if (this.data.type == Tensor.SCALAR_TYPE) {
            double x = this.data.scalar;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.RELU);
            var.setData(new Tensor(x > 0 ? x : 0D));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
    }

    public Variable addVec(Variable variable) {
        if (this.data.type > Tensor.SCALAR_TYPE && this.data.type == Tensor.MATRIX_TYPE) {
            INDArray x = this.data.tensor;
            INDArray y = variable.data.tensor;
            int[] _x_shape = x.shape();
            int[] _y_shape = y.shape();

            int x_row = _x_shape[0];
            int x_column = _x_shape[1];
            int y_column = _y_shape[1];

            if (x.rank() != 2) {
                throw new IllegalStateException("现在只支持矩阵加向量，此变量不是矩阵，张量阶：" + x.rank());
            }
            if (y.rank() != 2 || _y_shape[0] != 1) {
                throw new IllegalStateException("些操作只能与向量相加，此变量不是向量，张量阶：" + y.rank());
            }
            if (x_column != y_column) {
                throw new IllegalThreadStateException("列不相同，无法合并");
            }
            INDArray y_c = Nd4j.zeros(_y_shape);
            Nd4j.copy(y, y_c);
            y_c = y_c.reshape(1, y_column);

            //构造y向量到方阵对角线
            INDArray y_sq = Nd4j.zeros(new int[]{y_column, y_column});
            for (int i = 0; i < y_column; i++) {
                y_sq.putScalar(new int[]{i, i}, y_c.getDouble(0, i));
            }
            //构造全1的矩阵
            INDArray ones = Nd4j.ones(new int[]{x_row, x_column});
            INDArray stackVecMat = ones.mmul(y_sq);
            INDArray z = x.add(stackVecMat);

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.AddVec);
            var.setData(new Tensor(z));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.type + "]");
    }

}
