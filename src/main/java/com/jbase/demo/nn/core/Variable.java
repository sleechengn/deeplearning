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

    public boolean isRequireGrad = false;

    public Object data;

    public Object grad;

    public Operation operation;

    public Variable[] dependencies;

    public int backward = 0;

    public void applyTreeOperation(Consumer<Variable> oper) {
        oper.accept(this);
        if (dependencies != null) {
            for (Variable dependency : dependencies) {
                dependency.applyTreeOperation(oper);
            }
        }
    }

    public Variable(Object data) {
        this.isRequireGrad = false;
        this.data = data;
        this.operation = Operation.ASSIGN;
    }

    public Variable(Object data, boolean isRequireGrad) {
        this.isRequireGrad = isRequireGrad;
        this.operation = Operation.ASSIGN;
        this.data = data;
    }

    public Variable matMul(Variable variable) {
        INDArray x = (INDArray) this.data;
        INDArray y = (INDArray) variable.data;
        INDArray z = x.mmul(y);

        Variable var = new Variable();
        var.setDependencies(new Variable[]{this, variable});
        var.setOperation(Operation.MatMul);
        var.setData(z);
        return var;
    }

    public Variable add(Variable variable) {
        if (INDArray.class.isInstance(data)) {
            INDArray x = (INDArray) this.data;
            INDArray y = (INDArray) variable.data;
            INDArray z = x.add(y);
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Add);
            var.setData(z);
            return var;
        }
        if (Double.class.isInstance(data)) {
            Double x = (Double) data;
            Double y = (Double) variable.data;
            Double z = x + y;

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Add);
            var.setData(z);
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getClass().getName() + "]");
    }

    public Variable sub(Variable variable) {
        if (INDArray.class.isInstance(data)) {
            INDArray x = (INDArray) this.data;
            INDArray y = (INDArray) variable.data;
            INDArray z = x.sub(y);
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Sub);
            var.setData(z);
            return var;
        }
        if (Double.class.isInstance(data)) {
            Double x = (Double) data;
            Double y = (Double) variable.data;
            Double z = x - y;

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this, variable});
            var.setOperation(Operation.Sub);
            var.setData(z);
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getClass().getName() + "]");
    }

    public Variable sigmoid() {
        if (INDArray.class.isInstance(this.data)) {
            INDArray x = (INDArray) this.data;
            INDArray y = Nd4j.zeros(x.shape());
            Nd4j.copy(x, y);
            INDArray z = Activation.SIGMOID.getActivationFunction().getActivation(y, true);

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Sigmoid);
            var.setData(z);
            return var;
        }
        if (Double.class.isInstance(this.data)) {
            Double x = (Double) this.data;
            Double z, y;
            y = Calc.mathSigmoid(x);
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Sigmoid);
            var.setData(z);
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getClass().getName() + "]");
    }

    public Variable tanh() {
        if (INDArray.class.isInstance(this.data)) {
            INDArray x = (INDArray) this.data;
            INDArray y = Nd4j.zeros(x.shape());
            Nd4j.copy(x, y);
            INDArray z = Activation.TANH.getActivationFunction().getActivation(y, true);

            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.TanH);
            var.setData(z);
            return var;
        }
        if (Double.class.isInstance(this.data)) {
            Double x = (Double) this.data;
            Double z, y;
            y = mathTanh(x);
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.TanH);
            var.setData(z);
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getClass().getName() + "]");
    }

    public Variable mean() {
        if (INDArray.class.isInstance(this.data)) {
            INDArray x = (INDArray) this.data;
            Double y, z;
            y = x.meanNumber().doubleValue();
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Mean);
            var.setData(z);
            return var;
        }
        if (Double.class.isInstance(this.data)) {
            return this;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getClass().getName() + "]");
    }

    public Variable sum() {
        if (INDArray.class.isInstance(this.data)) {
            INDArray x = (INDArray) this.data;
            Double y, z;
            y = x.sumNumber().doubleValue();
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Sum);
            var.setData(z);
            return var;
        }
        if (Double.class.isInstance(this.data)) {
            return this;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getClass().getName() + "]");
    }

    public Variable square() {
        if (INDArray.class.isInstance(this.data)) {
            INDArray x = (INDArray) this.data;
            INDArray y, z;
            y = x.mul(x);
            z = y;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Square);
            var.setData(z);
            return var;
        }
        if (Double.class.isInstance(this.data)) {
            Double x = (Double) this.data;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.Square);
            var.setData(Math.pow(x, 2));
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getClass().getName() + "]");
    }

    public Variable relu() {
        if (INDArray.class.isInstance(this.data)) {
            INDArray x = (INDArray) this.data;
            INDArray y = Nd4j.create(x.shape());
            Nd4j.copy(x, y);
            INDArray z = Activation.RELU.getActivationFunction().getActivation(y, true);
            ;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.RELU);
            var.setData(z);
            return var;
        }
        if (Double.class.isInstance(this.data)) {
            Double x = (Double) this.data;
            Variable var = new Variable();
            var.setDependencies(new Variable[]{this});
            var.setOperation(Operation.RELU);
            var.setData(x > 0 ? x : 0D);
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getClass().getName() + "]");
    }

    public Variable addVec(Variable variable) {
        if (INDArray.class.isInstance(this.data)) {
            INDArray x = (INDArray) this.data;
            INDArray y = (INDArray) variable.data;
            int[] _x_shape = x.shape();
            int[] _y_shape = y.shape();

            int x_row = _x_shape[0];
            int x_column = _x_shape[1];
            int y_column = _y_shape[1];

            if (_x_shape.length != 2) {
                throw new IllegalStateException("现在只支持矩阵加向量，此变量不是矩阵，张量阶：" + _x_shape.length);
            }
            if (_y_shape.length != 2 || _y_shape[0] != 1) {
                throw new IllegalStateException("些操作只能与向量相加，此变量不是向量，张量阶：" + _y_shape.length);
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
            var.setData(z);
            return var;
        }
        throw new IllegalStateException("错误的值类型,不支持[" + data.getClass().getName() + "]");
    }

}
