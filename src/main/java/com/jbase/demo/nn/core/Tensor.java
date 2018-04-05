package com.jbase.demo.nn.core;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

@Data
public class Tensor {

    public final static int SCALAR_TYPE = 0;
    public final static int MATRIX_TYPE = 2;

    public int type;
    public double scalar;
    public INDArray tensor;

    public Tensor(INDArray tensor) {
        this.tensor = tensor;
        this.type = tensor.rank();
    }

    public Tensor(double scalar) {
        this.scalar = scalar;
        this.type = SCALAR_TYPE;
    }
}
