package com.jbase.demo.nn.core;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * Created by lee on 2018/3/27.
 */
@AllArgsConstructor
public enum Operation {

    MatMul("矩阵-乘积"),
    Dot("矩阵-标量积"),
    Hadamard("矩阵-哈达马积"),
    Kronecker("张量-直积"),
    Sum("矩阵-求和"),
    Mean("矩阵-求平均"),
    Add("同类-加法"),
    AddVec("矩阵-加向量"),
    Sub("同类-减法"),
    Sigmoid("Sigmoid函數"),
    TanH("Tanh函數"),
    Square("平方"),
    RELU("Relu函数"),
    ASSIGN("指定");

    private String name;

}
