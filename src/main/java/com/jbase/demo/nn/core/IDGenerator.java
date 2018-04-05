package com.jbase.demo.nn.core;

public class IDGenerator {

    private static volatile long currentId = 0L;

    public static synchronized long nextId() {
        return currentId++;
    }

}
