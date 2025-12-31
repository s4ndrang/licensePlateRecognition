package com.example.lplateapp;

import android.graphics.RectF;

public class Detection {
    public RectF rect;  // Bounding box
    public float score; // Confidence

    public Detection(RectF rect, float score) {
        this.rect = rect;
        this.score = score;
    }
}
