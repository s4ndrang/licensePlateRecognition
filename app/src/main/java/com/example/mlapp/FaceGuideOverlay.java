package com.example.mlapp;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import androidx.annotation.Nullable;

public class FaceGuideOverlay extends View {

    private final Paint strokePaint;
    private RectF ovalRect;
    private float ovalWidthFraction = 0.6f;   // fraction of view width
    private float ovalHeightFraction = 0.55f; // fraction of view height
    private float verticalOffsetPx = 0f;      // positive = move down, negative = move up

    public FaceGuideOverlay(Context context) {
        this(context, null);
    }

    public FaceGuideOverlay(Context context, @Nullable AttributeSet attrs) {
        super(context, attrs);

        // Paint for the oval outline
        strokePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        strokePaint.setStyle(Paint.Style.STROKE);
        strokePaint.setStrokeWidth(dpToPx(4));         // stroke thickness
        strokePaint.setColor(0xFF00DDFF);              // #00DDFF (opaque)
        strokePaint.setAlpha(255);                     // fully visible

        // We want the View itself to be transparent, so no background drawing here.
        setWillNotDraw(false); // allow onDraw to be called
    }

    // Optional setters so you can adjust from code:
    public void setOvalSizeFractions(float widthFraction, float heightFraction) {
        this.ovalWidthFraction = widthFraction;
        this.ovalHeightFraction = heightFraction;
        invalidate();
    }

    public void setVerticalOffsetPx(float offsetPx) {
        this.verticalOffsetPx = offsetPx;
        invalidate();
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        // Compute the rectangle for the oval when the view size is known
        float ovalW = w * ovalWidthFraction;
        float ovalH = h * ovalHeightFraction;

        float left = (w - ovalW) / 2f;
        float top = (h - ovalH) / 2f + verticalOffsetPx;
        float right = left + ovalW;
        float bottom = top + ovalH;

        ovalRect = new RectF(left, top, right, bottom);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (ovalRect != null) {
            canvas.drawOval(ovalRect, strokePaint);
        }
    }

    // Utility: convert dp to pixels
    private float dpToPx(float dp) {
        return dp * getResources().getDisplayMetrics().density;
    }
}
