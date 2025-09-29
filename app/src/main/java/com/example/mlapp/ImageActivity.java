package com.example.mlapp;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.example.mlapp.databinding.ActivityImageBinding;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ImageActivity extends AppCompatActivity {

    private ActivityImageBinding binding;
    private Interpreter tflite;
    private Map<Integer, String> imageMap;
    private List<Integer> intImages;
    private int currentImage;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        binding = ActivityImageBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
        loadIntImages();

        binding.back.setOnClickListener(v -> {
            finish();
        });

        binding.takePhoto.setOnClickListener(v -> {
            Toast.makeText(this, "TAKING PHOTO NOW...", Toast.LENGTH_SHORT).show();
        });

        binding.photoLibrary.setOnClickListener(v -> {
            currentImage = intImages.get(((int) (Math.random() * intImages.size())) - 1);
            binding.photo.setImageResource(currentImage);
            binding.predict.setVisibility(View.VISIBLE);
            binding.takePhoto.setVisibility(View.GONE);
            binding.photoLibrary.setVisibility(View.GONE);
        });

        binding.predict.setOnClickListener(v -> {
            Toast.makeText(this, "PREDICTING NOW...", Toast.LENGTH_SHORT).show();
            // 1️⃣ Decode drawable into a Bitmap
            Bitmap bitmap = BitmapFactory.decodeResource(getResources(), currentImage);
            String prediction = doInference(bitmap);
            binding.output.setText(prediction);
        });

        loadImageMap();

        //Create tflite object, loaded from the model file
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public String doInference(Bitmap image) {
        // Preprocess your image (Bitmap -> float array)
        float[][][] input = new float[1][28][28]; // 1 batch, 28x28, 1 channel
        for (int y=0; y<28; y++) {
            for (int x=0; x<28; x++) {
                int pixel = image.getPixel(x, y);
                // grayscale conversion, invert if needed
                float value = 255 - ((pixel >> 16) & 0xFF);
                input[0][y][x] = value / 255.0f;
            }
        }

        // Prepare output array
        float[][] output = new float[1][10];  // 10 classes (0-9)

        // Run inference
        tflite.run(input, output);  // THIS IS THE KEY STEP

        // Find predicted class
        int maxIndex = 0;
        float maxProb = 0;
        for (int i = 0; i < 10; i++) {
            System.out.println("i = " + imageMap.get(i) + "Predicted float: " + output[0][i]);
            if (output[0][i] > maxProb) {
                maxProb = output[0][i];
                maxIndex = i;
            }
        }
        System.out.println("Predicted class: " + imageMap.get(maxIndex));

        return imageMap.get(maxIndex);
    }

    //Memory-map the model file in Assets
    private MappedByteBuffer loadModelFile() throws IOException {
        //Open the model using an input stream and memory map it to load
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("tf_lite_float_16_model.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private void loadImageMap() {
        imageMap = new HashMap<>();
        imageMap.put(0, "T-shirt//top");
        imageMap.put(1, "Trouser");
        imageMap.put(2, "Pullover");
        imageMap.put(3, "Dress");
        imageMap.put(4, "Coat");
        imageMap.put(5, "Sandal");
        imageMap.put(6, "Shirt");
        imageMap.put(7, "Sneaker");
        imageMap.put(8, "Bag");
        imageMap.put(9, "Ankle boot");
    }

    private void loadIntImages() {
        intImages = new ArrayList<>();
        intImages.add(R.drawable.bag);
        intImages.add(R.drawable.shoe_pointing_left);
        intImages.add(R.drawable.shoe_pointing_right);
        intImages.add(R.drawable.trouser);
        intImages.add(R.drawable.tshirt);
        intImages.add(R.drawable.shirt);
        intImages.add(R.drawable.pullover);
    }

}
