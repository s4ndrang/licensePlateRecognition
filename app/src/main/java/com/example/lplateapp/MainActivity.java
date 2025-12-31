package com.example.lplateapp;


import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;

import androidx.activity.EdgeToEdge;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;

import com.example.lplateapp.databinding.ActivityMainBinding;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.Text;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicInteger;


public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding binding;
    private Uri destUri;
    private File photoFile;
    private File imageFile;
    private TextRecognizer recognizer;
    private Interpreter tflite;
    private List<String> lpInfoList = new ArrayList<>();
    private List<String> registeredLPs = new ArrayList<>();

    private final ActivityResultLauncher<String> cameraRequestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(),
                    isGranted -> {
                        if (isGranted) {
                            openCamera();
                        }
                    });
    private final ActivityResultLauncher<Intent> cameraActivityResultLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                    result -> {
                        if (result.getResultCode() == Activity.RESULT_OK) {
                            Bitmap bitmap = null;
                            try {
                                bitmap = getCorrectlyOrientedBitmap(photoFile);
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                            if (bitmap != null) {
                                lpInfoList.clear();
                                binding.name.setText("");
                                performTextRecognition(bitmap);
                            }
                        }
                    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        registerLPs(registeredLPs);

        recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);
        binding.cameraButton.setOnClickListener(v -> {
            if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                cameraRequestPermissionLauncher.launch(Manifest.permission.CAMERA);
            } else {
                openCamera();
            }
        });
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void registerLPs(List<String> registeredLPs) {
        registeredLPs.add("007-CMD-228");
        registeredLPs.add("DK-1234-AB");
        registeredLPs.add("DK 4444 H");
        registeredLPs.add("AA-012-BC");
        registeredLPs.add("AA-796-CX");
    }

    private void openCamera() {
        createImageFile();
        try {
            destUri = FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", photoFile);
            ActivityResultContracts.TakePicture takePicture = new ActivityResultContracts.TakePicture();
            Intent takePictureIntent = takePicture.createIntent(this, destUri);
            if (takePictureIntent.resolveActivity(this.getPackageManager()) != null) {
                cameraActivityResultLauncher.launch(takePictureIntent);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void createImageFile() {
        try {
            photoFile = File.createTempFile(UUID.randomUUID() + "_", ".png", getCacheDir());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Bitmap getCorrectlyOrientedBitmap(File file) throws IOException {
        Bitmap bitmap = BitmapFactory.decodeFile(file.getAbsolutePath());
        ExifInterface exif = new ExifInterface(file.getAbsolutePath());
        int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        int rotationDegrees = exifToDegrees(orientation);

        if (rotationDegrees != 0) {
            Matrix matrix = new Matrix();
            matrix.postRotate(rotationDegrees);
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        }
        return bitmap;
    }

    private int exifToDegrees(int exifOrientation) {
        if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_90) return 90;
        else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_180) return 180;
        else if (exifOrientation == ExifInterface.ORIENTATION_ROTATE_270) return 270;
        return 0;
    }

    private void runOcrOnPlate(Bitmap plateBitmap, AtomicInteger remainingTasks) {
        InputImage image = InputImage.fromBitmap(plateBitmap, 0);

        recognizer.process(image)
                .addOnSuccessListener(visionText -> {
                    String registered = "";

                    List<Text.Line> allLines = new ArrayList<>();
                    for (Text.TextBlock block : visionText.getTextBlocks()) {
                        allLines.addAll(block.getLines());
                    }

                    // Sort by height descending
                    allLines.sort((line1, line2) -> {
                        Rect r1 = line1.getBoundingBox();
                        Rect r2 = line2.getBoundingBox();
                        if (r1 == null || r2 == null) return 0;
                        int h1 = r1.bottom - r1.top;
                        int h2 = r2.bottom - r2.top;
                        return Integer.compare(h2, h1); // shortest first
                    });

                    //validate the concatenation of the 2 largest lines
                    List<String> textsToValidate = getStrings(allLines);

                    // Now process lines in order of height
                    for (int i = 0; i < textsToValidate.size(); i++) {
                        String text = textsToValidate.get(i);
                        if (text.matches("^[A-Z0-9\\-:• ]{1,20}$")) {
                            //append all text within license plate to one single string
                            boolean isValidLP = validateLP(text, registeredLPs);
                            if (isValidLP) {
                                registered += "REGISTERED:\n" + text;
                                break;
                            } else {
                                lpInfoList.add("UNREGISTERED:\n" + text);
                            }
                            Log.i("LICENSEPLATE", "Detected text: " + text);
                        }
                    }

                    if (!registered.isEmpty()) {
                        synchronized (lpInfoList) {  // thread safety
                            lpInfoList.add(registered);
                        }
                    }

                    if (remainingTasks.decrementAndGet() == 0) {
                        StringBuilder sb = new StringBuilder();
                        for (String lpInfo : lpInfoList) {
                            sb.append(lpInfo).append("\n\n");
                        }
                        binding.name.setText((!registered.isEmpty()) ? registered : sb.toString());
                    }
                })
                .addOnFailureListener(Throwable::printStackTrace);
    }

    @NonNull
    private List<String> getStrings(List<Text.Line> allLines) {
        String largestText = allLines.get(0).getText().trim();
        List<String> textsToValidate = new ArrayList<>();
        textsToValidate.add(largestText);
        if (allLines.size() > 1) {
            String secondLargestText = allLines.get(1).getText().trim();
            String largestAndSecondLargestText = allLines.get(0).getText().trim() + allLines.get(1).getText().trim();
            String secondLargestAndLargestText = allLines.get(1).getText().trim() + allLines.get(0).getText().trim();
            textsToValidate.add(secondLargestText);
            textsToValidate.add(largestAndSecondLargestText);
            textsToValidate.add(secondLargestAndLargestText);
        }
        return textsToValidate;
    }

    public void performTextRecognition(Bitmap input) {
        List<Detection> detections = doInference(input); // YOLO detections
        AtomicInteger remainingTasks = new AtomicInteger(detections.size());
        Bitmap outputBitmap = drawDetections(input, detections); // optional: draw boxes
        binding.photo.setImageBitmap(outputBitmap);

        lpInfoList.clear(); // clear previous results

        for (Detection d : detections) {
            RectF r = d.rect;

            // Clamp to image bounds
            int left   = Math.max(0, (int) r.left);
            int top    = Math.max(0, (int) r.top);
            int right  = Math.min(outputBitmap.getWidth(), (int) r.right);
            int bottom = Math.min(outputBitmap.getHeight(), (int) r.bottom);

            int width  = right - left;
            int height = bottom - top;
            if (width <= 0 || height <= 0) continue;

            // Crop plate from the bitmap
            Bitmap plateBitmap = Bitmap.createBitmap(outputBitmap, left, top, width, height);

            // Optional: upscale for better OCR
            plateBitmap = Bitmap.createScaledBitmap(plateBitmap, width*2, height*2, true);

            // Run OCR asynchronously
            runOcrOnPlate(plateBitmap, remainingTasks);
        }
    }

    private boolean validateLP(String text, List<String> registeredLPs) {
        if (registeredLPs.contains(text)) {
            return true;
        }
        text = text.replace("-", "").replace(":", "").toUpperCase().trim();
        for (String registered : registeredLPs) {
            registered = registered.replace("-", "").replace(":", "").toUpperCase().trim();
            if (text.contains(registered)) {
                return true;
            }
        }
        return false;
    }


    private MappedByteBuffer loadModelFile() throws IOException {
        //Open the model using an input stream and memory map it to load
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("best_lp_int8.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public List<Detection> doInference(Bitmap image) {
        List<Detection> detections = new ArrayList<>();
        // Resize image
        int inputSize = 640;
        Bitmap resized = Bitmap.createScaledBitmap(image, inputSize, inputSize, true);

        // Prepare input
        float[][][][] input = new float[1][640][640][3];
        int[] pixels = new int[inputSize * inputSize];
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize);

        int pixelIndex = 0;
        for (int y = 0; y < inputSize; y++) {
            for (int x = 0; x < inputSize; x++) {
                int pixel = pixels[pixelIndex++];

                // Extract RGB
                float r = ((pixel >> 16) & 0xFF) / 255.0f;
                float g = ((pixel >> 8) & 0xFF) / 255.0f;
                float b = (pixel & 0xFF) / 255.0f;

                input[0][y][x][0] = r;
                input[0][y][x][1] = g;
                input[0][y][x][2] = b;
            }
        }

        // Prepare output array
        float[][][] output = new float[1][5][8400];

        // Run inference
        tflite.run(input, output);

        int numBoxes = output[0][0].length; // 8400
        double THRESHOLD = 0.08;
        int count = 0;
        for (int i = 0; i < numBoxes; i++) {
            float score = output[0][4][i];
            if (score < THRESHOLD) continue; // filter low confidence
            count++;
            float x = output[0][0][i];
            float y = output[0][1][i];
            float w = output[0][2][i];
            float h = output[0][3][i];

            // convert to original bitmap coordinates
            float left = (x - w / 2) * image.getWidth();
            float top = (y - h / 2) * image.getHeight();
            float right = (x + w / 2) * image.getWidth();
            float bottom = (y + h / 2) * image.getHeight();

            boolean shouldAdd = true;

            for (Detection d : detections) {
                if (areBoxesSimilar(d.rect, x, y, image)) {
                    shouldAdd = false;
                    break;
                }
            }

            if (shouldAdd) {
                detections.add(new Detection(new RectF(left, top, right, bottom), score));
            }

        }
        return detections;
    }

    //measures if centers are similar
    private boolean areBoxesSimilar(RectF r, float x, float y, Bitmap image) {
        float cx = x * image.getWidth();
        float cy = y * image.getHeight();

        float rcx = (r.left + r.right) / 2f;
        float rcy = (r.top + r.bottom) / 2f;

        float eps = 400f; // pixels

        return Math.abs(cx - rcx) < eps &&
                Math.abs(cy - rcy) < eps;
    }

    public Bitmap drawDetections(Bitmap bitmap, List<Detection> detections) {
        Bitmap outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(outputBitmap);
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4);

        for (Detection d : detections) {
            canvas.drawRect(d.rect, paint);
        }

        return outputBitmap;
    }


}
