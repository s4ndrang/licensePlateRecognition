package com.example.mlapp;

import static android.view.View.GONE;
import static android.view.View.VISIBLE;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.video.Quality;
import androidx.camera.video.QualitySelector;
import androidx.camera.video.Recorder;
import androidx.camera.video.VideoCapture;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;

import com.bumptech.glide.Glide;
import com.example.mlapp.databinding.ActivityFaceBinding;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.gms.tasks.Task;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.Interpreter;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutionException;

public class FaceActivity extends AppCompatActivity {

    private ActivityFaceBinding binding;
    private Interpreter tflite;
    private Uri destUri;
    private Uri videoUri;
    private File photoFile;
    private File videoFile;
    private File imageFile;
    private List<float[]> patientEmbeddings = new ArrayList<>();
    private float[] embeddingNew;
    private String name;

    // High-accuracy landmark detection and face classification
    FaceDetectorOptions highAccuracyOpts =
            new FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                    .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL) //landmarks ex. eyebrows, nose etc
                    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE) //facial expressions
                    .setMinFaceSize(0.1f)
                    .build();

    FaceDetector detector;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        binding = ActivityFaceBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        Intent intent = getIntent();
        name = intent.getStringExtra("name");
        Log.i("TFLiteDebug", "onCreate called FaceActivity");
        binding.back.setOnClickListener(v -> {
            finish();
        });

        binding.takePhoto.setOnClickListener(this::takePhoto);
        binding.ok.setOnClickListener(this::openVideoRecorder);

        //Create tflite object, loaded from the model file
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (Exception e) {
            e.printStackTrace();
        }
        detector = FaceDetection.getClient(highAccuracyOpts);
        patientEmbeddings.clear();
    }

    public float[] doInference(Bitmap image) {
        // Resize image
        Bitmap resized = Bitmap.createScaledBitmap(image, 112, 112, true);

        // Prepare input
        float[][][][] input = new float[2][112][112][3];
        for (int y = 0; y < 112; y++) {
            for (int x = 0; x < 112; x++) {
                int pixel = resized.getPixel(x, y);
                input[0][y][x][0] = (((pixel >> 16) & 0xFF) - 127.5f) / 128.0f;
                input[0][y][x][1] = (((pixel >> 8) & 0xFF) - 127.5f) / 128.0f;
                input[0][y][x][2] = ((pixel & 0xFF) - 127.5f) / 128.0f;
            }
        }

        // Prepare output array
        float[][] output = new float[2][192];

        // Run inference
        tflite.run(input, output);  // THIS IS THE KEY STEP

        float[] embedding = output[0];

        //normalize embedding vector
        float sum = 0f;
        for (float v : embedding) {
            sum += v * v;
        }
        float norm = (float) Math.sqrt(sum);
        for (int i = 0; i < embedding.length; i++) {
            embedding[i] /= norm;
        }
        return embedding;
    }

    private boolean compareFaces(float[] embeddingNew) {
        float[] embeddingNewNorm = l2Normalize(embeddingNew);

        double minDistance = Double.MAX_VALUE;

        for (float[] known : patientEmbeddings) {
            float[] knownNorm = l2Normalize(known);
            double distance = 0;
            for (int i = 0; i < known.length; i++) {
                double diff = knownNorm[i] - embeddingNewNorm[i];
                distance += diff * diff;
            }
            distance = Math.sqrt(distance);
            if (distance < minDistance) minDistance = distance;
        }
        Log.i("TFLiteDebug", "Min distance = " + minDistance);
        double threshold = 0.9; // Loosen threshold slightly if averaging multiple faces
        return minDistance < threshold;
    }

    private float[] l2Normalize(float[] embedding) {
        float sum = 0f;
        for (float v : embedding) sum += v * v;
        float norm = (float) Math.sqrt(sum);
        float[] normalized = new float[embedding.length];
        for (int i = 0; i < embedding.length; i++) normalized[i] = embedding[i] / norm;
        return normalized;
    }


    //Memory-map the model file in Assets
    private MappedByteBuffer loadModelFile() throws IOException {
        //Open the model using an input stream and memory map it to load
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("MobileFaceNet.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void performFaceDetection(Bitmap input) {
        InputImage image = null;
        try {
            image = InputImage.fromFilePath(this, destUri);
        } catch (IOException e) {
            e.printStackTrace();
        }
        if (image == null) return;
        Task<List<Face>> result =
                detector.process(image)
                        .addOnSuccessListener(
                                new OnSuccessListener<List<Face>>() {
                                    @Override
                                    public void onSuccess(List<Face> faces) {
                                        for (Face face : faces) {
                                            Rect bounds = face.getBoundingBox();
                                            embeddingNew = performFaceRecognition(bounds, input);
//                                            Log.i("TFLiteDebug", "EmbeddingNew: " + Arrays.toString(embeddingNew));
                                            boolean isPatient = compareFaces(embeddingNew);
                                            binding.output.setText(isPatient ? name : "NOT " + name);
                                        }
                                    }
                                })
                        .addOnFailureListener(
                                new OnFailureListener() {
                                    @Override
                                    public void onFailure(@NonNull Exception e) {
                                        // Task failed with an exception
                                        // ...
                                    }
                                });
    }

    public void performFaceDetectionFromVideo(Bitmap input) {
        InputImage image = InputImage.fromBitmap(input, 0);
        Task<List<Face>> result =
                detector.process(image)
                        .addOnSuccessListener(
                                new OnSuccessListener<List<Face>>() {
                                    @Override
                                    public void onSuccess(List<Face> faces) {
                                        for (Face face : faces) {
                                            Rect bounds = face.getBoundingBox();
                                            performFaceRecognitionFromVideo(bounds, input);
                                        }
                                    }
                                })
                        .addOnFailureListener(
                                new OnFailureListener() {
                                    @Override
                                    public void onFailure(@NonNull Exception e) {
                                        // Task failed with an exception
                                        // ...
                                    }
                                });
    }
    public float[] performFaceRecognition(Rect bound, Bitmap input) {
        if (bound.top < 0) {
            bound.top = 0;
        }
        if (bound.left < 0) {
            bound.left = 0;
        }
        if (bound.right > input.getWidth()) {
            bound.right = input.getWidth() - 1;
        }
        if (bound.bottom > input.getHeight()) {
            bound.bottom = input.getHeight() - 1;
        }
        Bitmap croppedFace = Bitmap.createBitmap(input, bound.left, bound.top, bound.width(), bound.height());
        binding.currentPhoto.setImageBitmap(croppedFace);
        binding.executePendingBindings();
        float[] embeddingsNew = doInference(croppedFace);
//        Log.i("TFLiteDebug", "patientEmbeddings : " + Arrays.toString(embeddingsNew));
        return embeddingsNew;
    }
    public void performFaceRecognitionFromVideo(Rect bound, Bitmap input) {
        if (bound.top < 0) {
            bound.top = 0;
        }
        if (bound.left < 0) {
            bound.left = 0;
        }
        if (bound.right > input.getWidth()) {
            bound.right = input.getWidth() - 1;
        }
        if (bound.bottom > input.getHeight()) {
            bound.bottom = input.getHeight() - 1;
        }
        Bitmap croppedFace = Bitmap.createBitmap(input, bound.left, bound.top, bound.width(), bound.height());
        float[] embedding = doInference(croppedFace);
//        Log.i("TFLiteDebug", "patientEmbeddings : " + Arrays.toString(embedding));
        patientEmbeddings.add(embedding);
    }


    private final ActivityResultLauncher<String> cameraRequestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(),
                    isGranted -> {
                        if (isGranted) {
                            openCamera();
                        }
                    });

    private final ActivityResultLauncher<String[]> videoRequestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(),
                            result -> {
                boolean cameraGranted = result.getOrDefault(Manifest.permission.CAMERA, false);
                boolean audioGranted = result.getOrDefault(Manifest.permission.RECORD_AUDIO, false);
                if (cameraGranted && audioGranted) {
                    binding.normalSection.setVisibility(GONE);
                    binding.videoSection.setVisibility(VISIBLE);
                } else {
                    Toast.makeText(this, "Camera and audio permissions are required", Toast.LENGTH_SHORT).show();
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
                                performFaceDetection(bitmap);
                            }
                        }
                    });

    private final ActivityResultLauncher<Intent> videoActivityResultLauncher =
            registerForActivityResult(new ActivityResultContracts.StartActivityForResult(),
                    result -> {
                if (result.getResultCode() == Activity.RESULT_OK) {
                    try {
                        extractFrames(videoUri);
                        setPatientPhoto(videoUri);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                } else {
                    Toast.makeText(this, "Camera and audio permissions are required", Toast.LENGTH_SHORT).show();
                }
            });

    private void bindPhoto(View view) {
        imageFile = new File(view.getContext().getCacheDir(), photoFile.getName()); // Get full path
        Glide
                .with(view.getContext())
                .load(imageFile)
                .into((ImageView) view); //data binding
    }

    //@Override
    public void takePhoto(View view) {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            cameraRequestPermissionLauncher.launch(Manifest.permission.CAMERA);
        } else {
            openCamera();
        }
    }

    private void takeVideo(View view) {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED || checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) { //if user did not select Allow Camera for always;
            videoRequestPermissionLauncher.launch(new String[]{Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO});
        } else {
            openVideoRecorder(view);
        }
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

    private void openVideoRecorder(View view) {
        startCameraWithOverlay();
        createVideoFile();
        try {
            videoUri = FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", videoFile);
            ActivityResultContracts.CaptureVideo takeVideo = new ActivityResultContracts.CaptureVideo();
            Intent takeVideoIntent = takeVideo.createIntent(this, videoUri);
            takeVideoIntent.putExtra(MediaStore.EXTRA_DURATION_LIMIT, 10); //10sec
            takeVideoIntent.putExtra(MediaStore.EXTRA_VIDEO_QUALITY, 1);
            if (takeVideoIntent.resolveActivity(this.getPackageManager()) != null) {
                videoActivityResultLauncher.launch(takeVideoIntent);
                binding.normalSection.setVisibility(VISIBLE);
                binding.videoSection.setVisibility(GONE);
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

    private void createVideoFile() {
        try {
            videoFile = File.createTempFile(UUID.randomUUID() + "_", ".mp4", getCacheDir());
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

    private void extractFrames(Uri videoUri) throws IOException {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        retriever.setDataSource(this, videoUri);

        long durationMs = Long.parseLong(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION));

        //Extract frame every quarter of a sec
        for (int i = 0; i < durationMs; i += 250) {
            Bitmap frame = retriever.getFrameAtTime(i * 1000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC); //microseconds
            if (frame != null) {
                Log.i("TFLiteDebug", "performFaceDetectionFromVideo " + i);
                performFaceDetectionFromVideo(frame);
            }
        }
        retriever.release();
    }

    private void setPatientPhoto(Uri videoUri) {
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        retriever.setDataSource(this, videoUri);
        Bitmap mainPhotoBmp = retriever.getFrameAtTime(0, MediaMetadataRetriever.OPTION_CLOSEST_SYNC);
        createImageFile();
        try (FileOutputStream out = new FileOutputStream(photoFile)) {
            mainPhotoBmp.compress(Bitmap.CompressFormat.PNG, 100, out);
            out.flush();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        destUri = FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", photoFile);
        bindPhoto(binding.patientPhoto);
        binding.name.setText(name);
    }

    private void startCameraWithOverlay() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture =
                ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                cameraProvider.unbindAll();

                Preview preview = new Preview.Builder().build();
                CameraSelector cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA;

                VideoCapture<Recorder> videoCapture =
                        new VideoCapture.Builder(
                                new Recorder.Builder()
                                        .setQualitySelector(QualitySelector.from(Quality.HD))
                                        .build()
                        ).build();

                preview.setSurfaceProvider(binding.previewView.getSurfaceProvider());

                cameraProvider.bindToLifecycle(
                        this,
                        cameraSelector,
                        preview,
                        videoCapture
                );

                // Show overlay on top of preview
                binding.overlayOval.bringToFront();
                binding.overlayOval.setZ(100f);

            } catch (ExecutionException | InterruptedException e) {
                e.printStackTrace();
            }
        }, ContextCompat.getMainExecutor(this));
    }



}

