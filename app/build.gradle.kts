plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.example.mlapp"
    compileSdk = 36

    buildFeatures{
        dataBinding = true
        buildConfig = true

    }

    defaultConfig {
        applicationId = "com.example.mlapp"
        minSdk = 34
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    androidResources {
        noCompress += "tflite"
    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
    implementation(libs.glide)
    implementation(libs.tensorflow.lite)
    implementation(libs.face.detection)
    implementation(libs.play.services.mlkit.text.recognition)
}