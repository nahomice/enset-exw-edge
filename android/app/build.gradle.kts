plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "et.edu.aau.exw"
    compileSdk = 34

    defaultConfig {
        applicationId = "et.edu.aau.exw"
        minSdk = 26                       // Android 12 target device (Galaxy A13) is far above this
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
    }
    buildTypes {
        release { isMinifyEnabled = false }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += "tflite" }   // the model must stay uncompressed to be memory-mapped
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    testImplementation("junit:junit:4.13.2")
}
