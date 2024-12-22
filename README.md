<div align="center">

# Implementasi Algoritma Machine Learning
## Tugas Besar 2 IF3070 Dasar Inteligensi Artifisial

</div>

## 📋 Ringkasan Proyek

Proyek ini merupakan implementasi komprehensif dari algoritma machine learning untuk klasifikasi URL phishing. Kami mengembangkan dua model utama - K-Nearest Neighbors (KNN) dan Gaussian Naive-Bayes - dengan pendekatan implementasi dari awal (from scratch) dan membandingkannya dengan implementasi menggunakan pustaka scikit-learn.

## 🔍 Fitur-Fitur Kunci

### 1. Implementasi Algoritma Custom
- **K-Nearest Neighbors (KNN)**
  - Mendukung berbagai metrik jarak:
    - Euclidean Distance
    - Manhattan Distance
    - Chebyshev Distance
    - Minkowski Distance
- **Gaussian Naive-Bayes**
  - Implementasi perhitungan probabilitas berbasis distribusi Gaussian
  - Optimasi untuk performa klasifikasi

### 2. Integrasi dengan Scikit-learn
- Implementasi paralel menggunakan:
  - `KNeighborsClassifier` untuk KNN
  - `GaussianNB` untuk Naive-Bayes
- Perbandingan performa antara implementasi custom dan scikit-learn

### 3. Manajemen Model
- Penyimpanan model terlatih dalam format `.pkl`
- Sistem loading model yang efisien menggunakan `ModelLoader`
- Kemudahan dalam deployment dan penggunaan ulang model

### 4. Preprocessing Data
- Pembersihan data komprehensif:
  - Penanganan nilai yang hilang
  - Eliminasi data duplikat
  - Encoding fitur kategorikal
  - Normalisasi dan scaling fitur
  - Feature engineering untuk optimasi performa

### 5. Sistem Evaluasi
Evaluasi performa model menggunakan berbagai metrik:
- Accuracy
- Precision
- Recall
- F1-Score

## 🚀 Panduan Penggunaan

### Persiapan Awal
```bash
# Clone repository
git clone https://github.com/IlllIllIIIlIlIll/Traienta_IF3070.git

# Pindah ke direktori proyek
cd Traienta_IF3070/src/public
```

### Menjalankan Program
1. Buka Jupyter Notebook di direktori proyek
2. Akses file `Tubes2_Kelompok30.ipynb`
3. Jalankan seluruh sel dalam notebook secara berurutan

## 👥 Tim Pengembang - BebanKaggle (Kelompok 21)

|   NIM    |            Nama                |                                      Kontribusi                                       |
|----------|--------------------------------|---------------------------------------------------------------------------------------|
| 18222035 | Lydia Gracia                   | Pipeline Development, Data Processing, Model Implementation, Documentation            |
| 18222049 | Willhelmina Rachel Silalahi    | Machine Learning Pipeline, Data Analysis, Model Development, Technical Documentation  |
| 18222070 | Favian Izza Diasputra          | Data Engineering, Model Architecture, Pipeline Integration, Documentation             |
| 18222100 | Ervina Limka                   | Feature Engineering, Model Optimization, Pipeline Development, Documentation Support  |