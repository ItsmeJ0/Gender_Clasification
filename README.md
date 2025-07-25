# 👩‍🦰👨‍🦱 Gender Classification using CNN (AlexNet, GoogleNet, ResNet, VGG)

Proyek ini merupakan tugas Gender Classification untuk mengklasifikasikan gambar wajah menjadi Male atau Female menggunakan arsitektur CNN populer:
- AlexNet
- GoogleNet (Inception)
- ResNet
- VGG

----------------------------------------------------------
## ✅ Tujuan
- Implementasi dan perbandingan performa model CNN untuk klasifikasi gender.
- Mengukur akurasi, loss, dan waktu pelatihan setiap model.

----------------------------------------------------------
## 📂 Dataset
Dataset yang digunakan:
- CelebA HQ Face Gender Dataset
- Struktur folder:
dataset/
├── train/
│   ├── male/
│   └── female/
└── test/
    ├── male/
    └── female/

----------------------------------------------------------
## 🛠 Teknologi
- Python 3.x
- PyTorch atau TensorFlow/Keras
- CUDA (opsional) untuk GPU
- Matplotlib, NumPy, Pandas
- scikit-learn untuk evaluasi

----------------------------------------------------------
## ⚙️ Cara Menjalankan

1. Clone Repository:
git clone https://github.com/username/gender-classification-cnn.git
cd gender-classification-cnn

2. Install Dependencies:
pip install -r requirements.txt

3. Pastikan Struktur Folder:
project/
├── models/           # alexnet.py, googlenet.py, resnet.py, vgg.py
├── dataset/          # folder train & test
├── train.py          # script training
├── evaluate.py       # script evaluasi
└── README.md

4. Jalankan Training Model:
# Training dengan AlexNet
python train.py --model alexnet --epochs 20

# Training dengan VGG
python train.py --model vgg16 --epochs 20

# Training dengan ResNet
python train.py --model resnet50 --epochs 20

# Training dengan GoogleNet
python train.py --model googlenet --epochs 20

----------------------------------------------------------
## 📌 Hyperparameter
- Batch size: 32
- Learning rate: 0.001
- Optimizer: Adam
- Loss Function: CrossEntropyLoss

----------------------------------------------------------
## 📊 Evaluasi Model
Metrik:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Contoh Hasil:
AlexNet   : 92.1% accuracy
GoogleNet : 94.5% accuracy
ResNet    : 95.8% accuracy
VGG16     : 93.4% accuracy

----------------------------------------------------------
## ✅ Contoh Prediksi
# Jalankan prediksi untuk satu gambar
python predict.py --model resnet50 --image sample.jpg

