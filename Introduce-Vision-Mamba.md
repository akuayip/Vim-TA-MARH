# 🧠 Pengenalan Vision Mamba (ViM) - Dijelaskan dengan Mudah

> **Belajar tentang Vision Mamba seperti kamu menjelaskan kepada teman SMP!** 🎓

---

## 📚 Daftar Isi

1. [Apa itu Vision Mamba?](#1-apa-itu-vision-mamba)
2. [Mengapa Vision Mamba Penting?](#2-mengapa-vision-mamba-penting)
3. [Arsitektur Vision Mamba](#3-arsitektur-vision-mamba)
4. [Cara Kerja Vision Mamba](#4-cara-kerja-vision-mamba)
5. [Vision Mamba untuk Object Detection](#5-vision-mamba-untuk-object-detection)
6. [Perbandingan dengan Model Lain](#6-perbandingan-dengan-model-lain)

---

## 1. Apa itu Vision Mamba?

### 🎯 Bayangkan Seperti Ini...

**Analogi Sederhana:**

Bayangkan kamu punya **buku komik** yang tebal. Untuk memahami ceritanya:

- **Cara Lama (Transformer/ViT)**: Kamu harus **membaca semua halaman sekaligus**, membandingkan setiap halaman dengan halaman lainnya. Ini seperti mengingat semua detail di kepala sekaligus - **capek dan lambat**! 😵

- **Cara Vision Mamba**: Kamu **membaca dari depan ke belakang** dan **dari belakang ke depan** secara bersamaan, sambil mengingat hal-hal penting saja. Seperti punya **2 teman** yang membaca dari arah berbeda dan saling berbagi informasi penting. **Lebih cepat dan hemat energi**! ⚡

### 🔬 Definisi Teknis

**Vision Mamba (ViM)** adalah arsitektur deep learning untuk **computer vision** yang menggunakan:

- **Bidirectional State Space Models (SSM)** sebagai pengganti self-attention
- **Mamba blocks** untuk memproses urutan visual secara efisien
- **Position embeddings** untuk memahami lokasi objek dalam gambar

**Paper**: [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417) - ICML 2024

---

## 2. Mengapa Vision Mamba Penting?

### 🚀 Keunggulan Dibanding Model Tradisional

#### **Problem dengan Vision Transformer (ViT)**

Vision Transformer yang populer memiliki masalah:

| Problem                       | Penjelasan                                                              | Dampak                              |
| ----------------------------- | ----------------------------------------------------------------------- | ----------------------------------- |
| **Komputasi Mahal**           | Self-attention harus membandingkan setiap patch dengan semua patch lain | $O(N^2)$ kompleksitas               |
| **Memory Boros**              | Harus menyimpan attention matrix yang besar                             | Butuh GPU kuat                      |
| **Lambat untuk Gambar Besar** | Semakin besar gambar, semakin lambat                                    | Tidak efisien untuk high-resolution |

**Contoh Konkrit:**

```
Gambar 224x224 (ViT):
- Jumlah patches: 14 x 14 = 196
- Attention comparisons: 196 x 196 = 38,416 operasi

Gambar 1248x1248 (ViT):
- Jumlah patches: 78 x 78 = 6,084
- Attention comparisons: 6,084 x 6,084 = 37,014,656 operasi! 😱

Vision Mamba:
- Kompleksitas: Linear O(N)
- Jauh lebih cepat dan hemat memory!
```

#### **Solusi Vision Mamba**

✅ **2.8x lebih cepat** dari Vision Transformer (DeiT)
✅ **86.8% hemat GPU memory** untuk gambar resolusi tinggi
✅ **Akurasi sama atau lebih baik** di ImageNet, COCO, ADE20K
✅ **Scalable** untuk gambar beresolusi tinggi

---

## 3. Arsitektur Vision Mamba

### 🏗️ Komponen Utama

Vision Mamba terdiri dari **5 komponen utama**:

```
Input Gambar
    ↓
┌───────────────────────────────────────┐
│  1. PATCH EMBEDDING                   │
│  Ubah gambar jadi potongan-potongan   │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  2. POSITION EMBEDDING                │
│  Kasih "alamat" setiap potongan       │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  3. CLASS TOKEN (opsional)            │
│  Token khusus untuk klasifikasi       │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  4. MAMBA BLOCKS (24 layer)           │
│  Proses bidirectional dengan SSM      │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│  5. OUTPUT HEAD                       │
│  Klasifikasi atau feature extraction  │
└───────────────────────────────────────┘
    ↓
Output (Class / Features)
```

---

### 📐 Komponen 1: Patch Embedding

**Apa itu?**
Mengubah gambar besar menjadi **potongan-potongan kecil** (patches).

**Analogi:**
Bayangkan kamu punya **puzzle besar**. Kamu potong jadi kotak-kotak kecil ukuran 16×16 pixel.

**Cara Kerja:**

```python
# Input: Gambar RGB 224x224x3
# Output: 196 patches, masing-masing 768 dimensi

Gambar (224, 224, 3)
    ↓
Potong jadi patches 16×16
    ↓
Dapat 14×14 = 196 patches
    ↓
Setiap patch di-flatten & project
    ↓
Output: (196, 768)
```

**Visualisasi:**

```
Gambar Asli (224×224):
┌─────────────────────────────┐
│ [Gambar Kucing]             │
│                             │
│                             │
└─────────────────────────────┘

Setelah Patch Embedding (14×14 patches):
┌───┬───┬───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │...│13 │14 │
├───┼───┼───┼───┼───┼───┼───┤
│15 │16 │17 │18 │...│27 │28 │
├───┼───┼───┼───┼───┼───┼───┤
│...│...│...│...│...│...│...│
└───┴───┴───┴───┴───┴───┴───┘
Total: 196 patches
```

**Code Implementasi:**

```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=192):
        super().__init__()
        # Jumlah patches: (224/16) × (224/16) = 14 × 14 = 196
        self.num_patches = (img_size // patch_size) ** 2

        # Convolution untuk potong & project patches
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (Batch, 3, 224, 224)
        x = self.proj(x)  # → (Batch, 192, 14, 14)
        x = x.flatten(2).transpose(1, 2)  # → (Batch, 196, 192)
        return x
```

---

### 📍 Komponen 2: Position Embedding

**Apa itu?**
Memberi **"alamat"** atau **koordinat** untuk setiap patch, agar model tahu **posisi** mereka.

**Mengapa Penting?**
Tanpa position embedding, model tidak tahu apakah suatu patch ada di **pojok kiri atas** atau **tengah gambar**.

**Analogi:**
Seperti memberi **nomor rumah** di perumahan. Tanpa nomor, kamu tidak tahu rumah mana yang mana!

**Cara Kerja:**

```python
# 1. Buat learnable position embeddings
pos_embed = nn.Parameter(torch.zeros(1, 196, 192))

# 2. Tambahkan ke patch embeddings
x = patch_embed + pos_embed

# Sekarang setiap patch punya informasi:
# - Konten visual (dari patch embedding)
# - Posisi spatial (dari position embedding)
```

**Visualisasi:**

```
Tanpa Position Embedding:
Patch 1: [0.5, 0.2, 0.8, ...] → Gambar apa? Di mana? 🤷
Patch 2: [0.1, 0.9, 0.3, ...] → Gambar apa? Di mana? 🤷

Dengan Position Embedding:
Patch 1: [0.5, 0.2, 0.8] + [Pos: Kiri-Atas] → Mata kucing di kiri! 😺
Patch 2: [0.1, 0.9, 0.3] + [Pos: Tengah] → Hidung kucing! 😸
```

---

### 🎯 Komponen 3: Class Token

**Apa itu?**
**Token khusus** yang ditambahkan untuk **merangkum informasi** seluruh gambar.

**Analogi:**
Seperti **ketua kelas** yang mengumpulkan informasi dari semua siswa, lalu melaporkan kesimpulannya ke guru.

**Vision Mamba menggunakan:**

- **Middle Class Token**: Token ditempatkan di **tengah sequence** (bukan di awal/akhir)
- Ini membantu **bidirectional processing** lebih efektif

**Cara Kerja:**

```python
# Buat class token
cls_token = nn.Parameter(torch.zeros(1, 1, 192))

# Expand untuk batch
cls_token = cls_token.expand(batch_size, -1, -1)

# Insert di tengah sequence
# Original: [P1, P2, P3, ..., P196]
# Dengan CLS: [P1, ..., P98, [CLS], P99, ..., P196]

mid_position = num_patches // 2
x = torch.cat([x[:, :mid_position], cls_token, x[:, mid_position:]], dim=1)
```

---

### 🧩 Komponen 4: Mamba Blocks (INTI!)

**Ini adalah JANTUNG dari Vision Mamba!** 💓

#### **Apa itu Mamba Block?**

Mamba Block menggunakan **Selective State Space Model (SSM)** - sebuah cara untuk:

1. **Memproses data secara sequential** (berurutan)
2. **Mengingat informasi penting** sambil **melupakan yang tidak penting**
3. **Efisien** dalam komputasi dan memory

---

### 🔄 Cara Kerja State Space Model (SSM)

#### **Analogi: Aliran Sungai**

Bayangkan SSM seperti **aliran sungai**:

```
Hulu (Input) → [Filter Air] → [Tampung di Wadah] → Hilir (Output)
    ↓              ↓                ↓                    ↓
  Patch 1      Pilih info       Simpan state        Feature 1
  Patch 2      penting         Update state        Feature 2
  Patch 3      Filter         Remember/Forget      Feature 3
```

**Komponen SSM:**

1. **State (h)**: "Ingatan" model - apa yang dia simpan dari sebelumnya
2. **Input (x)**: Data baru yang masuk (patch saat ini)
3. **Selection Mechanism**: Memilih info mana yang penting untuk diingat
4. **Update Rule**: Cara update state berdasarkan input baru

#### **Formula Matematika (Disederhanakan)**

```python
# State Space Model Formula

# 1. Update state (ingatan)
h_t = A * h_{t-1} + B * x_t

Di mana:
- h_t = state baru (ingatan sekarang)
- h_{t-1} = state lama (ingatan sebelumnya)
- x_t = input baru (patch sekarang)
- A = matrix untuk "lupa" sebagian info lama
- B = matrix untuk "ingat" info baru

# 2. Hitung output
y_t = C * h_t + D * x_t

Di mana:
- y_t = output (feature yang dihasilkan)
- C = matrix untuk extract info dari state
- D = skip connection (koneksi langsung input→output)
```

**Dalam bahasa sederhana:**

> "Ingatan baru = (Sebagian ingatan lama yang penting) + (Info baru yang masuk)"

---

### 🎭 Bidirectional Processing

**Apa itu?**

Vision Mamba memproses gambar dari **2 arah sekaligus**:

1. **Forward**: Kiri → Kanan, Atas → Bawah
2. **Backward**: Kanan → Kiri, Bawah → Atas

**Mengapa Bidirectional?**

Karena dalam gambar, **konteks dari semua arah penting**!

**Analogi:**

Bayangkan kamu cari orang hilang di kerumunan:

- **Unidirectional**: Cuma lihat dari kiri ke kanan → Bisa ketinggalan orang di kanan
- **Bidirectional**: Lihat dari kiri ke kanan DAN kanan ke kiri → Lebih lengkap!

**Visualisasi:**

```
Gambar Patches (4×4 untuk contoh):
┌────┬────┬────┬────┐
│ P1 │ P2 │ P3 │ P4 │
├────┼────┼────┼────┤
│ P5 │ P6 │ P7 │ P8 │
├────┼────┼────┼────┤
│ P9 │P10 │P11 │P12 │
├────┼────┼────┼────┤
│P13 │P14 │P15 │P16 │
└────┴────┴────┴────┘

Forward Pass (→):
P1 → P2 → P3 → P4 → P5 → P6 → ... → P16

Backward Pass (←):
P16 → P15 → P14 → P13 → ... → P2 → P1

Hasil Gabungan:
Setiap patch mendapat informasi dari KEDUA arah!
```

**Code Implementasi:**

```python
class BiMambaBlock(nn.Module):
    def forward(self, x):
        # x shape: (Batch, Num_patches, Embed_dim)

        # Forward pass
        forward_out = self.mamba_forward(x)

        # Backward pass (flip sequence)
        x_reverse = torch.flip(x, dims=[1])
        backward_out = self.mamba_backward(x_reverse)
        backward_out = torch.flip(backward_out, dims=[1])

        # Gabungkan hasil forward & backward
        out = (forward_out + backward_out) / 2

        return out
```

---

### 🔁 Struktur 1 Mamba Block

Setiap Mamba Block punya struktur:

```
Input
  ↓
┌─────────────────────────┐
│  Layer Normalization    │  ← Normalisasi input
└─────────────────────────┘
  ↓
┌─────────────────────────┐
│  Selective SSM          │  ← Proses dengan State Space
│  (Bidirectional)        │
└─────────────────────────┘
  ↓
┌─────────────────────────┐
│  Residual Connection    │  ← Tambahkan input asli
└─────────────────────────┘
  ↓
Output
```

**Code:**

```python
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(dim, d_state=d_state)

    def forward(self, x, residual=None):
        # Residual connection
        if residual is None:
            residual = x
        else:
            residual = residual + x

        # Normalize
        x = self.norm(residual)

        # Apply Mamba (SSM)
        x = self.mamba(x)

        return x, residual
```

---

### 📚 Stacking Multiple Blocks

Vision Mamba menggunakan **24 Mamba Blocks** yang di-stack (ditumpuk):

```
Input Patches (196, 192)
    ↓
Block 1  → Belajar fitur sederhana (edges, textures)
    ↓
Block 2  → Belajar fitur lebih kompleks
    ↓
Block 3  → ...
    ↓
  ...
    ↓
Block 24 → Fitur high-level (object parts, semantics)
    ↓
Output Features (196, 192)
```

**Semakin dalam, semakin abstrak:**

- **Layer awal** (1-8): Deteksi **garis, sudut, warna**
- **Layer tengah** (9-16): Deteksi **pola, tekstur, bentuk sederhana**
- **Layer dalam** (17-24): Deteksi **objek, bagian tubuh, konsep abstrak**

---

### 🎓 Komponen 5: Output Head

**Untuk Klasifikasi (ImageNet):**

```python
# Ambil class token atau global pooling
cls_features = x[:, mid_position]  # Class token di tengah

# Linear layer untuk klasifikasi
output = self.head(cls_features)  # (Batch, 1000 classes)
```

**Untuk Object Detection (ViMDet):**

```python
# Ekstrak features dari semua patches
features = self.norm(x)  # (Batch, 196, 192)

# Reshape ke spatial format
features = features.reshape(B, 14, 14, 192)
features = features.permute(0, 3, 1, 2)  # (B, 192, 14, 14)

# Kirim ke detection head (FPN, RPN, ROI Head)
detections = detection_head(features)
```

---

## 4. Cara Kerja Vision Mamba

### 🔄 Alur Lengkap Step-by-Step

Mari kita ikuti **1 gambar** dari awal sampai akhir!

**Input: Gambar kucing 224×224×3**

---

#### **Step 1: Patch Embedding**

```
Gambar Kucing (224, 224, 3)
    ↓
[Potong jadi 16×16 patches]
    ↓
14×14 = 196 patches
    ↓
[Project dengan Conv2D]
    ↓
Output: (196, 192)

Contoh:
Patch 1: [Telinga kiri] → [0.5, 0.2, 0.8, ..., 192 nilai]
Patch 2: [Mata kiri] → [0.1, 0.9, 0.3, ..., 192 nilai]
...
Patch 196: [Ekor] → [0.7, 0.4, 0.1, ..., 192 nilai]
```

---

#### **Step 2: Add Position Embedding**

```
Patch Embeddings (196, 192)
    +
Position Embeddings (196, 192)
    ↓
Patches with Position (196, 192)

Sekarang setiap patch tahu:
- Apa isinya (visual content)
- Di mana lokasinya (spatial position)

Contoh:
Patch 1: [Telinga kiri] + [Posisi: Kiri-Atas]
Patch 98: [Hidung] + [Posisi: Tengah]
Patch 196: [Ekor] + [Posisi: Kanan-Bawah]
```

---

#### **Step 3: Insert Class Token**

```
Original: [P1, P2, ..., P196]
    ↓
Insert CLS di tengah (posisi 98)
    ↓
[P1, P2, ..., P97, [CLS], P98, ..., P196]
    ↓
Total: 197 tokens
```

---

#### **Step 4: Process dengan Mamba Blocks**

**Mamba Block 1 (Shallow Layer):**

```python
# Input: (197, 192)

# Forward pass (→)
state_fwd = [0, 0, 0, ..., 0]  # Initial state

for token in [P1, P2, ..., CLS, ..., P196]:
    # Update state
    state_fwd = A * state_fwd + B * token

    # Generate output
    out_fwd = C * state_fwd + D * token

# Backward pass (←)
state_bwd = [0, 0, 0, ..., 0]

for token in reversed([P1, P2, ..., CLS, ..., P196]):
    state_bwd = A * state_bwd + B * token
    out_bwd = C * state_bwd + D * token

# Combine
output_block1 = (out_fwd + out_bwd) / 2
```

**Yang Dipelajari Block 1:**

- Edges (tepi mata, telinga)
- Basic textures (bulu halus)
- Simple colors

**Mamba Block 2-8:**

- Menggabungkan edges jadi shapes
- Deteksi pola bulu kucing
- Deteksi kontur tubuh

**Mamba Block 9-16:**

- Deteksi bagian tubuh (mata, hidung, telinga)
- Pola keseluruhan
- Relasi antar bagian

**Mamba Block 17-24:**

- Semantic understanding: "Ini KUCING"
- High-level features
- Object-level representation

---

#### **Step 5: Output**

**Untuk Klasifikasi:**

```python
# Ambil class token (di posisi tengah)
cls_feature = output[:, 98, :]  # Shape: (1, 192)

# Normalisasi
cls_feature = LayerNorm(cls_feature)

# Klasifikasi
logits = Linear(cls_feature)  # (1, 1000)

# Prediksi
predicted_class = argmax(logits)
# Output: "Cat" (confidence: 98.5%)
```

---

### 🧮 Kompleksitas Komputasi

**Vision Transformer (ViT):**

```
Self-Attention: O(N²)
N = jumlah patches = 196
Operasi: 196 × 196 = 38,416 comparisons per layer

Total (12 layers): 38,416 × 12 = 460,992 operasi
```

**Vision Mamba:**

```
Selective SSM: O(N)
N = 196 patches
Operasi: 196 operasi per layer

Total (24 layers): 196 × 24 = 4,704 operasi

Speedup: 460,992 / 4,704 = 98x lebih efisien! 🚀
```

---

## 5. Vision Mamba untuk Object Detection

### 🎯 Dari Klasifikasi ke Detection

**Object Detection = Klasifikasi + Lokalisasi**

Kita perlu:

1. **Deteksi objek APA** (klasifikasi) → "Kucing"
2. **Deteksi objek DI MANA** (lokalisasi) → Bounding box: (x, y, w, h)

---

### 🏗️ Arsitektur ViMDet (Vision Mamba Detector)

```
Input Image (800×1333)
    ↓
┌──────────────────────────────────┐
│  VISION MAMBA BACKBONE           │
│  - Patch Embedding               │
│  - 24 Mamba Blocks               │
│  - Extract Multi-scale Features  │
└──────────────────────────────────┘
    ↓
Multi-scale Feature Maps
    ↓
┌──────────────────────────────────┐
│  FPN (Feature Pyramid Network)   │
│  - Fuse features dari berbagai   │
│    level untuk deteksi multi-    │
│    scale objects                 │
└──────────────────────────────────┘
    ↓
Pyramid Features (P2, P3, P4, P5)
    ↓
┌──────────────────────────────────┐
│  RPN (Region Proposal Network)   │
│  - Propose candidate boxes       │
└──────────────────────────────────┘
    ↓
~1000 Proposals
    ↓
┌──────────────────────────────────┐
│  ROI Head (Cascade R-CNN)        │
│  - Refine boxes (3 stages)       │
│  - Classify objects              │
│  - Generate masks (optional)     │
└──────────────────────────────────┘
    ↓
Final Detections
- Bounding boxes
- Class labels
- Confidence scores
- Masks (for instance segmentation)
```

---

### 🔍 Penjelasan Setiap Komponen Detection

#### **1. Vision Mamba Backbone**

**Fungsi:** Extract **visual features** dari gambar input.

**Proses:**

```python
# Input: Gambar (1, 3, 800, 1333)

# Patch embedding dengan stride berbeda untuk multi-scale
patches = PatchEmbed(img, patch_size=16, stride=16)
# Output: (B, N_patches, 192)

# Process dengan 24 Mamba blocks
for block in mamba_blocks:
    patches = block(patches)

# Extract features di berbagai level
features = {
    'level1': patches_from_block_6,   # High resolution, low semantic
    'level2': patches_from_block_12,  # Medium resolution
    'level3': patches_from_block_18,  # Low resolution, high semantic
    'level4': patches_from_block_24,  # Lowest resolution, highest semantic
}
```

**Output:** Multi-scale feature maps

---

#### **2. FPN (Feature Pyramid Network)**

**Fungsi:** Menggabungkan features dari berbagai level untuk **deteksi objek berbagai ukuran**.

**Mengapa Penting?**

```
Objek Kecil (mouse):
- Perlu HIGH resolution features (banyak detail spatial)
- Dari layer awal/tengah

Objek Besar (gajah):
- Perlu LOW resolution tapi HIGH semantic features
- Dari layer dalam

FPN → Gabungkan keduanya!
```

**Cara Kerja:**

```
Top-down pathway:
┌──────────┐
│ P5 (8×8) │ ← Semantic tinggi, resolution rendah
└──────────┘
    ↑ upsample
    │
┌──────────┐
│ P4(16×16)│ ← + lateral connection dari C4
└──────────┘
    ↑ upsample
    │
┌──────────┐
│ P3(32×32)│ ← + lateral connection dari C3
└──────────┘
    ↑ upsample
    │
┌──────────┐
│ P2(64×64)│ ← + lateral connection dari C2
└──────────┘

Result: Multi-scale features yang punya:
- Resolution tinggi (P2) → Deteksi objek kecil
- Semantic tinggi (P5) → Deteksi objek besar
```

---

#### **3. RPN (Region Proposal Network)**

**Fungsi:** **Mengusulkan kandidat** lokasi yang mungkin ada objek.

**Analogi:**
Seperti **pemindai cepat** yang bilang: "Eh, kayaknya di sini ada objek deh!"

**Cara Kerja:**

```python
# Untuk setiap lokasi di feature map
for location in feature_map:
    # Generate anchor boxes (kotak referensi) berbagai ukuran
    anchors = [
        (32×32), (64×64), (128×128),  # Ukuran
        ratio: [0.5, 1.0, 2.0]         # Aspect ratio
    ]

    # Untuk setiap anchor, prediksi:
    for anchor in anchors:
        objectness_score = sigmoid(...)  # Ada objek? (0-1)
        box_offset = regression(...)     # Koreksi posisi anchor

        if objectness_score > 0.7:
            proposals.add(anchor + box_offset)

# Hasil: ~1000-2000 proposals
```

**Visualisasi:**

```
Feature Map:
┌───────────────────────────┐
│   [Gambar dengan Kucing]  │
│                           │
│   🔴 🔴 🔴 ← Proposals   │
│   🔴 😺 🔴    (kucing)    │
│   🔴 🔴 🔴               │
│                           │
│   🔵 🔵 ← Proposals       │
│   🔵 🔵   (background)    │
└───────────────────────────┘

Setelah filtering:
- Keep: Boxes dengan objectness > threshold
- NMS: Hapus duplikasi
Result: ~1000 high-quality proposals
```

---

#### **4. ROI Head (Cascade R-CNN)**

**Fungsi:** **Memperhalus** proposals dan **mengklasifikasi** objek.

**Cascade = 3 Tahap Refinement:**

```
Stage 1 (IoU threshold = 0.5):
Input: 1000 proposals
↓
[ROI Pooling] → Extract features untuk setiap proposal
↓
[Classification Head] → Prediksi class
[Regression Head] → Refine bounding box
↓
Output: Refined 500 proposals

Stage 2 (IoU threshold = 0.6):
Input: 500 proposals dari stage 1
↓
[ROI Pooling] dengan threshold lebih ketat
↓
[Classification + Regression]
↓
Output: Refined 200 proposals

Stage 3 (IoU threshold = 0.7):
Input: 200 proposals dari stage 2
↓
[ROI Pooling] dengan threshold SANGAT ketat
↓
[Classification + Regression]
↓
Final Output: ~100 high-quality detections
```

**Code Contoh:**

```python
class CascadeROIHead(nn.Module):
    def __init__(self):
        self.box_head_1 = FastRCNNConvFCHead(...)
        self.box_head_2 = FastRCNNConvFCHead(...)
        self.box_head_3 = FastRCNNConvFCHead(...)

        self.box_predictor_1 = FastRCNNOutputLayers(...)
        self.box_predictor_2 = FastRCNNOutputLayers(...)
        self.box_predictor_3 = FastRCNNOutputLayers(...)

    def forward(self, features, proposals):
        # Stage 1
        box_features_1 = self.box_head_1(features, proposals)
        pred_class_1, pred_boxes_1 = self.box_predictor_1(box_features_1)
        proposals = self.refine_proposals(proposals, pred_boxes_1)

        # Stage 2
        box_features_2 = self.box_head_2(features, proposals)
        pred_class_2, pred_boxes_2 = self.box_predictor_2(box_features_2)
        proposals = self.refine_proposals(proposals, pred_boxes_2)

        # Stage 3
        box_features_3 = self.box_head_3(features, proposals)
        pred_class_3, pred_boxes_3 = self.box_predictor_3(box_features_3)

        return pred_class_3, pred_boxes_3
```

---

### 🎬 Contoh Konkrit: Deteksi Kucing & Anjing

**Input: Gambar dengan 1 kucing & 1 anjing**

```
1. Vision Mamba Backbone:
   - Extract features
   - Output: Multi-scale feature maps

2. FPN:
   - P2: Resolution tinggi (64×64) → Deteksi detail kecil
   - P3: Medium (32×32)
   - P4: Medium-low (16×16)
   - P5: Low (8×8) → Deteksi semantic

3. RPN:
   - Scan semua lokasi di P2-P5
   - Generate ~1500 proposals
   - Contoh proposals:
     * Box 1: (100, 150, 200, 200) - objectness: 0.95 → Kucing!
     * Box 2: (400, 180, 250, 220) - objectness: 0.92 → Anjing!
     * Box 3: (50, 50, 80, 80) - objectness: 0.15 → Background
     * ...

4. ROI Head (Cascade):

   Stage 1:
   - Input: 1500 proposals
   - Classification:
     * Box 1: Cat (85%), Dog (10%), Background (5%)
     * Box 2: Dog (88%), Cat (8%), Background (4%)
   - Refine boxes:
     * Box 1: (102, 152, 198, 198) ← Lebih presisi!
     * Box 2: (398, 182, 248, 218)

   Stage 2:
   - Input: Top 500 proposals
   - Classification (lebih akurat):
     * Box 1: Cat (92%), Dog (5%), Background (3%)
     * Box 2: Dog (94%), Cat (4%), Background (2%)
   - Refine boxes lagi:
     * Box 1: (103, 153, 197, 197)
     * Box 2: (399, 183, 247, 217)

   Stage 3:
   - Input: Top 200 proposals
   - Final classification:
     * Box 1: Cat (96.5%) ✓
     * Box 2: Dog (97.2%) ✓
   - Final boxes:
     * Box 1: (103, 154, 196, 196)
     * Box 2: (400, 184, 246, 216)

5. Output:
   Detection 1: {
     class: "Cat",
     confidence: 0.965,
     bbox: [103, 154, 196, 196]
   }

   Detection 2: {
     class: "Dog",
     confidence: 0.972,
     bbox: [400, 184, 246, 216]
   }
```

---

### 📊 Input Resolution untuk Object Detection

**ViMDet menggunakan Multi-scale Training:**

```python
# Training
INPUT_SIZE_TRAIN = [640, 672, 704, 736, 768, 800]  # Short side
MAX_SIZE_TRAIN = 1333  # Long side

# Test/Inference
INPUT_SIZE_TEST = 800
MAX_SIZE_TEST = 1333

# Contoh:
Gambar 1920×1080:
→ Resize short side ke 800
→ Long side = 800 * (1920/1080) = 1422
→ Clip ke max 1333
→ Final: 1107×800 (keep aspect ratio)
```

**Mengapa Multi-scale?**

```
Objek Kecil (mouse, bird):
- Butuh resolusi tinggi (800+)
- Supaya tidak hilang detail

Objek Sedang (cat, dog):
- OK dengan 640-800
- Balance antara speed & accuracy

Objek Besar (car, person):
- OK dengan resolusi lebih rendah (640)
- Lebih cepat, tetap akurat

Multi-scale training → Model belajar handle semua ukuran!
```

---

## 6. Perbandingan dengan Model Lain

### 📈 Vision Mamba vs Vision Transformer (ViT)

| Aspek                   | Vision Transformer (ViT) | Vision Mamba (ViM)             |
| ----------------------- | ------------------------ | ------------------------------ |
| **Mechanism**           | Self-Attention           | Selective State Space Model    |
| **Kompleksitas**        | O(N²)                    | O(N)                           |
| **Memory**              | Tinggi                   | Rendah (86.8% lebih hemat)     |
| **Speed**               | Lambat untuk high-res    | 2.8x lebih cepat               |
| **Accuracy (ImageNet)** | DeiT-S: 79.8%            | ViM-S: 80.5%                   |
| **COCO Detection (AP)** | ViT-Det-B: 51.6          | ViMDet-B: ~48-50 (competitive) |
| **Best For**            | Standard resolution      | High-resolution images         |

---

### 🏆 Keunggulan Vision Mamba

✅ **Efisiensi Komputasi**

- Linear complexity O(N) vs quadratic O(N²)
- Lebih cepat untuk inference

✅ **Memory Efficiency**

- Hemat 86.8% GPU memory untuk gambar 1248×1248
- Bisa process gambar lebih besar

✅ **Scalability**

- Tidak bottleneck di high-resolution
- Cocok untuk medical imaging, satellite imagery

✅ **Bidirectional Context**

- Capture information dari semua arah
- Better spatial understanding

✅ **Competitive Accuracy**

- Setara atau lebih baik dari ViT di berbagai task

---

### ⚠️ Limitasi Vision Mamba

❌ **Masih Relatif Baru**

- Ecosystem belum selengkap Transformer
- Tools & library masih berkembang

❌ **Training Complexity**

- SSM training butuh tuning khusus
- Pretrained weights masih terbatas

❌ **Interpretability**

- State space tidak se-intuitive attention maps
- Lebih susah di-visualize

---

## 🎓 Kesimpulan

### 📝 Rangkuman

**Vision Mamba adalah:**

1. **Alternatif efisien** untuk Vision Transformer
2. Menggunakan **Bidirectional State Space Models** untuk process gambar
3. **2.8x lebih cepat** dan **86.8% lebih hemat memory**
4. **Cocok untuk high-resolution images** dan real-time applications
5. **Competitive accuracy** di ImageNet, COCO detection, segmentation

---

### 🚀 Kenapa Menarik untuk Research?

1. **Efficiency**: Solusi untuk bottleneck Transformer
2. **Scalability**: Handle gambar resolusi tinggi
3. **Novel Architecture**: Bukan attention, tapi SSM!
4. **Strong Performance**: Competitive dengan SOTA
5. **Future Potential**: Bisa jadi backbone untuk vision foundation models

---

### 💡 Key Takeaways

> **Ingat ini:**

1. **Patch Embedding** → Potong gambar jadi pieces
2. **Position Embedding** → Kasih alamat setiap piece
3. **Mamba Blocks** → Process dengan SSM (bidirectional)
4. **State Space Model** → Ingat info penting, lupakan yang nggak penting
5. **Bidirectional** → Lihat dari depan DAN belakang
6. **Object Detection** → ViM Backbone + FPN + RPN + ROI Head

---

### 🎯 Analogi Final

**Vision Mamba seperti:**

> Bayangkan kamu baca buku komik tebal. **Vision Transformer** seperti setiap kali buka halaman baru, kamu harus **ingat semua halaman sebelumnya** dan **bandingkan satu-satu** → capek & lambat! 😵
>
> **Vision Mamba** seperti kamu punya **2 teman** yang baca dari arah berbeda:
>
> - Teman 1 baca dari halaman 1 → 100
> - Teman 2 baca dari halaman 100 → 1
>
> Mereka cuma **ingat poin penting** dan **share informasi** satu sama lain → **cepat & efisien**! ⚡

---

### 📚 Referensi

**Paper:**

- [Vision Mamba (ICML 2024)](https://arxiv.org/abs/2401.09417)
- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)

**Code:**

- [Vision Mamba Official Repo](https://github.com/hustvl/Vim)
- [Mamba SSM](https://github.com/state-spaces/mamba)

**Related Work:**

- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [DeiT](https://arxiv.org/abs/2012.12877)
- [Detectron2](https://github.com/facebookresearch/detectron2)

---

## 🙋 FAQ

**Q: Apakah Vision Mamba bisa replace Vision Transformer sepenuhnya?**

A: Belum tentu. ViT masih punya kelebihan di interpretability dan ecosystem yang mature. Tapi untuk use case yang butuh efisiensi (edge devices, high-res images), Vision Mamba lebih unggul.

---

**Q: Apakah Vision Mamba lebih sulit untuk training?**

A: Tidak jauh berbeda. Cuma perlu tuning hyperparameter yang sedikit berbeda, terutama untuk SSM-related parameters (d_state, dt_rank).

---

**Q: Bisa digunakan untuk video understanding?**

A: Sangat cocok! Karena SSM memang dirancang untuk sequential data. Vision Mamba bisa process video frames secara efisien.

---

**Q: GPU apa yang recommended?**

A: Minimal RTX 3060 (12GB) untuk inference. Untuk training, RTX 3090 atau A100 recommended.

---

**Good luck exploring Vision Mamba! 🚀🔬**

_Semoga penjelasan ini membantu memahami Vision Mamba dengan mudah!_

---

**Dibuat dengan ❤️ untuk pembelajaran**
_Last updated: November 2025_
