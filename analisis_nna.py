import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy.spatial import cKDTree
from scipy import stats

# ============================================================
# 1. PARAMETER WILAYAH
# ============================================================
AREA_WIDTH = 50.0
AREA_HEIGHT = 50.0
TOTAL_AREA = AREA_WIDTH * AREA_HEIGHT

# ============================================================
# 2. MEMBUAT DATA TITIK KECELAKAAN
# ============================================================
rng = np.random.default_rng()

# Menentukan tiga pusat klaster
center_1 = rng.uniform(10.0, 25.0, size=2) + np.array([0.0, 15.0])
center_2 = rng.uniform(25.0, 40.0, size=2) - np.array([0.0, 10.0])
center_3 = rng.uniform(20.0, 32.0, size=2)

# Klaster 1: 75 titik
n_c1 = 75
c1_coords = rng.normal(
    loc=center_1,
    scale=rng.uniform(2.2, 3.8),
    size=(n_c1, 2)
)

# Klaster 2: 65 titik
n_c2 = 65
c2_coords = rng.normal(
    loc=center_2,
    scale=rng.uniform(2.5, 4.0),
    size=(n_c2, 2)
)

# Klaster 3: 50 titik
n_c3 = 50
c3_coords = rng.normal(
    loc=center_3,
    scale=rng.uniform(2.0, 3.5),
    size=(n_c3, 2)
)

# Titik acak latar belakang: 40 titik
n_bg = 40
bg_coords = rng.uniform(
    low=2.0,
    high=48.0,
    size=(n_bg, 2)
)

# Menggabungkan seluruh koordinat
all_coords = np.vstack([
    c1_coords,
    c2_coords,
    c3_coords,
    bg_coords
])

# Membatasi koordinat pada wilayah 1–49 km
all_coords = np.clip(all_coords, 1.0, 49.0)

print("Data berhasil dibuat.")
print("Jumlah titik:", len(all_coords))
print("Luas wilayah:", TOTAL_AREA, "km²")

# ============================================================
# 3. FUNGSI NEAREST NEIGHBOR ANALYSIS (NNA)
# ============================================================
def calculate_nna(coords, area_size):
    n = len(coords)
    tree = cKDTree(coords)

    # k=2 karena titik pertama adalah titik itu sendiri
    distances, indices = tree.query(coords, k=2)
    nn_dists = distances[:, 1]

    d_obs = np.mean(nn_dists)
    density = n / area_size
    d_exp = 1.0 / (2.0 * np.sqrt(density))
    r_index = d_obs / d_exp

    se = 0.26136 / np.sqrt(n * density)
    z_score = (d_obs - d_exp) / se
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return {
        'n': n,
        'd_obs': d_obs,
        'd_exp': d_exp,
        'R_index': r_index,
        'SE': se,
        'Z_score': z_score,
        'p_value': p_val,
        'nn_indices': indices[:, 1],
        'nn_dists': nn_dists
    }

# ============================================================
# 4. MENJALANKAN ANALISIS
# ============================================================
nna_result = calculate_nna(all_coords, TOTAL_AREA)

print("\n" + "=" * 60)
print("HASIL AVERAGE NEAREST NEIGHBOR ANALYSIS (NNA)")
print("=" * 60)
print(f"Jumlah Titik Sampel (n)  : {nna_result['n']}")
print(f"Jarak Teramati           : {nna_result['d_obs']:.3f} km")
print(f"Jarak Harapan CSR        : {nna_result['d_exp']:.3f} km")
print(f"Indeks Rasio Tetangga (R): {nna_result['R_index']:.4f}")
print(f"Z-Score                  : {nna_result['Z_score']:.3f}")
print(f"P-Value                  : {nna_result['p_value']:.4e}")

# ============================================================
# 5. INTERPRETASI HASIL
# ============================================================
R = nna_result['R_index']
Z = nna_result['Z_score']
p = nna_result['p_value']

print("\nINTERPRETASI:")

if p < 0.05 and R < 1 and Z < -1.96:
    print("Pola titik MENGELOMPOK secara signifikan.")
    print("Jarak antartitik cenderung lebih dekat daripada pola acak.")
elif p < 0.05 and R > 1 and Z > 1.96:
    print("Pola titik TERATUR secara signifikan.")
    print("Jarak antartitik cenderung lebih jauh daripada pola acak.")
else:
    print("Pola tidak terbukti berbeda signifikan dari pola acak.")
    print("Bukti statistik belum cukup untuk menyatakan pola mengelompok atau teratur.")

print("\nCatatan:")
print("NNA hanya menganalisis jarak tetangga terdekat.")
print("Metode ini tidak cukup untuk menjelaskan seluruh pola spasial kompleks.")
