import os
import shutil

# Folder asal dataset
folder_asal = "dataset_asli"
folder_tujuan = "dataset"

# Label baru dan mapping dari nama folder asli
mapping_label = {
    "Highly Fresh": "Segar",
    "Fresh": "Segar",
    "Not Fresh": "TidakSegar"
}

# Buat folder tujuan
for label in mapping_label.values():
    os.makedirs(os.path.join(folder_tujuan, label), exist_ok=True)

# Proses pemindahan dan penggabungan file
for nama_folder in os.listdir(folder_asal):
    for kategori_asli, label_baru in mapping_label.items():
        if kategori_asli in nama_folder:
            folder_lengkap = os.path.join(folder_asal, nama_folder)
            for nama_file in os.listdir(folder_lengkap):
                asal = os.path.join(folder_lengkap, nama_file)
                # Tambahkan nama jenis ikan ke nama file agar tidak bentrok
                nama_baru = nama_folder.replace(" ", "_") + "_" + nama_file
                tujuan = os.path.join(folder_tujuan, label_baru, nama_baru)
                shutil.copy(asal, tujuan)
