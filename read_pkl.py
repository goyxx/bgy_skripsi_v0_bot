import pickle

# Ganti 'nama_file.pkl' dengan nama file kamu
with open('./model/svm_model.pkl', 'rb') as f:
    data = pickle.load(f)

# Menampilkan isi
print(data)
