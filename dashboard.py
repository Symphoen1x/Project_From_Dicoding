import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
sns.set(style='dark')

order_produk_ds = pd.read_csv("order_produk_ds.csv")


st.header('Almad Collection Dashboard :sparkles:')

st.subheader('Analisis berdasarkan dua pertanyaan bisnis')

st.subheader('1. Seberapa berpengaruh nilai berat produk, lebar produk, tinggi produk, dan panjang produk terhadap harga produk?')
plt.figure(figsize=(10, 5))
plt.title("Visualisasi Heatmap berdasarkan relasi antar multivariate variabel",
          loc="center", fontsize=20)
c = order_produk_ds[['Berat_Produk', 'Panjang_Produk',
                     'Tinggi_Produk', 'Lebar_Produk', 'Harga_Produk']].corr()
sns.heatmap(c, cmap="RdYlGn", annot=True)
c
st.pyplot(plt)


st.subheader(
    '2. Apa saja kategori produk yang nilai pengirimanya top 10 teratas?')
kategori_terpilih = order_produk_ds.groupby(
    'Kategori_Produk')['Nilai_Pengiriman'].sum().reset_index()
# Pengurutan berdasarkan total nilai produk secara menurun
kategori_terpilih = kategori_terpilih.sort_values(
    by='Nilai_Pengiriman', ascending=False)
# Memilih 10 kategori teratas
top_10_teratas = kategori_terpilih.head(10)
print(top_10_teratas)
plt.figure(figsize=(15, 5))
plt.plot(
    top_10_teratas["Kategori_Produk"],
    top_10_teratas["Nilai_Pengiriman"],
    marker='o',
    linewidth=2,
    color="#72BCD4"
)
plt.title("Top 10 kategori produk dengan nilai pengiriman teratas",
          loc="center", fontsize=20)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()
st.pyplot(plt)


X = top_10_teratas[['Nilai_Pengiriman']]

n_clusters = 10

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
top_10_teratas['Cluster'] = kmeans.fit_predict(X)

top_10_teratas = top_10_teratas.sort_values(
    by='Nilai_Pengiriman', ascending=False)

# Data visualization untuk melihat hasil clustering
plt.figure(figsize=(12, 6))
for cluster_label in range(n_clusters):
    cluster_data = top_10_teratas[top_10_teratas['Cluster'] == cluster_label]
    plt.bar(cluster_data['Kategori_Produk'],
            cluster_data['Nilai_Pengiriman'], label=f'Cluster {cluster_label + 1}')

plt.title('Clustering Kategori Produk berdasarkan Nilai Pengiriman')
plt.xlabel('Kategori Produk')
plt.ylabel('Nilai Pengiriman')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.show()
st.pyplot(plt)
