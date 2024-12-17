import pandas as pd
import numpy as np
import pickle
import streamlit as st
import matplotlib.pyplot as plt

pickle_vec = open('vectorizer.pkl', 'rb')
vectorizer = pickle.load(pickle_vec)

pickle_in = open('decision_tree.pkl', 'rb')
classifier = pickle.load(pickle_in)

def prediction(content, score):
    """
    Fungsi untuk melakukan prediksi.
    :param content: string, isi ulasan dari user
    :param score: numeric, nilai rating ulasan (1-5)
    :return: hasil prediksi
    """
    
    content_vectorized = vectorizer.transform([content]).toarray()
    features = np.hstack((content_vectorized, np.array([[score]])))
    prediction = classifier.predict(features)
    return prediction[0]

def plot_eda(df):
    """
    Fungsi untuk menampilkan EDA pie chart distribusi sentimen.
    :param df: DataFrame, dataset dengan kolom 'sentimen'
    """
    fig, ax = plt.subplots()
    df['sentimen'].value_counts().plot(kind='pie', autopct='%1.1f%%', explode=[0,0.1], colors=['red', 'green'], ax=ax)
    ax.legend(['Tidak Bisa Dipercaya', 'Bisa Dipercaya'])
    st.pyplot(fig)

def main():
    """
    Fungsi utama untuk Streamlit app.
    """

    html_temp = """
    <div style='background-color:blue; padding:13px; margin-bottom: 20px;'>
    <h1 style='text-align:center;'>Masih Percayakah Kamu dengan DANA?</h1>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    st.markdown(
       """
       <p style="text-align: justify; text-justify: inter-word;">
       Aplikasi DANA merupakan dompet digital yang populer di Indonesia, digunakan untuk berbagai transaksi seperti pembayaran, transfer uang, dan pembelian produk. Meskipun demikian, DANA menghadapi tantangan terkait penurunan tingkat kepercayaan pengguna setelah tidak lagi diawasi oleh Otoritas Jasa Keuangan (OJK). Hal ini menyebabkan munculnya kekhawatiran mengenai keamanan transaksi, seperti kasus kehilangan uang secara tiba-tiba tanpa perlindungan yang memadai.
        </p>
        """,
        unsafe_allow_html=True,
    )

    if st.checkbox('Bagaimana kepercayaan masyarakat terhadap DANA sebagai sarana transaksi online?'):
        dfx = pd.read_csv('data_bersih.csv', sep=';')

        if 'sentimen' in dfx.columns:
            plot_eda(dfx)
        else:
            st.error("Kolom 'sentimen' tidak ditemukan dalam dataset!")

    st.title('Kalau Menurutmu Bagaimana Nih?')

    content = st.text_input('Isi Ulasan dengan Kata Baku', 'Tulis ulasan Anda di sini...')
    score = st.number_input('Rating (1-5)', min_value=1, max_value=5, step=1)
    result = ''

    if st.button('Prediksi'):
        result = prediction(content, score)
        st.success('Hasil Prediksi adalah: DANA {}'.format(result))

if __name__ == '__main__':
    main()
