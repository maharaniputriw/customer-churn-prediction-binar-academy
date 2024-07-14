import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import seaborn as sns
import streamlit as st

# Memuat data
data = pd.read_csv('Data_Train.csv')

# Fungsi untuk memuat model dari file pickle
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

# Memuat Model
model = load_model('churn_model.pkl')

# Fungsi untuk menghasilkan deskripsi data
def generate_data_description(data):
    st.title("Deskripsi Dataset")
    st.subheader("Deskripsi File")
    text = '''
    - **train.csv** - set latih.
    
    Terdiri dari 4250 baris dengan 20 kolom. 3652 sampel (85.93%) termasuk dalam kelas churn=tidak dan 598 sampel (14.07%) termasuk dalam kelas churn=ya.
    
    - **test.csv** - set tes.
    
    Berisi 750 baris dengan 20 kolom: indeks setiap sampel dan 19 fitur (tidak termasuk variabel target "churn").
    '''
    st.write(text)
    st.subheader("Bidang data")
    text = '''
    - **state**, *string*. Kode 2 huruf negara bagian AS tempat tinggal pelanggan
    - **account_length**, *numerical*. Jumlah bulan pelanggan telah menggunakan penyedia layanan telekomunikasi saat ini
    - **area_code**, *string*. "area_code_AAA" di mana AAA = 3 digit kode area.
    - **international_plan**, *(yes/no)*. Pelanggan memiliki paket internasional.
    - **voice_mail_plan**, *(yes/no)*. Pelanggan memiliki paket pesan suara.
    - **number_vmail_messages**, *numerical*. Jumlah pesan pesan suara.
    - **total_day_minutes**, *numerical*. Total menit panggilan dalam sehari.
    - **total_day_calls**, *numerical*. Jumlah total panggilan dalam sehari.
    - **total_day_charge**, *numerical*. Total biaya panggilan harian.
    - **total_eve_minutes**, *numerical*. Total menit panggilan sore.
    - **total_eve_calls**, *numerical*. Jumlah total panggilan sore.
    - **total_eve_charge**, *numerical*. Total biaya panggilan sore.
    - **total_night_minutes**, *numerical*. Total menit panggilan malam.
    - **total_night_calls**, *numerical*. Jumlah total panggilan malam.
    - **total_night_charge**, *numerical*. Total biaya panggilan malam.
    - **total_intl_minutes**, *numerical*. Total menit panggilan internasional.
    - **total_intl_calls**, *numerical*. Jumlah total panggilan internasional.
    - **total_intl_charge**, *numerical*. Total biaya panggilan internasional.
    - **number_customer_service_calls**, *numerical*. Jumlah panggilan ke layanan pelanggan
    - **churn**, *(yes/no)*. Perpindahan (churn) pelanggan - variabel target.
    '''
    st.write(text)
    st.subheader("Sumber Dataset")
    st.write("File-file tersebut diunduh dari link kaggle berikut. https://www.kaggle.com/competitions/customer-churn-prediction-2020/data")
    # Membuat checkbox jika user ingin melihat datanya
    if st.checkbox("Tampilkan Keseluruhan Data Latih (Data Train)"):
        st.write(data)
    # Membuat statistika deskriptif data
    st.write("Statistika Deskriptif Data:")
    st.write(data.describe().transpose())

# Fungsi untuk mengisi input data
def input_data_form():
    st.subheader('Masukkan Data Pelanggan')
    area = st.selectbox('Tinggal di kode area (Area_Code) mana?', [408, 415, 510])
    area_code_mapping = {'415': [1, 0, 0], '408': [0, 1, 0], '510': [0, 0, 1]}
    area_code = area_code_mapping[str(area)]
    
    international = st.radio('Apakah memiliki paket internasional?', ['Iya', 'Tidak'])
    international_plan_mapping = {'Tidak': [1, 0], 'Iya': [0, 1]}
    international_plan = international_plan_mapping[str(international)]
    
    voice_mail = st.radio('Apakah memiliki paket pesan suara?', ['Iya', 'Tidak'])
    voice_mail_plan_mapping = {'Tidak': [1, 0], 'Iya': [0, 1]}
    voice_mail_plan = voice_mail_plan_mapping[str(voice_mail)]
    
    total_day_minutes = st.number_input('Berapa total menit panggilan dalam sehari?', min_value=0.0, step=0.1)
    total_day_calls = st.number_input('Berapa jumlah panggilan dalam sehari?', min_value=0, step=1)
    total_day_charge = st.number_input('Berapa total biaya panggilan harian?', min_value=0.00, step=0.01)
    total_eve_minutes = st.number_input('Berapa total menit panggilan di sore hari?', min_value=0.0, step=0.1)
    total_eve_calls = st.number_input('Berapa jumlah panggilan di sore hari?', min_value=0, step=1)
    total_eve_charge = st.number_input('Berapa total biaya panggilan di sore hari?', min_value=0.00, step=0.01)
    total_night_minutes = st.number_input('Berapa total menit panggilan di malam hari?', min_value=0.0, step=0.1)
    total_night_calls = st.number_input('Berapa jumlah panggilan di malam hari?', min_value=0, step=1)
    total_night_charge = st.number_input('Berapa total biaya panggilan di malam hari?', min_value=0.00, step=0.01)
    total_intl_minutes = st.number_input('Berapa total menit panggilan internasional?', min_value=0.0, step=0.1)
    total_intl_calls = st.number_input('Berapa jumlah panggilan internasional?', min_value=0, step=1)
    total_intl_charge = st.number_input('Berapa total biaya panggilan internasional?', min_value=0.00, step=0.01)
    number_customer_service_calls = st.number_input('Berapa kali melakukan panggilan ke layanan pelanggan?', min_value=0, step=1)
    
    # Memisahkan array menjadi DataFrame
    area_code_df = pd.DataFrame([area_code], columns=['area_code_area_code_408', 'area_code_area_code_415', 'area_code_area_code_510'])
    international_plan_df = pd.DataFrame([international_plan], columns=['international_plan_no', 'international_plan_yes'])
    voice_mail_plan_df = pd.DataFrame([voice_mail_plan], columns=['voice_mail_plan_no', 'voice_mail_plan_yes'])
    
    data = {
        'total_day_minutes': total_day_minutes,
        'total_day_calls': total_day_calls,
        'total_day_charge': total_day_charge,
        'total_eve_minutes': total_eve_minutes,
        'total_eve_calls': total_eve_calls,
        'total_eve_charge': total_eve_charge,
        'total_night_minutes': total_night_minutes,
        'total_night_calls': total_night_calls,
        'total_night_charge': total_night_charge,
        'total_intl_minutes': total_intl_minutes,
        'total_intl_calls': total_intl_calls,
        'total_intl_charge': total_intl_charge,
        'number_customer_service_calls': number_customer_service_calls
    }
    
    # Membuat DataFrame dari data
    input_data = pd.DataFrame(data, index=[0])
    
    # Menggabungkan semua DataFrame menjadi satu
    input_data = pd.concat([area_code_df, international_plan_df, voice_mail_plan_df, input_data], axis=1)
    
    return input_data
    
# Fungsi untuk membuat klasifikasi
def visualisasi(data):
    st.title("Visualisasi Data")
    
    # Membuat Pie Chart
    st.subheader("Jumlah *Customer Churn*")
    # Hitung jumlah data di kolom churn
    churn_counts = data['churn'].value_counts(normalize=True)
    churn_counts = churn_counts.reset_index()
    churn_counts.columns = ['Churn', 'Percentage']
    # Memisahkan data dan label
    nilai = churn_counts['Percentage']
    keys = churn_counts['Churn']
    # Menentukan warna dan efek pecah
    explode = [0, 0.1]  # Anda mungkin ingin menyesuaikan ini sesuai kebutuhan
    # Mengatur warna palette menggunakan Seaborn
    palette_color = sns.color_palette('bright')
    # Membuat pie chart tanpa keterangan "Yes" dan "No" pada chart
    fig, ax = plt.subplots(figsize=(4, 4), facecolor='none', edgecolor='none')
    ax.pie(nilai, labels=None, colors=palette_color, 
            explode=explode, autopct='%.0f%%')
    # Menambahkan legenda berdasarkan label "Yes" dan "No" dengan ukuran yang lebih kecil
    ax.legend(title="Churn", labels=keys, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize='x-small')
    # Menampilkan chart di Streamlit
    st.pyplot(fig)
    st.write("Dapat dilihat bahwa pada data tersebut terdapat sebanyak 14% orang yang termasuk dalam kategori churn sedangkan sisanya tidak")
    
    # Membuat Histogram Chart
    st.subheader("Jumlah *Churn* dan *Not Churn* pada setiap state")
    # Menampilkan histogram dengan Seaborn
    fig, ax = plt.subplots(figsize=(19, 8), facecolor='none', edgecolor='none')
    sns.histplot(data=data, x='state', hue='churn', kde=True, bins=30, palette=['red', 'blue'], alpha=0.4, ax=ax)
    # Menampilkan chart di Streamlit
    # Mengubah warna label sumbu X menjadi abu-abu
    plt.xlabel("State", color="#fff")
    # Mengubah warna label sumbu Y menjadi abu-abu
    plt.ylabel("Count", color="#fff")
    # Mengubah warna teks pada sumbu menjadi abu-abu
    plt.tick_params(colors="#fff")
    st.pyplot(fig)
    st.write("Dari histogram persebaran data pelanggan (churn dan tidak) pada setiap negara bagian AS diatas. Dapat dilihat bahwa negara bagian dengan kode WV menjadi negara bagian yang memiliki jumlah pelanggan (churn dan tidak) paling besar yaitu sekitar 139 orang diikuti oleh negara bagian MN sekitar 108 orang. Sedangkan negara bagian CA memiliki jumlah pelanggan (churn dan tidak) paling sedikit yaitu sekitar 39 orang.")
    
    # Membuat Histogram Chart
    st.subheader("Distribusi persebaran jumlah *Churn* dan *Not Churn*")
    variabel = st.selectbox('Variabel', ["total_day_minutes", "total_day_calls", "total_day_charge", "total_eve_minutes", "total_eve_calls", "total_night_minutes", "total_night_calls", "total_night_charge", "total_intl_minutes", "total_intl_calls", "total_intl_charge"])
    # Menyiapkan data
    data['churn'] = data['churn'].astype(str)
    # Membuat chart menggunakan Altair
    account_length_hist = alt.Chart(data).mark_bar().encode(
        x=alt.X(variabel, bin=alt.Bin(maxbins=30)),
        y='count()',
        color='churn'
    ).properties(
        width=600,
        height=400
    )
    # Menampilkan chart di Streamlit
    st.altair_chart(account_length_hist, use_container_width=True)
    if variabel == "total_intl_calls":
        st.write("Dapat dilihat bahwa distribusi pada histogram tersebut mengalami positive skewness. Hal tersebut menunjukkan bahwa rata-ratanya lebih besar dari median, dan modusnya lebih kecil dari mean dan median.")
    else:
        st.write("Dapat dilihat bahwa distribusi pada histogram tersebut dapat dikatakan sebagai distribusi normal karena kurva tersebut memiliki distribusi yang berbentuk simetris atau menyerupai kurva lonceng. Hal tersebut menunjukkan bahwa sebagian besar data mengelompok di sekitar rata-rata, dan distribusinya merata di kedua sisi.")
    
    # Membuat Bar Chart
    variabel2 = st.selectbox('Variabel', ["area_code", "international_plan", "voice_mail_plan", "number_customer_service_calls"])
    # Menampilkan count plot menggunakan Seaborn
    fig, ax = plt.subplots(facecolor='none', edgecolor='none')
    ax = sns.countplot(x=variabel2, data=data, hue="churn", palette="bright", edgecolor="white")
    # Menambahkan teks di atas setiap bar
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 5),
                    textcoords='offset points')
    # Memberi judul pada plot
    plt.title('Jumlah Churn Berdasarkan {}'.format(variabel2), color='#fff')
    # Mengatur warna dan style sumbu
    ax.tick_params(colors='#fff')
    plt.xlabel(variabel2, color='#fff')
    plt.ylabel('Count', color='#fff')
    # Menghilangkan background sumbu dan label
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    # Menampilkan chart di Streamlit
    st.pyplot(fig)
    # Interpretasi
    if variabel2 == "area_code":
        st.write("Area code 415 memiliki lebih banyak pelanggan yang churn dan tidak sedangkan area code 510 memiliki jumlah pelanggan yang churn lebih banyak daripada area 408, namun memiliki jumlah pelanggan paling sedikit.")
    elif variabel2 == "international_plan":
        st.write("Dapat dilihat bahwa banyak pelanggan yang tidak memiliki paket internasional mengalami *Churn* maupun *Not Churn* jika dibandingkan dengan pelanggan yang memiliki paket internasional.")
    elif variabel2 == "voice_mail_plan":
        st.write("Dapat dilihat bahwa banyak pelanggan yang tidak memiliki paket pesan suara mengalami *Churn* maupun *Not Churn* jika dibandingkan dengan pelanggan yang memiliki paket pesan suara.")
    elif variabel2 == "number_customer_service_calls":
        st.write("Dapat dilihat bahwa pelanggan yang tidak mengalami *Churn* cenderung melakukan panggilan ke layanan pelanggan sebanyak 1 sampai 3 kali. Sedangkan pelanggan yang melakukan panggilan ke layanan pelanggan lebih dari 4 kali cenderung mengalami *churn*.")
    
# Fungsi untuk melakukan prediksi dengan model yang dimuat
def predict_with_model(model, input_data):
    prediction = model.predict(input_data)
    return prediction
    
# Fungsi untuk melakukan klasifikasi
def klasifikasi():
    st.title('Prediksi Customer Churn')
    
    # Form input data
    input_data = input_data_form()

    # Tombol untuk melakukan prediksi
    if st.button('Prediksi'):
        # Lakukan prediksi dengan model
        prediction = model.predict(input_data)
        st.subheader('Hasil Prediksi:')
        if prediction[0] == "yes":
            st.write("Hasil prediksi menunjukkan bahwa pelanggan tersebut **akan melakukan pindah layanan (Churn)**")
        else:
            st.write("Hasil prediksi menunjukkan bahwa pelanggan tersebut **tidak akan melakukan pindah layanan (churn)**")

# Sidebar
st.sidebar.image("logo.png", caption="Binar Academy")
st.sidebar.title("Navigasi")

page = st.sidebar.radio("", ("Deskripsi Data", "Visualisasi", "Klasifikasi"))

# Routing
if page == "Deskripsi Data":
    generate_data_description(data)
elif page == "Visualisasi":
    visualisasi(data)  
elif page == "Klasifikasi":
    klasifikasi()
    
# Tema 
hide_st_style="""
<style>
#MainMenu {visibility:hidden}
footer {visibility:hidden}
header {visibility:hidden}
</style>
"""