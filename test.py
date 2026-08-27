from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf

def predictDigit(image, labels):
    model = tf.keras.models.load_model("VGG16(2).h5")
    
    # Resize gambar menjadi (200, 200)
    img = image.resize((200, 200))
    
    # Mengonversi gambar menjadi array numpy
    img_array = np.array(img, dtype='float32')
    
    # Normalisasi gambar
    img_array = img_array / 255.0
    
    # Menambahkan dimensi batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Menampilkan gambar untuk verifikasi
    plt.imshow(img_array[0])
    plt.show()
    
    # Melakukan prediksi
    pred = model.predict(img_array)
    result = np.argmax(pred[0])
    
    # Mendapatkan label yang sesuai dengan hasil prediksi
    label = list(labels.keys())[list(labels.values()).index(result)]
    return label

# Streamlit 
st.set_page_config(page_title='Handwritten Aksara Sunda Recognition', layout='wide')
st.title('Handwritten Aksara Sunda Recognition')
st.subheader("Draw the Aksara Sunda on canvas and click on 'Predict Now'")

# Keterangan labels
labels = {'ba': 0, 'ca': 1, 'da': 2, 'ga': 3, 'ha': 4, 'ja': 5, 'ka': 6, 'la': 7, 'ma': 8, 'na': 9, 'nga': 10, 'nya': 11, 'pa': 12, 'ra': 13, 'sa': 14, 'ta': 15, 'wa': 16, 'ya': 17}

# Membuat dua kolom
col1, col2 = st.columns([2, 1])

with col1:
    # Add canvas component
    drawing_mode = "freedraw"
    stroke_width = st.slider('Select Stroke Width', 1, 30, 15)
    stroke_color = '#FFFFFF'
    bg_color = '#000000'
    
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=500,
        width=500,
        key="canvas",
    )

with col2:
    st.image('aksara sunda ngalgena.png', caption='Contoh Aksara Sunda', width=None)


# Add "Predict Now" button
if st.button('Predict Now'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        
        # Konversi gambar ke RGB
        if input_image.mode == 'RGBA':
            input_image = input_image.convert('RGB')
        
        # Simpan gambar yang digambar ke dalam file
        input_image.save('prediction/img.png')
        
        # Baca gambar yang disimpan
        img = Image.open("prediction/img.png")
        
        # Lakukan prediksi
        res = predictDigit(img, labels)
        
        # Tampilkan hasil prediksi
        st.header('Predicted Aksara: ' + str(res))
    else:
        st.header('Please draw a digit on the canvas.')
