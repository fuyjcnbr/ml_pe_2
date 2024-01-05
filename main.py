
import io
import streamlit as st
from PIL import Image


TITLE_TEXT = "Классификация изображений"
BUTTON_TEXT = "Распознать изображение"
UPLOADER_LABEL = "Выберите изображение для распознавания"
RESULTS_TEXT = "**Результаты распознавания:**"

IMAGE_W = 224
IMAGE_H = 224

if __name__ == "__main__":

    from tensorflow.keras.applications import DenseNet169
    # from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
    # from tensorflow.keras.applications.efficientnet import decode_predictions
    import numpy as np


def load_image():
    """Создание формы для загрузки изображения"""
    # Форма для загрузки изображения средствами Streamlit
    uploaded_file = st.file_uploader(label=UPLOADER_LABEL) #'Выберите изображение для распознавания')
    if uploaded_file is not None:
        # Получение загруженного изображения
        image_data = uploaded_file.getvalue()
        # Показ загруженного изображения на Web-странице средствами Streamlit
        st.image(image_data)
        # Возврат изображения в формате PIL
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def preprocess_image(img):
    img = img.resize((IMAGE_W, IMAGE_H))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


if __name__ == "__main__":

    @st.cache(allow_output_mutation=True)
    def load_model():
        model = DenseNet169(weights='imagenet')
        return model


    def print_predictions(preds):
        classes = decode_predictions(preds, top=3)[0]
        for cl in classes:
            st.write(cl[1], cl[2])

    # Загружаем предварительно обученную модель
    model = load_model()
    # Выводим заголовок страницы
    st.title(TITLE_TEXT) #'Классификация изображений')
    # Выводим форму загрузки изображения и получаем изображение
    img = load_image()
    # Показывам кнопку для запуска распознавания изображения
    result = st.button(BUTTON_TEXT) #'Распознать изображение')
    # Если кнопка нажата, то запускаем распознавание изображения
    if result:
        # Предварительная обработка изображения
        x = preprocess_image(img)
        # Распознавание изображения
        preds = model.predict(x)
        # Выводим заголовок результатов распознавания жирным шрифтом
        # используя форматирование Markdown
        st.write(RESULTS_TEXT) #'**Результаты распознавания:**')
        # Выводим результаты распознавания
        print_predictions(preds)
