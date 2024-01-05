import numpy as np

from main import IMAGE_W, IMAGE_H


def test_clock():
        from keras.utils import load_img, img_to_array
        from tensorflow.keras.applications import DenseNet169
        from tensorflow.keras.applications.efficientnet import decode_predictions

        model = DenseNet169(weights='imagenet')

        image = load_img("tests/integrated/digital_clock.jpg", color_mode="rgb", target_size=(IMAGE_W, IMAGE_H))
        arr = img_to_array(image)
        x = np.expand_dims(arr, axis=0)
        preds = model.predict(x)
        classes = decode_predictions(preds, top=1)[0]
        assert classes[0][1] == "digital_clock"
