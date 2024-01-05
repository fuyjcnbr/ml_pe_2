
from main import TITLE_TEXT, BUTTON_TEXT, UPLOADER_LABEL, RESULTS_TEXT


def test_labels():
    assert TITLE_TEXT == "Классификация изображений"
    assert BUTTON_TEXT == "Распознать изображение"
    assert UPLOADER_LABEL == "Выберите изображение для распознавания"
    assert RESULTS_TEXT == "**Результаты распознавания:**"
