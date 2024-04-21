import os
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
import numpy as np

from src.labels import LABEL_MAP

MAPPING_ALL = {v: k for k, v in LABEL_MAP.items()}
MAX_LEN = 512
THRESHOLD = 0.2

BERT_PATH = Path("src/models/bert_ft")
LABEL_MAP_KEYS = list(MAPPING_ALL.keys())


class BertModel:
    def __init__(self):
        self._load_tokenizer = AutoTokenizer.from_pretrained(Path(BERT_PATH, 'tokenizer'))
        self._load_model = BertForSequenceClassification.from_pretrained(
            BERT_PATH,  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=len(MAPPING_ALL.keys()),  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            problem_type="multi_label_classification",
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )
        self._load_model.eval().cpu()

    def predict(self, input_text: str) -> list[str]:
        input_ids = self._load_tokenizer.encode(
            input_text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self._load_model(
                input_ids,
                attention_mask=torch.ones_like(input_ids)
            )

        logits = outputs.logits
        probs = torch.sigmoid(logits).numpy()

        predicted_classes = np.array(LABEL_MAP_KEYS)[(probs >= THRESHOLD)[0]]
        predicted_classes = [MAPPING_ALL[i] for i in predicted_classes]

        if not predicted_classes:
            predicted_classes = MAPPING_ALL[torch.argmax(logits, dim=1).item()]
        return predicted_classes


bert_model = BertModel()


if __name__ == "__main__":
    first_text = """
    В Microsoft показали, как созданная нейросетью Мона Лиза читает рэп
Команда исследователей искусственного интеллекта Research Asia компании Microsoft разработала приложение искусственного интеллекта (ИИ), преобразующее неподвижное изображение человека в правдоподобную анимацию. При наложении звуковой дорожки получившийся цифровой аватар проговорит или пропоет текст с правильной мимикой. Результаты работы опубликованы на портале научных материалов arXiv.
Новую нейросеть назвали VASA-1. Создатели проекта отметили, что ИИ может работать как с фотографиями, так и с рисунками. В качестве демонстрации возможностей группа представила ряд видеороликов, на которых созданные VASA-1 цифровые аватары поют и разговаривают. А «Мону Лизу» Леонардо да Винчи алгоритмы заставили зачитать рэп.
В каждой анимации выражение лица меняется вместе со словами, подчеркивая сказанное. Исследователи также отметили, что, несмотря на реалистичность видео, более пристальное рассмотрение может выявить недостатки и свидетельства того, что они были созданы искусственно.
По словам специалистов, инструмент генерирует видео разрешением 512 на 512 пикселей со скоростью 45 кадров в секунду, а для использования достаточно мощности потребительской видеокарты. Например, создание ролика с помощью графического процессора Nvidia RTX 4090 занимает около двух минут. Команда отметила, что VASA-1 можно использовать для создания дипфейков, поэтому система пока не будет общедоступной.
    """
    print(
        bert_model.predict(first_text)
    )
