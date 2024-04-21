from transformers import AutoModel, AutoTokenizer
import torch
import pickle
import numpy as np

from src.labels import LABEL_MAP

XGBOOST_PATH = "src/models/xgb_clf.pkl"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class XGBoostBertEmbedding:
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
        self._bert_model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")
        self._xgboost_model = pickle.load(open(XGBOOST_PATH, "rb"))

    def predict(self, input_text: str):
        tokenized_texts = self._tokenizer(
            input_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            input_ids = tokenized_texts["input_ids"]
            attention_mask = tokenized_texts["attention_mask"]

            outputs = self._bert_model(input_ids, attention_mask=attention_mask)
            sentence_embeddings = mean_pooling(outputs, attention_mask).cpu().numpy()
        preds = self._xgboost_model.predict(sentence_embeddings)
        mask = [True if i == 1 else False for i in preds[0]]
        if sum(mask) == 0:
            return []
        found_tags = [np.array(list(LABEL_MAP.keys()))[mask][0]] # TODO: check if empty!
        return found_tags


xgboost_model = XGBoostBertEmbedding()


if __name__ == "__main__":
    input_text = """В Microsoft показали, как созданная нейросетью Мона Лиза читает рэп
    Команда исследователей искусственного интеллекта Research Asia компании Microsoft разработала приложение искусственного интеллекта (ИИ), преобразующее неподвижное изображение человека в правдоподобную анимацию. При наложении звуковой дорожки получившийся цифровой аватар проговорит или пропоет текст с правильной мимикой. Результаты работы опубликованы на портале научных материалов arXiv.
    Новую нейросеть назвали VASA-1. Создатели проекта отметили, что ИИ может работать как с фотографиями, так и с рисунками. В качестве демонстрации возможностей группа представила ряд видеороликов, на которых созданные VASA-1 цифровые аватары поют и разговаривают. А «Мону Лизу» Леонардо да Винчи алгоритмы заставили зачитать рэп.
    В каждой анимации выражение лица меняется вместе со словами, подчеркивая сказанное. Исследователи также отметили, что, несмотря на реалистичность видео, более пристальное рассмотрение может выявить недостатки и свидетельства того, что они были созданы искусственно.
    По словам специалистов, инструмент генерирует видео разрешением 512 на 512 пикселей со скоростью 45 кадров в секунду, а для использования достаточно мощности потребительской видеокарты. Например, создание ролика с помощью графического процессора Nvidia RTX 4090 занимает около двух минут. Команда отметила, что VASA-1 можно использовать для создания дипфейков, поэтому система пока не будет общедоступной."""
    print(
        xgboost_model.predict(input_text)
    )
