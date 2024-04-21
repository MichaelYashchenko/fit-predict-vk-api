from dataclasses import dataclass
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch


@dataclass
class NERModel:
    _tokenizer = AutoTokenizer.from_pretrained("yqelz/xml-roberta-large-ner-russian")
    _model = AutoModelForTokenClassification.from_pretrained("yqelz/xml-roberta-large-ner-russian")

    def get_key_phrases(self, text: str) -> dict[str, list[str]]:
        inputs = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        predictions = outputs.logits.argmax(dim=2)

        key_phrases = []
        tokens = inputs.tokens()
        for token, prediction in zip(tokens, predictions[0].numpy()):
            if prediction != 6:
                key_phrases.append((token, self._model.config.id2label[prediction]))

        grouped_entities = {}
        for token, label in key_phrases:
            entity_type = label[2:]  # Убираем префикс "B-" или "I-"
            if entity_type not in grouped_entities:
                grouped_entities[entity_type] = [token[1:]]
            else:
                if label.startswith('B-'):
                    grouped_entities[entity_type].append(token[1:])
                else:
                    if token[0] == '▁':
                        token = ' ' + token[1:]
                    grouped_entities[entity_type][-1] += f'{token}'

        return grouped_entities


ner_model = NERModel()


if __name__ == "__main__":
    first_text = """
        В Microsoft показали, как созданная нейросетью Мона Лиза читает рэп
    Команда исследователей искусственного интеллекта Research Asia компании Microsoft разработала приложение искусственного интеллекта (ИИ), преобразующее неподвижное изображение человека в правдоподобную анимацию. При наложении звуковой дорожки получившийся цифровой аватар проговорит или пропоет текст с правильной мимикой. Результаты работы опубликованы на портале научных материалов arXiv.
    Новую нейросеть назвали VASA-1. Создатели проекта отметили, что ИИ может работать как с фотографиями, так и с рисунками. В качестве демонстрации возможностей группа представила ряд видеороликов, на которых созданные VASA-1 цифровые аватары поют и разговаривают. А «Мону Лизу» Леонардо да Винчи алгоритмы заставили зачитать рэп.
    В каждой анимации выражение лица меняется вместе со словами, подчеркивая сказанное. Исследователи также отметили, что, несмотря на реалистичность видео, более пристальное рассмотрение может выявить недостатки и свидетельства того, что они были созданы искусственно.
    По словам специалистов, инструмент генерирует видео разрешением 512 на 512 пикселей со скоростью 45 кадров в секунду, а для использования достаточно мощности потребительской видеокарты. Например, создание ролика с помощью графического процессора Nvidia RTX 4090 занимает около двух минут. Команда отметила, что VASA-1 можно использовать для создания дипфейков, поэтому система пока не будет общедоступной.
        """
    print(ner_model.get_key_phrases(first_text))
