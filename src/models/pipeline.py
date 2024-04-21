from src.models.bert import bert_model
from src.models.ner import ner_model
from src.models.xgboost import xgboost_model

from src.schemas import RequestModel, ResponseModel


def predict(input_: RequestModel) -> ResponseModel:
    input_text = input_.text
    bert_pred = bert_model.predict(input_text)
    xgb_pred = xgboost_model.predict(input_text)
    pred = list(set(bert_pred).union(set(xgb_pred)))
    kw = ner_model.get_key_phrases(input_text)
    return ResponseModel(tags=pred, key_words=kw)
