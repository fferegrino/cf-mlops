import bentoml
import pandas as pd


class HotelCancellationsModelRunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self, model: bentoml.Model) -> None:
        self.classifier = bentoml.sklearn.load_model(model)

    @bentoml.Runnable.method()
    def will_cancel(self, input_data: pd.DataFrame) -> pd.DataFrame:
        resultado = input_data[["reservation_id"]]
        predict_probas = self.classifier.predict_proba(input_data)
        negative_proba = predict_probas[:, 1]
        resultado["will_cancel"] = negative_proba
        return resultado
