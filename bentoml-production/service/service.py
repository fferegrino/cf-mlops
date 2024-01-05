import bentoml
import numpy as np
from bentoml.io import NumpyNdarray, PandasDataFrame

MODEL_TAG = "hotel-cancellations-model"

hotel_cancellations_model_runner = bentoml.sklearn.get(MODEL_TAG).to_runner()

hotel_cancellations_service = bentoml.Service("hotel-cancellations-service", runners=[hotel_cancellations_model_runner])


@hotel_cancellations_service.api(input=PandasDataFrame(), output=NumpyNdarray())
def predict(input_df):
    # Necesitamos convertir `children` a `np.float64` manualmente debido a las peculiaridades de la serialización JSON
    input_df["children"] = input_df["children"].astype(np.float64)

    return hotel_cancellations_model_runner.run(input_df)
