import bentoml
import numpy as np
from bentoml.io import PandasDataFrame

from hotel_cancellations_runner import HotelCancellationsModelRunner

MODEL_TAG = "hotel-cancellations-model"


hotel_cancellations_model = bentoml.sklearn.get(MODEL_TAG)
hotel_cancellations_model_runner = bentoml.Runner(
    HotelCancellationsModelRunner,
    models=[hotel_cancellations_model],
    runnable_init_params={"model": hotel_cancellations_model},
)

hotel_cancellations_service = bentoml.Service("hotel-cancellations-service", runners=[hotel_cancellations_model_runner])


@hotel_cancellations_service.api(input=PandasDataFrame(), output=PandasDataFrame())
def predict(input_df):
    # Necesitamos convertir `children` a `np.float64` manualmente debido a las peculiaridades de la serialización JSON
    input_df["children"] = input_df["children"].astype(np.float64)

    return hotel_cancellations_model_runner.will_cancel.run(input_df)
