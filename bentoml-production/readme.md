# 0. Entrena un nuevo modelo y guárdalo en *MLflow*

Inicia *mlflow*:

```bash
mlflow server
```

Ejecuta el flow que crea y registra modelos:

```bash
python hotel_cancellations_flow.py run --source-file data/original.csv
```

# 1. Guardar el modelo localmente

Crea un archivo llamado `download_bento_model.py`

```python
import bentoml
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

model_name = "hotel-cancellations-model"
model_version = 1
model_mlflow_uri = f"models:/{model_name}/{model_version}"

sklearn_model = mlflow.sklearn.load_model(model_uri=model_mlflow_uri)

bento_model = bentoml.sklearn.save_model(f"{model_name}:{model_version}", sklearn_model)
```

## 1.1 Comprobar que el modelo haya sido descargado

(No es necesario para producción)

```bash
bentoml models list
```

Cada modelo recibe una etiqueta (*tag*), es importante tener esto en cuenta.

# 2. Crea y ejecuta un servicio de BentoML

Crea una carpeta llamada `service`. 

Dentro de la carpeta crea un archivo llamado `service.py`:

```python
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
```

# 3. Crea un Bento

Dentro de tu carpeta `service` crea un archivo llamado `bentofile.yml`:

```yaml
service: "service:hotel_cancellations_service"
labels:
  owner: "Team Facilito"
include:
  - "*.py"
python:
  packages:
   - "scikit-learn"
```

```bash
bentoml build
```

Revisa que la creación se haya completado con:

```bash
bentoml list
```

Sirve el *bento* con:

```bash
bentoml serve hotel-cancellations-service:[tag]
```

# 4. Crea un contenedor de Docker a partir de tu Bento


```bash
bentoml containerize hotel-cancellations-service:[tag]
```

Sirve tu contenedor con:

```bash
docker run -p 3000:3000 hotel-cancellations-service:[tag]
```
