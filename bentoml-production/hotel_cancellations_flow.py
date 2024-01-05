from metaflow import FlowSpec, Parameter, current, step


class HotelCancellationsFlow(FlowSpec):
    source_file = Parameter("source-file", help="Source CSV file")
    tracking_uri = Parameter("tracking-uri", help="MLflow tracking URI", default="http://127.0.0.1:5000")
    train_proportion = Parameter("train-proportion", help="Proportion of the dataset to use for training", default=0.6)
    test_proportion = Parameter(
        "test-proportion", help="Proportion of the remaining dataset to use for testing", default=0.5
    )

    @step
    def start(self):
        import mlflow

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment("/hotel-facilito/BookingCancellations")
        run = mlflow.start_run()
        self.mlflow_run_id = run.info.run_id
        mlflow.set_tag("metaflow.runNumber", current.run_id)
        mlflow.set_tag("metaflow.flowName", current.flow_name)
        print(f"Primer paso – started run: {self.mlflow_run_id}")
        self.next(self.load_data)

    @step
    def load_data(self):
        import mlflow
        import pandas as pd

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            hotel_bookings = pd.read_csv(self.source_file)

            # Remove personal information of customers
            hotel_bookings = hotel_bookings.drop(["name", "email", "phone-number", "credit_card"], axis=1)

            # Avoid data leakage
            hotel_bookings = hotel_bookings.drop(["reservation_status", "reservation_status_date"], axis=1)

            # Convert objects to strings
            object_columns = hotel_bookings.select_dtypes("object").columns
            hotel_bookings[object_columns] = hotel_bookings[object_columns].astype(str)

            self.hotel_bookings = hotel_bookings

        self.next(self.split_dataset)

    @step
    def split_dataset(self):
        import mlflow
        from sklearn.model_selection import train_test_split

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            is_canceled = self.hotel_bookings["is_canceled"].copy()
            hotel_data = self.hotel_bookings.drop(["is_canceled"], axis=1)

            original_count = len(self.hotel_bookings)
            training_size = int(original_count * self.train_proportion)
            test_size = int((1 - self.train_proportion) * self.test_proportion * training_size)

            train_x, rest_x, train_y, rest_y = train_test_split(hotel_data, is_canceled, train_size=training_size)
            test_x, validate_x, test_y, validate_y = train_test_split(rest_x, rest_y, train_size=test_size)

            mlflow.log_params(
                {
                    "dataset_size": original_count,
                    "training_set_size": len(train_x),
                    "validate_set_size": len(validate_x),
                    "test_set_size": len(test_x),
                }
            )

            self.train_x = train_x
            self.train_y = train_y
            self.validate_x = validate_x
            self.validate_y = validate_y
            self.test_x = test_x
            self.test_y = test_y

        self.next(self.model_training)

    @step
    def model_training(self):
        import tempfile

        import mlflow
        from joblib import dump

        from feature_pipeline import build_pipeline

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            self.training_pipeline = build_pipeline()
            self.training_pipeline.fit(self.train_x, self.train_y)

            with tempfile.TemporaryDirectory() as temp:
                dump(self.training_pipeline, f"{temp}/inference_pipeline.joblib")
                mlflow.log_artifact(f"{temp}/inference_pipeline.joblib")

        self.next(self.model_validation)

    @step
    def model_validation(self):
        import mlflow
        from sklearn.metrics import accuracy_score, recall_score

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            train_pred_y = self.training_pipeline.predict(self.train_x)
            validate_pred_y = self.training_pipeline.predict(self.validate_x)

            train_accuracy = accuracy_score(train_pred_y, self.train_y)
            train_recall = recall_score(train_pred_y, self.train_y)

            validate_accuracy = accuracy_score(validate_pred_y, self.validate_y)
            validate_recall = recall_score(validate_pred_y, self.validate_y)

            print("Train accuracy", train_accuracy)
            print("Train recall", train_recall)

            print("Validate accuracy", validate_accuracy)
            print("Validate recall", validate_recall)

            metrics = {
                "train_accuracy": train_accuracy,
                "train_recall": train_recall,
                "validate_accuracy": validate_accuracy,
                "validate_recall": validate_recall,
            }

            mlflow.log_metrics(metrics)

        self.next(self.register_model)

    @step
    def register_model(self):
        import mlflow
        import mlflow.sklearn
        from mlflow.models.signature import infer_signature

        mlflow.set_tracking_uri(self.tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            signature = infer_signature(self.train_x, self.train_y)

            mlflow.sklearn.log_model(
                sk_model=self.training_pipeline,
                artifact_path="hotel-cancellations-model",
                signature=signature,
                registered_model_name="hotel-cancellations-model",
            )

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    HotelCancellationsFlow()
