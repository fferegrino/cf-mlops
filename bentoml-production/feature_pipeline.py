def build_pipeline():
    import mlflow
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.preprocessing import Binarizer, OneHotEncoder, RobustScaler

    # One-hot encoder
    internal_one_hot_encoding = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    columns_to_encode = [
        "hotel",
        "meal",
        "distribution_channel",
        "reserved_room_type",
        "assigned_room_type",
        "customer_type",
    ]

    mlflow.log_param("one_hot_encoded_columns", columns_to_encode)
    encoder_params = internal_one_hot_encoding.get_params()
    mlflow.log_params({f"encoder__{key}": value for key, value in encoder_params.items()})

    one_hot_encoding = ColumnTransformer([("one_hot_encode", internal_one_hot_encoding, columns_to_encode)])

    # Binarizer
    internal_binarizer = Binarizer()
    columns_to_binarize = [
        "total_of_special_requests",
        "required_car_parking_spaces",
        "booking_changes",
        "previous_bookings_not_canceled",
        "previous_cancellations",
    ]
    internal_encoder_binarizer = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

    binarizer = ColumnTransformer([("binarizer", internal_binarizer, columns_to_binarize)])

    one_hot_binarized = Pipeline(
        [
            ("binarizer", binarizer),
            ("one_hot_encoder", internal_encoder_binarizer),
        ]
    )

    # Scaler
    internal_scaler = RobustScaler()
    columns_to_scale = ["adr"]

    scaler = ColumnTransformer([("scaler", internal_scaler, columns_to_scale)])

    # Passthrough columns
    pass_columns = [
        "stays_in_week_nights",
        "stays_in_weekend_nights",
    ]

    passthrough = ColumnTransformer([("pass_columns", "passthrough", pass_columns)])

    # Full pipeline
    feature_engineering_pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("categories", one_hot_encoding),
                        ("binaries", one_hot_binarized),
                        ("scaled", scaler),
                        ("passthrough", passthrough),
                    ]
                ),
            )
        ]
    )

    # Machine learning model
    model = RandomForestClassifier(n_estimators=100)

    model_params = model.get_params()
    mlflow.log_params({f"model__{key}": value for key, value in model_params.items()})

    # Full pipeline
    final_pipeline = Pipeline([("feature_engineering", feature_engineering_pipeline), ("model", model)])

    return final_pipeline
