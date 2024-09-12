from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_model, split_data, train_model, quality_drift_check, prediction_drift_check, report_plotly


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_dataset", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs="y_pred",
                name="evaluate_model_node",
            ),
            node(
                func=quality_drift_check,
                inputs=["X_train", "X_test"],
                outputs="data_drift",
                name="data_quality_check",
            ),
            node(
                func= prediction_drift_check,
                inputs=["y_test", "y_pred"],
                outputs="pred_drift",
                name="pred_drift_check"
            ),
            node(
                func= report_plotly,
                inputs=["data_drift", "pred_drift"],
                outputs=None,
                name="report_plotly"
            )
           
        ]
    )
