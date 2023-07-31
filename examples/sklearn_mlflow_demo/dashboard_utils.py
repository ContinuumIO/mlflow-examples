import logging
import warnings

import numpy as np
import pandas as pd
import panel as pn
import requests

from ae5_tools import demand_env_var, load_ae5_user_secrets

warnings.filterwarnings("ignore")
np.random.seed(42)

logging.basicConfig(level=logging.WARN)

# Load user specific configuration.
load_ae5_user_secrets()


# Invokes the REST Endpoint
def invoke_rest_endpoint(input_data: dict) -> dict:
    endpoint_url: str = demand_env_var(name="SELF_HOSTED_MODEL_ENDPOINT")
    response = requests.post(url=f"{endpoint_url}/invocations", json=input_data, verify=False)
    if response.status_code != 200:
        raise Exception(f"Received status code: ({response.status_code}), Failed to call prediction: {response.text}")
    return response.json()


# Get prediction for the  given input.
def predict(data_x: pd.DataFrame) -> pd.DataFrame:
    # Build the prediction request
    payload: list[dict] = data_x.to_dict(orient="records")
    json_payload: dict = {"dataframe_records": payload}

    # Call prediction service
    predictions: dict = invoke_rest_endpoint(input_data=json_payload)
    return pd.DataFrame(predictions)


# Define our sliders
fixed_acidity_float_slider = pn.widgets.EditableFloatSlider(name="Fixed Acidity", start=0, end=15, step=0.1, value=7.0)
volatile_acidity_float_slider = pn.widgets.EditableFloatSlider(
    name="Volatile Acidity", start=0, end=2, step=0.1, value=1.0
)
citric_acid_float_slider = pn.widgets.EditableFloatSlider(name="Citric Acid", start=0, end=2, step=0.1, value=1.0)
residual_sugar_float_slider = pn.widgets.EditableFloatSlider(
    name="Residual Sugar", start=0, end=100, step=0.1, value=50.0
)
chlorides_float_slider = pn.widgets.EditableFloatSlider(name="Chlorides", start=0, end=1, step=0.1, value=0.25)
free_sulfur_dioxide_float_slider = pn.widgets.EditableFloatSlider(
    name="Free Sulfur Dioxide", start=0, end=1000, step=0.1, value=100.0
)
total_sulfur_dioxide_float_slider = pn.widgets.EditableFloatSlider(
    name="Total Sulfur Dioxide", start=0, end=1000, step=0.1, value=200.0
)
density_float_slider = pn.widgets.EditableFloatSlider(name="Density", start=0, end=2, step=0.1, value=0.9)
pH_float_slider = pn.widgets.EditableFloatSlider(name="pH", start=0, end=14, step=0.1, value=2.0)
sulphates_float_slider = pn.widgets.EditableFloatSlider(name="Sulphates", start=0, end=2, step=0.1, value=0.5)
alcohol_float_slider = pn.widgets.EditableFloatSlider(name="Alcohol", start=0, end=100, step=0.1, value=7.0)


def build_blank_result_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fixed acidity": [],
                "volatile acidity": [],
                "citric acid": [],
                "residual sugar": [],
                "chlorides": [],
                "free sulfur dioxide": [],
                "total sulfur dioxide": [],
                "density": [],
                "pH": [],
                "sulphates": [],
                "alcohol": [],
                "prediction": [],
            }
        ]
    )


results_df_pane = pn.widgets.DataFrame(build_blank_result_row(), width=1300)


def clear_btn_action(event):
    results_df_pane.value = build_blank_result_row()


def submit_btn_action(event):
    feature_data: list[dict] = [
        {
            "fixed acidity": fixed_acidity_float_slider.value,
            "volatile acidity": volatile_acidity_float_slider.value,
            "citric acid": citric_acid_float_slider.value,
            "residual sugar": residual_sugar_float_slider.value,
            "chlorides": chlorides_float_slider.value,
            "free sulfur dioxide": free_sulfur_dioxide_float_slider.value,
            "total sulfur dioxide": total_sulfur_dioxide_float_slider.value,
            "density": density_float_slider.value,
            "pH": pH_float_slider.value,
            "sulphates": sulphates_float_slider.value,
            "alcohol": alcohol_float_slider.value,
            "prediction": [],
        }
    ]
    json_payload: dict = {"dataframe_records": feature_data}
    predictions: dict = invoke_rest_endpoint(input_data=json_payload)
    predicted_quality: float = predictions["predictions"][0]
    feature_data[0]["prediction"] = predicted_quality

    row_df = pd.DataFrame(feature_data)

    if results_df_pane.value.loc[[0]]["prediction"][0] == []:
        results_df_pane.value = row_df
    else:
        results_df_pane.value = pd.concat([results_df_pane.value, row_df], ignore_index=True)


# Define our buttons
submit_btn = pn.widgets.Button(name="Submit", button_type="primary")
submit_btn.on_click(submit_btn_action)

clear_btn = pn.widgets.Button(name="Clear", button_type="primary")
clear_btn.on_click(clear_btn_action)
