"""Fraud Detection Dashboard"""

from functools import partial

import panel as pn
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

from fraud_detection.contracts.dto.dashboard.state import DashboardState

# Dashboard State
dashboard = DashboardState()

def update_stream_data(params) -> None:
    """Update state and view for data stream."""

    transactions_df_pane, graph_source, fraud_cases_df_pane = params
    dashboard.update()

    transactions_df_pane.value = dashboard.transactions
    fraud_cases_df_pane.value = dashboard.fraud_review_cases

    update = {"x": dashboard.transactions["time"], "y": dashboard.transactions["amount"]}
    graph_source.data.update(update)


def update_mlflow_data(params):
    """Update state and view for mlflow reports."""

    workflow_report_pane, model_report_pane = params
    dashboard.update_mlflow_data()

    workflow_report_pane.value = dashboard.mlflow_workflow_runs.drop(labels=["name"], axis=1)
    model_report_pane.value = dashboard.mlflow_models.drop(labels=["name", "run_id", "tags"], axis=1)


def reset_btn_action(event):
    """Reset State of Dashboard and Data Steam API"""
    dashboard.reset()


def panel_app():
    """Panel Application Definition"""

    # Panes
    transactions_df_pane = pn.widgets.DataFrame(dashboard.transactions, show_index=False, width=200, sizing_mode="stretch_height")
    fraud_cases_df_pane = pn.widgets.DataFrame(dashboard.fraud_review_cases, show_index=False, width=200, sizing_mode="stretch_height")
    model_report_pane = pn.widgets.DataFrame(dashboard.mlflow_models, show_index=False, sizing_mode="stretch_width", min_height=300)
    workflow_report_pane = pn.widgets.DataFrame(dashboard.mlflow_workflow_runs, show_index=False, sizing_mode="stretch_width", min_height=300)

    # Graphs
    graph_source: ColumnDataSource = ColumnDataSource({"x": [], "y": []})
    graph_tooltips = [
        ("Amount", "$y"),
        ("Time", "@x"),
    ]
    graph: figure = figure(x_axis_label="Time", y_axis_label="Amount", min_width=700, min_height=350, sizing_mode="scale_both", tooltips=graph_tooltips)
    graph.line(x="x", y="y", source=graph_source)

    # Seed initial state
    update_stream_data((transactions_df_pane, graph_source, fraud_cases_df_pane))
    update_mlflow_data((workflow_report_pane, model_report_pane))

    # Callbacks
    cb = pn.state.add_periodic_callback(partial(update_stream_data, [transactions_df_pane, graph_source, fraud_cases_df_pane]), period=1000)
    cb_mlflow = pn.state.add_periodic_callback(partial(update_mlflow_data, [workflow_report_pane, model_report_pane]), period=10000)

    # Buttons
    toggle_btn = pn.widgets.Toggle(name="Toggle", value=True)
    toggle_btn.link(cb, bidirectional=True, value="running")
    toggle_btn.link(cb_mlflow, bidirectional=True, value="running")
    reset_btn = pn.widgets.Button(name="Reset", button_type="primary")
    reset_btn.on_click(reset_btn_action)

    # Layout
    return pn.Column(
        pn.Row(pn.pane.HTML("<h1>Fraud Detection Dashboard</h1>"), sizing_mode="stretch_width"),
        pn.Row(
            pn.Column(
                pn.pane.HTML("<h2>Transaction Amounts vs. Time</h2>"),
                graph,
                sizing_mode="stretch_width"
            ),
            pn.Column(
                pn.pane.HTML("<h2>Transactions</h2>"),
                transactions_df_pane
            ),
            pn.Column(
                pn.pane.HTML("<h2>Suspect Transactions</h2>"),
                fraud_cases_df_pane
            ),
        ),
        pn.Row(
            pn.Column(pn.pane.HTML("<h2>Models</h2>"), model_report_pane),
            pn.Column(pn.pane.HTML("<h2>Training Runs</h2>"), workflow_report_pane)
        ),
        pn.Row(toggle_btn, reset_btn),
        sizing_mode="stretch_both"
    )


panel_app().servable()
