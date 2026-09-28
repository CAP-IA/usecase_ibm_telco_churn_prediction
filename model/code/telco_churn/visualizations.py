import numpy as np
import pandas as pd
import plotly.graph_objects as go


def create_risk_gauge(probability: float, threshold: float = 0.35) -> go.Figure:
    """Create a gauge comparing churn probability with the decision threshold.

    Args:
        probability: Predicted churn probability, between 0 and 1.
        threshold: Probability at or above which churn is predicted. Defaults to 0.35.

    Returns:
        A Plotly figure showing the probability and threshold as percentages.
    """
    probability = float(np.clip(probability, 0, 1))
    threshold = float(np.clip(threshold, 0, 1))

    probability_percent = probability * 100
    threshold_percent = threshold * 100

    if probability >= threshold:
        color = "#ff6366"
    else:
        color = "#42ce7a"

    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            domain={"x": [0, 1], "y": [0.24, 0.88]},
            value=probability_percent,
            number={
                "suffix": " %",
                "valueformat": ".1f",
                "font": {"size": 42, "color": "#f8fafc"},
            },
            title={
                "text": (
                    "Churn risk"
                    f"<br><span style='font-size:0.75em'>Threshold: {threshold_percent:.0f}%</span>"
                ),
                "font": {"size": 22, "color": "#f8fafc"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "ticksuffix": " %",
                    "tickcolor": "#cbd5e1",
                },
                "bar": {"color": color, "thickness": 0.35},
                "bgcolor": "rgba(255,255,255,0.08)",
                "borderwidth": 0,
                "steps": [{"range": [0, threshold_percent], "color": "rgba(66,206,122,0.15)"},
                          {"range": [threshold_percent, 100], "color": "rgba(255,99,102,0.15)"}],

                "threshold": {
                    "value": threshold_percent,
                    "thickness": 0.8,
                    "line": {"color": "#f8fafc", "width": 4},
                },
            },
        )
    )

    figure.add_annotation(
        x=0.5,
        y=0.015,
        xref="paper",
        yref="paper",
        text=(
            "The gauge compares the predicted churn probability with the decision threshold.<br>"
            f"Values at or above <b>{threshold:.0%}</b> are classified as likely to churn."
        ),
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        align="center",
        font={"size": 12, "color": "rgba(255,255,255,0.65)"},
    )

    figure.update_layout(
        template="plotly_dark",
        height=570,
        margin={"l": 35, "r": 35, "t": 70, "b": 35},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
    )
    return figure


def format_feature_name(feature_name: str) -> str:
    """Make a feature name easier to read by replacing underscores with spaces.

    Args:
        feature_name: Name of a model feature.

    Returns:
        The feature name with underscores replaced by spaces.
    """

    return feature_name.replace("_", " ")


def create_shap_chart(factors, top_n: int = 8) -> go.Figure:
    """Show the features with the largest impact on one churn prediction.

    Positive SHAP values increase the model's churn score, negative values decrease it. These contributions are to the model score, 
    not directly to the predicted probability.

    Args:
        factors: List of dictionaries from explain_customer(). Each dictionary contains "feature", "value", and "effect".
        top_n: Maximum number of features to display. Defaults to 8.

    Returns:
        A Plotly bar chart of the selected SHAP contributions.
    """
    shap_data = pd.DataFrame(factors)
    shap_data["contribution"] = shap_data["effect"]
    shap_data["absolute_contribution"] = shap_data["contribution"].abs()
    shap_data["display_name"] = shap_data.apply(lambda row: f"{format_feature_name(row['feature'])} = {row['value']}", axis=1,)

    top_features = shap_data.nlargest(top_n, "absolute_contribution").sort_values("absolute_contribution")
    positive_features = top_features[top_features["contribution"] >= 0]
    negative_features = top_features[top_features["contribution"] < 0]

    figure = go.Figure()

    if not negative_features.empty:
        figure.add_trace(
            go.Bar(
                x=negative_features["contribution"],
                y=negative_features["display_name"],
                orientation="h",
                name="Decreases churn risk",
                marker_color="#42ce7a",
                text=[f"{value:+.3f}" for value in negative_features["contribution"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "%{y}<br>"
                    "SHAP contribution: %{x:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    if not positive_features.empty:
        figure.add_trace(
            go.Bar(
                x=positive_features["contribution"],
                y=positive_features["display_name"],
                orientation="h",
                name="Increases churn risk",
                marker_color="#ff6366",
                text=[f"{value:+.3f}" for value in positive_features["contribution"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate=(
                    "%{y}<br>"
                    "SHAP contribution: %{x:.4f}"
                    "<extra></extra>"
                ),
            )
        )
            
    figure.add_vline(
        x=0,
        line_width=2,
        line_color="#f8fafc",
    )

    figure.add_annotation(
        x=0.5,
        y=-0.16,
        xref="paper",
        yref="paper",
        text=(
            "SHAP values measure how each feature moves this customer's prediction away from the model baseline.<br>"
            "Red increases the churn score, while green decreases it."
        ),
        showarrow=False,
        xanchor="center",
        yanchor="top",
        align="center",
        font={"size": 12, "color": "rgba(255,255,255,0.65)"},
    )

    figure.update_yaxes(
        categoryorder="array",
        categoryarray=top_features["display_name"].tolist(),
        automargin=True,
    )

    figure.update_xaxes(
        title="SHAP contribution to the model score",
        gridcolor="rgba(255,255,255,0.10)",
        zeroline=False,
    )

    figure.update_layout(
        template="plotly_dark",
        barmode="relative",
        height=max(380, 55 * len(top_features) + 130),
        margin={"l": 190, "r": 70, "t": 90, "b": 120},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
    )

    return figure