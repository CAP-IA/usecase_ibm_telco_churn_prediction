from telco_churn.visualizations import create_risk_gauge, create_shap_chart


def test_create_risk_gauge_changes_color_at_decision_threshold():
    below = create_risk_gauge(probability=0.34, threshold=0.35).data[0]
    at_threshold = create_risk_gauge(probability=0.35, threshold=0.35).data[0]

    assert below.value == 34.0
    assert below.gauge.threshold.value == 35.0
    assert below.gauge.bar.color == "#42ce7a"
    assert at_threshold.value == 35.0
    assert at_threshold.gauge.bar.color == "#ff6366"


def test_create_shap_chart_displays_positive_factors_without_negative_factors():
    factors = [{"feature": "Contract", "value": "Month-to-month", "effect": 0.4}]
    chart = create_shap_chart(factors)
    assert len(chart.data) == 1
    assert chart.data[0].name == "Increases churn risk"
    assert list(chart.data[0].x) == [0.4]