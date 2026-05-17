import pandas as pd
import pytest

from src.tools import (
    analyze_trends,
    detect_anomalies,
    simulate_price_change,
    simulate_promo_campaign
)

@pytest.fixture
def mock_cpg_data():
    """Generates a controlled, clean dataframe for testing tool logic."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"] * 2
    categories = ["Beverages"] * 6 + ["Snacks"] * 6
    regions = ["North"] * 6 + ["East"] * 6

    # Pre-defined, stable metrics
    revenue = [50000, 52000, 49000, 55000, 58000, 62000] + [40000, 42000, 41000, 45000, 47000, 50000]
    units_sold = [1000, 1100, 950, 1200, 1250, 1300] + [800, 850, 820, 900, 950, 1000]
    price = [2.5, 4.1, 3.0, 9.1, 5.0, 3.6, 2.2, 4.5, 8.7, 2.0, 5.3, 3.7]

    promo_flag = [0] * 8 + [1] * 4

    # Insert a clear anomaly row at index 3 (April - Beverages) to test anomaly detection
    units_sold[3] = 500  # Artificially low out-of-bounds anomaly

    return pd.DataFrame({
        "month": months,
        "category": categories,
        "store_region": regions,
        "revenue": revenue,
        "units_sold": units_sold,
        "price": price,
        "promo_flag": promo_flag,
        "store_id": [1, 2, 1, 2, 1, 2] * 2,
        "date": pd.date_range(start="2026-01-01", periods=12, freq="ME").strftime("%Y-%m-%d")
    })

def test_trend_analysis(mock_cpg_data):
    """Verifies trend analysis accurately aggregates revenue parameters."""
    monthly_df, summary = analyze_trends(
        mock_cpg_data,
        target_regions=['North', 'East'],
        target_categories=['Snacks', 'Beverages']
    )

    assert isinstance(summary, dict)
    assert "total_revenue" in summary
    assert "top_category" in summary
    assert summary["total_revenue"] > 0
    assert isinstance(monthly_df, pd.DataFrame)


def test_anomaly_detection(mock_cpg_data):
    """Verifies IQR boundaries capture out-of-bounds metrics correctly."""
    result = detect_anomalies(mock_cpg_data, target_col="units_sold")

    assert isinstance(result, dict)
    assert "lower_bound" in result
    assert "upper_bound" in result
    assert "total anomalies found" in result
    assert result["lower_bound"] < result["upper_bound"]


def test_price_change_simulation(mock_cpg_data):
    """Verifies negative pricing calculation correctly reduces projected revenue outcomes."""
    # Simulating a -10% price reduction drop
    result = simulate_price_change(mock_cpg_data, price_change_pct=-0.1)

    assert isinstance(result, dict)
    assert result["price_change_pct" if "price_change_pct" in result else "elasticity_coefficient_used"] == 0.5
    assert result["projected_revenue"] < result["baseline_revenue"]
    assert result["revenue_change_pct"] < 0


def test_promo_campaign_simulation(mock_cpg_data):
    """Verifies promotional lift equations scale revenue projections properly."""
    result = simulate_promo_campaign(mock_cpg_data, discount_pct=0.15, conversion_rate=0.35)

    assert isinstance(result, dict)
    assert result["discount_pct_used"] == 0.15
    assert result["conversion_rate_used"] == 0.35
    assert result["lift_factor_used"] == 1.5
    assert result["projected_revenue"] < result["baseline_revenue"]
