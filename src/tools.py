import pandas as pd
import numpy as np

def analyze_trends(df, target_regions=None, target_categories=None):
    """
    Performs monthly trend aggregation and computes portfolio-wide KPIs, with optional
    filtering for specific store regions and product categories.

    Parameters:
    - df: The baseline pandas DataFrame containing sales records.
    - target_regions: List of strings, specific store regions to filter by (e.g., ['East', 'South']). If None, applies to all.
    - target_categories: List of strings, specific categories to filter by (e.g., ['Dairy', 'Snacks']). If None, applies to all.

    Returns:
    - monthly: A DataFrame aggregated by month and category with total units sold and revenue.
    - summary: A dictionary containing the filtered subset's total revenue, total units, and the top-performing category.
    """
    df_filtered = df.copy()
    df_filtered['date'] = pd.to_datetime(df_filtered['date'])
    df_filtered['month'] = df_filtered['date'].dt.to_period('M')

    # Apply optional store region filter
    if target_regions:
        df_filtered = df_filtered[df_filtered['store_region'].isin(target_regions)]

    # Apply optional product category filter
    if target_categories:
        df_filtered = df_filtered[df_filtered['category'].isin(target_categories)]

    # Check if the filtered DataFrame is empty to avoid aggregation errors
    if df_filtered.empty:
        return pd.DataFrame(columns=['month', 'category', 'units_sold', 'revenue']), {
            'total_revenue': 0.0,
            'total_units': 0,
            'top_category': None
        }

    # Aggregate units sold and revenue by month and category
    monthly = df_filtered.groupby(['month', 'category']).agg({
        'units_sold': 'sum',
        'revenue': 'sum'
    }).reset_index()

    monthly['month'] = monthly['month'].astype(str)

    # Calculate portfolio KPI metrics for the selected scope
    summary = {
        'total_revenue': round(df_filtered['revenue'].sum(), 2),
        'total_units': int(df_filtered['units_sold'].sum()),
        'top_category': df_filtered.groupby('category')['revenue'].sum().sort_values(ascending=False).index[0]
    }

    return monthly, summary

def detect_anomalies(df, target_col='units_sold', target_regions=None, target_categories=None):
    """
    Identifies historical anomalies or outliers in daily store sales using the Inter-quartile Range (IQR) method,
    with optional filtering for specific store regions and product categories.

    Parameters:
    - df: The baseline pandas DataFrame containing sales records.
    - target_col: String, the numerical column to evaluate for anomalies (defaults to 'units_sold').
    - target_regions: List of strings, specific store regions to isolate (e.g., ['East', 'South']). If None, applies to all.
    - target_categories: List of strings, specific categories to isolate (e.g., ['Dairy', 'Snacks']). If None, applies to all.

    Returns:
    - summary: A dictionary containing the IQR thresholds, total anomalies found, and the top 5 outliers below/above the bounds.
    """
    df_filtered = df.copy()

    # Apply optional store region filter
    if target_regions:
        df_filtered = df_filtered[df_filtered['store_region'].isin(target_regions)]

    # Apply optional product category filter
    if target_categories:
        df_filtered = df_filtered[df_filtered['category'].isin(target_categories)]

    # Check if the filtered DataFrame is empty to avoid empty evaluation errors
    if df_filtered.empty:
        return {
            'lower_bound': None,
            'upper_bound': None,
            'total anomalies found': 0,
            'top 5 anomalies below the lower bound': pd.DataFrame(),
            'top 5 anomalies above the upper bound': pd.DataFrame()
        }

    # Group by date and store to analyze daily transaction velocity
    store_daily = df_filtered.groupby(['date', 'store_id'])[target_col].sum().reset_index()

    # Compute IQR bounds based on the filtered scope
    q1 = store_daily[target_col].quantile(0.25)
    q3 = store_daily[target_col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Separate anomalies outside acceptable statistical bounds
    anomalies = store_daily[
        (store_daily[target_col] < lower_bound) | (store_daily[target_col] > upper_bound)].reset_index(drop=True)

    # Extract bottom outliers (below lower bound)
    anomalies_below_lower_bound = store_daily[(store_daily[target_col] < lower_bound)]
    if not anomalies_below_lower_bound.empty:
        anomalies_below_lower_bound = anomalies_below_lower_bound.sort_values(target_col, ascending=False).reset_index(
            drop=True).head(5)
    else:
        anomalies_below_lower_bound = pd.DataFrame()

    # Extract top outliers (above upper bound)
    anomalies_above_upper_bound = store_daily[(store_daily[target_col] > upper_bound)]
    if not anomalies_above_upper_bound.empty:
        anomalies_above_upper_bound = anomalies_above_upper_bound.sort_values(target_col, ascending=False).reset_index(
            drop=True).head(5)
    else:
        anomalies_above_upper_bound = pd.DataFrame()

    # Consolidate baseline statistical boundaries and flagged row summaries
    summary = {
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'total anomalies found': anomalies.shape[0],
        'top 5 anomalies below the lower bound': anomalies_below_lower_bound,
        'top 5 anomalies above the upper bound': anomalies_above_upper_bound
    }

    return summary

def simulate_price_change(df, price_change_pct=0.15, elasticity=0.5, target_categories=None):
    """
    Simulates a price change and its subsequent impact on demand (units sold) and revenue.

    Parameters:
    - df: The baseline pandas DataFrame.
    - price_change_pct: Float, percentage change in price (e.g., 0.15 for 15% increase, -0.1 for 10% decrease).
    - elasticity: Float, demand elasticity coefficient (change in demand = price_change_pct * elasticity).
    - target_categories: List of strings, specific categories to apply the price change to. If None, applies to all.

    Returns:
    A dictionary containing the calculated financial impact metrics:
        - 'baseline_revenue' (float): Total revenue of the original dataset before changes.
        - 'projected_revenue' (float): Total portfolio revenue after combining simulated
          changes for target categories and baseline values for non-target categories.
        - 'revenue_change_pct' (float): The global percentage change in total revenue
          compared to the baseline.
        - 'elasticity_coefficient_used' (float): The elasticity factor applied in the simulation.
    """
    df_sim = df.copy()
    baseline_revenue = df_sim['revenue'].sum()

    # Create a mask for target rows
    if target_categories:
        mask = df_sim['category'].isin(target_categories)
    else:
        mask = pd.Series(True, index=df_sim.index)

    # Apply price hike/change
    df_sim.loc[mask, 'new_price'] = np.round(df_sim.loc[mask, 'price'] * (1 + price_change_pct), 2)

    # Calculate demand drop factor based on price change and elasticity
    demand_factor = 1 - (price_change_pct * elasticity)

    # Adjust units sold and calculate projected revenue for targeted items
    df_sim.loc[mask, 'projected_units_sold'] = np.maximum(0, np.round(
        df_sim.loc[mask, 'units_sold'] * demand_factor).astype(int))
    df_sim.loc[mask, 'projected_revenue'] = np.round(
        df_sim.loc[mask, 'projected_units_sold'] * df_sim.loc[mask, 'new_price'], 2)

    # Fill non-targeted rows with their original revenue to ensure a correct global total
    df_sim['projected_revenue'] = df_sim['projected_revenue'].fillna(df_sim['revenue'])

    # Calculate total projected revenue across the entire portfolio
    projected_revenue = df_sim['projected_revenue'].sum()

    impact = {
        'baseline_revenue': round(baseline_revenue, 2),
        'projected_revenue': round(projected_revenue, 2),
        'revenue_change_pct': round(((projected_revenue - baseline_revenue) / baseline_revenue) * 100, 2),
        'elasticity_coefficient_used': elasticity
    }

    return impact


def simulate_promo_campaign(df, promo_type='FlashSale', discount_pct=0.20, target_categories=None, conversion_rate=0.30,
                            lift_factor=1.5):
    """
    Simulates introducing a promotion campaign to a subset of currently non-promotional rows.

    Parameters:
    - df: The baseline pandas DataFrame.
    - promo_type: String, type of promo to apply ('FlashSale', 'Discount', 'BuyOneGetOne').
    - discount_pct: Float, discount multiplier applied to the baseline price.
    - target_categories: List of strings, the specific categories to run the promo on. If None, applies to all.
    - conversion_rate: Float, fraction of eligible rows to randomly convert to promo status.
    - lift_factor: Float, lift factor to simulate the demand hike in units sold due to promo campaign.

    Returns:
    A dictionary containing the following simulated campaign impact metrics:
        - 'baseline_revenue' (float): Total revenue of the original dataset before changes.
        - 'projected_revenue' (float): Total portfolio revenue after incorporating simulated
          promo spikes and price reductions for the converted rows.
        - 'revenue_change_pct' (float): The global percentage change in total revenue
          compared to the baseline.
        - 'discount_pct_used' (float): The discount rate applied to the converted promotional items.
        - 'conversion_rate_used' (float): The target conversion rate used to select promotional items.
        - 'lift_factor_used' (float): The lift factor used to simulate promotional demand spike in units sold.
    """
    df_sim = df.copy()
    baseline_revenue = df_sim['revenue'].sum()

    # Find rows eligible for a new promotion (currently have no promo)
    if target_categories:
        eligible_mask = (df_sim['category'].isin(target_categories)) & (df_sim['promo_flag'] == 0)
    else:
        eligible_mask = (df_sim['promo_flag'] == 0)

    eligible_indices = df_sim[eligible_mask].index

    num_to_convert = int(len(eligible_indices) * conversion_rate)

    if num_to_convert > 0:
        converted_indices = df_sim.loc[eligible_indices].sort_values(by='revenue', ascending=False).head(
            num_to_convert).index

        # Update promo markers
        df_sim.loc[converted_indices, 'promo_flag'] = 1
        df_sim.loc[converted_indices, 'promo_type'] = promo_type

        # Apply discount to price
        df_sim.loc[converted_indices, 'new_price'] = np.round(
            df_sim.loc[converted_indices, 'price'] * (1 - discount_pct), 2)

        # Simulate promotional demand spike using lift factor
        new_units = np.ceil(df_sim.loc[converted_indices, 'units_sold'] * lift_factor).astype(int)
        df_sim.loc[converted_indices, 'new_units_sold'] = new_units

        # Calculate revenue for the converted promo rows
        df_sim.loc[converted_indices, 'projected_revenue'] = np.round(
            df_sim.loc[converted_indices, 'new_units_sold'] * df_sim.loc[converted_indices, 'new_price'], 2)

    # Fill unconverted and non-targeted rows with their original baseline revenue
    df_sim['projected_revenue'] = df_sim['projected_revenue'].fillna(df_sim['revenue'])

    # Calculate total projected revenue across the entire portfolio
    projected_revenue = df_sim['projected_revenue'].sum()

    impact = {
        'baseline_revenue': round(baseline_revenue, 2),
        'projected_revenue': round(projected_revenue, 2),
        'revenue_change_pct': round(((projected_revenue - baseline_revenue) / baseline_revenue) * 100, 2),
        'discount_pct_used': discount_pct,
        'conversion_rate_used': conversion_rate,
        'lift_factor_used': lift_factor
    }

    return impact
