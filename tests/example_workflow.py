# tests/example_test_workflow.py
"""
Example demonstrating the complete testing workflow with visualization.

This shows how to:
1. Test individual functions
2. Visualize transformations
3. Run the full pipeline
4. Generate reports

Run with: pytest tests/example_test_workflow.py -v -s
"""

import sys
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest
import polars as pl
from factorlabs.data.ingest_yf import (
    normalize_column_names,
    normalize_date_column,
    data_quality_fixing,
    normalize_prices,
)
from test_utils import (
    DataFrameComparator,
    PipelineVisualizer,
    visualize_pipeline,
    assert_no_nulls_in_columns,
    assert_column_types,
    assert_values_in_range,
)


# ============================================================================
# EXAMPLE 1: Visual Debugging of a Transformation
# ============================================================================

@pytest.mark.visual
def test_visualize_column_normalization():
    """See exactly what happens during column normalization"""
    # Create messy input data
    raw_df = pl.DataFrame({
        "('date', '')": ["2024-01-01", "2024-01-02"],
        "('Close', 'AAPL')": [100.0, 101.0],
        "('Volume', 'AAPL')": [1000000, 1100000],
        "('Close', 'MSFT')": [200.0, 201.0],
    })
    
    print("\n" + "🔍 VISUALIZING COLUMN NORMALIZATION")
    print("="*80)
    print("\nINPUT (raw yfinance data):")
    print(raw_df)
    
    # Apply transformation
    result = normalize_column_names(raw_df)
    
    print("\nOUTPUT (normalized):")
    print(result)
    
    # Compare before/after
    comparison = DataFrameComparator.compare(raw_df, result, "Column Normalization")
    DataFrameComparator.print_comparison(comparison)
    
    # Assertions
    assert "date" in result.columns
    assert "close_aapl" in result.columns
    assert "volume_aapl" in result.columns


# ============================================================================
# EXAMPLE 2: Testing Data Quality Fixes with Visual Output
# ============================================================================

@pytest.mark.visual
def test_visualize_data_quality_fixing():
    """See which rows get removed and why"""
    # Create data with quality issues
    dirty_df = pl.DataFrame({
        "ticker": ["AAPL", "AAPL", "MSFT", "TSLA"],
        "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 1)],
        "open": [100.0, None, 200.0, 150.0],
        "high": [101.0, None, 201.0, 151.0],
        "low": [99.0, None, 199.0, 149.0],
        "close": [100.5, None, 200.5, 150.5],
        "volume": [1000000, None, 2000000, 1500000],
    })
    
    print("\n" + "🧹 VISUALIZING DATA QUALITY FIXING")
    print("="*80)
    print("\nBEFORE CLEANING:")
    print(dirty_df)
    print(f"\n📊 Stats:")
    print(f"  - Total rows: {len(dirty_df)}")
    print(f"  - Rows with nulls: {dirty_df.null_count().sum_horizontal()[0]}")
    
    # Apply cleaning
    clean_df = data_quality_fixing(dirty_df)
    
    print("\nAFTER CLEANING:")
    print(clean_df)
    print(f"\n📊 Stats:")
    print(f"  - Total rows: {len(clean_df)}")
    print(f"  - Rows removed: {len(dirty_df) - len(clean_df)}")
    
    # Show which row was removed
    print("\n❌ REMOVED ROWS:")
    removed = dirty_df.join(
        clean_df.with_columns(pl.lit(True).alias("kept")),
        on=["ticker", "date"],
        how="left"
    ).filter(pl.col("kept").is_null())
    print(removed.select(["ticker", "date", "open", "high", "low", "close"]))
    
    # Assertions
    assert len(clean_df) == 3  # One row removed
    assert clean_df.filter(
        (pl.col("ticker") == "AAPL") & 
        (pl.col("date") == date(2024, 1, 2))
    ).height == 0  # The all-null row is gone


# ============================================================================
# EXAMPLE 3: Full Pipeline Visualization
# ============================================================================

@pytest.mark.integration
@pytest.mark.visual
def test_full_pipeline_with_visualization():
    """Run the complete pipeline and see each transformation"""
    # Create realistic raw data
    raw_df = pl.DataFrame({
        "('Date', '')": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "('Open', 'AAPL')": [100.0, None, 102.0],
        "('High', 'AAPL')": [101.0, None, 103.0],
        "('Low', 'AAPL')": [99.0, None, 101.0],
        "('Close', 'AAPL')": [100.5, None, 102.5],
        "('Volume', 'AAPL')": [1000000, None, 1200000],
    })
    
    print("\n" + "🚀 FULL PIPELINE VISUALIZATION")
    
    # Use the visualizer
    viz = PipelineVisualizer()
    
    df = raw_df.clone()
    df = viz.add_step(normalize_column_names, "Step 1: Normalize Columns", df)
    df = viz.add_step(normalize_date_column, "Step 2: Normalize Dates", df)
    df = viz.add_step(data_quality_fixing, "Step 3: Quality Fixes", df)
    
    viz.visualize(verbose=True)
    
    # Save report
    report_path = Path("tests/pipeline_report.json")
    viz.save_report(report_path)
    
    # Final assertions
    assert "date" in df.columns
    assert df["date"].dtype == pl.Date
    assert len(df) == 2  # Removed the all-null row


# ============================================================================
# EXAMPLE 4: Property-Based Testing with Assertions
# ============================================================================

@pytest.mark.unit
def test_data_quality_properties():
    """Test that data quality invariants hold"""
    # Create test data
    df = pl.DataFrame({
        "ticker": ["AAPL", "MSFT", "TSLA"],
        "date": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        "open": [100.0, 200.0, 150.0],
        "high": [101.0, 201.0, 151.0],
        "low": [99.0, 199.0, 149.0],
        "close": [100.5, 200.5, 150.5],
        "volume": [1000000, 2000000, 1500000],
    })
    
    result = data_quality_fixing(df)
    
    # Property 1: No null values in required columns
    assert_no_nulls_in_columns(result, ["ticker", "date"])
    
    # Property 2: Correct data types
    assert_column_types(result, {
        "ticker": pl.Utf8,
        "date": pl.Date,
    })
    
    # Property 3: Prices are reasonable
    assert_values_in_range(result, "open", 0, 10000)
    assert_values_in_range(result, "high", 0, 10000)
    
    print("\n✅ All data quality properties verified!")


# ============================================================================
# EXAMPLE 5: Comparative Testing - Before vs After
# ============================================================================

@pytest.mark.visual
def test_compare_raw_vs_normalized():
    """Side-by-side comparison of raw and normalized data"""
    raw_df = pl.DataFrame({
        "Date": ["2024-01-01", "2024-01-02"],
        "Open": [100.0, 101.0],
        "Close": [100.5, 101.5],
    })
    
    normalized_df = normalize_prices(raw_df)
    
    print("\n" + "📊 RAW vs NORMALIZED COMPARISON")
    print("="*80)
    
    print("\n📥 RAW DATA:")
    print(f"  Columns: {raw_df.columns}")
    print(f"  Dtypes: {raw_df.dtypes}")
    print(raw_df)
    
    print("\n📤 NORMALIZED DATA:")
    print(f"  Columns: {normalized_df.columns}")
    print(f"  Dtypes: {normalized_df.dtypes}")
    print(normalized_df)
    
    # Detailed comparison
    comp = DataFrameComparator.compare(raw_df, normalized_df, "Full Normalization")
    DataFrameComparator.print_comparison(comp)


# ============================================================================
# EXAMPLE 6: Parameterized Testing with Multiple Scenarios
# ============================================================================

@pytest.mark.parametrize("scenario,expected_rows", [
    ("all_valid", 3),
    ("one_null_row", 2),
    ("all_null_rows", 0),
])
def test_multiple_data_quality_scenarios(scenario, expected_rows):
    """Test different data quality scenarios"""
    scenarios = {
        "all_valid": pl.DataFrame({
            "ticker": ["A", "B", "C"],
            "date": [date(2024, 1, i) for i in [1, 2, 3]],
            "close": [100.0, 101.0, 102.0],
        }),
        "one_null_row": pl.DataFrame({
            "ticker": ["A", "B", "C"],
            "date": [date(2024, 1, i) for i in [1, 2, 3]],
            "close": [100.0, None, 102.0],
            "open": [99.0, None, 101.0],
        }),
        "all_null_rows": pl.DataFrame({
            "ticker": ["A", "B"],
            "date": [date(2024, 1, 1), date(2024, 1, 2)],
            "close": [None, None],
            "open": [None, None],
        }),
    }
    
    df = scenarios[scenario]
    result = data_quality_fixing(df)
    
    print(f"\n📝 Scenario: {scenario}")
    print(f"   Expected rows: {expected_rows}")
    print(f"   Actual rows: {len(result)}")
    
    assert len(result) == expected_rows


if __name__ == "__main__":
    # Run all visual tests
    pytest.main([__file__, "-v", "-s", "-m", "visual"])