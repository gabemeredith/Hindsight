# tests/test_utils.py
"""
Utilities for visualizing and debugging data pipeline transformations.
"""

import polars as pl
from typing import Callable, Dict, Any
from pathlib import Path
import json


class DataFrameComparator:
    """Compare DataFrames before and after transformations"""
    
    @staticmethod
    def compare(
        before: pl.DataFrame,
        after: pl.DataFrame,
        name: str = "Transformation"
    ) -> Dict[str, Any]:
        """
        Compare two DataFrames and return a summary of changes.
        
        Returns
        -------
        dict with keys:
            - row_count_change
            - column_changes
            - dtype_changes
            - null_count_changes
            - summary
        """
        comparison = {
            "name": name,
            "row_count_before": len(before),
            "row_count_after": len(after),
            "row_count_change": len(after) - len(before),
            "columns_before": before.columns,
            "columns_after": after.columns,
            "columns_added": list(set(after.columns) - set(before.columns)),
            "columns_removed": list(set(before.columns) - set(after.columns)),
            "dtype_changes": {},
            "null_count_changes": {},
        }
        
        # Check dtype changes for common columns
        common_cols = set(before.columns) & set(after.columns)
        for col in common_cols:
            before_dtype = before[col].dtype
            after_dtype = after[col].dtype
            if before_dtype != after_dtype:
                comparison["dtype_changes"][col] = {
                    "before": str(before_dtype),
                    "after": str(after_dtype)
                }
            
            # Check null count changes
            before_nulls = before[col].null_count()
            after_nulls = after[col].null_count()
            if before_nulls != after_nulls:
                comparison["null_count_changes"][col] = {
                    "before": before_nulls,
                    "after": after_nulls,
                    "change": after_nulls - before_nulls
                }
        
        return comparison
    
    @staticmethod
    def print_comparison(comparison: Dict[str, Any]) -> None:
        """Pretty print the comparison results"""
        print("\n" + "="*80)
        print(f"📊 {comparison['name']}")
        print("="*80)
        
        # Row changes
        print(f"\n📏 Rows: {comparison['row_count_before']} → {comparison['row_count_after']}", end="")
        if comparison['row_count_change'] != 0:
            sign = "+" if comparison['row_count_change'] > 0 else ""
            print(f" ({sign}{comparison['row_count_change']})")
        else:
            print(" (no change)")
        
        # Column changes
        if comparison['columns_added']:
            print(f"\n➕ Columns added: {comparison['columns_added']}")
        if comparison['columns_removed']:
            print(f"\n➖ Columns removed: {comparison['columns_removed']}")
        
        # Dtype changes
        if comparison['dtype_changes']:
            print("\n🔄 Data type changes:")
            for col, change in comparison['dtype_changes'].items():
                print(f"   {col}: {change['before']} → {change['after']}")
        
        # Null count changes
        if comparison['null_count_changes']:
            print("\n🔍 Null count changes:")
            for col, change in comparison['null_count_changes'].items():
                sign = "+" if change['change'] > 0 else ""
                print(f"   {col}: {change['before']} → {change['after']} ({sign}{change['change']})")
        
        print("="*80 + "\n")


class PipelineVisualizer:
    """Visualize an entire data pipeline step by step"""
    
    def __init__(self):
        self.steps = []
    
    def add_step(
        self,
        func: Callable,
        name: str,
        df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Apply a transformation and record it for visualization.
        
        Parameters
        ----------
        func : callable
            Transformation function to apply
        name : str
            Name of the transformation step
        df : pl.DataFrame
            Input DataFrame
        
        Returns
        -------
        pl.DataFrame
            Transformed DataFrame
        """
        before = df.clone()
        after = func(df)
        
        comparison = DataFrameComparator.compare(before, after, name)
        self.steps.append({
            "name": name,
            "comparison": comparison,
            "before_shape": before.shape,
            "after_shape": after.shape,
            "before_sample": before.head(3),
            "after_sample": after.head(3),
        })
        
        return after
    
    def visualize(self, verbose: bool = True) -> None:
        """Print visualization of all pipeline steps"""
        print("\n" + "🔧 "*40)
        print("DATA PIPELINE VISUALIZATION")
        print("🔧 "*40 + "\n")
        
        for i, step in enumerate(self.steps, 1):
            print(f"\n{'='*80}")
            print(f"STEP {i}: {step['name']}")
            print(f"{'='*80}")
            
            DataFrameComparator.print_comparison(step['comparison'])
            
            if verbose:
                print("📄 Sample Data BEFORE:")
                print(step['before_sample'])
                print("\n📄 Sample Data AFTER:")
                print(step['after_sample'])
                print()
    
    def save_report(self, filepath: Path) -> None:
        """Save pipeline report to JSON file"""
        report = {
            "total_steps": len(self.steps),
            "steps": []
        }
        
        for step in self.steps:
            report["steps"].append({
                "name": step["name"],
                "shape_before": step["before_shape"],
                "shape_after": step["after_shape"],
                "comparison": {
                    k: v for k, v in step["comparison"].items()
                    if k not in ["columns_before", "columns_after"]  # Skip large lists
                }
            })
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Report saved to: {filepath}")


def visualize_pipeline(df: pl.DataFrame) -> pl.DataFrame:
    """
    Run the full normalization pipeline with visualization.
    
    Usage in tests:
        from test_utils import visualize_pipeline
        result = visualize_pipeline(raw_df)
    """
    from factorlabs.data.ingest_yf import (
        normalize_column_names,
        normalize_date_column,
        data_quality_fixing
    )
    
    viz = PipelineVisualizer()
    
    df = viz.add_step(normalize_column_names, "Normalize Column Names", df)
    df = viz.add_step(normalize_date_column, "Normalize Date Column", df)
    df = viz.add_step(data_quality_fixing, "Data Quality Fixing", df)
    
    viz.visualize(verbose=True)
    
    return df


# Assertion helpers for data quality
def assert_no_nulls_in_columns(df: pl.DataFrame, columns: list) -> None:
    """Assert that specified columns have no null values"""
    for col in columns:
        null_count = df[col].null_count()
        assert null_count == 0, f"Column '{col}' has {null_count} null values"


def assert_column_types(df: pl.DataFrame, type_map: Dict[str, type]) -> None:
    """Assert that columns have expected data types"""
    for col, expected_type in type_map.items():
        actual_type = df[col].dtype
        assert actual_type == expected_type, \
            f"Column '{col}' has type {actual_type}, expected {expected_type}"


def assert_no_duplicate_keys(df: pl.DataFrame, key_cols: list) -> None:
    """Assert that there are no duplicate combinations of key columns"""
    duplicates = df.group_by(key_cols).agg(pl.count().alias("count")).filter(pl.col("count") > 1)
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate key combinations"


def assert_values_in_range(df: pl.DataFrame, col: str, min_val: float, max_val: float) -> None:
    """Assert that all values in a column are within a range"""
    out_of_range = df.filter(
        (pl.col(col) < min_val) | (pl.col(col) > max_val)
    )
    assert len(out_of_range) == 0, \
        f"Column '{col}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]"