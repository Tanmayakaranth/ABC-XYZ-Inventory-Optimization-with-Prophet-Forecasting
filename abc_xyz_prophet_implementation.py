"""
Enhanced ABC-XYZ Analysis with Prophet Forecasting for Inventory Optimization
Production-Ready Implementation with Advanced Features
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from datetime import datetime, timedelta
import logging

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ABCXYZAnalyzer:
    """
    Comprehensive ABC-XYZ Analysis with AI-powered forecasting
    """
    
    def __init__(self, file_path, base_year=2024):
        """Initialize the analyzer with dataset path"""
        self.file_path = file_path
        self.base_year = base_year
        self.df = None
        self.month_columns = None  # Will be auto-detected
        self.forecasts = {}
        self.optimization_results = None
        self.full_inventory = None  # NEW: full inventory table
        
    def load_data(self):
        """Load and validate dataset"""
        logger.info("Loading dataset...")
        try:
            self.df = pd.read_csv(self.file_path)
            logger.info(f"Dataset loaded: {self.df.shape[0]} items, {self.df.shape[1]} columns")
            
            # Auto-detect monthly demand columns
            self.month_columns = [col for col in self.df.columns if '_Demand' in col]
            if len(self.month_columns) != 12:
                logger.warning(f"Expected 12 monthly columns, found {len(self.month_columns)}: {self.month_columns}")
            else:
                logger.info(f"Monthly columns detected: {self.month_columns}")
            
            # Validate required columns
            required_cols = ['Item_ID', 'Total_Annual_Units', 'Price_Per_Unit']
            missing_cols = [col for col in required_cols if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def perform_abc_classification(self, threshold_a=80, threshold_b=95):
        """ABC Classification based on Pareto Principle"""
        logger.info("Performing ABC Classification...")
        
        if 'Annual_Consumption_Value' not in self.df.columns:
            self.df['Annual_Consumption_Value'] = (
                self.df['Total_Annual_Units'] * self.df['Price_Per_Unit']
            )
        
        self.df = self.df.sort_values('Annual_Consumption_Value', ascending=False).reset_index(drop=True)
        total_value = self.df['Annual_Consumption_Value'].sum()
        self.df['Value_Percentage'] = (self.df['Annual_Consumption_Value'] / total_value) * 100
        self.df['Cumulative_Percentage'] = self.df['Value_Percentage'].cumsum()
        
        def classify_abc(cum_pct):
            if cum_pct <= threshold_a: return 'A'
            elif cum_pct <= threshold_b: return 'B'
            else: return 'C'
        
        self.df['ABC_Category'] = self.df['Cumulative_Percentage'].apply(classify_abc)
        abc_dist = self.df['ABC_Category'].value_counts().sort_index()
        logger.info(f"ABC Distribution:\n{abc_dist}")
        return self.df
    
    def perform_xyz_classification(self, threshold_x=10, threshold_y=25):
        """XYZ Classification based on Coefficient of Variation"""
        logger.info("Performing XYZ Classification...")
        
        if not self.month_columns:
            raise ValueError("Monthly demand columns not detected.")
        
        self.df['Mean_Monthly_Demand'] = self.df[self.month_columns].mean(axis=1)
        self.df['StdDev_Monthly_Demand'] = self.df[self.month_columns].std(axis=1)
        
        self.df['Coefficient_of_Variation'] = np.where(
            self.df['Mean_Monthly_Demand'] > 0,
            (self.df['StdDev_Monthly_Demand'] / self.df['Mean_Monthly_Demand']) * 100,
            0
        )
        
        def classify_xyz(cv):
            if cv < threshold_x: return 'X'
            elif cv < threshold_y: return 'Y'
            else: return 'Z'
        
        self.df['XYZ_Category'] = self.df['Coefficient_of_Variation'].apply(classify_xyz)
        xyz_dist = self.df['XYZ_Category'].value_counts().sort_index()
        logger.info(f"XYZ Distribution:\n{xyz_dist}")
        return self.df
    
    def create_combined_matrix(self):
        """Create ABC-XYZ combined classification matrix"""
        logger.info("Creating ABC-XYZ Matrix...")
        
        self.df['ABC_XYZ_Category'] = self.df['ABC_Category'] + self.df['XYZ_Category']
        
        policies = {
            'AX': {'service_level': 0.99, 'priority': 'Critical', 'review_freq': 'Daily'},
            'AY': {'service_level': 0.98, 'priority': 'Critical', 'review_freq': 'Daily'},
            'AZ': {'service_level': 0.95, 'priority': 'High', 'review_freq': 'Weekly'},
            'BX': {'service_level': 0.95, 'priority': 'Medium', 'review_freq': 'Weekly'},
            'BY': {'service_level': 0.90, 'priority': 'Medium', 'review_freq': 'Weekly'},
            'BZ': {'service_level': 0.85, 'priority': 'Medium', 'review_freq': 'Bi-weekly'},
            'CX': {'service_level': 0.85, 'priority': 'Low', 'review_freq': 'Monthly'},
            'CY': {'service_level': 0.80, 'priority': 'Low', 'review_freq': 'Monthly'},
            'CZ': {'service_level': 0.75, 'priority': 'Low', 'review_freq': 'Quarterly'}
        }
        
        for key in ['service_level', 'priority', 'review_freq']:
            col_name = key.replace('_', ' ').title().replace(' ', '_')
            self.df[col_name] = self.df['ABC_XYZ_Category'].map(lambda x: policies.get(x, {}).get(key, None))
        
        matrix = pd.crosstab(self.df['ABC_Category'], self.df['XYZ_Category'], margins=True)
        logger.info(f"ABC-XYZ Matrix:\n{matrix}")
        return self.df, policies
    
    def prepare_prophet_data(self, item_id):
        """Prepare time series data for Prophet"""
        item_row = self.df[self.df['Item_ID'] == item_id]
        if item_row.empty:
            raise ValueError(f"Item {item_id} not found")
        
        dates = pd.date_range(start=f'{self.base_year}-01-01', periods=12, freq='MS')
        prophet_df = pd.DataFrame({
            'ds': dates,
            'y': item_row[self.month_columns].values[0]
        })
        return prophet_df
    
    def forecast_item(self, item_id, periods=6, include_uncertainty=True):
        """Forecast demand for a single item using Prophet"""
        try:
            ts_data = self.prepare_prophet_data(item_id)
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                interval_width=0.95 if include_uncertainty else 0.80,
                changepoint_prior_scale=0.05
            )
            model.fit(ts_data)
            future = model.make_future_dataframe(periods=periods, freq='MS')
            forecast = model.predict(future)
            
            item_info = self.df[self.df['Item_ID'] == item_id].iloc[0]
            self.forecasts[item_id] = {
                'model': model,
                'forecast': forecast,
                'historical': ts_data,
                'category': item_info['ABC_XYZ_Category'],
                'cv': item_info['Coefficient_of_Variation'],
                'service_level': item_info['Service_Level']
            }
            return forecast
        except Exception as e:
            logger.error(f"Error forecasting {item_id}: {str(e)}")
            return None
    
    def forecast_priority_items(self, max_items=20, categories=['AX', 'AY', 'BX'], periods=6):
        """Forecast multiple priority items"""
        logger.info(f"Forecasting priority items (categories: {categories})...")
        priority_df = self.df[self.df['ABC_XYZ_Category'].isin(categories)].head(max_items)
        logger.info(f"Selected {len(priority_df)} items for forecasting")
        
        success_count = 0
        for _, row in priority_df.iterrows():
            item_id = row['Item_ID']
            result = self.forecast_item(item_id, periods=periods)
            if result is not None:
                success_count += 1
                if success_count % 5 == 0:
                    logger.info(f"Completed {success_count}/{len(priority_df)} forecasts")
        logger.info(f"Successfully forecasted {success_count} items")
        return self.forecasts
    
    def calculate_inventory_metrics(self, lead_time_days=7):
        """Legacy: only for forecasted items"""
        logger.info("Calculating inventory optimization metrics for forecasted items...")
        results = []
        for item_id, data in self.forecasts.items():
            forecast = data['forecast']
            service_level = data['service_level']
            future_forecast = forecast.tail(6)
            avg_monthly_demand = future_forecast['yhat'].mean()
            avg_daily_demand = avg_monthly_demand / 30
            interval_std = (future_forecast['yhat_upper'] - future_forecast['yhat_lower']) / (2 * 1.96)
            prophet_std = interval_std.mean()
            item_info = self.df[self.df['Item_ID'] == item_id].iloc[0]
            historical_std = item_info['StdDev_Monthly_Demand']
            lead_time_months = lead_time_days / 30.0
            fallback_std = historical_std * np.sqrt(lead_time_months)
            forecast_std = max(prophet_std, fallback_std, 0.1)
            z_score = stats.norm.ppf(service_level)
            safety_stock = z_score * forecast_std
            reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
            annual_demand = avg_monthly_demand * 12
            holding_cost = item_info['Price_Per_Unit'] * 0.25
            ordering_cost = 50
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost) if holding_cost > 0 else 0

            results.append({
                'Item_ID': item_id,
                'ABC_XYZ_Category': data['category'],
                'Service_Level': service_level,
                'CV': data['cv'],
                'Avg_Monthly_Forecast': round(avg_monthly_demand, 2),
                'Avg_Daily_Demand': round(avg_daily_demand, 2),
                'Safety_Stock': round(safety_stock, 2),
                'Reorder_Point': round(reorder_point, 2),
                'EOQ': round(eoq, 2),
                'Lead_Time_Days': lead_time_days
            })
        self.optimization_results = pd.DataFrame(results)
        logger.info(f"Calculated metrics for {len(results)} items")
        return self.optimization_results

    def get_full_inventory_table(self, lead_time_days=7):
        """NEW: Safety stock for ALL items (9 categories)"""
        logger.info("Building full inventory table for all 9 ABC-XYZ categories...")
        results = []

        for _, row in self.df.iterrows():
            item_id = row['Item_ID']
            service_level = row['Service_Level']
            avg_monthly = row['Mean_Monthly_Demand']
            avg_daily = avg_monthly / 30.0

            # Uncertainty
            historical_std = row['StdDev_Monthly_Demand']
            lead_time_months = lead_time_days / 30.0
            fallback_std = historical_std * np.sqrt(lead_time_months)

            if item_id in self.forecasts:
                future = self.forecasts[item_id]['forecast'].tail(6)
                interval_std = (future['yhat_upper'] - future['yhat_lower']) / (2 * 1.96)
                prophet_std = interval_std.mean()
                forecast_std = max(prophet_std, fallback_std, 0.1)
            else:
                forecast_std = max(fallback_std, 0.1)

            z = stats.norm.ppf(service_level)
            safety_stock = z * forecast_std
            reorder_point = (avg_daily * lead_time_days) + safety_stock

            annual_demand = avg_monthly * 12
            holding_cost = row['Price_Per_Unit'] * 0.25
            ordering_cost = 50
            eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost) if holding_cost > 0 else 0

            results.append({
                'Item_ID': item_id,
                'ABC_XYZ_Category': row['ABC_XYZ_Category'],
                'Service_Level': service_level,
                'CV': row['Coefficient_of_Variation'],
                'Avg_Monthly_Demand': round(avg_monthly, 2),
                'Safety_Stock': round(safety_stock, 2),
                'Reorder_Point': round(reorder_point, 2),
                'EOQ': round(eoq, 2)
            })

        self.full_inventory = pd.DataFrame(results)
        logger.info(f"Full inventory table built – {len(self.full_inventory)} rows")
        return self.full_inventory

    def visualize_distributions(self, save_path='abc_xyz_distribution.png'):
        """Create comprehensive distribution visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        abc_counts = self.df['ABC_Category'].value_counts().sort_index()
        colors_abc = ['#2ecc71', '#f39c12', '#e74c3c']
        axes[0, 0].bar(abc_counts.index, abc_counts.values, color=colors_abc, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('ABC Classification Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Category'); axes[0, 0].set_ylabel('Number of Items')
        axes[0, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(abc_counts.values): axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        xyz_counts = self.df['XYZ_Category'].value_counts().sort_index()
        colors_xyz = ['#3498db', '#e67e22', '#9b59b6']
        axes[0, 1].bar(xyz_counts.index, xyz_counts.values, color=colors_xyz, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('XYZ Classification Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Category'); axes[0, 1].set_ylabel('Number of Items')
        axes[0, 1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(xyz_counts.values): axes[0, 1].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        matrix_data = pd.crosstab(self.df['ABC_Category'], self.df['XYZ_Category'])
        sns.heatmap(matrix_data, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 0], cbar_kws={'label': 'Item Count'})
        axes[1, 0].set_title('ABC-XYZ Matrix Heatmap', fontsize=14, fontweight='bold')
        
        abc_values = self.df.groupby('ABC_Category')['Annual_Consumption_Value'].sum().sort_index()
        axes[1, 1].pie(abc_values, labels=abc_values.index, autopct='%1.1f%%', colors=colors_abc, startangle=90)
        axes[1, 1].set_title('Value Distribution by ABC Category', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved distribution visualization: {save_path}")
        return fig
    
    def plot_forecast(self, item_id, save_path=None):
        """Plot forecast for specific item"""
        if item_id not in self.forecasts:
            logger.error(f"No forecast available for {item_id}")
            return None
        
        data = self.forecasts[item_id]
        model = data['model']
        forecast = data['forecast']
        
        fig = model.plot(forecast, figsize=(14, 6))
        plt.title(f'Demand Forecast: {item_id} ({data["category"]}) | CV: {data["cv"]:.2f}%', fontsize=14, fontweight='bold')
        plt.xlabel('Date'); plt.ylabel('Demand (Units)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None: save_path = f'forecast_{item_id}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved forecast plot: {save_path}")
        return fig
    
    def generate_report(self, output_path='abc_xyz_comprehensive_report.csv'):
        """Generate comprehensive final report"""
        logger.info("Generating comprehensive report...")
        if self.optimization_results is not None:
            report_df = self.df.merge(
                self.optimization_results[['Item_ID', 'Avg_Monthly_Forecast', 'Safety_Stock', 'Reorder_Point', 'EOQ']],
                on='Item_ID', how='left'
            )
        else:
            report_df = self.df.copy()
        
        key_cols = ['Item_ID', 'Item_Name', 'Category', 'ABC_Category', 'XYZ_Category', 'ABC_XYZ_Category',
                    'Annual_Consumption_Value', 'Coefficient_of_Variation', 'Service_Level', 'Priority', 'Review_Freq']
        if 'Avg_Monthly_Forecast' in report_df.columns:
            key_cols.extend(['Avg_Monthly_Forecast', 'Safety_Stock', 'Reorder_Point', 'EOQ'])
        
        final_report = report_df[key_cols]
        final_report.to_csv(output_path, index=False)
        logger.info(f"Report saved: {output_path}")
        return final_report
    
    def run_complete_analysis(self, forecast_periods=6, max_forecast_items=20):
        """Execute complete ABC-XYZ analysis workflow"""
        logger.info("="*80)
        logger.info("STARTING COMPLETE ABC-XYZ ANALYSIS")
        logger.info("="*80)
        
        self.load_data()
        self.perform_abc_classification()
        self.perform_xyz_classification()
        self.create_combined_matrix()
        self.forecast_priority_items(max_items=max_forecast_items, periods=forecast_periods)
        if self.forecasts:
            self.calculate_inventory_metrics()
        
        # NEW: Build full inventory table for all 9 categories
        self.get_full_inventory_table()
        
        self.visualize_distributions()
        if self.forecasts:
            for item_id in list(self.forecasts.keys())[:3]:
                self.plot_forecast(item_id)
        
        final_report = self.generate_report()
        
        logger.info("="*80)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("="*80)
        return final_report


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    analyzer = ABCXYZAnalyzer(file_path='abc_xyz_dataset.csv', base_year=2024)
    final_report = analyzer.run_complete_analysis(forecast_periods=6, max_forecast_items=15)
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total Items Analyzed: {len(analyzer.df)}")
    print(f"Items Forecasted: {len(analyzer.forecasts)}")
    print("\nABC-XYZ Distribution:")
    print(analyzer.df['ABC_XYZ_Category'].value_counts().sort_index())
    
    if analyzer.full_inventory is not None:
        print("\nFull Inventory Summary (All 9 Categories):")
        summary = analyzer.full_inventory.groupby('ABC_XYZ_Category').agg(
            Items=('Item_ID', 'count'),
            Avg_Safety_Stock=('Safety_Stock', 'mean'),
            Avg_Reorder_Point=('Reorder_Point', 'mean'),
            Avg_EOQ=('EOQ', 'mean')
        ).round(2)
        print(summary)