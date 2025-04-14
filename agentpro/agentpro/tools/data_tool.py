import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import json
from typing import Dict, List, Optional, Union, Any
import tempfile
from .base import LLMTool
class DataAnalysisTool(LLMTool):
    name: str = "Data Analysis Tool"
    description: str = "A tool that can analyze data files (CSV, Excel, etc.) and provide insights. It can generate statistics, visualizations, and exploratory data analysis."
    arg: str = "Either a file path or a JSON object with parameters for analysis. If providing a path, supply the full path to the data file. If providing parameters, use the format: {'file_path': 'path/to/file', 'analysis_type': 'basic|correlation|visualization', 'columns': ['col1', 'col2'], 'target': 'target_column'}"
    # Path to the currently loaded dataframe
    _current_file: str = None
    _df: Optional[pd.DataFrame] = None
    def load_data(self, file_path: str) -> str:
        """Load data from the specified file path."""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.csv':
                self._df = pd.read_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                self._df = pd.read_excel(file_path)
            elif file_ext == '.json':
                self._df = pd.read_json(file_path)
            elif file_ext == '.parquet':
                self._df = pd.read_parquet(file_path)
            elif file_ext == '.sql':
                # For SQL files, we expect a SQLite database
                import sqlite3
                conn = sqlite3.connect(file_path)
                self._df = pd.read_sql("SELECT * FROM main_table", conn)
                conn.close()
            else:
                return f"Unsupported file format: {file_ext}. Supported formats: .csv, .xlsx, .xls, .json, .parquet, .sql"
            self._current_file = file_path
            return f"Successfully loaded data from {file_path}. Shape: {self._df.shape}. Columns: {', '.join(self._df.columns.tolist())}"
        except Exception as e:
            return f"Error loading data: {str(e)}"
    def generate_basic_stats(self, columns: Optional[List[str]] = None) -> Dict:
        """Generate basic statistics for the dataframe or specified columns."""
        if self._df is None:
            return "No data loaded. Please load data first."
        try:
            if columns:
                # Filter to only include columns that exist in the dataframe
                valid_columns = [col for col in columns if col in self._df.columns]
                if not valid_columns:
                    return f"None of the specified columns {columns} exist in the dataframe."
                df_subset = self._df[valid_columns]
            else:
                df_subset = self._df
            numeric_stats = df_subset.describe().to_dict()
            null_counts = df_subset.isnull().sum().to_dict()
            categorical_columns = df_subset.select_dtypes(include=['object', 'category']).columns
            unique_counts = {col: df_subset[col].nunique() for col in categorical_columns}
            stats = {
                "shape": self._df.shape,
                "columns": self._df.columns.tolist(),
                "numeric_stats": numeric_stats,
                "null_counts": null_counts,
                "unique_counts": unique_counts
            }
            return stats
        except Exception as e:
            return f"Error generating basic statistics: {str(e)}"
    def generate_correlation_analysis(self, columns: Optional[List[str]] = None) -> Dict:
        """Generate correlation analysis for numeric columns."""
        if self._df is None:
            return "No data loaded. Please load data first."
        try:
            numeric_df = self._df.select_dtypes(include=[np.number])
            if columns:
                # Filter to only include numeric columns that were specified
                valid_columns = [col for col in columns if col in numeric_df.columns]
                if not valid_columns:
                    return f"None of the specified columns {columns} are numeric or exist in the dataframe."
                numeric_df = numeric_df[valid_columns]
            if numeric_df.empty:
                return "No numeric columns found in the dataset for correlation analysis."
            corr_matrix = numeric_df.corr().to_dict()
            corr_df = numeric_df.corr().abs()
            upper_tri = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
            high_corr = [(col1, col2, upper_tri.loc[col1, col2]) 
                         for col1 in upper_tri.index 
                         for col2 in upper_tri.columns 
                         if upper_tri.loc[col1, col2] > 0.7]
            high_corr.sort(key=lambda x: x[2], reverse=True)
            return {"correlation_matrix": corr_matrix, "high_correlations": high_corr}
        except Exception as e:
            return f"Error generating correlation analysis: {str(e)}"
    def generate_visualization(self, viz_type: str, columns: Optional[List[str]] = None, target: Optional[str] = None) -> str:
        """Generate visualization based on the specified type and columns."""
        if self._df is None:
            return "No data loaded. Please load data first."
        try:
            # Create a temporary directory for the visualization
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                output_path = tmp.name
            plt.figure(figsize=(10, 6))
            # Handle different visualization types
            if viz_type == 'histogram':
                if not columns or len(columns) == 0:
                    # If no columns specified, use all numeric columns
                    numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
                    if not numeric_cols:
                        return "No numeric columns found for histogram."
                    # Limit to 4 columns for readability
                    columns = numeric_cols[:4]
                # Filter to valid columns
                valid_columns = [col for col in columns if col in self._df.columns]
                if not valid_columns:
                    return f"None of the specified columns {columns} exist in the dataframe."
                for col in valid_columns:
                    if pd.api.types.is_numeric_dtype(self._df[col]):
                        plt.hist(self._df[col].dropna(), alpha=0.5, label=col)
                plt.legend()
                plt.title(f"Histogram of {', '.join(valid_columns)}")
                plt.tight_layout()
            elif viz_type == 'scatter':
                if not columns or len(columns) < 2:
                    return "Scatter plot requires at least two columns."
                # Check if columns exist
                if columns[0] not in self._df.columns or columns[1] not in self._df.columns:
                    return f"One or more of the specified columns {columns[:2]} do not exist in the dataframe."
                # Create scatter plot
                x_col, y_col = columns[0], columns[1]
                plt.scatter(self._df[x_col], self._df[y_col], alpha=0.5)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.title(f"Scatter Plot: {x_col} vs {y_col}")
                # Color by target if provided
                if target and target in self._df.columns:
                    if pd.api.types.is_numeric_dtype(self._df[target]):
                        scatter = plt.scatter(self._df[x_col], self._df[y_col], 
                                            c=self._df[target], alpha=0.5)
                        plt.colorbar(scatter, label=target)
                    else:
                        # For categorical targets, create multiple scatters
                        categories = self._df[target].unique()
                        for category in categories:
                            mask = self._df[target] == category
                            plt.scatter(self._df.loc[mask, x_col], self._df.loc[mask, y_col], alpha=0.5, label=str(category))
                        plt.legend()
                plt.tight_layout()
            elif viz_type == 'correlation':
                # Generate correlation heatmap
                numeric_df = self._df.select_dtypes(include=[np.number])
                if columns:
                    # Filter to valid numeric columns
                    valid_columns = [col for col in columns if col in numeric_df.columns]
                    if not valid_columns:
                        return f"None of the specified columns {columns} are numeric or exist in the dataframe."
                    numeric_df = numeric_df[valid_columns]
                if numeric_df.empty:
                    return "No numeric columns found for correlation heatmap."
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title("Correlation Heatmap")
                plt.tight_layout()
            elif viz_type == 'boxplot':
                if not columns or len(columns) == 0:
                    # If no columns specified, use all numeric columns
                    numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
                    if not numeric_cols:
                        return "No numeric columns found for boxplot."
                    # Limit to 5 columns for readability
                    columns = numeric_cols[:5]
                # Filter to valid columns
                valid_columns = [col for col in columns if col in self._df.columns]
                if not valid_columns:
                    return f"None of the specified columns {columns} exist in the dataframe."
                # Create boxplot
                self._df[valid_columns].boxplot()
                plt.title("Boxplot of Selected Columns")
                plt.xticks(rotation=45)
                plt.tight_layout()
            elif viz_type == 'pairplot':
                # Create a pair plot for multiple columns
                if not columns or len(columns) < 2:
                    # Use first 4 numeric columns if not specified
                    numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) < 2:
                        return "Not enough numeric columns for a pairplot."
                    columns = numeric_cols[:min(4, len(numeric_cols))]
                # Filter to valid columns
                valid_columns = [col for col in columns if col in self._df.columns]
                if len(valid_columns) < 2:
                    return f"Not enough valid columns in {columns} for a pairplot."
                # Use seaborn pairplot
                plt.close()  # Close previous figure
                # Create a temporary directory for the visualization
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    output_path = tmp.name
                if target and target in self._df.columns:
                    g = sns.pairplot(self._df[valid_columns + [target]], hue=target, height=2.5)
                else:
                    g = sns.pairplot(self._df[valid_columns], height=2.5)
                plt.suptitle("Pair Plot of Selected Features", y=1.02)
                plt.tight_layout()
            else:
                return f"Unsupported visualization type: {viz_type}. Supported types: histogram, scatter, correlation, boxplot, pairplot"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"Visualization saved to: {output_path}"
        except Exception as e:
            return f"Error generating visualization: {str(e)}"
    def generate_data_insights(self) -> str:
        """Generate AI-powered insights about the data."""
        if self._df is None:
            return "No data loaded. Please load data first."
        try:
            # Get a sample and info about the data to send to the LLM
            df_sample = self._df.head(5).to_string()
            df_info = {
                "shape": self._df.shape,
                "columns": self._df.columns.tolist(),
                "dtypes": {col: str(self._df[col].dtype) for col in self._df.columns},
                "missing_values": self._df.isnull().sum().to_dict(),
                "numeric_stats": self._df.describe().to_dict() if not self._df.select_dtypes(include=[np.number]).empty else {},
            }
            
            prompt = f"""
            Analyze this dataset and provide key insights.
            
            Dataset Sample:
            {df_sample}
            
            Dataset Info:
            {json.dumps(df_info, indent=2)}
            
            Your task:
            1. Identify the dataset type and potential use cases
            2. Summarize the basic characteristics (rows, columns, data types)
            3. Highlight key statistics and distributions
            4. Point out missing data patterns if any
            5. Suggest potential relationships or correlations worth exploring
            6. Recommend next steps for deeper analysis
            7. Note any data quality issues or anomalies
            
            Provide a comprehensive but concise analysis with actionable insights.
            """
            
        #    response = self.client.chat.completions.create(
        #        model="gpt-4",
        #        messages=[
        #            {"role": "system", "content": "You are a data science expert specializing in exploratory data analysis and deriving insights from datasets."},
        #            {"role": "user", "content": prompt}
        #        ],
        #        max_tokens=3000)
        #    return response.choices[0].message.content
            openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
            model_name = os.environ.get("MODEL_NAME", "gpt-4")  # Default to gpt-4 if MODEL_NAME is not set
            try:
                if openrouter_api_key:
                    print(f"Using OpenRouter with model: {model_name} for data insights")
                    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a data science expert specializing in exploratory data analysis and deriving insights from datasets."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=3000)
                else: # Fall back to default OpenAI client
                    print("OpenRouter API key not found, using default OpenAI client with gpt-4")
                    response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a data science expert specializing in exploratory data analysis and deriving insights from datasets."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=3000)
                return response.choices[0].message.content
            except Exception as e:
                print(f"Error with OpenRouter: {e}")
                print("Falling back to default OpenAI client with gpt-4")
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a data science expert specializing in exploratory data analysis and deriving insights from datasets."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=3000)
                    return response.choices[0].message.content
                except Exception as e2:
                    return f"Error generating data insights with fallback model: {str(e2)}"
        except Exception as e:
            return f"Error analyzing data for insights: {str(e)}"            
    def run(self, prompt: Union[str, Dict]) -> str:
        """Run the data analysis tool."""
        print(f"Calling Data Analysis Tool with prompt: {prompt}")
        try: # If prompt is a string, try to parse it as JSON or treat it as a file path
            if isinstance(prompt, str):
                try:
                    params = json.loads(prompt)
                except json.JSONDecodeError: # Treat as file path
                    return self.load_data(prompt)
            else:
                params = prompt
            # Handle different parameter options
            if 'file_path' in params:
                file_path = params['file_path']
                # Load the data first
                load_result = self.load_data(file_path)
                if "Successfully" not in load_result:
                    return load_result
            # If no analysis type is specified, generate insights
            if 'analysis_type' not in params:
                return self.generate_data_insights()
            analysis_type = params['analysis_type'].lower()
            columns = params.get('columns', None)
            target = params.get('target', None)
            if analysis_type == 'basic':
                stats = self.generate_basic_stats(columns)
                return json.dumps(stats, indent=2)
            elif analysis_type == 'correlation':
                corr_analysis = self.generate_correlation_analysis(columns)
                return json.dumps(corr_analysis, indent=2)
            elif analysis_type == 'visualization':
                viz_type = params.get('viz_type', 'histogram')
                return self.generate_visualization(viz_type, columns, target)
            elif analysis_type == 'insights':
                return self.generate_data_insights()
            else:
                return f"Unsupported analysis type: {analysis_type}. Supported types: basic, correlation, visualization, insights"
        except Exception as e:
            return f"Error executing data analysis: {str(e)}"
