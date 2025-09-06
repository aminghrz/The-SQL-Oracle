import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
import json
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class ChartGenerator:
    def __init__(self):
        # self.chart_types = {
        #     'bar': self._create_bar_chart,
        #     'line': self._create_line_chart,
        #     'pie': self._create_pie_chart,
        #     'scatter': self._create_scatter_plot,
        #     'table': self._create_table
        # }
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL")
        )
        
    def generate_visualization(self, data: pd.DataFrame, 
                            intent: Any) -> Optional[Dict[str, Any]]:
        """Generate appropriate visualization based on data and intent"""
        if data.empty:
            return None
        
        # Determine best chart type and parameters using LLM
        chart_type, viz_score, parameters = self._select_chart_type(data, intent)
        
        if viz_score < 0.6:
            logger.info(f"Visualization score too low: {viz_score}")
            return None
        
        try:
            # Generate chart with parameters
            fig = self._create_chart_with_params(chart_type, data, parameters)
            
            # Convert to JSON for frontend
            return {
                'type': chart_type,
                'figure': fig.to_json(),
                'score': viz_score,
                'parameters': parameters
            }
            
        except Exception as e:
            logger.error(f"Failed to generate {chart_type} chart: {e}")
            return None

    def _create_chart_with_params(self, chart_type: str, data: pd.DataFrame, 
                                params: Dict[str, Any]) -> go.Figure:
        """Create chart using LLM-provided parameters"""
        if chart_type == 'bar':
            return self._create_bar_chart_with_params(data, params)
        elif chart_type == 'line':
            return self._create_line_chart_with_params(data, params)
        elif chart_type == 'pie':
            return self._create_pie_chart_with_params(data, params)
        elif chart_type == 'scatter':
            return self._create_scatter_plot_with_params(data, params)
        else:
            return self._create_table(data, {})
    
    def _select_chart_type(self, data: pd.DataFrame, intent: Any) -> Tuple[str, float, Dict[str, Any]]:
        """Select the best chart type and parameters using LLM"""
        try:
            # Prepare data analysis
            analysis = self._analyze_data_for_llm(data)
            
            prompt = f"""
    You are an expert data analyst. Your task is to select the most appropriate visualization type and parameters for the given data and query intent.

    Query Intent: {intent.intent_type.value}
    Expected Result Type: {intent.expected_result_type}

    Data Analysis:
    - Shape: {analysis['shape']['rows']} rows, {analysis['shape']['columns']} columns
    - Column Types: {json.dumps(analysis['column_types'])}
    - Numeric Columns: {analysis['numeric_columns']}
    - Categorical Columns: {analysis['categorical_columns']}

    Categorical Column Cardinalities:
    {json.dumps(analysis['cardinalities'], indent=2)}

    Sample Data (top 10 rows):
    {analysis['sample_data']}

    Available chart types:
    - bar: standard bar chart
    - line: line chart for trends
    - pie: pie chart for proportions
    - scatter: scatter plot for relationships
    - table: data table

    Return a JSON object with:
    {{
        "chart_type": "one of: bar, line, pie, scatter, table",
        "confidence": float between 0.0 and 1.0,
        "reasoning": "brief explanation",
        "parameters": {{
            "x": "column name for x-axis",
            "y": "column name for y-axis",
            "color": "column name for grouping/color (optional)",
            "title": "chart title",
            "barmode": "group or stack (for bar charts only)",
            "orientation": "v or h (for bar charts only)",
            "labels": {{"column_name": "display_name"}},
            "hover_data": ["additional columns to show on hover"]
        }}
    }}

    For stacked bar charts showing breakdowns by time and category, set:
    - x: time/date column
    - y: value column
    - color: category column
    - barmode: "stack"
    """

            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are a data visualization expert who selects the most appropriate chart type and parameters."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=400
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Parse response
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            result = json.loads(response_text.strip())
            
            chart_type = result.get('chart_type', 'table')
            confidence = float(result.get('confidence', 0.7))
            parameters = result.get('parameters', {})
            reasoning = result.get('reasoning', '')
            
            logger.info(f"LLM selected {chart_type} chart (confidence: {confidence:.2f}): {reasoning}")
            logger.debug(f"Chart parameters: {json.dumps(parameters, indent=2)}")
            
            # Calculate final viz score
            num_rows = len(data)
            data_quality = 1.0 if 0 < num_rows < 1000 else 0.5
            row_count_score = 1.0 if 1 < num_rows < 1000 else 0.5
            
            viz_score = (
                0.4 * data_quality +
                0.3 * row_count_score +
                0.3 * confidence
            )
            
            return chart_type, viz_score, parameters
            
        except Exception as e:
            logger.error(f"LLM chart selection failed: {e}")
            # Fallback to simple logic
            return self._fallback_chart_selection(data, intent) + ({},)
    
    def _analyze_data_for_llm(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data characteristics for LLM"""
        analysis = {
            'shape': {
                'rows': len(data),
                'columns': len(data.columns)
            },
            'column_types': {},
            'numeric_columns': [],
            'categorical_columns': [],
            'cardinalities': {},
            'sample_data': ''
        }
        
        # Analyze columns
        for col in data.columns:
            dtype = str(data[col].dtype)
            analysis['column_types'][col] = dtype
            
            if data[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                analysis['numeric_columns'].append(col)
            else:
                analysis['categorical_columns'].append(col)
                
                # Get cardinality and unique values for categorical columns
                unique_values = data[col].unique()
                cardinality = len(unique_values)
                analysis['cardinalities'][col] = {
                    'count': cardinality,
                    'values': list(unique_values) if cardinality <= 20 else f"Too many ({cardinality}) to list"
                }
        
        # Get sample data (top 10 rows)
        sample_df = data.head(10)
        analysis['sample_data'] = sample_df.to_string()
        
        return analysis
    
    def _fallback_chart_selection(self, data: pd.DataFrame, intent: Any) -> Tuple[str, float]:
        """Simple fallback chart selection"""
        num_rows = len(data)
        numeric_cols = data.select_dtypes(include=['number']).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Simple rules
        if intent.intent_type.value == 'trend' and self._has_time_column(data):
            return 'line', 0.7
        elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1 and num_rows <= 50:
            return 'bar', 0.7
        elif num_rows <= 10 and len(numeric_cols) >= 1:
            return 'pie', 0.6
        else:
            return 'table', 0.6
    
    def _has_time_column(self, data: pd.DataFrame) -> bool:
        """Check if dataframe has a time-based column"""
        for col in data.columns:
            if data[col].dtype == 'datetime64[ns]':
                return True
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month']):
                return True
        return False
    
    def _create_bar_chart(self, data: pd.DataFrame, intent: Any) -> go.Figure:
        """Create a bar chart"""
        # Identify x and y columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            x_col = categorical_cols[0]
            y_col = numeric_cols[0]
            
            # Check if we have a second categorical column for grouping
            if len(categorical_cols) > 1:
                # This is a grouped bar chart
                color_col = categorical_cols[1]
                fig = px.bar(data, x=x_col, y=y_col, color=color_col,
                           title=f"{y_col} by {x_col} and {color_col}",
                           barmode='group')
            else:
                fig = px.bar(data, x=x_col, y=y_col,
                           title=f"{y_col} by {x_col}")
        else:
            # Fallback to first two columns
            fig = px.bar(data, x=data.columns[0], y=data.columns[1])
            
        fig.update_layout(showlegend=True)
        return fig
    
    def _create_line_chart(self, data: pd.DataFrame, intent: Any) -> go.Figure:
        """Create a line chart"""
        # Find time column
        time_col = None
        for col in data.columns:
            if self._has_time_column(pd.DataFrame(data[col])):
                time_col = col
                break
                
        if not time_col:
            time_col = data.columns[0]
            
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            y_col = numeric_cols[0]
            fig = px.line(data, x=time_col, y=y_col,
                        title=f"{y_col} over {time_col}")
        else:
            fig = px.line(data, x=time_col, y=data.columns[1])
            
        return fig
    
    def _create_pie_chart(self, data: pd.DataFrame, intent: Any) -> go.Figure:
        """Create a pie chart"""
        # Need a category and a value column
        if len(data.columns) >= 2:
            labels_col = data.columns[0]
            values_col = data.columns[1]
            
            fig = px.pie(data, names=labels_col, values=values_col,
                       title=f"Distribution of {values_col}")
        else:
            # Single column - use value counts
            fig = px.pie(values=data.iloc[:, 0].value_counts().values,
                       names=data.iloc[:, 0].value_counts().index)
                       
        return fig
    
    def _create_scatter_plot(self, data: pd.DataFrame, intent: Any) -> go.Figure:
        """Create a scatter plot"""
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) >= 2:
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            fig = px.scatter(data, x=x_col, y=y_col,
                           title=f"{y_col} vs {x_col}")
        else:
            # Fallback
            fig = px.scatter(data, x=data.columns[0], y=data.columns[1])
            
        return fig
    
    def _create_table(self, data: pd.DataFrame, intent: Any) -> go.Figure:
        """Create a table visualization"""
        # Limit rows for display
        display_data = data.head(100)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(display_data.columns),
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[display_data[col] for col in display_data.columns],
                fill_color='lavender',
                align='left'
            )
        )])
        
        fig.update_layout(title="Query Results")
        return fig
    
    def _create_bar_chart_with_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create bar chart with LLM-provided parameters"""
        x_col = params.get('x')
        y_col = params.get('y')
        color_col = params.get('color')
        title = params.get('title', 'Query Results')
        barmode = params.get('barmode', 'group')
        orientation = params.get('orientation', 'v')
        labels = params.get('labels', {})
        hover_data = params.get('hover_data', [])
        
        # Validate columns exist
        if not x_col or x_col not in data.columns:
            x_col = data.columns[0]
        if not y_col or y_col not in data.columns:
            y_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        # Create figure based on parameters
        if color_col and color_col in data.columns:
            # Grouped/stacked bar chart
            fig = px.bar(data, 
                        x=x_col, 
                        y=y_col, 
                        color=color_col,
                        title=title,
                        barmode=barmode,
                        orientation=orientation,
                        labels=labels,
                        hover_data=hover_data)
        else:
            # Simple bar chart
            fig = px.bar(data, 
                        x=x_col, 
                        y=y_col,
                        title=title,
                        orientation=orientation,
                        labels=labels,
                        hover_data=hover_data)
        
        fig.update_layout(showlegend=True)
        return fig

    def _create_line_chart_with_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create line chart with LLM-provided parameters"""
        x_col = params.get('x', data.columns[0])
        y_col = params.get('y', data.columns[1] if len(data.columns) > 1 else data.columns[0])
        color_col = params.get('color')
        title = params.get('title', 'Query Results')
        labels = params.get('labels', {})
        
        if color_col and color_col in data.columns:
            fig = px.line(data, x=x_col, y=y_col, color=color_col, 
                        title=title, labels=labels)
        else:
            fig = px.line(data, x=x_col, y=y_col, 
                        title=title, labels=labels)
        
        return fig

    def _create_pie_chart_with_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create pie chart with LLM-provided parameters"""
        names_col = params.get('x', data.columns[0])
        values_col = params.get('y', data.columns[1] if len(data.columns) > 1 else data.columns[0])
        title = params.get('title', 'Query Results')
        
        fig = px.pie(data, names=names_col, values=values_col, title=title)
        return fig

    def _create_scatter_plot_with_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> go.Figure:
        """Create scatter plot with LLM-provided parameters"""
        x_col = params.get('x', data.columns[0])
        y_col = params.get('y', data.columns[1] if len(data.columns) > 1 else data.columns[0])
        color_col = params.get('color')
        size_col = params.get('size')
        title = params.get('title', 'Query Results')
        labels = params.get('labels', {})
        
        kwargs = {
            'x': x_col,
            'y': y_col,
            'title': title,
            'labels': labels
        }
        
        if color_col and color_col in data.columns:
            kwargs['color'] = color_col
        if size_col and size_col in data.columns:
            kwargs['size'] = size_col
            
        fig = px.scatter(data, **kwargs)
        return fig