import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Optional

class Visualizer:
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, 
                        x_column: str, 
                        y_column: str, 
                        title: Optional[str] = None) -> go.Figure:
        """Create an interactive bar chart."""
        fig = px.bar(df, x=x_column, y=y_column, title=title)
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column,
            template='plotly_white'
        )
        return fig

    @staticmethod
    def create_line_plot(df: pd.DataFrame, 
                        x_column: str, 
                        y_columns: List[str], 
                        title: Optional[str] = None) -> go.Figure:
        """Create an interactive line plot."""
        fig = px.line(df, x=x_column, y=y_columns, title=title)
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=', '.join(y_columns),
            template='plotly_white'
        )
        return fig

    @staticmethod
    def create_scatter_plot(df: pd.DataFrame, 
                          x_column: str, 
                          y_column: str, 
                          color_column: Optional[str] = None,
                          title: Optional[str] = None) -> go.Figure:
        """Create an interactive scatter plot."""
        fig = px.scatter(df, x=x_column, y=y_column, 
                        color=color_column if color_column else None,
                        title=title)
        fig.update_layout(
            xaxis_title=x_column,
            yaxis_title=y_column,
            template='plotly_white'
        )
        return fig
