import streamlit as st
import pandas as pd
import io
# from utils.data_processor import DataProcessor
# from utils.visualizer import Visualizer




import numpy as np
from typing import Union, Optional



import google.generativeai as genai


import os

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Optional

class DataChatAnalyzer:
    def __init__(self):
        api_key = "AIzaSyA-suP5ATD-ZTgzadlVeU2GSNKLiFRVwH0"
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.chat = self.model.start_chat(history=[])

    def analyze_data(self, df: pd.DataFrame, user_query: str) -> str:
        """Generate insights about the data based on user query."""
        try:
            # Create a data summary
            data_info = (
                f"DataFrame Info:\n"
                f"- Shape: {df.shape}\n"
                f"- Columns: {', '.join(df.columns)}\n"
                f"- Data Types:\n{df.dtypes.to_string()}\n\n"
                f"First few rows:\n{df.head().to_string()}\n\n"
                f"Summary Statistics:\n{df.describe().to_string()}\n"
            )

            # Prepare the prompt
            prompt = f"""
            As a friendly data analysis assistant developed by afraz , help me analyze this dataset:
            
            {data_info}
            
            my Question: {user_query}
            
            Please provide a clear and concise analysis based on the data and the  question just give simple answer.
            """

            # Get response from Gemini
            response = self.chat.send_message(prompt)
            return response.text
            
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

    def get_chat_history(self):
        """Return the chat history."""
        return [(msg.parts[0].text, msg.role) for msg in self.chat.history]






class DataProcessor:
    @staticmethod
    def load_data(file) -> pd.DataFrame:
        """Load data from uploaded file."""
        try:
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                return pd.read_excel(file)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")

    @staticmethod
    def get_file_info(file) -> dict:
        """Get file metadata."""
        return {
            "filename": file.name,
            "size": f"{file.size / 1024:.2f} KB",
            "format": file.name.split('.')[-1].upper()
        }

    @staticmethod
    def clean_data(df: pd.DataFrame,
                   remove_duplicates: bool = False,
                   fill_missing: Optional[str] = None,
                   fill_value: Union[str, float, None] = None) -> pd.DataFrame:
        """Clean the dataset based on specified options."""
        if remove_duplicates:
            df = df.drop_duplicates()

        if fill_missing:
            for column in df.columns:
                if df[column].isnull().any():
                    if fill_missing == 'mean' and pd.api.types.is_numeric_dtype(df[column]):
                        df[column] = df[column].fillna(df[column].mean())
                    elif fill_missing == 'median' and pd.api.types.is_numeric_dtype(df[column]):
                        df[column] = df[column].fillna(df[column].median())
                    elif fill_missing == 'mode':
                        df[column] = df[column].fillna(df[column].mode()[0])
                    elif fill_missing == 'custom' and fill_value is not None:
                        df[column] = df[column].fillna(fill_value)

        return df

    @staticmethod
    def export_data(df: pd.DataFrame, format: str) -> Union[str, bytes]:
        """Export data in specified format."""
        if format == 'csv':
            return df.to_csv(index=False)
        elif format == 'excel':
            return df.to_excel(index=False, engine='xlsxwriter')
        else:
            raise ValueError("Unsupported export format")





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
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#FAFAFA'
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
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#FAFAFA'
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
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#FAFAFA'
        )
        return fig


def main():
    st.set_page_config(
        page_title="Data Processing & Visualization Tool",
        page_icon="📊",
        layout="wide"
    )

    # Initialize session state for chat
    if 'chat_analyzer' not in st.session_state:
        st.session_state.chat_analyzer = DataChatAnalyzer()
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []

    st.title("📊 Data Processing & Visualization Tool")
    st.write("Upload, clean, visualize, and export your data with ease!")

    # File upload section
    st.header("1. Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls']
    )

    if uploaded_file is not None:
        try:
            # Display file info
            file_info = DataProcessor.get_file_info(uploaded_file)
            st.write("File Information:")
            col1, col2, col3 = st.columns(3)
            col1.metric("Filename", file_info["filename"])
            col2.metric("Size", file_info["size"])
            col3.metric("Format", file_info["format"])

            # Load and preview data
            df = DataProcessor.load_data(uploaded_file)

            st.header("2. Data Preview")
            preview_cols = st.multiselect(
                "Select columns to preview",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:5]
            )
            st.dataframe(df[preview_cols].head())

            # Data cleaning options
            st.header("3. Data Cleaning")
            with st.expander("Clean your data"):
                remove_duplicates = st.checkbox("Remove duplicate entries")

                fill_missing = st.selectbox(
                    "Handle missing values",
                    options=[None, 'mean', 'median', 'mode', 'custom']
                )

                fill_value = None
                if fill_missing == 'custom':
                    fill_value = st.text_input("Enter custom value for missing data")

                if st.button("Apply Cleaning"):
                    df = DataProcessor.clean_data(
                        df,
                        remove_duplicates,
                        fill_missing,
                        fill_value
                    )
                    st.success("Data cleaning completed!")
                    st.dataframe(df.head())

            # Visualization section
            st.header("4. Data Visualization")
            with st.expander("Create visualizations"):
                viz_type = st.selectbox(
                    "Select visualization type",
                    options=['Bar Chart', 'Line Plot', 'Scatter Plot']
                )

                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

                if viz_type == 'Bar Chart':
                    x_col = st.selectbox("Select X-axis column", options=df.columns.tolist())
                    y_col = st.selectbox("Select Y-axis column", options=numeric_cols)
                    title = st.text_input("Enter chart title", "Bar Chart")
                    fig = Visualizer.create_bar_chart(df, x_col, y_col, title)
                    st.plotly_chart(fig, use_container_width=True)

                elif viz_type == 'Line Plot':
                    x_col = st.selectbox("Select X-axis column", options=df.columns.tolist())
                    y_cols = st.multiselect("Select Y-axis column(s)", options=numeric_cols)
                    title = st.text_input("Enter chart title", "Line Plot")
                    if y_cols:
                        fig = Visualizer.create_line_plot(df, x_col, y_cols, title)
                        st.plotly_chart(fig, use_container_width=True)

                elif viz_type == 'Scatter Plot':
                    x_col = st.selectbox("Select X-axis column", options=numeric_cols)
                    y_col = st.selectbox("Select Y-axis column", options=numeric_cols)
                    color_col = st.selectbox("Select color column (optional)",
                                              options=[None] + df.columns.tolist())
                    title = st.text_input("Enter chart title", "Scatter Plot")
                    fig = Visualizer.create_scatter_plot(df, x_col, y_col, color_col, title)
                    st.plotly_chart(fig, use_container_width=True)

            # Add new Data Chat Analysis section
            st.header("5. Data Analysis Chat")
            with st.expander("Chat with your data"):
                if 'df' in locals():  # Check if data is loaded
                    st.write("Ask questions about your data and get AI-powered insights!")

                    # Display chat history
                    for message in st.session_state.chat_messages:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])

                    # Chat input
                    if prompt := st.chat_input("Ask about your data..."):
                        # Display user message
                        with st.chat_message("user"):
                            st.write(prompt)
                        st.session_state.chat_messages.append({"role": "user", "content": prompt})

                        # Get and display assistant response
                        with st.chat_message("assistant"):
                            response = st.session_state.chat_analyzer.analyze_data(df, prompt)
                            st.write(response)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response})

                else:
                    st.warning("Please upload and process data first to use the chat feature.")

            # Export section (now section 6)
            st.header("6. Export Data")
            with st.expander("Export your data"):
                export_format = st.selectbox(
                    "Select export format",
                    options=['csv', 'excel']
                )

                if st.button("Export Data"):
                    try:
                        if export_format == 'csv':
                            csv = DataProcessor.export_data(df, 'csv')
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="processed_data.csv",
                                mime="text/csv"
                            )
                        else:
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                df.to_excel(writer, index=False)
                            st.download_button(
                                label="Download Excel",
                                data=buffer.getvalue(),
                                file_name="processed_data.xlsx",
                                mime="application/vnd.ms-excel"
                            )
                    except Exception as e:
                        st.error(f"Error exporting data: {str(e)}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()