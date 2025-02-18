import streamlit as st
import pandas as pd
import io
from utils.data_processor import DataProcessor
from utils.visualizer import Visualizer

def main():
    st.set_page_config(
        page_title="Data Processing & Visualization Tool",
        page_icon="📊",
        layout="wide"
    )

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

            # Export sectionn
            st.header("5. Export Data")
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
