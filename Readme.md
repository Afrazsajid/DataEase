# Data Processing & Visualization Tool

## 📊 Overview
This is a **Streamlit** application that allows users to **upload, clean, visualize, and export** their data with ease. It supports **CSV** and **Excel** file formats and provides interactive visualizations using **Plotly**.

## 🚀 Features
- **Upload & Preview Data**: Supports CSV and Excel files.
- **Data Cleaning**:
  - Remove duplicate entries.
  - Handle missing values (mean, median, mode, or custom input).
- **Data Visualization**:
  - Bar Chart
  - Line Plot
  - Scatter Plot
- **Export Processed Data**:
  - Download as CSV or Excel.

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Afrazsajid/DataEase/data-processing-tool.git
   cd data-processing-tool
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage
Run the Streamlit app using the command:
```bash
streamlit run app.py
```

## 📌 File Structure
```
📂 data-processing-tool
│── app.py                # Main Streamlit app
│── requirements.txt      # Required dependencies
│── utils/
│   ├── data_processor.py  # Data processing functions
│   ├── visualizer.py      # Visualization functions
```

## 📦 Dependencies
- **Streamlit**
- **Pandas**
- **Plotly**
- **XlsxWriter**

## 🎨 Visualizations
The app provides various visualizations, including:
- **Bar Chart**: Compare categorical data.
- **Line Plot**: Track trends over time.
- **Scatter Plot**: Identify relationships between variables.

## 📝 License
This project is licensed under the MIT License.

## 🤝 Contributing
Feel free to fork the repository and contribute! Pull requests are welcome.

## 📧 Contact
For any issues or suggestions, please open an issue or reach out via email: `afrazsajid55@gmail.com`.

