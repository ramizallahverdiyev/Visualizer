import streamlit as st
import pandas as pd
import plotly.express as px

# Page Configurations
st.set_page_config(page_title="CSV Data Visualizer", layout="wide")

# Title
st.title("📊Data Visualizer")

# Function to get the stored data and column types from session state
def get_data():
    df_ = st.session_state.df
    categorical_cols_ = st.session_state.categorical_cols
    numerical_cols_ = st.session_state.numerical_cols
    return df_, numerical_cols_, categorical_cols_

# Function to add a new graph type
def add_graph():
    st.session_state.graphs.append("Count Plot")  # Default to "Count Plot"

# Function to generate a graph based on type
def generate_graph(graph_type, graph_index,filtered_df):
    df, numerical_cols, categorical_cols = get_data()

    if graph_type == "Count Plot":
        selected_col = st.selectbox(f"Select Categorical Column for Count Plot",
                                    categorical_cols, key=f"count_{graph_index}")
        if selected_col:
            count_data = filtered_df[selected_col].value_counts().reset_index()
            count_data.columns = [selected_col, "Count"]
            fig = px.bar(count_data, x=selected_col, y="Count", title=f"Count Plot of {selected_col}")
            st.plotly_chart(fig, key=f"plotly_chart_{graph_index}")

    elif graph_type == "Pie Chart":
        selected_col = st.selectbox(f"Select Categorical Column for Pie Chart",
                                    categorical_cols, key=f"pie_{graph_index}")
        if selected_col:
            fig = px.pie(filtered_df, names=selected_col, title=f"Pie Chart of {selected_col}")
            st.plotly_chart(fig, key=f"plotly_chart_{graph_index}")

    elif graph_type == "Tree Map":
        selected_col = st.selectbox(f"Select Categorical Column for Treemap",
                                    categorical_cols, key=f"treemap_{graph_index}")
        if selected_col:
            count_data = filtered_df[selected_col].value_counts().reset_index()
            count_data.columns = [selected_col, "Count"]
            fig = px.treemap(count_data, path=[selected_col], values="Count", title=f"Treemap of {selected_col}")
            st.plotly_chart(fig, key=f"plotly_chart_{graph_index}")

    elif graph_type == "Histogram":
        selected_col = st.selectbox(f"Select Numerical Column for Histogram",
                                    numerical_cols, key=f"histogram_{graph_index}")
        if selected_col:
            fig = px.histogram(filtered_df, x=selected_col, title=f"Histogram of {selected_col}", nbins=30)
            st.plotly_chart(fig, key=f"plotly_chart_{graph_index}")

    elif graph_type == "Box Plot":
        selected_col = st.selectbox(f"Select Numerical Column for Box Plot",
                                    numerical_cols, key=f"boxplot_{graph_index}")
        if selected_col:
            fig = px.box(filtered_df, x=selected_col, title=f"Box Plot of {selected_col}")
            st.plotly_chart(fig, key=f"plotly_chart_{graph_index}")

    elif graph_type == "Scatter Plot":
        x_col = st.selectbox(f"Select X Axis for Scatter Plot",
                             numerical_cols, key=f"scatter_x_{graph_index}")
        y_col = st.selectbox(f"Select Y Axis for Scatter Plot",
                             numerical_cols, key=f"scatter_y_{graph_index}")
        if x_col and y_col:
            fig = px.scatter(filtered_df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
            st.plotly_chart(fig, key=f"plotly_chart_{graph_index}")

    elif graph_type == "Line Chart":
        x_col = st.selectbox(f"Select X Axis for Line Chart",
                             numerical_cols, key=f"line_x_{graph_index}")
        y_col = st.selectbox(f"Select Y Axis for Line Chart",
                             numerical_cols, key=f"line_y_{graph_index}")
        if x_col and y_col:
            fig = px.line(filtered_df, x=x_col, y=y_col, title=f"Line Chart: {x_col} vs {y_col}")
            st.plotly_chart(fig, key=f"plotly_chart_{graph_index}")

    elif graph_type == "Density Plot":
        selected_col = st.selectbox(f"Select Numerical Column for Density Plot",
                                    numerical_cols, key=f"density_{graph_index}")
        if selected_col:
            fig = px.density_contour(filtered_df, x=selected_col, title=f"Density Plot of {selected_col}")
            st.plotly_chart(fig, key=f"plotly_chart_{graph_index}")

    elif graph_type == "Heatmap":
        selected_cols = st.multiselect(f"Select Numerical Columns for Heatmap",
                                       numerical_cols, key=f"heatmap_{graph_index}")
        if len(selected_cols) >= 2:
            fig = px.imshow(filtered_df[selected_cols].corr(), labels=dict(color="Correlation"),
                            x=selected_cols, y=selected_cols, title="Heatmap of Selected Numerical Features",
                            color_continuous_scale="Blues", text_auto=".3f")
            st.plotly_chart(fig, key=f"plotly_chart_{graph_index}")
        else:
            st.warning("Select at least two numerical columns for a heatmap.")


if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
    st.session_state.numerical_cols = []
    st.session_state.categorical_cols = []
    st.session_state.graphs = []


uploaded_file = st.file_uploader("📂 Upload your data file", type=["csv", "xlsx", "xls", "json", "tsv", "txt"])

if uploaded_file is not None:
    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        elif file_name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        elif file_name.endswith(".tsv"):
            df = pd.read_csv(uploaded_file, sep="\t")
        elif file_name.endswith(".txt"):
            df = pd.read_csv(uploaded_file, delimiter=",")  # or adjust delimiter as needed
        else:
            st.error("Unsupported file format.")
            st.stop()

        # Store in session state
        st.session_state.df = df
        st.session_state.numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
        st.session_state.categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()


if not st.session_state.df.empty:
    # Display uploaded data preview
    df, numerical_cols, categorical_cols = get_data()

    with st.sidebar.expander("🔍 Filter Data", expanded=False):
        st.markdown("Use the filters below to refine the dataset view.")

        filtered_df = df.copy()

        # Grouped Checkboxes for Categorical Filters
        if categorical_cols:
            st.markdown("**Categorical Filters**")
            for col in categorical_cols:
                unique_vals = df[col].dropna().unique().tolist()
                selected_vals = []

                # Display "Select All" checkbox for each column
                select_all = st.checkbox(f"Select All {col}", value=True, key=f"all_{col}")

                # Add checkboxes for individual values in the column
                if select_all:
                    selected_vals = unique_vals
                else:
                    for val in unique_vals:
                        if st.checkbox(f"{val}", key=f"{col}_{val}"):
                            selected_vals.append(val)

                # Filter the dataframe based on selected values
                filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]

        # Sliders for Numerical Filters
        if numerical_cols:
            st.markdown("**Numerical Filters**")
            for col in numerical_cols:
                min_val, max_val = df[col].min(), df[col].max()
                selected_range = st.slider(
                    f"{col}", float(min_val), float(max_val),
                    (float(min_val), float(max_val)), key=f"num_filter_{col}"
                )
                filtered_df = filtered_df[
                    (filtered_df[col] >= selected_range[0]) & (filtered_df[col] <= selected_range[1])
                    ]

    # Toggle button to switch between full data or preview
    if 'show_full_table' not in st.session_state:
        st.session_state.show_full_table = False

    if st.button("Show Full Data Table" if not st.session_state.show_full_table else "Show Preview"):
        st.session_state.show_full_table = not st.session_state.show_full_table
        st.rerun()

    # Display either full table or preview based on the button click
    if st.session_state.show_full_table:
        st.dataframe(df, use_container_width=True,hide_index=True)  # Show full table
    else:
        st.dataframe(df.head(5), hide_index=True, use_container_width=True)  # Show preview


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📂 Categorical Columns")
        st.dataframe(pd.DataFrame(categorical_cols, columns=["Categorical Columns"]), hide_index=True)

    with col2:
        st.subheader("🔢 Numerical Columns")
        st.dataframe(pd.DataFrame(numerical_cols, columns=["Numerical Columns"]), hide_index=True)

    st.write("## 📈 Dataset Summary & Insights")

    # Shape
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # Duplicates & missing
    st.write(f"**Duplicate Rows:** {df.duplicated().sum()}")
    st.write("**Missing Values per Column:**")
    st.dataframe(df.isnull().sum().reset_index().rename(columns={"index": "Column", 0: "Missing Values"}),
                 hide_index=True)

    # Descriptive statistics for numerical columns
    st.write("### 🔢 Descriptive Statistics (Numerical Columns)")
    st.dataframe(df[numerical_cols].describe().T.style.format(precision=2), use_container_width=True)

    # Categorical column insights
    if categorical_cols:
        st.write("### 📂 Categorical Columns Overview")
        for col in categorical_cols:
            st.write(
                f"**{col}** – Unique: {df[col].nunique()}, Top: {df[col].mode().iloc[0]} (Freq: {df[col].value_counts().iloc[0]})")

    
    st.write("### ✅ Filtered Data Preview")
    st.dataframe(filtered_df.head(10), use_container_width=True,hide_index=True)

    # Display all added graphs
    for i, graph_type in enumerate(st.session_state.graphs):
        st.write(f"### Graph {i + 1}")

    
        selected_graph = st.selectbox(
            f"Select Graph Type",
            ["Count Plot", "Pie Chart", "Tree Map", "Histogram", "Box Plot", "Scatter Plot",
                "Line Chart", "Density Plot", "Heatmap"],
            index=["Count Plot", "Pie Chart", "Tree Map", "Histogram", "Box Plot",
                    "Scatter Plot", "Line Chart", "Density Plot", "Heatmap"].index(graph_type),
            key=f"graph_type_{i}"
        )

        st.session_state.graphs[i] = selected_graph
        generate_graph(selected_graph, i,filtered_df)

    
        # Add a delete button for each graph
        if st.button("Delete Graph 🗑️", key=f"delete_{i}"):
            st.session_state.graphs.pop(i)
            st.rerun() # Remove graph from session state

    # Button to add a new graph
    st.write("---")
    if st.button("➕ Add Graph"):
        add_graph()
        st.rerun()
