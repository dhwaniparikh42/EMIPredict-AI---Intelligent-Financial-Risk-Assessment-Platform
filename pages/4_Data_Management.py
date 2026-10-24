"""Session-local CRUD for the sample dataset."""
import pandas as pd
import streamlit as st

from app_utils import load_sample_data

st.set_page_config(page_title="Data Management - EMIPredict AI", layout="wide")
st.title("Data Management")
st.caption(
    "Changes are kept only in this Streamlit session. Download the edited CSV before "
    "closing the session if you want to retain them; this page does not modify the training data."
)

try:
    base_df = load_sample_data()
except FileNotFoundError as exc:
    st.warning(str(exc))
    st.code(
        "python scripts/build_frontend_sample.py "
        "--input /path/to/backend/data/processed/train.parquet "
        "--output data/EMI_sample.csv",
        language="bash",
    )
    st.stop()

if "crud_df" not in st.session_state:
    st.session_state.crud_df = base_df.copy()

df = st.session_state.crud_df
if df.empty:
    st.warning("The sample dataset is empty.")
    st.stop()

tab_read, tab_create, tab_update, tab_delete = st.tabs(["Read", "Create", "Update", "Delete"])

with tab_read:
    st.subheader("Browse records")
    if "emi_scenario" in df.columns:
        scenario_options = ["All"] + sorted(df["emi_scenario"].dropna().astype(str).unique().tolist())
        scenario = st.selectbox("Filter by scenario", scenario_options)
        view = df if scenario == "All" else df[df["emi_scenario"].astype(str) == scenario]
    else:
        view = df
    st.dataframe(view, use_container_width=True, height=400)
    st.download_button(
        "Download current session dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="emi_dataset_session.csv",
        mime="text/csv",
    )

with tab_create:
    st.subheader("Add a complete record")
    st.caption("Fields are generated from the loaded dataset schema. Target columns are labels and should only be entered when known.")
    with st.form("create_form"):
        new_values = {}
        columns = st.columns(3)
        for i, column in enumerate(df.columns):
            with columns[i % 3]:
                series = df[column]
                if pd.api.types.is_numeric_dtype(series):
                    numeric = pd.to_numeric(series, errors="coerce").dropna()
                    default = float(numeric.median()) if not numeric.empty else 0.0
                    is_integer = pd.api.types.is_integer_dtype(series)
                    if is_integer:
                        new_values[column] = st.number_input(
                            column, value=int(round(default)), step=1, key=f"create_{column}"
                        )
                    else:
                        new_values[column] = st.number_input(
                            column, value=float(default), step=100.0 if "amount" in column or "salary" in column else 1.0,
                            format="%.3f", key=f"create_{column}"
                        )
                else:
                    options = sorted(series.dropna().astype(str).unique().tolist())
                    if not options:
                        options = ["Unknown"]
                    default_index = 0
                    new_values[column] = st.selectbox(column, options, index=default_index, key=f"create_{column}")
        create_submitted = st.form_submit_button("Add record", type="primary")

    if create_submitted:
        st.session_state.crud_df = pd.concat(
            [st.session_state.crud_df, pd.DataFrame([new_values], columns=df.columns)],
            ignore_index=True,
        )
        st.success("Record added to the session dataset.")
        st.rerun()

with tab_update:
    st.subheader("Update a record")
    row_index = st.number_input("Row index", min_value=0, max_value=len(df) - 1, value=0, step=1)
    original = df.iloc[[int(row_index)]].reset_index(drop=True)
    edited = st.data_editor(original, use_container_width=True, num_rows="fixed", key="update_editor")
    if st.button("Apply update", type="primary"):
        for column in df.columns:
            st.session_state.crud_df.loc[int(row_index), column] = edited.iloc[0][column]
        st.success(f"Record {int(row_index)} updated in this session.")
        st.rerun()

with tab_delete:
    st.subheader("Delete a record")
    del_index = st.number_input(
        "Row index to delete", min_value=0, max_value=len(df) - 1,
        value=0, step=1, key="delete_row_index"
    )
    st.dataframe(df.iloc[[int(del_index)]], use_container_width=True)
    if st.button("Delete record", type="primary"):
        st.session_state.crud_df = st.session_state.crud_df.drop(
            index=st.session_state.crud_df.index[int(del_index)]
        ).reset_index(drop=True)
        st.success(f"Record {int(del_index)} deleted from this session.")
        st.rerun()
