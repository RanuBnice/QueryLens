import streamlit as st
import pandas as pd
import json
import os
import sys
from collections import defaultdict
import numpy as np
import io

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# The core logic functions from the original script
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

# NOTE: The Groq client and API key usage should be managed securely,
# e.g., using Streamlit secrets. For this example, it's hardcoded.
# If you don't have Groq installed, run `pip install groq`.
from groq import Groq

# Place your Groq API key here or use st.secrets.
# client = Groq(api_key=st.secrets["groq_api_key"])
client = Groq(api_key='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')

def get_report_table_columns(report_name, sql, client):
    """Function to interact with the Groq API for SQL analysis."""
    prompt = (
        f"You are a professional SQL analyst.\\n"
        f"Your task is to analyze the provided report name and SQL query, and extract ONLY the real, original, physical table names from the query. You must EXCLUDE all temporary tables, CTEs (Common Table Expressions), derived tables, subquery aliases, or any tables created, dropped, or defined within the query itself. Only include tables that exist independently in the database schema.\\n\\n"
        f"For each real table, list all columns used in the query, including those referenced via aliases or in subqueries, but only for the real, physical tables.\\n\\n"
        f"Additionally, for each join condition in the SQL query (whether in the ON or WHERE clause, regardless of whether the word 'join' is present), identify:\\n"
        f"- The real/original table names and columns involved in the join (not the aliases).\\n"
        f"- If different aliases refer to the same table, always use the real table name in the output.\\n"
        f"- Find all join conditions (usually in the ON or WHERE clause). For each join, map the alias back to the real table name and record the columns used.\\n\\n"
        f"IMPORTANT OUTPUT GUIDELINES (strictly follow):\\n"
        f"- Output MUST be a single valid JSON object and nothing else.\\n"
        f"- The report name MUST be the top-level key in the JSON object.\\n"
        f"- Each real table name MUST be a key mapping to a list of its columns.\\n"
        f"- There MUST be an additional key called 'joins' (at the same level as the table names), which maps to a list of join descriptions. Each join description is a dict with keys: 'left_table', 'left_column', 'right_table', 'right_column'.\\n"
        f"- DO NOT include any explanations, pretext, posttext, or formatting such as triple quotes, markdown, or code blocks—output ONLY the JSON object.\\n"
        f"- DO NOT add any extra text before or after the JSON object.\\n"
        f"- Be thorough in identifying real tables and their columns, even if referenced through aliases or subqueries.\\n"
        f"- If you are unsure, err on the side of including all possible real tables and columns used in the query.\\n"
        f"- You must NOT include any temporary tables, CTEs, or derived tables in your output under any circumstances. Only include tables that are permanent and exist in the database schema.\\n\\n"
        f"Example output (your output must match this structure exactly):\\n"
        f'{{\\n  "{report_name}": {{\\n    "table1": ["col1", "col2"],\\n    "table2": ["colA", "colB"],\\n    "joins": [{{"left_table": "table1", "left_column": "col1", "right_table": "table2", "right_column": "colA"}}]\\n  }}\\n}}\\n\\n'
        f"Report Name: {report_name}\\n"
        f"SQL:\\n{sql}\\n"
    )
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        content = ""
        for chunk in completion:
            content += chunk.choices[0].delta.content or ""
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            return json.loads(json_str)
        except Exception as e:
            st.error(f"Failed to parse JSON from Groq response for report '{report_name}': {e}")
            st.code(f"Raw response: {content}")
            return None
    except Exception as api_e:
        st.error(f"Groq API call failed for report '{report_name}': {api_e}")
        return None

# --- Helper functions for data processing ---
def flatten_results(results):
    flattened_rows = []
    for res in results:
        for report_name, tables in res.items():
            for table_name, columns in tables.items():
                if table_name == "joins":
                    continue
                flattened_rows.append({
                    "Report Name": report_name,
                    "Table Name": table_name,
                    "Column Names": ', '.join(columns)
                })
    return flattened_rows

def build_report_to_tables(output_df):
    return output_df.groupby('Report Name')['Table Name'].apply(set).to_dict()

def build_initial_groups(report_to_tables):
    groups = []
    for report, tables in report_to_tables.items():
        groups.append({
            'merged_reports': set([report]),
            'table_names': set(tables)
        })
    return groups

def can_merge(group1, group2, max_diff=3):
    diff = group1['table_names'].symmetric_difference(group2['table_names'])
    return len(diff) <= max_diff

def merge_groups(groups, min_reports, max_reports):
    while True:
        merged = False
        used = set()
        new_groups = []
        i = 0
        while i < len(groups):
            if i in used:
                i += 1
                continue
            group1 = groups[i]
            merged_this_round = False
            for j in range(i+1, len(groups)):
                if j in used:
                    continue
                group2 = groups[j]
                merged_reports_size = len(group1['merged_reports'] | group2['merged_reports'])
                if can_merge(group1, group2, max_diff=3) and merged_reports_size <= max_reports:
                    merged_group = {
                        'merged_reports': group1['merged_reports'] | group2['merged_reports'],
                        'table_names': group1['table_names'] | group2['table_names']
                    }
                    new_groups.append(merged_group)
                    used.add(i)
                    used.add(j)
                    merged = True
                    merged_this_round = True
                    break
            if not merged_this_round and i not in used:
                new_groups.append(group1)
                used.add(i)
            i += 1
        groups = new_groups
        sizes = [len(g['merged_reports']) for g in groups]
        min_size_in_groups = min(sizes) if sizes else 0
        if min_size_in_groups < min_reports:
            merged = True
            used = set()
            new_groups = []
            for i, group1 in enumerate(groups):
                if i in used:
                    continue
                if len(group1['merged_reports']) >= min_reports:
                    new_groups.append(group1)
                    used.add(i)
                    continue
                min_diff = float('inf')
                min_j = None
                for j, group2 in enumerate(groups):
                    if j == i or j in used:
                        continue
                    diff = len(group1['table_names'].symmetric_difference(group2['table_names']))
                    merged_reports_size = len(group1['merged_reports'] | group2['merged_reports'])
                    if merged_reports_size <= max_reports and diff < min_diff:
                        min_diff = diff
                        min_j = j
                if min_j is not None:
                    group2 = groups[min_j]
                    merged_group = {
                        'merged_reports': group1['merged_reports'] | group2['merged_reports'],
                        'table_names': group1['table_names'] | group2['table_names']
                    }
                    new_groups.append(merged_group)
                    used.add(i)
                    used.add(min_j)
                else:
                    new_groups.append(group1)
                    used.add(i)
            groups = new_groups
        if not merged:
            break
    return groups

def deduplicate_groups(groups, min_reports, max_reports):
    final_groups = [g for g in groups if min_reports <= len(g['merged_reports']) <= max_reports]
    unique_merged_groups = []
    seen = set()
    for group in final_groups:
        key = (tuple(sorted(group['merged_reports'])), tuple(sorted(group['table_names'])))
        if key not in seen:
            unique_merged_groups.append(group)
            seen.add(key)
    return unique_merged_groups

def build_table_to_columns(output_df):
    table_to_columns = {}
    for idx, row in output_df.iterrows():
        table = row['Table Name']
        cols = row['Column Names']
        if isinstance(cols, str):
            col_list = [c.strip() for c in cols.split(',')]
        else:
            col_list = list(cols)
        if table in table_to_columns:
            table_to_columns[table].update(col_list)
        else:
            table_to_columns[table] = set(col_list)
    return table_to_columns

def collect_all_group_columns(merged_df, table_to_columns):
    column_names_list = []
    for idx, row in merged_df.iterrows():
        all_columns = []
        for table in row['GroupTableNames']:
            cols = list(table_to_columns.get(table, set()))
            all_columns.extend(cols)
        seen = set()
        unique_columns = [x for x in all_columns if not (x in seen or seen.add(x))]
        column_names_list.append(unique_columns)
    return column_names_list

def assign_report_numbers(merged_df):
    all_reports_in_groups = set()
    for group in merged_df['MergedReportNames']:
        all_reports_in_groups.update(group)
    report_to_number = {report: idx+1 for idx, report in enumerate(sorted(all_reports_in_groups))}
    return report_to_number

def build_report_table_mapping(merged_df, output_df, report_to_number):
    report_tables_lookup = output_df.groupby('Report Name')['Table Name'].apply(list).to_dict()
    nested_report_tables = []
    for idx, row in merged_df.iterrows():
        group_reports = row['MergedReportNames']
        nested_list = []
        for report in group_reports:
            report_num = report_to_number[report]
            tables = report_tables_lookup.get(report, [])
            nested_list.append([report_num, tables])
        nested_report_tables.append(nested_list)
    return nested_report_tables

def assign_table_numbers(merged_df):
    all_tables_in_groups = set()
    for group in merged_df['GroupTableNames']:
        all_tables_in_groups.update(group)
    table_to_number = {table: idx+1 for idx, table in enumerate(sorted(all_tables_in_groups))}
    return table_to_number

def build_table_column_mapping(merged_df, table_to_columns, table_to_number):
    nested_table_columns = []
    for idx, row in merged_df.iterrows():
        group_tables = row['GroupTableNames']
        nested_list = []
        for table in group_tables:
            table_num = table_to_number[table]
            columns = list(table_to_columns.get(table, set()))
            columns = sorted(set([c.strip() for c in columns if c.strip()]))
            nested_list.append([table_num, columns])
        nested_table_columns.append(nested_list)
    return nested_table_columns

def convert_joins_to_column_map(joins):
    join_map = defaultdict(list)
    for join in joins:
        if all(k in join for k in ('left_table', 'left_column', 'right_table', 'right_column')):
            join_map[join['left_column']].append([join['left_table'], join['right_table']])
    return dict(join_map)

def build_report_joins_df(output_df, report_joins_dict, report_to_number):
    report_joins_list = []
    for report in output_df['Report Name'].unique():
        joins = report_joins_dict.get(report, [])
        report_joins_list.append({
            "Report Name": report,
            "Report Unique Value": report_to_number.get(report, None),
            "Joins": joins
        })
    return pd.DataFrame(report_joins_list)

def build_nested_report_joins(merged_df, report_to_number, report_joins_dict):
    nested_report_joins = []
    for idx, row in merged_df.iterrows():
        group_reports = row['MergedReportNames']
        nested_list = []
        for report in group_reports:
            report_num = report_to_number[report]
            joins = report_joins_dict.get(report, [])
            joins_dict = convert_joins_to_column_map(joins)
            nested_list.append([report_num, joins_dict])
        nested_report_joins.append(nested_list)
    return nested_report_joins

def calculate_group_accuracy_v2(row):
    report_table_nested = row['ReportTableMapping']
    group_table_names = set(row['GroupTableNames'])
    tables_in_reports = set()
    for report_info in report_table_nested:
        if len(report_info) == 2 and isinstance(report_info[1], list):
            tables_in_reports.update([t for t in report_info[1] if t])
    if tables_in_reports == group_table_names:
        return 100.0
    else:
        return 0.0

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# The main logic wrapped for Streamlit
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def process_data(uploaded_file, min_reports, max_reports):
    """Main function to run the data processing with Streamlit."""
    try:
        # Read the Excel file from the uploaded file object
        df = pd.read_excel(uploaded_file, sheet_name='Sorted one')
        st.success("Successfully read the Excel file!")
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return

    st.info(f"Using **min_reports={min_reports}**, **max_reports={max_reports}**")
    
    results = []
    report_joins_dict = {}
    report_result_dict = {}

    if not df.empty:
        total_rows = len(df)
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, row in df.iterrows():
            report_name = str(row['Name'])
            sql = row['SQL']
            
            progress_percent = (idx + 1) / total_rows
            progress_bar.progress(progress_percent)
            status_text.text(f"Processing report {idx+1} of {total_rows}: {report_name}...")

            if pd.notnull(sql) and str(sql).strip():
                result = get_report_table_columns(report_name, sql, client)
                if result:
                    results.append(result)
                    if report_name in result:
                        joins = result[report_name].get('joins', [])
                        report_joins_dict[report_name] = joins
                        report_result_dict[report_name] = result[report_name]
        
        status_text.text("All reports processed. Generating output...")
        progress_bar.empty()

    flattened_rows = flatten_results(results)
    output_df = pd.DataFrame(flattened_rows)
    st.subheader("Output Data")
    st.dataframe(output_df)

    report_to_tables = build_report_to_tables(output_df)
    groups = build_initial_groups(report_to_tables)
    groups = merge_groups(groups, min_reports, max_reports)
    unique_merged_groups = deduplicate_groups(groups, min_reports, max_reports)

    merged_df = pd.DataFrame([
        {
            'MergedReportNames': sorted(list(group['merged_reports'])),
            'GroupTableNames': sorted(list(group['table_names']))
        }
        for group in unique_merged_groups
    ])
    
    table_to_columns = build_table_to_columns(output_df)
    merged_df['AllGroupColumnNames'] = collect_all_group_columns(merged_df, table_to_columns)
    report_to_number = assign_report_numbers(merged_df)
    merged_df['ReportTableMapping'] = build_report_table_mapping(merged_df, output_df, report_to_number)
    table_to_number = assign_table_numbers(merged_df)
    merged_df['TableColumnMapping'] = build_table_column_mapping(merged_df, table_to_columns, table_to_number)
    report_joins_df = build_report_joins_df(output_df, report_joins_dict, report_to_number)
    merged_df['Join'] = build_nested_report_joins(merged_df, report_to_number, report_joins_dict)
    merged_df['Accuracy'] = merged_df.apply(calculate_group_accuracy_v2, axis=1)

    desired_order = [
        'ReportTableMapping', 'MergedReportNames', 'GroupTableNames',
        'TableColumnMapping', 'AllGroupColumnNames', 'Join', 'Accuracy'
    ]
    merged_df = merged_df[desired_order]

    all_report_rows = []
    report_id = 1
    for idx, row in merged_df.iterrows():
        merged_reports = row['MergedReportNames']
        group_tables = row['GroupTableNames']
        table_column_mapping = row.get('TableColumnMapping', [])
        join_data = row.get('Join', [])
        
        tables_accesses_all = group_tables
        column_accesses_all = []
        if isinstance(table_column_mapping, list):
            for mapping in table_column_mapping:
                if isinstance(mapping, list) and len(mapping) == 2:
                    table_name = table_to_number.get(mapping[0])
                    columns = mapping[1]
                    if isinstance(columns, list):
                        for col in columns:
                            column_accesses_all.append(f"{table_name}.{col}")

        all_joins_list = []
        if isinstance(join_data, list):
            for item in join_data:
                if isinstance(item, list) and len(item) == 2:
                    all_joins_list.append(item[1])
                else:
                    all_joins_list.append(np.nan)

        unique_joins_dict = {}
        for join_val in all_joins_list:
            if isinstance(join_val, dict):
                for k, v in join_val.items():
                    if k not in unique_joins_dict:
                        unique_joins_dict[k] = []
                    if isinstance(v, list):
                        for pair in v:
                            if pair not in unique_joins_dict[k]:
                                unique_joins_dict[k].append(pair)
        
        for i, report in enumerate(merged_reports):
            join_val = all_joins_list[i] if i < len(all_joins_list) else np.nan
            
            joins_without_temp = {}
            if isinstance(join_val, dict):
                for k, v_list in join_val.items():
                    filtered_v = [p for p in v_list if p[0] in tables_accesses_all and p[1] in tables_accesses_all]
                    if filtered_v:
                        joins_without_temp[k] = filtered_v

            if i == 0:
                all_report_rows.append({
                    'ReportID': report_id, 'ReportName': report,
                    'Tables accesses': tables_accesses_all,
                    'Column Accesses': column_accesses_all,
                    'All_Joins': join_val,
                    'Joins_without_Temp': joins_without_temp if joins_without_temp else np.nan,
                    'Unique_Joins': unique_joins_dict if unique_joins_dict else np.nan
                })
            else:
                all_report_rows.append({
                    'ReportID': report_id, 'ReportName': report,
                    'Tables accesses': np.nan, 'Column Accesses': np.nan,
                    'All_Joins': join_val,
                    'Joins_without_Temp': joins_without_temp if joins_without_temp else np.nan,
                    'Unique_Joins': np.nan
                })
        report_id += 1

    new_df1 = pd.DataFrame(all_report_rows)
    st.success("Script finished successfully!")
    st.download_button(
        label="Download Output as CSV",
        data=new_df1.to_csv(index=False).encode('utf-8'),
        file_name='testing_v2.csv',
        mime='text/csv',
    )
    st.subheader("Final Processed Data")
    st.dataframe(new_df1)


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
# Streamlit UI
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

st.title("SQL Analysis Script Runner")

st.markdown("Upload an Excel file with SQL queries and specify bucket values to run the analysis.")

uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

col1, col2 = st.columns(2)
with col1:
    min_reports = st.number_input("Minimum Bucket Value", min_value=1, value=3, step=1)
with col2:
    max_reports = st.number_input("Maximum Bucket Value", min_value=1, value=10, step=1)

if st.button("Run Analysis"):
    if uploaded_file is None:
        st.error("Please upload an Excel file to proceed.")
    elif min_reports > max_reports:
        st.error("The minimum bucket value cannot be greater than the maximum value.")
    else:
        with st.spinner("Processing reports... This may take a while."):
            process_data(uploaded_file, min_reports, max_reports)