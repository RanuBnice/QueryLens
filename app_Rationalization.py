import streamlit as st
import pandas as pd
import json
import os
import logging
import time
from collections import defaultdict
import numpy as np
from dotenv import load_dotenv
import concurrent.futures
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from openai import OpenAI

# Simple client initialization
@st.cache_resource
def get_openai_client():
    logger.info("Initializing OpenAI client")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY not found. Please set it in your .env file")
        logger.error("OPENAI_API_KEY not found")
        return None
    return OpenAI(api_key=api_key)

def get_report_table_columns(report_name, sql, client):
    """Enhanced function to interact with the OPENAI API for SQL analysis."""
    logger.info(f"Processing SQL for report: {report_name}")
    prompt = (
        f"You are an expert SQL schema analyst with deep knowledge of database structures.\n"
        f"Your sole task is to parse the given SQL query and extract ONLY the PERMANENT, PHYSICAL table names "
        f"that exist independently in the database schema. STRICTLY EXCLUDE:\n"
        f"- Any temporary tables (e.g., starting with 'ZZ' like ZZSP00, ZZMD00).\n"
        f"- CTEs (Common Table Expressions, e.g., WITH clause aliases).\n"
        f"- Derived/subquery tables (e.g., (SELECT ...) AS alias).\n"
        f"- Views or any non-physical constructs unless explicitly permanent tables.\n"
        f"- Aliases; always resolve to the original table name.\n\n"
        f"For each identified real table:\n"
        f"- List ALL unique columns referenced from it (SELECT, WHERE, JOIN, GROUP BY, etc.), "
        f"including those in subqueries or through aliases. Deduplicate columns.\n"
        f"- Columns must be exact names as they appear, without qualifiers unless necessary.\n\n"
        f"For JOIN conditions (explicit JOIN ON or implicit in WHERE):\n"
        f"- Identify EVERY pairing of columns between two real tables.\n"
        f"- Use 'left_table' and 'right_table' as the real table names (not aliases).\n"
        f"- 'left_column' and 'right_column' as the exact column names in the condition (e.g., t1.id = t2.parent_id).\n"
        f"- Direction: Assume left is the first table in the condition; list each unique pair once.\n"
        f"- Handle multi-column joins by creating separate entries.\n"
        f"- Ignore self-joins or conditions not linking tables.\n\n"
        f"EDGE CASES TO HANDLE:\n"
        f"- Nested subqueries: Only extract from innermost real tables.\n"
        f"- Functions/aggregates: Extract base columns (e.g., COUNT(id) -> 'id').\n"
        f"- Aliased columns: Trace back to real table/column.\n"
        f"- UNION/INTERSECT: Process each branch separately.\n"
        f"- If no real tables or joins, output empty lists.\n\n"
        f"OUTPUT RULES (CRITICAL - DEVIATE AND OUTPUT WILL FAIL):\n"
        f"- Respond with ONLY a single, valid JSON object. NO text, code blocks, or extras.\n"
        f"- Structure: {{\"{report_name}\": {{ \"table1\": [\"col1\", \"col2\"], \"table2\": [\"colA\"], \"joins\": [{{ \"left_table\": \"table1\", \"left_column\": \"id\", \"right_table\": \"table2\", \"right_column\": \"parent_id\" }}] }}\n"
        f"- Tables: Keys are exact real table names; values are sorted lists of unique columns.\n"
        f"- Joins: List of dicts at the same level; each dict has exactly those 4 keys.\n"
        f"- Sort table keys alphabetically; sort column lists alphabetically.\n"
        f"- If unsure about a table/column, EXCLUDE it to avoid false positives.\n"
        f"- Deduplicate columns within each table and sort them alphabetically.\n"
        f"- Deduplicate join conditions (same table-column pairs) and sort by left_table, left_column.\n"
        f"- Return empty lists for tables or joins if none qualify.\n\n"
        f"EXAMPLE INPUT/OUTPUT:\n"
        f"Report: SalesReport\n"
        f"SQL: SELECT s.id, c.name FROM sales s JOIN customers c ON s.cust_id = c.id WHERE s.date > '2023-01-01'\n"
        f"OUTPUT: {{\n  \"SalesReport\": {{\n    \"sales\": [\"cust_id\", \"date\", \"id\"],\n    \"customers\": [\"id\", \"name\"],\n    \"joins\": [{{ \"left_table\": \"sales\", \"left_column\": \"cust_id\", \"right_table\": \"customers\", \"right_column\": \"id\" }}]\n  }}\n}}\n\n"
        f"Report Name: {report_name}\nSQL Query:\n{sql}\n\n"
        f"Output the JSON now."
    )
    try:
        start_time = time.time()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are precise and concise. Follow instructions exactly."},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
            top_p=1,
            stream=False,
            stop=None,
        )
        logger.info(f"LLM call for {report_name} completed in {time.time() - start_time:.2f} seconds")
        content = completion.choices[0].message.content.strip()
        if not content.startswith('{'):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end != 0:
                content = content[start:end]
        result = json.loads(content)
        
        # Enhanced filtering: Double-check for ZZ tables in tables and joins
        if report_name in result:
            filtered_result = {report_name: {}}
            for key, value in result[report_name].items():
                if key == "joins":
                    filtered_joins = []
                    if isinstance(value, list):
                        for join in value:
                            if isinstance(join, dict):
                                left_table = join.get('left_table', '').upper()
                                right_table = join.get('right_table', '').upper()
                                if not left_table.startswith('ZZ') and not right_table.startswith('ZZ'):
                                    join_tuple = (join['left_table'], join['left_column'], join['right_table'], join['right_column'])
                                    if join_tuple not in [(j['left_table'], j['left_column'], j['right_table'], j['right_column']) for j in filtered_joins]:
                                        filtered_joins.append(join)
                    filtered_result[report_name]["joins"] = sorted(filtered_joins, key=lambda x: (x['left_table'], x['left_column']))
                elif not str(key).upper().startswith('ZZ') and isinstance(value, list):
                    unique_cols = sorted(list(set(value)))
                    filtered_result[report_name][key] = unique_cols
            filtered_result[report_name] = {k: v for k, v in filtered_result[report_name].items() if k != "joins" or len(v) > 0}
            if sum(1 for k in filtered_result[report_name] if k != "joins") == 0 and not filtered_result[report_name].get("joins"):
                logger.warning(f"No valid tables or joins for {report_name}")
                return None
            return filtered_result
        return result
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error for '{report_name}': {e}. Raw: {content[:200]}...")
        logger.error(f"JSON parse error for {report_name}: {e}")
        return None
    except Exception as api_e:
        st.error(f"API error for '{report_name}': {api_e}")
        logger.error(f"API error for {report_name}: {api_e}")
        return None

def process_single_report(args):
    """Wrapper for parallel processing of a single report."""
    report_name, sql, client = args
    logger.info(f"Starting processing for report: {report_name}")
    result = get_report_table_columns(report_name, sql, client)
    logger.info(f"Finished processing for report: {report_name}")
    return result

def flatten_results(results):
    logger.info("Flattening LLM results")
    flattened_rows = []
    for res in results:
        if res is None:
            continue
        for report_name, tables in res.items():
            for table_name, columns in tables.items():
                if table_name == "joins":
                    continue
                flattened_rows.append({
                    "Report Name": report_name,
                    "Table Name": table_name,
                    "Column Names": ', '.join(columns) if isinstance(columns, list) else columns
                })
    return flattened_rows

def build_report_to_tables(output_df):
    logger.info("Building report-to-tables mapping")
    return output_df.groupby('Report Name')['Table Name'].agg(set).to_dict()

def build_initial_groups(report_to_tables):
    logger.info("Building initial groups")
    return [
        {'merged_reports': {report}, 'table_names': tables}
        for report, tables in report_to_tables.items()
    ]

def can_merge(group1, group2, max_diff=3):
    diff = group1['table_names'].symmetric_difference(group2['table_names'])
    return len(diff) <= max_diff

def merge_groups(groups, min_reports, max_reports):
    logger.info("Merging groups")
    start_time = time.time()
    iteration = 0
    while True:
        iteration += 1
        logger.info(f"Merge iteration {iteration}")
        merged = False
        used = set()
        new_groups = []
        for i, group1 in enumerate(groups):
            if i in used:
                continue
            merged_this_round = False
            for j, group2 in enumerate(groups[i+1:], start=i+1):
                if j in used:
                    continue
                merged_reports_size = len(group1['merged_reports'] | group2['merged_reports'])
                if can_merge(group1, group2, max_diff=3) and merged_reports_size <= max_reports:
                    new_groups.append({
                        'merged_reports': group1['merged_reports'] | group2['merged_reports'],
                        'table_names': group1['table_names'] | group2['table_names']
                    })
                    used.add(i)
                    used.add(j)
                    merged = True
                    merged_this_round = True
                    break
            if not merged_this_round and i not in used:
                new_groups.append(group1)
                used.add(i)
        groups = new_groups
        sizes = [len(g['merged_reports']) for g in groups]
        min_size_in_groups = min(sizes) if sizes else 0
        if min_size_in_groups < min_reports:
            logger.info("Enforcing minimum report size")
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
                    new_groups.append({
                        'merged_reports': group1['merged_reports'] | group2['merged_reports'],
                        'table_names': group1['table_names'] | group2['table_names']
                    })
                    used.add(i)
                    used.add(min_j)
                else:
                    new_groups.append(group1)
                    used.add(i)
            groups = new_groups
        if not merged:
            break
    logger.info(f"Group merging completed in {time.time() - start_time:.2f} seconds")
    return groups

def deduplicate_groups(groups, min_reports, max_reports):
    logger.info("Deduplicating groups")
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
    logger.info("Building table-to-columns mapping")
    table_to_columns = {}
    for _, row in output_df.iterrows():
        table = row['Table Name']
        cols = row['Column Names']
        col_list = [c.strip() for c in cols.split(',')] if isinstance(cols, str) else list(cols or [])
        table_to_columns.setdefault(table, set()).update(col_list)
    return table_to_columns

def collect_all_group_columns(merged_df, table_to_columns):
    logger.info("Collecting group columns")
    column_names_list = []
    for _, row in merged_df.iterrows():
        all_columns = set()
        for table in row['GroupTableNames']:
            all_columns.update(table_to_columns.get(table, set()))
        column_names_list.append(sorted(all_columns))
    return column_names_list

def assign_report_numbers(merged_df):
    logger.info("Assigning report numbers")
    all_reports = set()
    for group in merged_df['MergedReportNames']:
        all_reports.update(group)
    return {report: idx + 1 for idx, report in enumerate(sorted(all_reports))}

def build_report_table_mapping(merged_df, output_df, report_to_number):
    logger.info("Building report-table mapping")
    report_tables_lookup = output_df.groupby('Report Name')['Table Name'].agg(list).to_dict()
    nested_report_tables = []
    for _, row in merged_df.iterrows():
        group_reports = row['MergedReportNames']
        nested_list = [[report_to_number[report], report_tables_lookup.get(report, [])] for report in group_reports]
        nested_report_tables.append(nested_list)
    return nested_report_tables

def assign_table_numbers(merged_df):
    logger.info("Assigning table numbers")
    all_tables = set()
    for group in merged_df['GroupTableNames']:
        all_tables.update(group)
    return {table: idx + 1 for idx, table in enumerate(sorted(all_tables))}

def build_table_column_mapping(merged_df, table_to_columns, table_to_number):
    logger.info("Building table-column mapping")
    nested_table_columns = []
    for _, row in merged_df.iterrows():
        group_tables = row['GroupTableNames']
        nested_list = [[table_to_number[table], sorted(set(table_to_columns.get(table, [])))] for table in group_tables]
        nested_table_columns.append(nested_list)
    return nested_table_columns

def convert_joins_to_column_map(joins):
    join_map = defaultdict(list)
    for join in joins:
        if all(k in join for k in ('left_table', 'left_column', 'right_table', 'right_column')):
            join_map[join['left_column']].append([join['left_table'], join['right_table']])
    return dict(join_map)

def build_report_joins_df(output_df, report_joins_dict, report_to_number):
    logger.info("Building report joins DataFrame")
    report_joins_list = [
        {
            "Report Name": report,
            "Report Unique Value": report_to_number.get(report, None),
            "Joins": report_joins_dict.get(report, [])
        }
        for report in output_df['Report Name'].unique()
    ]
    return pd.DataFrame(report_joins_list)

def build_nested_report_joins(merged_df, report_to_number, report_joins_dict):
    logger.info("Building nested report joins")
    nested_report_joins = []
    for _, row in merged_df.iterrows():
        group_reports = row['MergedReportNames']
        nested_list = [
            [report_to_number[report], convert_joins_to_column_map(report_joins_dict.get(report, []))]
            for report in group_reports
        ]
        nested_report_joins.append(nested_list)
    return nested_report_joins

def calculate_group_accuracy_v2(row):
    report_table_nested = row['ReportTableMapping']
    group_table_names = set(row['GroupTableNames'])
    tables_in_reports = set()
    for report_info in report_table_nested:
        if len(report_info) == 2 and isinstance(report_info[1], list):
            tables_in_reports.update([t for t in report_info[1] if t])
    return 100.0 if tables_in_reports == group_table_names else 0.0

def process_data(uploaded_file, min_reports, max_reports, client):
    """Main function to run the data processing with parallelization for speed."""
    logger.info("Starting data processing")
    try:
        start_time = time.time()
        df = pd.read_excel(uploaded_file, sheet_name='Sorted one')
        st.success("✅ Successfully read the Excel file!")
        logger.info(f"Excel file read in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        st.error(f"❌ Error reading Excel file: {e}")
        logger.error(f"Error reading Excel file: {e}")
        return

    st.info(f"📊 Using min_reports={min_reports}, max_reports={max_reports}")
    logger.info(f"Parameters: min_reports={min_reports}, max_reports={max_reports}")
    
    results = []
    report_joins_dict = {}

    if not df.empty:
        process_args = [(str(row['Name']), row['SQL'], client) for _, row in df.iterrows() if pd.notnull(row['SQL']) and str(row['SQL']).strip()]
        total_rows = len(process_args)
        if total_rows == 0:
            st.warning("⚠️ No valid SQL queries found.")
            logger.warning("No valid SQL queries found")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"🔄 Processing {total_rows} reports in parallel...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_report = {executor.submit(process_single_report, args): args[0] for args in process_args}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_report):
                report_name = future_to_report[future]
                try:
                    result = future.result(timeout=30)
                    if result:
                        results.append(result)
                        if report_name in result:
                            report_joins_dict[report_name] = result[report_name].get('joins', [])
                except Exception as e:
                    st.error(f"❌ Error processing {report_name}: {e}")
                    logger.error(f"Error processing {report_name}: {e}")
                completed += 1
                progress_percent = completed / total_rows
                progress_bar.progress(progress_percent)
                status_text.text(f"🔄 Processed {completed}/{total_rows} reports...")

        progress_bar.empty()
        status_text.text("📈 Generating raw output...")
        logger.info("Parallel processing completed, generating raw output")

    if not results:
        st.error("❌ No reports were processed successfully!")
        logger.error("No reports processed successfully")
        return

    output_df = pd.DataFrame(flatten_results(results))
    st.subheader("📋 Raw Output Data")
    st.dataframe(output_df, use_container_width=True)
    logger.info("Raw output DataFrame displayed")

    # Post-processing with progress feedback
    st_progress = st.progress(0)
    st_status = st.empty()
    post_process_steps = 8
    step = 0

    st_status.text("🔄 Building report-to-tables mapping...")
    report_to_tables = build_report_to_tables(output_df)
    step += 1
    st_progress.progress(step / post_process_steps)

    st_status.text("🔄 Initializing groups...")
    groups = build_initial_groups(report_to_tables)
    step += 1
    st_progress.progress(step / post_process_steps)

    st_status.text("🔄 Merging groups...")
    groups = merge_groups(groups, min_reports, max_reports)
    step += 1
    st_progress.progress(step / post_process_steps)

    st_status.text("🔄 Deduplicating groups...")
    unique_merged_groups = deduplicate_groups(groups, min_reports, max_reports)
    step += 1
    st_progress.progress(step / post_process_steps)

    merged_df = pd.DataFrame([
        {
            'MergedReportNames': sorted(list(group['merged_reports'])),
            'GroupTableNames': sorted(list(group['table_names']))
        }
        for group in unique_merged_groups
    ])
    logger.info("Merged groups DataFrame created")

    st_status.text("🔄 Building table-to-columns mapping...")
    table_to_columns = build_table_to_columns(output_df)
    merged_df['AllGroupColumnNames'] = collect_all_group_columns(merged_df, table_to_columns)
    step += 1
    st_progress.progress(step / post_process_steps)

    st_status.text("🔄 Assigning report and table numbers...")
    report_to_number = assign_report_numbers(merged_df)
    table_to_number = assign_table_numbers(merged_df)
    step += 1
    st_progress.progress(step / post_process_steps)

    st_status.text("🔄 Building report and table mappings...")
    merged_df['ReportTableMapping'] = build_report_table_mapping(merged_df, output_df, report_to_number)
    merged_df['TableColumnMapping'] = build_table_column_mapping(merged_df, table_to_columns, table_to_number)
    step += 1
    st_progress.progress(step / post_process_steps)

    st_status.text("🔄 Building joins and accuracy...")
    report_joins_df = build_report_joins_df(output_df, report_joins_dict, report_to_number)
    merged_df['Join'] = build_nested_report_joins(merged_df, report_to_number, report_joins_dict)
    merged_df['Accuracy'] = merged_df.apply(calculate_group_accuracy_v2, axis=1)
    step += 1
    st_progress.progress(step / post_process_steps)

    st_progress.empty()
    st_status.text("📈 Generating final output...")

    all_report_rows = []
    report_id = 1
    for _, row in merged_df.iterrows():
        logger.info(f"Processing group {report_id}")
        merged_reports = row['MergedReportNames']
        group_tables = row['GroupTableNames']
        table_column_mapping = row.get('TableColumnMapping', [])
        join_data = row.get('Join', [])
        
        tables_accesses_all = group_tables
        column_accesses_all = []
        for mapping in table_column_mapping:
            if isinstance(mapping, list) and len(mapping) == 2:
                columns = mapping[1]
                if isinstance(columns, list):
                    column_accesses_all.extend([f"None.{col}" for col in columns])

        all_joins_list = [item[1] if isinstance(item, list) and len(item) == 2 else np.nan for item in join_data]

        unique_joins_dict = {}
        for join_val in all_joins_list:
            if isinstance(join_val, dict):
                for k, v in join_val.items():
                    if k not in unique_joins_dict:
                        unique_joins_dict[k] = []
                    if isinstance(v, list):
                        for pair in v:
                            rev_pair = [pair[1], pair[0]]
                            if pair not in unique_joins_dict[k] and rev_pair not in unique_joins_dict[k]:
                                unique_joins_dict[k].append(pair)
                                unique_joins_dict[k].append(rev_pair)

        for i, report in enumerate(merged_reports):
            join_val = all_joins_list[i] if i < len(all_joins_list) else np.nan
            joins_without_temp = {}
            if isinstance(join_val, dict):
                for k, v_list in join_val.items():
                    filtered_v = [p for p in v_list if p[0] in group_tables and p[1] in group_tables]
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
    logger.info("Final DataFrame created")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Groups", len(merged_df))
    with col2:
        st.metric("Avg Group Size", f"{new_df1.groupby('ReportID').size().mean():.1f}")
    with col3:
        st.metric("Avg Accuracy", f"{merged_df['Accuracy'].mean():.1f}%")
    with col4:
        st.metric("Total Reports", len(new_df1))

    st.success("🎉 Script finished successfully!")
    logger.info("Script completed successfully")
    csv = new_df1.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Output as CSV",
        data=csv,
        file_name='testing_v2_fixed.csv',
        mime='text/csv',
    )
    st.subheader("📊 Final Processed Data")
    st.dataframe(new_df1, use_container_width=True, height=600)
    logger.info("Final output displayed and available for download")

# Streamlit UI
st.set_page_config(page_title="SQL Analysis Tool", page_icon="🔍", layout="wide")
st.sidebar.image("nssicon.ico", use_container_width=True, caption="SQL Analysis Tool")  # Add logo here
st.title("🔍 SQL Report Analysis Tool")
st.markdown("**Upload an Excel file** containing SQL queries in the 'Sorted one' sheet...")
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("📁 Upload Excel File", type=['xlsx'], help="Upload your Excel file with SQL in 'Sorted one' sheet.")
col1, col2 = st.sidebar.columns(2)
with col1:
    min_reports = st.number_input("Min Reports per Group", min_value=1, value=3, step=1, help="Minimum number of reports in a group.")
with col2:
    max_reports = st.number_input("Max Reports per Group", min_value=1, value=10, step=1, help="Maximum number of reports in a group.")

if st.sidebar.button("🚀 Run Analysis", type="primary"):
    if uploaded_file is None:
        st.error("❌ Please upload an Excel file in the sidebar.")
        logger.error("No file uploaded")
    elif min_reports > max_reports:
        st.error("❌ Minimum cannot exceed maximum.")
        logger.error("Invalid parameters: min_reports > max_reports")
    else:
        with st.spinner("🔄 Initializing analysis..."):
            client = get_openai_client()
            if client is None:
                st.stop()
            process_data(uploaded_file, min_reports, max_reports, client)

st.markdown("---")
st.markdown("*Built using Streamlit & OpenAI*")
