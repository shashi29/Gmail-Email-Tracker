import streamlit as st
import os
import json
import pandas as pd

# Utility functions
def load_json_files(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r') as file:
                file_data = json.load(file)
                data.extend(file_data)
    return data

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, ', '.join(map(str, v))))
        else:
            items.append((new_key, v))
    return dict(items)

# Page functions
def display_job_listing(job):
    st.header(f"{job['job_title']} at {job['company']}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Information")
        st.write(f"**Location:** {job['location']}")
        st.write(f"**Full Location:** {job['full_location']['city']}, {job['full_location']['state']}, {job['full_location']['country']}")
        st.write(f"**Job Type:** {', '.join(job['job_type'])}")
        st.write(f"**Employment Type:** {', '.join(job['emp_type'])}")
        st.write(f"**Date Posted:** {job['date_posted']}")
        st.write(f"**Source:** {job['source']}")
        st.write(f"**Tag:** {job['tag']}")
    
    with col2:
        st.subheader("Contact Information")
        st.write(f"**Contact Person:** {job['contact_person']}")
        st.write(f"**Email:** {job['email']}")
    
    st.subheader("Job Details")
    details = job['job_details']
    for key, value in details.items():
        if isinstance(value, list):
            st.write(f"**{key.replace('_', ' ').title()}:** {', '.join(value)}")
        else:
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")
    
    st.subheader("Skills")
    skills = job['skills']
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Core Skills:**")
        st.write(", ".join(skills['core']))
        st.write("**Primary Skills:**")
        st.write(", ".join(skills['primary']))
    with col2:
        st.write("**Secondary Skills:**")
        st.write(", ".join(skills['secondary']))
    
    with st.expander("View Full Job Description"):
        st.write(job['jd'])

def detail_view_page(job_data):
    st.title("Job Listings - Detailed View")

    if not job_data:
        st.warning("No job listings found in the specified folder.")
        return

    # Create a selectbox for choosing a job listing
    job_titles = [f"{job['job_title']} at {job['company']}" for job in job_data]
    selected_job = st.selectbox("Select a job listing:", job_titles)

    # Display the selected job listing
    selected_index = job_titles.index(selected_job)
    display_job_listing(job_data[selected_index])

    # Add a download button for the full JSON data
    st.download_button(
        label="Download Full JSON Data",
        data=json.dumps(job_data, indent=2),
        file_name="job_listings.json",
        mime="application/json"
    )

def table_view_page(job_data):
    st.title("Job Listings - Table View")

    if not job_data:
        st.warning("No job listings found in the specified folder.")
        return

    # Flatten the nested dictionaries
    flattened_data = [flatten_dict(job) for job in job_data]

    # Convert to DataFrame
    df = pd.DataFrame(flattened_data)

    # Allow users to select columns to display
    all_columns = df.columns.tolist()
    default_columns = ['job_title', 'company', 'location', 'date_posted', 'emp_type']
    selected_columns = st.multiselect("Select columns to display:", all_columns, default=default_columns)

    # Display the selected columns
    if selected_columns:
        st.dataframe(df[selected_columns])
    else:
        st.warning("Please select at least one column to display.")

    # Add a download button for the full CSV data
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download Full CSV Data",
        data=csv,
        file_name="job_listings.csv",
        mime="text/csv"
    )

# Main app
def main():
    st.set_page_config(layout="wide", page_title="Job Listings Viewer")

    # Set the folder path containing JSON files
    folder_path = "/root/Gmail-Email-Tracker/saved_jobs/2024-10-01/"

    # Load JSON data
    job_data = load_json_files(folder_path)

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ("Detailed View", "Table View"))

    if page == "Detailed View":
        detail_view_page(job_data)
    elif page == "Table View":
        table_view_page(job_data)

if __name__ == "__main__":
    main()