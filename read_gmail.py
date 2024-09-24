import re
import csv
import logging
import pandas as pd
from simplegmail import Gmail
from simplegmail.query import construct_query
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='email_processing.log', 
                    filemode='a')

# Initialize Gmail object
gmail = Gmail()

# Get today's date in the required format for Gmail query
today = datetime.now().strftime("%Y/%m/%d")

# Construct a query for unread emails from today
query_params = {
    "newer_than": (1, "day"),  # Newer than today
    "unread": True
}

# Retrieve unread messages
try:
    logging.info("Attempting to retrieve unread messages.")
    messages = gmail.get_messages(query=construct_query(query_params))
    logging.info(f"Successfully retrieved {len(messages)} unread messages.")
except Exception as e:
    logging.error(f"Failed to retrieve unread messages: {str(e)}")
    messages = []

def clean_and_remove_patterns(text):
    """
    Function to clean text by removing specific patterns and extra whitespace characters.
    """
    try:
        # Define the pattern to match unwanted text like 'Remove ... From' and 'Sign Up'
        pattern = r"Remove.\s*.*\s*From.\s*.*\s*.*\s*.*\s*.*\s*|.*Sign.\s*Up\s*.*"
        
        # Remove the matched patterns
        cleaned_text = re.sub(pattern, "", text, 0, re.MULTILINE)
        
        # Remove all excess spaces (extra spaces, tabs, newlines, etc.)
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        
        # Optionally, remove any remaining tabs, newlines, and spaces
        cleaned_text = re.sub(r"[\r\n\t\s]+", " ", cleaned_text).strip()
        
        return cleaned_text
    except Exception as e:
        logging.error(f"Error cleaning text: {str(e)}")
        return text

def process_message(message):
    """
    Function to process each email message, clean its content and return a dictionary.
    """
    try:
        out = clean_and_remove_patterns(message.plain)
        return {
            "Date": message.date,
            "Subject": message.subject,
            "From": message.sender,
            "Message": out
        }
    except Exception as e:
        logging.error(f"Error processing message from {message.sender}: {str(e)}")
        return None

def parallel_process_messages(messages, max_workers=4):
    """
    Function to process email messages in parallel using a thread pool.
    """
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use the map method to process the messages in parallel
            processed_messages = list(executor.map(process_message, messages))
            # Filter out any None values in case of processing errors
            return [msg for msg in processed_messages if msg]
    except Exception as e:
        logging.error(f"Error in parallel processing: {str(e)}")
        return []

# Process messages in parallel
cleaned_data = parallel_process_messages(messages, max_workers=6)

# Convert the cleaned data into a pandas DataFrame
if cleaned_data:
    df = pd.DataFrame(cleaned_data)
    logging.info("DataFrame created successfully with cleaned email data.")
else:
    logging.warning("No cleaned data available to create a DataFrame.")

# Optionally, save the DataFrame to a CSV file
try:
    df.to_csv("cleaned_emails.csv", index=False)
    logging.info("Cleaned emails saved to 'cleaned_emails.csv'.")
except Exception as e:
    logging.error(f"Failed to save DataFrame to CSV: {str(e)}")
