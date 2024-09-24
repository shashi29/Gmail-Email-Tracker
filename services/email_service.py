import os
import re
import json
import logging
from simplegmail import Gmail
from services.observer import EmailObserver
from simplegmail.query import construct_query
from concurrent.futures import ThreadPoolExecutor

class EmailService:
    """Service for retrieving and processing emails."""
    def __init__(self):
        self.gmail = Gmail()
        self.observer = EmailObserver()  # Observer to track processed emails
        self.saved_jobs_folder = "saved_jobs"
        os.makedirs(self.saved_jobs_folder, exist_ok=True)

    def save_job_details(self, message, job_details, classification):
        """Save job details and classification to a JSON file."""
        try:
            job_data = {
                "job_details": job_details,
                "classification": classification
            }
            file_path = os.path.join(self.saved_jobs_folder, f"job_{message.id}.json")
            with open(file_path, "w") as json_file:
                json.dump(job_details, json_file, indent=4)
            logging.info(f"Job details saved to {file_path}")
            self.observer.track_processed_email(message)  # Track email as processed after saving
        except Exception as e:
            logging.error(f"Error saving job details for email {message.id}: {str(e)}")

    def process_message(self, message, job_classifier, job_extractor):
        """Process each email message: classify and extract job details."""
        try:
            cleaned_content = self.clean_and_remove_patterns(message.plain)
            classification = job_classifier.classify_job(message.subject, cleaned_content)
            logging.info(f"Job Classification : {classification}")
            if classification["classification"] != "Other":
                job_details = job_extractor.extract_job_details(message, cleaned_content)
                logging.info("Job details extracted using llm")
                job_details["tag"] = classification["classification"]
                self.save_job_details(message, job_details, classification)
                return job_details, classification
            else:
                logging.info(f"Non-job email ignored: {message.subject}")
        except Exception as e:
            logging.error(f"Error processing email from {message.sender}: {str(e)}")

    def fetch_unread_emails(self):
        """Fetch unread emails using the Gmail API."""
        query_params = {
            "newer_than": (1, "day"),
            "unread": True
        }
        try:
            logging.info("Fetching unread emails.")
            messages = self.gmail.get_messages(query=construct_query(query_params))
            return [msg for msg in messages if not self.observer.is_email_processed(msg.id)]
        except Exception as e:
            logging.error(f"Error fetching unread emails: {str(e)}")
            return []
        
    def clean_and_remove_patterns(self, text):
        """
        Function to clean text by removing specific patterns and extra whitespace characters.
        """
        try:
            # Define the pattern to match unwanted text like 'Remove ... From' and 'Sign Up'
            pattern = r"Remove.\s*.*\s*|.*Sign.\s*Up\s*.*"
            
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
        
    def parallel_process_messages(self, messages, job_classifier, job_extractor, max_workers=1):
        """Process email messages in parallel."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda msg: self.process_message(msg, job_classifier, job_extractor),
                messages
            ))
            return [result for result in results if result]