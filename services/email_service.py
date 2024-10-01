import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from simplegmail import Gmail
from simplegmail.query import construct_query
from tqdm import tqdm

from services.observer import EmailObserver

class EmailService:
    """Service for retrieving and processing emails."""
    
    def __init__(self):
        self.gmail = Gmail()
        self.observer = EmailObserver()
        self.s3_client = boto3.client('s3')
        self.sqs_client = boto3.client('sqs')
        
        self.base_folder = os.getenv('SAVE_FOLDER', 'saved_jobs')
        self.s3_bucket = os.getenv('S3_BUCKET', 'job-history-data')
        self.sqs_queue_url = os.getenv('SQS_QUEUE_URL','https://sqs.us-east-1.amazonaws.com/247640998427/S3JobHistoryData')
        
        self.enable_s3_copy = True#os.getenv('ENABLE_S3_COPY', 'False').lower() == 'true'
        self.enable_sqs_message = True#os.getenv('ENABLE_SQS_MESSAGE', 'False').lower() == 'true'
        
        self._setup_folders()
        self._setup_logging()
        self.processed_files = self._load_processed_files()

    def _setup_folders(self):
        """Set up the folder structure for saving job details."""
        current_date = datetime.now().strftime('%Y-%m-%d')
        self.save_folder = os.path.join(self.base_folder, current_date)
        os.makedirs(self.save_folder, exist_ok=True)

    def _setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='email_service.log'
        )

    def _load_processed_files(self) -> Set[str]:
        """Load the set of processed file names from a tracking file."""
        tracking_file = os.path.join(self.base_folder, 'processed_files.json')
        if os.path.exists(tracking_file):
            with open(tracking_file, 'r') as f:
                return set(json.load(f))
        return set()

    def _save_processed_files(self):
        """Save the set of processed file names to a tracking file."""
        tracking_file = os.path.join(self.base_folder, 'processed_files.json')
        with open(tracking_file, 'w') as f:
            json.dump(list(self.processed_files), f)

    def save_job_details(self, message, job_details, classification):
        """Save job details and classification to a JSON file."""
        try:
            job_details_list = list()
            job_details_list.append(job_details)
            
            file_name = f"job_{message.id}.json"
            file_path = os.path.join(self.save_folder, file_name)
            
            with open(file_path, "w") as json_file:
                json.dump(job_details_list, json_file, indent=4)
            
            logging.info(f"Job details saved to {file_path}")
            
            if file_name not in self.processed_files:
                if self.enable_s3_copy:
                    self._copy_to_s3(file_path, file_name)
                
                if self.enable_sqs_message:
                    self._send_sqs_message(file_name)
                
                self.processed_files.add(file_name)
                self._save_processed_files()
            else:
                logging.info(f"File {file_name} already processed. Skipping S3 copy and SQS message.")
            
            self.observer.track_processed_email(message)
        except Exception as e:
            logging.error(f"Error saving job details for email {message.id}: {str(e)}")

    def _copy_to_s3(self, file_path, file_name):
        """Copy the JSON file to S3."""
        try:
            self.s3_client.upload_file(file_path, self.s3_bucket, file_name)
            logging.info(f"File {file_name} uploaded to S3 bucket {self.s3_bucket}")
        except (BotoCoreError, ClientError) as e:
            logging.error(f"Error uploading file {file_name} to S3: {str(e)}")

    def _send_sqs_message(self, file_name):
        """Send an SQS message for the processed JSON file."""
        try:
            message_body = json.dumps({
                "Records": [{
                    "s3": {
                        "bucket": {"name": self.s3_bucket},
                        "object": {"key": file_name}
                    }
                }]
            })
            self.sqs_client.send_message(QueueUrl=self.sqs_queue_url, MessageBody=message_body)
            logging.info(f"SQS message sent for file {file_name}")
        except (BotoCoreError, ClientError) as e:
            logging.error(f"Error sending SQS message for file {file_name}: {str(e)}")

    def process_message(self, message, job_classifier, job_extractor):
        """Process each email message: classify and extract job details."""
        try:
            cleaned_content = self.clean_and_remove_patterns(message.plain)
            classification = job_classifier.classify_job(message.subject, cleaned_content)
            logging.info(f"Job Classification: {classification}")
            
            if classification["classification"] != "Other":
                job_details = job_extractor.extract_job_details(message, cleaned_content)
                logging.info("Job details extracted using LLM")
                job_details["tag"] = classification["classification"]
                self.save_job_details(message, job_details, classification)
                return job_details, classification
            else:
                logging.info(f"Non-job email ignored: {message.subject}")
                self.observer.track_processed_email(message)
                return None
        except Exception as e:
            logging.error(f"Error processing email from {message.sender}: {str(e)}")
            return None

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
        """Clean text by removing specific patterns and extra whitespace characters."""
        try:
            # Define the pattern to match unwanted text like 'Remove ... From' and 'Sign Up'
            pattern = r"Remove.\s*.*\s*|.*Sign.\s*Up\s*.*"
            
            # Remove the matched patterns
            cleaned_text = re.sub(pattern, "", text, 0, re.MULTILINE)
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
            cleaned_text = re.sub(r"[\r\n\t\s]+", " ", cleaned_text).strip()
            return cleaned_text
        except Exception as e:
            logging.error(f"Error cleaning text: {str(e)}")
            return text
        
    def parallel_process_messages(self, messages, job_classifier, job_extractor, max_workers=1):
        """Process email messages in parallel with progress tracking."""
        #Update folder with current date
        self._setup_folders()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda msg: self.process_message(msg, job_classifier, job_extractor),
                tqdm(messages, desc="Processing messages", total=len(messages))
            ))
            return [result for result in results if result]