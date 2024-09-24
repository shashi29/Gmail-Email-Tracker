import pandas as pd
import os
import logging

class EmailObserver:
    """Observer to track processed emails and saved job files."""
    def __init__(self):
        self.processed_emails_file = 'processed_emails.csv'
        if os.path.exists(self.processed_emails_file):
            self.processed_df = pd.read_csv(self.processed_emails_file)
        else:
            self.processed_df = pd.DataFrame(columns=['Email_ID'])

    def is_email_processed(self, email_id):
        """Check if an email has already been processed."""
        return email_id in self.processed_df['Email_ID'].values

    def track_processed_email(self, email_id):
        """Mark an email as processed."""
        if not self.is_email_processed(email_id):
            self.processed_df = self.processed_df.append({'Email_ID': email_id}, ignore_index=True)
            self.processed_df.to_csv(self.processed_emails_file, index=False)
            logging.info(f"Email {email_id} tracked as processed.")
