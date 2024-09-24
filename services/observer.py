import pandas as pd
import os
import logging

class EmailObserver:
    """Observer to track processed emails and saved job files."""
    
    def __init__(self):
        self.processed_emails_file = 'processed_emails.csv'
        
        # Define the columns to track email information
        self.columns = ['Email_ID', 'Subject', 'Sender', 'Date', 'Body']
        
        # Load the file if it exists, otherwise create an empty DataFrame
        if os.path.exists(self.processed_emails_file):
            self.processed_df = pd.read_csv(self.processed_emails_file)
        else:
            self.processed_df = pd.DataFrame(columns=self.columns)

    def is_email_processed(self, email_id: str) -> bool:
        """Check if an email has already been processed."""
        return email_id in self.processed_df['Email_ID'].values

    def track_processed_email(self, message):
        """Mark an email as processed and store the full content."""
        if not self.is_email_processed(message.id):
            # Create a new row with the email details
            new_row = pd.DataFrame({
                'Email_ID': [message.id],
                'Subject': [message.subject],
                'Sender': [message.sender],
                'Date': [message.date],
                'Body': [message.plain]
            })
            
            # Use pd.concat to add the new row to the DataFrame
            self.processed_df = pd.concat([self.processed_df, new_row], ignore_index=True)
            
            # Save the updated DataFrame to the CSV file
            self.processed_df.to_csv(self.processed_emails_file, index=False)
            logging.info(f"Email {message.id} from {message.sender} tracked as processed.")

