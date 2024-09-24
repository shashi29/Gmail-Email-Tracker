import time
import logging
from services.service_manager import ServiceManager

# Set logging
logging.basicConfig(level=logging.INFO, filename="email_processing.log", filemode='a')

def run_service():
    """Main function to run the service every 30 minutes."""
    manager = ServiceManager()

    email_service = manager.get_email_service()
    job_classifier = manager.get_job_classifier()
    job_extractor = manager.get_job_extractor()

    while True:
        logging.info("Checking for new unread emails.")
        unread_emails = email_service.fetch_unread_emails()

        if unread_emails:
            logging.info(f"Processing {len(unread_emails)} unread emails.")
            email_service.parallel_process_messages(unread_emails, job_classifier, job_extractor, max_workers=1)
        else:
            logging.info("No new unread emails.")

        logging.info("Sleeping for 30 minutes before the next check.")
        time.sleep(30 * 60)

if __name__ == "__main__":
    run_service()
