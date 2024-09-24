import logging
from services.email_service import EmailService
from services.job_service import JobClassifierService, JobDetailsExtractorService

class ServiceManager:
    """Singleton class to manage all services (email, job classification, extraction)."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ServiceManager, cls).__new__(cls)
            cls._instance.email_service = EmailService()
            cls._instance.job_classifier = JobClassifierService()
            cls._instance.job_extractor = JobDetailsExtractorService()
            logging.info("ServiceManager initialized.")
        return cls._instance

    def get_email_service(self):
        return self.email_service

    def get_job_classifier(self):
        return self.job_classifier

    def get_job_extractor(self):
        return self.job_extractor
