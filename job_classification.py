import os
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama.llms import OllamaLLM
from typing import Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobClassification(BaseModel):
    classification: str = Field(description="Job category")
    explanation: str = Field(description="Brief reason for classification")

class JobClassificationResponse(BaseModel):
    status: int = Field(default=200)
    data: JobClassification
    
class JobClassifierService:
    def __init__(self):
        self.model = OllamaLLM(model="qwen2.5:7b-instruct-q2_K")
        self.parser = JsonOutputParser(pydantic_object=JobClassificationResponse)
        
        self.prompt = PromptTemplate(
            template="""Categorize the following job into one of these categories: Java Dev, .NET Dev, Python Dev, Pega Dev, Oracle DBA, DB Admin, BA/Scrum Master/PM, AWS Data Engineer, Azure Data Engineer, GCP Data Engineer, or Other.

            Analyze the job subject and description, focusing on the main requirements and technologies.

            {format_instructions}

            Subject: {subject}
            Description: {description}

            Provide your response as JSON only.
            """,
            input_variables=["subject", "description"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = self.prompt | self.model | self.parser

    def classify_job(self, subject: str, description: str) -> Dict[str, str]:
        """
        Classify the job based on the subject and description and return the result
        in a consistent JSON format.
        
        Args:
            subject (str): The subject of the email/job.
            description (str): The job description.

        Returns:
            Dict[str, str]: Consistent JSON output with "classification" and "explanation".
        """
        logger.info("Classifying job")
        try:
            classification: JobClassification = self.chain.invoke({
                "subject": subject,
                "description": description
            })

            # Return a dictionary ensuring the correct format
            return {
                "classification": classification.classification,
                "explanation": classification.explanation
            }
        except Exception as e:
            logger.error(f"Error classifying job: {str(e)}")
            return {
                "classification": "Other",
                "explanation": "Failed to classify due to error."
            }

# Example usage
if __name__ == "__main__":
    classifier = JobClassifierService()
    
    subject = "Senior Python Developer"
    description = "We are looking for an experienced Python developer with expertise in Django and Flask. The ideal candidate should have a strong understanding of RESTful APIs, database design, and cloud technologies like AWS."
    
    result = classifier.classify_job(subject, description)
    print(result)
