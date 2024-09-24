import os
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobClassification(BaseModel):
    classification: str = Field(description="Job category")
    explanation: str = Field(description="Brief reason for classification")

class JobClassifierService:
    def __init__(self):
        # self.model = ChatOpenAI(
        #     api_key=os.getenv("OPENAI_API_KEY"),
        #     temperature=0.2,
        #     model_name="gpt-3.5-turbo"
        # )
        self.model = OllamaLLM(model="qwen2.5:7b-instruct-q2_K")
        self.parser = JsonOutputParser(pydantic_object=JobClassification)
        
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

    def classify_job(self, subject: str, description: str) -> JobClassification:
        logger.info("Classifying job")
        classification = self.chain.invoke({
            "subject": subject,
            "description": description
        })
        
        return classification

# Example usage
if __name__ == "__main__":
    classifier = JobClassifierService()
    
    subject = "Senior Python Developer"
    description = "We are looking for an experienced Python developer with expertise in Django and Flask. The ideal candidate should have a strong understanding of RESTful APIs, database design, and cloud technologies like AWS."
    
    result = classifier.classify_job(subject, description)
    print(result)
    print(result["properties"]["classification"])