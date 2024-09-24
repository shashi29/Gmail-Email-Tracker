import os
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama.llms import OllamaLLM
from typing import List, Optional, Dict

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

class Location(BaseModel):
    city: str = Field(default="")
    state: str = Field(default="")
    country: str = Field(default="")
    remote: Optional[bool] = Field(default=None)

class JobDetails(BaseModel):
    employment_type: str = Field(default="")
    job_code: str = Field(default="")
    experience_required: str = Field(default="")
    degree_required: str = Field(default="")
    visa_sponsorship: str = Field(default="")
    notice_period: str = Field(default="")
    duration: str = Field(default="")
    rate: str = Field(default="")

class CandidateInfo(BaseModel):
    client_name: str = Field(default="")

class Skills(BaseModel):
    core: List[str] = Field(default_factory=list)
    primary: List[str] = Field(default_factory=list)
    secondary: List[str] = Field(default_factory=list)
    all: List[str] = Field(default_factory=list)
    with_experience: List[str] = Field(default_factory=list)

class JobInformation(BaseModel):
    job_title: str = Field(description="The title of the job")
    location: Location = Field(description="Location details of the job")
    job_details: JobDetails = Field(description="Detailed information about the job")
    candidate_info: CandidateInfo = Field(description="Information about the candidate")
    skills: Skills = Field(description="Skills required for the job")
    job_description_full: str = Field(description="The full job description")

class JobResponse(BaseModel):
    status: int = Field(default=200)
    data: JobInformation

class JobDetailsExtractorService:
    def __init__(self):
        self.model = OllamaLLM(model="qwen2.5:7b-instruct-q2_K")
        self.parser = JsonOutputParser(pydantic_object=JobResponse)
        
        self.prompt = PromptTemplate(
            template="""Extract detailed information from the following job posting. 
            Analyze the email subject and job description to fill in as many fields as possible.
            If information is not available, use empty strings for text fields, null for boolean fields, and empty lists for list fields.
            Include all mentioned skills in the "all" array of the skills section.
            Only include explicitly stated or reasonably inferred information.

            {format_instructions}

            Email Subject: {subject}
            Job Description: {description}

            Provide your response as JSON only, with a status field set to 200 and all job information nested under a "data" field.
            """,
            input_variables=["subject", "description"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = self.prompt | self.model | self.parser

    def extract_job_details(self, subject: str, description: str) -> Dict:
        logger.info("Extracting job details")
        job_response = self.chain.invoke({
            "subject": subject,
            "description": description
        })
        
        return job_response
    
