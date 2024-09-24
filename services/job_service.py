import os
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama.llms import OllamaLLM
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from services.config import Settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JobClassification(BaseModel):
    classification: str = Field(description="Job category")
    explanation: str = Field(description="Brief reason for classification")

class JobClassifierService:
    def __init__(self):
        self.model = ChatOpenAI(api_key=Settings.OPENAI_API_KEY,
                                temperature=Settings.OPENAI_TEMPERATURE,
                                model_name=Settings.OPENAI_MODEL,
                                top_p=Settings.OPENAI_TOP_P)
        # self.model = OllamaLLM(model="qwen2.5:7b-instruct-q2_K")
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
    city: str = Field(default="", description="The city where the job is physically located, not remote")
    state: str = Field(default="", description="The state where the job is physically located, not remote")
    country: str = Field(default="USA", description="The country where the job is physically located, not remote")

class JobDetails(BaseModel):
    employment_type: List[str] = Field(default_factory=list, description="The employment types for the job (e.g., 'third party', 'contract', 'full-time', 'part-time')")
    job_code: str = Field(default="", description="The job code or ID")
    experience_required: str = Field(default="", description="The required level of experience for the job")
    degree_required: str = Field(default="", description="The required degree for the job")
    visa_sponsorship: str = Field(default="", description="Whether the employer offers visa sponsorship")
    notice_period: str = Field(default="", description="The required notice period for the job")
    duration: str = Field(default="", description="The duration of the job")
    rate: str = Field(default="", description="The rate of pay for the job")

class Skills(BaseModel):
    core: List[str] = Field(default_factory=list, description="The core skills required for the job")
    primary: List[str] = Field(default_factory=list, description="The primary skills required for the job")
    secondary: List[str] = Field(default_factory=list, description="The secondary skills required for the job")
    all: List[str] = Field(default_factory=list, description="All the skills required for the job")
    with_experience: List[str] = Field(default_factory=list, description="The skills for which experience is required")

class JobInformation(BaseModel):
    #source: str = Field(default="linkedin", description="The source where the job was found")
    company: str = Field(default="", description="The name of the company offering the job")
    #date_posted: str = Field(default="", description="The date the job was posted")
    #unique_id: str = Field(default="", description="A unique identifier for the job")
    job_title: str = Field(default="", description="The title of the job")
    location: str = Field(description="The location of the job, derived from the full_location fields")
    full_location: Location = Field(default_factory=Location, description="The actual physical location of the job, not remote")
    job_details: JobDetails = Field(default_factory=JobDetails, description="Detailed information about the job")
    skills: Skills = Field(default_factory=Skills, description="The skills required for the job")
    job_type: List[str] = Field(default_factory=list, description="The job types (e.g., 'remote', 'onsite', 'hybrid')")
    contact_person: str = Field(default="", description="The contact person for the job")
    email: str = Field(default="", description="The email address of the contact person")
    jd: str = Field(default="", description="The full job description, extracted directly from the email content without any additional information")
    emp_type: List[str] = Field(default_factory=list, description="The employment types for the job (e.g., 'third party', 'contract', 'full-time', 'part-time')")
    tag: str = Field(default="", description="Any tags or keywords associated with the job")

class JobResponse(BaseModel):
    status: int = Field(default=200, description="The status code of the response")
    data: JobInformation = Field(description="The job information extracted from the job posting")


class JobResponse(BaseModel):
    status: int = Field(default=200, description="The status code of the response")
    data: JobInformation = Field(description="The job information extracted from the job posting")

class JobDetailsExtractorService:
    def __init__(self):
        self.model = ChatOpenAI(api_key=Settings.OPENAI_API_KEY,
                                temperature=Settings.OPENAI_TEMPERATURE,
                                model_name=Settings.OPENAI_MODEL,
                                top_p=Settings.OPENAI_TOP_P)        
        # self.model = OllamaLLM(model="qwen2.5:7b-instruct-q2_K")

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

    def extract_job_details(self, message, description) -> Dict:
        logger.info("Extracting job details")
        job_response = self.chain.invoke({
            "subject": message.subject,
            "description": description
        })
        
        #Add aditional fields
        job_response["source"] = "Email"
        job_response["date_posted"] = message.date
        job_response["unique_id"] = message.id
        #job_response["email"] =  message.sender
        
        return job_response
    
