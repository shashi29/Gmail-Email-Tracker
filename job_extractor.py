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

# Example usage
if __name__ == "__main__":
    extractor = JobDetailsExtractorService()
    
    subject = "Re:Data Modeler"
    description = """Data ModelerChicago, illinois(Onsite) Please reach me at Diwakar@fpmtechnologies.com Phone: 973-381-5547 Job Summary: We are seeking a highly skilled and detail-oriented Data Modeler to join our team. The Data Modeler will be responsible for designing, implementing, and optimizing data models to support our organization's business processes and decision-making. This role involves working closely with business stakeholders, data engineers, and analysts to ensure that data structures are aligned with business requirements and performance goals.

    Key Responsibilities:
    Data Modeling: Design and develop conceptual, logical, and physical data models for relational databases, data warehouses, and data lakes that meet the organization's needs.
    Data Analysis: Collaborate with business analysts and stakeholders to gather and analyze requirements, translating them into data models that align with business processes.
    Database Design: Create and optimize database schemas, including tables, views, indexes, and stored procedures, ensuring efficient data storage and retrieval.
    Data Governance: Ensure data integrity, consistency, and accuracy across all data models and implement best practices for data management and governance.
    Documentation: Document data models, data dictionaries, and data flow diagrams, ensuring they are easily accessible and understandable by both technical and non-technical stakeholders.
    Collaboration: Work closely with data engineers, ETL developers, and BI teams to implement data models and support data integration, transformation, and reporting processes.
    Performance Tuning: Monitor and optimize data models and database performance, identifying and resolving bottlenecks and improving query performance.
    Technology Evaluation: Stay updated with the latest trends, tools, and technologies in data modeling and database design, recommending and implementing improvements as needed.
    Training and Support: Provide guidance and training to team members and stakeholders on data modeling best practices and data model usage.

    Qualifications:
    Education: Bachelor's degree in Computer Science, Information Technology, Data Science, or a related field. A Master's degree is a plus.
    Experience:
    Minimum of 3-5 years of experience in data modeling, database design, or a related role.
    Proven experience with data modeling tools (e.g., ER/Studio, ERwin, or similar).
    Strong understanding of relational databases, SQL, and database design principles.
    Technical Skills:
    Proficiency in SQL and experience with one or more database management systems (e.g., Oracle, SQL Server, MySQL, PostgreSQL).
    Experience with data warehousing concepts and technologies (e.g., Snowflake, Redshift, BigQuery).
    Familiarity with ETL processes and data integration tools.
    Knowledge of big data platforms (e.g., Hadoop, Spark) and NoSQL databases is a plus.
    Soft Skills:
    Excellent analytical and problem-solving skills.
    Strong communication skills, with the ability to translate complex technical concepts into understandable terms for non-technical stakeholders.
    Ability to work independently as well as collaboratively in a team environment.
    Attention to detail and commitment to delivering high-quality work.
    Preferred Qualifications:
    Experience in agile or iterative development methodologies.
    Certifications in data modeling, database management, or related areas.
    Experience with cloud-based data solutions (e.g., AWS, Azure, Google Cloud)."""
    
    result = extractor.extract_job_details(subject, description)
    print(result)