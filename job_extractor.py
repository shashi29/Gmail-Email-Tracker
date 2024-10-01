import os
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama.llms import OllamaLLM
from langchain_openai import ChatOpenAI
from typing import List, Optional, Dict
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class Location(BaseModel):
    city: str = Field(
        default="",
        description=(
            "Determine the job's city location using this logical process:\n"
            "1. Look for explicit mentions of a city in the job posting.\n"
            "2. If multiple cities are listed, use the following priority:\n"
            "   a) The city listed first or emphasized as the primary location.\n"
            "   b) The city where the company's main office is located, if mentioned.\n"
            "3. For remote jobs with a required proximity to a specific city, use that city.\n"
            "4. If no specific city is mentioned, but a region or area is (e.g., 'Greater Boston Area'), use the main city of that region.\n"
            "5. For truly remote positions with no location preference, leave this field empty.\n"
            "6. Ensure the city name is spelled correctly and use the most common English name for international cities.\n"
            "7. Do not include state or country information in this field.\n"
            "Examples: 'New York', 'San Francisco', 'London', ''\n"
            "Note: This field is for the physical location, not for remote work arrangements."
        )
    )

    state: str = Field(
        default="",
        description=(
            "Specify the job's state location using this process:\n"
            "1. Identify the state corresponding to the city specified in the 'city' field.\n"
            "2. For US locations, use the two-letter state abbreviation (e.g., 'CA' for California, 'NY' for New York).\n"
            "3. For non-US locations, consider the following:\n"
            "   a) If the country has states or provinces, use the full name of the state/province.\n"
            "   b) If the country doesn't use states, or if the state is not relevant/mentioned, leave this field empty.\n"
            "4. If multiple states are mentioned in the job posting, use the state that corresponds to the chosen city.\n"
            "5. For remote jobs with no specific location, leave this field empty.\n"
            "6. Ensure consistency between the city and state fields.\n"
            "Examples: 'CA', 'NY', 'Ontario', ''\n"
            "Note: Use official state names or abbreviations, not colloquial terms."
        )
    )

    country: str = Field(
        default="USA",
        description=(
            "Determine the job's country location through this logical flow:\n"
            "1. Look for explicit mentions of a country in the job posting.\n"
            "2. If no country is specified, consider the following:\n"
            "   a) If a US city or state is mentioned, use 'USA'.\n"
            "   b) If a non-US location is clearly implied, use that country's name.\n"
            "3. For multinational postings, use the country of the primary job location.\n"
            "4. For remote positions:\n"
            "   a) If there's a country restriction, use that country.\n"
            "   b) If it's global, use 'Global' instead of a specific country.\n"
            "6. Ensure the country name is spelled correctly and use the most common English name.\n"
            "Note: Default is 'USA' unless there's clear indication otherwise."
        )
    )

class JobDetails(BaseModel):
    employment_type: List[str] = Field(
    default_factory=list,
    description = (
    """Use the following decision tree to determine the employment type(s):
    1. Third Party Assessment:
    - Is an agency, consulting firm, or C2C arrangement mentioned?
    - Are terms like "vendor," "consultant," or "third-party provider" used?
    If Yes to any -> Include 'third party' in the list of types
    Regardless of answer, proceed to step 2

    2. Contract/Temporary Assessment:
    - Is the position described as temporary, project-based, or fixed-term?
    - Is there mention of a specific contract duration or end date?
    - Is the role described using terms like "contractor" or "freelancer"?
    If Yes to any -> Include 'contract' in the list of types
    Regardless of answer, proceed to step 3

    3. Full-Time Assessment:
    - Are full-time hours (typically 35-40 per week) mentioned?
    - Is there reference to comprehensive benefits, typical of full-time roles?
    - Are terms like "permanent," "regular," or "full-time employee" used?
    If Yes to any -> Include 'full-time' in the list of types
    Regardless of answer, proceed to step 4

    4. Part-Time Assessment:
    - Is the position explicitly described as part-time?
    - Are reduced or flexible hours mentioned?
    - Is there indication of fewer benefits compared to full-time roles?
    If Yes to any -> Include 'part-time' in the list of types
    Regardless of answer, proceed to step 5

    5. Compensation Structure Assessment:
    - Is an hourly rate mentioned without clear indication of employment type?
    - Is the compensation described as "per project" or "per assignment"?
    If Yes to any -> Consider adding 'contract' if not already included
    If 'contract' is added here and step 1 was Yes, consider also including 'third party'

    6. Final Review:
    - If no types have been determined, mark as "unclear"
    - Check for logical consistency:
        * 'third party' can coexist with any other type
        * 'contract' can coexist with 'third party' but not with 'full-time' or 'part-time'
        * 'full-time' and 'part-time' are mutually exclusive
    - If any inconsistencies are found, prioritize based on the strength of the evidence in the job information

    Follow this decision tree for the given job information. You may end up with multiple types (e.g., 'third party' and 'contract'), a single type, or no clear type ("unclear").
    """
    )
    )
    
    job_code: str = Field(
    default="",
    description=(
        "Determine the job code or ID using the following process:\n"
        "1. Check if a specific job code is provided in the job posting.\n"
        "2. If not explicitly stated, look for any unique identifier associated with the position.\n"
        "3. Consider the format: Is it alphanumeric, numeric only, or following a specific pattern?\n"
        "4. Verify that the code is unique within the organization's job listings.\n"
        "5. If no code is found, leave this field empty.\n"
        "Example: 'JD2023-CS-001' or 'TechLead-NYC-52'\n"
        "Note: Do not create or invent a job code if one is not provided."
    )
    )

    experience_required: str = Field(
    default="",
    description=(
        "Specify the required experience level using this logical flow:\n"
        "1. Look for explicit statements about years of experience or seniority level.\n"
        "2. If years are specified, use the exact number (e.g., '3 years').\n"
        "3. If a range is given, use the full range (e.g., '3-5 years').\n"
        "4. For seniority levels, use these categories:\n"
        "   - Entry-level: 0-2 years or explicitly stated as entry-level\n"
        "   - Mid-level: 3-5 years or explicitly stated as mid-level\n"
        "   - Senior-level: 6+ years or explicitly stated as senior-level\n"
        "5. If both years and level are provided, include both (e.g., 'Senior-level, 8+ years').\n"
        "6. If no clear experience requirement is stated, use 'Not specified'.\n"
        "Examples: '2 years', 'Entry-level', 'Senior-level, 10+ years', 'Not specified'"
    )
    )

    degree_required: str = Field(
        default="",
        description=(
            "Determine the required degree using this process:\n"
            "1. Check for explicit degree requirements in the job description.\n"
            "2. If a specific degree is mentioned, include the level and field (e.g., 'Bachelor's in Computer Science').\n"
            "3. If multiple degrees are accepted, list all (e.g., 'Bachelor's or Master's in Engineering').\n"
            "4. For general requirements, use phrases like 'Any relevant degree' or 'Degree in a related field'.\n"
            "5. If no degree is required but it's explicitly stated, use 'No degree required'.\n"
            "6. If education requirements are not mentioned, use 'Not specified'.\n"
            "7. Include any alternatives to degrees if mentioned (e.g., 'Degree or equivalent experience').\n"
            "Examples: 'Bachelor's in Computer Science', 'Any STEM degree', 'No degree required', 'Not specified'"
        )
        )

    visa_sponsorship: str = Field(
        default="",
        description=(
            "Determine the visa sponsorship status using this logical approach:\n"
            "1. Look for explicit statements about visa sponsorship or work authorization requirements.\n"
            "2. If sponsorship is offered, use 'Visa sponsorship available'.\n"
            "3. If sponsorship is explicitly not offered, use 'No visa sponsorship'.\n"
            "4. If the job requires existing work authorization, use 'Must have existing work authorization'.\n"
            "5. If the posting is silent on the issue, use 'Not specified'.\n"
            "6. If there are conditions or limitations, include them (e.g., 'Sponsorship available for certain visas only').\n"
            "Examples: 'Visa sponsorship available', 'No visa sponsorship', 'Must have existing work authorization', 'Not specified'"
        )
    )

    notice_period: str = Field(
        default="",
        description=(
            "Specify the required notice period using this process:\n"
            "1. Check if the job posting mentions a specific notice period requirement.\n"
            "2. If stated, use the exact period mentioned (e.g., '2 weeks', '1 month').\n"
            "3. If a range is given, use the full range (e.g., '2-4 weeks').\n"
            "4. If the posting asks for 'immediate joining', use 'Immediate'.\n"
            "5. If flexible or negotiable, state this (e.g., 'Flexible, up to 1 month').\n"
            "6. If not mentioned in the posting, use 'Not specified'.\n"
            "7. If the posting states no notice period is required, use 'No notice period required'.\n"
            "Examples: '2 weeks', '1-3 months', 'Immediate', 'Flexible', 'Not specified'"
        )
    )

    duration: str = Field(
        default="",
        description=(
            "Determine the job duration using this logical flow:\n"
            "1. Look for explicit statements about the job's timeframe or nature (temporary, permanent, etc.).\n"
            "2. For temporary positions, specify the exact duration if given (e.g., '6 months', '1 year contract').\n"
            "3. For ongoing roles, use 'Permanent' or 'Ongoing'.\n"
            "4. If project-based, specify this and include duration if known (e.g., 'Project-based, approximately 9 months').\n"
            "5. If the duration is extendable or has the possibility of becoming permanent, include this information.\n"
            "6. If not clearly stated, use 'Not specified'.\n"
            "7. For seasonal jobs, specify the season and year if given.\n"
            "Examples: '6-month contract', 'Permanent', 'Project-based, 1 year with possibility of extension', 'Seasonal (Summer 2024)', 'Not specified'"
        )
    )

    rate: str = Field(
        default="",
        description=(
            "Specify the rate of pay using this process:\n"
            "1. Identify if the rate is provided as hourly, annual salary, or project-based fee.\n"
            "2. If a specific amount is given, use the exact figure and specify the period (e.g., '$25/hour', '$75,000/year').\n"
            "3. If a range is provided, use the full range (e.g., '$60,000 - $80,000/year').\n"
            "4. For project-based roles, specify the total project fee if given.\n"
            "5. If the rate is dependent on experience, indicate this (e.g., 'Competitive salary based on experience').\n"
            "6. If benefits are mentioned as part of compensation, include a note about this.\n"
            "7. If no specific rate is provided, use 'Not specified'.\n"
            "8. If the posting states the rate is negotiable, include this information.\n"
            "Examples: '$30-$40/hour', '$80,000-$100,000/year', 'Project fee: $10,000', 'Competitive salary + benefits', 'Not specified'"
        )
    )

class Skills(BaseModel):
    core: List[str] = Field(
        default_factory=list,
        description=(
            "Identify core skills using this process:\n"
            "1. Analyze the job description for skills mentioned as 'essential', 'required', or 'must-have'.\n"
            "2. Look for skills that appear in the job title or are repeatedly emphasized.\n"
            "3. Consider skills that are fundamental to the primary job functions.\n"
            "4. Typically include 3-5 core skills, unless the job is highly specialized.\n"
            "5. Use specific, industry-standard terms (e.g., 'Python' instead of 'programming').\n"
            "6. If a skill is listed as 'X or Y', include both as separate core skills.\n"
            "7. Do not include soft skills or general attributes (e.g., 'team player') as core skills.\n"
            "Example: ['Java', 'Spring Framework', 'SQL', 'RESTful APIs'] for a Java Developer position."
        )
    )

    primary: List[str] = Field(
        default_factory=list,
        description=(
            "Determine primary skills through this logical flow:\n"
            "1. Identify skills mentioned as 'important', 'preferred', or 'strongly desired'.\n"
            "2. Include skills that are frequently mentioned but not labeled as 'essential'.\n"
            "3. Consider skills that support or enhance the core skills.\n"
            "4. Look for skills related to secondary job functions or responsibilities.\n"
            "5. Aim for 5-8 primary skills, depending on the job's complexity.\n"
            "6. Include domain-specific knowledge or technologies relevant to the role.\n"
            "7. Can include some advanced or specialized versions of core skills.\n"
            "Example: ['Docker', 'Kubernetes', 'Microservices Architecture', 'JUnit', 'Maven'] for a Java Developer."
        )
    )

    secondary: List[str] = Field(
        default_factory=list,
        description=(
            "Identify secondary skills using this approach:\n"
            "1. Look for skills mentioned as 'nice-to-have', 'bonus', or 'a plus'.\n"
            "2. Include skills that are mentioned briefly or in passing.\n"
            "3. Consider skills that might be useful for future growth in the role.\n"
            "4. Include relevant soft skills or general technical skills.\n"
            "5. Add skills that are common in the industry but not specific to this role.\n"
            "6. Don't limit the number, but typically ranges from 5-10 secondary skills.\n"
            "7. Can include skills that are not directly related but potentially beneficial.\n"
            "Example: ['Agile Methodologies', 'GraphQL', 'CI/CD', 'Cloud Platforms', 'Technical Writing'] for a Java Developer."
        )
    )

    all: List[str] = Field(
        default_factory=list,
        description=(
            "Compile a comprehensive list of all skills using this process:\n"
            "1. Start by combining all skills from core, primary, and secondary lists.\n"
            "2. Review the entire job description to catch any missed skills.\n"
            "3. Include all technical skills, technologies, and tools mentioned.\n"
            "4. Add relevant soft skills and general competencies.\n"
            "5. Include industry-specific knowledge areas or certifications.\n"
            "6. Ensure no duplication; each skill should appear only once.\n"
            "7. Maintain consistent terminology and specificity across all skills.\n"
            "8. Order the skills from most to least important if possible.\n"
            "Note: This list should be exhaustive, typically containing 15-30 skills or more, depending on the job's complexity."
        )
    )

    with_experience: List[str] = Field(
        default_factory=list,
        description=(
            "Identify skills requiring prior experience using this logical flow:\n"
            "1. Look for phrases like 'X years of experience in...' or 'proven experience with...'.\n"
            "2. Include skills where the job asks for demonstrable proficiency or expertise.\n"
            "3. Consider skills where portfolio examples or past projects are requested.\n"
            "4. Include skills tied to specific experience levels (e.g., 'senior-level expertise in...').\n"
            "5. Look for skills where the job asks for leadership or mentoring abilities.\n"
            "6. Include skills where advanced or in-depth knowledge is explicitly required.\n"
            "7. If a skill is listed in core or primary and requires experience, include it here.\n"
            "Example: ['Java (5+ years)', 'Spring Framework', 'Team Leadership'] for a Senior Java Developer position.\n"
            "Note: This list may be shorter than others, focusing only on skills with explicit experience requirements."
        )
    )


class JobInformation(BaseModel):
    company: str = Field(
        default="",
        description=(
            "Determine the company name using this process:\n"
            "1. Look for explicit mentions of the company name in the job posting.\n"
            "2. If multiple company names are mentioned (e.g., in case of staffing agencies), use the following priority:\n"
            "   a) The company where the employee will actually work.\n"
            "   b) The hiring company, if different from the staffing agency.\n"
            "3. Use the full official company name, avoiding abbreviations unless they are part of the official name.\n"
            "4. Do not include legal entity types (e.g., 'Inc.', 'LLC') unless they are always used as part of the company's name.\n"
            "5. For well-known companies, use their commonly recognized name (e.g., 'Google' instead of 'Alphabet Inc.').\n"
            "6. If the company name is not explicitly stated, leave this field empty.\n"
            "Example: 'Amazon', 'Apple', 'Startup XYZ'\n"
            "Note: Accuracy in company name is crucial for job seekers to identify the employer correctly."
        )
    )

    job_title: str = Field(
        default="",
        description=(
            "Specify the job title using this logical flow:\n"
            "1. Use the exact job title as stated in the job posting.\n"
            "2. If multiple titles are listed, use the following priority:\n"
            "   a) The title that appears first or is most emphasized.\n"
            "   b) The more specific or senior title if multiple levels are mentioned.\n"
            "3. Maintain consistent capitalization (typically title case).\n"
            "4. Include level or seniority if it's part of the official title (e.g., 'Senior', 'Lead', 'Junior').\n"
            "5. If the job posting uses internal titles, consider using a more standard industry title in parentheses.\n"
            "6. Do not add extra words not included in the original title.\n"
            "7. If no clear title is provided, use the most accurate description based on the job responsibilities.\n"
            "Examples: 'Software Engineer', 'Senior Marketing Manager', 'Data Scientist (Machine Learning Specialist)'\n"
            "Note: The job title should accurately reflect the position as described in the posting."
        )
    )

    location: str = Field(
        description=(
            "Determine the job location using this process:\n"
            "1. Use the information from the 'full_location' field to construct this field.\n"
            "2. The format should be 'City, State, USA' for US locations.\n"
            "3. If the job is entirely remote with no specific location requirement:\n"
            "   a) Use only 'USA' as the location.\n"
            "   b) Do not include the term 'Remote' in this field.\n"
            "4. For non-US locations, use the format 'City, Country'.\n"
            "5. If a specific office location is mentioned for a primarily remote job, use that location.\n"
            "6. Do not include multiple locations in this field; choose the primary or first-mentioned location.\n"
            "7. Ensure consistency between this field and the 'full_location' field.\n"
            "Examples: 'New York, NY, USA', 'San Francisco, CA, USA', 'USA' (for remote jobs), 'London, United Kingdom'\n"
            "Note: This field is for quick reference; detailed location info is in 'full_location'."
        )
    )

    full_location: Location = Field(
        default_factory=Location,
        description="Detailed location information as defined in the Location class."
    )

    job_details: JobDetails = Field(
        default_factory=JobDetails,
        description="Comprehensive job details as defined in the JobDetails class."
    )

    skills: Skills = Field(
        default_factory=Skills,
        description="Required skills for the job as defined in the Skills class."
    )

    job_type: List[str] = Field(
        default_factory=list,
        description=(
            "Determine the job type(s) using this logical process:\n"
            "1. Analyze the job description for explicit mentions of work arrangement.\n"
            "2. Categorize into one or more of: 'remote', 'onsite', 'hybrid'.\n"
            "3. Use the following criteria:\n"
            "   a) 'Remote': If the job can be done entirely from a location of choice.\n"
            "   b) 'Onsite': If physical presence at a specific location is required full-time.\n"
            "   c) 'Hybrid': If the job combines remote and onsite work.\n"
            "4. If multiple types are possible, include all that apply.\n"
            "5. For jobs that offer flexibility, include all relevant types (e.g., ['onsite', 'hybrid']).\n"
            "6. If the job type is not explicitly stated, infer from context clues in the description.\n"
            "7. If truly unable to determine, leave the list empty.\n"
            "Examples: ['remote'], ['onsite'], ['hybrid'], ['onsite', 'hybrid']\n"
            "Note: This field should reflect the actual work arrangement, not just mentioned possibilities."
        )
    )

    contact_person: str = Field(
        default="",
        description=(
            "Identify the contact person using this approach:\n"
            "1. Look for explicit mentions of a contact person in the job posting.\n"
            "2. If multiple names are mentioned, prioritize:\n"
            "   a) The person designated for job-related queries.\n"
            "   b) The hiring manager or recruiter, if specified.\n"
            "3. Use the full name if provided, in the format 'First Last'.\n"
            "4. If only a first name or last name is given, use what's available.\n"
            "5. Do not include titles (e.g., Mr., Ms., Dr.) unless they are part of the name in the posting.\n"
            "6. If no specific person is mentioned, leave this field empty.\n"
            "7. Do not infer or create a name if none is provided.\n"
            "Examples: 'John Smith', 'Sarah', 'Taylor, J.', ''\n"
            "Note: Respect privacy by only using publicly provided contact information."
        )
    )

    email: str = Field(
        default="",
        description=(
            "Specify the contact email using this logical flow:\n"
            "1. Look for an email address specifically provided for job inquiries.\n"
            "2. If multiple email addresses are given, prioritize:\n"
            "   a) The address designated for applications or questions.\n"
            "   b) The most specific email (e.g., a person's email over a general one).\n"
            "3. Ensure the email address is correctly formatted (contains '@' and a domain).\n"
            "4. Do not modify or 'clean up' the email address; use it exactly as provided.\n"
            "5. If no email is provided but a web form is mentioned, leave this field empty.\n"
            "6. Do not include any additional text or instructions with the email address.\n"
            "7. If truly no contact method is provided, leave this field empty.\n"
            "Examples: 'jobs@company.com', 'john.smith@company.com', ''\n"
            "Note: Only use email addresses explicitly provided in the job posting."
        )
    )

    jd: str = Field(
        default="",
        description=(
            "Extract the complete job description using this process:\n"
            "1. Identify the start and end of the actual job description within the email content.\n"
            "2. Include all relevant information about the job, such as:\n"
            "   - Job title and company name\n"
            "   - Location and work arrangement (remote/onsite/hybrid)\n"
            "   - Responsibilities and requirements\n"
            "   - Qualifications and skills needed\n"
            "   - Employment type and duration\n"
            "   - Compensation and benefits information (if provided)\n"
            "   - Application instructions\n"
            "3. Maintain the original formatting as much as possible, including paragraphs and bullet points.\n"
            "4. Do not include:\n"
            "   - Email headers or footers\n"
            "   - Personal messages from the sender\n"
            "   - Recruiter contact information (unless it's part of the application process)\n"
            "   - Confidentiality disclaimers or email signatures\n"
            "5. If the job description is in a different language, include it as-is without translation.\n"
            "6. Do not summarize or modify the content; extract it verbatim.\n"
            "7. If the job description seems incomplete, include only what is provided without adding assumptions.\n"
            "Note: The goal is to capture the complete, unaltered job description as it appears in the original posting."
        )
    )

class JobResponse(BaseModel):
    status: int = Field(default=200, description="The status code of the response") 
    data: JobInformation = Field(description="The job information extracted from the job posting")


class JobDetailsExtractorService:
    def __init__(self):
        self.model = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                                temperature=os.getenv('OPENAI_TEMPERATURE'),
                                model_name=os.getenv('OPENAI_MODEL'),
                                top_p=os.getenv('OPENAI_TOP_P'))        
        #self.model = OllamaLLM(model="qwen2.5:7b-instruct-q2_K")

        self.parser = JsonOutputParser(pydantic_object=JobResponse)
        
        self.prompt = PromptTemplate(
            template="""Analyze the provided job posting information using the following logical process:

        1. Initial Parsing:
        a) Read the email subject carefully, noting any key information like job title, company name, or location.
        b) Thoroughly examine the job description, identifying sections for responsibilities, requirements, benefits, etc.

        2. Information Extraction:
        a) Company: Identify the hiring company name, prioritizing the actual employer over any recruiting agency.
        b) Job Title: Extract the exact title, maintaining original capitalization and including any seniority level.
        c) Location: Determine the job location, noting if it's remote, onsite, or hybrid.
        d) Full Location: Break down the location into city, state, and country components.
        e) Job Details: Extract information on employment type, experience required, education requirements, etc.
        f) Skills: Categorize skills into core, primary, and secondary, ensuring all mentioned skills are included in the "all" array.
        g) Job Type: Classify as remote, onsite, or hybrid based on the description.
        h) Contact Information: Note any provided contact person and email address.

        3. Data Validation:
        a) Ensure all extracted information directly corresponds to content in the subject or description.
        b) Do not infer information unless explicitly instructed and there's strong contextual evidence.
        c) Use empty strings for missing text fields, null for unknown boolean fields, and empty lists for list fields without data.

        4. Special Considerations:
        a) Skills: Include ALL mentioned skills in the "all" array, even if not categorized elsewhere.
        b) Job Description: Include the full, unmodified job description in the "jd" field.

        5. Output Formatting:
        a) Structure the response as a JSON object.
        b) Include a "status" field set to 200.
        c) Nest all job information under a "data" field.
        d) Follow the structure defined in the format instructions precisely.

        {format_instructions}

        Email Subject: {subject}
        Job Description: {description}

        Provide your response as a JSON object only, adhering to the structure outlined above.
        """,
            input_variables=["subject", "description"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

        self.chain = self.prompt | self.model | self.parser

    def extract_job_details(self, subject, description) -> Dict:
        logger.info("Extracting job details")
        job_response = self.chain.invoke({
            "subject": subject,
            "description": description
        })
        
        #Add aditional fields
        job_response["data"]["source"] = "Email"
        # job_response["data"]["date_posted"] = message.date
        # job_response["data"]["unique_id"] = message.id
        # #job_response["data"]["email"] =  message.sender
        job_response["data"]["emp_type"] = job_response["data"]["job_details"]["employment_type"]
        return job_response["data"]

# Example usage
if __name__ == "__main__":
    extractor = JobDetailsExtractorService()
    
    subject = "Immediate Need for  PostgreSQL DBA which is located in Plano, TX (Onsite)"
    description = """ From:

		                                   Abdul Rahman,

		                                   ConvexTech                                            

									       abdul.rahman@convextech.com

									       Reply to: Â Â abdul.rahman@convextech.com
								Hi,Hope you are doing great,Â Please go through the Requirement and let me know if youâ€™re interested. And if you are comfortable with the requirement please attach your updated profileÂ Position Title: PostgreSQL DBA/ArchitectLocation: Plano, TX (Onsite)Â  Expected Duration: 06 months Contract minimum possibly longer.Visa acceptance: USC or Green card Holders only.Rate:Â  $58/hr. on C2CÂ Â Job Description:Technical Requirements:â€¢ PostgreSQL database Administration, in-depth hands-on experience is a must.â€¢ Good experience PostgreSQL DBA activities.â€¢ Strong experience in PostgreSQL Performance Tuning and Optimizationâ€¢ Hands on experience backup strategy and high availability in production databasesâ€¢ Skilled at optimizing large, complicated SQL statements in PostgreSQL.â€¢ Strong experience in performance tuning on SQL queries.â€¢ Experience in table partitionsâ€¢ Strong experience in JOINS and Sub Queriesâ€¢ Skilled at optimizing large, complicated SQL statements and stored procedures, functions.â€¢ Strong with common database procedures such as upgrade, patch, recovery, migration, High Availabilityâ€¢ Ensure performance, security, and availability of databases.â€¢ Need to have strong experience in PostgreSQL HIGH Availability Solutions.â€¢ Design and develop PostgreSQL systems including statements, data modelling, tables, views, and indexes.â€¢ Knowledge of best practices when dealing with relational databases PostgreSQL.â€¢ Setting up the configuration parameters in PostgreSQL Database.Â Qualifications:â€¢ 10+ years of experience in PostgreSQL Administrationâ€¢ Extensive experience in PostgreSQL SQL and PLSQL (PG/PLSQL) Coding.â€¢ Experience in conversion of ETL mappings from Oracle to Postgres.â€¢ Good analytical and problem-solving skills for design, creation, and testing of programs.â€¢ Good communication skills to interact with team members, support personnel, and provide technical guidance and expertise to customers and management.â€¢ Good interpersonal skills to interact with customers and team members.â€¢ Ability to work in a self-directed work environment.â€¢ Ability to work in a team environment.Â ConvexTechAbdul RahmanTalent Acquisition | Staffing & RecruitmentEmail: abdul.rahman@convextech.com 11200 Broadway Suite 2743, Pearland TX 77584 USAhttps://convextech.com/Â Note: The content of this email is confidential and intended for the recipient specified in the message only. It is strictly forbidden to share any part of this message with any third party, without a written consent of the sender. If you received this message by mistake, please reply to this message and follow with its deletion, so that we can ensure such a mistake does not occur in the future.Â 
"""
    
    result = extractor.extract_job_details(subject, description)
    print(result)