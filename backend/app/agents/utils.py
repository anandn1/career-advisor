"""
Utility file for shared components, like the Resume Parser.
This allows any agent to import and use these tools.
"""

import os
import re
import PyPDF2
from pathlib import Path

class ResumeParser:
    """Parse resume files and extract skills"""
    
    # A comprehensive set of common skills
    COMMON_SKILLS = {
        # Programming Languages
        'python', 'javascript', 'java', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
        'typescript', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',
        
        # Web Technologies
        'html', 'css', 'react', 'react.js', 'vue', 'vue.js', 'angular', 'angular.js',
        'node.js', 'express', 'django', 'flask', 'fastapi', 'spring', 'asp.net',
        
        # Databases
        'sql', 'mysql', 'postgresql', 'mongodb', 'cassandra', 'redis', 'elasticsearch',
        'dynamodb', 'firestore', 'sqlite',
        
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 'github',
        'ci/cd', 'terraform', 'ansible', 'vagrant', 'aws lambda', 'azure functions',
        
        # Data & AI
        'machine learning', 'deep learning', 'nlp', 'natural language processing', 'cv', 'computer vision',
        'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'pandas', 'numpy', 'scipy',
        'spark', 'hadoop', 'kafka', 'data analysis', 'data science', 'data visualization',
        
        # APIs & Architecture
        'rest', 'restful', 'restful api', 'restful apis', 'graphql', 'soap', 'microservices',
        'grpc', 'websocket', 'api', 'apis',
        
        # Tools & Practices
        'git', 'agile', 'agile methodology', 'agile methodologies', 'scrum', 'kanban',
        'jira', 'confluence', 'postman', 'swagger', 'linux', 'bash', 'shell scripting',
        
        # Soft Skills
        'communication', 'teamwork', 'team collaboration', 'problem-solving',
        'leadership', 'collaboration', 'project management', 'time management', 'mentoring',
        
        # Testing
        'unit testing', 'integration testing', 'automated testing', 'jest', 'pytest',
        'selenium', 'junit', 'cypress', 'mocha',
        
        # Other
        'next.js', 'laravel', 'automated testing', 'user experience', 'ux', 'ui', 'user interface',
        'figma', 'sketch', 'blockchain'
    }
    
    @staticmethod
    def extract_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"[ERROR] Error extracting PDF: {e}")
            return ""
    
    @staticmethod
    def extract_from_txt(file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"[ERROR] Error extracting TXT: {e}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Extract text based on file type"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            return cls.extract_from_pdf(file_path)
        elif file_ext == '.txt':
            return cls.extract_from_txt(file_path)
        else:
            raise ValueError(f"[ERROR] Unsupported file type: {file_ext}. Use PDF or TXT")
    
    @classmethod
    def extract_skills(cls, resume_text: str) -> list:
        """Extract skills from resume text"""
        resume_lower = resume_text.lower()
        found_skills = set()
        
        # Find skills by matching common skills list
        for skill in cls.COMMON_SKILLS:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, resume_lower):
                # Use title case for consistency
                found_skills.add(skill.title()) 
        
        return sorted(list(found_skills))
    
    @classmethod
    def parse_resume(cls, file_path: str) -> dict:
        """Main method to parse resume and extract skills"""
        print(f"\n[RESUME PARSER] Reading resume from: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"[ERROR] Resume file not found: {file_path}")
            return {"skills": [], "error": "File not found"}
        
        # Extract text
        resume_text = cls.extract_text(file_path)
        
        if not resume_text:
            print("[ERROR] Could not extract text from resume")
            return {"skills": [], "error": "Could not extract text"}
        
        print(f"[RESUME PARSER] Extracted {len(resume_text)} characters from resume")
        
        # Extract skills
        skills = cls.extract_skills(resume_text)
        print(f"[RESUME PARSER] Found {len(skills)} skills: {', '.join(skills)}")
        
        return {
            "skills": skills,
            "error": None
        }