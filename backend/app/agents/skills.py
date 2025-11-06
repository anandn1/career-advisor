import os
import requests
import base64
from pprint import pprint
from openai import OpenAI 
from dotenv import load_dotenv

import pdfplumber 
from docx import Document

load_dotenv()



GITHUB_ACCESS_TOKEN = os.environ.get('GITHUB_ACCESS_TOKEN')
HF_TOKEN = os.environ.get('HF_TOKEN') 


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)


def get_github_repo_languages(username: str, token: str = None) -> list[str]:
    """
    Fetches all unique languages (skills) used in a user's public repos.
    
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    repos_url = f"https://api.github.com/users/{username}/repos?per_page=100"
    response = requests.get(repos_url, headers=headers)

    if response.status_code != 200:
        print(f"Error: Could not fetch repo list. Status code: {response.status_code}")
        return [] # Return empty list on failure

    repos = response.json()
    all_languages = set() # Use a set to store unique languages

    for repo in repos:
        lang_url = repo['languages_url']
        lang_response = requests.get(lang_url, headers=headers)
        
        if lang_response.status_code == 200:
            languages = lang_response.json()
            for lang in languages.keys():
                all_languages.add(lang)
        else:
            pass 

    return list(all_languages)

def get_readme_content(username: str, token: str = None) -> str | None:
    """
    Fetches the raw text content of a user's profile README.md.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    readme_url = f"https://api.github.com/repos/{username}/{username}/contents/README.md"
    response = requests.get(readme_url, headers=headers)

    if response.status_code == 404:
        print(f"Info: User {username} does not have a profile README.")
        return None 
        
    if response.status_code != 200:
        print(f"Error: Could not fetch README. Status code: {response.status_code}")
        return None 

    try:
        content_base64 = response.json()['content']
        decoded_bytes = base64.b64decode(content_base64)
        readme_text = decoded_bytes.decode('utf-8')
        return readme_text
    except Exception as e:
        print(f"Error decoding README content: {e}")
        return None

def get_combined_skills_llm(
    repo_languages: list[str], 
    readme_text: str | None, 
) -> set[str]:
    """
    Sends the repo languages AND README text to the HF Router
    using the OpenAI client.
    """
    if not HF_TOKEN:
        print("Error: HF_TOKEN is not set. Cannot query LLM.")
        return set()
        
    

    lang_string = ", ".join(repo_languages)
    
    if not readme_text:
        readme_text = "No README content provided."

    system_prompt = """You are an expert tech skill extractor.
Your task is to create one single, deduplicated list of technical skills.

You must use two sources of information:
1.  **Repo Languages:** A list of languages found in the user's code.
2.  **Profile README:** The user's self-described profile.

Combine both sources to create a final, clean list of all unique skills.
Return *only* a single, simple, comma-separated list.
Do not add any explanation, categories, or preamble."""

    user_prompt = f"""Here is the data:
Repo Languages: "{lang_string}"
Profile README: "{readme_text}"
"""

    try:
        
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct", # The model ID
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        #  Get the response content
        generated_text = completion.choices[0].message.content
        
        
        skill_list = [
            skill.strip().strip("'\"") 
            for skill in generated_text.split(',')
        ]
        
        return set(skill for skill in skill_list if skill)

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return set()

def read_resume_file(file_path: str) -> str | None:
    """
    Reads a resume file (.pdf, .docx) and returns its text.
    """
    print(f"\n Reading resume file: {file_path} ")
    try:
        if file_path.endswith(".pdf"):
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        
        elif file_path.endswith(".docx"):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text

        else:
            print(f"Error: Unsupported file format: {file_path}")
            return None
            
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def extract_skills_from_resume(resume_text: str) -> set[str]:
    """
    Sends resume text to the HF Router to extract skills.
    """
    if not HF_TOKEN:
        print("Error: HF_TOKEN is not set. Cannot query LLM.")
        return set()
        

    
    system_prompt = """You are an expert tech recruiter and skill extractor.
Your task is to read a resume and extract all relevant technical skills from the given resume.
Focus on programming languages, frameworks, databases, tools, and technical concepts.
Return *only* a single, simple, comma-separated list.
Do not add any explanation, categories, or preamble."""

    user_prompt = f"Here is the resume text:\n\n{resume_text}"

    try:
        # This is the same API call as your other function
        completion = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=200 # You might need more tokens for a long resume
        )
        
        generated_text = completion.choices[0].message.content
        
    
        skill_list = [
            skill.strip().strip("'\"") 
            for skill in generated_text.split(',')
        ]
        
        return set(skill for skill in skill_list if skill)

    except Exception as e:
        print(f"Error calling LLM: {e}")
        return set()



# main function
def get_final_skills_data(username: str, resume_file_path: str = None) -> set[str]:
    """
    Main orchestrator to get skills from either a resume file or GitHub.
    """
    final_skills = set() # Initialize an empty set

    if resume_file_path:
        print(f" Processing Resume: {resume_file_path} ")
        resume_text = read_resume_file(resume_file_path)
        
        if resume_text:
            # Send the text to the LLM
            final_skills = extract_skills_from_resume(resume_text)
        else:
            print(f"Could not process resume file: {resume_file_path}")
            

    else:
        print(f" Processing GitHub User: {username} ")
        
        if not GITHUB_ACCESS_TOKEN:
            print("Warning: GITHUB_TOKEN not set. You may hit rate limits.\n")
    
        repo_langs = get_github_repo_languages(username, GITHUB_ACCESS_TOKEN)
    
        if not repo_langs:
            print(" No languages found in user's repos ")

        readme_content = get_readme_content(username, GITHUB_ACCESS_TOKEN)
    
        # Send to LLM
        final_skills = get_combined_skills_llm(repo_langs, readme_content)
    return final_skills


if __name__ == "__main__":
    
    if not HF_TOKEN:
        print("Error: HF_TOKEN not set. LLM analysis will be skipped.\n")
        exit()

    
    USERNAME_TO_FETCH = "Satyajeet-Das" 
    
   
    RESUME_FILE_PATH = None 
    

    
    skills = get_final_skills_data(USERNAME_TO_FETCH, RESUME_FILE_PATH)
    
    
    if skills:
        print("\n Extracted Skills ")
        pprint(sorted(list(skills)))
    else:
        print("\nLLM analysis complete, but no skills were extracted.")