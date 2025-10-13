import os
import glob
from pathlib import Path
from huggingface_hub import InferenceClient
import requests



def generate_resume_tags(client: InferenceClient, tex_file_path: str) -> dict:
   
    with open(tex_file_path, 'r', encoding='utf-8') as f:
        resume_content = f.read()
    PREDEFINED_TAGS = [
    "Technology",
    "Business & Finance",
    "Engineering",
    "Design & User Experience",
    "Freelance",
    "Corporate",
    "Startup",
    "Software Development"
]

    system_instructions = "You are an expert HR analyst who is a perfect instruction follower."
    user_instructions = f"""Analyze the following LaTeX resume content and assign relevant tags ONLY from this list:
{', '.join(PREDEFINED_TAGS)}

Your response MUST be a single line of comma-separated tags and nothing else.
For example: Technology, Software Development, Startup

--- RESUME CONTENT ---
{resume_content}
--- END OF CONTENT ---"""
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_instructions},
    ]

    response = client.chat_completion(
        messages=messages,
        max_tokens=50,
        temperature=0.1
    )
    response_text = response.choices[0].message.content.strip()
    #llm_output = response.choices[0].message.content.strip()
    
    validated_tags = [
        tag.strip() for tag in response_text.split(',') 
        if tag.strip() in PREDEFINED_TAGS
    ]

    return {
        "filepath": tex_file_path,
        "content": resume_content,
        "tags": validated_tags
    }

# --- Function 2: Manages the loop and CALLS the worker function ---
def process_resumes_in_folder(directory_path: str):
    """
    Finds all .tex files in a directory and processes them one by one.
    """
    MODEL_NAME= "meta-llama/Meta-Llama-3-8B-Instruct"
    client = InferenceClient(model=MODEL_NAME)# Initialize the client once
    tex_files = glob.glob(os.path.join(directory_path, '*.tex'))
   
    
  
    all_results = []
    # Start the loop to go through the list of files
    for file_path in tex_files:
        try:
           
            result = generate_resume_tags(client, file_path)
            all_results.append(result)
            print(f"File: {os.path.basename(result['filepath'])}, Tags: {result['tags']}")
            
        except Exception as e:
            print(f"Failed to process {os.path.basename(file_path)}: {e}")
            
    return all_results

if __name__ == "__main__":
    script_file_path = Path(__file__).resolve()

  
    project_root = script_file_path.parent.parent.parent

    
    FOLDER_TO_PROCESS = project_root / "templates"
    #FOLDER_TO_PROCESS = "../../templates"
    processed_data = process_resumes_in_folder(FOLDER_TO_PROCESS)

   
