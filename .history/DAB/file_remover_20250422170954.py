import os

# List of files to be removed
files_to_remove = [
    'Prompt Extracting information from blog posts.md',
    'Prompting.md',
    'Prompt Engineering',
    'Covariance vs Correlation.md',
    'Cost-Sensitive Analysis.md',
    'F1 Score.md',
    'Comparing_Ensembles.py.md',
    'Covariance Structures.md',
    'Handling Missing Data.md',
    'Mermaid.md',
    'ER Diagrams.md',
    'Data Mining - CRISP.md',
    'Software Development Portal.md',
    'Data Orchestration.md',
'Dataview.md'
'ML_Tools.md',
'Queries.md'
]

# Directory where the files are located (replace with your folder path)
directory_path = './your-folder-path'  # Change this to your actual folder path

# Remove files
for file_name in files_to_remove:
    file_path = os.path.join(directory_path, file_name)
    
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"✅ Successfully deleted {file_name}")
        except Exception as e:
            print(f"❌ Failed to delete {file_name}: {e}")
    else:
        print(f"⚠️ {file_name} not found in the directory.")
