from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="models",         
    repo_id="Subh737/resume-analyzer-model",
    repo_type="model"
)

print("Models folder uploaded successfully!")