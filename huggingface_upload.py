# Training শেষে এই code add করুন
model_name = "fahadali32/bangla-hate-speech-distilbert"

# Model push করুন with proper error handling
try:
    model.push_to_hub(model_name, commit_message="Bengali hate speech detection model")
    tokenizer.push_to_hub(model_name, commit_message="Tokenizer for Bengali hate speech detection")
    print(f"✅ Model uploaded successfully!")
    print(f"🔗 Model link: https://huggingface.co/{model_name}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
    print("Make sure you're logged in: huggingface-cli login")

# Alternative method with authentication check
from huggingface_hub import whoami

try:
    user_info = whoami()
    print(f"Logged in as: {user_info['name']}")
    
    # Upload with version info
    model.push_to_hub(
        model_name, 
        commit_message="Bengali hate speech detection - DistilBERT fine-tuned"
    )
    tokenizer.push_to_hub(
        model_name,
        commit_message="Tokenizer for Bengali hate speech detection"
    )
    
    print(f"✅ Model successfully uploaded to: https://huggingface.co/{model_name}")
    
except Exception as e:
    print(f"❌ Authentication or upload error: {e}")
    print("Run this first: !huggingface-cli login")
