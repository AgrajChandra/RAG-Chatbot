import shutil
import os
from chatbot import InteractiveRAGChatbot

print("Fixing FAISS cache...")

# Remove cache directory if it exists
cache_dir = "cache"
if os.path.exists(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        print(f"Removed existing cache directory: {cache_dir}")
    except Exception as e:
        print(f"Warning: Could not remove cache directory: {str(e)}")

# Recreate cache directory
os.makedirs(cache_dir, exist_ok=True)
print(f"Created new cache directory: {cache_dir}")

# Reinitialize and load data
print("Reinitializing chatbot...")
bot = InteractiveRAGChatbot()
bot.load_all_data()

# Test query processing
print("Testing query processing...")
try:
    response = bot.answer_query("What is EOXS?")
    print(f"Test query successful: {response[:100]}...")
except Exception as e:
    print(f"Test query failed: {str(e)}")

print("Cache fix completed!") 