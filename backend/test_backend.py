from chatbot import InteractiveRAGChatbot

# Test chatbot data loading
print("Testing chatbot data loading...")
bot = InteractiveRAGChatbot()
bot.load_all_data()

# Check contexts
contexts_status = {}
for k, v in bot.contexts.items():
    contexts_status[k] = 'Loaded' if v['index'] is not None else 'Failed'

print("Context loading status:")
for context, status in contexts_status.items():
    print(f"  {context}: {status}")

# Test a simple query
print("\nTesting query processing...")
try:
    response = bot.answer_query("Hello")
    print(f"Query response: {response[:100]}...")
except Exception as e:
    print(f"Query failed: {str(e)}")

print("Test completed!") 