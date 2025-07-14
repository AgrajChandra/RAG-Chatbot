#!/usr/bin/env python3

"""
Test script to diagnose chatbot issues
"""

import sys
import traceback
from chatbot import InteractiveRAGChatbot

def test_chatbot():
    print("🔍 Testing EOXS Chatbot...")
    
    try:
        print("📁 Step 1: Creating chatbot instance...")
        bot = InteractiveRAGChatbot()
        print("✅ Chatbot instance created successfully")
        
        print("📊 Step 2: Loading all data...")
        bot.load_all_data()
        print("✅ Data loaded successfully")
        
        # Check if contexts are loaded
        print("🔍 Step 3: Checking loaded contexts...")
        for context_name, context_data in bot.contexts.items():
            index_status = "✅ Loaded" if context_data['index'] is not None else "❌ Failed"
            chunk_count = len(context_data['chunks']) if context_data['chunks'] else 0
            print(f"  {context_name}: {index_status} ({chunk_count} chunks)")
        
        print("💬 Step 4: Testing simple query...")
        response = bot.answer_query("What is EOXS?")
        print(f"✅ Query successful: {response[:100]}...")
        
        print("💬 Step 5: Testing employee query...")
        response2 = bot.answer_query("Who is Rajat Jain?")
        print(f"✅ Employee query successful: {response2[:100]}...")
        
        print("🎉 All tests passed! Chatbot is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print("📝 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_chatbot()
    sys.exit(0 if success else 1) 