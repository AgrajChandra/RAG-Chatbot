from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import InteractiveRAGChatbot
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the chatbot
chatbot = InteractiveRAGChatbot()
chatbot.load_all_data()  # Load all data at startup

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get response from chatbot
        response = chatbot.answer_query(data['message'])
        
        return jsonify({
            'response': response
        })
    except Exception as e:
        print("Error:", str(e))
        print(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/api/daily-update', methods=['POST'])
def add_daily_update():
    try:
        data = request.json
        required_fields = [
            'date', 'team', 'sub_team', 'project', 'present_members',
            'summary', 'tasks_completed', 'blockers', 'next_steps'
        ]
        
        # Check if all required fields are present
        if not all(field in data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in data]
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
        
        # Convert arrays to comma-separated strings
        present_members_str = ','.join(data['present_members']) if isinstance(data['present_members'], list) else data['present_members']
        tasks_completed_str = ','.join(data['tasks_completed']) if isinstance(data['tasks_completed'], list) else data['tasks_completed']
        blockers_str = ','.join(data['blockers']) if isinstance(data['blockers'], list) else data['blockers']
        
        # Add the update using the chatbot's method
        success = chatbot.append_update(
            date=data['date'],
            team=data['team'],
            sub_team=data['sub_team'],
            project=data['project'],
            present_members=present_members_str,
            summary=data['summary'],
            tasks_completed=tasks_completed_str,
            blockers=blockers_str,
            next_steps=data['next_steps']
        )
        
        if success:
            # Reload data after successful update
            chatbot.load_all_data()
            return jsonify({'message': 'Update added successfully'})
        else:
            return jsonify({'error': 'Failed to add update. Please check if daily_updates.json exists and is writable.'}), 500
            
    except Exception as e:
        print("Error:", str(e))
        print(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True) 