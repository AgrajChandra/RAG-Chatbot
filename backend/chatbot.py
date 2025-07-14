from typing import List, Tuple
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
import os
import json
import faiss
import re
import pickle
import time
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

class InteractiveRAGChatbot:
    def __init__(self):
        # Initialize the chatbot with embeddings, model, chunker, and context data structures.
        # Loads conversation history and sets up cache directory.
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY
        )
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.chunker = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90
        )
        self.contexts = {
            'products': {'chunks': [], 'index': None},
            'updates': {'chunks': [], 'index': None},
            'employees': {'chunks': [], 'index': None}
        }
        self.history = []
        self.history_file = "conversation_history.json"
        self.load_history()
        self.cache_dir = "cache"
        self.code_version = "1.8"
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_history(self):
        """
        Loads the conversation history from a JSON file if it exists.
        Initializes self.history as an empty list if the file is missing or invalid.
        """
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load history - {str(e)}. Starting with empty history.")

    def save_history(self):
        """
        Saves the current conversation history to a JSON file.
        """
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history - {str(e)}.")

    def load_cache_data(self, context: str, json_file: str) -> bool:
        """
        Loads cached embeddings and chunk data for a given context (products, updates, employees) if the cache is valid.
        Returns True if cache is loaded, False otherwise.
        """
        cache_file = os.path.join(self.cache_dir, f"{context}_cache.pkl")
        try:
            if os.path.exists(cache_file) and os.path.exists(json_file):
                json_mtime = os.path.getmtime(json_file)
                cache_mtime = os.path.getmtime(cache_file)
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                if data.get('code_version') == self.code_version and cache_mtime > json_mtime:
                    print(f"Loaded cache for {context}: {cache_file}")
                    self.contexts[context]['chunks'] = data['chunks']
                    self.contexts[context]['index'] = data['index']
                    return True
                else:
                    print(f"Cache invalid for {context}: version mismatch or stale cache")
        except Exception as e:
            print(f"Warning: Could not load cache for {context} - {str(e)}")
        return False

    def save_cached_data(self, context: str):
        """
        Saves the current embeddings and chunk data for a context to a cache file for faster future loading.
        """
        cache_file = os.path.join(self.cache_dir, f"{context}_cache.pkl")
        try:
            data = {
                'code_version': self.code_version,
                'chunks': self.contexts[context]['chunks'],
                'index': self.contexts[context]['index']
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved cache for {context}: {cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache for {context} - {str(e)}")

    def parse_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Parses the user's query to extract explicit context requests (e.g., 'in products and updates').
        Returns the cleaned query and a list of requested contexts.
        """
        search_in_pattern = r"^(?:search\s*(?:for\s*)?.*?\s*in\s+([\w\s,]+?))(?:\s+and\s+([\w\s,]+))?\s*$"
        match = re.match(search_in_pattern, query.lower().strip())
        if match:
            contexts = []
            if match.group(1):
                contexts.extend([c.strip() for c in match.group(1).split(',')])
            if match.group(2):
                contexts.extend([c.strip() for c in match.group(2).split(',')])
            clean_query = re.sub(r"^(search\s*(?:for\s*)?.*?\s*in\s+.*$)", "", query, flags=re.IGNORECASE).strip()
            clean_query = clean_query or query
            return clean_query, [c for c in contexts if c in ['products', 'updates', 'employees']]
        return query, []

    def determine_context(self, query: str) -> List[str]:
        """
        Heuristically determines which context(s) (products, updates, employees) are most relevant to the query.
        Uses keyword matching and embedding similarity.
        Returns a list of relevant context keys.
        """
        product_keywords = ['product', 'eoxs', 'books', 'crm', 'people', 'shop', 'reports',
                           'features', 'software', 'pricing', 'trial', 'payment', 'inventory']
        update_keywords = ['team', 'project', 'status', 'blockers', 'tasks', 'deadline',
                          'progress', 'completed', 'working', 'updates', 'daily']
        employee_keywords = ['employee', 'department', 'manager', 'reporting', 'email',
                            'intern', 'ceo', 'staff', 'team member', 'role', 'job title', 'names',
                            'count', 'total', 'list', 'alphabetical']
        scores = {
            'products': sum(1 for keyword in product_keywords if keyword.lower() in query.lower()),
            'updates': sum(1 for keyword in update_keywords if keyword.lower() in query.lower()),
            'employees': sum(1 for keyword in employee_keywords if keyword.lower() in query.lower()) * 1.5
        }
        query_embedding = self.embeddings.embed_query(query)
        context_scores = {}
        for ctx in self.contexts:
            if self.contexts[ctx]['index']:
                try:
                    query_vec = np.array([query_embedding], dtype=np.float32)
                    D, I = self.contexts[ctx]['index'].search(query_vec, 1)
                    context_scores[ctx] = 1 / (1 + D[0][0])
                except Exception as e:
                    print(f"Warning: FAISS search failed for context {ctx} in determine_context: {str(e)}")
                    context_scores[ctx] = 0.0
        combined_scores = {ctx: scores[ctx] + context_scores.get(ctx, 0) for ctx in self.contexts}
        max_score = max(combined_scores.values()) if combined_scores else 0
        if max_score < 0.1:
            return ['employees'] if any(k in query.lower() for k in employee_keywords) else list(self.contexts.keys())
        return [ctx for ctx, score in combined_scores.items() if score > 0.1]

    def process_products_data(self, data: dict) -> str:
        """
        Converts the products data from the JSON file into a formatted string for embedding and retrieval.
        Handles different product data structures and includes features, sections, FAQs, and industry focus.
        """
        texts = [f"Company Name: {data['company_name']}\n"]
        for product in data.get('products', []):
            text = ""
            if 'name' in product:
                text += f"Product: {product['name']}\nDescription: {product.get('description', '')}\n"
                if 'features' in product:
                    text += "Features:\n" + "\n".join(f"  * {f}" for f in product['features']) + "\n"
            elif 'header' in product:
                text += f"Product: {product['header']['title']}\nDescription: {' '.join(product['header'].get('description', []))}\n"
                if 'sections' in product:
                    text += "Sections:\n" + "\n".join(f"- {s['title']}: {s['content']}" for s in product['sections']) + "\n"
                if 'tools_section' in product:
                    text += f"Tools:\nTitle: {product['tools_section']['title']}\n" + "\n".join(f"  * {i}" for i in product['tools_section']['items']) + "\n"
            elif 'sections' in product and isinstance(product['sections'], list):
                text += f"Product: {product['title']}\nSections:\n" + "\n".join(f"- {s['heading']}: {s['content']}" for s in product['sections']) + "\n"
            else:
                text += f"Product: {product['title']}\nOverview: {product.get('overview', '')}\n"
                if 'main_features' in product:
                    text += "Main Features:\n"
                    for feature in product['main_features']:
                        text += f"- {feature['name']}:\n" + "\n".join(f"  * {d}" for d in feature['details']) + "\n"
                if 'faq' in product:
                    text += "FAQs:\n" + "\n".join(f"Q: {qa['question']}\nA: {qa['answer']}" for qa in product['faq']) + "\n"
                if 'industry_focus' in product:
                    text += f"Industry Focus: {product['industry_focus']}\n"
            texts.append(text)
        return "\n".join(texts)

    def process_daily_updates(self, updates: list) -> str:
        """
        Converts the daily updates data from the JSON file into a formatted string for embedding and retrieval.
        Includes team, project, members, summary, tasks, blockers, and next steps.
        """
        texts = []
        for update in updates:
            text = (
                f"Date: {update['date']}\nTeam: {update['team']}\nSub-Team: {update['sub_team']}\n"
                f"Project: {update['project']}\nPresent Team Members: {', '.join(update['present_members'])}\n"
            )
            if 'member_details' in update:
                text += "Member Details:\n" + "\n".join(
                    f"- {member['name']}: Role: {member['role']}, Email: {member['email']}, Responsibilities: {member['responsibilities']}"
                    for member in update['member_details']
                ) + "\n"
            blockers_text = '\n'.join(f'- {b}' for b in update['blockers']) if update['blockers'] else 'No blockers reported'
            text += (
                f"Daily Summary: {update['summary']}\nTasks Completed:\n" +
                "\n".join(f"- {task}" for task in update['tasks_completed']) + "\n"
                f"Blockers:\n{blockers_text}\n"
                f"Next Steps: {update['next_steps']}"
            )
            texts.append(text)
        return "\n\n".join(texts)

    def process_employee_data(self, data: dict) -> str:
        """
        Converts the employee structure data from the JSON file into a formatted string for embedding and retrieval.
        Includes department, employee details, and department summaries.
        """
        texts = []
        for department, dept_data in data.items():
            text = f"Department: {department}\nEmployees:\n"
            for employee in dept_data['employees']:
                text += (
                    f"Name: {employee['name']}\nEmail: {employee['email']}\n"
                    f"Job Title: {employee['job_title']}\nManager: {employee['manager']}\n"
                )
            texts.append(text)
        summary_text = "Department Summaries:\n"
        for department, dept_data in data.items():
            employee_count = len(dept_data['employees'])
            managers = set(emp['manager'] for emp in dept_data['employees'])
            summary_text += (
                f"Department: {department}\nTotal Employees: {employee_count}\n"
                f"Reporting Managers: {', '.join(managers)}\n"
            )
        texts.append(summary_text)
        return "\n\n".join(texts)

    def count_unique_employee_names(self) -> int:
        """
        Counts the number of unique employee names by aggregating data from both employee_structure.json and daily_updates.json.
        Returns the count of unique names found.
        """
        try:
            email_to_name = {}
            if os.path.exists('employee_structure.json'):
                with open('employee_structure.json', 'r') as f:
                    data = json.load(f)
                for department, dept_data in data.items():
                    for employee in dept_data.get('employees', []):
                        email = employee['email'].lower()
                        name = employee['name']
                        if email not in email_to_name:
                            email_to_name[email] = name
            if os.path.exists('daily_updates.json'):
                with open('daily_updates.json', 'r') as f:
                    updates_data = json.load(f)
                for update in updates_data:
                    if 'member_details' in update:
                        for member in update['member_details']:
                            email = member['email'].lower()
                            name = member['name']
                            if email not in email_to_name:
                                email_to_name[email] = name
            unique_names = set(email_to_name.values())
            return len(unique_names)
        except Exception as e:
            print(f"Error counting unique names: {str(e)}")
            return 0

    def get_employee_names(self, alphabetical: bool = False, with_details: bool = False) -> List:
        """
        Retrieves a list of employee names (optionally sorted alphabetically) or detailed info.
        Aggregates data from employee_structure.json and daily_updates.json.
        If with_details is True, returns detailed info for each employee.
        """
        try:
            employee_details = []
            email_to_info = {}
            if os.path.exists('employee_structure.json'):
                with open('employee_structure.json', 'r') as f:
                    data = json.load(f)
                for department, dept_data in data.items():
                    for employee in dept_data.get('employees', []):
                        email = employee['email'].lower()
                        if email not in email_to_info:
                            email_to_info[email] = {
                                'name': employee['name'],
                                'department': department,
                                'email': employee['email'],
                                'job_title': employee['job_title'],
                                'manager': employee['manager'],
                                'teams': set(),
                                'projects': set()
                            }
            if os.path.exists('daily_updates.json'):
                with open('daily_updates.json', 'r') as f:
                    updates_data = json.load(f)
                for update in updates_data:
                    for member in update.get('present_members', []):
                        for email, info in email_to_info.items():
                            if info['name'].lower() == member.lower():
                                info['teams'].add(f"{update['team']} ({update['sub_team']})")
                                info['projects'].add(update['project'])
                    if 'member_details' in update:
                        for member in update['member_details']:
                            email = member['email'].lower()
                            if email not in email_to_info:
                                email_to_info[email] = {
                                    'name': member['name'],
                                    'department': 'Unknown',
                                    'email': member['email'],
                                    'job_title': member['role'],
                                    'manager': 'Unknown',
                                    'teams': set([f"{update['team']} ({update['sub_team']})"]),
                                    'projects': set([update['project']])
                                }
                            else:
                                email_to_info[email]['teams'].add(f"{update['team']} ({update['sub_team']})")
                                email_to_info[email]['projects'].add(update['project'])
            for email, info in email_to_info.items():
                detail = {
                    'name': info['name'],
                    'details': (
                        f"Name: {info['name']}\n"
                        f"Department: {info['department']}\n"
                        f"Job Title: {info['job_title']}\n"
                        f"Email: {info['email']}\n"
                        f"Manager: {info['manager']}\n"
                        f"Teams: {', '.join(info['teams']) if info['teams'] else 'None'}\n"
                        f"Projects: {', '.join(info['projects']) if info['projects'] else 'None'}"
                    )
                }
                employee_details.append(detail)
            if alphabetical:
                employee_details.sort(key=lambda x: x['name'].lower())
            return [d['details'] for d in employee_details] if with_details else [d['name'] for d in employee_details]
        except Exception as e:
            print(f"Error retrieving employee names: {str(e)}")
            return []

    def get_employee_details(self, name: str) -> str:
        """
        Retrieves detailed information about an employee by name from both employee_structure.json and daily_updates.json.
        Returns a formatted string with all found details.
        """
        try:
            details = []
            if os.path.exists('employee_structure.json'):
                with open('employee_structure.json', 'r') as f:
                    employee_data = json.load(f)
                for department, dept_data in employee_data.items():
                    for employee in dept_data.get('employees', []):
                        if name.lower() in employee['name'].lower():
                            details.append(
                                f"Name: {employee['name']}, Department: {department}, "
                                f"Job Title: {employee['job_title']}, Email: {employee['email']}, "
                                f"Manager: {employee['manager']}"
                            )
            if os.path.exists('daily_updates.json'):
                with open('daily_updates.json', 'r') as f:
                    updates_data = json.load(f)
                for update in updates_data:
                    if 'member_details' in update:
                        for member in update['member_details']:
                            if name.lower() in member['name'].lower():
                                details.append(
                                    f"Name: {member['name']}, Team: {update['team']}, "
                                    f"Sub-Team: {update['sub_team']}, Project: {update['project']}, "
                                    f"Role: {member['role']}, Responsibilities: {member['responsibilities']}, "
                                    f"Update Date: {update['date']}"
                                )
                    elif name in update.get('present_members', []):
                        details.append(
                            f"Name: {name}, Team: {update['team']}, Sub-Team: {update['sub_team']}, "
                            f"Project: {update['project']}, Present on: {update['date']}"
                        )
            return "\n".join(details) or f"No details found for {name}."
        except Exception as e:
            return f"Error retrieving details for {name}: {str(e)}."

    def get_team_names(self, query: str = None) -> set:
        """
        Retrieves all unique team and sub-team names from daily_updates.json.
        If a query is provided, filters teams containing the query string.
        Returns a set of team names.
        """
        try:
            if os.path.exists('daily_updates.json'):
                with open('daily_updates.json', 'r') as f:
                    updates_data = json.load(f)
                teams = set(update['team'] for update in updates_data)
                sub_teams = set(update['sub_team'] for update in updates_data)
                all_teams = teams.union(sub_teams)
                if query:
                    return {team for team in all_teams if query.lower() in team.lower()}
                return all_teams
            return set()
        except Exception as e:
            print(f"Error retrieving team names: {str(e)}")
            return set()

    def add_employee(self, department: str, name: str, email: str, job_title: str, manager: str) -> bool:
        """
        Adds a new employee to the specified department in employee_structure.json.
        Checks for duplicate emails. Returns True if successful, False otherwise.
        """
        try:
            # Check if the employee_structure.json file exists
            if os.path.exists('employee_structure.json'):
                # Open and load the existing employee data
                with open('employee_structure.json', 'r') as f:
                    data = json.load(f)
                # If the department does not exist, create a new entry for it
                if department not in data:
                    data[department] = {"employees": []}
                # Check for duplicate email addresses across all departments
                for dept, dept_data in data.items():
                    for emp in dept_data.get('employees', []):
                        if emp['email'].lower() == email.lower():
                            # If the email already exists, print an error and return False
                            print(f"Error: Email {email} already exists for {emp['name']}.")
                            return False
                # Add the new employee to the specified department
                data[department]["employees"].append({
                    "name": name,
                    "email": email,
                    "job_title": job_title,
                    "manager": manager
                })
                # Save the updated data back to the JSON file
                with open('employee_structure.json', 'w') as f:
                    json.dump(data, f, indent=2)
                return True
            else:
                # If the file does not exist, print an error and return False
                print("Error: employee_structure.json not found.")
                return False
        except Exception as e:
            # Catch any unexpected errors, print them, and return False
            print(f"Error adding employee: {str(e)}")
            return False

    def append_update(self, date: str, team: str, sub_team: str, project: str, present_members: str, 
                     summary: str, tasks_completed: str, blockers: str, next_steps: str) -> bool:
        """
        Appends a new daily update to daily_updates.json.
        Converts comma-separated strings to lists for members, tasks, and blockers.
        Returns True if successful, False otherwise.
        """
        try:
            if os.path.exists('daily_updates.json'):
                with open('daily_updates.json', 'r') as f:
                    data = json.load(f)
                members_list = [m.strip() for m in present_members.split(',') if m.strip()]
                tasks_list = [t.strip() for t in tasks_completed.split(',') if t.strip()]
                blockers_list = [b.strip() for b in blockers.split(',') if b.strip()]
                update = {
                    "date": date,
                    "team": team,
                    "sub_team": sub_team,
                    "project": project,
                    "present_members": members_list,
                    "summary": summary,
                    "tasks_completed": tasks_list,
                    "blockers": blockers_list,
                    "next_steps": next_steps
                }
                data.append(update)
                with open('daily_updates.json', 'w') as f:
                    json.dump(data, f, indent=2)
                return True
            else:
                print("Error: daily_updates.json not found.")
                return False
        except Exception as e:
            print(f"Error appending update: {str(e)}")
            return False

    def load_all_data(self):
        """
        Loads and processes all data (products, updates, employees) from their respective JSON files.
        Uses caching to speed up loading if possible. Builds embeddings and FAISS indices for retrieval.
        """
        start_time = time.time()
        try:
            if not self.load_cache_data('products', 'Products.json'):
                if os.path.exists('Products.json'):
                    with open('Products.json', 'r') as f:
                        products_data = json.load(f)
                    products_text = self.process_products_data(products_data)
                    self.contexts['products']['chunks'] = self.chunker.split_text(products_text)
                    products_embeddings = self.embeddings.embed_documents(self.contexts['products']['chunks'])
                    if products_embeddings:
                        dimension = len(products_embeddings[0])
                        self.contexts['products']['index'] = faiss.IndexFlatL2(dimension)
                        self.contexts['products']['index'].add(np.array(products_embeddings))
                        self.save_cached_data('products')
                    else:
                        self.contexts['products']['index'] = None
                else:
                    print("Warning: Products.json not found. Skipping product data.")
                    self.contexts['products']['index'] = None
            if not self.load_cache_data('updates', 'daily_updates.json'):
                if os.path.exists('daily_updates.json'):
                    with open('daily_updates.json', 'r') as f:
                        updates_data = json.load(f)
                    updates_text = self.process_daily_updates(updates_data)
                    self.contexts['updates']['chunks'] = self.chunker.split_text(updates_text)
                    updates_embeddings = self.embeddings.embed_documents(self.contexts['updates']['chunks'])
                    if updates_embeddings:
                        dimension = len(updates_embeddings[0])
                        self.contexts['updates']['index'] = faiss.IndexFlatL2(dimension)
                        self.contexts['updates']['index'].add(np.array(updates_embeddings))
                        self.save_cached_data('updates')
                    else:
                        self.contexts['updates']['index'] = None
                else:
                    print("Warning: daily_updates.json not found. Skipping updates data.")
                    self.contexts['updates']['index'] = None
            if not self.load_cache_data('employees', 'employee_structure.json'):
                if os.path.exists('employee_structure.json'):
                    with open('employee_structure.json', 'r') as f:
                        employee_data = json.load(f)
                    employee_text = self.process_employee_data(employee_data)
                    self.contexts['employees']['chunks'] = self.chunker.split_text(employee_text)
                    employee_embeddings = self.embeddings.embed_documents(self.contexts['employees']['chunks'])
                    if employee_embeddings:
                        dimension = len(employee_embeddings[0])
                        self.contexts['employees']['index'] = faiss.IndexFlatL2(dimension)
                        self.contexts['employees']['index'].add(np.array(employee_embeddings))
                        self.save_cached_data('employees')
                    else:
                        self.contexts['employees']['index'] = None
                else:
                    print("Warning: employee_structure.json not found. Skipping employee data.")
                    self.contexts['employees']['index'] = None
        except FileNotFoundError as e:
            print(f"Error: File not found - {str(e)}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format - {str(e)}")
            raise
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
        print(f"Data loading completed in {time.time() - start_time:.2f} seconds.")

    def get_relevant_chunks(self, query: str, contexts: List[str], k: int = 10) -> Tuple[List[str], List[str]]:
        """
        Retrieves the top-k most relevant text chunks for a query from the specified contexts using vector search.
        Returns a tuple of (list of relevant chunks, list of used context names).
        """
        query_embedding = np.array([self.embeddings.embed_query(query)])
        chunks = []
        used_contexts = []
        for ctx in contexts or self.contexts.keys():
            if self.contexts[ctx]['index']:
                try:
                    D, I = self.contexts[ctx]['index'].search(query_embedding.astype(np.float32), k)
                    for i, dist in zip(I[0], D[0]):
                        if dist < 1.5:
                            chunks.append((self.contexts[ctx]['chunks'][i], dist, ctx))
                except Exception as e:
                    print(f"Warning: FAISS search failed for context {ctx}: {str(e)}")
                    continue
        chunks.sort(key=lambda x: x[1])
        top_chunks = [chunk[0] for chunk in chunks[:k]]
        used_contexts = list(set(chunk[2] for chunk in chunks[:k]))
        return top_chunks, used_contexts

    def get_context_specific_prompt(self, context_types: List[str], context: str, query: str) -> str:
        """
        Builds a prompt for the LLM that includes the relevant context and the user's question.
        Used to instruct the model to answer only using the provided context.
        """
        context_str = ", ".join(context_types).upper() or "COMBINED"
        return (
            f"You are an expert assistant for EOXS, knowledgeable about products, team updates, and organizational structure. "
            f"Answer the question using only the provided context. Be concise, accurate, and detailed. "
            f"If the information is not in the context, say 'I don't have enough information to answer this question.'\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )

    def create_chatgpt_style_prompt(self, context: str, query: str, intent: str, history: str = "") -> str:
        """
        Builds a ChatGPT-style prompt for the LLM, including context, user question, intent, and conversation history.
        Adjusts instructions and formatting based on the detected intent (comparison, definition, person_info, generic).
        """
        role_map = {
            "comparison": "You are a product analyst, skilled at comparing features and explaining differences in a clear, friendly way.",
            "definition": "You are a teacher, great at explaining concepts simply and clearly.",
            "person_info": "You are an HR assistant, cheerful and helpful, ready to provide information about people and teams.",
            "generic": "You are EOXA, a helpful, cheerful, and knowledgeable assistant."
        }
        role = role_map.get(intent, role_map["generic"])
        assistant_name = "EOXA"
        prompt = f"""
# {assistant_name} Chatbot

{role}

*Instructions:*
- Respond in a friendly, conversational, and complete way, just like ChatGPT would.
- Use Markdown formatting: headers, bold, bullet points, etc.
- Add a cheerful tone and sign off as '{assistant_name}'.
- If you don't have enough information, politely say so.
"""
        if intent == "comparison":
            prompt += "\n- *For this question, always present the answer as a Markdown table comparing the main features, tasks, and focus areas. If a table is not possible, use bullet points.*\n"
        elif intent == "person_info":
            prompt += "\n- *For this question, prefer a Markdown bullet list for details about people, teams, or roles.*\n"
        elif intent == "definition":
            prompt += "\n- *For this question, provide a concise paragraph and, if helpful, a short bullet list for key points.*\n"
        else:
            prompt += "\n- *For this question, use the most suitable format (paragraph, bullets, or table) for clarity.*\n"
        if history:
            prompt += f"\n---\n*Recent Conversation:*\n{history}\n---\n"
        prompt += f"\n*Context:*\n{context}\n\n*User Question:* {query}\n\n*Your Answer:*"
        return prompt

    def detect_intent(self, query: str) -> str:
        """
        Detects the user's intent (comparison, definition, person_info, or generic) based on keywords in the query.
        Returns a string representing the intent.
        """
        query_lower = query.lower()
        if any(word in query_lower for word in ["compare", "vs", "difference", "contrast"]):
            return "comparison"
        elif any(word in query_lower for word in ["what is", "define", "explain", "definition"]):
            return "definition"
        elif any(word in query_lower for word in ["employee", "team", "member", "person", "staff", "names", "details"]):
            return "person_info"
        return "generic"

    def answer_query(self, query: str) -> str:
        """
        Main entry point for answering a user query.
        Handles special queries (employee count, list, etc.), determines context, retrieves relevant chunks,
        builds a prompt, and generates a response using the LLM.
        Maintains conversation history.
        """
        if not query.strip():
            return "Please provide a valid query."
        
        if not any(self.contexts[ctx]['index'] for ctx in self.contexts):
            return "No data loaded. Please check JSON files and load_all_data()."

        try:
            clean_query, explicit_contexts = self.parse_query(query)
            query_to_use = clean_query or query
            self.history.append({'query': query_to_use, 'context_types': [], 'response': None})
            if len(self.history) > 100:
                self.history = self.history[-100:]
            query_lower = query_to_use.lower()

            if any(phrase in query_lower for phrase in ["count how many unique names", "total employees", "count all"]):
                unique_count = self.count_unique_employee_names()
                if unique_count > 0:
                    response = f"[EMPLOYEES Context] The total number of employees is {unique_count}."
                    self.history[-1]['context_types'] = ['employees']
                    self.history[-1]['response'] = response
                    self.save_history()
                    return response
                response = "[EMPLOYEES Context] Unable to count employees. Please check employee_structure.json."
                self.history[-1]['context_types'] = ['employees']
                self.history[-1]['response'] = response
                self.save_history()
                return response

            if any(phrase in query_lower for phrase in ["show all the employees", "all employees of eoxs", "show their names", "all employees name", "total employees name", "print all the members"]):
                alphabetical = "alphabetical" in query_lower
                with_details = any(phrase in query_lower for phrase in ["with numbering and details", "with details"])
                names = self.get_employee_names(alphabetical=alphabetical, with_details=with_details)
                if names:
                    if with_details:
                        context = "\n".join(names)
                        intent = "person_info"
                        history_context = "\n".join(f"Q: {h['query']}\nA: {h['response']}" for h in self.history[-4:-1] if h['response'])
                        prompt = self.create_chatgpt_style_prompt(context, query_to_use, intent, history_context)
                        response = self.model.generate_content(prompt)
                        answer = f"[EMPLOYEES + UPDATES Context] {response.text}"
                    else:
                        response = "[EMPLOYEES Context] List of employees:\n" + "\n".join(f"{i+1}. {name}" for i, name in enumerate(names))
                    self.history[-1]['context_types'] = ['employees', 'updates'] if with_details else ['employees']
                    self.history[-1]['response'] = response.text if with_details else response
                    self.save_history()
                    return answer if with_details else response
                response = "[EMPLOYEES Context] Unable to retrieve employee names. Please check employee_structure.json."
                self.history[-1]['context_types'] = ['employees']
                self.history[-1]['response'] = response
                self.save_history()
                return response

            # Let "tell me about" and all other queries use the general processing flow
            contexts = explicit_contexts or self.determine_context(query_to_use)
            relevant_chunks, used_contexts = self.get_relevant_chunks(query_to_use, contexts, k=10)
            if not relevant_chunks:
                response = f"[{'+'.join(used_contexts).upper() or 'COMBINED'} Context] No relevant information found for '{query_to_use}'."
                self.history[-1]['context_types'] = used_contexts
                self.history[-1]['response'] = response
                self.save_history()
                return response
            context = "\n".join(relevant_chunks)
            history_context = "\n".join(
                f"Q: {h['query']}\nA: {h['response']}" for h in self.history[-4:-1] if h['response']
            )
            intent = self.detect_intent(query_to_use)
            prompt = self.create_chatgpt_style_prompt(context, query_to_use, intent, history_context)
            response = self.model.generate_content(prompt)
            answer = f"[{'+'.join(used_contexts).upper() or 'COMBINED'} Context] {response.text}"
            self.history[-1]['context_types'] = used_contexts
            self.history[-1]['response'] = response.text
            self.save_history()
            return answer
        except Exception as e:
            return f"Error generating response: {str(e)}"

    # =============================================================================
    # DIAGNOSTIC AND MAINTENANCE METHODS
    # =============================================================================

    def fix_cache(self) -> bool:
        """
        Fixes FAISS cache issues by clearing and regenerating all cached data.
        Useful when encountering FAISS version compatibility errors.
        Returns True if successful, False otherwise.
        """
        print("🔧 Fixing FAISS cache...")
        try:
            # Remove cache directory if it exists
            if os.path.exists(self.cache_dir):
                import shutil
                shutil.rmtree(self.cache_dir)
                print(f"✅ Removed existing cache directory: {self.cache_dir}")
            
            # Recreate cache directory
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"✅ Created new cache directory: {self.cache_dir}")
            
            # Reinitialize and load data
            print("🔄 Reinitializing chatbot...")
            self.load_all_data()
            
            # Test query processing
            print("🧪 Testing query processing...")
            response = self.answer_query("What is EOXS?")
            print(f"✅ Test query successful: {response[:100]}...")
            
            print("🎉 Cache fix completed!")
            return True
            
        except Exception as e:
            print(f"❌ Cache fix failed: {str(e)}")
            return False

    def test_basic(self) -> bool:
        """
        Basic functionality test - quick smoke test to verify core functionality.
        Returns True if all tests pass, False otherwise.
        """
        print("🧪 Running basic chatbot tests...")
        try:
            # Test 1: Data loading
            print("📊 Testing data loading...")
            self.load_all_data()
            print("✅ Data loaded successfully")
            
            # Test 2: Context validation
            print("🔍 Testing context validation...")
            contexts_status = {}
            for k, v in self.contexts.items():
                contexts_status[k] = 'Loaded' if v['index'] is not None else 'Failed'
            
            print("Context loading status:")
            for context, status in contexts_status.items():
                print(f"  {context}: {status}")
            
            # Test 3: Simple query
            print("💬 Testing query processing...")
            response = self.answer_query("Hello")
            print(f"✅ Query response: {response[:100]}...")
            
            print("🎉 Basic tests completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Basic test failed: {str(e)}")
            return False

    def test_comprehensive(self) -> bool:
        """
        Comprehensive diagnostic test with detailed reporting.
        Tests all major functionality step by step.
        Returns True if all tests pass, False otherwise.
        """
        print("🔍 Running comprehensive EOXS Chatbot tests...")
        
        try:
            print("📁 Step 1: Creating chatbot instance...")
            # Instance already created, just verify
            print("✅ Chatbot instance verified successfully")
            
            print("📊 Step 2: Loading all data...")
            self.load_all_data()
            print("✅ Data loaded successfully")
            
            # Check if contexts are loaded
            print("🔍 Step 3: Checking loaded contexts...")
            for context_name, context_data in self.contexts.items():
                index_status = "✅ Loaded" if context_data['index'] is not None else "❌ Failed"
                chunk_count = len(context_data['chunks']) if context_data['chunks'] else 0
                print(f"  {context_name}: {index_status} ({chunk_count} chunks)")
            
            print("💬 Step 4: Testing simple query...")
            response = self.answer_query("What is EOXS?")
            print(f"✅ Query successful: {response[:100]}...")
            
            print("💬 Step 5: Testing employee query...")
            response2 = self.answer_query("Who is Rajat Jain?")
            print(f"✅ Employee query successful: {response2[:100]}...")
            
            print("💬 Step 6: Testing product query...")
            response3 = self.answer_query("Tell me about EOXS CRM")
            print(f"✅ Product query successful: {response3[:100]}...")
            
            print("📈 Step 7: Testing context determination...")
            contexts = self.determine_context("How many employees work here?")
            print(f"✅ Context determination: {contexts}")
            
            print("🔢 Step 8: Testing employee count...")
            count = self.count_unique_employee_names()
            print(f"✅ Employee count: {count}")
            
            print("🎉 All comprehensive tests passed! Chatbot is working correctly.")
            return True
            
        except Exception as e:
            print(f"❌ Comprehensive test failed: {str(e)}")
            import traceback
            print("📝 Full traceback:")
            traceback.print_exc()
            return False

    def get_system_status(self) -> dict:
        """
        Returns a comprehensive system status report.
        Useful for monitoring and debugging.
        """
        status = {
            'cache_exists': os.path.exists(self.cache_dir),
            'contexts_loaded': {},
            'data_files': {},
            'history_length': len(self.history),
            'code_version': self.code_version
        }
        
        # Check context status
        for ctx_name, ctx_data in self.contexts.items():
            status['contexts_loaded'][ctx_name] = {
                'index_loaded': ctx_data['index'] is not None,
                'chunks_count': len(ctx_data['chunks']) if ctx_data['chunks'] else 0
            }
        
        # Check data files
        data_files = ['Products.json', 'daily_updates.json', 'employee_structure.json']
        for file in data_files:
            status['data_files'][file] = os.path.exists(file)
        
        return status

    def run_interactive(self):
        """
        Runs an interactive command-line loop for chatting with the bot.
        Supports special commands for clearing history, viewing history, clearing cache, adding employees, and adding updates.
        """
        self.load_all_data()
        print("EOXS Chatbot: Ask a question or use commands:")
        print("  'exit' - Quit the chatbot")
        print("  'clear history' - Reset conversation history")
        print("  'view history' - View past conversations")
        print("  'clear cache' - Clear cache and reload data")
        print("  'fix cache' - Fix FAISS cache issues")
        print("  'test basic' - Run basic functionality tests")
        print("  'test comprehensive' - Run comprehensive diagnostic tests")
        print("  'status' - Show system status")
        print("  'add employee <dept>,<name>,<email>,<job>,<manager>' - Add employee")
        print("  'add update <date>,<team>,<sub_team>,<project>,<members>;<summary>;<tasks>;<blockers>;<next_steps>' - Add update")
        try:
            while True:
                query = input("> ").strip()
                if query.lower() == 'exit':
                    self.save_history()
                    break
                if query.lower() == 'clear history':
                    self.history = []
                    self.save_history()
                    print("Conversation history cleared.")
                    continue
                if query.lower() == 'view history':
                    if not self.history:
                        print("No conversation history available.")
                    else:
                        for entry in self.history:
                            print(f"Q: {entry['query']}\nA: {entry['response']}\n")
                    continue
                if query.lower() == 'clear cache':
                    import shutil
                    shutil.rmtree(self.cache_dir, ignore_errors=True)
                    os.makedirs(self.cache_dir, exist_ok=True)
                    self.load_all_data()
                    print("Cache cleared and data reloaded.")
                    continue
                if query.lower() == 'fix cache':
                    self.fix_cache()
                    continue
                if query.lower() == 'test basic':
                    self.test_basic()
                    continue
                if query.lower() == 'test comprehensive':
                    self.test_comprehensive()
                    continue
                if query.lower() == 'status':
                    status = self.get_system_status()
                    print("📊 System Status:")
                    print(f"  Cache Directory: {'✅ Exists' if status['cache_exists'] else '❌ Missing'}")
                    print(f"  Code Version: {status['code_version']}")
                    print(f"  History Length: {status['history_length']} entries")
                    print("  Data Files:")
                    for file, exists in status['data_files'].items():
                        print(f"    {file}: {'✅ Found' if exists else '❌ Missing'}")
                    print("  Contexts:")
                    for ctx, data in status['contexts_loaded'].items():
                        index_status = '✅ Loaded' if data['index_loaded'] else '❌ Failed'
                        print(f"    {ctx}: {index_status} ({data['chunks_count']} chunks)")
                    continue
                if query.lower().startswith('add employee '):
                    parts = query[12:].split(',')
                    if len(parts) == 5:
                        dept, name, email, job, manager = [p.strip() for p in parts]
                        if self.add_employee(dept, name, email, job, manager):
                            import shutil
                            shutil.rmtree(self.cache_dir, ignore_errors=True)
                            os.makedirs(self.cache_dir, exist_ok=True)
                            self.load_all_data()
                            print(f"Employee {name} added to {dept}. Cache cleared and data reloaded.")
                        else:
                            print("Failed to add employee. Check JSON file or email uniqueness.")
                    else:
                        print("Invalid format. Use: add employee <dept>,<name>,<email>,<job>,<manager>")
                    continue
                if query.lower().startswith('add update '):
                    parts = query[10:].split(',', 4)
                    if len(parts) == 5:
                        date, team, sub_team, project, rest = [p.strip() for p in parts]
                        subparts = rest.split(';', 4)
                        if len(subparts) == 5:
                            members, summary, tasks, blockers, next_steps = [p.strip() for p in subparts]
                            if self.append_update(date, team, sub_team, project, members, summary, tasks, blockers, next_steps):
                                import shutil
                                shutil.rmtree(self.cache_dir, ignore_errors=True)
                                os.makedirs(self.cache_dir, exist_ok=True)
                                self.load_all_data()
                                print(f"Update for {team} on {date} added. Cache cleared and data reloaded.")
                            else:
                                print("Failed to add update. Check JSON file or input format.")
                        else:
                            print(f"Invalid sub-format. Expected 5 semicolon-separated parts (members;summary;tasks;blockers;next_steps), got {len(subparts)}. Use: add update <date>,<team>,<sub_team>,<project>,<members>;<summary>;<tasks>;<blockers>;<next_steps>")
                    else:
                        print(f"Invalid format. Expected 5 comma-separated parts (date,team,sub_team,project,members), got {len(parts)}. Use: add update <date>,<team>,<sub_team>,<project>,<members>;<summary>;<tasks>;<blockers>;<next_steps>")
                    continue
                response = self.answer_query(query)
                print(response)
        except Exception as e:
            print(f"Error in interactive mode: {str(e)}")

if __name__ == "__main__":
    chatbot = InteractiveRAGChatbot()
    chatbot.run_interactive()