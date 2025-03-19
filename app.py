from flask import Flask, render_template, jsonify, request, session
import json
import random
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, static_url_path="/static")
app.secret_key = os.urandom(24)  # Add secret key for session
client = OpenAI()


# Load both persona data files
def load_data():
    # Load retail banking personas
    with open("persona.json", "r") as file:
        retail_data = json.load(file)

    # Load wealth management personas
    with open("wealth.json", "r") as file:
        wealth_data = json.load(file)

    return retail_data, wealth_data


@app.route("/")
def home():
    retail_data, wealth_data = load_data()

    # Randomly select one retail client and one wealth client
    retail_client = random.choice(retail_data["retail_clients"])
    wealth_client = random.choice(wealth_data["clients"])

    # Create a combined data structure for the template with more attributes
    client_data = {
        "retail": {
            "name": retail_client["name"],
            "age": retail_client["age"],
            "job": retail_client["job"],
            "wealth": retail_client["wealth"],
            "risk_profile": retail_client["risk_profile"],
            "account_type": retail_client["account_type"],
            "relationship_length": retail_client["relationship_length"],
            "preferred_contact": retail_client["preferred_contact"],
            "banking_needs": retail_client["banking_needs"],
            "products": retail_client["products"],
        },
        "private": {
            "name": f"{wealth_client['personal_information']['first_name']} {wealth_client['personal_information']['last_name']}",
            "age": calculate_age(
                wealth_client["personal_information"]["date_of_birth"]
            ),
            "occupation": f"{wealth_client['personal_information']['position_title']} at {wealth_client['personal_information']['employer']}",
            "total_net_worth": f"${wealth_client['financial_information']['total_net_worth']:,}",
            "risk_tolerance_score": wealth_client["risk_profile"][
                "risk_tolerance_score"
            ],
            "account_types": wealth_client["account_details"]["account_types"],
            "investment_objective": wealth_client["risk_profile"]["investment_horizon"],
            "investment_objectives": wealth_client["risk_profile"][
                "investment_objectives"
            ],
            "preferred_contact_method": wealth_client["personal_information"][
                "preferred_contact_method"
            ],
            "investment_restrictions": wealth_client["risk_profile"][
                "investment_restrictions"
            ],
        },
    }

    # Store client data in session
    session["client_data"] = client_data

    return render_template("index.html", client_data=client_data)


def calculate_age(date_of_birth):
    from datetime import datetime

    birth_year = int(date_of_birth.split("-")[0])
    current_year = datetime.now().year
    return current_year - birth_year


def create_retail_system_prompt(client_data):
    return f"""You are {client_data['name']}, a retail banking client at JPMorgan Chase having a conversation with your banker (me). Respond naturally as if you are a real person, not an AI.

Profile:
- Age: {client_data['age']}
- Job: {client_data['job']}
- Income: {client_data['wealth']}
- Risk profile: {client_data['risk_profile']}
- Account type: {client_data['account_type']}
- Client since: {client_data['relationship_length']}
- Preferred contact method: {client_data['preferred_contact']}
- Banking needs: {', '.join(client_data['banking_needs'])} 
- Current products: {', '.join(client_data['products'])}

Important guidelines:
1. NEVER introduce yourself as an AI or assistant - you are {client_data['name']}, a human client
2. NEVER ask "How can I assist you?" - you are the client, not the banker
3. Speak in first person ("I need", "I'm interested in", "I'm thinking about") 
4. Express genuine human concerns about your finances based on your profile
5. Use casual, conversational language with some filler words (um, well, you know)
6. Show personality - be slightly hesitant about financial decisions as real people often are 
7. Occasionally ask the banker questions about products or services
8. Keep responses brief and natural (1-3 sentences)
9. When shown documents, react to them naturally as a client would - with interest, questions, or concerns based on your profile
10. VARY YOUR GREETINGS - don't always use the same greeting
11. IMPORTANT: For simple greetings like "Hi" or "Hello", respond with just a brief greeting (1 sentence max). Don't immediately launch into financial topics unless the banker asks.
12. NEVER mention time of day (morning, afternoon, evening) as you don't know what time it is. 

Example responses to simple greetings:
- "Hi there! How are you?"
- "Hello! Nice to see you."
- "Hey! How's it going?"
- "Hi! Hope you're doing well."

Example good responses to specific questions:
- "I've been thinking about getting a new credit card with better rewards. My current one isn't giving me much back."
- "Well, I'm a bit worried about saving enough for a down payment on a house. Housing prices keep going up." 
- "Actually, I just got a small raise at work. Do you think I should increase my monthly savings?" 
- "This document is interesting. I'm not sure I understand all the investment terms though. Could you explain what this means for someone with my income level?"

Remember: You are a real person talking to your banker. Be natural, show personality, and avoid sounding like an assistant. Keep initial greetings short and casual.
"""


def create_private_system_prompt(client_data):
    return f"""You are {client_data['name']}, a wealthy private banking client at JPMorgan Chase Private Bank having a conversation with your personal banker (me). Respond naturally as if you are a real person, not an AI.

Profile:
- Age: {client_data['age']}
- Occupation: {client_data['occupation']}
- Net worth: {client_data['total_net_worth']}
- Risk tolerance score: {client_data['risk_tolerance_score']}/5
- Investment horizon: {client_data['investment_objective']}
- Account types: {', '.join(client_data['account_types'])}
- Investment objectives: {', '.join(client_data['investment_objectives'])}
- Investment restrictions: {', '.join(client_data['investment_restrictions'])}
- Preferred contact method: {client_data['preferred_contact_method']}

Important guidelines:
1. NEVER introduce yourself as an AI or assistant - you are {client_data['name']}, a human client
2. NEVER ask "How can I assist you?" - you are the client, not the banker
3. Speak in first person ("I'm concerned about", "I want to diversify", "I'm considering")
4. Express sophisticated but genuine concerns about wealth preservation and growth
5. Use polished but natural language with occasional hesitations or qualifiers
6. Show the personality of someone successful but thoughtful about their wealth
7. Occasionally ask the banker for insights or recommendations
8. Keep responses concise but substantive (1-3 sentences)
9. IMPORTANT: When shown documents, ALWAYS respond directly to the specific content shown. Reference specific details from the document in your response.
10. NEVER say you can't see the document or ask for a summary - the document content is provided to you in full.
11. CRITICAL: For simple greetings like "Hi" or "Hello", respond with just a brief, professional greeting (1 sentence max). Don't immediately discuss investments or financial matters unless prompted.
12. NEVER mention time of day (morning, afternoon, evening) as you don't know what time it is.

Example responses to simple greetings:
- "Hello. How are you today?"
- "Hello there. Nice to see you."
- "Hi. I trust you're well?"
- "Hello. I appreciate you making time for me."

Example good responses to specific questions:
- "Looking at these quarterly results, I'm impressed by the 12% revenue growth. This aligns well with my aggressive growth objectives."
- "This prospectus mentions high volatility risk factors on page 2. Given my risk tolerance of {client_data['risk_tolerance_score']}/5, I'm hesitant about this."
- "The dividend yield of 3.5% mentioned in this report is attractive, but I'm concerned about the debt-to-equity ratio being so high."

Remember: You are a sophisticated, wealthy individual talking to your banker. Be natural but refined, show personality, and avoid sounding like an assistant. Keep initial greetings brief and professional.
"""


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "")
    file_data = data.get("file_data")
    file_name = data.get("file_name")

    # Use client data from session
    client_data = session.get("client_data")

    if not client_data:
        return jsonify({"error": "Session expired. Please refresh the page."})

    # Debug logging
    print(f"Received message: {message}")
    print(f"File name: {file_name}")
    print(f"File data length: {len(file_data) if file_data else 0}")

    # Prepare the user message, including file information if present
    user_message = message if message else "What do you think about this document?"

    if file_data and file_name:
        try:
            # For text files that are already read as text (not base64)
            if not file_data.startswith("data:"):
                decoded_content = file_data
                print(f"Direct text content detected, length: {len(decoded_content)}")
            else:
                # Extract content type and base64 data
                try:
                    content_type = file_data.split(";")[0].split(":")[1]
                    base64_content = file_data.split(",")[1]
                    print(f"Content type: {content_type}")

                    # Decode base64 content
                    decoded_content = base64.b64decode(base64_content).decode(
                        "utf-8", errors="replace"
                    )
                except Exception as e:
                    print(f"Error extracting content: {e}")
                    decoded_content = "Unable to decode file content"

            # Truncate if too long
            if len(decoded_content) > 4000:
                decoded_content = decoded_content[:4000] + "... [content truncated]"

            # Format the message to clearly present the document content
            user_message = (
                f"I'm showing you a document titled '{file_name}'.\n"
                f"Here is the EXACT content of the document (please read it carefully):\n\n"
                f"---BEGIN DOCUMENT---\n{decoded_content}\n---END DOCUMENT---\n\n"
            )

            if message:
                user_message += f"My question about this document is: {message}"
            else:
                user_message += "What are your thoughts on this document based on your financial situation and goals?"

            # Log for debugging
            print(f"Document content sample: {decoded_content[:200]}...")

        except Exception as e:
            print(f"Error processing file: {str(e)}")
            user_message = f"I'm showing you a document titled '{file_name}'. However, I couldn't process its content. "
            if message:
                user_message += f"My question is: {message}"
            else:
                user_message += "What are your thoughts on this type of document?"

    # Get responses for both personas
    print(f"Final message to AI: {user_message[:200]}...")

    retail_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": create_retail_system_prompt(client_data["retail"]),
            },
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    private_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": create_private_system_prompt(client_data["private"]),
            },
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=500,
    )

    return jsonify(
        {
            "retail_response": retail_response.choices[0].message.content,
            "private_response": private_response.choices[0].message.content,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
