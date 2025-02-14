CIC Community Chatbot
A Streamlit-powered chatbot that integrates speech-to-text, RAG (Retrieval-Augmented Generation) with Pinecone, and response generation using Meta Llama 3.2 3B Instruct.

🚀 Features
✅ Document Processing: Extracts text from PDFs and Word documents
✅ Speech-to-Text: Converts voice input to text using IBM Watson
✅ RAG-powered Search: Retrieves relevant document chunks using Pinecone
✅ AI Response Generation: Generates accurate responses with Meta Llama 3.2 3B Instruct
✅ Secure API Handling: Uses environment variables for authentication

🛠 Installation
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/cic-community-chatbot.git
cd cic-community-chatbot
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Set Up Environment Variables
Create a .env file in the project directory and add:

bash
Copy
Edit
HUG_TOKEN=your-huggingface-token
IBM_API_KEY=your-ibm-key
IBM_URL=your-ibm-url
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=your-pinecone-index
RAG_FILE=your-rag-file-path
ADMIN_PSW=your-admin-password
4️⃣ Run the App
bash
Copy
Edit
streamlit run app.py
