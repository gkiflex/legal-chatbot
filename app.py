# app.py - Main application file
import gradio as gr
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import warnings
warnings.filterwarnings("ignore")

class LegalKnowledgeBase:
    def __init__(self):
        self.legal_terms = {}
        self.legal_documents = []
        
    def create_basic_legal_terms(self):
        """Create a basic legal terms dictionary"""
        basic_terms = {
            "contract": "A legally binding agreement between two or more parties that creates mutual obligations enforceable by law.",
            "liability": "Legal responsibility for one's acts or omissions, which may result in damages or compensation.",
            "jurisdiction": "The official power to make legal decisions and judgments within a specific geographic area or over certain types of legal cases.",
            "damages": "Money awarded by a court as compensation for a loss or injury caused by another party's wrongful act.",
            "breach of contract": "Failure to perform any duty or obligation specified in a contract without a legal excuse.",
            "negligence": "Failure to exercise the care that a reasonably prudent person would exercise in like circumstances.",
            "plaintiff": "The person who brings a case against another in a court of law, seeking legal remedy.",
            "defendant": "The person against whom a legal action is brought in a court of law.",
            "statute of limitations": "A law that sets the maximum time after an event within which legal proceedings may be initiated.",
            "due process": "Fair treatment through the normal judicial system, especially as a citizen's entitlement.",
            "indemnify": "To compensate someone for harm or loss, or to provide security against legal responsibility for their actions.",
            "force majeure": "Unforeseeable circumstances that prevent a party from fulfilling a contract, like natural disasters.",
            "arbitration": "A method of dispute resolution where parties agree to have their case decided by an impartial third party.",
            "consideration": "Something of value exchanged between parties to make a contract legally binding.",
            "easement": "A legal right to use another person's land for a specific limited purpose."
        }
        return basic_terms
        
    def load_sample_clauses(self):
        """Load sample legal clauses for analysis"""
        sample_clauses = [
            {
                "type": "Rental Agreement",
                "clause": "The Tenant shall pay rent in the amount of $1,200.00 per month, due on the first day of each month.",
                "explanation": "This means you must pay $1,200 every month by the 1st day of the month as your rent payment."
            },
            {
                "type": "Terms of Service",
                "clause": "User agrees to indemnify and hold harmless the Company from any claims arising from User's use of the service.",
                "explanation": "This means if someone sues the company because of something you did while using their service, you agree to pay for the company's legal costs and any damages."
            },
            {
                "type": "Employment Contract",
                "clause": "Employee agrees to a non-compete clause for a period of 12 months following termination.",
                "explanation": "After you stop working for this company, you cannot work for a competing company for 12 months."
            },
            {
                "type": "Purchase Agreement",
                "clause": "This agreement shall be governed by the laws of the State of California.",
                "explanation": "If there are any legal disputes about this agreement, California state laws will be used to resolve them."
            }
        ]
        return sample_clauses

class LegalRAGSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
        
    def build_index(self, legal_terms: dict, legal_clauses: list):
        """Build FAISS index from legal knowledge base"""
        documents = []
        
        # Add legal terms
        for term, definition in legal_terms.items():
            documents.append({
                "type": "term",
                "title": term,
                "content": definition,
                "text": f"{term}: {definition}"
            })
            
        # Add legal clauses
        for clause_data in legal_clauses:
            documents.append({
                "type": "clause",
                "title": clause_data["type"],
                "content": clause_data["explanation"],
                "text": f"{clause_data['type']}: {clause_data['clause']} | {clause_data['explanation']}"
            })
            
        self.documents = documents
        print(f"Building index for {len(documents)} documents...")
        
        # Generate embeddings
        texts = [doc["text"] for doc in documents]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        self.embeddings = embeddings
        print("Index built successfully!")
        
    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> list:
        """Retrieve most relevant documents for a query"""
        if self.index is None:
            return []
            
        query_embedding = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        relevant_docs = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['relevance_score'] = float(distances[0][i])
                relevant_docs.append(doc)
                
        return relevant_docs

class LegalLLMHandler:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.load_model()
        
    def load_model(self):
        """Load the language model"""
        try:
            print(f"Loading language model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Create pipeline for text generation
            self.pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                device=-1  # Use CPU
            )
            print("Language model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using the LLM"""
        if self.pipeline is None:
            return "Model not available. Please check your setup."
            
        # Construct prompt with context
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nExplain this legal concept in simple, everyday language that anyone can understand:"
        else:
            full_prompt = f"Question: {prompt}\n\nExplain this legal concept in simple, everyday language:"
        
        try:
            result = self.pipeline(full_prompt, max_length=400, num_return_sequences=1)
            response = result[0]['generated_text']
            
            # Clean up response
            if response.startswith(full_prompt):
                response = response[len(full_prompt):].strip()
            
            # Add disclaimer
            disclaimer = "\n\n⚠️ **Disclaimer**: This information is for educational purposes only and does not constitute legal advice. Please consult with a qualified attorney for legal matters."
            
            return response + disclaimer
            
        except Exception as e:
            return f"I apologize, but I encountered an error while generating a response: {str(e)}. Please try rephrasing your question."

class LegalEaseBot:
    def __init__(self):
        print("Initializing LegalEaseBot...")
        self.knowledge_base = LegalKnowledgeBase()
        self.rag_system = LegalRAGSystem()
        self.llm_handler = LegalLLMHandler()
        self.initialize_system()
        
    def initialize_system(self):
        """Initialize the bot with legal knowledge"""
        # Load legal terms and clauses
        legal_terms = self.knowledge_base.create_basic_legal_terms()
        legal_clauses = self.knowledge_base.load_sample_clauses()
        
        # Build RAG index
        self.rag_system.build_index(legal_terms, legal_clauses)
        
        print("LegalEaseBot initialized successfully!")
        
    def process_query(self, user_input: str, chat_history: list) -> tuple:
        """Process user query and return response"""
        if not user_input.strip():
            return chat_history, ""
            
        try:
            # Retrieve relevant documents
            relevant_docs = self.rag_system.retrieve_relevant_docs(user_input, top_k=3)
            
            # Build context from retrieved documents
            context = ""
            if relevant_docs:
                context_parts = []
                for doc in relevant_docs[:2]:  # Use top 2 most relevant
                    context_parts.append(doc['content'])
                context = " ".join(context_parts)
                
            # Generate response using LLM
            response = self.llm_handler.generate_response(user_input, context)
            
            # Add sources if available
            if relevant_docs:
                response += "\n\n📚 **Related Legal Topics**: "
                topics = [doc['title'].title() for doc in relevant_docs[:3]]
                response += ", ".join(topics)
            
            # Update chat history
            chat_history.append([user_input, response])
            
        except Exception as e:
            error_response = f"I apologize, but I encountered an error: {str(e)}. Please try again."
            chat_history.append([user_input, error_response])
        
        return chat_history, ""
        
    def analyze_clause(self, clause_text: str) -> str:
        """Analyze a specific legal clause"""
        if not clause_text.strip():
            return "Please provide a legal clause to analyze."
            
        try:
            prompt = f"Explain this legal clause in simple terms: {clause_text}"
            relevant_docs = self.rag_system.retrieve_relevant_docs(clause_text, top_k=2)
            
            context = ""
            if relevant_docs:
                context = relevant_docs[0]['content']
                
            response = self.llm_handler.generate_response(prompt, context)
            return response
            
        except Exception as e:
            return f"I encountered an error while analyzing the clause: {str(e)}. Please try again."

def create_interface():
    """Create Gradio interface for LegalEaseBot"""
    
    print("Creating Gradio interface...")
    
    # Initialize bot (this may take a few moments)
    bot = LegalEaseBot()
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="LegalEaseBot - Legal Terms Made Simple",
        css="""
        .gradio-container {
            max-width: 1200px;
            margin: auto;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 📚 LegalEaseBot
        *Your conversational guide to understanding legal terms and clauses in plain English*
        """)
        
        with gr.Tab("💬 Chat with Bot"):
            gr.Markdown("Ask me about legal terms, concepts, or paste a legal clause for explanation!")
            
            chatbot = gr.Chatbot(
                value=[],
                height=500,
                label="Legal Assistant",
                show_label=False,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask me about legal terms, clauses, or paste a document section...",
                    label="Your Question",
                    lines=2,
                    scale=4,
                    show_label=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                
            gr.Examples(
                examples=[
                    "What is a contract?",
                    "Explain liability in simple terms",
                    "What does 'breach of contract' mean?",
                    "What is arbitration?",
                    "Help me understand this clause: 'Party agrees to indemnify and hold harmless the Company'",
                    "What does 'force majeure' mean?"
                ],
                inputs=msg,
                label="Try these example questions:"
            )
            
        with gr.Tab("📋 Clause Analyzer"):
            gr.Markdown("### Paste a legal clause below for a plain-language explanation")
            
            with gr.Row():
                with gr.Column():
                    clause_input = gr.Textbox(
                        placeholder="Example: 'The party of the first part agrees to indemnify and hold harmless the party of the second part from any and all claims...'",
                        label="Legal Clause Text",
                        lines=6
                    )
                    analyze_btn = gr.Button("Analyze Clause", variant="primary", size="lg")
                    
                with gr.Column():
                    clause_output = gr.Textbox(
                        label="Plain Language Explanation", 
                        lines=10,
                        show_label=True
                    )
            
            gr.Examples(
                examples=[
                    "The Tenant shall pay rent in the amount of $1,200.00 per month, due on the first day of each month.",
                    "User agrees to indemnify and hold harmless the Company from any claims arising from User's use of the service.",
                    "This agreement shall be governed by the laws of the State of California.",
                    "Employee agrees to a non-compete clause for a period of 12 months following termination."
                ],
                inputs=clause_input,
                label="Try these example clauses:"
            )
            
        with gr.Tab("ℹ️ About & Help"):
            gr.Markdown("""
            ## About LegalEaseBot
            
            LegalEaseBot is designed to help you understand legal terms and document clauses in plain, everyday language. 
            This tool uses artificial intelligence to break down complex legal jargon into simple explanations.
            
            ### 🎯 Key Features:
            - **💬 Conversational Interface**: Ask questions about legal terms naturally
            - **📋 Clause Analysis**: Get plain-language explanations of legal text
            - **🎓 Educational Focus**: Learn legal concepts in simple terms
            - **🔒 Privacy-First**: Runs entirely on your device - no data sent to external servers
            - **📚 Knowledge Base**: Built-in database of common legal terms and concepts
            
            ### 📖 How to Use:
            
            **Chat Tab:**
            - Ask questions like "What is a contract?" or "Explain liability"
            - Paste legal clauses and ask for explanations
            - Have multi-turn conversations to dive deeper into topics
            
            **Clause Analyzer:**
            - Copy and paste legal text from contracts, agreements, or documents
            - Get instant plain-language explanations
            - Understand what legal clauses actually mean for you
            
            ### 💡 Tips for Best Results:
            - Be specific in your questions
            - Feel free to ask follow-up questions
            - Try different phrasings if you don't get the answer you need
            - Use the clause analyzer for complex legal text
            
            ### 🔧 Technical Details:
            - **Language Model**: Google FLAN-T5 (running locally)
            - **Knowledge Retrieval**: Sentence transformers + FAISS
            - **Interface**: Gradio web framework
            - **Privacy**: All processing happens on your device
            
            ### ⚠️ Important Legal Disclaimer:
            
            **This tool is for educational and informational purposes only.**
            
            - The information provided does NOT constitute legal advice
            - Do not rely on this tool for legal decisions
            - Always consult with a qualified attorney for legal matters
            - Laws vary by jurisdiction and change over time
            - This tool may not have the most current legal information
            
            ### 🤝 Developed by:
            - **Aemunathan** - Backend & LLM Development
            - **Pallavi** - Documentation & Project Management  
            - **Karthikeyan** - Frontend & Deployment
            
            *University of the Cumberlands - Human-Computer Interaction Course Project*
            """)
            
        # Event handlers
        def respond(message, chat_history):
            if not message.strip():
                return chat_history, message
            return bot.process_query(message, chat_history)
            
        def analyze_clause_handler(clause):
            if not clause.strip():
                return "Please enter a legal clause to analyze."
            return bot.analyze_clause(clause)
        
        def clear_chat():
            return []
            
        # Connect events
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        send_btn.click(respond, [msg, chatbot], [chatbot, msg])
        analyze_btn.click(analyze_clause_handler, clause_input, clause_output)
        
        # Add clear button for chat
        with gr.Tab("💬 Chat with Bot"):
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                clear_btn.click(clear_chat, outputs=chatbot)
        
    return demo

if __name__ == "__main__":
    print("Starting LegalEaseBot...")
    print("This may take a few moments to load the models...")
    
    demo = create_interface()
    
    print("\n" + "="*50)
    print("🎉 LegalEaseBot is ready!")
    print("📱 Opening web interface...")
    print("🌐 Access your bot at: http://127.0.0.1:7860")
    print("="*50 + "\n")
    
    # Launch the app
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # Set to True if you want to create a public link
        show_error=True,
        debug=False,
        quiet=False
    )