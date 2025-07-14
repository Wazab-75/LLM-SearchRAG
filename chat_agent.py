#!/usr/bin/env python3

import os
import json
import requests
import re
from pathlib import Path
from colorama import init, Fore, Style
from typing import Optional, List, Mapping, Any, Dict, Tuple
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent results
DetectorFactory.seed = 0

# Initialize
init()

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain.llms.base import LLM
from langchain.schema import Document

# Local imports
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

@dataclass
class SearchResult:
    """Represents a search result with metadata"""
    document: Document
    score: float
    relevance_score: float
    search_type: str  # 'semantic', 'keyword', 'hybrid'

@dataclass
class DocumentContext:
    """Stores context for a specific document"""
    document_id: str
    filename: str
    relevant_chunks: List[Document]
    last_accessed: datetime
    query_history: List[str]

class Colors:
    CYAN = Fore.CYAN
    GREEN = Fore.GREEN
    YELLOW = Fore.YELLOW
    RED = Fore.RED
    GRAY = Fore.LIGHTBLACK_EX
    WHITE = Fore.WHITE
    BLUE = Fore.BLUE
    MAGENTA = Fore.MAGENTA
    BOLD = Style.BRIGHT
    RESET = Style.RESET_ALL

class OllamaLLM(LLM):
    """Custom Ollama LLM wrapper for LangChain"""
    
    model_name: str = "mistral"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    
    @property
    def _llm_type(self) -> str:
        return "ollama"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call Ollama API with optimized settings"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Lower for more focused responses
                        "num_predict": 200,  # Shorter responses
                        "top_k": 5,  # More focused
                        "top_p": 0.8,
                        "repeat_penalty": 1.2,  # Reduce repetition
                        "num_ctx": 3072  # Adequate context
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return f"Error: HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama. Make sure Ollama is running with: ollama serve"
        except Exception as e:
            return f"❌ Error calling Ollama: {str(e)}"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters"""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.temperature
        }

class ImprovedLocalChatAgent:
    def __init__(self, model_name="mistral", pdf_dir="data/database/manual/fr", vector_store_path="data/Vectorized"):
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.model_name = model_name
        self.pdf_dir = pdf_dir
        self.vector_store_path = vector_store_path
        
        # Global conversation memory
        self.conversation_history: List[Dict[str, str]] = []
        
        # Search configuration
        self.search_config = {
            'semantic_weight': 0.7,
            'keyword_weight': 0.3,
            'rerank_top_k': 10,
            'final_top_k': 4
        }
        
        
        # Ensure directories exist
        Path(self.pdf_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
        
        # Cache for search optimization
        self.search_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def print_colored(self, text: str, color: str = Colors.RESET, bold: bool = False):
        style = Colors.BOLD if bold else ""
        print(f"{style}{color}{text}{Colors.RESET}")
    
    def update_conversation_memory(self, question: str, answer: str, sources: List[Document]):
        """Update conversation memory"""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'sources': [doc.metadata.get('filename', 'Unknown') for doc in sources[:2]]
        })
        
        # Keep only last 10 exchanges
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_memory(self) -> str:
        """Get conversation memory for context"""
        if not self.conversation_history:
            return ""
        
        # Get last 2 exchanges
        recent_exchanges = self.conversation_history[-2:]
        memory_parts = []
        
        for exchange in recent_exchanges:
            memory_parts.append(f"Q: {exchange['question']}")
            memory_parts.append(f"A: {exchange['answer'][:100]}...")
        
        return '\n'.join(memory_parts)
    
    def detect_language(self, text: str) -> str:
        """Detect language of input text using langdetect library"""
        # Clean text for better detection
        clean_text = re.sub(r'[^\w\s]', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        # Need at least some text for detection
        if len(clean_text) < 10:
            return "english"  # Default fallback
            
        try:
            detected_lang = detect(clean_text)
            
            # Map langdetect codes to our language names
            language_mapping = {
                'fr': 'french',
                'en': 'english', 
                'es': 'spanish',
                'de': 'german',
                'it': 'italian'
            }
            
            return language_mapping.get(detected_lang, 'english')
            
        except LangDetectException:
            return 'english'  # Default fallback
    
    def initialize_llm(self):
        """Initialize Ollama LLM"""
        try:
            self.print_colored(f"🦙 Connecting to Ollama ({self.model_name})...", Colors.CYAN)
            self.llm = OllamaLLM(model_name=self.model_name)
            
            # Test the connection
            test_response = self.llm._call("Hello")
            if "❌" in test_response:
                self.print_colored("❌ Failed to connect to Ollama", Colors.RED)
                return False
            
            self.print_colored("✅ Connected to Ollama", Colors.GREEN)
            return True
            
        except Exception:
            self.print_colored("❌ Failed to connect to Ollama", Colors.RED)
            return False
    
    def load_vector_store(self):
        """Load existing vector store"""
        vectorstore_path = os.path.join(self.vector_store_path, "index.faiss")
        
        if not os.path.exists(vectorstore_path):
            self.print_colored("❌ No vector store found", Colors.RED)
            return False
        
        try:
            self.print_colored("📚 Loading vector store...", Colors.CYAN)
            metadata_path = os.path.join(self.vector_store_path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                embedding_model = metadata.get("embedding_model", "all-MiniLM-L6-v2")
            else:
                embedding_model = "all-MiniLM-L6-v2"
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'batch_size': 32}
            )
            
            self.vectorstore = FAISS.load_local(
                self.vector_store_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            self.print_colored("✅ Vector store loaded", Colors.GREEN)
            return True
            
        except Exception as e:
            self.print_colored(f"❌ Failed to load vector store: {str(e)}", Colors.RED)
            return False
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter - keep words longer than 2 characters
        keywords = []
        for word in words:
            if len(word) > 2:
                keywords.append(word)
        
        # Extract important patterns
        patterns = [
            r'\b[A-Z]{2,}[\s\-]?\d{2,}\b',  # Model numbers
            r'\b\d{4,6}\b',  # Product codes
            r'\b\d+\s*(?:mm|cm|m|kg|g|A|V|W|Hz|°C|bar|psi|rpm)\b',  # Measurements
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            keywords.extend(matches)
        
        return list(set(keywords))
    
    def keyword_search(self, keywords: List[str], k: int = 10) -> List[Document]:
        """Perform keyword-based search"""
        all_results = []
        
        for keyword in keywords[:5]:  # Limit to top 5 keywords
            try:
                results = self.vectorstore.similarity_search(keyword, k=k//len(keywords[:5]))
                all_results.extend(results)
            except:
                continue
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for doc in all_results:
            doc_id = f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', '')}"
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
        
        return unique_results
    
    def hybrid_search(self, question: str, k: int = 10) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword search"""
        # Check cache first
        cache_key = f"{question}_{k}"
        if cache_key in self.search_cache:
            cached_time, cached_results = self.search_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                return cached_results
        
        # Extract keywords
        keywords = self.extract_keywords(question)
        
        # Semantic search
        semantic_results = self.vectorstore.similarity_search_with_score(question, k=k)
        
        # Keyword search
        keyword_docs = self.keyword_search(keywords, k=k)
        
        # Combine results
        search_results = []
        
        # Process semantic results
        for doc, distance in semantic_results:
            # Convert distance to similarity score (lower distance = higher similarity)
            similarity_score = 1 / (1 + distance)
            search_results.append(SearchResult(
                document=doc,
                score=similarity_score * self.search_config['semantic_weight'],
                relevance_score=similarity_score,
                search_type='semantic'
            ))
        
        # Process keyword results
        for doc in keyword_docs:
            # Calculate keyword relevance
            content_lower = doc.page_content.lower()
            keyword_score = sum(1 for kw in keywords if kw.lower() in content_lower) / max(len(keywords), 1)
            
            search_results.append(SearchResult(
                document=doc,
                score=keyword_score * self.search_config['keyword_weight'],
                relevance_score=keyword_score,
                search_type='keyword'
            ))
        
        # Remove duplicates and combine scores
        combined_results = {}
        for result in search_results:
            doc_id = f"{result.document.metadata.get('source', '')}_{result.document.metadata.get('page', '')}"
            if doc_id in combined_results:
                combined_results[doc_id].score += result.score
                combined_results[doc_id].relevance_score = max(
                    combined_results[doc_id].relevance_score,
                    result.relevance_score
                )
                if result.search_type == 'keyword':
                    combined_results[doc_id].search_type = 'hybrid'
            else:
                combined_results[doc_id] = result
        
        # Sort by combined score
        final_results = sorted(combined_results.values(), key=lambda x: x.score, reverse=True)
        
        # Cache results
        self.search_cache[cache_key] = (time.time(), final_results[:k])
        
        return final_results[:k]
    
    def rerank_results(self, question: str, results: List[SearchResult]) -> List[SearchResult]:
        """Simple reranking - focus on recent sources and relevance"""
        question_lower = question.lower()
        reranked = []
        
        for result in results:
            doc = result.document
            content_lower = doc.page_content.lower()
            
            # Calculate simple ranking factors
            bonus_score = 0
            
            # Prefer documents from recent conversation
            if self.conversation_history:
                recent_sources = []
                for exchange in self.conversation_history[-2:]:
                    recent_sources.extend(exchange.get('sources', []))
                
                doc_filename = doc.metadata.get('filename', 'Unknown')
                if doc_filename in recent_sources:
                    bonus_score += 0.4  # Strong preference for recent sources
            
            # Exact phrase matching
            if len(question_lower.split()) > 2:
                for i in range(len(question_lower.split()) - 2):
                    three_word_phrase = ' '.join(question_lower.split()[i:i+3])
                    if three_word_phrase in content_lower:
                        bonus_score += 0.3
            
            # Early page preference
            page_num = doc.metadata.get('page', 999)
            if page_num <= 3:
                bonus_score += 0.2
            
            # Update score
            result.score += bonus_score
            reranked.append(result)
        
        return sorted(reranked, key=lambda x: x.score, reverse=True)
    
    
    def get_contextual_search(self, question: str) -> List[SearchResult]:
        """Simple contextual search - let Mistral handle the understanding"""
        return self.hybrid_search(question, k=self.search_config['final_top_k'])
    
    def enhanced_answer_generation(self, question: str, context: str) -> str:
        """Generate answer as a conversational agent with memory"""
        # Detect the language of the question
        detected_language = self.detect_language(question)
        
        # Get conversation memory
        conversation_memory = self.get_conversation_memory()
        
        # Language-specific instructions
        language_instructions = {
            'french': "Répondez en français. Fournissez une réponse concise et utile (2-4 phrases).",
            'english': "Respond in English. Provide a concise, helpful answer (2-4 sentences).",
            'spanish': "Responda en español. Proporcione una respuesta concisa y útil (2-4 oraciones).",
            'german': "Antworten Sie auf Deutsch. Geben Sie eine prägnante, hilfreiche Antwort (2-4 Sätze).",
            'italian': "Rispondi in italiano. Fornisci una risposta concisa e utile (2-4 frasi)."
        }
        
        # Get language-specific instruction
        lang_instruction = language_instructions.get(detected_language, language_instructions['english'])
        
        # Create dynamic prompt that adapts to the user's language
        prompt = f"""You are a helpful technical assistant. You can reference previous parts of our conversation and understand context naturally.

{f"Previous conversation:\n{conversation_memory}\n" if conversation_memory else ""}Document context:
{context}

Question: {question}

{lang_instruction} If you're referring to something we discussed before, you can naturally reference it.

Answer:"""
        
        return self.llm._call(prompt)
    
    def intelligent_search(self, question: str) -> Tuple[str, List[Document]]:
        """Perform intelligent search with all enhancements"""
        start_time = time.time()
        
        try:
            # Step 1: Contextual hybrid search
            search_results = self.get_contextual_search(question)
            
            # Step 2: Rerank results
            reranked_results = self.rerank_results(question, search_results)
            
            # Step 3: Get top results
            final_docs = []
            seen_content = set()
            
            for result in reranked_results[:self.search_config['final_top_k']]:
                # Avoid duplicate content
                content_hash = hash(result.document.page_content[:200])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    final_docs.append(result.document)
            
            # Step 4: Build context for LLM (more concise)
            context_parts = []
            for doc in final_docs[:3]:  # Limit to 3 most relevant docs
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 0)
                filename = os.path.basename(source) if source != 'Unknown' else 'Document'
                
                header = f"\n[{filename}, Page {page + 1}]"
                content = doc.page_content[:600]  # Smaller chunks for focused answers
                
                context_parts.append(f"{header}\n{content}")
            
            context = "\n".join(context_parts)
            
            # Step 5: Generate answer
            answer = self.enhanced_answer_generation(question, context)
            
            # Step 6: Update conversation memory
            self.update_conversation_memory(question, answer, final_docs)
            
            # Show timing
            total_time = time.time() - start_time
            self.print_colored(f"⏱️ Response time: {total_time:.2f}s", Colors.GRAY)
            
            return answer, final_docs
            
        except Exception as e:
            self.print_colored(f"❌ Search error: {str(e)}", Colors.RED)
            return f"Error: {str(e)}", []
    
    
    def setup_retrieval_chain(self):
        """Setup the conversational retrieval chain"""
        return self.vectorstore and self.llm
    
    def show_sources(self, source_documents):
        """Display source documents with enhanced formatting"""
        if not source_documents:
            self.print_colored("No sources found", Colors.GRAY)
            return
        
        self.print_colored(f"\n📚 Sources ({len(source_documents)} documents):", Colors.CYAN)
        
        # Group by source file
        sources_by_file = defaultdict(list)
        for doc in source_documents:
            source = doc.metadata.get('source', 'Unknown')
            if source != 'Unknown':
                filename = os.path.basename(source)
            else:
                filename = doc.metadata.get('filename', 'Unknown')
            sources_by_file[filename].append(doc)
        
        for filename, docs in sources_by_file.items():
            self.print_colored(f"\n📄 {filename}:", Colors.YELLOW, bold=True)
            for doc in sorted(docs, key=lambda x: x.metadata.get('page', 0)):
                page = doc.metadata.get('page', 0)
                # Fix the concatenation bug by ensuring page is treated as int
                if isinstance(page, (int, float)):
                    page_display = f"Page {int(page) + 1}"
                else:
                    page_display = f"Page {page}"
                self.print_colored(f"  • {page_display}", Colors.GREEN)
                content_preview = doc.page_content[:150].replace('\n', ' ')
                print(f"    {content_preview}...")
    
    
    def chat_loop(self):
        """Main chat interaction loop"""
        self.print_colored("\n💬 Ready to answer questions!", Colors.GREEN)
        
        while True:
            try:
                user_input = input(f"\n{Colors.CYAN}You: {Colors.RESET}")
                
                if not user_input.strip():
                    continue
                
                # Get enhanced answer
                self.print_colored("🤔 Thinking...", Colors.YELLOW)
                answer, sources = self.intelligent_search(user_input)
                
                self.print_colored(f"Mistral: {answer}", Colors.GREEN)
                
                # Offer to show sources
                if sources:
                    show_sources = input(f"\n{Colors.GRAY}Show sources? (y/n): {Colors.RESET}").lower()
                    if show_sources in ['y', 'yes']:
                        self.show_sources(sources)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.print_colored(f"\n❌ Error: {str(e)}", Colors.RED)
    
    
    def run(self):
        """Main application entry point"""
        
        # Initialize Ollama LLM
        if not self.initialize_llm():
            return
        
        # Load vector store
        if not self.load_vector_store():
            return
        
        # Setup retrieval chain
        if not self.setup_retrieval_chain():
            return
        
        # Start chat loop
        self.chat_loop()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Local PDF Chat Agent')
    parser.add_argument('--model', default='mistral', 
                       help='Ollama model to use (default: mistral)')
    parser.add_argument('--pdf-dir', default='data/database/manual/fr',
                       help='Directory containing PDF files')
    parser.add_argument('--vector-dir', default='data/Vectorized',
                       help='Directory containing vector store')
    
    args = parser.parse_args()
    
    # Create and run enhanced chat agent
    agent = ImprovedLocalChatAgent(
        model_name=args.model,
        pdf_dir=args.pdf_dir,
        vector_store_path=args.vector_dir
    )
    agent.run()

if __name__ == "__main__":
    main()