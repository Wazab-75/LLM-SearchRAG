#!/usr/bin/env python3

import os
import json
from pathlib import Path
from colorama import init, Fore, Style
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import torch
from typing import List, Dict, Any
import gc
import re
import hashlib
from collections import defaultdict
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Set seed for consistent results
DetectorFactory.seed = 0

# Set CUDA memory allocation strategy for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Initialize
init()

# LangChain imports
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS, Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

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


class EnhancedDatabasePDFProcessor:
    def __init__(self, 
                 database_dir="data/database", 
                 vector_store_path="data/Vectorized",
                 num_workers=None, 
                 batch_size=256, 
                 use_gpu=True, 
                 vector_store_type="faiss"):
        self.database_dir = database_dir
        self.vector_store_path = vector_store_path
        self.num_workers = num_workers or mp.cpu_count()
        self.batch_size = batch_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.vector_store_type = vector_store_type
        
        # Choose best embedding model based on hardware
        if self.use_gpu:
            self.embedding_model_name = "BAAI/bge-small-en-v1.5"
            self.model_kwargs = {'device': 'cuda'}
            self.encode_kwargs = {'batch_size': min(batch_size, 32), 'normalize_embeddings': True}
        else:
            self.embedding_model_name = "all-MiniLM-L6-v2"
            self.model_kwargs = {'device': 'cpu'}
            self.encode_kwargs = {'batch_size': min(batch_size, 64), 'normalize_embeddings': True}
        
        # Ensure directories exist
        Path(self.vector_store_path).mkdir(parents=True, exist_ok=True)
    
    def print_colored(self, text: str, color: str = Colors.RESET, bold: bool = False):
        style = Colors.BOLD if bold else ""
        print(f"{style}{color}{text}{Colors.RESET}")
    
    
    def load_pdf_enhanced(self, pdf_path: Path) -> List[Document]:
        """Load PDF and extract French content"""
        try:
            loader = PDFPlumberLoader(str(pdf_path))
            pages = loader.load()
            
            if not pages:
                return []
            
            all_documents = []
            
            for page_num, page in enumerate(pages):
                # Skip very short pages
                if len(page.page_content.strip()) < 50:
                    continue
                
                # Detect language and only include French pages
                try:
                    detected_lang = detect(page.page_content)
                    if detected_lang != 'fr':
                        continue  # Skip non-French pages
                except LangDetectException:
                    # If detection fails, include it (might be technical content)
                    pass
                
                # Add basic metadata
                page.metadata.update({
                    'source': str(pdf_path),
                    'filename': pdf_path.name,
                    'page': page_num,
                    'subdirectory': str(pdf_path.parent.relative_to(self.database_dir))
                })
                
                all_documents.append(page)
            
            return all_documents
            
        except Exception as e:
            self.print_colored(f"Error loading {pdf_path.name}: {e}", Colors.RED)
            return []
    
    
    def process_pdfs_parallel(self, pdf_files: List[Path]) -> List[Document]:
        """Process PDFs in parallel"""
        all_documents = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_pdf = {executor.submit(self.load_pdf_enhanced, pdf): pdf 
                           for pdf in pdf_files}
            
            for future in tqdm(as_completed(future_to_pdf), 
                             total=len(pdf_files), 
                             desc="Processing PDFs"):
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                except Exception as e:
                    self.print_colored(f"Error in batch processing: {e}", Colors.RED)
        
        return all_documents
    
    def create_embeddings_model(self):
        """Create optimized embeddings model"""
        self.print_colored(f"🧠 Loading embedding model: {self.embedding_model_name}", Colors.CYAN)
        
        try:
            if self.use_gpu:
                torch.cuda.empty_cache()
                time.sleep(1)
                
                self.print_colored("🚀 Attempting GPU acceleration...", Colors.CYAN)
                
                embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs=self.model_kwargs,
                    encode_kwargs=self.encode_kwargs
                )
                
                self.print_colored("✅ GPU acceleration enabled!", Colors.GREEN)
                return embeddings
                
        except Exception as e:
            self.print_colored(f"⚠️  GPU failed ({str(e)[:50]}...), falling back to CPU", Colors.YELLOW)
        
        # Fallback to CPU
        self.print_colored("🖥️  Using CPU for embeddings", Colors.CYAN)
        self.use_gpu = False
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'batch_size': min(self.batch_size, 64), 'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )
        
        return embeddings
    
    def create_vector_store_with_ids(self, texts: List[Document], embeddings) -> FAISS:
        """Create vector store with document IDs for better tracking"""
        # Assign unique IDs
        for i, doc in enumerate(texts):
            doc.metadata['doc_id'] = f"doc_{i}"
        
        # Create vector store
        if self.vector_store_type == "faiss":
            vectorstore = FAISS.from_documents(texts[:1000], embeddings)
            
            # Add remaining documents in batches
            for i in tqdm(range(1000, len(texts), 1000), desc="Creating vector store"):
                batch = texts[i:i + 1000]
                batch_vectorstore = FAISS.from_documents(batch, embeddings)
                vectorstore.merge_from(batch_vectorstore)
                del batch_vectorstore
                gc.collect()
                if self.use_gpu:
                    torch.cuda.empty_cache()
            
            return vectorstore
        else:
            # Chroma implementation
            vectorstore = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=embeddings
            )
            
            for i in tqdm(range(0, len(texts), 1000), desc="Adding to vector store"):
                batch = texts[i:i + 1000]
                vectorstore.add_documents(batch)
                vectorstore.persist()
                gc.collect()
            
            return vectorstore
    
    def get_all_pdfs(self):
        """Get all PDFs recursively from database directory"""
        pdf_files = []
        subdirs = {}
        total_size = 0
        
        for pdf_path in Path(self.database_dir).glob("**/*.pdf"):
            pdf_files.append(pdf_path)
            total_size += pdf_path.stat().st_size
            
            subdir = str(pdf_path.parent.relative_to(self.database_dir))
            if subdir not in subdirs:
                subdirs[subdir] = []
            subdirs[subdir].append(pdf_path)
        
        return pdf_files, subdirs, total_size
    
    def analyze_pdfs(self):
        """Analyze PDF distribution and sizes"""
        pdf_files, subdirs, total_size = self.get_all_pdfs()
        
        self.print_colored(f"\n📊 PDF Analysis for {self.database_dir}:", Colors.CYAN, bold=True)
        self.print_colored(f"{'='*60}", Colors.CYAN)
        
        for subdir, files in sorted(subdirs.items()):
            subdir_size = sum(f.stat().st_size for f in files)
            size_mb = subdir_size / (1024 * 1024)
            self.print_colored(f"📁 {subdir}:", Colors.GREEN)
            self.print_colored(f"   Files: {len(files)} PDFs", Colors.WHITE)
            self.print_colored(f"   Size: {size_mb:.1f} MB", Colors.WHITE)
        
        self.print_colored(f"\n📊 Total:", Colors.CYAN, bold=True)
        self.print_colored(f"   Files: {len(pdf_files)} PDFs", Colors.WHITE)
        self.print_colored(f"   Size: {total_size / (1024 * 1024 * 1024):.2f} GB", Colors.WHITE)
        
        return pdf_files, subdirs, total_size
    
    def process_database(self, force_reload=False):
        """Process all PDFs with enhanced extraction and indexing"""
        start_time = time.time()
        
        # Check existing vector store
        if not force_reload:
            vectorstore_path = os.path.join(self.vector_store_path, "index.faiss")
            if os.path.exists(vectorstore_path):
                self.print_colored("✅ Vector store already exists. Use --force to rebuild", Colors.GREEN)
                return True
        
        # Analyze PDFs
        pdf_files, subdirs, total_size = self.analyze_pdfs()
        
        if not pdf_files:
            self.print_colored(f"❌ No PDF files found in {self.database_dir}/", Colors.RED)
            return False
        
        self.print_colored(f"\n⚙️  Enhanced Processing Configuration:", Colors.CYAN, bold=True)
        self.print_colored(f"  ⚡ Workers: {self.num_workers}", Colors.WHITE)
        self.print_colored(f"  📦 Batch size: {self.batch_size}", Colors.WHITE)
        self.print_colored(f"  🎯 Vector store: {self.vector_store_type.upper()}", Colors.WHITE)
        self.print_colored(f"  🖥️  Device: {'GPU (CUDA)' if self.use_gpu else 'CPU'}", Colors.WHITE)
        self.print_colored(f"  🧠 Smart chunking: Enabled", Colors.WHITE)
        self.print_colored(f"  📊 Metadata extraction: Enhanced", Colors.WHITE)
        
        try:
            # Step 1: Load all PDFs
            self.print_colored("\n🔄 Loading PDFs...", Colors.CYAN)
            documents = self.process_pdfs_parallel(pdf_files)
            self.print_colored(f"📖 Loaded {len(documents)} pages", Colors.GREEN)
            
            # Step 2: Create chunks using standard text splitter
            self.print_colored("✂️ Creating chunks...", Colors.CYAN)
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
            )
            
            texts = text_splitter.split_documents(documents)
            self.print_colored(f"📝 Created {len(texts)} chunks", Colors.GREEN)
            
            # Step 3: Create embeddings
            if self.use_gpu:
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)
            
            embeddings = self.create_embeddings_model()
            
            # Step 4: Create vector store
            self.print_colored("🔗 Creating vector store...", Colors.CYAN)
            vectorstore = self.create_vector_store_with_ids(texts, embeddings)
            
            # Step 5: Save vector store
            if self.vector_store_type == "faiss":
                vectorstore.save_local(self.vector_store_path)
            else:
                vectorstore.persist()
            
            # Save metadata
            processing_time = time.time() - start_time
            
            metadata = {
                "database_dir": self.database_dir,
                "total_pdfs": len(pdf_files),
                "total_chunks": len(texts),
                "total_size_gb": total_size / (1024 * 1024 * 1024),
                "subdirectories": {k: len(v) for k, v in subdirs.items()},
                "embedding_model": self.embedding_model_name,
                "processing_time": processing_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            with open(os.path.join(self.vector_store_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Display results
            self.print_colored("\n✅ Processing completed!", Colors.GREEN, bold=True)
            self.print_colored(f"⏱️  Total time: {processing_time / 60:.1f} minutes", Colors.GREEN)
            self.print_colored(f"⚡ Speed: {len(pdf_files) / (processing_time / 60):.1f} PDFs/minute", Colors.GREEN)
            self.print_colored(f"📊 Chunks created: {len(texts):,}", Colors.GREEN)
            self.print_colored(f"💾 Vector store saved to: {self.vector_store_path}", Colors.GREEN)
            
            return True
            
        except Exception as e:
            self.print_colored(f"❌ Error processing PDFs: {str(e)}", Colors.RED)
            import traceback
            traceback.print_exc()
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Database PDF Processor')
    parser.add_argument('--force', action='store_true', 
                       help='Force rebuild even if vector store exists')
    parser.add_argument('--database-dir', default='data/database',
                       help='Database directory containing all PDFs')
    parser.add_argument('--output-dir', default='data/Vectorized',
                       help='Directory to save vector store')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for embeddings')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--vector-store', choices=['faiss', 'chroma'], default='faiss',
                       help='Vector store type')
    
    args = parser.parse_args()
    
    processor = EnhancedDatabasePDFProcessor(
        database_dir=args.database_dir,
        vector_store_path=args.output_dir,
        num_workers=args.workers,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu,
        vector_store_type=args.vector_store
    )
    
    processor.print_colored("\n🚀 Enhanced Database PDF Processor", Colors.CYAN, bold=True)
    processor.print_colored("="*60, Colors.CYAN)
    processor.print_colored("📚 Processing with smart chunking and metadata extraction", Colors.GREEN, bold=True)
    
    if processor.process_database(force_reload=args.force):
        processor.print_colored("\n🎉 Success! Enhanced processing complete", Colors.GREEN, bold=True)
    else:
        processor.print_colored("\n❌ Processing failed", Colors.RED, bold=True)

if __name__ == "__main__":
    main()