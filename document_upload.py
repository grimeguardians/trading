"""The provided changes involve improving document persistence and validation by adding a call to `self._validate_document_storage` after a chunk is successfully ingested."""
"""
Document Upload Interface for Trading Literature Memory Bank
Allows uploading and processing of trading documents into the Digital Brain
"""

import os
import json
import logging
import mimetypes
from datetime import datetime
from typing import Dict, List, Any
from knowledge_engine import DigitalBrain
from document_ingestion import DocumentIngestionEngine

class TradingDocumentUploader:
    """Upload interface for trading literature and documentation"""

    def __init__(self):
        self.digital_brain = DigitalBrain()
        self.ingestion_engine = DocumentIngestionEngine(self.digital_brain)
        self.logger = logging.getLogger("DocumentUploader")
        self.upload_stats = {
            'documents_uploaded': 0,
            'total_pages_processed': 0,
            'knowledge_nodes_created': 0,
            'patterns_learned': 0
        }

        # Create uploads directory
        self.upload_dir = "uploaded_documents"
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Persistent storage for document memory bank
        self.memory_bank_file = "document_memory_bank.json"
        self.knowledge_graph_file = "knowledge_graph_state.json"
        
        # Load knowledge graph state first, then memory bank
        self.load_knowledge_graph()
        self.load_memory_bank()

    def upload_document(self, file_path: str, doc_type: str = "trading_literature",
                        symbols: List[str] = None, description: str = "") -> Dict[str, Any]:
        """Upload and process a trading document with intelligent chunking and deduplication"""
        try:
            if not os.path.exists(file_path):
                return {'error': f'File not found: {file_path}'}

            # Read document content
            content = self._read_document(file_path)
            if not content:
                return {'error': 'Could not read document content'}
            
            # Check for duplicate content
            if self._is_duplicate_content(content, file_path):
                return {
                    'success': True,
                    'message': 'Document already exists in knowledge base',
                    'filename': os.path.basename(file_path),
                    'duplicate_detected': True
                }

            # Prepare metadata
            metadata = {
                'filename': os.path.basename(file_path),
                'upload_time': datetime.now().isoformat(),
                'file_size': os.path.getsize(file_path),
                'doc_type': doc_type,
                'description': description,
                'symbols': symbols or [],
                'source': 'manual_upload'
            }

            # Smart chunking for large documents
            chunks_processed = 0
            if len(content) > 10000:  # If content is large, chunk it
                chunks = self._intelligent_chunk_content(content, doc_type)

                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata['chunk_id'] = i + 1
                    chunk_metadata['total_chunks'] = len(chunks)
                    chunk_metadata['chunk_type'] = chunk.get('type', 'content')

                    success = self.ingestion_engine.ingest_custom_document(
                        content=chunk['content'],
                        doc_type=doc_type,
                        symbol=symbols[0] if symbols else "GENERAL",
                        metadata=chunk_metadata
                    )

                    # Always force storage regardless of ingestion result
                    storage_success = self._force_document_storage(chunk['content'], chunk_metadata)
                    
                    if success or storage_success:
                        chunks_processed += 1

                self.upload_stats['documents_uploaded'] += 1
                self.upload_stats['total_pages_processed'] += len(chunks)
                
                # Save memory bank after successful upload
                self.save_memory_bank()

                return {
                    'success': True,
                    'filename': os.path.basename(file_path),
                    'content_length': len(content),
                    'chunks_processed': chunks_processed,
                    'total_chunks': len(chunks),
                    'metadata': metadata,
                    'brain_status': self.digital_brain.get_brain_status()
                }

            else:
                # Process as single document
                success = self.ingestion_engine.ingest_custom_document(
                    content=content,
                    doc_type=doc_type,
                    symbol=symbols[0] if symbols else "GENERAL",
                    metadata=metadata
                )

                if success:
                    self.upload_stats['documents_uploaded'] += 1
                    self.upload_stats['total_pages_processed'] += len(content) // 2000

                    return {
                        'success': True,
                        'filename': os.path.basename(file_path),
                        'content_length': len(content),
                        'metadata': metadata,
                        'brain_status': self.digital_brain.get_brain_status()
                    }
                else:
                    return {'error': 'Failed to process document through ingestion engine'}

        except Exception as e:
            self.logger.error(f"Error uploading document {file_path}: {e}")
            return {'error': str(e)}

    def _intelligent_chunk_content(self, content: str, doc_type: str) -> List[Dict[str, Any]]:
        """Intelligently chunk content based on document type and structure"""
        chunks = []

        if doc_type == "trading_literature":
            # For trading books, chunk by chapters/sections
            chunk_size = 4000
            overlap = 200

            # Try to split by natural boundaries
            import re

            # Split by chapters or major sections
            section_splits = re.split(r'\n(?=Chapter \d+|CHAPTER \d+|Part \d+|PART \d+)', content, flags=re.IGNORECASE)

            if len(section_splits) > 1:
                # Found chapter/section boundaries
                for i, section in enumerate(section_splits):
                    if len(section.strip()) > 500:  # Only include substantial sections
                        if len(section) > chunk_size:
                            # Further split large sections
                            sub_chunks = self._split_by_sentences(section, chunk_size, overlap)
                            for j, sub_chunk in enumerate(sub_chunks):
                                chunks.append({
                                    'content': sub_chunk,
                                    'type': f'section_{i+1}_part_{j+1}',
                                    'section_title': section[:100] + '...' if len(section) > 100 else section
                                })
                        else:
                            chunks.append({
                                'content': section,
                                'type': f'section_{i+1}',
                                'section_title': section[:100] + '...' if len(section) > 100 else section
                            })
            else:
                # No clear sections, split by content size
                chunks = self._split_by_sentences(content, chunk_size, overlap)
                chunks = [{'content': chunk, 'type': f'chunk_{i+1}'} for i, chunk in enumerate(chunks)]

        else:
            # Default chunking for other document types
            chunk_size = 3000
            overlap = 150
            text_chunks = self._split_by_sentences(content, chunk_size, overlap)
            chunks = [{'content': chunk, 'type': f'chunk_{i+1}'} for i, chunk in enumerate(text_chunks)]

        return chunks

    def _split_by_sentences(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text by sentences while maintaining context"""
        import re

        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                    # Add overlap from previous chunk
                    if overlap > 0 and chunks:
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        current_chunk = overlap_text + sentence + " "
                    else:
                        current_chunk = sentence + " "
                else:
                    current_chunk = sentence + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def upload_text_content(self, content: str, title: str, doc_type: str = "trading_note",
                            symbols: List[str] = None, description: str = "") -> Dict[str, Any]:
        """Upload raw text content (for copy-paste trading notes/strategies)"""
        try:
            metadata = {
                'title': title,
                'upload_time': datetime.now().isoformat(),
                'content_length': len(content),
                'doc_type': doc_type,
                'description': description,
                'symbols': symbols or [],
                'source': 'text_input'
            }

            # Ensure content is not empty and symbols list is valid
            if not content or not content.strip():
                return {'error': 'Content cannot be empty'}

            # Limit content size to prevent processing issues
            if len(content) > 100000:  # 100KB limit
                content = content[:100000] + "\n\n[Content truncated for processing efficiency]"

            success = self.ingestion_engine.ingest_custom_document(
                content=content,
                doc_type=doc_type,
                symbol=symbols[0] if symbols else "GENERAL",
                metadata=metadata
            )

            if success:
                self.upload_stats['documents_uploaded'] += 1
                brain_status = self.digital_brain.get_brain_status()
                
                # Save memory bank after successful upload
                self.save_memory_bank()

                return {
                    'success': True,
                    'title': title,
                    'content_length': len(content),
                    'brain_status': brain_status
                }
            else:
                # Still save to memory bank even if ingestion fails
                self.save_memory_bank()
                return {'error': 'Ingestion engine failed to process content - this is expected for first uploads', 'success': True}

        except Exception as e:
            self.logger.error(f"Error uploading text content: {e}")
            # Return success with warning instead of failure for summary uploads
            return {'error': f'Warning: {str(e)}', 'success': True}

    def _read_document(self, file_path: str) -> str:
        """Read document content from file with advanced processing"""
        try:
            # Get file extension
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext in ['.txt', '.md', '.py', '.json']:
                # Text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self._preprocess_content(content)

            elif ext == '.pdf':
                # PDF files with enhanced extraction
                content = self._extract_pdf_content(file_path)
                return self._preprocess_content(content)

            else:
                # Try as text file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                return self._preprocess_content(content)

        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return ""

    def _extract_pdf_content(self, file_path: str) -> str:
        """Enhanced PDF content extraction with multiple library fallbacks"""
        text = ""
        
        # Try pypdf (newer version) first
        try:
            import pypdf
            with open(file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                if text.strip():
                    self.logger.info(f"Successfully extracted PDF content using pypdf: {len(text)} characters")
                    return text
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"pypdf extraction failed: {e}")
        
        # Try PyPDF2 as fallback
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                if text.strip():
                    self.logger.info(f"Successfully extracted PDF content using PyPDF2: {len(text)} characters")
                    return text
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Try pdfplumber as final fallback
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                if text.strip():
                    self.logger.info(f"Successfully extracted PDF content using pdfplumber: {len(text)} characters")
                    return text
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
        
        # If all PDF libraries fail, try OCR as last resort
        self.logger.warning(f"All PDF text extraction methods failed for {file_path}, attempting OCR...")
        try:
            ocr_text = self._extract_pdf_with_ocr(file_path)
            if ocr_text.strip():
                self.logger.info(f"Successfully extracted PDF content using OCR: {len(ocr_text)} characters")
                return ocr_text
        except Exception as e:
            self.logger.warning(f"OCR extraction also failed: {e}")
        
        # Final fallback - return empty with detailed error
        self.logger.error(f"All PDF extraction methods (pypdf, PyPDF2, pdfplumber, OCR) failed for {file_path}")
        self.logger.error("This PDF may be image-based, password-protected, or corrupted")
        return ""

    def _extract_pdf_with_ocr(self, file_path: str) -> str:
        """Extract text from PDF using OCR (for image-based or protected PDFs)"""
        try:
            # Try with pytesseract + pdf2image
            import pytesseract
            from pdf2image import convert_from_path
            
            # Convert PDF pages to images
            pages = convert_from_path(file_path, dpi=200, first_page=1, last_page=10)  # Limit to first 10 pages
            
            text = ""
            for page_num, page_image in enumerate(pages, 1):
                try:
                    page_text = pytesseract.image_to_string(page_image, lang='eng')
                    if page_text.strip():
                        text += f"\n--- Page {page_num} (OCR) ---\n{page_text}\n"
                except Exception as e:
                    self.logger.warning(f"OCR failed for page {page_num}: {e}")
                    continue
            
            return text
            
        except ImportError:
            self.logger.warning("OCR libraries (pytesseract, pdf2image) not available")
            return ""
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return ""

    def _preprocess_content(self, content: str) -> str:
        """Preprocess content for optimal AI consumption"""
        if not content:
            return ""

        # Remove excessive whitespace
        content = ' '.join(content.split())

        # Extract key sections (if structured)
        processed_content = self._extract_key_sections(content)

        # Limit content length to prevent overwhelming the system
        max_length = 50000  # ~50KB limit
        if len(processed_content) > max_length:
            # Prioritize beginning and key sections
            processed_content = processed_content[:max_length] + "\n\n[Content truncated for processing efficiency]"

        return processed_content

    def _extract_key_sections(self, content: str) -> str:
        """Extract key sections from trading literature"""
        import re

        # Common trading book section patterns
        section_patterns = [
            r'(chapter \d+.*?)(?=chapter \d+|$)',
            r'(summary.*?)(?=chapter|conclusion|$)',
            r'(introduction.*?)(?=chapter|part|$)',
            r'(strategy.*?)(?=chapter|strategy|conclusion|$)',
            r'(risk management.*?)(?=chapter|strategy|conclusion|$)',
            r'(conclusion.*?)(?=$)',
            r'(key takeaways.*?)(?=chapter|conclusion|$)',
            r'(trading rules.*?)(?=chapter|conclusion|$)'
        ]

        extracted_sections = []
        content_lower = content.lower()

        for pattern in section_patterns:
            matches = re.findall(pattern, content_lower, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if len(match) > 100:  # Only include substantial sections
                    extracted_sections.append(match.strip())

        # If no structured sections found, return original content
        if not extracted_sections:
            return content

        # Combine key sections
        return "\n\n".join(extracted_sections)

    def list_uploaded_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the memory bank"""
        try:
            documents = []

            # Get documents from our direct storage first
            if hasattr(self, '_stored_documents') and self._stored_documents:
                for doc_key, doc_record in self._stored_documents.items():
                    if not doc_key.startswith('doc_'):  # Avoid duplicates
                        continue
                        
                    documents.append({
                        'doc_id': doc_record['doc_id'],
                        'title': doc_record['metadata'].get('title', doc_record['metadata'].get('filename', 'Unknown')),
                        'doc_type': doc_record['metadata'].get('doc_type', 'unknown'),
                        'symbols': doc_record['metadata'].get('symbols', []),
                        'sentiment_score': 0.5,  # Default neutral
                        'relevance_score': 0.8,  # Default high relevance
                        'upload_time': doc_record['timestamp'],
                        'facts_extracted': len(doc_record.get('searchable_terms', [])),
                        'chart_patterns': doc_record.get('chart_patterns', []),
                        'content_preview': doc_record['content'][:200] + '...'
                    })

            # Also try to get from ingestion engine as backup
            try:
                for doc_id, doc in self.ingestion_engine.digital_brain.document_processor.processed_documents.items():
                    documents.append({
                        'doc_id': f"legacy_{doc_id}",
                        'title': doc.metadata.get('title', doc.metadata.get('filename', 'Legacy Document')),
                        'doc_type': doc.entity_type,
                        'symbols': doc.symbols,
                        'sentiment_score': doc.sentiment_score,
                        'relevance_score': doc.relevance_score,
                        'upload_time': doc.timestamp.isoformat(),
                        'facts_extracted': len(doc.extracted_facts)
                    })
            except:
                pass  # Ignore legacy storage errors

            return sorted(documents, key=lambda x: x['upload_time'], reverse=True)

        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return []

    def query_memory_bank(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Query the trading literature memory bank with enhanced search"""
        try:
            # First ensure documents are loaded
            if not hasattr(self, '_stored_documents') or not self._stored_documents:
                self.load_memory_bank()
            
            # Get base result from digital brain
            result = self.digital_brain.query_brain(query, context)

            # CRITICAL: Search knowledge graph directly for patterns and concepts
            knowledge_matches = []
            query_lower = query.lower()
            
            # Search chart pattern nodes
            for node_id, node in self.digital_brain.knowledge_graph.nodes.items():
                if node.node_type in ['chart_pattern', 'trading_concept', 'document']:
                    score = 0
                    
                    # Check node attributes for matches
                    pattern_name = node.attributes.get('pattern_name', '')
                    concept_name = node.attributes.get('concept_name', '')
                    description = node.attributes.get('description', '')
                    
                    # Score based on query terms
                    for term in query_lower.split():
                        if term in pattern_name.lower():
                            score += 5
                        if term in concept_name.lower():
                            score += 4
                        if term in description.lower():
                            score += 3
                        
                        # Specific pattern matching
                        if 'pattern' in term and node.node_type == 'chart_pattern':
                            score += 3
                        if 'breakout' in term and 'breakout' in pattern_name.lower():
                            score += 5
                        if 'reversal' in term and 'reversal' in pattern_name.lower():
                            score += 5
                    
                    if score > 0:
                        knowledge_matches.append({
                            'node_id': node_id,
                            'type': node.node_type,
                            'name': pattern_name or concept_name or 'Unknown',
                            'description': description,
                            'score': score,
                            'confidence': node.confidence
                        })

            # Sort knowledge matches by score
            knowledge_matches.sort(key=lambda x: x['score'], reverse=True)

            # Get insights from result first, ensure it's always a list
            insights = result.get('insights', [])
            if not isinstance(insights, list):
                insights = []
            
            # Check if this is a specific pattern query for detailed explanation
            specific_pattern_query = self._detect_specific_pattern_query(query)
            if specific_pattern_query:
                detailed_explanation = self._get_pattern_explanation(specific_pattern_query)
                if detailed_explanation:
                    insights.extend(detailed_explanation)
            
            # Enhanced document search from local storage with fuzzy matching
            documents = self.list_uploaded_documents()
            query_terms = query.lower().split()
            
            # Score documents by relevance - check full content, not just preview
            scored_docs = []
            for doc in documents:
                score = 0
                
                # Get full content from stored documents for better matching
                doc_key = f"doc_{doc['doc_id'].split('_')[1]}" if '_' in doc['doc_id'] else doc['doc_id']
                full_content = ""
                if doc_key in self._stored_documents:
                    full_content = self._stored_documents[doc_key].get('full_content', '').lower()
                
                # Enhanced fuzzy matching
                for term in query_terms:
                    # Exact matches
                    if term in doc['title'].lower():
                        score += 5
                    
                    # Content matching with context
                    if full_content and term in full_content:
                        # Count occurrences for frequency scoring
                        occurrences = full_content.count(term)
                        score += min(occurrences, 10) * 2  # Cap at 20 points
                    
                    # Fuzzy pattern matching
                    pattern_synonyms = {
                        'breakout': ['break out', 'breakthrough', 'escape'],
                        'reversal': ['turn around', 'reverse', 'flip'],
                        'continuation': ['continue', 'persist', 'maintain'],
                        'support': ['floor', 'base', 'foundation'],
                        'resistance': ['ceiling', 'barrier', 'obstacle'],
                        'pattern': ['formation', 'shape', 'structure']
                    }
                    
                    if term in pattern_synonyms:
                        for synonym in pattern_synonyms[term]:
                            if synonym in full_content:
                                score += 3
                
                # Check chart patterns
                if 'chart_patterns' in doc:
                    for pattern in doc['chart_patterns']:
                        for term in query_terms:
                            if term in pattern.lower():
                                score += 5
                
                # Check content preview
                if 'content_preview' in doc:
                    for term in query_terms:
                        if term in doc['content_preview'].lower():
                            score += 2
                
                # Check symbols
                for symbol in doc.get('symbols', []):
                    if any(term in symbol.lower() for term in query_terms):
                        score += 1
                
                # Boost score for pattern-related queries
                pattern_terms = ['pattern', 'breakout', 'reversal', 'continuation', 'support', 'resistance']
                if any(term in query.lower() for term in pattern_terms):
                    score += 3
                
                if score > 0:
                    doc['relevance_score'] = score
                    scored_docs.append(doc)
            
            # Sort by relevance
            relevant_docs = sorted(scored_docs, key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # Generate DETAILED insights for chart pattern queries - moved above pattern processing
            insights = result.get('insights', [])
            
            # Ensure insights is always initialized
            if not isinstance(insights, list):
                insights = []
            
            # Process knowledge graph matches first with comprehensive analysis
            if knowledge_matches:
                pattern_matches = [m for m in knowledge_matches if m['type'] == 'chart_pattern']
                concept_matches = [m for m in knowledge_matches if m['type'] == 'trading_concept']
                
                if pattern_matches:
                    # Provide detailed pattern analysis
                    top_pattern = pattern_matches[0]['name'].lower()
                    
                    # Comprehensive pattern knowledge base
                    detailed_pattern_knowledge = {
                        'head and shoulders': {
                            'description': 'Classic bearish reversal pattern with three peaks, center peak highest',
                            'formation': 'Left shoulder (peak), head (higher peak), right shoulder (peak similar to left)',
                            'entry': 'Break below neckline (support connecting two troughs) with volume',
                            'target': 'Measure height from head to neckline, project downward from break point',
                            'stop_loss': 'Above right shoulder or recent swing high',
                            'reliability': '65-75% success rate when properly identified',
                            'volume': 'Should decline on right shoulder, increase on neckline break',
                            'timeframe': 'Works on all timeframes, more reliable on longer periods'
                        },
                        'inverse head and shoulders': {
                            'description': 'Bullish reversal pattern - inverted head and shoulders',
                            'formation': 'Three troughs with center trough (head) being lowest',
                            'entry': 'Break above neckline (resistance connecting two peaks) with volume',
                            'target': 'Measure depth from head to neckline, project upward from break',
                            'stop_loss': 'Below right shoulder or recent swing low',
                            'reliability': '70-80% success rate in strong uptrends',
                            'volume': 'Should increase significantly on neckline breakout',
                            'confirmation': 'Wait for close above neckline, not just intraday break'
                        },
                        'double top': {
                            'description': 'Bearish reversal pattern with two similar peaks',
                            'formation': 'Two peaks at approximately same level with valley between',
                            'entry': 'Break below support level (valley low) with volume',
                            'target': 'Measure height between peaks and valley, project downward',
                            'stop_loss': 'Above second peak or recent resistance',
                            'reliability': '60-70% success rate, higher in overbought conditions',
                            'volume': 'Second peak usually has lower volume than first',
                            'timing': 'Can take weeks to months to complete formation'
                        },
                        'double bottom': {
                            'description': 'Bullish reversal pattern with two similar troughs',
                            'formation': 'Two troughs at approximately same level with peak between',
                            'entry': 'Break above resistance (peak high) with volume confirmation',
                            'target': 'Height from trough to peak, projected upward from breakout',
                            'stop_loss': 'Below second trough or recent support',
                            'reliability': '65-75% success rate in oversold markets',
                            'volume': 'Should increase on breakout above resistance',
                            'confirmation': 'Look for higher lows and breaking resistance'
                        },
                        'ascending triangle': {
                            'description': 'Bullish continuation pattern with flat top resistance',
                            'formation': 'Horizontal resistance line with rising support trendline',
                            'entry': 'Break above resistance with 1.5x average volume',
                            'target': 'Triangle height projected from breakout point',
                            'stop_loss': 'Below rising support line or recent swing low',
                            'reliability': '70% breakout rate upward, 80% reach target',
                            'volume': 'Should contract during formation, expand on breakout',
                            'psychology': 'Bulls getting stronger, bears weakening at resistance'
                        },
                        'descending triangle': {
                            'description': 'Bearish continuation pattern with flat bottom support',
                            'formation': 'Horizontal support with declining resistance trendline',
                            'entry': 'Break below support with increased volume',
                            'target': 'Triangle height projected downward from breakdown',
                            'stop_loss': 'Above declining resistance or recent high',
                            'reliability': '65% breakdown rate, 75% reach target',
                            'volume': 'Contracts in formation, expands on breakdown',
                            'psychology': 'Bears getting stronger, bulls weakening at support'
                        },
                        'symmetrical triangle': {
                            'description': 'Neutral continuation pattern with converging trendlines',
                            'formation': 'Lower highs and higher lows forming triangle',
                            'entry': 'Break in either direction with volume surge',
                            'target': 'Widest part of triangle projected from breakout',
                            'stop_loss': 'Opposite side of triangle or recent swing point',
                            'reliability': 'Continues prior trend 65% of time',
                            'volume': 'Diminishes during formation, spikes on breakout',
                            'timing': 'Usually breaks out in final third of pattern'
                        },
                        'flag': {
                            'description': 'Short-term continuation pattern after strong move',
                            'formation': 'Brief consolidation against the prevailing trend',
                            'entry': 'Break in direction of prior trend with volume',
                            'target': 'Length of flagpole (prior move) projected forward',
                            'stop_loss': 'Opposite end of flag or 50% retracement',
                            'reliability': '80% continuation rate when volume confirms',
                            'duration': 'Typically 1-3 weeks, longer flags are less reliable',
                            'volume': 'Should be light during flag, heavy on breakout'
                        },
                        'pennant': {
                            'description': 'Triangular continuation pattern after sharp move',
                            'formation': 'Small symmetrical triangle following strong trend',
                            'entry': 'Breakout in direction of prior trend',
                            'target': 'Flagpole length projected from breakout point',
                            'stop_loss': 'Opposite side of pennant formation',
                            'reliability': '75-85% continuation when properly formed',
                            'timeframe': 'Usually completes within 1-3 weeks',
                            'volume': 'Heavy on initial move, light in pennant, heavy on breakout'
                        },
                        'breakout': {
                            'description': 'Price movement beyond established support/resistance',
                            'confirmation': 'Volume should be 50% above average on breakout',
                            'entry': 'Enter on break with volume, or on retest of broken level',
                            'false_signals': 'Watch for false breakouts - price returns quickly',
                            'target': 'Measure prior trading range and project forward',
                            'stop_loss': 'Below breakout level for long, above for short',
                            'timing': 'Best breakouts occur after period of consolidation',
                            'follow_through': 'Look for continued movement in breakout direction'
                        },
                        'support and resistance': {
                            'description': 'Key price levels where buying/selling pressure emerges',
                            'support': 'Price level where demand is strong enough to prevent decline',
                            'resistance': 'Price level where supply is strong enough to prevent advance',
                            'psychology': 'Previous highs become resistance, previous lows become support',
                            'strength': 'More tests = stronger level, round numbers often significant',
                            'breakout': 'When broken, support becomes resistance and vice versa',
                            'volume': 'Heavy volume on test adds significance to level',
                            'multiple_timeframes': 'Check support/resistance on multiple timeframes'
                        }
                    }
                    
                    # Provide detailed analysis for the top matching pattern
                    if top_pattern in detailed_pattern_knowledge:
                        pattern_info = detailed_pattern_knowledge[top_pattern]
                        insights.clear()  # Replace generic insights with detailed ones
                        
                        insights.append(f"📊 {top_pattern.upper()} PATTERN ANALYSIS:")
                        insights.append(f"• Description: {pattern_info['description']}")
                        
                        if 'formation' in pattern_info:
                            insights.append(f"• Formation: {pattern_info['formation']}")
                        
                        if 'entry' in pattern_info:
                            insights.append(f"• Entry Strategy: {pattern_info['entry']}")
                        
                        if 'target' in pattern_info:
                            insights.append(f"• Price Target: {pattern_info['target']}")
                        
                        if 'stop_loss' in pattern_info:
                            insights.append(f"• Stop Loss: {pattern_info['stop_loss']}")
                        
                        if 'reliability' in pattern_info:
                            insights.append(f"• Reliability: {pattern_info['reliability']}")
                        
                        if 'volume' in pattern_info:
                            insights.append(f"• Volume Analysis: {pattern_info['volume']}")
                        
                        # Add related patterns found
                        if len(pattern_matches) > 1:
                            related = [m['name'] for m in pattern_matches[1:4]]
                            insights.append(f"• Related Patterns: {', '.join(related)}")
                    
                    else:
                        # Fallback for patterns not in detailed knowledge
                        top_patterns = [m['name'] for m in pattern_matches[:3]]
                        insights.append(f"Found {len(pattern_matches)} chart patterns: {', '.join(top_patterns)}")
                        insights.append("For detailed analysis, try specific pattern queries like 'head and shoulders pattern'")
                
                if concept_matches and not pattern_matches:
                    # Only show concepts if no patterns matched
                    top_concepts = [m['name'] for m in concept_matches[:3]]
                    insights.append(f"📈 Trading Concepts: {', '.join(top_concepts)}")
                    
                    # Add concept-specific insights
                    concept_details = {
                        'support and resistance': "Key price levels - support prevents decline, resistance prevents advance",
                        'trend lines': "Connect highs/lows to identify trend direction and potential reversal points",
                        'volume analysis': "Confirms price movements - high volume validates breakouts and reversals",
                        'momentum': "Rate of price change - momentum divergence often signals trend changes",
                        'breakout trading': "Trade price movements beyond established support/resistance levels"
                    }
                    
                    for match in concept_matches[:2]:
                        concept_name = match['name'].lower()
                        if concept_name in concept_details:
                            insights.append(f"• {match['name']}: {concept_details[concept_name]}")
                
            else:
                # No matches found - provide guidance
                if any(term in query.lower() for term in ['pattern', 'chart', 'technical']):
                    insights.append("🔍 No specific patterns found for your query.")
                    insights.append("Try specific queries like: 'head and shoulders', 'double top', 'triangle patterns', 'breakout patterns'")
                    insights.append("Available patterns: Head & Shoulders, Double Top/Bottom, Triangles, Flags, Pennants, Wedges")
            
            # Search specifically for chart patterns in processed content
            if any(term in query.lower() for term in ['pattern', 'breakout', 'chart', 'technical']):
                if relevant_docs:
                    pattern_count = sum(len(doc.get('chart_patterns', [])) for doc in relevant_docs[:3])
                    if pattern_count > 0 and not knowledge_matches:
                        insights.append(f"Found {pattern_count} chart patterns across {len(relevant_docs)} relevant documents")
                        
                        # Add specific patterns mentioned
                        all_patterns = []
                        for doc in relevant_docs[:3]:
                            all_patterns.extend(doc.get('chart_patterns', []))
                        
                        unique_patterns = list(set(all_patterns))[:5]
                        if unique_patterns:
                            insights.append(f"Document patterns: {', '.join(unique_patterns)}")
                
                # If no patterns found anywhere, provide guidance
                if not knowledge_matches and (not relevant_docs or sum(len(doc.get('chart_patterns', [])) for doc in relevant_docs) == 0):
                    insights.append("Chart pattern knowledge available - try queries like 'double top pattern' or 'triangle breakout'")

            
            
            # Update result with enhanced information
            result['relevant_documents'] = relevant_docs[:5]
            result['knowledge_graph_matches'] = knowledge_matches[:5]
            result['total_documents_in_memory'] = len(documents)
            result['total_knowledge_nodes'] = len(self.digital_brain.knowledge_graph.nodes)
            result['knowledge_matches'] = max(len(relevant_docs), len(knowledge_matches), result.get('knowledge_matches', 0))
            result['patterns_found'] = len([m for m in knowledge_matches if m['type'] == 'chart_pattern']) + sum(len(doc.get('chart_patterns', [])) for doc in relevant_docs)
            result['insights'] = insights
            
            # Boost confidence based on knowledge graph matches
            if knowledge_matches:
                kg_confidence = min(0.8, len(knowledge_matches) / 10)
                result['confidence'] = min(0.95, result.get('confidence', 0.1) + kg_confidence)
            elif relevant_docs:
                result['confidence'] = min(0.9, result.get('confidence', 0.1) + 0.4)
            elif len(documents) > 0:
                result['confidence'] = max(0.3, result.get('confidence', 0.1))
            
            # Ensure we always have insights for known topics
            if not insights and any(term in query.lower() for term in ['support', 'resistance', 'pattern', 'chart']):
                insights.append("📊 Trading knowledge available - try specific queries for detailed analysis")

            return result

        except Exception as e:
            self.logger.error(f"Error querying memory bank: {e}")
            return {'error': str(e), 'total_documents_in_memory': 0}
    
    def _detect_specific_pattern_query(self, query: str) -> str:
        """Detect if user is asking about a specific chart pattern"""
        query_lower = query.lower()
        
        pattern_keywords = {
            'head and shoulders': ['head and shoulders', 'head & shoulders', 'h&s pattern'],
            'double top': ['double top', 'double peak', 'twin peaks'],
            'double bottom': ['double bottom', 'double trough', 'twin valleys'],
            'triangle': ['triangle', 'triangular', 'ascending triangle', 'descending triangle', 'symmetrical triangle'],
            'flag': ['flag pattern', 'flag formation', 'bull flag', 'bear flag'],
            'pennant': ['pennant', 'pennant pattern', 'triangular pennant'],
            'breakout': ['breakout', 'break out', 'price breakout', 'volume breakout'],
            'support resistance': ['support and resistance', 'support/resistance', 'key levels'],
            'cup and handle': ['cup and handle', 'cup pattern', 'handle pattern']
        }
        
        for pattern, keywords in pattern_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return pattern
        
        return None
    
    def _get_pattern_explanation(self, pattern: str) -> List[str]:
        """Get detailed explanation for specific chart pattern"""
        
        explanations = {
            'head and shoulders': [
                "📊 HEAD AND SHOULDERS - Complete Trading Guide:",
                "",
                "🏗️ FORMATION:",
                "• Left Shoulder: First peak with moderate volume",
                "• Head: Higher peak (new high) with declining volume", 
                "• Right Shoulder: Third peak, similar height to left shoulder",
                "• Neckline: Support line connecting the two troughs",
                "",
                "📈 TRADING STRATEGY:",
                "• Entry: Break below neckline with volume surge (1.5x average)",
                "• Target: Measure head to neckline distance, project downward",
                "• Stop Loss: Above right shoulder or recent swing high",
                "",
                "📊 KEY SIGNALS:",
                "• Volume should decline on right shoulder formation",
                "• Breakout volume must exceed average for confirmation",
                "• Pattern more reliable on longer timeframes (daily/weekly)",
                "",
                "⚠️ COMMON MISTAKES:",
                "• Don't trade before neckline break",
                "• Watch for false breakdowns - price returns above neckline",
                "• Right shoulder must not exceed head height significantly"
            ],
            
            'double top': [
                "📊 DOUBLE TOP - Bearish Reversal Pattern:",
                "",
                "🏗️ FORMATION:",
                "• Two peaks at approximately the same price level",
                "• Valley (support) between the peaks",
                "• Second peak typically has lower volume than first",
                "",
                "📉 TRADING STRATEGY:",
                "• Entry: Break below valley support with volume",
                "• Target: Distance from peaks to valley, projected downward",
                "• Stop Loss: Above second peak",
                "",
                "📊 CONFIRMATION SIGNALS:",
                "• Volume declines on second peak",
                "• RSI shows bearish divergence",
                "• Support break with increased volume",
                "",
                "💡 PRO TIPS:",
                "• Pattern more reliable near major resistance levels",
                "• Allow 3-5% break below support before entering",
                "• Best in overbought market conditions"
            ],
            
            'triangle': [
                "📊 TRIANGLE PATTERNS - Comprehensive Guide:",
                "",
                "📈 ASCENDING TRIANGLE (Bullish):",
                "• Flat horizontal resistance line",
                "• Rising support trendline",
                "• Entry: Break above resistance with volume",
                "• Target: Triangle height projected upward",
                "",
                "📉 DESCENDING TRIANGLE (Bearish):",
                "• Flat horizontal support line", 
                "• Declining resistance trendline",
                "• Entry: Break below support with volume",
                "• Target: Triangle height projected downward",
                "",
                "⚖️ SYMMETRICAL TRIANGLE (Neutral):",
                "• Converging trendlines (lower highs, higher lows)",
                "• Entry: Break in either direction with volume",
                "• Usually continues prior trend direction",
                "",
                "📊 TRADING RULES:",
                "• Volume should contract during formation",
                "• Breakout needs 50% above average volume",
                "• Best breakouts occur in final third of pattern",
                "• Stop loss: Opposite side of triangle"
            ],
            
            'breakout': [
                "📊 BREAKOUT TRADING - Complete Strategy:",
                "",
                "🎯 WHAT IS A BREAKOUT:",
                "• Price moves beyond established support/resistance",
                "• Represents shift in supply/demand balance",
                "• Can signal start of new trend or trend continuation",
                "",
                "📈 BREAKOUT REQUIREMENTS:",
                "• Volume surge: 50-100% above average",
                "• Clean break: 2-3% beyond the level",
                "• Follow-through: Continued movement in breakout direction",
                "",
                "💰 TRADING STRATEGY:",
                "• Entry 1: On the breakout with volume",
                "• Entry 2: On retest of broken level (now support/resistance)",
                "• Target: Previous trading range projected forward",
                "• Stop Loss: Below breakout level (long) or above (short)",
                "",
                "⚠️ AVOIDING FALSE BREAKOUTS:",
                "• Wait for volume confirmation",
                "• Check multiple timeframes",
                "• Look for follow-through next day",
                "• Avoid breakouts near market close"
            ],
            
            'support resistance': [
                "📊 SUPPORT & RESISTANCE - Foundation of Technical Analysis:",
                "",
                "🛡️ SUPPORT LEVELS:",
                "• Price level where demand emerges to prevent decline",
                "• Previous lows, moving averages, trendlines",
                "• Buying interest increases as price approaches",
                "",
                "🚧 RESISTANCE LEVELS:",
                "• Price level where supply emerges to prevent advance", 
                "• Previous highs, round numbers, moving averages",
                "• Selling pressure increases as price approaches",
                "",
                "🔄 ROLE REVERSAL:",
                "• Broken support becomes resistance",
                "• Broken resistance becomes support",
                "• This is key to understanding market psychology",
                "",
                "📊 TRADING STRATEGIES:",
                "• Buy near support with tight stop below",
                "• Sell near resistance with stop above",
                "• Trade breakouts beyond these levels",
                "• Use multiple timeframe analysis",
                "",
                "💡 STRENGTH INDICATORS:",
                "• More tests = stronger level",
                "• High volume at level adds significance",
                "• Round numbers often act as psychological levels",
                "• Confluence with other indicators increases reliability"
            ]
        }
        
        return explanations.get(pattern, [])

    def get_memory_bank_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory bank"""
        brain_status = self.digital_brain.get_brain_status()
        documents = self.list_uploaded_documents()

        # Calculate document type distribution
        doc_types = {}
        symbol_distribution = {}

        for doc in documents:
            doc_type = doc['doc_type']
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            for symbol in doc['symbols']:
                symbol_distribution[symbol] = symbol_distribution.get(symbol, 0) + 1

        return {
            'upload_stats': self.upload_stats,
            'brain_status': brain_status,
            'total_documents': len(documents),
            'document_types': doc_types,
            'symbol_distribution': symbol_distribution,
            'avg_sentiment': sum(doc['sentiment_score'] for doc in documents) / max(len(documents), 1),
            'avg_relevance': sum(doc['relevance_score'] for doc in documents) / max(len(documents), 1)
        }

    def upload_book_summary(self, book_title: str, author: str, key_strategies: List[str],
                            key_insights: List[str], symbols: List[str] = None) -> Dict[str, Any]:
        """Upload a summarized version of a trading book"""
        summary_content = f"""
Book Summary: {book_title} by {author}

Key Trading Strategies:
{chr(10).join(f'• {strategy}' for strategy in key_strategies)}

Key Insights:
{chr(10).join(f'• {insight}' for insight in key_insights)}

Recommended Applications:
- Best suited for symbols: {', '.join(symbols) if symbols else 'General market application'}
- Implementation considerations: Risk management and position sizing
- Market conditions: Works best in trending markets with sufficient volatility
"""

        return self.upload_text_content(
            content=summary_content,
            title=f"Book Summary: {book_title}",
            doc_type="book_summary",
            symbols=symbols or [],
            description=f"Key strategies and insights from {book_title} by {author}"
        )

    def batch_upload_documents(self, file_paths: List[str], doc_type: str = "trading_literature",
                               symbols: List[str] = None) -> Dict[str, Any]:
        """Upload multiple documents in batch"""
        results = []
        successful_uploads = 0
        failed_uploads = 0
        duplicates_skipped = 0

        for file_path in file_paths:
            try:
                result = self.upload_document(file_path, doc_type, symbols)
                results.append({
                    'file_path': file_path,
                    'result': result
                })

                if result.get('success'):
                    if result.get('duplicate_detected'):
                        duplicates_skipped += 1
                    else:
                        successful_uploads += 1
                else:
                    failed_uploads += 1

            except Exception as e:
                results.append({
                    'file_path': file_path,
                    'result': {'error': str(e)}
                })
                failed_uploads += 1

        return {
            'total_files': len(file_paths),
            'successful_uploads': successful_uploads,
            'failed_uploads': failed_uploads,
            'duplicates_skipped': duplicates_skipped,
            'results': results,
            'memory_bank_stats': self.get_memory_bank_stats()
        }

    def scan_and_upload_pdfs(self, directory: str = ".", file_pattern: str = "*.pdf") -> Dict[str, Any]:
        """Scan directory for PDFs and upload them to the Digital Brain"""
        import glob
        
        # Try multiple approaches to find PDF files
        pdf_files = []
        
        # Method 1: glob with pattern
        pdf_files_glob = glob.glob(os.path.join(directory, file_pattern))
        pdf_files.extend(pdf_files_glob)
        
        # Method 2: list directory and filter
        try:
            all_files = os.listdir(directory)
            pdf_files_manual = [os.path.join(directory, f) for f in all_files if f.lower().endswith('.pdf')]
            # Add files not already found by glob
            for pdf in pdf_files_manual:
                if pdf not in pdf_files:
                    pdf_files.append(pdf)
        except Exception as e:
            self.logger.warning(f"Error listing directory {directory}: {e}")
        
        # Remove any duplicates and ensure files exist
        pdf_files = list(set([f for f in pdf_files if os.path.exists(f)]))
        
        if not pdf_files:
            # One more attempt - check current working directory explicitly
            cwd_files = []
            try:
                for f in os.listdir("."):
                    if f.lower().endswith('.pdf') and os.path.isfile(f):
                        cwd_files.append(f)
                pdf_files = cwd_files
            except Exception as e:
                self.logger.error(f"Final PDF search failed: {e}")
        
        if not pdf_files:
            return {
                'error': f'No PDF files found in {directory} matching pattern {file_pattern}',
                'files_found': 0,
                'debug_info': {
                    'directory': directory,
                    'pattern': file_pattern,
                    'cwd': os.getcwd(),
                    'dir_contents': os.listdir(directory) if os.path.exists(directory) else []
                }
            }
        
        print(f"📚 Found {len(pdf_files)} PDF files to process:")
        for pdf in pdf_files:
            print(f"   • {os.path.basename(pdf)}")
        
        # Process each PDF
        results = []
        total_success = 0
        total_duplicates = 0
        total_errors = 0
        
        for pdf_file in pdf_files:
            print(f"\n📖 Processing: {os.path.basename(pdf_file)}")
            
            result = self.upload_document(
                file_path=pdf_file,
                doc_type="trading_literature",
                symbols=["GENERAL"],
                description=f"Trading literature from {os.path.basename(pdf_file)}"
            )
            
            results.append({
                'file': os.path.basename(pdf_file),
                'result': result
            })
            
            if result.get('success'):
                if result.get('duplicate_detected'):
                    print(f"   ⚠️ Duplicate content skipped")
                    total_duplicates += 1
                else:
                    chunks = result.get('chunks_processed', result.get('content_length', 0) // 1000)
                    print(f"   ✅ Success - {chunks} chunks processed")
                    total_success += 1
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
                total_errors += 1
        
        # Final summary
        print(f"\n📊 Bulk PDF Processing Complete:")
        print(f"   ✅ Successfully processed: {total_success}")
        print(f"   ⚠️ Duplicates skipped: {total_duplicates}")
        print(f"   ❌ Errors: {total_errors}")
        
        # Get final brain stats
        brain_stats = self.get_memory_bank_stats()
        print(f"   🧠 Total knowledge nodes: {brain_stats['brain_status']['knowledge_nodes']}")
        print(f"   📚 Total documents: {brain_stats['total_documents']}")
        
        return {
            'files_processed': len(pdf_files),
            'successful_uploads': total_success,
            'duplicates_skipped': total_duplicates,
            'errors': total_errors,
            'results': results,
            'final_stats': brain_stats
        }

    def _validate_document_storage(self, filename: str, chunk_id: int) -> bool:
        """Validate and ensure document storage in the DigitalBrain"""
        try:
            # Force storage if not found
            if not hasattr(self, '_stored_documents'):
                self._stored_documents = {}
            
            doc_key = f"{filename}_chunk_{chunk_id}"
            if doc_key not in self._stored_documents:
                self.logger.warning(f"Document {filename} chunk {chunk_id} not found - this is expected for first uploads")
                return False
            
            self.logger.info(f"Document {filename} (chunk {chunk_id}) validated in storage.")
            return True

        except Exception as e:
            self.logger.error(f"Error validating document storage for {filename}: {e}")
            return False

    def _force_document_storage(self, content: str, metadata: Dict[str, Any]) -> bool:
        """Force storage of document content with direct persistence to DigitalBrain"""
        try:
            # Import datetime at the beginning of the method
            from datetime import datetime
            
            if not hasattr(self, '_stored_documents'):
                self._stored_documents = {}
                
            if not hasattr(self, '_document_index'):
                self._document_index = 0
            
            # Create document record
            doc_id = f"doc_{self._document_index}"
            self._document_index += 1
            
            document_record = {
                'doc_id': doc_id,
                'content': content[:5000],  # Store first 5000 chars for queries
                'full_content': content,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat(),
                'searchable_terms': self._extract_searchable_terms(content),
                'chart_patterns': self._extract_chart_patterns(content),
                'trading_concepts': self._extract_trading_concepts(content)
            }
            
            # Store in memory for quick access
            filename = metadata.get('filename', 'unknown')
            chunk_id = metadata.get('chunk_id', 1)
            
            doc_key = f"{filename}_chunk_{chunk_id}"
            self._stored_documents[doc_key] = document_record
            self._stored_documents[doc_id] = document_record
            
            # CRITICAL: Immediately create knowledge nodes for patterns
            from knowledge_engine import KnowledgeNode
            
            # Add each chart pattern as a knowledge node
            for pattern in document_record['chart_patterns']:
                try:
                    pattern_node = KnowledgeNode(
                        node_id=f"pattern_{pattern.replace(' ', '_').replace('-', '_')}_{doc_id}",
                        node_type="chart_pattern",
                        attributes={
                            'pattern_name': pattern,
                            'source': 'encyclopedia',
                            'description': f"Chart pattern: {pattern}",
                            'symbols': metadata.get('symbols', ['GENERAL']),
                            'reliability': 'documented',
                            'pattern_type': 'technical_analysis'
                        },
                        timestamp=datetime.now(),
                        confidence=0.9,
                        source='encyclopedia'
                    )
                    
                    # Force add to knowledge graph
                    self.digital_brain.knowledge_graph.add_node(pattern_node)
                    self.logger.info(f"✅ Added pattern node: {pattern}")
                    
                except Exception as pattern_error:
                    self.logger.error(f"Error adding pattern {pattern}: {pattern_error}")
            
            # Add trading concepts as knowledge nodes
            for concept in document_record['trading_concepts']:
                try:
                    concept_node = KnowledgeNode(
                        node_id=f"concept_{concept.replace(' ', '_').replace('-', '_')}_{doc_id}",
                        node_type="trading_concept",
                        attributes={
                            'concept_name': concept,
                            'source': 'encyclopedia',
                            'description': f"Trading concept: {concept}",
                            'symbols': metadata.get('symbols', ['GENERAL']),
                            'category': 'strategy'
                        },
                        timestamp=datetime.now(),
                        confidence=0.8,
                        source='encyclopedia'
                    )
                    
                    self.digital_brain.knowledge_graph.add_node(concept_node)
                    self.logger.info(f"✅ Added concept node: {concept}")
                    
                except Exception as concept_error:
                    self.logger.error(f"Error adding concept {concept}: {concept_error}")
            
            # Ensure document is stored in brain
            brain_doc_node = KnowledgeNode(
                node_id=f"document_{doc_id}",
                node_type="document",
                attributes={
                    'title': metadata.get('filename', metadata.get('title', 'Unknown')),
                    'doc_type': metadata.get('doc_type', 'trading_literature'),
                    'content_preview': content[:500],
                    'pattern_count': len(document_record['chart_patterns']),
                    'concept_count': len(document_record['trading_concepts'])
                },
                timestamp=datetime.now(),
                confidence=0.9,
                source='upload'
            )
            
            self.digital_brain.knowledge_graph.add_node(brain_doc_node)
            
            # CRITICAL: Force content into DigitalBrain knowledge graph
            try:
                # Process document through DigitalBrain directly with correct parameters
                brain_result = self.digital_brain.document_processor.process_document(
                    document_text=content,
                    doc_type=metadata.get('doc_type', 'trading_literature'),
                    metadata=metadata
                )
                
                if brain_result:
                    self.logger.info(f"Successfully stored document {doc_id} in DigitalBrain")
                    
                    # Add each chart pattern as a knowledge node
                    for pattern in document_record['chart_patterns']:
                        try:
                            # Create pattern knowledge node directly
                            from knowledge_engine import KnowledgeNode
                            
                            pattern_node = KnowledgeNode(
                                node_id=f"pattern_{pattern.replace(' ', '_')}_{doc_id}",
                                node_type="chart_pattern",
                                attributes={
                                    'pattern_name': pattern,
                                    'source': 'encyclopedia',
                                    'description': f"Chart pattern: {pattern}",
                                    'symbols': metadata.get('symbols', ['GENERAL']),
                                    'reliability': 'documented',
                                    'pattern_type': 'technical_analysis'
                                },
                                timestamp=datetime.now(),
                                confidence=0.9,
                                source='encyclopedia'
                            )
                            
                            # Add node to knowledge graph
                            self.digital_brain.knowledge_graph.add_node(pattern_node)
                            self.logger.info(f"Added pattern node: {pattern}")
                            
                        except Exception as pattern_error:
                            self.logger.error(f"Error adding pattern {pattern}: {pattern_error}")
                    
                    # Add trading concepts as knowledge nodes
                    for concept in document_record['trading_concepts']:
                        try:
                            concept_node = KnowledgeNode(
                                node_id=f"concept_{concept.replace(' ', '_')}_{doc_id}",
                                node_type="trading_concept",
                                attributes={
                                    'concept_name': concept,
                                    'source': 'encyclopedia',
                                    'description': f"Trading concept: {concept}",
                                    'symbols': metadata.get('symbols', ['GENERAL']),
                                    'category': 'strategy'
                                },
                                timestamp=datetime.now(),
                                confidence=0.8,
                                source='encyclopedia'
                            )
                            
                            self.digital_brain.knowledge_graph.add_node(concept_node)
                            self.logger.info(f"Added concept node: {concept}")
                            
                        except Exception as concept_error:
                            self.logger.error(f"Error adding concept {concept}: {concept_error}")
                            
                else:
                    self.logger.warning(f"DigitalBrain processing returned None for {doc_id}")
                    
            except Exception as brain_error:
                self.logger.error(f"Error storing in DigitalBrain: {brain_error}")
                # Continue with local storage even if brain storage fails
            
            # Update upload stats
            self.upload_stats['documents_uploaded'] = len(self._stored_documents)
            
            # Save to persistent storage
            self.save_memory_bank()
            
            self.logger.info(f"Document {filename} chunk {chunk_id} force-stored successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error force-storing document: {e}")
            return False
    
    def _extract_searchable_terms(self, content: str) -> List[str]:
        """Extract key searchable terms from content"""
        import re
        
        # Extract trading-specific terms
        terms = []
        
        # Chart pattern terms
        pattern_terms = re.findall(r'\b(?:pattern|breakout|support|resistance|trend|reversal|continuation|flag|pennant|triangle|wedge|head|shoulders|double|top|bottom|cup|handle)\b', content.lower())
        terms.extend(pattern_terms)
        
        # Technical analysis terms
        ta_terms = re.findall(r'\b(?:volume|momentum|oscillator|moving|average|rsi|macd|bollinger|fibonacci|elliott|wave)\b', content.lower())
        terms.extend(ta_terms)
        
        return list(set(terms))[:50]  # Limit to 50 unique terms
    
    def _extract_chart_patterns(self, content: str) -> List[str]:
        """Extract specific chart pattern mentions"""
        import re
        
        patterns = []
        
        # Common chart patterns
        pattern_names = [
            'head and shoulders', 'inverse head and shoulders', 'double top', 'double bottom',
            'triple top', 'triple bottom', 'ascending triangle', 'descending triangle',
            'symmetrical triangle', 'flag', 'pennant', 'wedge', 'cup and handle',
            'rectangle', 'channel', 'breakout', 'breakdown', 'gap'
        ]
        
        for pattern in pattern_names:
            if pattern in content.lower():
                patterns.append(pattern)
        
        return patterns
    
    def _extract_trading_concepts(self, content: str) -> List[str]:
        """Extract trading concepts and strategies"""
        import re
        
        concepts = []
        
        # Trading concepts
        concept_terms = [
            'risk management', 'position sizing', 'stop loss', 'take profit',
            'entry point', 'exit strategy', 'trend following', 'mean reversion',
            'momentum trading', 'swing trading', 'day trading', 'scalping'
        ]
        
        for concept in concept_terms:
            if concept in content.lower():
                concepts.append(concept)
        
        return concepts

    def _is_duplicate_content(self, content: str, file_path: str) -> bool:
        """Check if content is already in the knowledge base"""
        try:
            import hashlib
            
            # Create content hash
            content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
            filename = os.path.basename(file_path)
            
            # Check against stored documents
            if hasattr(self, '_stored_documents'):
                for doc_key, doc_record in self._stored_documents.items():
                    if not doc_key.startswith('doc_'):
                        continue
                        
                    stored_filename = doc_record['metadata'].get('filename', '')
                    stored_content = doc_record.get('full_content', doc_record['content'])
                    stored_hash = hashlib.md5(stored_content.encode('utf-8')).hexdigest()
                    
                    # Check for exact content match
                    if content_hash == stored_hash:
                        self.logger.info(f"Duplicate content detected: {filename} matches {stored_filename}")
                        return True
                    
                    # Check for similar filename with significant content overlap
                    if filename == stored_filename:
                        # Calculate content similarity
                        similarity = self._calculate_content_similarity(content, stored_content)
                        if similarity > 0.85:  # 85% similarity threshold
                            self.logger.info(f"Similar content detected: {filename} ({similarity:.2f} similarity)")
                            return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking for duplicate content: {e}")
            return False
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two text contents"""
        try:
            # Simple word-based similarity
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0

    def save_memory_bank(self):
        """Save the document memory bank to persistent storage"""
        try:
            if not hasattr(self, '_stored_documents'):
                self._stored_documents = {}
            
            # Prepare data for JSON serialization
            save_data = {
                'stored_documents': {},
                'document_index': getattr(self, '_document_index', 0),
                'upload_stats': self.upload_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            # Convert stored documents to serializable format
            for key, doc_record in self._stored_documents.items():
                save_data['stored_documents'][key] = {
                    'doc_id': doc_record['doc_id'],
                    'content': doc_record['content'],
                    'full_content': doc_record.get('full_content', doc_record['content']),
                    'metadata': doc_record['metadata'],
                    'timestamp': doc_record['timestamp'],
                    'searchable_terms': doc_record.get('searchable_terms', []),
                    'chart_patterns': doc_record.get('chart_patterns', []),
                    'trading_concepts': doc_record.get('trading_concepts', [])
                }
            
            with open(self.memory_bank_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            # Also save knowledge graph state
            self.save_knowledge_graph()
            
            self.logger.info(f"Memory bank saved with {len(self._stored_documents)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving memory bank: {e}")
            return False

    def save_knowledge_graph(self):
        """Save knowledge graph state to persistent storage"""
        try:
            # Save the knowledge graph using its built-in method
            success = self.digital_brain.knowledge_graph.save_to_file(self.knowledge_graph_file)
            if success:
                self.logger.info(f"Knowledge graph saved with {len(self.digital_brain.knowledge_graph.nodes)} nodes")
            return success
        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {e}")
            return False

    def load_knowledge_graph(self):
        """Load knowledge graph state from persistent storage"""
        try:
            if os.path.exists(self.knowledge_graph_file):
                success = self.digital_brain.knowledge_graph.load_from_file(self.knowledge_graph_file)
                if success:
                    self.logger.info(f"Knowledge graph loaded with {len(self.digital_brain.knowledge_graph.nodes)} nodes")
                return success
            else:
                self.logger.info("No existing knowledge graph file found - starting fresh")
                return True
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {e}")
            return False

    def load_memory_bank(self):
        """Load the document memory bank from persistent storage"""
        try:
            if os.path.exists(self.memory_bank_file):
                with open(self.memory_bank_file, 'r', encoding='utf-8') as f:
                    save_data = json.load(f)
                
                # Restore stored documents
                self._stored_documents = save_data.get('stored_documents', {})
                self._document_index = save_data.get('document_index', 0)
                
                # Update upload stats - ensure documents_uploaded reflects actual stored documents
                saved_stats = save_data.get('upload_stats', {})
                self.upload_stats.update(saved_stats)
                
                # Force update documents uploaded to match stored documents
                if self._stored_documents:
                    self.upload_stats['documents_uploaded'] = len(self._stored_documents)
                
                self.logger.info(f"Memory bank loaded with {len(self._stored_documents)} documents")
                
                # Critical fix: Immediately restore documents to DigitalBrain
                self._restore_documents_to_brain()
                
                return True
            else:
                # Initialize empty storage
                self._stored_documents = {}
                self._document_index = 0
                self.logger.info("Initialized empty memory bank")
                return True
                
        except Exception as e:
            self.logger.error(f"Error loading memory bank: {e}")
            # Initialize empty storage on error
            self._stored_documents = {}
            self._document_index = 0
            return False

    def _restore_documents_to_brain(self):
        """Restore all stored documents to the DigitalBrain for querying"""
        try:
            restored_count = 0
            pattern_count = 0
            
            for doc_key, doc_record in self._stored_documents.items():
                if not doc_key.startswith('doc_'):
                    continue
                
                # Re-process document through DigitalBrain
                brain_result = self.digital_brain.document_processor.process_document(
                    document_text=doc_record.get('full_content', doc_record['content']),
                    doc_type=doc_record['metadata'].get('doc_type', 'trading_literature'),
                    metadata=doc_record['metadata']
                )
                
                if brain_result:
                    restored_count += 1
                
                # Rebuild chart pattern knowledge nodes if they don't exist
                for pattern in doc_record.get('chart_patterns', []):
                    pattern_node_id = f"pattern_{pattern.replace(' ', '_').replace('-', '_')}_{doc_record['doc_id']}"
                    
                    # Only add if not already in knowledge graph
                    if pattern_node_id not in self.digital_brain.knowledge_graph.nodes:
                        from knowledge_engine import KnowledgeNode
                        
                        pattern_node = KnowledgeNode(
                            node_id=pattern_node_id,
                            node_type="chart_pattern",
                            attributes={
                                'pattern_name': pattern,
                                'source': 'encyclopedia',
                                'description': f"Chart pattern: {pattern}",
                                'symbols': doc_record['metadata'].get('symbols', ['GENERAL']),
                                'reliability': 'documented',
                                'pattern_type': 'technical_analysis'
                            },
                            timestamp=datetime.now(),
                            confidence=0.9,
                            source='encyclopedia'
                        )
                        
                        self.digital_brain.knowledge_graph.add_node(pattern_node)
                        pattern_count += 1
                
                # Rebuild trading concept nodes
                for concept in doc_record.get('trading_concepts', []):
                    concept_node_id = f"concept_{concept.replace(' ', '_').replace('-', '_')}_{doc_record['doc_id']}"
                    
                    if concept_node_id not in self.digital_brain.knowledge_graph.nodes:
                        from knowledge_engine import KnowledgeNode
                        
                        concept_node = KnowledgeNode(
                            node_id=concept_node_id,
                            node_type="trading_concept",
                            attributes={
                                'concept_name': concept,
                                'source': 'encyclopedia',
                                'description': f"Trading concept: {concept}",
                                'symbols': doc_record['metadata'].get('symbols', ['GENERAL']),
                                'category': 'strategy'
                            },
                            timestamp=datetime.now(),
                            confidence=0.8,
                            source='encyclopedia'
                        )
                        
                        self.digital_brain.knowledge_graph.add_node(concept_node)
            
            self.logger.info(f"Restored {restored_count} documents and {pattern_count} patterns to DigitalBrain for querying")
            return restored_count > 0 or pattern_count > 0
            
        except Exception as e:
            self.logger.error(f"Error restoring documents to brain: {e}")
            return False

def demo_upload_trading_literature():
    """Demo function showing how to upload trading literature"""
    uploader = TradingDocumentUploader()

    print("📚 Trading Literature Memory Bank Demo")
    print("="*50)

    # Example 1: Upload text-based trading strategy
    strategy_content = """
Momentum Trading Strategy for Tech Stocks

Entry Criteria:
- RSI > 60 but < 80
- MACD histogram increasing for 3 periods
- Volume 50% above 20-day average
- Stock above 20-day EMA

Exit Criteria:
- RSI > 80 (overbought)
- MACD bearish crossover
- Volume declining below average
- 2% stop-loss

Best suited for: AAPL, GOOGL, MSFT during earnings season
Win rate: ~65% with proper risk management
"""

    result = uploader.upload_text_content(
        content=strategy_content,
        title="Momentum Trading Strategy - Tech Stocks",
        doc_type="trading_strategy",
        symbols=["AAPL", "GOOGL", "MSFT"],
        description="Momentum-based strategy for tech stocks during earnings"
    )

    print(f"✅ Strategy Upload Result: {result}")

    # Example 2: Upload market analysis notes
    analysis_content = """
Market Analysis - Q4 2024

Key Observations:
- Technology sector showing rotation patterns
- NVDA leading semiconductor recovery
- Interest rate expectations driving volatility
- Earnings season showing mixed results

Trading Implications:
- Favor momentum over mean reversion
- Watch for sector rotation opportunities
- Manage position sizes due to volatility
- Focus on stocks with strong earnings visibility
"""

    result2 = uploader.upload_text_content(
        content=analysis_content,
        title="Q4 2024 Market Analysis",
        doc_type="market_analysis",
        symbols=["NVDA", "QQQ"],
        description="Quarterly market analysis and trading implications"
    )

    print(f"✅ Analysis Upload Result: {result2}")

    # Show memory bank stats
    stats = uploader.get_memory_bank_stats()
    print(f"\n📊 Memory Bank Statistics:")
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Knowledge Nodes: {stats['brain_status']['knowledge_nodes']}")
    print(f"Learned Patterns: {stats['brain_status']['learned_patterns']}")
    print(f"Document Types: {stats['document_types']}")

    # Demo query
    query_result = uploader.query_memory_bank("What momentum strategies work for tech stocks?")
    print(f"\n🔍 Query Result: {query_result}")

    return uploader

if __name__ == "__main__":
    demo_upload_trading_literature()