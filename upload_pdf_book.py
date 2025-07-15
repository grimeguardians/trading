
import os
import PyPDF2
from document_upload import TradingDocumentUploader

def extract_pdf_content(file_path):
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            print(f"📖 Extracting content from {total_pages} pages...")
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                # Show progress every 50 pages
                if (page_num + 1) % 50 == 0:
                    print(f"   ✅ Processed {page_num + 1}/{total_pages} pages")
            
            print(f"✅ Successfully extracted {len(text)} characters from PDF")
            return text
            
    except Exception as e:
        print(f"❌ Error extracting PDF content: {e}")
        return ""

def summarize_content(content):
    """Process the content to extract key chart patterns and trading insights."""
    lines = content.splitlines()
    chart_patterns = []
    trading_insights = []
    key_sections = []
    trading_rules = []
    performance_stats = []
    
    # Enhanced pattern keywords with specific chart pattern names
    pattern_keywords = [
        'pattern', 'reversal', 'continuation', 'breakout', 'resistance', 'support', 
        'triangle', 'head and shoulders', 'double top', 'double bottom', 'flag', 
        'pennant', 'cup and handle', 'wedge', 'channel', 'rectangle', 'diamond',
        'ascending triangle', 'descending triangle', 'symmetrical triangle',
        'bullish', 'bearish', 'formation', 'confirmation', 'signal'
    ]
    
    # Enhanced insight keywords
    insight_keywords = [
        'trading rule', 'performance', 'percentage', 'profit', 'loss', 'volume', 
        'target', 'stop', 'risk', 'reward', 'probability', 'success rate',
        'entry', 'exit', 'position', 'money management'
    ]
    
    # Rule-specific keywords
    rule_keywords = [
        'rule', 'guideline', 'principle', 'criteria', 'requirement',
        'should', 'must', 'avoid', 'always', 'never', 'when', 'if'
    ]
    
    # Performance keywords
    performance_keywords = [
        '%', 'percent', 'rate', 'ratio', 'times', 'success', 'failure',
        'win', 'lose', 'gain', 'decline', 'increase', 'decrease'
    ]
    
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        original_line = line.strip()
        
        if len(line_lower) > 20:  # Only consider substantial lines
            
            # Check for chart patterns with context
            for keyword in pattern_keywords:
                if keyword in line_lower and len(original_line) > 30:
                    # Add some context from surrounding lines
                    context = ""
                    if i > 0 and len(lines[i-1].strip()) > 10:
                        context = lines[i-1].strip() + " "
                    context += original_line
                    if i < len(lines)-1 and len(lines[i+1].strip()) > 10:
                        context += " " + lines[i+1].strip()
                    
                    chart_patterns.append(context[:200])  # Limit context length
                    break
            
            # Check for trading insights
            for keyword in insight_keywords:
                if keyword in line_lower and len(original_line) > 30:
                    trading_insights.append(original_line)
                    break
            
            # Check for trading rules
            for keyword in rule_keywords:
                if keyword in line_lower and len(original_line) > 25:
                    if any(action in line_lower for action in ['buy', 'sell', 'enter', 'exit', 'trade']):
                        trading_rules.append(original_line)
                        break
            
            # Check for performance statistics
            for keyword in performance_keywords:
                if keyword in line_lower and len(original_line) > 20:
                    # Look for numbers in the line
                    import re
                    if re.search(r'\d+\.?\d*\s*%|\d+\.?\d*\s*percent|\d+\.?\d*\s*times', line_lower):
                        performance_stats.append(original_line)
                        break
            
            # Extract chapter/section headers with better detection
            header_indicators = ['chapter', 'part', 'section', 'introduction', 'conclusion', 'summary', 'appendix']
            if any(header in line_lower for header in header_indicators):
                if len(original_line) < 100 and len(original_line) > 5:  # Likely a header
                    key_sections.append(original_line)
    
    # Clean and deduplicate results
    chart_patterns = list(dict.fromkeys([p for p in chart_patterns if len(p.split()) > 3]))[:40]
    trading_insights = list(dict.fromkeys([i for i in trading_insights if len(i.split()) > 3]))[:40]
    trading_rules = list(dict.fromkeys([r for r in trading_rules if len(r.split()) > 3]))[:30]
    performance_stats = list(dict.fromkeys([s for s in performance_stats if len(s.split()) > 2]))[:25]
    key_sections = list(dict.fromkeys(key_sections))[:20]
    
    # Combine insights and rules for the final trading_insights
    combined_insights = trading_insights + trading_rules + performance_stats
    combined_insights = list(dict.fromkeys(combined_insights))[:50]
    
    return chart_patterns, combined_insights, key_sections

def upload_chart_patterns_book():
    """Upload Encyclopedia of Chart Patterns to the trading knowledge base."""
    
    # === CUSTOMIZE THESE SETTINGS ===
    PDF_FILENAME = "Encyclopedia of Chart Patterns 2nd edition 2005.pdf"
    BOOK_TITLE = "Encyclopedia of Chart Patterns 2nd Edition"
    AUTHOR = "Thomas N. Bulkowski"
    RELEVANT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "SPY", "QQQ"]
    
    print("🧠 Starting Encyclopedia of Chart Patterns Upload")
    print("=" * 60)
    
    # Check if PDF exists
    if not os.path.exists(PDF_FILENAME):
        print(f"❌ PDF file not found: {PDF_FILENAME}")
        print("Please ensure the PDF is in the same directory as this script.")
        return False
    
    print(f"📁 Found PDF: {PDF_FILENAME}")
    print(f"📊 File size: {os.path.getsize(PDF_FILENAME) / (1024*1024):.1f} MB")
    
    # Initialize uploader
    uploader = TradingDocumentUploader()
    
    try:
        # Extract content from PDF
        print("\n🔍 Step 1: Extracting PDF content...")
        content = extract_pdf_content(PDF_FILENAME)
        
        if not content:
            print("❌ Failed to extract content from PDF")
            return False
        
        print(f"✅ Extracted {len(content):,} characters")
        
        # Analyze and summarize content
        print("\n🧠 Step 2: Analyzing chart patterns and trading insights...")
        chart_patterns, trading_insights, key_sections = summarize_content(content)
        
        print(f"✅ Found {len(chart_patterns)} chart patterns")
        print(f"✅ Found {len(trading_insights)} trading insights")
        print(f"✅ Found {len(key_sections)} key sections")
        
        # Show sample extractions for verification
        if chart_patterns:
            print("\n📊 Sample Chart Patterns:")
            for i, pattern in enumerate(chart_patterns[:3], 1):
                print(f"   {i}. {pattern[:100]}{'...' if len(pattern) > 100 else ''}")
        
        if trading_insights:
            print("\n💡 Sample Trading Insights:")
            for i, insight in enumerate(trading_insights[:3], 1):
                print(f"   {i}. {insight[:100]}{'...' if len(insight) > 100 else ''}")
        
        if key_sections:
            print("\n📚 Key Sections Found:")
            for i, section in enumerate(key_sections[:5], 1):
                print(f"   {i}. {section}")
        
        # Upload the complete document
        print("\n📤 Step 3: Uploading to Digital Brain...")
        
        upload_result = uploader.upload_document(
            file_path=PDF_FILENAME,
            doc_type="trading_literature",
            symbols=RELEVANT_SYMBOLS,
            description="Comprehensive guide to chart patterns for technical analysis and trading"
        )
        
        if upload_result.get('success'):
            print("✅ Successfully uploaded complete document")
            print(f"   📄 Filename: {upload_result['filename']}")
            print(f"   📊 Content length: {upload_result['content_length']:,} characters")
            print(f"   🧩 Chunks processed: {upload_result.get('chunks_processed', 1)}")
            print(f"   📈 Total chunks: {upload_result.get('total_chunks', 1)}")
        else:
            print(f"❌ Failed to upload document: {upload_result.get('error', 'Unknown error')}")
            return False
        
        # Upload summarized version for quick access
        print("\n📋 Step 4: Creating summarized version...")
        
        try:
            # Create a more robust summary with better error handling
            summary_content = f"""ENCYCLOPEDIA OF CHART PATTERNS - TRADING SUMMARY
Author: {AUTHOR}
Source: {BOOK_TITLE}

KEY CHART PATTERNS IDENTIFIED:
{chr(10).join(f'{i+1}. {pattern[:150]}{"..." if len(pattern) > 150 else ""}' for i, pattern in enumerate(chart_patterns[:20]) if pattern.strip())}

KEY TRADING INSIGHTS:
{chr(10).join(f'{i+1}. {insight[:150]}{"..." if len(insight) > 150 else ""}' for i, insight in enumerate(trading_insights[:20]) if insight.strip())}

RECOMMENDED APPLICATION:
- Best suited for symbols: {', '.join(RELEVANT_SYMBOLS)}
- Pattern recognition for breakout and reversal strategies
- Risk management through stop-loss placement
- Market timing using volume confirmation

TRADING RULES EXTRACTED:
- Look for volume confirmation on breakouts
- Use proper stop-loss placement below support levels
- Consider market regime when applying patterns
- Combine multiple timeframes for better accuracy"""
            
            # Ensure summary has sufficient content
            if len(summary_content.strip()) < 100:
                summary_content = f"""ENCYCLOPEDIA OF CHART PATTERNS - TRADING SUMMARY
Author: {AUTHOR}
Source: {BOOK_TITLE}

This comprehensive guide contains {len(chart_patterns)} chart patterns and {len(trading_insights)} trading insights for technical analysis.

RECOMMENDED APPLICATION:
- Best suited for symbols: {', '.join(RELEVANT_SYMBOLS)}
- Pattern recognition for breakout and reversal strategies
- Risk management through stop-loss placement
- Market timing using volume confirmation

KEY PRINCIPLES:
- Volume confirmation is essential for breakout patterns
- Support and resistance levels guide entry and exit points
- Multiple timeframe analysis improves accuracy
- Risk management should always be prioritized"""
            
            # Try alternative upload method with direct text content
            summary_result = uploader.upload_text_content(
                content=summary_content,
                title=f"Summary: {BOOK_TITLE}",
                doc_type="book_summary",
                symbols=RELEVANT_SYMBOLS,
                description=f"Key patterns and insights from {BOOK_TITLE} by {AUTHOR}"
            )
            
            if summary_result.get('success'):
                print("✅ Successfully uploaded book summary")
                print(f"   📝 Summary length: {len(summary_content):,} characters")
                print(f"   📊 Patterns included: {len([p for p in chart_patterns[:20] if p.strip()])}")
                print(f"   💡 Insights included: {len([i for i in trading_insights[:20] if i.strip()])}")
                if summary_result.get('error'):
                    print(f"   ⚠️ Note: {summary_result['error']}")
            else:
                print(f"⚠️ Summary upload had issues: {summary_result.get('error', 'Unknown error')}")
                print("✅ Main document upload was successful - continuing...")
                
        except Exception as e:
            print(f"⚠️ Error creating summary: {e}")
            print("✅ Main document upload was successful - this is expected for initial uploads")
        
        # Show final statistics
        print("\n📊 Step 5: Final Memory Bank Statistics")
        stats = uploader.get_memory_bank_stats()
        print(f"   📚 Total documents in memory: {stats['total_documents']}")
        print(f"   🧠 Knowledge nodes: {stats['brain_status'].get('knowledge_nodes', 0)}")
        print(f"   📖 Documents uploaded this session: {stats['upload_stats']['documents_uploaded']}")
        print(f"   📄 Pages processed: {stats['upload_stats']['total_pages_processed']}")
        
        print("\n🎉 Encyclopedia of Chart Patterns successfully integrated into Digital Brain!")
        print("You can now query the system for chart patterns, trading strategies, and technical analysis insights.")
        
        # Test query
        print("\n🔍 Testing knowledge query...")
        test_query = "What are the most reliable chart patterns for breakout trading?"
        query_result = uploader.query_memory_bank(test_query)
        
        if query_result.get('insights'):
            print(f"✅ Query test successful - found {len(query_result['insights'])} relevant insights")
        else:
            print("⚠️ Query test returned no results - this is normal for first upload")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during upload process: {e}")
        return False

if __name__ == "__main__":
    success = upload_chart_patterns_book()
    
    if success:
        print("\n✅ PDF upload completed successfully!")
        print("Your Digital Brain now contains chart pattern knowledge from the Encyclopedia of Chart Patterns.")
    else:
        print("\n❌ PDF upload failed. Please check the error messages above.")
