
#!/usr/bin/env python3
"""
Bulk PDF Upload Script for Digital Brain Knowledge Base
Automatically processes all PDFs in directory and adds them to the brain
"""

from document_upload import TradingDocumentUploader
import os
import logging

def main():
    """Main function for bulk PDF processing"""
    print("📚 Digital Brain - Bulk PDF Processing")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize uploader
    uploader = TradingDocumentUploader()
    
    # Get current brain status
    stats = uploader.get_memory_bank_stats()
    print(f"🧠 Current Brain Status:")
    print(f"   Documents: {stats['total_documents']}")
    print(f"   Knowledge Nodes: {stats['brain_status']['knowledge_nodes']}")
    print(f"   Memory Health: {stats['brain_status']['memory_health']}")
    
    # First, let's see what files are actually in the directory
    import glob
    import os
    
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Check for PDF files with different approaches
    pdf_files = glob.glob("*.pdf")
    all_files = os.listdir(".")
    pdf_files_alt = [f for f in all_files if f.lower().endswith('.pdf')]
    
    print(f"📄 Files found with glob pattern '*.pdf': {len(pdf_files)}")
    if pdf_files:
        for pdf in pdf_files:
            print(f"   • {pdf}")
    
    print(f"📄 Files found with .pdf extension: {len(pdf_files_alt)}")
    if pdf_files_alt:
        for pdf in pdf_files_alt:
            print(f"   • {pdf}")
    
    if not pdf_files_alt:
        print("❌ No PDF files found at all!")
        return
    
    # Process all PDFs in current directory
    result = uploader.scan_and_upload_pdfs(
        directory=".",
        file_pattern="*.pdf"
    )
    
    if result.get('error'):
        print(f"\n❌ {result['error']}")
        return
    
    # Show detailed results
    print(f"\n📈 Processing Results:")
    print(f"   Files Found: {result['files_processed']}")
    print(f"   Successfully Uploaded: {result['successful_uploads']}")
    print(f"   Duplicates Skipped: {result['duplicates_skipped']}")
    print(f"   Errors: {result['errors']}")
    
    # Show final brain state
    final_stats = result['final_stats']
    print(f"\n🧠 Final Brain State:")
    print(f"   Total Documents: {final_stats['total_documents']}")
    print(f"   Knowledge Nodes: {final_stats['brain_status']['knowledge_nodes']}")
    print(f"   Learned Patterns: {final_stats['brain_status']['learned_patterns']}")
    
    # Test a few queries to verify knowledge
    print(f"\n🔍 Testing Knowledge Retrieval:")
    
    test_queries = [
        "What are breakout patterns?",
        "How do triangle patterns work?",
        "Tell me about support and resistance"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = uploader.query_memory_bank(query)
        
        if result.get('error'):
            print(f"   ❌ Error: {result['error']}")
        else:
            confidence = result.get('confidence', 0)
            matches = result.get('knowledge_matches', 0)
            print(f"   ✅ Confidence: {confidence:.2f}, Matches: {matches}")
            
            insights = result.get('insights', [])
            if insights:
                print(f"   💡 Key insight: {insights[0][:100]}...")

if __name__ == "__main__":
    main()
