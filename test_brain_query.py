
#!/usr/bin/env python3
"""
Test script for querying the Digital Brain knowledge base
"""

from document_upload import TradingDocumentUploader
import json
import os
import logging

def test_knowledge_queries():
    """Test various knowledge queries on the uploaded Digital Brain data"""
    print("🧠 Testing Digital Brain Knowledge Queries")
    print("=" * 60)

    # Set up logging to see what's happening
    logging.basicConfig(level=logging.INFO)

    # Initialize the uploader (gives us access to Digital Brain)
    uploader = TradingDocumentUploader()

    # First, check the Digital Brain status directly
    brain_status = uploader.digital_brain.get_brain_status()
    print(f"🧠 Digital Brain Status:")
    print(f"   Knowledge nodes: {brain_status['knowledge_nodes']}")
    print(f"   Knowledge edges: {brain_status['knowledge_edges']}")
    print(f"   Learned patterns: {brain_status['learned_patterns']}")
    print(f"   Processed documents: {brain_status['processed_documents']}")
    print(f"   Memory health: {brain_status['memory_health']}")

    # Check knowledge graph nodes directly
    kg = uploader.digital_brain.knowledge_graph
    print(f"\n🔍 Knowledge Graph Inspection:")
    print(f"   Total nodes: {len(kg.nodes)}")
    print(f"   Total edges: {len(kg.edges)}")
    
    # Count node types
    node_types = {}
    pattern_nodes = []
    concept_nodes = []
    
    for node in kg.nodes.values():
        node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        if node.node_type == 'chart_pattern':
            pattern_nodes.append(node)
        elif node.node_type == 'trading_concept':
            concept_nodes.append(node)
    
    print(f"   Node types: {node_types}")
    
    # Show sample pattern nodes
    if pattern_nodes:
        print(f"\n📊 Sample Chart Pattern Nodes:")
        for i, node in enumerate(pattern_nodes[:5]):
            pattern_name = node.attributes.get('pattern_name', 'Unknown')
            print(f"      {i+1}. {pattern_name} (confidence: {node.confidence})")
    
    # Show sample concept nodes
    if concept_nodes:
        print(f"\n💡 Sample Trading Concept Nodes:")
        for i, node in enumerate(concept_nodes[:5]):
            concept_name = node.attributes.get('concept_name', 'Unknown')
            print(f"      {i+1}. {concept_name} (confidence: {node.confidence})")

    # Check memory bank file
    if os.path.exists(uploader.memory_bank_file):
        print(f"\n📁 Memory Bank File: {uploader.memory_bank_file}")
        with open(uploader.memory_bank_file, 'r') as f:
            try:
                data = json.load(f)
                stored_docs = data.get('stored_documents', {})
                print(f"📊 Stored documents: {len(stored_docs)}")
            except json.JSONDecodeError:
                print("❌ Memory bank file is corrupted")
    else:
        print(f"❌ Memory bank file not found: {uploader.memory_bank_file}")

    # Test queries about chart patterns and trading
    test_queries = [
        "What are reliable breakout patterns?",
        "How do head and shoulders patterns work?",
        "What are triangle patterns?",
        "Tell me about double top patterns",
        "What reversal patterns should I know?",
        "How do flag patterns work?",
        "What are continuation patterns?",
        "Tell me about support and resistance"
    ]
    
    print(f"\n🔍 Testing Queries on Digital Brain Knowledge...")
    print("=" * 60)
    
    successful_queries = 0
    failed_queries = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 40)
        
        try:
            # Query the memory bank
            result = uploader.query_memory_bank(query)
            
            if 'error' in result:
                print(f"❌ Query failed: {result['error']}")
                failed_queries += 1
                continue
            
            # Display results
            print(f"✅ Query successful!")
            print(f"📊 Results:")
            print(f"   - Knowledge matches: {result.get('knowledge_matches', 0)}")
            print(f"   - Patterns found: {result.get('patterns_found', 0)}")
            print(f"   - Confidence: {result.get('confidence', 0):.2f}")
            print(f"   - Total documents: {result.get('total_documents_in_memory', 0)}")
            print(f"   - Knowledge nodes: {result.get('total_knowledge_nodes', 0)}")
            
            # Show insights
            insights = result.get('insights', [])
            if insights:
                print(f"🔍 Insights:")
                for insight in insights[:3]:  # Show top 3 insights
                    print(f"   • {insight}")
            
            # Show knowledge graph matches
            kg_matches = result.get('knowledge_graph_matches', [])
            if kg_matches:
                print(f"🧠 Knowledge Graph Matches:")
                for match in kg_matches[:3]:  # Show top 3 matches
                    print(f"   • {match['name']} ({match['type']}) - score: {match['score']}")
            
            # Show relevant documents
            relevant_docs = result.get('relevant_documents', [])
            if relevant_docs:
                print(f"📚 Relevant Documents:")
                for doc in relevant_docs[:2]:  # Show top 2 documents
                    title = doc.get('title', 'Unknown Document')
                    print(f"   • {title}")
            
            successful_queries += 1
                
        except Exception as e:
            print(f"❌ Error during query: {e}")
            failed_queries += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"✅ Successful queries: {successful_queries}")
    print(f"❌ Failed queries: {failed_queries}")
    print(f"📈 Success rate: {successful_queries / len(test_queries) * 100:.1f}%")
    
    # Final memory bank statistics
    try:
        stats = uploader.get_memory_bank_stats()
        print(f"\n📚 Final Memory Bank Statistics:")
        print(f"   Total Documents: {stats['total_documents']}")
        print(f"   Document Types: {stats.get('document_types', {})}")
        
        brain_status = stats.get('brain_status', {})
        print(f"   Knowledge Nodes: {brain_status.get('knowledge_nodes', 0)}")
        print(f"   Knowledge Edges: {brain_status.get('knowledge_edges', 0)}")
        print(f"   Memory Health: {brain_status.get('memory_health', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Error getting final stats: {e}")

    # Test direct brain query as fallback
    print(f"\n🧠 Testing Direct Brain Query...")
    try:
        direct_result = uploader.digital_brain.query_brain(
            "What are chart patterns?", 
            {'query_type': 'pattern'}
        )
        print(f"✅ Direct brain query successful!")
        print(f"   Confidence: {direct_result.get('confidence', 0):.2f}")
        print(f"   Knowledge matches: {direct_result.get('knowledge_matches', 0)}")
        print(f"   Insights: {len(direct_result.get('insights', []))}")
        
    except Exception as e:
        print(f"❌ Direct brain query failed: {e}")

if __name__ == "__main__":
    test_knowledge_queries()
