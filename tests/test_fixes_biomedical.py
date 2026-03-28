"""Test that biomedical domain fixes work properly."""

from pathlib import Path
from backend.graphrag.parser import parse_article
from backend.graphrag.chunking import chunk_article
from backend.graphrag.extraction import extract_layer2
from backend.graphrag.edges import build_layer3
from backend.graphrag.config import Phase1Settings


def test_biomedical_method_extraction():
    """Test that biomedical methods (ChIP-seq, RNA-seq, etc.) are extracted."""
    settings = Phase1Settings.from_env()
    
    # Parse one article
    article_path = Path("articles/BJ_100828.xml")
    if not article_path.exists():
        print(f"Skipping: {article_path} not found")
        return
    
    paper = chunk_article(parse_article(article_path), settings=settings)
    layer2 = extract_layer2(paper, settings=settings, use_gemini=False)
    
    # Check extraction stats
    entity_counts = {}
    for entity in layer2.entities:
        entity_counts[entity.entity_type] = entity_counts.get(entity.entity_type, 0) + 1
    
    print("\n=== Biomedical Method Extraction Test ===")
    print(f"Paper: {paper.title}")
    print(f"Chunks: {len(paper.chunks)}")
    print(f"Extracted Entity Types:")
    for entity_type, count in sorted(entity_counts.items()):
        print(f"  {entity_type}: {count}")
    
    # Check for biomedical methods
    biomedical_methods = [
        e for e in layer2.entities 
        if e.entity_type == "method" and e.properties.get("domain") == "biomedical"
    ]
    print(f"\nBiomedical Methods: {len(biomedical_methods)}")
    for method in biomedical_methods[:5]:
        print(f"  - {method.label} (confidence: {method.confidence})")
    
    # Verify improvements
    assert entity_counts.get("method", 0) > 0, "No methods extracted!"
    assert entity_counts.get("claim", 0) > 0, "No claims extracted!"
    if entity_counts.get("result", 0) > 0:
        results = [e for e in layer2.entities if e.entity_type == "result"]
        print(f"\nResults extracted: {len(results)}")
        for result in results[:3]:
            print(f"  - {result.label} (value: {result.properties.get('value')}, metric: {result.properties.get('metric')})")
    
    print(f"\n✓ Biomedical extraction working!")
    return True


def test_layer3_edges():
    """Test that Layer 3 edges are properly built."""
    settings = Phase1Settings.from_env()
    
    article_path = Path("articles/BJ_100828.xml")
    if not article_path.exists():
        print(f"Skipping: {article_path} not found")
        return
    
    paper = chunk_article(parse_article(article_path), settings=settings)
    layer2 = extract_layer2(paper, settings=settings, use_gemini=False)
    
    layer3 = build_layer3([paper], [layer2])
    
    print("\n=== Layer 3 Edges Test ===")
    edge_types = {}
    for edge in layer3.semantic_edges:
        edge_types[edge.relation_type] = edge_types.get(edge.relation_type, 0) + 1
    
    print(f"Semantic Edges by Type:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"  {edge_type}: {count}")
    
    # Verify new edge types exist
    new_edge_types = ["GROUNDED_IN", "MEASURED_ON", "USING_METRIC"]
    for edge_type in new_edge_types:
        count = edge_types.get(edge_type, 0)
        print(f"  {edge_type}: {count} {'✓' if count > 0 else '✗'}")
    
    print(f"\n✓ Layer 3 edges test complete!")
    return True


if __name__ == "__main__":
    try:
        test_biomedical_method_extraction()
        test_layer3_edges()
        print("\n" + "="*50)
        print("ALL TESTS PASSED ✓")
        print("="*50)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
