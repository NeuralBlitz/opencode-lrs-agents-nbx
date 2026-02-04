"""
Example 4: Intent Comparison
Demonstrates measuring intent similarity and differences.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neuralblitz import IntentVector
from neuralblitz.advanced import compare_intents


def main():
    print("=" * 60)
    print("NeuralBlitz V50 Minimal - Example 4: Intent Comparison")
    print("=" * 60)

    # Define example intents
    print("\n1. Defining test intents...")

    creative_intent = IntentVector(
        phi1_dominance=0.3,
        phi2_harmony=0.4,
        phi3_creation=0.9,
        phi4_preservation=0.2,
        phi5_transformation=0.8,
        phi6_knowledge=0.3,
        phi7_connection=0.5,
        metadata={"name": "Creative"},
    )

    analytical_intent = IntentVector(
        phi1_dominance=0.7,
        phi2_harmony=0.3,
        phi3_creation=0.2,
        phi4_preservation=0.6,
        phi5_transformation=0.3,
        phi6_knowledge=0.9,
        phi7_connection=0.2,
        metadata={"name": "Analytical"},
    )

    social_intent = IntentVector(
        phi1_dominance=0.2,
        phi2_harmony=0.9,
        phi3_creation=0.5,
        phi4_preservation=0.4,
        phi5_transformation=0.3,
        phi6_knowledge=0.4,
        phi7_connection=0.9,
        metadata={"name": "Social"},
    )

    balanced_intent = IntentVector(
        phi1_dominance=0.5,
        phi2_harmony=0.5,
        phi3_creation=0.5,
        phi4_preservation=0.5,
        phi5_transformation=0.5,
        phi6_knowledge=0.5,
        phi7_connection=0.5,
        metadata={"name": "Balanced"},
    )

    intense_creative = IntentVector(
        phi1_dominance=0.2,
        phi2_harmony=0.3,
        phi3_creation=0.95,
        phi4_preservation=0.1,
        phi5_transformation=0.9,
        phi6_knowledge=0.2,
        phi7_connection=0.4,
        metadata={"name": "Intense Creative"},
    )

    intents = [
        creative_intent,
        analytical_intent,
        social_intent,
        balanced_intent,
        intense_creative,
    ]

    for intent in intents:
        print(
            f"   ✓ {intent.metadata['name']}: [{', '.join(f'{x:.1f}' for x in intent.to_vector())}]"
        )

    # Compare all pairs
    print("\n2. Pairwise similarity matrix:")
    print("-" * 60)
    print(f"{'Intent':<20}", end="")
    for i in intents:
        print(f"{i.metadata['name'][:8]:<10}", end="")
    print()
    print("-" * 60)

    for i, intent1 in enumerate(intents):
        print(f"{intent1.metadata['name']:<20}", end="")
        for j, intent2 in enumerate(intents):
            comparison = compare_intents(intent1, intent2)
            similarity = comparison["similarity_score"]
            print(f"{similarity:<10.2f}", end="")
        print()

    # Detailed comparisons
    print("\n3. Detailed comparisons:")

    print("\n   Creative vs Analytical (should be different):")
    comp = compare_intents(creative_intent, analytical_intent)
    print(f"   ✓ Cosine similarity: {comp['cosine_similarity']:.3f}")
    print(f"   ✓ Euclidean distance: {comp['euclidean_distance']:.3f}")
    print(f"   ✓ Overall similarity: {comp['similarity_score']:.1%}")

    print("\n   Creative vs Intense Creative (should be similar):")
    comp = compare_intents(creative_intent, intense_creative)
    print(f"   ✓ Cosine similarity: {comp['cosine_similarity']:.3f}")
    print(f"   ✓ Euclidean distance: {comp['euclidean_distance']:.3f}")
    print(f"   ✓ Overall similarity: {comp['similarity_score']:.1%}")

    print("\n   Balanced vs Any (should be moderate):")
    comp = compare_intents(balanced_intent, social_intent)
    print(f"   ✓ Cosine similarity: {comp['cosine_similarity']:.3f}")
    print(f"   ✓ Euclidean distance: {comp['euclidean_distance']:.3f}")
    print(f"   ✓ Overall similarity: {comp['similarity_score']:.1%}")

    # Find most similar
    print("\n4. Finding most similar intent pairs:")
    similarities = []
    for i, intent1 in enumerate(intents):
        for j, intent2 in enumerate(intents):
            if i < j:
                comp = compare_intents(intent1, intent2)
                similarities.append(
                    (
                        comp["similarity_score"],
                        intent1.metadata["name"],
                        intent2.metadata["name"],
                    )
                )

    similarities.sort(reverse=True)
    print("   Most similar:")
    for sim, name1, name2 in similarities[:3]:
        print(f"   ✓ {name1} ↔ {name2}: {sim:.1%}")

    print("\n   Most different:")
    for sim, name1, name2 in similarities[-3:]:
        print(f"   ✓ {name1} ↔ {name2}: {sim:.1%}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
