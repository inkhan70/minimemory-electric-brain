from minimemory import FederatedBrain


with FederatedBrain("knowledge") as brain:
    brain.remember(
        "Photosynthesis allows plants to convert light energy into chemical energy.",
        domain="science",
        source="science-book",
        confidence=0.95,
    )
    brain.remember(
        "Forests are ecosystems containing trees, plants, animals and other organisms.",
        domain="nature",
        source="nature-book",
        confidence=0.94,
    )
    brain.remember(
        "Forests store carbon in vegetation and soils and are relevant to climate systems.",
        domain="environment",
        source="environment-book",
        confidence=0.93,
    )

    result = brain.ask(
        "How do plants use sunlight in forests?",
        max_domains=2,
        top_k=6,
    )
    print("ANSWER:")
    print(result["answer"])
    print("\nDOMAINS:", result["domains"])
    print("PROVENANCE:")
    for item in result["provenance"]:
        print(item)
