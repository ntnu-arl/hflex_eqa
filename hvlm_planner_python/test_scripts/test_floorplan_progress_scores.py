#!/usr/bin/env python3
"""Quick check for EQA planner semantic floorplan progress scores."""

import argparse

import matplotlib.pyplot as plt
import networkx as nx

from hvlm_planner_python.eqa_planner.eqa_planner import (
    EQAPlanner,
    EQAPlannerConfig,
    FloorPlanGraph,
    unique_room_names,
)
from hvlm_planner_python.eqa_planner.eqa_planner_output import (
    EQAPlannerOutput,
    OutputMode,
)


def make_example_floorplan() -> FloorPlanGraph:
    """Create a small floorplan with duplicate semantic room types."""
    return FloorPlanGraph(
        nodes=[
            "office 0",
            "bathroom 0",
            "study 0",
            "bedroom 0",
            "tie: living room & hall/stairwell 0",
            "bathroom 1",
            "office 1",
            "kitchen 0",
            "bedroom 1",
            "living room 0",
            "bedroom 2",
            "undefined 0",
            "tie: bedroom & living room 0",
            "bathroom 2",
            "bedroom 3",
            "living room 1",
            "undefined 1",
            "kitchen 1",
            "hallway 0",
            "bedroom 4",
        ],
        edges=[
            ["bathroom 0", "bathroom 2"],
            ["bathroom 0", "living room 1"],
            ["study 0", "bedroom 0"],
            ["study 0", "tie: living room & hall/stairwell 0"],
            ["study 0", "bedroom 4"],
            ["bedroom 0", "tie: living room & hall/stairwell 0"],
            ["bedroom 0", "bedroom 4"],
            ["tie: living room & hall/stairwell 0", "bathroom 1"],
            ["tie: living room & hall/stairwell 0", "office 1"],
            ["office 1", "kitchen 0"],
            ["kitchen 0", "bedroom 1"],
            ["kitchen 0", "tie: bedroom & living room 0"],
            ["bedroom 1", "living room 0"],
            ["bedroom 1", "tie: bedroom & living room 0"],
            ["living room 0", "bedroom 2"],
            ["living room 0", "tie: bedroom & living room 0"],
            ["bedroom 2", "tie: bedroom & living room 0"],
            ["undefined 0", "bathroom 2"],
            ["undefined 0", "living room 1"],
            ["bedroom 3", "hallway 0"],
            ["living room 1", "undefined 1"],
            ["kitchen 1", "hallway 0"],
            ["kitchen 1", "bedroom 4"],
        ],
    )


def visualize_floor_plan_graph(graph: FloorPlanGraph) -> None:
    G = nx.Graph()

    G.add_nodes_from(graph.nodes)
    G.add_edges_from(graph.edges)

    plt.figure(figsize=(8, 6))

    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=1800,
        font_size=10,
        font_weight="bold",
        edgecolors="black",
    )

    plt.title("Floor Plan Graph")
    plt.axis("off")
    plt.show()


def make_planner_for_scoring(floorplan_graph: FloorPlanGraph) -> EQAPlanner:
    """Build a minimal planner object without creating VLM/CLIP components."""
    planner = EQAPlanner.__new__(EQAPlanner)
    planner._config = EQAPlannerConfig(floorplan_graph=floorplan_graph)
    planner._unique_room_names = unique_room_names(floorplan_graph.nodes)[0]
    return planner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test semantic room-type floorplan progress scoring."
    )
    parser.add_argument(
        "--current-room",
        default="kitchen",
        help="Current semantic room label.",
    )
    parser.add_argument(
        "--target-room",
        default="bedroom",
        help="Desired semantic room label.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Whether to visualize the floor plan graph.",
    )
    args = parser.parse_args()

    floorplan = make_example_floorplan()
    planner = make_planner_for_scoring(floorplan)

    output = EQAPlannerOutput(
        answer="",
        answered=False,
        mode=OutputMode.FIND_ROOM,
        valid=True,
        target_room_label=args.target_room,
    )
    planner._add_find_room_output_fields(output, args.current_room)

    print("Original floorplan nodes:")
    for node in floorplan.nodes:
        print(f"  {node}")

    print("\nOriginal floorplan edges:")
    for a, b in floorplan.edges:
        print(f"  {a} -- {b}")

    print(
        f"\nSoftmax next-label scores from current room type '{args.current_room}' "
        f"toward '{args.target_room}':"
    )
    for label, score in zip(
        output.floorplan_room_labels, output.floorplan_progress_scores, strict=True
    ):
        print(f"  {label}: {score:.3f}")

    if args.visualize:
        visualize_floor_plan_graph(floorplan)


if __name__ == "__main__":
    main()
