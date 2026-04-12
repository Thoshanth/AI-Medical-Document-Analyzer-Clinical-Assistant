import json
import networkx as nx
from pathlib import Path
from backend.logger import get_logger

logger = get_logger("medical_graph.store")

GRAPH_DIR = Path("graph_data")
GRAPH_DIR.mkdir(exist_ok=True)

FOUNDATION_PATH = GRAPH_DIR / "medical_foundation.json"


def save_graph(G: nx.DiGraph, path: Path):
    """Saves NetworkX graph to JSON."""
    data = nx.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Graph saved | path='{path}'")


def load_graph(path: Path) -> nx.DiGraph:
    """Loads NetworkX graph from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Graph not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    G = nx.node_link_graph(data, directed=True)
    logger.info(
        f"Graph loaded | nodes={G.number_of_nodes()} | "
        f"edges={G.number_of_edges()}"
    )
    return G


def get_document_graph_path(document_id: int) -> Path:
    return GRAPH_DIR / f"medical_doc_{document_id}.json"


def get_combined_graph_path(document_id: int) -> Path:
    return GRAPH_DIR / f"medical_combined_{document_id}.json"


def get_or_build_foundation() -> nx.DiGraph:
    """
    Returns foundation graph.
    Builds and saves it on first call, loads from disk after.
    """
    if FOUNDATION_PATH.exists():
        return load_graph(FOUNDATION_PATH)

    from backend.medical_graph.foundation_graph import build_foundation_graph
    G = build_foundation_graph()
    save_graph(G, FOUNDATION_PATH)
    return G