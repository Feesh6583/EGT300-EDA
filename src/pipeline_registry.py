# src/eda_pipeline/pipeline_registry.py
from kedro.pipeline import Pipeline
from .pipeline import create_nodes

def register_pipelines() -> dict:
    nodes = create_nodes()
    pipeline = Pipeline(nodes)
    return {"__default__": pipeline}
