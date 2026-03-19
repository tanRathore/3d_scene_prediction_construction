# Future 3D Semantic Scene Graph Prediction ( incomplete ongoing )

This project explores how to predict the **future state of a 3D scene** from RGB-D video using an object-centric representation.

The core idea is to move beyond static reconstruction and instead model how a scene evolves over time. Given a sequence of RGB-D frames, the system builds a **3D semantic scene graph** (objects, positions, visibility, and relations) and then predicts how that graph will look in the future.


video link for demo : 
---

## Pipeline Overview

The system is built as an end-to-end pipeline:

- Object detection from RGB frames  
- Mask-aware 3D lifting using depth  
- Lightweight tracking across frames  
- Scene graph construction  
- Temporal windowing  
- Future graph prediction using a transformer-based model  
- Comparison against simple baselines (copy-last and constant velocity)  
- Room-scale visualization using reconstructed scene meshes  

One key focus of this project is understanding **when learning actually helps**. In mostly static scenes, simple persistence is surprisingly strong at short horizons. However, at longer horizons and in dynamic cases (movement, occlusion changes), the learned model performs significantly better.

---

## Current Status

This project is currently in a **minimum viable research prototype stage**.

The full end-to-end pipeline is working, including:

- Real RGB-D sequence ingestion  
- 3D graph construction  
- Temporal forecasting  
- Quantitative evaluation  
- Qualitative 3D visualization  

However, it is still limited in scope:

- Experiments are based on a small number of scenes  
- The graph representation is relatively simple  
- The model predicts object-centric future graphs, not full scene geometry  

---

## Project Structure

Below is the high-level structure of the repository:
future_scene_graphs_mvp/
│
├── src/
│ ├── data/ # Dataset loading, RGB-D parsing, intrinsics/poses handling
│ ├── detection/ # Object detection (YOLO) and segmentation integration
│ ├── lifting/ # 2D → 3D lifting using depth + camera geometry
│ ├── tracking/ # Object tracking across frames
│ ├── graph/ # Scene graph construction (nodes, edges, features)
│ ├── temporal/ # Window creation and sequence handling
│ ├── models/ # Transformer-based forecasting model
│ ├── evaluation/ # Metrics (L2, visibility F1, edge F1, etc.)
│ └── utils/ # Helper functions (geometry, IO, visualization helpers)
│
├── scripts/
│ ├── 01_extract_graphs.py # Build graphs from RGB-D sequences
│ ├── 02_make_windows.py # Create temporal windows for training
│ ├── 03_train_model.py # Train forecasting model
│ ├── 04_eval.py # Run evaluation and compute metrics
│ ├── 05_visualize.py # Local visualization (triptych, overlays)
│ └── 17_make_report_assets.py # Generate figures/tables for report
│
├── runs/
│ ├── graphs/ # Extracted scene graphs
│ ├── windows/ # Temporal datasets
│ ├── models/ # Saved model checkpoints
│ ├── eval/ # Evaluation outputs
│ └── report_assets/ # Figures, tables, and visualization outputs
│
├── configs/ # Config files for training/evaluation
├── requirements.txt # Dependencies
└── README.md

### What each part does

- **src/data/** → Handles loading RGB, depth, poses, and intrinsics  
- **src/detection/** → Detects objects and integrates segmentation masks  
- **src/lifting/** → Converts 2D detections into 3D object estimates  
- **src/tracking/** → Maintains object identity across frames  
- **src/graph/** → Builds per-frame scene graphs  
- **src/temporal/** → Groups graphs into time windows  
- **src/models/** → Learns to predict future graphs  
- **src/evaluation/** → Computes metrics and breakdowns  
- **scripts/** → Runs each stage of the pipeline step-by-step  
- **runs/** → Stores outputs, models, and visualizations  

---

## Ongoing Work

The long-term goal is to move toward **full 3D scene understanding and continuation**.

Future directions include:

- Predicting full 3D scene structure, not just object graphs  
- Extending scenes beyond observed views (e.g., continuing a hallway)  
- Improving segmentation and object-level reasoning  
- Scaling to multiple scenes and larger datasets  
- Building richer graph representations with stronger semantics  

---

## Why this matters

This work sits at the intersection of:

- 3D vision  
- scene graphs  
- temporal prediction  
- embodied AI  

A system that can **understand and predict how a scene evolves** is a step toward more general world models for robotics and intelligent agents.