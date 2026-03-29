"""
Pre-computa los embeddings de texto de YOLO-World durante el build de Docker.
Requiere CLIP instalado. En runtime, CLIP NO es necesario.

Guarda dos archivos:
- /app/embeddings_full.pt    → clases completas para exploración
- /app/embeddings_nav.pt     → clases compactas para navegación
"""
import sys
import os
sys.path.insert(0, '/app')

import torch
from ultralytics import YOLOWorld
from pathlib import Path

# Importar listas de clases
from app.services.object_detection_service import WORLD_CLASSES_ES, NAVIGATION_WORLD_CLASSES_ES

model_name = os.environ.get('YOLO_MODEL', 'yolov8s-worldv2.pt')
print(f"Cargando modelo {model_name}...")
model = YOLOWorld(model_name)

# --- Clases completas (exploración) ---
full_classes = list({**WORLD_CLASSES_ES, **NAVIGATION_WORLD_CLASSES_ES}.keys())
print(f"Computando embeddings para {len(full_classes)} clases completas...")
model.set_classes(full_classes)
torch.save(
    {'classes': full_classes, 'txt_feats': model.model.txt_feats.cpu()},
    '/app/embeddings_full.pt'
)
print("embeddings_full.pt guardado.")

# --- Clases de navegación (compactas) ---
nav_classes = list(NAVIGATION_WORLD_CLASSES_ES.keys())
print(f"Computando embeddings para {len(nav_classes)} clases de navegación...")
model.set_classes(nav_classes)
torch.save(
    {'classes': nav_classes, 'txt_feats': model.model.txt_feats.cpu()},
    '/app/embeddings_nav.pt'
)
print("embeddings_nav.pt guardado.")

print("Precomputo completado. CLIP no será necesario en runtime.")
