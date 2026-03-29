"""
Prepara el modelo YOLO-World con las clases personalizadas durante el build.
Requiere CLIP instalado. Guarda el modelo COMPLETO (pesos + head reconfigurado).
En runtime, se carga el modelo guardado directamente — sin CLIP.
"""
import sys
import os
sys.path.insert(0, '/app')

from ultralytics import YOLOWorld

# Importar lista de clases
from app.services.object_detection_service import WORLD_CLASSES_ES, NAVIGATION_WORLD_CLASSES_ES

model_name = os.environ.get('YOLO_MODEL', 'yolov8s-worldv2.pt')
print(f"Cargando modelo {model_name}...")
model = YOLOWorld(model_name)

# Usar clases completas (exploración + navegación unificadas)
# configure_for_navigation filtra por relevancia en la capa de guía, no en YOLO
full_classes = list({**WORLD_CLASSES_ES, **NAVIGATION_WORLD_CLASSES_ES}.keys())
print(f"Configurando {len(full_classes)} clases...")
model.set_classes(full_classes)

# Guardar modelo COMPLETO (pesos + detection head reconfigurado para N clases)
output_path = '/app/yolov8s-worldv2-custom.pt'
model.save(output_path)
print(f"Modelo guardado en {output_path}")
print("Runtime no necesitará CLIP.")
