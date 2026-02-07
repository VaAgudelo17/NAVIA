#!/bin/bash
# ============================================================================
# NAVIA - Script de Inicio
# ============================================================================
# Ejecuta el backend (FastAPI) y frontend (Next.js) simultáneamente
# Uso: ./start.sh
# ============================================================================

# Colores para mensajes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # Sin color

# Directorio base
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$BASE_DIR/navia-backend"
FRONTEND_DIR="$BASE_DIR/navia-fronted"

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}       NAVIA - Iniciando Servidores        ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Función para limpiar procesos al salir
cleanup() {
    echo ""
    echo -e "${YELLOW}Deteniendo servidores...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}Servidores detenidos.${NC}"
    exit 0
}

# Capturar Ctrl+C para limpiar
trap cleanup SIGINT SIGTERM

# Iniciar Backend
echo -e "${GREEN}[1/2] Iniciando Backend (FastAPI)...${NC}"
cd "$BACKEND_DIR"
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo -e "      PID: $BACKEND_PID"
echo -e "      URL: http://localhost:8000"
echo -e "      Docs: http://localhost:8000/docs"
echo ""

# Esperar a que el backend inicie
sleep 3

# Iniciar Frontend
echo -e "${GREEN}[2/2] Iniciando Frontend (Next.js)...${NC}"
cd "$FRONTEND_DIR"
npm run dev -- -p 3002 &
FRONTEND_PID=$!
echo -e "      PID: $FRONTEND_PID"
echo -e "      URL: http://localhost:3002"
echo ""

echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✓ Ambos servidores están corriendo${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Backend:  ${YELLOW}http://localhost:8000${NC}"
echo -e "Frontend: ${YELLOW}http://localhost:3002${NC}"
echo -e "API Docs: ${YELLOW}http://localhost:8000/docs${NC}"
echo ""
echo -e "Presiona ${YELLOW}Ctrl+C${NC} para detener ambos servidores"
echo ""

# Mantener el script corriendo
wait
