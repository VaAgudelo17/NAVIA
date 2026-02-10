#!/bin/bash
# ============================================================================
# NAVIA Mobile - Script de Inicio
# ============================================================================
# Ejecuta el backend y la app móvil Expo
# Uso: ./start-mobile.sh
# ============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$BASE_DIR/navia-backend"
MOBILE_DIR="$BASE_DIR/navia-mobile"

# Obtener IP local
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}    NAVIA Mobile - Iniciando Servicios     ${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

cleanup() {
    echo ""
    echo -e "${YELLOW}Deteniendo servicios...${NC}"
    kill $BACKEND_PID 2>/dev/null
    echo -e "${GREEN}Servicios detenidos.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Iniciar Backend
echo -e "${GREEN}[1/2] Iniciando Backend (FastAPI)...${NC}"
cd "$BACKEND_DIR"
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo -e "      URL: http://$LOCAL_IP:8000"
echo ""

sleep 3

# Iniciar App Móvil
echo -e "${GREEN}[2/2] Iniciando Expo (App Móvil)...${NC}"
cd "$MOBILE_DIR"
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}✓ Backend listo${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Tu IP local: ${YELLOW}$LOCAL_IP${NC}"
echo ""
echo -e "Configura en tu app:"
echo -e "  ${YELLOW}src/constants/config.ts${NC}"
echo -e "  API_BASE_URL = 'http://$LOCAL_IP:8000'"
echo ""
echo -e "Presiona ${YELLOW}Ctrl+C${NC} para detener"
echo ""

# Iniciar Expo
npx expo start

wait
