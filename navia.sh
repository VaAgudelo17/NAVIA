#!/bin/bash
# ============================================================================
# NAVIA - Iniciar Todo (Backend + Mobile)
# ============================================================================
# Ejecuta los 2 servicios simultaneamente con un solo comando:
#   ./navia.sh
# ============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$BASE_DIR/navia-backend"
MOBILE_DIR="$BASE_DIR/navia-mobile"

# Obtener IP local (para la app movil)
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")

echo ""
echo -e "${CYAN}  _   _    _    __     __ ___    _   ${NC}"
echo -e "${CYAN} | \ | |  / \   \ \   / /|_ _|  / \  ${NC}"
echo -e "${CYAN} |  \| | / _ \   \ \ / /  | |  / _ \ ${NC}"
echo -e "${CYAN} | |\  |/ ___ \   \ V /   | | / ___ \\${NC}"
echo -e "${CYAN} |_| \_/_/   \_\   \_/   |___/_/   \_\\${NC}"
echo ""
echo -e "${BLUE}  Navegacion Inteligente Visual con IA${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# PIDs para cleanup
BACKEND_PID=""

cleanup() {
    echo ""
    echo -e "${YELLOW}Deteniendo todos los servicios...${NC}"
    [ -n "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null
    # Expo se detiene solo al cerrar el script
    echo -e "${GREEN}Todos los servicios detenidos.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

# ============================================================================
# 1. BACKEND
# ============================================================================
echo -e "${GREEN}[1/2] Iniciando Backend (FastAPI)...${NC}"
cd "$BACKEND_DIR"
if [ -d "venv" ]; then
    source venv/bin/activate
fi
python3 main.py &
BACKEND_PID=$!
echo -e "      PID: $BACKEND_PID"
echo ""

# Esperar a que el backend inicie
sleep 15

# ============================================================================
# 2. APP MOVIL
# ============================================================================
echo -e "${GREEN}[2/2] Iniciando App Movil (Expo)...${NC}"
echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}  Los servicios de NAVIA estan activos${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "  Backend:   ${YELLOW}http://localhost:8000${NC}"
echo -e "  API Docs:  ${YELLOW}http://localhost:8000/docs${NC}"
echo -e "  IP Local:  ${YELLOW}$LOCAL_IP${NC} (para app movil)"
echo ""
echo -e "  ${CYAN}Historial:      GET  http://localhost:8000/api/v1/history${NC}"
echo -e "  ${CYAN}Preferencias:   GET  http://localhost:8000/api/v1/preferences${NC}"
echo ""
echo -e "  Presiona ${YELLOW}Ctrl+C${NC} para detener todo"
echo ""

cd "$MOBILE_DIR"
npx expo start --clear

# Si expo se cierra, limpiar todo
cleanup
