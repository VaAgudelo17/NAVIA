from PIL import Image, ImageDraw, ImageFont
import io, requests, json

API_URL = 'https://patient-charisma-production.up.railway.app/api/v1/analyze/lectura'

def create_test_image(lines, bg_color=(255,255,255), text_color=(0,0,0), font_size=20):
    # Avoid pure white (248+,248+,248+) which triggers the brightness quality check
    if bg_color == (255,255,255):
        bg_color = (230, 230, 230)
    img = Image.new('RGB', (800, 600), color=bg_color)
    draw = ImageDraw.Draw(img)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill=text_color)
        y += 30
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=95)
    buf.seek(0)
    return buf

def test_reading(name, image_buf):
    try:
        r = requests.post(
            API_URL,
            files={'image': ('test.jpg', image_buf, 'image/jpeg')},
            timeout=60
        )
        return r.status_code, r.json()
    except Exception as e:
        return None, {"error": str(e)}

test_cases = [
    {
        "name": "1. Chat WhatsApp",
        "expected_doc_type": "chat",  # exact match
        "bg_color": (26, 26, 26),
        "text_color": (255, 255, 255),
        "lines": [
            "asi. Creeme que tanto tu como ella estan ansiosos",
            "de estar bien y hablarse y muchas veces no es",
            "hacerse de menos o algo si no que simplemente",
            "tomar la iniciativa ve mira la cague o tu",
            "la cagaste pero hablemos. Independiente de",
            "quien sea el error/culpa mientras no sea algo",
            "no negociable ya sea una infidelidad...",
            "8:13 AM",
        ]
    },
    {
        "name": "2. Nequi transfer",
        "expected_doc_type": "app_banking",  # Nequi is a banking app
        "bg_color": (255, 255, 255),
        "text_color": (0, 0, 0),
        "lines": [
            "Escanea este QR con Nequi para verificar tu envio",
            "Para",
            "Valentina",
            "Cuanto?",
            "$ 150.000,00",
            "Numero Nequi",
            "350 431 4868",
            "Fecha",
            "29 de julio 2025, 10:42",
        ]
    },
    {
        "name": "3. CV / Hoja de vida",
        "expected_doc_type": "hoja_de_vida",
        "bg_color": (255, 255, 255),
        "text_color": (0, 0, 0),
        "lines": [
            "CONTACTO",
            "Identificacion: 123456    hola@unsitioestupendo.com.co",
            "Calle Cualquiera 123      (316) 212-3456",
            "RESUMEN PROFESIONAL",
            "Profesional con experiencia en gestion de proyectos",
            "EXPERIENCIA",
            "2030 - Gerente en Empresa ABC",
            "Liderazgo de equipos de 20 personas",
            "EDUCACION",
            "2025 - Universidad Nacional - Ingenieria",
        ]
    },
    {
        "name": "4. Product label (Cetaphil)",
        "expected_doc_type": "etiqueta",
        "bg_color": (255, 255, 255),
        "text_color": (0, 0, 0),
        "lines": [
            "Cetaphil",
            "OIL CONTROL",
            "Hidratante Facial Matificante Antimanchas",
            "0.5% Acido Salicilico",
            "Pieles mixtas, grasas y/o con barros y espinillas",
            "8H control de brillo, Rapida absorcion",
            "Uso externo. 88ml",
            "Acido Salicilico y Bisabolol",
        ]
    },
    {
        "name": "5. English vocabulary book",
        "expected_doc_type": "desconocido",  # no libro subtype, reads as paragraph
        "bg_color": (255, 255, 255),
        "text_color": (0, 0, 0),
        "lines": [
            "FAST ENGLISH",
            "TURN UP    RAISE THE VOLUME",
            "TURN OFF   SWITCH OFF ELECTRICITY",
            "TURN ON    SWITCH ON THE ELECTRICITY",
            "CALL ON    VISIT",
            "GET OVER   RECOVER FROM SICKNESS",
            "GO OVER    REVIEW",
            "LOOK AFTER TAKE CARE OF",
            "RUN INTO   MEET BY CHANCE",
            "CATCH UP WITH  REACH SAME LEVEL",
        ]
    },
    {
        "name": "6. WhatsApp meme/post",
        "expected_doc_type": "desconocido",  # no meme subtype, reads as paragraph
        "bg_color": (26, 26, 26),
        "text_color": (255, 255, 255),
        "lines": [
            "me siento ni vieja ni joven...",
            "es como si fuera seminueva, preanciana ......",
            "Demasiado vieja para ser joven",
            "y demasiado joven para ser vieja....",
            "Lo que si es verdad es que",
            "tengo mas de 18 y menos de 100",
            "(El resto son chismes)",
        ]
    },
    {
        "name": "7. Factura / Recibo",
        "expected_doc_type": "factura",
        "bg_color": (255, 255, 255),
        "text_color": (0, 0, 0),
        "lines": [
            "FACTURA DE VENTA",
            "Empresa: Tienda ABC S.A.S",
            "NIT: 900.123.456-7",
            "Fecha: 09/04/2026",
            "Cliente: Juan Perez",
            "Producto: Laptop HP    $2.500.000",
            "Producto: Mouse        $45.000",
            "Subtotal:              $2.545.000",
            "IVA 19%:               $483.550",
            "TOTAL:                 $3.028.550",
        ]
    },
    {
        "name": "8. Medication label",
        "expected_doc_type": "receta_medica",
        "bg_color": (255, 255, 255),
        "text_color": (0, 0, 0),
        "lines": [
            "AMOXICILINA 500mg",
            "Capsulas",
            "Indicaciones: Infecciones bacterianas",
            "Dosis: 1 capsula cada 8 horas",
            "Duracion: 7 dias",
            "Doctor: Dr. Martinez",
            "Paciente: Ana Lopez",
            "No suspender sin consultar al medico",
        ]
    },
]

results = []

for tc in test_cases:
    print(f"\n{'='*60}")
    print(f"Testing: {tc['name']}")
    image_buf = create_test_image(
        tc["lines"],
        bg_color=tc["bg_color"],
        text_color=tc["text_color"]
    )
    status_code, response = test_reading(tc["name"], image_buf)

    doc_type = response.get("document_type", response.get("doc_type", "N/A"))
    narrative = response.get("narrative", response.get("description", "N/A"))

    # Simple correctness heuristic: check if expected keyword appears in doc_type
    expected_keyword = tc["expected_doc_type"].split("/")[0].strip().lower()
    returned_lower = doc_type.lower() if isinstance(doc_type, str) else ""
    correct = expected_keyword in returned_lower

    print(f"  HTTP Status   : {status_code}")
    print(f"  doc_type      : {doc_type}")
    print(f"  Expected type : {tc['expected_doc_type']}")
    print(f"  Correct?      : {'YES' if correct else 'NO'}")
    print(f"  narrative     : {str(narrative)[:200]}")
    if not correct:
        print(f"  FULL RESPONSE : {json.dumps(response, ensure_ascii=False, indent=2)}")

    results.append({
        "name": tc["name"],
        "http_status": status_code,
        "expected_doc_type": tc["expected_doc_type"],
        "returned_doc_type": doc_type,
        "correct": correct,
        "narrative": narrative,
        "full_response": response
    })

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for r in results:
    mark = "PASS" if r["correct"] else "FAIL"
    print(f"[{mark}] {r['name']}")
    print(f"       Expected : {r['expected_doc_type']}")
    print(f"       Got      : {r['returned_doc_type']}")

passed = sum(1 for r in results if r["correct"])
print(f"\nTotal: {passed}/{len(results)} passed")

# Save full results to JSON
with open('/Users/valentinaagudelo/Desktop/UNIVERSIDAD-USB/NAVIA-PRODUCTO/NAVIA/test_reading_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nFull results saved to test_reading_results.json")
