import json
import os
from datetime import datetime, timezone

# 1. Esquema JSON de Fermentação da Cacaulogia (v2 - Altamente Compatível Draft-07)
CACAULOGY_FERMENTATION_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "CacaulogyFermentationTelemetry",
    "description": "Payload de telemetria termodinâmica e química para monitoramento de bioprocessos de fermentação de cacau.",
    "type": "object",
    "required": [
        "lot_id",
        "sensor_node_id",
        "timestamp",
        "fermentation_hour",
        "internal_mass_temperature_c",
        "pulp_ph",
        "cotyledon_ph"
    ],
    "properties": {
        "lot_id": {
            "type": "string",
            "description": "Identificador único do lote."
        },
        "sensor_node_id": {
            "type": "string",
            "description": "Identificador da sonda IoT."
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "Data e hora ISO 8601."
        },
        "fermentation_hour": {
            "type": "number",
            "minimum": 0,
            "description": "Horas decorridas."
        },
        "internal_mass_temperature_c": {
            "type": "number",
            "minimum": 15.0,
            "maximum": 60.0,
            "description": "Temperatura interna em °C."
        },
        "ambient_temperature_c": {
            "type": "number"
        },
        "ambient_humidity_pct": {
            "type": "number",
            "minimum": 0,
            "maximum": 100
        },
        "pulp_ph": {
            "type": "number",
            "minimum": 2.0,
            "maximum": 7.0
        },
        "cotyledon_ph": {
            "type": "number",
            "minimum": 3.0,
            "maximum": 8.0
        },
        "turning_event_detected": {
            "type": "boolean"
        }
    }
}

# Tenta carregar o arquivo JSON local se ele existir
def carregar_schema_arquivo():
    caminhos_busca = [
        "cacaulogy-fermentation-schema-v2.json",
        "cacaulogy-fermentation-schema.json",
        "artifacts/cacaulogy-fermentation-schema-v2.json",
        "../artifacts/cacaulogy-fermentation-schema-v2.json"
    ]
    for caminho in caminhos_busca:
        if os.path.exists(caminho):
            try:
                with open(caminho, "r", encoding="utf-8") as f:
                    print(f"📥 [INFO] Carregando esquema JSON a partir do arquivo: '{caminho}'")
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ [AVISO] Falha ao ler arquivo {caminho}: {e}. Usando esquema padrão embutido.")
    print("ℹ️ [INFO] Arquivo de esquema JSON externo não encontrado. Usando esquema estático embutido.")
    return CACAULOGY_FERMENTATION_SCHEMA

# Detecta se a biblioteca 'jsonschema' está disponível
try:
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    print("⚠️ [AVISO] Biblioteca 'jsonschema' não instalada localmente.")
    print("👉 Para instalar, execute no terminal: pip install jsonschema")
    print("ℹ️ O simulador continuará sendo executado utilizando validação programática de contingência (fallback).\n")

# 2. Geração de Dados Sintéticos de Telemetria (Sem avisos de depreciação do python utcnow)
def obter_timestamp_atual_utc():
    # Compatível com Python 3.8+ e livre de avisos de depreciação no Python 3.12+
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

telemetria_pico_ideal = {
    "lot_id": "LOT-AMZ-2026-0907A",
    "sensor_node_id": "ESP32-COCHO-04",
    "timestamp": obter_timestamp_atual_utc(),
    "fermentation_hour": 84.0,                  # Janela crítica de pico metabólico
    "internal_mass_temperature_c": 48.5,        # Pico térmico ideal mapeado
    "ambient_temperature_c": 28.2,
    "ambient_humidity_pct": 82.0,
    "pulp_ph": 3.45,                            # Polpa ácida pela ação de bactérias acéticas
    "cotyledon_ph": 4.80,                       # Declínio do pH interno indicando penetração ácida
    "turning_event_detected": False
}

telemetria_com_falha_termica = {
    "lot_id": "LOT-AMZ-2026-0907B",
    "sensor_node_id": "ESP32-COCHO-05",
    "timestamp": obter_timestamp_atual_utc(),
    "fermentation_hour": 84.0,
    "internal_mass_temperature_c": 39.2,        # Temperatura muito baixa para 84h (anomalia)
    "pulp_ph": 4.20,
    "cotyledon_ph": 5.80
}

# 3. Validador programático alternativo em caso de ausência da biblioteca jsonschema
def validar_manualmente(instance, schema):
    """Validador básico de contingência."""
    required_fields = schema.get("required", [])
    for field in required_fields:
        if field not in instance:
            raise ValueError(f"Campo obrigatório '{field}' ausente no payload.")
            
    # Valida tipos básicos
    properties = schema.get("properties", {})
    for key, val in instance.items():
        if key in properties:
            expected_type = properties[key].get("type")
            if expected_type == "string" and not isinstance(val, str):
                raise TypeError(f"O campo '{key}' deve ser string, recebeu {type(val).__name__}.")
            elif expected_type == "number" and not isinstance(val, (int, float)):
                raise TypeError(f"O campo '{key}' deve ser numérico, recebeu {type(val).__name__}.")
            elif expected_type == "boolean" and not isinstance(val, bool):
                raise TypeError(f"O campo '{key}' deve ser booleano, recebeu {type(val).__name__}.")
                
            # Valida limites numéricos básicos
            minimum = properties[key].get("minimum")
            maximum = properties[key].get("maximum")
            if minimum is not None and val < minimum:
                raise ValueError(f"O campo '{key}' possui valor {val} abaixo do limite mínimo ({minimum}).")
            if maximum is not None and val > maximum:
                raise ValueError(f"O campo '{key}' possui valor {val} acima do limite máximo ({maximum}).")

# 4. Função de Processamento e Validação no Back-End
def processar_telemetria_fermentacao(payload_json):
    print(f"\n[BACK-END] Recebendo dados do lote: {payload_json.get('lot_id')}...")
    
    # Carrega esquema (do arquivo ou estático)
    schema = carregar_schema_arquivo()
    
    # Passo A: Validação de Esquema
    try:
        if JSONSCHEMA_AVAILABLE:
            validate(instance=payload_json, schema=schema)
            print("✅ Passo 1/2: Validação de estrutura JSON (via jsonschema) aprovada!")
        else:
            validar_manualmente(instance=payload_json, schema=schema)
            print("✅ Passo 1/2: Validação de contingência manual aprovada! (Sem biblioteca externa)")
    except Exception as e:
        print(f"❌ Erro de Validação de Estrutura: {str(e)}")
        return {"status": "REJECTED", "reason": "INVALID_JSON_STRUCTURE", "details": str(e)}

    # Passo B: Validação de Alvo Científico (Pico de Bioprocesso)
    temp_atual = payload_json["internal_mass_temperature_c"]
    horas_decorridas = payload_json["fermentation_hour"]
    
    # Regra de negócio: Na janela entre 72h e 96h, a massa DEVE atingir a faixa ótima (45°C - 50°C)
    if 72.0 <= horas_decorridas <= 96.0:
        if temp_atual >= 48.5:
            print(f"🔥 Passo 2/2: Alvo térmico de Cacaulogia ALCANÇADO! Temperatura: {temp_atual}°C em {horas_decorridas}h.")
            return {
                "status": "APPROVED",
                "bioprocess_evaluation": "OPTIMAL_THERMAL_PEAK_REACHED",
                "message": "Condições bioquímicas ideais para síntese de precursores aromáticos frutados."
            }
        elif temp_atual < 44.0:
            print(f"⚠️ Passo 2/2: ALERTA DE BIOPROCESSO! Temperatura abaixo do crítico na hora {horas_decorridas}: {temp_atual}°C.")
            return {
                "status": "WARNING",
                "bioprocess_evaluation": "THERMAL_PEAK_DEFICIT",
                "message": "Sub-fermentação detectada. Risco de alta adstringência e acidez volátil residual."
            }
        else:
            print(f"ℹ️ Passo 2/2: Temperatura aceitável, mas abaixo do pico máximo histórico de 48.5°C: {temp_atual}°C.")
            return {"status": "APPROVED", "bioprocess_evaluation": "ACCEPTABLE_THERMAL_RANGE"}
            
    return {"status": "APPROVED", "bioprocess_evaluation": "OUT_OF_PEAK_WINDOW"}

if __name__ == "__main__":
    print("=== SIMULANDO PAYLOAD 1: LOTE COM BIOPROCESSO PERFEITO ===")
    resultado_1 = processar_telemetria_fermentacao(telemetria_pico_ideal)
    print(f"Resultado final do processamento: {resultado_1}")

    print("\n=== SIMULANDO PAYLOAD 2: LOTE COM FALHA DE BIOPROCESSO ===")
    resultado_2 = processar_telemetria_fermentacao(telemetria_com_falha_termica)
    print(f"Resultado final do processamento: {resultado_2}")
