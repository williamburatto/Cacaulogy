import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuração visual e de layout profissional
st.set_page_config(
    page_title="Cacaulogy SaaS v16.0 - Smart Contracts, Low-Fee FX & Amazonian Biorefinery",
    page_icon="🍫",
    layout="wide"
)

# Estilização editorial (Warm Chocolate, Cream and Gold/Orange accents)
st.markdown("""
<style>
    .reportview-container { background: #fcfaf7; }
    h1, h2, h3 { color: #403228 !important; font-family: 'Georgia', serif; }
    .stButton>button { background-color: #cc5500; color: white; border-radius: 8px; font-weight: bold; }
    .metric-card {
        background-color: white; border-left: 5px solid #cc5500;
        padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .metric-card h4 { margin: 0; color: #998c82; font-size: 11px; text-transform: uppercase; }
    .metric-card h2 { margin: 5px 0; color: #403228; font-size: 22px; font-weight: bold; }
    .metric-card p { margin: 0; color: #5a5a5a; font-size: 12px; }
    .vector-card {
        background-color: #faf7f3; border: 1px solid #e1dbd6;
        padding: 16px; border-radius: 8px; margin-bottom: 15px;
    }
    .vector-card h4 { color: #cc5500; margin-top: 0; font-size: 16px; font-weight: bold; }
    .badge-tag {
        background-color: #403228; color: #fcfaf7; padding: 3px 8px;
        border-radius: 4px; font-size: 11px; font-weight: bold; display: inline-block; margin-right: 5px;
    }
    .timeline-card {
        background-color: #ffffff; border: 1px solid #e1dbd6; border-left: 4px solid #27ae60;
        padding: 12px 16px; border-radius: 6px; margin-bottom: 10px;
    }
    .fx-card {
        background-color: #f0f7f4; border: 1px solid #b8e0d2; border-left: 5px solid #2e7d32;
        padding: 15px; border-radius: 8px; margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Dicionário de Internacionalização Completo (6 Idiomas)
LANGUAGES = {
    "Português": {
        "title": "🍫 Plataforma Cacaulogy SaaS v16.0",
        "subtitle": "Biorrefinaria de Cacau Amazônico, Consórcio de Empresas, Contratos Inteligentes & Conversão Cambial Automática (XRPL / CBDC)",
        "sidebar_header": "🌐 Idioma & Parâmetros do Consórcio",
        "select_lang": "Escolha o Idioma / Select Language:",
        "fresh_fruit_vol": "Volume de Frutos Frescos de Cacau (Tons/ano):",
        "active_vectors_label": "Vetores Ativos no Consórcio:",
        "tab_advances": "📊 Avanços (v1-v16)",
        "tab_biorefinery": "🌱 Biorrefinaria (8 Vetores)",
        "tab_finance": "📈 Viabilidade & Economia FX",
        "tab_smart_contracts": "🤝 Contratos Inteligentes & FX (XRPL)",
        "tab_equipment": "🛠️ Equipamentos & Empresas de Belém/PA",
        "min_vol_viable": "Volume Mínimo Viável",
        "min_capex": "Investimento Mínimo (CAPEX)",
        "estimated_profit": "Lucro Estimado / ano",
        "key_equipments": "Equipamentos Específicos",
        "regional_partners": "Empresas Parceiras (Região de Belém/PA)",
        "consortium_summary": "Resumo de Viabilidade do Consórcio",
        "total_capex": "CAPEX Total do Consórcio",
        "total_revenue": "Receita Bruta Projetada",
        "total_profit": "Lucro Líquido Retido",
        "payback_months": "Payback Estimado",
        "lots_title": "🔍 Lotes Homogêneos de Amêndoas de Exportação",
        "fx_savings_title": "💡 Economia em Taxas Cambiais com XRPL Smart Contracts"
    },
    "English": {
        "title": "🍫 Cacaulogy SaaS v16.0 Platform",
        "subtitle": "Amazonian Cocoa Biorefinery, Industrial Consortium, Smart Contracts & Automated Low-Fee FX Settlement (XRPL / CBDC)",
        "sidebar_header": "🌐 Language & Consortium Parameters",
        "select_lang": "Choose Language / Escolha o Idioma:",
        "fresh_fruit_vol": "Fresh Cocoa Fruit Volume (Tons/year):",
        "active_vectors_label": "Active Consortium Vectors:",
        "tab_advances": "📊 Advances (v1-v16)",
        "tab_biorefinery": "🌱 Biorefinery (8 Vectors)",
        "tab_finance": "📈 Viability & FX Savings",
        "tab_smart_contracts": "🤝 Smart Contracts & FX (XRPL)",
        "tab_equipment": "🛠️ Equipment & Belém/PA Companies",
        "min_vol_viable": "Minimum Viable Volume",
        "min_capex": "Minimum Investment (CAPEX)",
        "estimated_profit": "Estimated Profit / year",
        "key_equipments": "Specific Equipment",
        "regional_partners": "Partner Companies (Belém/PA Region)",
        "consortium_summary": "Consortium Viability Summary",
        "total_capex": "Total Consortium CAPEX",
        "total_revenue": "Projected Gross Revenue",
        "total_profit": "Retained Net Profit",
        "payback_months": "Estimated Payback",
        "lots_title": "🔍 Standardized Export Bean Lots",
        "fx_savings_title": "💡 FX Fee Savings with XRPL Smart Contracts"
    },
    "中文": {
        "title": "🍫 Cacaulogy SaaS v16.0 平台",
        "subtitle": "亚马逊可可生物精炼厂、企业财团、智能合约与自动低费率汇率转换 (XRPL / CBDC)",
        "sidebar_header": "🌐 语言与财团参数",
        "select_lang": "选择语言 / Select Language:",
        "fresh_fruit_vol": "新鲜可可果实年产量 (吨/年):",
        "active_vectors_label": "财团活跃向量:",
        "tab_advances": "📊 演进历程 (v1-v16)",
        "tab_biorefinery": "🌱 生物精炼厂 (8大向量)",
        "tab_finance": "📈 可行性与外汇节省",
        "tab_smart_contracts": "🤝 智能合约与外汇 (XRPL)",
        "tab_equipment": "🛠️ 设备与贝伦 (Belém) 区域企业",
        "min_vol_viable": "最低可行产量",
        "min_capex": "最低投资额 (CAPEX)",
        "estimated_profit": "预估年利润",
        "key_equipments": "专用设备",
        "regional_partners": "合作企业 (贝伦及帕拉州区域)",
        "consortium_summary": "财团可行性摘要",
        "total_capex": "财团总资本支出",
        "total_revenue": "预测总收入",
        "total_profit": "保留净利润",
        "payback_months": "预估投资回收期",
        "lots_title": "🔍 标准化出口可可豆批次",
        "fx_savings_title": "💡 使用 XRPL 智能合约节省的手续费"
    },
    "Español": {
        "title": "🍫 Plataforma Cacaulogy SaaS v16.0",
        "subtitle": "Biorrefinería de Cacao Amazónico, Consorcio Industrial, Contratos Inteligentes & Conversión Cambiaria Automática (XRPL / CBDC)",
        "sidebar_header": "🌐 Idioma y Parámetros del Consorcio",
        "select_lang": "Seleccionar Idioma / Select Language:",
        "fresh_fruit_vol": "Volumen de Frutos Frescos de Cacao (Tons/año):",
        "active_vectors_label": "Vectores Activos en el Consorcio:",
        "tab_advances": "📊 Avances (v1-v16)",
        "tab_biorefinery": "🌱 Biorrefinería (8 Vectores)",
        "tab_finance": "📈 Viabilidad & Ahorro FX",
        "tab_smart_contracts": "🤝 Contratos Inteligentes & FX (XRPL)",
        "tab_equipment": "🛠️ Equipos & Empresas de Belém/PA",
        "min_vol_viable": "Volumen Mínimo Viable",
        "min_capex": "Inversión Mínima (CAPEX)",
        "estimated_profit": "Beneficio Estimado / año",
        "key_equipments": "Equipamiento Específico",
        "regional_partners": "Empresas Aliadas (Región de Belém/PA)",
        "consortium_summary": "Resumen de Viabilidad del Consorcio",
        "total_capex": "CAPEX Total del Consorcio",
        "total_revenue": "Ingresos Brutos Proyectados",
        "total_profit": "Beneficio Neto Retenido",
        "payback_months": "Payback Estimado",
        "lots_title": "🔍 Lotes Homogéneos de Granos de Exportación",
        "fx_savings_title": "💡 Ahorro de Comisiones Cambiarias con Smart Contracts XRPL"
    },
    "Italiano": {
        "title": "🍫 Piattaforma Cacaulogy SaaS v16.0",
        "subtitle": "Bioraffineria del Cacao Amazzonico, Consorzio Industriale, Smart Contract & Conversione Valutaria Automatica (XRPL / CBDC)",
        "sidebar_header": "🌐 Lingua e Parametri del Consorzio",
        "select_lang": "Seleziona Lingua / Select Language:",
        "fresh_fruit_vol": "Volume di Frutti Freschi di Cacao (Tonnellate/anno):",
        "active_vectors_label": "Vettori Attivi nel Consorzio:",
        "tab_advances": "📊 Progressi (v1-v16)",
        "tab_biorefinery": "🌱 Bioraffineria (8 Vettori)",
        "tab_finance": "📈 Viabilità & Risparmio FX",
        "tab_smart_contracts": "🤝 Smart Contract & FX (XRPL)",
        "tab_equipment": "🛠️ Attrezzature & Aziende di Belém/PA",
        "min_vol_viable": "Volume Minimo Viabile",
        "min_capex": "Investimento Minimo (CAPEX)",
        "estimated_profit": "Profitto Stimato / anno",
        "key_equipments": "Attrezzatura Specifica",
        "regional_partners": "Aziende Partner (Regione di Belém/PA)",
        "consortium_summary": "Riepilogo della Viabilità del Consorzio",
        "total_capex": "CAPEX Totale del Consorzio",
        "total_revenue": "Ricavi Lordi Proiettati",
        "total_profit": "Profitto Netto Trattenuto",
        "payback_months": "Payback Stimato",
        "lots_title": "🔍 Lotti Omogenei di Cacao da Esportazione",
        "fx_savings_title": "💡 Risparmio sulle Commissioni di Cambio con Smart Contract XRPL"
    },
    "Français": {
        "title": "🍫 Plateforme Cacaulogy SaaS v16.0",
        "subtitle": "Bioraffinerie de Cacao Amazonien, Consortium Industriel, Smart Contracts & Conversion Devise Automatique (XRPL / CBDC)",
        "sidebar_header": "🌐 Langue & Paramètres du Consortium",
        "select_lang": "Choisir la Langue / Select Language:",
        "fresh_fruit_vol": "Volume de Fruits Frais de Cacao (Tonnes/an):",
        "active_vectors_label": "Vecteurs Actifs du Consortium:",
        "tab_advances": "📊 Progrès (v1-v16)",
        "tab_biorefinery": "🌱 Bioraffinerie (8 Vecteurs)",
        "tab_finance": "📈 Viabilité & Économie FX",
        "tab_smart_contracts": "🤝 Smart Contracts & FX (XRPL)",
        "tab_equipment": "🛠️ Équipements & Entreprises de Belém/PA",
        "min_vol_viable": "Volume Minimum Viable",
        "min_capex": "Investissement Minimum (CAPEX)",
        "estimated_profit": "Profit Estimé / an",
        "key_equipments": "Équipements Spécifiques",
        "regional_partners": "Entreprises Partenaires (Région de Belém/PA)",
        "consortium_summary": "Résumé de Viabilité du Consortium",
        "total_capex": "CAPEX Total du Consortium",
        "total_revenue": "Revenus Brut Projetés",
        "total_profit": "Profit Net Retenu",
        "payback_months": "Payback Estimé",
        "lots_title": "🔍 Lots Homogènes de Fèves d'Exportation",
        "fx_savings_title": "💡 Économies sur Frais de Change via Smart Contracts XRPL"
    }
}

# Sidebar - Seletor de Idioma
selected_lang = st.sidebar.selectbox(
    "🌐 Language / Idioma:",
    ["Português", "English", "中文", "Español", "Italiano", "Français"]
)
L = LANGUAGES[selected_lang]

st.title(L["title"])
st.subheader(L["subtitle"])

st.sidebar.markdown("---")
st.sidebar.header(L["sidebar_header"])

# Controle Deslizante de Volume de Fruto Fresco
fresh_fruit_volume = st.sidebar.slider(
    L["fresh_fruit_vol"],
    min_value=50.0,
    max_value=5000.0,
    value=500.0,
    step=50.0
)

# Cálculo Derivado de Biomassa
dry_bean_volume = fresh_fruit_volume * 0.20  # 20% do fruto fresco = amêndoa seca
dry_husk_volume = fresh_fruit_volume * 0.16  # 16% do fruto fresco = casca seca

st.sidebar.markdown(f"**📊 Biomassa Resultante (Calculada):**")
st.sidebar.write(f"• **Amêndoas Secas:** {dry_bean_volume:,.1f} Tons/ano")
st.sidebar.write(f"• **Casca Seca (Biomassa):** {dry_husk_volume:,.1f} Tons/ano")

# Base de Dados dos 8 Vetores da Biorrefinaria
VECTORS_DATA = {
    "Vetor 1: Alimentos de Especialidade": {
        "name_pt": "Vetor 1: Alimentos de Especialidade (Chocolates Gourmet & Manteiga Fina)",
        "min_vol_fresh": 50.0,
        "min_capex_brl": 120000.0,
        "revenue_per_ton_fresh": 11600.0,
        "opex_ratio": 0.45,
        "equipments": [
            "Cochos de Fermentação de Madeira com Sondas IoT (ESP32/DS18B20)",
            "Estufa Solar Automatizada com Exaustores Mecânicos (SHT31)",
            "Torrador Industrial por Convecção Controlada (110-135°C)",
            "Descascadora e Sopradora de Amêndoas (Winnowing Machine)",
            "Moinho de Pedras de Granito para Refino (Melanger)",
            "Concheira e Temperadeira Automatizada com Controle Térmico"
        ],
        "partners_belem": [
            "Gaudens Chocolateria (Parque Cacaulógico de Belém)",
            "CAMTA - Cooperativa Agrícola Mista de Tomé-Açu",
            "COOPATRANS (Medicilândia / Calha da Transamazônica)",
            "Chocolates da Amazônia / MendoÁ Belém"
        ]
    },
    "Vetor 2: Compostos Bioativos": {
        "name_pt": "Vetor 2: Compostos Bioativos (Polifenóis Totais & Antocianinas)",
        "min_vol_fresh": 125.0,
        "min_capex_brl": 280000.0,
        "revenue_per_ton_fresh": 14400.0,
        "opex_ratio": 0.40,
        "equipments": [
            "Reator de Extração Assistida por Ultrassom (25 kHz, 50°C)",
            "Centrífuga Industrial de Discos de Alta Velocidade (14.000 rpm)",
            "Sistema de Filtração Tangencial por Membrana (UF/NF)",
            "Evaporador e Concentrador a Vácuo Rotativo Industrial"
        ],
        "partners_belem": [
            "Beraca Ingredientes Naturais (Distrito Industrial de Ananindeua)",
            "Laboratório de Extração de Óleos e Extratos da UFPA (PCT Guamá)",
            "Ideflor-Bio / Instituto de Desenvolvimento Florestal do Pará"
        ]
    },
    "Vetor 3: Cosméticos de Alta Performance": {
        "name_pt": "Vetor 3: Cosméticos de Alta Performance (Skincare & Manteiga Virgem)",
        "min_vol_fresh": 150.0,
        "min_capex_brl": 210000.0,
        "revenue_per_ton_fresh": 9500.0,
        "opex_ratio": 0.42,
        "equipments": [
            "Prensa Hidráulica Contínua de Extração Lipídica a Frio",
            "Reator de Desodorização Física Branda e Clarificação",
            "Homogeneizador de Alta Pressão para Emulsões Cosméticas",
            "Envasadora e Blistadeira de Cosméticos Automatizada"
        ],
        "partners_belem": [
            "Natura Ecoparque (Unidade de Benevides/Belém)",
            "Amazon Oil (Distrito Industrial de Ananindeua)",
            "Floras da Amazônia / Laboratórios de Biocosméticos PCT Guamá"
        ]
    },
    "Vetor 4: Fármacos": {
        "name_pt": "Vetor 4: Fármacos (Metilxantinas Purificadas: Teobromina & Cafeína)",
        "min_vol_fresh": 200.0,
        "min_capex_brl": 450000.0,
        "revenue_per_ton_fresh": 18500.0,
        "opex_ratio": 0.38,
        "equipments": [
            "Reator Alcalino de Extração com Solvente Biocompatível (pH 10, 50°C)",
            "Unidade de Purificação Cromatográfica Preparativa (HPLC/Prep-LC)",
            "Sistema de Cristalização e Precipitação Contínua",
            "Liofilizador Industrial de Bancada (-50°C, 0.04 mbar)"
        ],
        "partners_belem": [
            "LAFEPA - Laboratório Farmacêutico do Estado do Pará",
            "Faculdade de Farmácia & Laboratório de Fármacos da UFPA",
            "Polo Farma do Distrito Industrial de Belém"
        ]
    },
    "Vetor 5: Fitoterápicos & Nutracêuticos": {
        "name_pt": "Vetor 5: Fitoterápicos & Nutracêuticos (Cápsulas e Sachês Liofilizados)",
        "min_vol_fresh": 100.0,
        "min_capex_brl": 320000.0,
        "revenue_per_ton_fresh": 13200.0,
        "opex_ratio": 0.40,
        "equipments": [
            "Liofilizador Industrial de Grande Porte (-50°C a -76°C)",
            "Moinho Analítico de Facas Ultrafino (40 mesh / 420 µm)",
            "Misturador de Pós e Granulados em Duplo Cônico",
            "Capsuladora Automática de Alta Velocidade e Seladora de Blisters"
        ],
        "partners_belem": [
            "Belém Nutracêuticos / Ervas da Amazônia",
            "FITOAMAZÔNIA Indústria de Fitoterápicos",
            "Laboratório de Bioprodutos do Instituto Evandro Chagas (IEC / Belém)"
        ]
    },
    "Vetor 6: Biocombustíveis & Bioenergia": {
        "name_pt": "Vetor 6: Biocombustíveis & Bioenergia (Vapor Térmico & Biometano da Casca)",
        "min_vol_fresh": 125.0,
        "min_capex_brl": 180000.0,
        "revenue_per_ton_fresh": 3800.0,
        "opex_ratio": 0.35,
        "equipments": [
            "Triturador e Picador de Casca Fresca de Cacau",
            "Secador Rotativo Contínuo alimentado por Biomassa",
            "Caldeira Industrial a Biomassa (HHV 16.79-17.26 MJ/kg)",
            "Biodigestor Anaeróbio Mesofílico com Purificador de Biogás (PSA) para CH4"
        ],
        "partners_belem": [
            "Belém Bioenergia Brasil (BBB / Marituba)",
            "Grupo Agropalma (Divisão de Bioenergia)",
            "Centrais Geradoras do Distrito Industrial de Ananindeua",
            "Equatorial Energia Pará (Projetos de Eficiência Energética)"
        ]
    },
    "Vetor 7: Bioinsumos & Fertilizantes Organominerais": {
        "name_pt": "Vetor 7: Bioinsumos & Fertilizantes Organominerais (Compostagem & Biofertilizante)",
        "min_vol_fresh": 100.0,
        "min_capex_brl": 85000.0,
        "revenue_per_ton_fresh": 2800.0,
        "opex_ratio": 0.30,
        "equipments": [
            "Compostadeira Industrial Automatizada com Revolvimento Mecânico",
            "Separador de Efluentes e Digestato de Biodigestor",
            "Tanque de Enriquecimento Microbiológico NPK e Inoculação",
            "Envasadora de Biofertilizante Líquido e Adubos Organominerais"
        ],
        "partners_belem": [
            "EMBRAPA Amazônia Oriental (Belém/PA)",
            "BioAmazônia Fertilizantes Orgânicos",
            "Cooperativa Agrícola de Castanhal / Região Metropolitana de Belém",
            "Grupo Formosa Agro"
        ]
    },
    "Vetor 8: Biomateriais & Embalagens Biodegradáveis": {
        "name_pt": "Vetor 8: Biomateriais & Embalagens Biodegradáveis (Bioplásticos de Pectina/Lignina)",
        "min_vol_fresh": 300.0,
        "min_capex_brl": 350000.0,
        "revenue_per_ton_fresh": 6800.0,
        "opex_ratio": 0.42,
        "equipments": [
            "Reator Quimio-Enzimático de Extração de Pectina e Lignina",
            "Misturador e Extrusora Monorosca de Biopolímeros",
            "Injetora de Bioplásticos Biodegradáveis para Moldagem",
            "Sopradora de Filmes Plásticos Ecológicos e Embalagens"
        ],
        "partners_belem": [
            "Plastipak Amazônia (Ananindeua/Belém)",
            "Solução Bioplásticos (Parque Tecnológico Guamá)",
            "Indústrias de Embalagens do Distrito Industrial de Ananindeua"
        ]
    }
}

# Seletor de Vetores Ativos na Sidebar
st.sidebar.markdown(f"**{L['active_vectors_label']}**")
selected_vectors = []
for vec_key, vec_info in VECTORS_DATA.items():
    default_check = True if vec_key in ["Vetor 1: Alimentos de Especialidade", "Vetor 2: Compostos Bioativos", "Vetor 3: Cosméticos de Alta Performance", "Vetor 6: Biocombustíveis & Bioenergia"] else False
    if st.sidebar.checkbox(vec_key, value=default_check):
        selected_vectors.append(vec_key)

# Tabs Principais do Aplicativo v16.0
tab_advances, tab1, tab2, tab3, tab4 = st.tabs([
    L["tab_advances"],
    L["tab_biorefinery"],
    L["tab_finance"],
    L["tab_smart_contracts"],
    L["tab_equipment"]
])

# ==============================================================================
# TAB 0: HISTÓRICO DE AVANÇOS DO SOFTWARE (v1.0 a v16.0)
# ==============================================================================
with tab_advances:
    st.markdown("### 📜 Linha do Tempo e Evolução Tecnológica do Cacaulogy SaaS (v1.0 ➔ v16.0)")
    st.write("Conheça como a plataforma evoluiu de uma simples calculadora de agrupamento até se tornar um ecossistema completo de Biorrefinaria, Smart Contracts e Interoperabilidade CBDC sem fricção cambial.")

    st.markdown("""
    <div class='timeline-card'>
        <h4>🚀 v1.0 - v3.0 | Núcleo Quimiométrico e Clustering K-Means</h4>
        <p>• Agrupamento de 200 famílias produtoras amazônicas utilizando Z-Score normalizado.<br>
        • Mapeamento de perfis moleculares: Extrato Etéreo (%EE), Polifenóis Totais, Razão Teobromina/Cafeína (TB/CF) e Nota QDA.<br>
        • Classificação dos Lotes de Exportação: Lote A Premium (R$ 58/kg), Lote B Antioxidante (R$ 38/kg) e Lote C Trinitário (R$ 32/kg).</p>
    </div>
    <div class='timeline-card'>
        <h4>📡 v4.0 - v7.0 | Telemetria IoT e Validação de Bioprocessos em Tempo Real</h4>
        <p>• Integração com sondas IoT (ESP32 + DS18B20) para monitorar o pico térmico de fermentação (48,5 °C às 84h).<br>
        • Sensores SHT31/DHT22 em estufas solares de secagem para rastreio de umidade e temperatura microclimática.<br>
        • Esqueletos JSON Schema Draft 2020-12 / Draft-07 para validação da integridade estrutural dos dados de campo.</p>
    </div>
    <div class='timeline-card'>
        <h4>🌐 v8.0 - v11.0 | Internacionalização Multilíngue e Adequação Aduaneira</h4>
        <p>• Suporte nativo e instantâneo a 6 idiomas: Português, Inglês, Chinês (Hengqin), Espanhol, Italiano e Francês.<br>
        • Emissão automatizada do Passaporte Cacaulógico QR Code com polígonos de Desmatamento Zero (EUDR EU 2023/1115).<br>
        • Conformidade sanitária e toxicológica (isento de cádmio) para desembaraço expresso na China (GACC Decretos 248/249).</p>
    </div>
    <div class='timeline-card'>
        <h4>🌱 v12.0 - v15.0 | Biorrefinaria Integrada de 8 Vetores e Consórcio Agroindustrial</h4>
        <p>• Aproveitamento de 100% da biomassa do fruto (20% amêndoa seca + 16% casca seca com HHV de 16,79 a 17,26 MJ/kg).<br>
        • 8 Vetores Industriais: Alimentos, Bioativos, Cosméticos, Fármacos, Fitoterápicos, Biocombustíveis, Bioinsumos e Biomateriais.<br>
        • Mapeamento do Consórcio de Empresas da Região Metropolitana de Belém/PA (Gaudens, CAMTA, Beraca, Natura, PCT Guamá, UFPA, EMBRAPA).</p>
    </div>
    <div class='timeline-card' style='border-left: 5px solid #cc5500;'>
        <h4>⚡ v16.0 | Smart Contracts XRPL, CUSUM Escrow e Conversão Cambial de Baixa Taxa (CBDC)</h4>
        <p>• <b>Contratos Inteligentes no XRP Ledger (XRPL):</b> Liquidação automatizada e condicionada à prova biológica de terroir.<br>
        • <b>Filtro Estatístico CUSUM:</b> Congelamento automático em Escrow se houver fraude ou alteração no perfil do lote.<br>
        • <b>Conversão Cambial Automática com Baixas Taxas:</b> Troca instantânea em pools AMM (XLS-30) com taxa de apenas 0,1% a 0,3% (vs. 5% de câmbio bancário tradicional), economizando centenas de milhares de Reais para o consórcio.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# TAB 1: BIORREFINARIA DE CACAU (8 VETORES)
# ==============================================================================
with tab1:
    st.markdown("### 🌱 Biorrefinaria de Cacau Amazônico v16.0")
    st.write("A plataforma Cacaulogy v16.0 converte 100% da biomassa do cacau (20% amêndoa seca + 80% casca/resíduos) em 8 vetores de valor agregado, eliminando o descarte e multiplicando a receita da cooperativa ou consórcio.")

    st.markdown("---")
    
    # Exibição dos Vetores em Cards Informativos
    for vec_key in selected_vectors:
        v = VECTORS_DATA[vec_key]
        
        is_viable = fresh_fruit_volume >= v["min_vol_fresh"]
        status_badge = "✅ VIÁVEL PARA SEU VOLUME" if is_viable else "⚠️ REQUER MAIOR VOLUME"
        status_color = "#2e7d32" if is_viable else "#d32f2f"
        
        st.markdown(f"""
        <div class='vector-card'>
            <div style='float: right; color: {status_color}; font-weight: bold; font-size: 13px;'>{status_badge}</div>
            <h4>{v['name_pt']}</h4>
            <p><b>{L['min_vol_viable']}:</b> {v['min_vol_fresh']} Tons de Frutos Frescos/ano | <b>{L['min_capex']}:</b> R$ {v['min_capex_brl']:,.2f}</p>
            <p><b>{L['key_equipments']}:</b> {", ".join(v['equipments'][:3])} e outros.</p>
            <p><b>{L['regional_partners']}:</b> {", ".join(v['partners_belem'][:2])}.</p>
        </div>
        """, unsafe_allow_html=True)

    # Lotes de Exportação Tradicionais
    st.markdown(f"### {L['lots_title']}")
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.markdown("""
        <div class='metric-card'>
            <h4>LOTE A - FLORAL & FRUTADO</h4>
            <h2>R$ 58,00 / kg</h2>
            <p><b>Bioquímica:</b> Lípides >55%, QDA >8.0, Polifenóis Baixos.</p>
            <p><b>Mercado:</b> Chocolates gourmet de alto luxo (Europa/Ásia).</p>
        </div>
        """, unsafe_allow_html=True)
    with col_l2:
        st.markdown("""
        <div class='metric-card'>
            <h4>LOTE B - ANTIOXIDANTE</h4>
            <h2>R$ 38,00 / kg</h2>
            <p><b>Bioquímica:</b> Polifenóis >10.000 mg GAE/100g, TB/CF >9.0.</p>
            <p><b>Mercado:</b> Cosméticos ativos e fitoterápicos funcionais.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_l3:
        st.markdown("""
        <div class='metric-card'>
            <h4>LOTE C - TRINITÁRIO TRADICIONAL</h4>
            <h2>R$ 32,00 / kg</h2>
            <p><b>Bioquímica:</b> Perfil equilibrado, QDA ~7.1, EUDR Rastreável.</p>
            <p><b>Mercado:</b> Indústria de transformação com selo verde.</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: VIABILIDADE DO CONSÓRCIO & ECONOMIA FX
# ==============================================================================
with tab2:
    st.markdown("### 📈 Simulador Financeiro do Consórcio & Comparativo de Taxas Cambiais")
    st.write("Simule a viabilidade econômica do consórcio e observe a economia líquida obtida ao utilizar liquidação por Smart Contracts no XRPL em comparação ao câmbio bancário tradicional.")

    # Cálculos Consolidados
    total_consortium_capex = sum([VECTORS_DATA[k]["min_capex_brl"] for k in selected_vectors])
    
    vec_revenues = {}
    vec_profits = {}
    for k in selected_vectors:
        v = VECTORS_DATA[k]
        rev = fresh_fruit_volume * v["revenue_per_ton_fresh"]
        opex = rev * v["opex_ratio"]
        profit = rev - opex
        vec_revenues[k] = rev
        vec_profits[k] = profit

    total_gross_revenue = sum(vec_revenues.values())
    total_net_profit = sum(vec_profits.values())
    
    # Comparativo de Taxas de Câmbio / Intermediação Financeira
    # Câmbio Tradicional Swift / Bancário: ~4.5% de spread + taxas
    # XRPL AMM Smart Contract FX Fee: ~0.2% de taxa total
    traditional_fx_fees = total_gross_revenue * 0.045
    xrpl_fx_fees = total_gross_revenue * 0.002
    fx_net_savings = traditional_fx_fees - xrpl_fx_fees
    
    # Lucro Ajustado com Economia de FX
    adjusted_profit_with_xrpl = total_net_profit + fx_net_savings

    commodity_revenue = dry_bean_volume * 1000 * 15.00
    added_value_multiplier = ((total_gross_revenue / commodity_revenue) - 1) * 100 if commodity_revenue > 0 else 0
    
    payback_years = total_consortium_capex / adjusted_profit_with_xrpl if adjusted_profit_with_xrpl > 0 else 0
    payback_months = payback_years * 12

    # Métricas no Topo
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"<div class='metric-card'><h4>{L['total_capex']}</h4><h2>R$ {total_consortium_capex:,.2f}</h2><p>{len(selected_vectors)} vetores ativos</p></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><h4>{L['total_revenue']}</h4><h2>R$ {total_gross_revenue:,.2f}</h2><p>vs R$ {commodity_revenue:,.2f} commodity</p></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><h4>LUCRO LÍQUIDO + ECONOMIA FX</h4><h2>R$ {adjusted_profit_with_xrpl:,.2f}</h2><p style='color:#2e7d32; font-weight:bold;'>+ R$ {fx_net_savings:,.2f} economizados em FX</p></div>", unsafe_allow_html=True)
    with m4:
        st.markdown(f"<div class='metric-card'><h4>{L['payback_months']}</h4><h2>{payback_months:.1f} Meses</h2><p>Retorno acelerado por Smart Contracts</p></div>", unsafe_allow_html=True)

    st.markdown("---")

    # Destaque da Economia de Taxas Cambiais
    st.markdown(f"#### {L['fx_savings_title']}")
    fx1, fx2, fx3 = st.columns(3)
    with fx1:
        st.markdown(f"""
        <div class='fx-card' style='border-left-color: #d32f2f;'>
            <h4 style='color: #d32f2f;'>🏦 Câmbio Tradicional (Swift / Bancos)</h4>
            <p><b>Taxa Média de Spread/FX:</b> 4.5%</p>
            <p><b>Custo Anual em Taxas:</b> R$ {traditional_fx_fees:,.2f}</p>
            <p><b>Tempo de Liquidação:</b> 15 a 60 dias úteis</p>
        </div>
        """, unsafe_allow_html=True)
    with fx2:
        st.markdown(f"""
        <div class='fx-card'>
            <h4 style='color: #2e7d32;'>⚡ Cacaulogy Smart Contracts (XRPL)</h4>
            <p><b>Taxa Média AMM (XLS-30):</b> 0.2%</p>
            <p><b>Custo Anual em Taxas:</b> R$ {xrpl_fx_fees:,.2f}</p>
            <p><b>Tempo de Liquidação:</b> 3 a 5 segundos</p>
        </div>
        """, unsafe_allow_html=True)
    with fx3:
        st.markdown(f"""
        <div class='fx-card' style='background-color: #fff9c4; border-color: #fbc02d; border-left-color: #f57f17;'>
            <h4 style='color: #f57f17;'>💰 Lucro Extra Revertido ao Consórcio</h4>
            <p><b>Economia Líquida Anual:</b> R$ {fx_net_savings:,.2f}</p>
            <p><b>Aumento de Margem:</b> + 4.3% direto no caixa</p>
            <p><b>Destino:</b> Repasse direto aos produtores rurais</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Gráficos
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        df_rev = pd.DataFrame({
            "Vetor": list(vec_revenues.keys()),
            "Receita Bruta (R$)": list(vec_revenues.values())
        })
        # Usando lista de cores customizada em hex para evitar atributo inexistente em Plotly
        custom_colors = ["#cc5500", "#403228", "#998c82", "#27ae60", "#2980b9", "#8e44ad", "#d35400", "#f39c12"]
        fig_pie = px.pie(df_rev, names="Vetor", values="Receita Bruta (R$)", title="Distribuição da Receita por Vetor do Consórcio", color_discrete_sequence=custom_colors)
        fig_pie.update_layout(paper_bgcolor="white", plot_bgcolor="#faf7f3")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_g2:
        years = [f"Ano {i}" for i in range(1, 6)]
        cum_cash = [-total_consortium_capex]
        for y in range(1, 6):
            cum_cash.append(cum_cash[-1] + adjusted_profit_with_xrpl * (1 + 0.10)**(y-1))
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=["Setup"] + years, y=cum_cash, mode="lines+markers", line=dict(color="#cc5500", width=3)))
        fig_line.add_hline(y=0, line_dash="dash", line_color="green", annotation_text="Payback Point")
        fig_line.update_layout(title="Fluxo de Caixa Acumulado do Consórcio com Smart Contracts (R$)", paper_bgcolor="white", plot_bgcolor="#faf7f3")
        st.plotly_chart(fig_line, use_container_width=True)

    # Tabela DRE
    st.markdown("#### 📄 DRE Detalhado do Consórcio com Economia Cambial")
    dre_rows = []
    for k in selected_vectors:
        v = VECTORS_DATA[k]
        rev = vec_revenues[k]
        prof = vec_profits[k]
        opex = rev - prof
        fx_sav = rev * 0.043 # 4.3% de economia líquida em FX
        adj_prof = prof + fx_sav
        viable_str = "✅ Viável" if fresh_fruit_volume >= v["min_vol_fresh"] else "⚠️ Sub-volume"
        dre_rows.append({
            "Vetor": k,
            "Volume Mínimo (Tons)": v["min_vol_fresh"],
            "Status de Viabilidade": viable_str,
            "CAPEX (R$)": v["min_capex_brl"],
            "Receita Bruta (R$)": rev,
            "OPEX Anual (R$)": opex,
            "Economia FX XRPL (R$)": fx_sav,
            "Lucro Ajustado (R$)": adj_prof
        })
    df_dre = pd.DataFrame(dre_rows)
    st.dataframe(df_dre.style.format({
        "Volume Mínimo (Tons)": "{:,.1f}",
        "CAPEX (R$)": "R$ {:,.2f}",
        "Receita Bruta (R$)": "R$ {:,.2f}",
        "OPEX Anual (R$)": "R$ {:,.2f}",
        "Economia FX XRPL (R$)": "R$ {:,.2f}",
        "Lucro Ajustado (R$)": "R$ {:,.2f}"
    }))

# ==============================================================================
# TAB 3: CONTRATOS INTELIGENTES & CONVERSÃO AUTOMÁTICA (XRPL & CBDC)
# ==============================================================================
with tab3:
    st.markdown("### 🤝 Arquitetura de Contratos Inteligentes, Escrow CUSUM & Conversão Cambial Automática")
    st.write("A plataforma Cacaulogy v16.0 utiliza o **XRP Ledger (XRPL)** como uma camada de evidência agnóstica de moeda (*CBDC-Agnostic Evidence Layer*). Os pagamentos de exportação para a Europa (Euro Digital) e China (e-CNY via Hengqin) são executados via **Smart Contracts de Escrow** condicionados a provas científicas de terroir.")

    st.markdown("#### 🏗️ Arquitetura em 3 Camadas de Interoperabilidade")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='vector-card'>
            <h4>1️⃣ Camada de Evidência & RWA</h4>
            <p>• <b>Geolocalização EUDR:</b> Polígonos WKT de Desmatamento Zero.<br>
            • <b>Genotipagem 96-SNP VCF:</b> Hash SHA3-512 da identidade varietal.<br>
            • <b>IoT Fermentação:</b> Curva térmica de pico de 48,5 °C.<br>
            • <b>Nota QDA & Cádmio:</b> Dossiê GACC digitalizado.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='vector-card'>
            <h4>2️⃣ Camada Ledger (XRPL)</h4>
            <p>• <b>Passaporte RWA XLS-20:</b> Minting de NFT de lote imutável.<br>
            • <b>Contrato Inteligente Escrow:</b> Fundos retidos até verificação de provas.<br>
            • <b>Filtro Estatístico CUSUM:</b> Trava automática de segurança em caso de anomalia ou fraude de mistura.<br>
            • <b>Mensageria ISO 20022:</b> Payloads pacs.008 em tempo real.</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='vector-card'>
            <h4>3️⃣ Camada Monetária (CBDC)</h4>
            <p>• <b>Euro Digital (BCE):</b> Liquidação direta em EUR na Europa.<br>
            • <b>e-CNY (PBoC):</b> Liquidação direta em Yuan Digital na China.<br>
            • <b>Drex / BRL Pix:</b> Conversão automática para Reais no Brasil.<br>
            • <b>Pools AMM XLS-30:</b> Taxas automatizadas de apenas 0,1% a 0,3%.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Simulador Interativo de Conversão Cambial por Smart Contract
    st.markdown("#### 💱 Simulador de Liquidação por Smart Contract & Conversão de Moeda")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        export_val_usd = st.number_input("Valor da Exportação do Lote (USD):", min_value=1000.0, max_value=1000000.0, value=50000.0, step=5000.0)
        target_currency = st.selectbox("Moeda do Comprador Internacional:", ["EUR (Euro Digital)", "CNY (e-CNY / Yuan Digital)", "BRL (Drex / Pix Brasil)"])
        
        # Cotações de Referência
        usd_brl = 5.20
        usd_eur = 0.92
        usd_cny = 7.15
        
        if "EUR" in target_currency:
            converted_amount = export_val_usd * usd_eur
            currency_symbol = "€"
        elif "CNY" in target_currency:
            converted_amount = export_val_usd * usd_cny
            currency_symbol = "¥"
        else:
            converted_amount = export_val_usd * usd_brl
            currency_symbol = "R$"
            
        fee_traditional = export_val_usd * usd_brl * 0.05  # 5% taxa tradicional
        fee_xrpl = export_val_usd * usd_brl * 0.002        # 0.2% taxa XRPL
        net_savings_sim = fee_traditional - fee_xrpl

    with col_s2:
        st.markdown(f"""
        <div class='fx-card'>
            <h4 style='color: #2e7d32;'>📋 Resultado da Liquidação Automática</h4>
            <p>• <b>Valor Bruto no Destino:</b> {currency_symbol} {converted_amount:,.2f}</p>
            <p>• <b>Taxa de Transação XRPL (Gas):</b> ~ 0.00001 XRP (< R$ 0,001)</p>
            <p><b>Taxa de Conversão AMM (0.2%):</b> R$ {fee_xrpl:,.2f}</p>
            <p style='color: #d32f2f;'><b>Custo em Câmbio Bancário Tradicional (5.0%):</b> R$ {fee_traditional:,.2f}</p>
            <hr>
            <h3 style='color: #2e7d32; margin-top: 5px;'>Economia Gerada: R$ {net_savings_sim:,.2f}</h3>
            <p><i>Este valor permanece 100% retido com a cooperativa produtora na Amazônia.</i></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### 🛡️ Como o Filtro CUSUM Protege o Contrato Inteligente contra Fraudes")
    st.info("O filtro estatístico CUSUM (Cumulative Sum, baseado no BIS Working Paper 1374) monitora continuamente os parâmetros do lote. Se uma amêndoa de baixa qualidade for misturada ao lote durante o transporte, a divergência quimiométrica dispara um **Security Freeze** no Smart Contract no XRPL, bloqueando a liberação dos valores retidos em Escrow até que seja realizada uma auditoria presencial.")

# ==============================================================================
# TAB 4: EQUIPAMENTOS & EMPRESAS REGIONAIS (BELÉM/PARÁ)
# ==============================================================================
with tab4:
    st.markdown("### 🛠️ Especificação de Equipamentos & Mapeamento de Parceiros (Região Metropolitana de Belém/PA)")
    st.write("Detalhamento físico e estratégico para formação do consórcio com indústrias, cooperativas e parques tecnológicos locais.")

    for k in selected_vectors:
        v = VECTORS_DATA[k]
        st.markdown(f"#### 📦 {v['name_pt']}")
        
        c_eq, c_part = st.columns(2)
        with c_eq:
            st.markdown("**🔧 Equipamentos Necessários:**")
            for eq in v["equipments"]:
                st.write(f"• {eq}")
        
        with c_part:
            st.markdown("**🏢 Empresas & Instituições Mapeadas em Belém/PA:**")
            for p in v["partners_belem"]:
                st.write(f"• {p}")
        
        st.markdown("---")
