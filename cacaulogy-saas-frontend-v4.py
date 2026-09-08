import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuração visual e de layout profissional
st.set_page_config(
    page_title="Cacaulogy SaaS - Cooperative Batch Aggregator & Financial Intelligence",
    page_icon="🍫",
    layout="wide"
)

# Estilização editorial (Warm Chocolate and Cream)
st.markdown("""
<style>
    .reportview-container { background: #fcfaf7; }
    h1, h2, h3 { color: #403228 !important; font-family: 'Georgia', serif; }
    .stButton>button { background-color: #cc5500; color: white; border-radius: 8px; }
    .metric-card {
        background-color: white; border-left: 5px solid #cc5500;
        padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    .metric-card h4 { margin: 0; color: #998c82; font-size: 11px; text-transform: uppercase; }
    .metric-card h2 { margin: 5px 0; color: #403228; font-size: 24px; font-weight: bold; }
    .metric-card p { margin: 0; color: #5a5a5a; font-size: 12px; }
    .equip-card {
        background-color: #faf7f3; border: 1px solid #e1dbd6;
        padding: 15px; border-radius: 8px; margin-bottom: 12px;
    }
    .equip-card h4 { color: #cc5500; margin-top: 0; }
</style>
""", unsafe_allow_html=True)

# ─── SISTEMA DE TRADUÇÃO MULTILÍNGUE (CACAULOGIA SaaS) ───
TRANSLATIONS = {
    "Português": {
        "title": "🍫 Plataforma de Inteligência Preditiva Cacaulogia",
        "subtitle": "SaaS para Agrupamento de Cooperativas, Padronização de Lotes de Exportação e Viabilidade Financeira",
        "config_panel": "🔧 Painel de Configuração",
        "num_clusters": "Número de Lotes de Exportação Alvo (Clusters)",
        "standardize": "Padronizar Variáveis Bioquímicas (Z-Score)",
        "about_features_title": "Sobre as Variáveis do Terroir",
        "about_features_text": """
*   **Extrato Etéreo (%)**: Teor de manteiga de cacau. Essencial para a fusão, textura e brilho na indústria.
*   **Polifenóis Totais (mg GAE/100g)**: Capacidade antioxidante, associada ao amargor e adstringência natural.
*   **Razão TB/CF**: Marcador quimiotaxonômico. Razão Teobromina/Cafeína para distinguir Criollo (<3), Trinitário (3-9) e Forastero (9-11).
*   **Nota Sensorial (QDA)**: Nota de avaliação sensorial calibrada de 0 a 10.
""",
        "tab_clustering": "📦 Agrupamento de Lotes (Clustering)",
        "tab_finance": "📈 Viabilidade & Projeção Financeira",
        "tab_equipment": "🛠️ Inventário de Equipamentos & Infraestrutura",
        "impact_overview": "📊 Visão Geral do Impacto da Plataforma",
        "metric_vol": "VOLUME AGREGADO",
        "metric_vol_desc": "De 200 microprodutores",
        "metric_comm": "ESTIMATIVA COMMODITY",
        "metric_comm_desc": "A R$ 15,00/kg padrão",
        "metric_saas": "VALORAÇÃO CACAULOGIA",
        "metric_saas_desc": "Lotes premium padronizados",
        "metric_added": "VALOR LÍQUIDO ADICIONADO",
        "metric_added_desc": "+ {:.1f}% multiplicador de valor",
        "chart_title": "🗺️ Mapa Quimiométrico de Terroir & Agrupamento",
        "chart_desc": "Cada ponto representa a colheita de uma família amazônica. O algoritmo de agrupamento os organiza em lotes homogêneos e padronizados para exportação comercial.",
        "polyphenols_axis": "Polifenóis Totais (mg GAE/100g) - [Dim II]",
        "sensory_axis": "Nota Sensorial (QDA) - [Dim IV]",
        "lot_category": "Categoria de Lote Padronizado",
        "lot_spec_title": "📦 Especificação Técnica dos Lotes de Exportação",
        "lot_explain_title": "🔍 Entendendo os Lotes de Exportação Cacaulogia",
        "lot_explain_desc": """
Abaixo, explicamos detalhadamente o perfil científico e comercial de cada lote agrupado:

*   **Lote A: Amazonian Floral-Fruity Premium (Premium Floral e Frutado da Amazônia):**
    *   *O que é:* Lotes com perfil de sabor excepcional (nota superior a 8,0/10) e alto teor de manteiga natural (>55%). Possui baixos polifenóis totais, reduzindo o amargor e ressaltando notas florais e de mel.
    *   *Genética:* Linhagens modernas de Criollo e Trinitário.
    *   *Destino:* Chocolates finos artesanais *bean-to-bar* premium e confeitaria gourmet europeia e asiática.
    *   *Preço de Venda:* **R$ 58,00/kg**.
*   **Lote B: Deep Forest Antioxidant (Antioxidante da Floresta Profunda):**
    *   *O que é:* Lotes com altíssimo poder funcional e antioxidante (polifenóis >10.000 mg GAE/100g) e alta razão Teobromina/Cafeína. Apresenta notas acentuadas de amargor natural e adstringência rica.
    *   *Genética:* Linhagens clássicas de Forastero (Amelonado).
    *   *Destino:* Indústria de cosméticos de luxo, nutracêuticos e chocolates funcionais de alto cacau (>80%).
    *   *Preço de Venda:* **R$ 38,00/kg**.
*   **Lote C: Traditional Balanced Trinitario (Trinitário Tradicional Equilibrado):**
    *   *O que é:* Lote com perfil equilibrado e clássico (nota sensorial próxima a 7,1/10). Excelente consistência de fusão industrial.
    *   *Genética:* Híbridos tradicionais Trinitários.
    *   *Destino:* Indústrias europeias e chinesas de chocolate fino em larga escala, agregando rastreabilidade e certificação de Desmatamento Zero (EUDR).
    *   *Preço de Venda:* **R$ 32,00/kg**.
""",
        "lot_summary_headers": {
            "Export_Lot_Category": "Categoria do Lote de Exportação",
            "Producers": "Produtores (Famílias)",
            "Total_Volume_kg": "Volume Total (kg)",
            "Mean_Sensory": "Nota Sensorial Média",
            "Mean_Polyphenols": "Polifenóis Médios",
            "Mean_Lipids": "Gordura Média (EE%)",
            "SaaS_Price": "Preço SaaS (R$/kg)",
            "Net_Added_Value": "Valor Líquido Adicionado (R$)"
        },
        "micro_allocations_title": "👪 Alocação Detalhada por Família Produtora",
        "micro_allocations_desc": "Baixe esta lista processada em formato Excel para alimentar portais alfandegários ou ERPs logísticos (EUDR/GACC).",
        "saas_price_col": "Preço SaaS (R$/kg)",
        "saas_total_col": "Valor Total SaaS (R$)",
        "net_added_col": "Valor Adicionado Líquido (R$)",
        
        # Finance Tab
        "finance_title": "📈 Simulador de Viabilidade e Prospecção Financeira (5 Anos)",
        "finance_desc": "Simule em tempo real os fluxos de caixa, custos de investimento (CAPEX) e retornos operacionais (ROI) gerados pelo framework Cacaulogia.",
        "op_params": "⚙️ Parâmetros Operacionais da Cooperativa / Indústria",
        "annual_volume": "Volume Anual Inicial de Amêndoas Processadas (Toneladas)",
        "growth_rate_lbl": "Crescimento Anual de Volume Esperado (%)",
        "comm_base_price_lbl": "Preço Base de Compra da Commodity (R$/kg)",
        "costs_structure": "💰 Estrutura de Custos de Implantação e Operação",
        "capex_title": "Investimentos de Capital Iniciais (CAPEX)",
        "opex_title": "Custos Operacionais Anuais (OPEX)",
        "capex_iot_lbl": "Sensores IoT para Cochos (R$)",
        "capex_dryers_lbl": "Automação de Estufas de Secagem (R$)",
        "capex_lab_lbl": "Equipamentos Analíticos de Laboratório (R$)",
        "capex_setup_lbl": "Treinamento & Instalação de TI (R$)",
        "capex_eudr_lbl": "Mapeamento Ambiental & EUDR (R$)",
        "capex_pkg_lbl": "Design de QR Codes & Embalagem (R$)",
        "total_capex_lbl": "Total CAPEX Inicial",
        "opex_saas_lbl": "Assinatura Anual do Software SaaS (R$)",
        "opex_maint_lbl": "Manutenção Física de Sensores (R$)",
        "opex_logistics_lbl": "Logística de Técnicos de Campo (R$)",
        "opex_audits_lbl": "Auditorias & Certificações Externas (R$)",
        "total_opex_lbl": "Total OPEX Anual",
        "projection_chart_title": "📊 Projeção Financeira & Margens Operacionais",
        "chart_rev_title": "Comparativo de Receita Bruta Anual (R$)",
        "chart_rev_comm": "Receita Tradicional (Commodity)",
        "chart_rev_saas": "Receita Cacaulogia SaaS",
        "chart_cash_title": "Fluxo de Caixa Líquido Acumulado (R$)",
        "chart_cash_flow": "Fluxo de Caixa Acumulado",
        "payback_lbl": "Ponto de Equilíbrio (Payback)",
        "dre_title": "📄 Demonstrativo de Resultados do Exercício Projetado (DRE)",
        "dre_cols": {
            "Ano": "Ano",
            "Volume Processed (kg)": "Volume Processado (kg)",
            "Receita Commodity (R$)": "Receita Commodity (R$)",
            "Receita Cacaulogia (R$)": "Receita Cacaulogia (R$)",
            "Custo Matéria-Prima Justa (R$)": "Custo Matéria-Prima Justa (R$)",
            "OPEX (R$)": "OPEX (R$)",
            "Lucro Líquido Retido (R$)": "Lucro Líquido Retido (R$)",
            "Fluxo de Caixa Acumulado (R$)": "Fluxo de Caixa Acumulado (R$)"
        },
        "dre_summary_text": """
*   **Prêmio Médio de Preço:** O preço médio ponderado gerado pelo portfólio de lotes Cacaulogia é de **R$ {:.2f}/kg** (contra R$ {:.2f}/kg da commodity).
*   **Impacto Social Compartilhado:** O Custo de Matéria-Prima Justa repassa um prêmio social direto de mais de **R$ {:,.2f} acumulados em 5 anos** para as famílias associadas, blindando a floresta e incentivando o vetor humano \\(K\\).
""",
        
        # Equipment Tab
        "equip_title": "🛠️ Ativos Físicos e Maquinários Requeridos em Campo",
        "equip_desc": "Para operacionalizar o rigor metodológico 'da narrativa à evidência' da Cacaulogia, a cooperativa ou indústria deve adquirir os seguintes ativos complementares além da plataforma digital:",
        "eq_section1": "📡 1. Infraestrutura IoT & Pós-Colheita (Dimensões I & III)",
        "eq_section1_list": """
*   **Sondas Térmicas DS18B20 Seladas:** Inseridas diretamente na massa de cacau fermentando nos cochos para traçar as curvas cinéticas de fermentação em tempo real.
*   **Microcontroladores ESP32 de Baixo Custo:** Sondas com caixas protetoras estanques que transmitem dados via rede local ou salvam de forma offline assíncrona.
*   **Sensores de Estufa (DHT22 / SHT31):** Instalados em estufas solares de secagem para registrar umidade relativa e temperatura interna do ar.
*   **Exaustores Automatizados:** Ativados automaticamente para exaurir a umidade do ar da estufa quando esta ultrapassa o ponto crítico de evaporação ácida.
""",
        "eq_section2": "🔬 2. Equipamentos Analíticos & Químicos de Laboratório (Dimensão II)",
        "eq_section2_list": """
*   **Liofilizador de Bancada (Lyophilizer):** Desidrata a frio cotilédones frescos (a -50 ºC e 0.04 mbar) preservando polifenóis ativos e alcaloides de degradações térmicas antes das análises.
*   **Balança Analítica de Precisão (0.1 mg):** Para pesagem de frações finas de amêndoas e controle quantitativo de reagentes.
*   **Moinho Analítico de Facas:** Tritura uniformemente amêndoas liofilizadas até a malha de 40 mesh (600 a 420 µm).
*   **Banho de Ultrassom Térmico:** Para extração assistida por ultrassom (25 Hz a 50 ºC) de flavonoides, teobromina e cafeína ativos.
*   **Centrífuga de Alta Velocidade (14.000 rpm):** Para separação de fases e purificação rápida dos extratos analíticos.
*   **Leitor de Microplacas (Espectrofotômetro):** Quantifica Polifenóis Totais (Folin-Ciocalteu a 760 nm) e Antocianinas (razão de absorbância a 460/530 nm).
""",
        "eq_section3": "👅 3. Estrutura para Análise Sensorial Calibrada (Dimensão IV)",
        "eq_section3_list": """
*   **Cabines de Degustação Padronizadas (ISO 8586):** Cabines isoladas termicamente (24 ± 0.2 ºC), com controle de ruído e umidade de 53% a 60% para neutralidade dos provadores.
*   **Kit de Referências Olfativas Comerciais:** Compostos puros de Ácido Isovalérico, Ácido Capróico, Álcool Isoamílico e Benzaldeído para calibração de QDA e detecção de defeitos.
""",
        "eq_partnerships": """
> **Parceria Tecnológica Regional:** A aquisição dos equipamentos laboratoriais analíticos de alta tecnologia pode ser barateada ou substituída através de **convênios de cooperação de pesquisa** com laboratórios públicos das universidades na Amazônia (UFPA, IFPA, UFRA, CEPLAC). A cooperativa utiliza sensores de IoT locais e envia amostras quinzenais para análise varietal nos laboratórios parceiros, mantendo o OPEX sob rígido controle.
""",
        "target_calib_title": "🎯 Alvos Quimiométricos de Qualidade",
        "target_calib_desc": "Os equipamentos analíticos descritos são calibrados para mapear o cacau dentro das faixas ótimas científicas:",
        "target_labels": ["Manteiga (%)", "Polifenóis Máximos", "Nota Sensorial QDA", "Acidez Volátil (pH)"],
        "target_suffix": "% do Alvo Ideal"
    },
    "English": {
        "title": "🍫 Cacaulogy Predictive Intelligence Platform",
        "subtitle": "SaaS for Cooperative Aggregation, Export Lot Standardization & Financial Viability Modeler",
        "config_panel": "🔧 Configuration Panel",
        "num_clusters": "Number of Target Export Lots (Clusters)",
        "standardize": "Standardize Biochemical Features (Z-Score)",
        "about_features_title": "About the Terroir Features",
        "about_features_text": """
*   **Ethereal Extract (%)**: Cocoa butter content. Essential for industrial melting, texture, and gloss.
*   **Total Polyphenols (mg GAE/100g)**: Antioxidant capacity, strongly related to bitterness and natural astringency.
*   **TB/CF Ratio**: Chemotaxonomic marker. Theobromine to Caffeine ratio to distinguish Criollo (<3), Trinitario (3-9), and Forastero (9-11).
*   **Sensory Score (QDA)**: Calibrated organoleptic evaluation score from 0 to 10.
""",
        "tab_clustering": "📦 Batch Aggregation (Clustering)",
        "tab_finance": "📈 Economic Viability & Financial Projections",
        "tab_equipment": "🛠️ Equipment Inventory & Infrastructure Guide",
        "impact_overview": "📊 Platform Impact Overview",
        "metric_vol": "TOTAL VOLUME AGGREGATED",
        "metric_vol_desc": "From 200 smallholders",
        "metric_comm": "COMMODITY ESTIMATIVE",
        "metric_comm_desc": "At standard R$ 15.00/kg rate",
        "metric_saas": "CACAULOGY SaaS VALUATION",
        "metric_saas_desc": "Standardized premium lots",
        "metric_added": "NET ADDED VALUE CREATED",
        "metric_added_desc": "+ {:.1f}% value multiplier",
        "chart_title": "🗺️ Terroir Chemometric Map & Clustering",
        "chart_desc": "Each node represents an individual Amazonian family's harvest. The clustering engine groups them into standardized and homogeneous export lots.",
        "polyphenols_axis": "Total Polyphenols (mg GAE/100g) - [Dim II]",
        "sensory_axis": "Sensory Score (QDA) - [Dim IV]",
        "lot_category": "Standardized Export Category",
        "lot_spec_title": "📦 Technical Specifications of Export Lots",
        "lot_explain_title": "🔍 Understanding the Cacaulogy Export Lots",
        "lot_explain_desc": """
Below we detail the scientific and commercial profile of each grouped lot:

*   **Lot A: Amazonian Floral-Fruity Premium:**
    *   *What it is:* Lots with exceptional flavor profile (score above 8.0/10) and high natural fat content (>55%). It has low total polyphenols, which reduces bitterness and highlights floral and honey notes.
    *   *Genetics:* Modern Criollo and Trinitario lineages.
    *   *Destination:* Premium bean-to-bar chocolates and European and Asian gourmet confectionery.
    *   *Selling Price:* **R$ 58.00/kg**.
*   **Lot B: Deep Forest Antioxidant:**
    *   *What it is:* Lots with extremely high functional and antioxidant power (polyphenols >10,000 mg GAE/100g) and high Theobromine/Caffeine ratio. Strong natural bitterness and rich astringency.
    *   *Genetics:* Traditional Forastero (Amelonado) lineages.
    *   *Destination:* Luxury cosmetics, nutraceuticals, and high-cocoa functional chocolates (>80%).
    *   *Selling Price:* **R$ 38.00/kg**.
*   **Lot C: Traditional Balanced Trinitario:**
    *   *What it is:* Lots with balanced and classic profiles (sensory score close to 7.1/10). Excellent consistency of industrial melting.
    *   *Genetics:* Traditional Trinitario hybrids.
    *   *Destination:* Large-scale fine chocolate industries in Europe and China, adding EUDR Deforestation-Free compliance and end-to-end traceability.
    *   *Selling Price:* **R$ 32.00/kg**.
""",
        "lot_summary_headers": {
            "Export_Lot_Category": "Export Lot Category",
            "Producers": "Producers (Families)",
            "Total_Volume_kg": "Total Volume (kg)",
            "Mean_Sensory": "Mean Sensory Score",
            "Mean_Polyphenols": "Mean Polyphenols",
            "Mean_Lipids": "Mean Fat (EE%)",
            "SaaS_Price": "SaaS Price (R$/kg)",
            "Net_Added_Value": "Net Added Value (R$)"
        },
        "micro_allocations_title": "👪 Detailed Allocations by Smallholder Family",
        "micro_allocations_desc": "Download this processed list in Excel format to feed customs portals or logistical ERPs (EUDR/GACC).",
        "saas_price_col": "SaaS Price (R$/kg)",
        "saas_total_col": "SaaS Total Value (R$)",
        "net_added_col": "Net Added Value (R$)",
        
        # Finance Tab
        "finance_title": "📈 Viability & Financial Projections (5 Years)",
        "finance_desc": "Simulate in real-time cash flows, capital expenditures (CAPEX), and operational returns (ROI) generated by the Cacaulogy framework.",
        "op_params": "⚙️ Operational Parameters of the Cooperative / Industry",
        "annual_volume": "Initial Annual Volume of Processed Cocoa Beans (Tons)",
        "growth_rate_lbl": "Expected Annual Volume Growth Rate (%)",
        "comm_base_price_lbl": "Commodity Purchase Base Price (R$/kg)",
        "costs_structure": "💰 Capital & Operational Costs Structure",
        "capex_title": "Initial Capital Expenditures (CAPEX)",
        "opex_title": "Annual Operational Expenditures (OPEX)",
        "capex_iot_lbl": "IoT Sonda Sensors for Cochos (R$)",
        "capex_dryers_lbl": "Drying Greenhouse Automation (R$)",
        "capex_lab_lbl": "Analytical Laboratory Equipment (R$)",
        "capex_setup_lbl": "IT Setup & Training (R$)",
        "capex_eudr_lbl": "Environmental Mapping & EUDR (R$)",
        "capex_pkg_lbl": "Design of Packaging & QR Codes (R$)",
        "total_capex_lbl": "Total Initial CAPEX",
        "opex_saas_lbl": "Annual SaaS Software Subscription (R$)",
        "opex_maint_lbl": "Sonda Physical Maintenance (R$)",
        "opex_logistics_lbl": "Field Technicians Logistics (R$)",
        "opex_audits_lbl": "External Audits & Certifications (R$)",
        "total_opex_lbl": "Total Annual OPEX",
        "projection_chart_title": "📊 Financial Projections & Operating Margins",
        "chart_rev_title": "Annual Gross Revenue Comparison (R$)",
        "chart_rev_comm": "Traditional Revenue (Commodity)",
        "chart_rev_saas": "Cacaulogy SaaS Revenue",
        "chart_cash_title": "Cumulative Net Cash Flow (R$)",
        "chart_cash_flow": "Cumulative Cash Flow",
        "payback_lbl": "Break-Even Point (Payback)",
        "dre_title": "📄 Projected Statement of Income (DRE)",
        "dre_cols": {
            "Ano": "Year",
            "Volume Processed (kg)": "Processed Volume (kg)",
            "Receita Commodity (R$)": "Commodity Revenue (R$)",
            "Receita Cacaulogia (R$)": "Cacaulogy Revenue (R$)",
            "Custo Matéria-Prima Justa (R$)": "Fair Raw Material Cost (R$)",
            "OPEX (R$)": "OPEX (R$)",
            "Lucro Líquido Retido (R$)": "Retained Net Profit (R$)",
            "Fluxo de Caixa Acumulado (R$)": "Cumulative Cash Flow (R$)"
        },
        "dre_summary_text": """
*   **Average Premium Price:** The weighted average price generated by the Cacaulogy portfolio is **R$ {:.2f}/kg** (vs. R$ {:.2f}/kg for standard commodity).
*   **Shared Social Impact:** The Fair Raw Material Cost repasses a direct social premium of over **R$ {:,.2f} accumulated over 5 years** to the associated families, protecting the forest and incentivizing the human vector \\(K\\).
""",
        
        # Equipment Tab
        "equip_title": "🛠️ Required Physical Assets & Hardware in the Field",
        "equip_desc": "To operationalize the scientific rigor of Cacaulogy 'from narrative to evidence', the cooperative or industry must acquire the following complementary assets besides the software:",
        "eq_section1": "📡 1. IoT & Post-Harvest Infrastructure (Dimensions I & III)",
        "eq_section1_list": """
*   **DS18B20 Sealed Thermal Probes:** Inserted directly into the fermenting cocoa mass in the cochos to trace kinetic temperature curves in real time.
*   **Low-Cost ESP32 Microcontrollers:** Probes with waterproof protective boxes that transmit via local network or save in offline asynchronous mode.
*   **Greenhouse Sensors (DHT22 / SHT31):** Installed in solar drying greenhouses to log relative humidity and internal air temperature.
*   **Automated Exhaust Fans:** Activated automatically to exhaust air humidity from the greenhouse when it exceeds the critical acid evaporation threshold.
""",
        "eq_section2": "🔬 2. Analytical & Chemical Laboratory Equipment (Dimension II)",
        "eq_section2_list": """
*   **Benchtop Lyophilizer (Freeze Dryer):** Cold-dehydrates fresh cotyledons (at -50 ºC and 0.04 mbar) to preserve active polyphenols and alkaloids from thermal degradation before analysis.
*   **Analytical Precision Balance (0.1 mg):** Essential for precise weighing of bean fractions and quantitative control of reagents.
*   **Analytical Knife Mill:** Uniformly grinds lyophilized beans to a 40 mesh grain size (600 to 420 µm).
*   **Thermal Ultrasonic Bath:** For ultrasound-assisted extraction (25 Hz at 50 ºC) of active flavonoids, theobromine, and caffeine.
*   **High-Speed Centrifuge (14,000 rpm):** For fast phase separation and purification of analytical extracts.
*   **Microplate Reader (Spectrophotometer):** Quantifies Total Polyphenols (Folin-Ciocalteu at 760 nm) and Anthocyanins (absorbance ratio at 460/530 nm).
""",
        "eq_section3": "👅 3. Structure for Calibrated Sensory Analysis (Dimension IV)",
        "eq_section3_list": """
*   **Standardized Tasting Booths (ISO 8586):** Thermally isolated booths (24 ± 0.2 ºC), with noise and relative humidity control (53% to 60%) to ensure panel neutrality.
*   **Commercial Olfactory References Kit:** Pure compounds of Isovaleric Acid, Caproic Acid, Isoamyl Alcohol, and Benzaldehyde for QDA calibration and defect detection.
""",
        "eq_partnerships": """
> **Regional Technology Partnership:** The acquisition of high-tech analytical laboratory equipment can be cheapened or replaced through **research cooperation agreements** with public laboratories of universities in the Amazon (UFPA, IFPA, UFRA, CEPLAC). The cooperative uses local IoT sensors and sends bi-weekly samples for varietal analysis in partner labs, keeping OPEX under tight control.
""",
        "target_calib_title": "🎯 Quality Chemometric Targets",
        "target_calib_desc": "The analytical equipment is calibrated to map cocoa within optimal scientific ranges:",
        "target_labels": ["Cocoa Butter (%)", "Max Polyphenols", "Sensory Score QDA", "Volatile Acidity (pH)"],
        "target_suffix": "% of Ideal Target"
    },
    "中文": {
        "title": "🍫 可可学预测智能平台",
        "subtitle": "用于合作社整合、出口批次标准化和财务可行性建模的SaaS平台",
        "config_panel": "🔧 配置面板",
        "num_clusters": "目标出口批次数量 (聚类数)",
        "standardize": "标准化生化特征 (Z-Score)",
        "about_features_title": "关于可可风土特征",
        "about_features_text": """
*   **醚提取物 (%)**: 可可脂含量。对于工业熔融、质地和光泽至关重要。
*   **总多酚 (mg GAE/100g)**: 抗氧化能力，与苦味和天然收敛性强相关。
*   **TB/CF 比率**: 化学分类学标记。可可碱与咖啡因的比率，用以区分克里奥罗 (<3)、特里尼达 (3-9) 和外来种 (9-11)。
*   **感官评分 (QDA)**: 校准的感官评估得分 (0 到 10)。
""",
        "tab_clustering": "📦 批次聚合 (聚类引擎)",
        "tab_finance": "📈 可行性与财务预测",
        "tab_equipment": "🛠️ 设备清单与基础设施指南",
        "impact_overview": "📊 平台影响概述",
        "metric_vol": "总聚合量",
        "metric_vol_desc": "来自200户小农",
        "metric_comm": "大宗商品估值",
        "metric_comm_desc": "按标准 15.00 雷亚尔/公斤计算",
        "metric_saas": "可可学SaaS估值",
        "metric_saas_desc": "标准化溢价批次",
        "metric_added": "创造的净附加值",
        "metric_added_desc": "+ {:.1f}% 价值乘数",
        "chart_title": "🗺️ 风土化学计量图与聚类",
        "chart_desc": "每个节点代表一个亚马逊家庭的收获。聚类引擎将其组织成标准化且均质的出口批次。",
        "polyphenols_axis": "总多酚 (mg GAE/100g) - [维度 II]",
        "sensory_axis": "感官评分 (QDA) - [维度 IV]",
        "lot_category": "标准化出口类别",
        "lot_spec_title": "📦 出口批次技术规范",
        "lot_explain_title": "🔍 了解可可学出口批次",
        "lot_explain_desc": """
下面我们详细介绍每个聚类批次的科学和商业特征：

*   **批次 A：亚马逊花果香溢价级 (Amazonian Floral-Fruity Premium):**
    *   *定义:* 具有卓越风味（得分高于 8.0/10）和高天然脂肪含量（>55%）的批次。其总多酚含量较低，减少了苦味并突出了花香和蜂蜜味。
    *   *遗传学:* 现代克里奥罗 (Criollo) 和特里尼达 (Trinitario) 谱系。
    *   *用途:* 溢价 bean-to-bar 巧克力以及欧洲 and 亚洲的高档糕点。
    *   *售价:* **58.00 雷亚尔/公斤**。
*   **批次 B：深林抗氧化级 (Deep Forest Antioxidant):**
    *   *定义:* 具有极高功能和抗氧化能力（多酚 >10,000 mg GAE/100g）以及高可可碱/咖啡因比率的批次。具有强烈的天然苦味和丰富的收敛性。
    *   *遗传学:* 传统外来种 (Forastero/Amelonado) 谱系。
    *   *用途:* 奢侈化妆品、保健品和高可可含量功能性巧克力（>80%）。
    *   *售价:* **38.00 雷亚尔/公斤**。
*   **批次 C：传统平衡特里尼达级 (Traditional Balanced Trinitario):**
    *   *定义:* 具有平衡和经典风味（感官评分接近 7.1/10）的批次。工业熔融一致性极佳。
    *   *遗传学:* 传统特里尼达杂交种。
    *   *用途:* 欧洲和中国的大规模精细巧克力产业，融入了欧盟反毁林法规 (EUDR) 的合规性和端到端追溯性。
    *   *售价:* **32.00 雷亚尔/公斤**。
""",
        "lot_summary_headers": {
            "Export_Lot_Category": "出口批次类别",
            "Producers": "生产者（家庭数）",
            "Total_Volume_kg": "总成交量 (kg)",
            "Mean_Sensory": "平均感官评分",
            "Mean_Polyphenols": "平均总多酚",
            "Mean_Lipids": "平均脂肪含量 (EE%)",
            "SaaS_Price": "SaaS 价格 (雷亚尔/公斤)",
            "Net_Added_Value": "净附加值 (雷亚尔)"
        },
        "micro_allocations_title": "👪 小农家庭详细分配",
        "micro_allocations_desc": "下载此 Excel 格式的处理后列表，以导入海关门户或物流 ERP（EUDR/GACC）。",
        "saas_price_col": "SaaS 价格 (雷亚尔/公斤)",
        "saas_total_col": "SaaS 总价值 (雷亚尔)",
        "net_added_col": "净附加值 (雷亚尔)",
        
        # Finance Tab
        "finance_title": "📈 财务可行性与5年效益预测模拟器",
        "finance_desc": "实时模拟由可可学 (Cacaulogy) 框架产生的现金流、资本支出 (CAPEX) 和运营回报 (ROI)。",
        "op_params": "⚙️ 合作社/加工厂的运营参数",
        "annual_volume": "初始年度可可豆加工量（吨）",
        "growth_rate_lbl": "预期年度产量增长率 (%)",
        "comm_base_price_lbl": "大宗商品采购基准价（雷亚尔/公斤）",
        "costs_structure": "💰 资本与运营成本结构",
        "capex_title": "初始资本支出 (CAPEX)",
        "opex_title": "年度运营支出 (OPEX)",
        "capex_iot_lbl": "发酵槽 IoT 传感器探针 (R$)",
        "capex_dryers_lbl": "干燥温室自动化系统 (R$)",
        "capex_lab_lbl": "实验室分析设备 (R$)",
        "capex_setup_lbl": "IT 部署与技术培训 (R$)",
        "capex_eudr_lbl": "环境制图与欧盟 EUDR 合规 (R$)",
        "capex_pkg_lbl": "二维码设计与包装升级 (R$)",
        "total_capex_lbl": "总初始资本支出",
        "opex_saas_lbl": "SaaS 软件年度订阅费 (R$)",
        "opex_maint_lbl": "物联网传感器物理维护 (R$)",
        "opex_logistics_lbl": "农技人员实地物流费用 (R$)",
        "opex_audits_lbl": "外部审计与认证费用 (R$)",
        "total_opex_lbl": "总年度运营支出",
        "projection_chart_title": "📊 财务预测与运营利润率",
        "chart_rev_title": "年度总收入对比 (雷亚尔)",
        "chart_rev_comm": "传统模式收入 (大宗商品)",
        "chart_rev_saas": "可可学 SaaS 模式收入",
        "chart_cash_title": "累计净现金流 (雷亚尔)",
        "chart_cash_flow": "累计现金流",
        "payback_lbl": "盈亏平衡点 (Payback)",
        "dre_title": "📄 5年财务损益预测表 (DRE)",
        "dre_cols": {
            "Ano": "年度",
            "Volume Processed (kg)": "加工量 (kg)",
            "Receita Commodity (R$)": "商品模式收入 (雷亚尔)",
            "Receita Cacaulogia (R$)": "可可学模式收入 (雷亚尔)",
            "Custo Matéria-Prima Justa (R$)": "公平原料采购成本 (雷亚尔)",
            "OPEX (R$)": "运营支出 (雷亚尔)",
            "Lucro Líquido Retido (R$)": "留存净利润 (雷亚尔)",
            "Fluxo de Caixa Acumulado (R$)": "累计现金流 (雷亚尔)"
        },
        "dre_summary_text": """
*   **平均溢价:** 可可学投资组合生成的加权平均价格为 **{:.2f} 雷亚尔/公斤**（大宗商品标准价为 {:.2f} 雷亚尔/公斤）。
*   **共享社会影响:** 公平原料采购成本在 **5年内累计向合作农户回馈超过 {:,.2f} 雷亚尔** 的社会溢价，切实保护森林并激励人类要素 \\(K\\).
""",
        
        # Equipment Tab
        "equip_title": "🛠️ 现场所需的物理资产与硬件设备清单",
        "equip_desc": "为了将可可学“从叙事转化为证据”的科学严谨性付诸实践，合作社或加工厂除数字平台外，还必须采购以下互补资产：",
        "eq_section1": "📡 1. 物联网与后收获基础设施（维度 I & III）",
        "eq_section1_list": """
*   **DS18B20 密封式温度探针:** 直接插入发酵槽中发酵的可可质量中，实时追踪发酵动力学温度曲线。
*   **低成本 ESP32 微控制器:** 带有防水保护盒的探针，可通过本地网络传输或在离线异步模式下保存。
*   **干燥温室传感器 (DHT22 / SHT31):** 安装在太阳能干燥温室中，记录相对湿度和内部空气温度。
*   **自动排气扇:** 当温室空气湿度超过临界酸挥发阈值时，自动启动排出水分。
""",
        "eq_section2": "🔬 2. 实验室分析与化学检测设备（维度 II）",
        "eq_section2_list": """
*   **台式冻干机 (Lyophilizer):** 冷冻脱水新鲜子叶（在 -50 ºC 和 0.04 mbar 下），在分析前保护活性多酚和生物碱免受热降解。
*   **分析天平 (0.1 mg):** 用于精准称量可可豆级分和试剂的定量控制。
*   **分析型刀磨机:** 将冻干豆均匀研磨至 40 目粒度（600 至 420 µm）。
*   **恒温超声波清洗器:** 用于超声辅助提取（25 Hz，50 ºC）活性黄酮类化合物、可可碱和咖啡因。
*   **高速离心机 (14,000 rpm):** 用于分析提取物的快速相分离和澄清。
*   **酶标仪 (分光光度计):** 定量总多酚（Folin-Ciocalteu 法，760 nm）和花青素（460/530 nm 处的吸光度比）。
""",
        "eq_section3": "👅 3. 校准感官分析室（维度 IV）",
        "eq_section3_list": """
*   **标准化感官品评室 (ISO 8586):** 隔热品评室（24 ± 0.2 ºC），具有噪声和相对湿度控制（53% 至 60%），确保品评人员的客观性。
*   **商用嗅觉标准品套件:** 异戊酸、己酸、异戊醇和苯甲醛的纯化合物，用于 QDA 感官校准和缺陷检测。
""",
        "eq_partnerships": """
> **区域技术伙伴关系:** 采购高科技分析实验室设备可以通过与亚马逊地区大学（UFPA、IFPA、UFRA、CEPLAC）的公共实验室签署**研究合作协议**来降低成本或进行替代。合作社利用本地物联网传感器，并将每两周的样品送往合作实验室进行品种分析，从而将运营支出控制在极低水平。
""",
        "target_calib_title": "🎯 生物计量质量目标",
        "target_calib_desc": "分析设备均经过校准，可将可可映射在最佳科学范围内：",
        "target_labels": ["可可脂含量 (%)", "最大多酚值", "QDA 感官得分", "挥发性酸度 (pH)"],
        "target_suffix": "% 的理想目标值"
    },
    "Español": {
        "title": "🍫 Plataforma de Inteligencia Predictiva Cacaulogy",
        "subtitle": "SaaS para Agrupación de Cooperativas, Estandarización de Lotes de Exportación y Modelado de Viabilidad Financiera",
        "config_panel": "🔧 Panel de Configuración",
        "num_clusters": "Número de Lotes de Exportación Objetivo (Clusters)",
        "standardize": "Estandarizar Variables Bioquímicas (Z-Score)",
        "about_features_title": "Sobre las Variables del Terroir",
        "about_features_text": """
*   **Extracto Etéreo (%)**: Contenido de manteca de cacao. Esencial para la fusión, textura y brillo en la industria.
*   **Polifenoles Totales (mg GAE/100g)**: Capacidad antioxidante, asociada al amargor y la astringencia natural.
*   **Relación TB/CF**: Marcador quimiotaxonómico. Relación Teobromina/Cafeína para distinguir Criollo (<3), Trinitario (3-9) y Forastero (9-11).
*   **Nota Sensorial (QDA)**: Nota de evaluación sensorial calibrada de 0 a 10.
""",
        "tab_clustering": "📦 Agrupación de Lotes (Clustering)",
        "tab_finance": "📈 Viabilidad y Proyección Financiera",
        "tab_equipment": "🛠️ Inventario de Equipos e Infraestructura",
        "impact_overview": "📊 Resumen de Impacto de la Plataforma",
        "metric_vol": "VOLUMEN AGREGADO",
        "metric_vol_desc": "De 200 microproductores",
        "metric_comm": "ESTIMACIÓN COMMODITY",
        "metric_comm_desc": "A tasa estándar de R$ 15,00/kg",
        "metric_saas": "VALORACIÓN CACAULOGY",
        "metric_saas_desc": "Lotes premium estandarizados",
        "metric_added": "VALOR NETO AGREGADO",
        "metric_added_desc": "+ {:.1f}% multiplicador de valor",
        "chart_title": "🗺️ Mapa Quimiométrico del Terroir y Agrupación",
        "chart_desc": "Cada punto representa la cosecha de una familia amazónica. El algoritmo los organiza en lotes homogéneos y estandarizados para exportación comercial.",
        "polyphenols_axis": "Polifenoles Totales (mg GAE/100g) - [Dim II]",
        "sensory_axis": "Nota Sensorial (QDA) - [Dim IV]",
        "lot_category": "Categoría de Lote Estandarizado",
        "lot_spec_title": "📦 Especificaciones Técnicas de los Lotes de Exportación",
        "lot_explain_title": "🔍 Entendiendo los Lotes de Exportación Cacaulogy",
        "lot_explain_desc": """
A continuación, explicamos en detalle el perfil científico y comercial de cada lote agrupado:

*   **Lote A: Amazonian Floral-Fruity Premium:**
    *   *Qué es:* Lotes con un perfil de sabor excepcional (nota superior a 8,0/10) y alto contenido de manteca natural (>55%). Posee bajos polifenoles totales, lo que reduce el amargor y resalta notas florales y de miel.
    *   *Genética:* Linajes modernos de Criollo y Trinitario.
    *   *Destino:* Chocolates finos artesanales *bean-to-bar* premium y repostería gourmet en Europa y Asia.
    *   *Precio de Venta:* **R$ 58,00/kg**.
*   **Lote B: Deep Forest Antioxidant:**
    *   *Qué es:* Lotes con un altísimo poder funcional y antioxidante (polifenoles >10.000 mg GAE/100g) y alta relación Teobromina/Cafeína. Notas pronunciadas de amargor natural y astringencia rica.
    *   *Genética:* Linajes clásicos de Forastero (Amelonado).
    *   *Destino:* Industria de cosméticos de lujo, nutracéuticos y chocolates funcionales con alto contenido de cacao (>80%).
    *   *Precio de Venta:* **R$ 38,00/kg**.
*   **Lote C: Traditional Balanced Trinitario:**
    *   *Qué es:* Lote con perfil equilibrado y clásico (nota sensorial cercana a 7,1/10). Excelente consistencia de fusión industrial.
    *   *Genética:* Híbridos tradicionales Trinitarios.
    *   *Destino:* Industrias europeas y chinas de chocolate fino a gran escala, agregando trazabilidad y certificación de Deforestación Cero (EUDR).
    *   *Precio de Venda:* **R$ 32,00/kg**.
""",
        "lot_summary_headers": {
            "Export_Lot_Category": "Categoría de Lote de Exportação",
            "Producers": "Productores (Familias)",
            "Total_Volume_kg": "Volumen Total (kg)",
            "Mean_Sensory": "Nota Sensorial Media",
            "Mean_Polyphenols": "Polifenoles Medios",
            "Mean_Lipids": "Grasa Media (EE%)",
            "SaaS_Price": "Precio SaaS (R$/kg)",
            "Net_Added_Value": "Valor Neto Agregado (R$)"
        },
        "micro_allocations_title": "👪 Asignación Detallada por Familia Productora",
        "micro_allocations_desc": "Descargue esta lista procesada en formato Excel para alimentar portales aduaneros o ERPs logísticos (EUDR/GACC).",
        "saas_price_col": "Precio SaaS (R$/kg)",
        "saas_total_col": "Valor Total SaaS (R$)",
        "net_added_col": "Valor Neto Agregado (R$)",
        
        # Finance Tab
        "finance_title": "📈 Simulador de Viabilidad y Proyección Financiera (5 Años)",
        "finance_desc": "Simule en tiempo real los flujos de caja, costos de inversión (CAPEX) y retornos operativos (ROI) generados por el framework Cacaulogy.",
        "op_params": "⚙️ Parâmetros Operativos de la Cooperativa / Industria",
        "annual_volume": "Volumen Anual Inicial de Granos Procesados (Toneladas)",
        "growth_rate_lbl": "Crecimiento Anual de Volumen Esperado (%)",
        "comm_base_price_lbl": "Precio Base de Compra del Commodity (R$/kg)",
        "costs_structure": "💰 Estructura de Costos de Implantación y Operación",
        "capex_title": "Inversiones de Capital Iniciales (CAPEX)",
        "opex_title": "Costos Operativos Anuales (OPEX)",
        "capex_iot_lbl": "Sensores IoT para Cajones (R$)",
        "capex_dryers_lbl": "Automatización de Marquesinas de Secado (R$)",
        "capex_lab_lbl": "Equipos Analíticos de Laboratorio (R$)",
        "capex_setup_lbl": "Capacitación e Instalación de TI (R$)",
        "capex_eudr_lbl": "Mapeo Ambiental y EUDR (R$)",
        "capex_pkg_lbl": "Diseño de Códigos QR y Empaque (R$)",
        "total_capex_lbl": "Total CAPEX Inicial",
        "opex_saas_lbl": "Suscripción Anual del Software SaaS (R$)",
        "opex_maint_lbl": "Mantenimiento Físico de Sensores (R$)",
        "opex_logistics_lbl": "Logística de Técnicos de Campo (R$)",
        "opex_audits_lbl": "Auditorías y Certificaciones Externas (R$)",
        "total_opex_lbl": "Total OPEX Anual",
        "projection_chart_title": "📊 Proyecciones Financieras y Márgenes Operativos",
        "chart_rev_title": "Comparación de Ingreso Bruto Anual (R$)",
        "chart_rev_comm": "Ingreso Tradicional (Commodity)",
        "chart_rev_saas": "Ingreso Cacaulogy SaaS",
        "chart_cash_title": "Flujo de Caja Neto Acumulado (R$)",
        "chart_cash_flow": "Flujo de Caja Acumulado",
        "payback_lbl": "Punto de Equilibrio (Payback)",
        "dre_title": "📄 Estado de Resultados Proyectado (DRE)",
        "dre_cols": {
            "Ano": "Año",
            "Volume Processed (kg)": "Volumen Procesado (kg)",
            "Receita Commodity (R$)": "Ingreso Commodity (R$)",
            "Receita Cacaulogia (R$)": "Ingreso Cacaulogy (R$)",
            "Custo Matéria-Prima Justa (R$)": "Costo de Materia Prima Justa (R$)",
            "OPEX (R$)": "OPEX (R$)",
            "Lucro Líquido Retido (R$)": "Utilidad Neta Retenida (R$)",
            "Fluxo de Caixa Acumulado (R$)": "Flujo de Caja Acumulado (R$)"
        },
        "dre_summary_text": """
*   **Prêmio Promedio de Precio:** El precio promedio ponderado obtenido por el portafolio Cacaulogy es **R$ {:.2f}/kg** (frente a R$ {:.2f}/kg del commodity).
*   **Impacto Social Compartido:** El Costo de Materia Prima Justa transfiere un premio social directo de más de **R$ {:,.2f} acumulados en 5 años** para las familias asociadas, protegiendo la selva e incentivando el vector humano \\(K\\).
""",
        
        # Equipment Tab
        "equip_title": "🛠️ Activos Físicos y Maquinaria Requeridos en Campo",
        "equip_desc": "Para operacionalizar el rigor científico de Cacaulogy 'de la narrativa a la evidencia', la cooperativa o industria debe adquirir los siguientes activos complementarios además del software:",
        "eq_section1": "📡 1. Infraestructura IoT y Postcosecha (Dimensiones I y III)",
        "eq_section1_list": """
*   **Sondas Térmicas Selladas DS18B20:** Insertadas directamente en la masa de cacao en fermentación para registrar curvas cinéticas térmicas en tiempo real.
*   **Microcontroladores ESP32 de Bajo Costo:** Sondas con cajas protectoras estancas que transmiten datos vía red local o guardan de forma offline asíncrona.
*   **Sensores de Estufa (DHT22 / SHT31):** Instalados en marquesinas solares de secado para registrar la humedad relativa y temperatura del aire.
*   **Extractores Automatizados:** Activados de forma automática para extraer la humedad del aire del secador solar cuando supera el límite crítico de evaporación ácida.
""",
        "eq_section2": "🔬 2. Equipos Analíticos y Químicos de Laboratorio (Dimensión II)",
        "eq_section2_list": """
*   **Liofilizador de Mesa (Freeze Dryer):** Deshidrata a baja temperatura cotiledones frescos (a -50 ºC y 0.04 mbar) preservando polifenoles y alcaloides activos antes del análisis.
*   **Balanza Analítica de Precisión (0.1 mg):** Esencial para pesajes precisos de fracciones finas de grano y control de reactivos.
*   **Molino Analítico de Cuchillas:** Tritura uniformemente granos liofilizados hasta malla 40 (600 a 420 µm).
*   **Baño Ultrasónico Térmico:** Para extracción asistida por ultrasonido (25 Hz a 50 ºC) de flavonoides, teobromina y cafeína activos.
*   **Centrífuga de Alta Velocidade (14.000 rpm):** Para separación de fases y purificación rápida de extractos analíticos.
*   **Lector de Microplacas (Espectrofotômetro):** Cuantifica Polifenoles Totais (Folin-Ciocalteu a 760 nm) y Antocianinas (razón de absorbancia a 460/530 nm).
""",
        "eq_section3": "👅 3. Estructura para Análisis Sensorial Calibrado (Dimensión IV)",
        "eq_section3_list": """
*   **Cabinas de Catación Estandarizadas (ISO 8586):** Cabinas aisladas térmicamente (24 ± 0.2 ºC), con control de ruido y humedad de 53% a 60% para neutralidad de los catadores.
*   **Kit de Referencias Olfativas Comerciales:** Compuestos puros de Ácido Isovalérico, Ácido Caproico, Alcohol Isoamílico y Benzaldehído para calibración de QDA y detección de defectos.
""",
        "eq_partnerships": """
> **Alianza Tecnológica Regional:** La adquisición de equipos de laboratorio analítico de alta tecnología puede abaratarse o sustituirse mediante **convenios de cooperación de investigación** con laboratorios públicos de universidades en la Amazonía (UFPA, IFPA, UFRA, CEPLAC). La cooperativa utiliza sensores de IoT locales y envía muestras quincenales para análisis varietal en laboratorios asociados, manteniendo el OPEX bajo control.
""",
        "target_calib_title": "🎯 Blancos Quimiométricos de Calidad",
        "target_calib_desc": "El equipo analítico está calibrado para mapear el cacao dentro de rangos científicos óptimos:",
        "target_labels": ["Manteca de Cacao (%)", "Polifenoles Máx", "Puntuación QDA", "Acidez Volátil (pH)"],
        "target_suffix": "% del Objetivo Ideal"
    },
    "Italiano": {
        "title": "🍫 Piattaforma di Intelligenza Predittiva Cacaulogy",
        "subtitle": "SaaS per Aggregazione di Cooperative, Standardizzazione di Lotti di Esportazione e Modellazione di Viabilità Finanziaria",
        "config_panel": "🔧 Pannello di Configurazione",
        "num_clusters": "Numero di Lotti di Esportazione Obiettivo (Cluster)",
        "standardize": "Standardizzare Variabili Biochimiche (Z-Score)",
        "about_features_title": "Informazioni sulle Variabili del Terroir",
        "about_features_text": """
*   **Estratto Etereo (%)**: Contenuto di burro di cacao. Essenziale per la fusione, consistenza e lucentezza nell'industria.
*   **Polifenoli Totali (mg GAE/100g)**: Capacità antiossidante, associata ad amarezza e astringenza naturale.
*   **Rapporto TB/CF**: Marcatore chemiotassonomico. Rapporto Teobromina/Caffeina per distinguere Criollo (<3), Trinitario (3-9) e Forastero (9-11).
*   **Punteggio Sensoriale (QDA)**: Punteggio di valutazione sensoriale calibrato da 0 a 10.
""",
        "tab_clustering": "📦 Aggregazione di Lotti (Clustering)",
        "tab_finance": "📈 Viabilità & Proiezione Finanziaria",
        "tab_equipment": "🛠️ Inventario di Attrezzature & Infrastruttura",
        "impact_overview": "📊 Panoramica dell'Impatto della Piattaforma",
        "metric_vol": "VOLUME AGGREGATO",
        "metric_vol_desc": "Da 200 microproduttori",
        "metric_comm": "STIMA COMMODITY",
        "metric_comm_desc": "Al tasso standard di R$ 15,00/kg",
        "metric_saas": "VALUTAZIONE CACAULOGY",
        "metric_saas_desc": "Lotti premium standardizzati",
        "metric_added": "VALORE NETTO AGGIUNTO",
        "metric_added_desc": "+ {:.1f}% moltiplicatore di valore",
        "chart_title": "🗺️ Mappa Chemiometrica del Terroir e Raggruppamento",
        "chart_desc": "Ogni punto rappresenta il raccolto di una famiglia amazzonica. L'algoritmo li organizza in lotti omogenei e standardizzati per l'esportazione commerciale.",
        "polyphenols_axis": "Polifenoli Totali (mg GAE/100g) - [Dim II]",
        "sensory_axis": "Punteggio Sensoriale (QDA) - [Dim IV]",
        "lot_category": "Categoria di Lotto Standardizzato",
        "lot_spec_title": "📦 Specifiche Tecniche dei Lotti di Esportazione",
        "lot_explain_title": "🔍 Comprendere i Lotti di Esportazione Cacaulogy",
        "lot_explain_desc": """
Di seguito spieghiamo in dettaglio il profilo scientifico e commerciale di ciascun lotto raggruppato:

*   **Lotto A: Amazonian Floral-Fruity Premium:**
    *   *Cos'è:* Lotti con profilo aromatico eccezionale (punteggio superiore a 8,0/10) e alto contenuto di burro naturale (>55%). Contiene pochi polifenoli totali, riducendo l'amarezza e mettendo in risalto note floreali e di miele.
    *   *Genetica:* Linee moderne Criollo e Trinitario.
    *   *Destinazione:* Cioccolati pregiati artigianali *bean-to-bar* premium e pasticceria gourmet in Europa e Asia.
    *   *Prezzo di Vendita:* **R$ 58,00/kg**.
*   **Lotto B: Deep Forest Antioxidant:**
    *   *Cos'è:* Lotti con un altissimo potere funzionale e antiossidante (polifenoli >10.000 mg GAE/100g) e alto rapporto Teobromina/Caffeina. Note marcate di amarezza naturale e ricca astringenza.
    *   *Genetica:* Linee classiche Forastero (Amelonado).
    *   *Destinazione:* Industria cosmetica di lusso, nutraceutici e cioccolati funzionali ad alto contenuto di cacao (>80%).
    *   *Prezzo di Vendita:* **R$ 38,00/kg**.
*   **Lotto C: Traditional Balanced Trinitario:**
    *   *Cos'è:* Lotto con profilo equilibrato e classico (punteggio sensoriale vicino a 7,1/10). Eccellente consistenza di fusione industriale.
    *   *Genetica:* Ibridi tradizionali Trinitario.
    *   *Destinazione:* Industrie europee e cinesi di cioccolato fine su larga scala, aggiungendo tracciabilità e certificazione di Deforestazione Zero (EUDR).
    *   *Prezzo di Vendita:* **R$ 32,00/kg**.
""",
        "lot_summary_headers": {
            "Export_Lot_Category": "Categoria Lotto di Esportazione",
            "Producers": "Produttori (Famiglie)",
            "Total_Volume_kg": "Volume Totale (kg)",
            "Mean_Sensory": "Punteggio Sensoriale Medio",
            "Mean_Polyphenols": "Polifenoli Medi",
            "Mean_Lipids": "Grasso Medio (EE%)",
            "SaaS_Price": "Prezzo SaaS (R$/kg)",
            "Net_Added_Value": "Valore Netto Aggiunto (R$)"
        },
        "micro_allocations_title": "👪 Assegnazione Dettagliata per Famiglia Produttrice",
        "micro_allocations_desc": "Scarica questo elenco elaborato in formato Excel per alimentare portali doganali o ERP logistici (EUDR/GACC).",
        "saas_price_col": "Prezzo SaaS (R$/kg)",
        "saas_total_col": "Valore Totale SaaS (R$)",
        "net_added_col": "Valore Netto Aggiunto (R$)",
        
        # Finance Tab
        "finance_title": "📈 Simulatore di Viabilità e Proiezione Finanziaria (5 Anni)",
        "finance_desc": "Simula in tempo reale i flussi di cassa, i costi di investimento (CAPEX) e i rendimenti operativi (ROI) generati dal framework Cacaulogy.",
        "op_params": "⚙️ Parametri Operativi della Cooperativa / Industria",
        "annual_volume": "Volume Annuo Iniziale di Fave Lavorate (Tonnellate)",
        "growth_rate_lbl": "Crescita Annua del Volume Prevista (%)",
        "comm_base_price_lbl": "Prezzo Base di Acquisto della Commodity (R$/kg)",
        "costs_structure": "💰 Struttura dei Costi di Impianto e Funzionamento",
        "capex_title": "Investimenti di Capitale Iniziali (CAPEX)",
        "opex_title": "Costi Operativi Annuali (OPEX)",
        "capex_iot_lbl": "Sensori IoT per Vasche (R$)",
        "capex_dryers_lbl": "Automazione delle Serre di Essiccazione (R$)",
        "capex_lab_lbl": "Apparecchiature Analitiche di Laboratorio (R$)",
        "capex_setup_lbl": "Formazione e Installazione IT (R$)",
        "capex_eudr_lbl": "Mappatura Ambientale e EUDR (R$)",
        "capex_pkg_lbl": "Design di Codici QR e Confezionamento (R$)",
        "total_capex_lbl": "Totale CAPEX Iniziale",
        "opex_saas_lbl": "Abbonamento Annuo al Software SaaS (R$)",
        "opex_maint_lbl": "Manutenzione Fisica dei Sensori (R$)",
        "opex_logistics_lbl": "Logistica dei Tecnici di Campo (R$)",
        "opex_audits_lbl": "Audit e Certificazioni Esterne (R$)",
        "total_opex_lbl": "Totale OPEX Annuo",
        "projection_chart_title": "📊 Proiezioni Finanziarie e Margini Operativi",
        "chart_rev_title": "Confronto dei Ricavi Lordi Annuali (R$)",
        "chart_rev_comm": "Ricavo Tradizionale (Commodity)",
        "chart_rev_saas": "Ricavo Cacaulogy SaaS",
        "chart_cash_title": "Flusso di Cassa Netto Accumulato (R$)",
        "chart_cash_flow": "Flusso di Cassa Accumulato",
        "payback_lbl": "Punto di Pareggio (Payback)",
        "dre_title": "📄 Conto Economico Previsionale (DRE)",
        "dre_cols": {
            "Ano": "Anno",
            "Volume Processed (kg)": "Volume Lavorato (kg)",
            "Receita Commodity (R$)": "Ricavo Commodity (R$)",
            "Receita Cacaulogia (R$)": "Ricavo Cacaulogy (R$)",
            "Custo Matéria-Prima Justa (R$)": "Costo della Materia Prima Equa (R$)",
            "OPEX (R$)": "OPEX (R$)",
            "Lucro Líquido Retido (R$)": "Utile Netto Trattenuto (R$)",
            "Fluxo de Caixa Acumulado (R$)": "Flusso di Cassa Accumulato (R$)"
        },
        "dre_summary_text": """
*   **Premio Medio di Prezzo:** Il prezzo medio ponderato ottenuto dal portafoglio Cacaulogy is **R$ {:.2f}/kg** (rispetto a R$ {:.2f}/kg del commodity).
*   **Impatto Sociale Condiviso:** Il Costo della Materia Prima Equa trasferisce un premio sociale diretto di oltre **R$ {:,.2f} accumulati in 5 anni** per le famiglie associate, proteggendo la foresta e incentivando il vettore umano \\(K\\).
""",
        
        # Equipment Tab
        "equip_title": "🛠️ Attrezzature Fisiche e Macchinari Richiesti sul Campo",
        "equip_desc": "Per rendere operativo il rigore scientifico di Cacaulogy 'dalla narrazione all'evidenza', la cooperativa o l'industria deve acquisire i seguenti asset complementari oltre al software:",
        "eq_section1": "📡 1. Infrastruttura IoT & Post-Raccolta (Dimensioni I & III)",
        "eq_section1_list": """
*   **Sondas Térmicas Sigillate DS18B20:** Inserite direttamente nella massa di cacao in fermentazione per registrare le curve cinetiche termiche in tempo reale.
*   **Microcontrollori ESP32 di Basso Costo:** Sonde con scatole protettive stagne che trasmettono dati via rete locale o salvano in modalità offline asincrona.
*   **Sensori di Serra (DHT22 / SHT31):** Installati nelle serre solari di essiccazione per registrare l'umidità relativa e la temperatura dell'aria.
*   **Estrattori Automatizzati:** Attivati automaticamente per estrarre l'umidità dall'aria della serra quando supera la soglia critica di evaporazione acida.
""",
        "eq_section2": "🔬 2. Apparecchiature Analitiche & Chimiche di Laboratorio (Dimensioni II)",
        "eq_section2_list": """
*   **Liofilizzatore da Banco (Freeze Dryer):** Disidrata a freddo cotiledoni freschi (a -50 ºC e 0,04 mbar) preservando polifenoli e alcaloidi attivi prima dell'analisi.
*   **Bilancia Analitica di Precisione (0.1 mg):** Essenziale per pesate precise di frazioni fini di fave e controllo dei reagenti.
*   **Mulino Analitico a Coltelli:** Macina uniformemente fave liofilizzate fino a una granulometria di 40 mesh (da 600 a 420 µm).
*   **Bagno Ultrasuoni Termico:** Per estrazione assistita da ultrasuoni (25 Hz a 50 ºC) di flavonoidi, teobromina e caffeina attivi.
*   **Centrifuga ad Alta Velocità (14.000 giri/min):** Per separazione di fase e purificazione rapida di estratti analitici.
*   **Lettore di Micropiastre (Spettrofotometro):** Quantifica i Polifenoli Totali (Folin-Ciocalteu a 760 nm) e gli Antociani (rapporto di assorbanza a 460/530 nm).
""",
        "eq_section3": "👅 3. Struttura per Analisi Sensoriale Calibrada (Dimensione IV)",
        "eq_section3_list": """
*   **Cabine di Degustazione Standardizzate (ISO 8586):** Cabine isolate termicamente (24 ± 0.2 ºC), con controllo del rumore e dell'umidità dal 53% al 60% per garantire la neutralità dei degustatori.
*   **Kit di Riferimenti Olfattivi Commerciali:** Composti puri di Acido Isovalerico, Acido Caproico, Alcol Isoamilico e Benzaldeide per calibrazione di QDA e rilevamento dei difetti.
""",
        "eq_partnerships": """
> **Partenariato Tecnologico Regionale:** L'acquisizione di attrezzature di laboratorio analitico ad alta tecnologia può essere ridotta o sostituita tramite **accordi di cooperazione di ricerca** con laboratori pubblici delle università in Amazzonia (UFPA, IFPA, UFRA, CEPLAC). La cooperativa utilizza sensori IoT locali e invia campioni bisettimanali per analisi varietali nei laboratori partner, mantenendo l'OPEX sotto stretto controllo.
""",
        "target_calib_title": "🎯 Target Chemiometrici di Qualità",
        "target_calib_desc": "L'apparecchiatura analitica è calibrata per mappare il cacao entro intervalli scientifici ottimali:",
        "target_labels": ["Burro di Cacao (%)", "Polifenoli Max", "Punteggio QDA", "Acidità Volatile (pH)"],
        "target_suffix": "% del Target Ideale"
    },
    "Français": {
        "title": "🍫 Plateforme d'Intelligence Prédictive Cacaulogy",
        "subtitle": "SaaS pour l'Agrégation de Coopératives, la Standardisation des Lots d'Exportation et la Viabilité Financière",
        "config_panel": "🔧 Panneau de Configuration",
        "num_clusters": "Nombre de Lots d'Exportation Cibles (Clusters)",
        "standardize": "Standardiser les Variables Biochimiques (Z-Score)",
        "about_features_title": "À propos des Variables du Terroir",
        "about_features_text": """
*   **Extrait Éthéré (%)**: Teneur en beurre de cacao. Essentiel pour la fusion, la texture et la brillance dans l'industrie.
*   **Polyphénols Totaux (mg GAE/100g)**: Capacité antioxydante, associée à l'amertume et à l'astringence naturelle.
*   **Rapport TB/CF**: Marqueur chimiotaxonomique. Rapport Théobromine/Caféine pour distinguer Criollo (<3), Trinitario (3-9) et Forastero (9-11).
*   **Note Sensorielle (QDA)**: Note d'évaluation sensorielle calibrée de 0 à 10.
""",
        "tab_clustering": "📦 Agrégation de Lots (Clustering)",
        "tab_finance": "📈 Viabilité & Projections Financières",
        "tab_equipment": "🛠️ Inventaire des Équipements & Infrastructure",
        "impact_overview": "📊 Aperçu de l'Impact de la Plateforme",
        "metric_vol": "VOLUME AGRÉGÉ",
        "metric_vol_desc": "De 200 microproducteurs",
        "metric_comm": "ESTIMATION COMMODITY",
        "metric_comm_desc": "Au taux standard de R$ 15,00/kg",
        "metric_saas": "ÉVALUATION CACAULOGY",
        "metric_saas_desc": "Lots premium standardisés",
        "metric_added": "VALEUR NETTE AJOUTÉE",
        "metric_added_desc": "+ {:.1f}% multiplicateur de valeur",
        "chart_title": "🗺️ Carte Chimiométrique du Terroir & Regroupement",
        "chart_desc": "Chaque point représente la récolte d'une famille amazonienne. L'algorithme les organise en lots homogènes et standardisés pour l'exportation commerciale.",
        "polyphenols_axis": "Polyphénols Totaux (mg GAE/100g) - [Dim II]",
        "sensory_axis": "Note Sensorielle (QDA) - [Dim IV]",
        "lot_category": "Catégorie de Lot Standardisé",
        "lot_spec_title": "📦 Spécifications Techniques des Lots d'Exportation",
        "lot_explain_title": "🔍 Comprendre les Lots d'Exportation Cacaulogy",
        "lot_explain_desc": """
Ci-dessous, nous expliquons en détail le profil scientifique et commercial de chaque lot regroupé :

*   **Lot A: Amazonian Floral-Fruity Premium :**
    *   *Qu'est-ce que c'est :* Lots au profil aromatique exceptionnel (note supérieure à 8,0/10) et à haute teneur en beurre naturel (>55%). Il contient peu de polyphénols totaux, réduisant l'amertume et mettant en valeur des notes florales et de miel.
    *   *Génétique :* Lignées modernes Criollo et Trinitario.
    *   *Destination :* Chocolats fins artisanaux *bean-to-bar* premium et pâtisserie de luxe en Europe et en Asie.
    *   *Prix de Vente :* **R$ 58,00/kg**.
*   **Lot B: Deep Forest Antioxidant :**
    *   *Qu'est-ce que c'est :* Lots à très haut pouvoir fonctionnel et antioxydant (polyphénols >10 000 mg GAE/100g) et rapport Théobromine/Caféine élevé. Notes marquées d'amertume naturelle et astringence riche.
    *   *Génétique :* Lignées classiques Forastero (Amelonado).
    *   *Destination :* Industrie cosmétique de luxe, nutraceutiques et chocolats fonctionnels à haute teneur en cacao (>80%).
    *   *Prix de Vente :* **R$ 38,00/kg**.
*   **Lot C: Traditional Balanced Trinitario :**
    *   *Qu'est-ce que c'est :* Lot au profil équilibré et classique (note sensorielle proche de 7,1/10). Excellente régularité de fusion industrielle.
    *   *Génétique :* Hybrides traditionnels Trinitario.
    *   *Destination :* Industries européennes et chinoises de chocolat fin à grande échelle, intégrant la traçabilité et la certification de Déforestation Zéro (EUDR).
    *   *Prix de Vente :* **R$ 32,00/kg**.
""",
        "lot_summary_headers": {
            "Export_Lot_Category": "Catégorie de Lot d'Exportation",
            "Producers": "Producteurs (Familles)",
            "Total_Volume_kg": "Volume Total (kg)",
            "Mean_Sensory": "Note Sensorielle Moyenne",
            "Mean_Polyphenols": "Polyphénols Moyens",
            "Mean_Lipids": "Matière Grasse Moyenne (EE%)",
            "SaaS_Price": "Prix SaaS (R$/kg)",
            "Net_Added_Value": "Valeur Nette Ajoutée (R$)"
        },
        "micro_allocations_title": "👪 Répartition Détaillée par Famille Productrice",
        "micro_allocations_desc": "Téléchargez cette liste traitée au format Excel pour alimenter les portails douaniers ou les ERP logistiques (EUDR/GACC).",
        "saas_price_col": "Prix SaaS (R$/kg)",
        "saas_total_col": "Valeur Totale SaaS (R$)",
        "net_added_col": "Valeur Nette Ajoutée (R$)",
        
        # Finance Tab
        "finance_title": "📈 Simulateur de Viabilité & Projections Financières (5 Ans)",
        "finance_desc": "Simulez en temps réel les flux de trésorerie, les dépenses d'investissement (CAPEX) et les retours opérationnels (ROI) générés par le framework Cacaulogy.",
        "op_params": "⚙️ Paramètres Opérationnels de la Coopérative / Industrie",
        "annual_volume": "Volume Annuel Initial de Fèves Traitées (Tonnes)",
        "growth_rate_lbl": "Croissance Annuelle du Volume Attendue (%)",
        "comm_base_price_lbl": "Prix de Base d'Achat de la Matière Première (R$/kg)",
        "costs_structure": "💰 Structure des Coûts d'Implantation et d'Exploitation",
        "capex_title": "Investissements Initiaux (CAPEX)",
        "opex_title": "Coûts Opérationnels Annuels (OPEX)",
        "capex_iot_lbl": "Capteurs IoT pour Bacs (R$)",
        "capex_dryers_lbl": "Automatisation des Serres de Séchage (R$)",
        "capex_lab_lbl": "Équipements Analytiques de Laboratoire (R$)",
        "capex_setup_lbl": "Formation & Installation Informatique (R$)",
        "capex_eudr_lbl": "Cartographie Environnementale & EUDR (R$)",
        "capex_pkg_lbl": "Conception de Codes QR & Emballages (R$)",
        "total_capex_lbl": "Total CAPEX Initial",
        "opex_saas_lbl": "Abonnement Annuel du Logiciel SaaS (R$)",
        "opex_maint_lbl": "Maintenance Physique des Capteurs (R$)",
        "opex_logistics_lbl": "Logistique des Techniciens de Terrain (R$)",
        "opex_audits_lbl": "Audits & Certifications Externes (R$)",
        "total_opex_lbl": "Total OPEX Annuel",
        "projection_chart_title": "📊 Projections Financières & Marges Opérationnelles",
        "chart_rev_title": "Comparaison du Chiffre d'Affaires Brut Annuel (R$)",
        "chart_rev_comm": "Revenu Traditionnel (Commodity)",
        "chart_rev_saas": "Revenu Cacaulogy SaaS",
        "chart_cash_title": "Flux de Trésorerie Net Cumulé (R$)",
        "chart_cash_flow": "Flux de Trésorerie Cumulé",
        "payback_lbl": "Seuil de Rentabilité (Payback)",
        "dre_title": "📄 Compte de Résultat Prévisionnel (DRE)",
        "dre_cols": {
            "Ano": "Année",
            "Volume Processed (kg)": "Volume Traité (kg)",
            "Receita Commodity (R$)": "Revenu Commodity (R$)",
            "Receita Cacaulogia (R$)": "Revenu Cacaulogy (R$)",
            "Custo Matéria-Prima Justa (R$)": "Coût de Matière Équitable (R$)",
            "OPEX (R$)": "OPEX (R$)",
            "Lucro Líquido Retido (R$)": "Bénéfice Net Conservé (R$)",
            "Fluxo de Caixa Acumulado (R$)": "Flux de Trésorerie Cumulé (R$)"
        },
        "dre_summary_text": """
*   **Prix Premium Moyen:** Le prix moyen préférentiel généré par le portefeuille Cacaulogy est de **R$ {:.2f}/kg** (contre R$ {:.2f}/kg pour la commodité).
*   **Impact Social Partagé:** Le Coût de la Matière Première Équitable reverse une prime sociale directe de plus de **R$ {:,.2f} cumulés en 5 ans** aux familles associées, protégeant la forêt et stimulant le vecteur humain \\(K\\).
""",
        
        # Equipment Tab
        "equip_title": "🛠️ Équipements Physiques et Machines Requis sur le Terrain",
        "equip_desc": "Pour opérationnaliser la rigueur scientifique de Cacaulogy 'du récit à la preuve', la coopérative ou l'industrie doit acquérir les actifs complémentaires suivants en plus du logiciel :",
        "eq_section1": "📡 1. Infrastructure IoT & Post-Récolte (Dimensions I & III)",
        "eq_section1_list": """
*   **Sondes Thermiques Scellées DS18B20 :** Insérées directement dans la masse de cacao en fermentation pour enregistrer les courbes cinétiques de fermentation en temps réel.
*   **Microcontrôleurs ESP32 de Bas Coût :** Sondes dotées de boîtiers de protection étanches transmettant via réseau local ou enregistrant en mode asynchrone hors ligne.
*   **Capteurs de Serre (DHT22 / SHT31) :** Installés dans les serres de séchage solaire pour enregistrer l'humidité relative et la température ambiante.
*   **Extracteurs Automatiques :** Activés automatiquement pour évacuer l'humidité de la serre solaire lorsqu'elle dépasse le seuil critique d'évaporation acide.
""",
        "eq_section2": "🔬 2. Équipements Analytiques & Chimiques de Laboratoire (Dimension II)",
        "eq_section2_list": """
*   **Lyophilisateur de Paillasse (Freeze Dryer) :** Déshydrate à froid les cotylédons frais (à -50 ºC et 0,04 mbar) pour préserver les polyphénols actifs et les alcaloïdes des dégradations thermiques avant l'analyse.
*   **Balance Analytique de Précision (0,1 mg) :** Indispensable pour peser de manière précise des fractions fines de fèves et contrôler les réactifs.
*   **Broyeur Analytique à Couteaux :** Broie uniformément les fèves lyophilisées jusqu'à une granulométrie de 40 mesh (600 à 420 µm).
*   **Bain à Ultrasons Thermique :** Pour l'extraction assistée par ultrasons (25 Hz à 50 ºC) des flavonoïdes, théobromine et caféine actifs.
*   **Centrifugeuse à Haute Vitesse (14 000 tr/min) :** Pour une séparation rapide des phases et la purification des extraits analytiques.
*   **Lecteur de Microplaques (Spectrophotomètre) :** Quantifie les Polyphénols Totaux (Folin-Ciocalteu à 760 nm) et les Anthocyanes (rapport d'absorbance à 460/530 nm).
""",
        "eq_section3": "👅 3. Structure pour l'Analyse Sensorielle Calibrée (Dimension IV)",
        "eq_section3_list": """
*   **Cabines de Dégustation Standardisées (ISO 8586) :** Cabines isolées thermiquement (24 ± 0,2 ºC), avec contrôle du bruit et de l'humidité (53% à 60%) pour garantir la neutralité des dégustateurs.
*   **Kit de Références Olfactives Commerciales :** Composés purs d'acide isovalérique, d'acide caproïque, d'alcool isoamylique et de benzaldéhyde pour l'étalonnage QDA et la détection des défauts.
""",
        "eq_partnerships": """
> **Partenariat Technologique Régional :** L'acquisition d'équipements de laboratoire analytique de haute technologie peut être rationalisée ou remplacée par des **accords de coopération de recherche** avec des laboratoires publics d'universités en Amazonie (UFPA, IFPA, UFRA, CEPLAC). La coopérative utilise les capteurs IoT locaux et envoie des échantillons toutes les deux semaines pour analyse variétale dans les laboratoires partenaires, maintenant l'OPEX sous contrôle.
""",
        "target_calib_title": "🎯 Cibles Chimiométriques de Qualité",
        "target_calib_desc": "L'équipement analytique est étalonné pour cartographier le cacao dans les plages scientifiques optimales :",
        "target_labels": ["Beurre de Cacao (%)", "Polyphénols Max", "Note QDA", "Acidité Volatile (pH)"],
        "target_suffix": "% de la Cible Idéale"
    }
}

# Escolha de Idioma na Barra Lateral
lang = st.sidebar.selectbox("🌐 Idioma / Language / 语言 / Idioma / Lingua / Langue", ["Português", "English", "中文", "Español", "Italiano", "Français"])
T = TRANSLATIONS[lang]

st.title(T["title"])
st.subheader(T["subtitle"])

# Sidebar para configurações
st.sidebar.header(T["config_panel"])
num_clusters = st.sidebar.slider(T["num_clusters"], min_value=2, max_value=5, value=3)
standardize_data = st.sidebar.checkbox(T["standardize"], value=True)

# Geração de dados sintéticos de exemplo
@st.cache_data
def get_synthetic_data():
    np.random.seed(42)
    n_farmers = 200
    farmers = [f"Família {i+1:03d}" for i in range(n_farmers)]
    municipalities = np.random.choice(["Medicilândia", "Tomé-Açu", "Santarém", "Altamira", "Novo Repartimento"], n_farmers)
    
    # Gerando dados correlacionados
    cluster_type = np.random.choice([0, 1, 2], n_farmers, p=[0.35, 0.45, 0.20])
    vol = np.random.gamma(shape=5, scale=120, size=n_farmers) # 100 a 1500 kg
    
    lipid_content = np.zeros(n_farmers)
    polyphenols = np.zeros(n_farmers)
    tb_cf_ratio = np.zeros(n_farmers)
    sensory = np.zeros(n_farmers)
    
    for i in range(n_farmers):
        if cluster_type[i] == 0: # Premium (Floral/Frutado)
            lipid_content[i] = np.random.normal(56.0, 1.5)
            polyphenols[i] = np.random.normal(5200, 600)
            tb_cf_ratio[i] = np.random.normal(3.2, 0.5)
            sensory[i] = np.random.normal(8.4, 0.4)
        elif cluster_type[i] == 1: # Funcional (Alto Antioxidante)
            lipid_content[i] = np.random.normal(48.0, 2.0)
            polyphenols[i] = np.random.normal(11000, 1500)
            tb_cf_ratio[i] = np.random.normal(11.2, 1.0)
            sensory[i] = np.random.normal(6.5, 0.5)
        else: # Equilibrado (Trinitário Tradicional)
            lipid_content[i] = np.random.normal(51.0, 1.5)
            polyphenols[i] = np.random.normal(7000, 800)
            tb_cf_ratio[i] = np.random.normal(6.5, 0.8)
            sensory[i] = np.random.normal(7.2, 0.4)
            
    sensory = np.clip(sensory, 0, 10)
    
    df = pd.DataFrame({
        "Farmer_ID": [f"AMZ-{1000+i}" for i in range(n_farmers)],
        "Family_Name": farmers,
        "Municipality": municipalities,
        "Volume_Wet_kg": vol.round(1),
        "Ethereal_Extract_Pct": lipid_content.round(2),
        "Total_Polyphenols_mgGAE_100g": polyphenols.round(1),
        "TB_CF_Ratio": tb_cf_ratio.round(2),
        "Sensory_Score": sensory.round(2)
    })
    return df

df = get_synthetic_data()

st.sidebar.markdown(f"### {T['about_features_title']}")
st.sidebar.markdown(T['about_features_text'])

# Execução do K-Means para Agrupamento
features = ["Ethereal_Extract_Pct", "Total_Polyphenols_mgGAE_100g", "TB_CF_Ratio", "Sensory_Score"]
X = df[features]

if standardize_data:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
else:
    X_scaled = X

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Classificação mercadológica inteligente dos clusters (Blindagem de colisões)
cluster_means = df.groupby("Cluster")[["Sensory_Score", "Total_Polyphenols_mgGAE_100g", "Ethereal_Extract_Pct"]].mean()

sensory_rank = cluster_means["Sensory_Score"].sort_values(ascending=False).index.tolist()
polyphenol_rank = cluster_means["Total_Polyphenols_mgGAE_100g"].sort_values(ascending=False).index.tolist()

premium_cluster = sensory_rank[0]
functional_cluster = polyphenol_rank[0]

if premium_cluster == functional_cluster:
    if len(polyphenol_rank) > 1:
        functional_cluster = polyphenol_rank[1]

# Tradução dinâmica dos nomes de Lotes com base no idioma escolhido
category_translations = {
    "Português": {
        "LotA": "Prêmio Floral-Frutado Amazônico (Lote A)",
        "LotB": "Antioxidante da Floresta Profunda (Lote B)",
        "LotC": "Trinitário Tradicional Equilibrado (Lote C)",
        "LotD": "Mescla Agroflorestal Padrão"
    },
    "English": {
        "LotA": "Amazonian Floral-Fruity Premium (Lot A)",
        "LotB": "Deep Forest Antioxidant (Lot B)",
        "LotC": "Traditional Balanced Trinitario (Lot C)",
        "LotD": "Standard Agroforest Blend"
    },
    "中文": {
        "LotA": "亚马逊花果香溢价级 (批次 A)",
        "LotB": "深林抗氧化级 (批次 B)",
        "LotC": "传统平衡特里尼达级 (批次 C)",
        "LotD": "标准农林混作调配豆"
    },
    "Español": {
        "LotA": "Floral-Frutado Amazónico Premium (Lote A)",
        "LotB": "Antioxidante de la Selva Profunda (Lote B)",
        "LotC": "Trinitario Tradicional Equilibrado (Lote C)",
        "LotD": "Mezcla Agroforestal Estándar"
    },
    "Italiano": {
        "LotA": "Premium Floreale-Fruttato Amazzonico (Lotto A)",
        "LotB": "Antiossidante della Foresta Profonda (Lotto B)",
        "LotC": "Trinitario Tradizionale Bilanciato (Lotto C)",
        "LotD": "Miscela Agroforestale Standard"
    },
    "Français": {
        "LotA": "Premium Floral-Fruité Amazonien (Lot A)",
        "LotB": "Antioxydant de la Forêt Profonde (Lot B)",
        "LotC": "Trinitario Traditionnel Équilibré (Lot C)",
        "LotD": "Mélange Agroforestier Standard"
    }
}

ct = category_translations[lang]

cluster_name_mapping = {}
for c in range(num_clusters):
    if c == premium_cluster:
        cluster_name_mapping[c] = ct["LotA"]
    elif c == functional_cluster:
        cluster_name_mapping[c] = ct["LotB"]
    elif num_clusters >= 3 and c == sensory_rank[min(1 if len(sensory_rank) > 1 else 0, len(sensory_rank)-1)] and c not in cluster_name_mapping.values():
        cluster_name_mapping[c] = ct["LotC"]
    else:
        cluster_name_mapping[c] = f"{ct['LotD']} (D-{c})"

df["Export_Lot_Category"] = df["Cluster"].map(cluster_name_mapping)

# Valoração Econômica baseada no Modelo de Negócio
def calculate_pricing(row):
    category = row["Export_Lot_Category"]
    vol = row["Volume_Wet_kg"]
    
    if ct["LotA"] in category:
        price_per_kg = 58.00
    elif ct["LotB"] in category:
        price_per_kg = 38.00
    elif ct["LotC"] in category:
        price_per_kg = 32.00
    else:
        price_per_kg = 22.00
        
    commodity_val = vol * 15.00
    saas_val = vol * price_per_kg
    added_value = saas_val - commodity_val
    return pd.Series({
        "SaaS_Price_BRL": price_per_kg,
        "SaaS_Total_Value_BRL": saas_val,
        "Net_Added_Value_BRL": added_value
    })

pricing_df = df.apply(calculate_pricing, axis=1)
df["SaaS_Price_BRL"] = pricing_df["SaaS_Price_BRL"]
df["SaaS_Total_Value_BRL"] = pricing_df["SaaS_Total_Value_BRL"]
df["Net_Added_Value_BRL"] = pricing_df["Net_Added_Value_BRL"]

# Organização da Plataforma em Tabs (Guias)
tab_clustering, tab_finance, tab_equipment = st.tabs([
    T["tab_clustering"], 
    T["tab_finance"], 
    T["tab_equipment"]
])

# ==============================================================================
# TAB 1: AGRUPAMENTO DE LOTES (CLUSTERING ENGINE)
# ==============================================================================
with tab_clustering:
    st.markdown(f"### {T['impact_overview']}")
    col1, col2, col3, col4 = st.columns(4)

    total_volume = df["Volume_Wet_kg"].sum() / 1000  # em toneladas
    total_saas_val = df["SaaS_Total_Value_BRL"].sum()
    total_comm_val = df["Volume_Wet_kg"].sum() * 15.00
    net_added_value = total_saas_val - total_comm_val

    with col1:
        st.markdown(f"<div class='metric-card'><h4>{T['metric_vol']}</h4><h2>{total_volume:.1f} Tons</h2><p>{T['metric_vol_desc']}</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h4>{T['metric_comm']}</h4><h2>R$ {total_comm_val:,.2f}</h2><p>{T['metric_comm_desc']}</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h4>{T['metric_saas']}</h4><h2>R$ {total_saas_val:,.2f}</h2><p>{T['metric_saas_desc']}</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><h4>{T['metric_added']}</h4><h2>R$ {net_added_value:,.2f}</h2><p style='color:#cc5500; font-weight:bold;'>{T['metric_added_desc'].format((total_saas_val/total_comm_val-1)*100)}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### {T['chart_title']}")
    st.write(T["chart_desc"])

    # Cores dinâmicas coerentes
    discrete_color_map = {
        ct["LotA"]: "#cc5500",
        ct["LotB"]: "#403228",
        ct["LotC"]: "#998c82"
    }
    for cat in df["Export_Lot_Category"].unique():
        if cat not in discrete_color_map:
            discrete_color_map[cat] = "#d5c9c1"

    fig = px.scatter(
        df, 
        x="Total_Polyphenols_mgGAE_100g", 
        y="Sensory_Score", 
        color="Export_Lot_Category",
        size="Volume_Wet_kg",
        hover_name="Family_Name",
        hover_data=["Municipality", "Ethereal_Extract_Pct", "TB_CF_Ratio", "SaaS_Price_BRL"],
        color_discrete_map=discrete_color_map,
        labels={
            "Total_Polyphenols_mgGAE_100g": T["polyphenols_axis"],
            "Sensory_Score": T["sensory_axis"],
            "Export_Lot_Category": T["lot_category"]
        }
    )
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="#faf7f3", font_color="#403228")
    st.plotly_chart(fig, use_container_width=True)

    # Explicação detalhada e desmistificação dos Lotes na primeira página
    st.markdown("---")
    st.markdown(f"### {T['lot_explain_title']}")
    st.markdown(T["lot_explain_desc"])

    # Detalhes das Categorias de Lotes em Tabela
    st.markdown(f"### {T['lot_spec_title']}")
    lot_summary = df.groupby("Export_Lot_Category").agg(
        Producers=("Farmer_ID", "count"),
        Total_Volume_kg=("Volume_Wet_kg", "sum"),
        Mean_Sensory=("Sensory_Score", "mean"),
        Mean_Polyphenols=("Total_Polyphenols_mgGAE_100g", "mean"),
        Mean_Lipids=("Ethereal_Extract_Pct", "mean"),
        SaaS_Price=("SaaS_Price_BRL", "first"),
        Net_Added_Value=("Net_Added_Value_BRL", "sum")
    ).reset_index()

    # Traduzir cabeçalhos da tabela do resumo
    headers_map = T["lot_summary_headers"]
    lot_summary_renamed = lot_summary.rename(columns=headers_map)

    # Formatar dados numéricos para a exibição amigável
    st.dataframe(lot_summary_renamed.style.format({
        headers_map["Total_Volume_kg"]: "{:,.1f} kg",
        headers_map["Mean_Sensory"]: "{:.2f} / 10",
        headers_map["Mean_Polyphenols"]: "{:.1f} mg/100g",
        headers_map["Mean_Lipids"]: "{:.2f}%",
        headers_map["SaaS_Price"]: "R$ {:.2f}/kg",
        headers_map["Net_Added_Value"]: "R$ {:,.2f}"
    }))

    # Tabela detalhada por produtor
    st.markdown(f"### {T['micro_allocations_title']}")
    st.write(T["micro_allocations_desc"])
    
    # Traduzir colunas da tabela individual
    producer_cols_map = {
        "Farmer_ID": "ID",
        "Family_Name": "Família / Family",
        "Municipality": "Município / Location",
        "Volume_Wet_kg": "Volume (kg)",
        "Export_Lot_Category": T["lot_category"],
        "SaaS_Price_BRL": T["saas_price_col"],
        "SaaS_Total_Value_BRL": T["saas_total_col"],
        "Net_Added_Value_BRL": T["net_added_col"]
    }
    st.dataframe(df[list(producer_cols_map.keys())].rename(columns=producer_cols_map))


# ==============================================================================
# TAB 2: VIABILIDADE ECONÔMICA & PROJEÇÃO DE LUCROS (ECONOMIC MODELER)
# ==============================================================================
with tab_finance:
    st.markdown(f"### {T['finance_title']}")
    st.write(T["finance_desc"])

    # Parâmetros Editáveis pelo Usuário
    st.markdown(f"#### {T['op_params']}")
    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        annual_volume_tons = st.slider(T["annual_volume"], min_value=10.0, max_value=1000.0, value=float(total_volume), step=10.0)
    with col_input2:
        growth_rate = st.slider(T["growth_rate_lbl"], min_value=5, max_value=50, value=15, step=5)
    with col_input3:
        comm_base_price = st.number_input(T["comm_base_price_lbl"], min_value=5.0, max_value=50.0, value=15.00, step=0.50)

    # Detalhes de CAPEX e OPEX
    st.markdown(f"#### {T['costs_structure']}")
    col_costs1, col_costs2 = st.columns(2)
    
    with col_costs1:
        st.markdown(f"**{T['capex_title']}**")
        capex_iot = st.number_input(T["capex_iot_lbl"], value=10000.0)
        capex_dryers = st.number_input(T["capex_dryers_lbl"], value=15000.0)
        capex_lab = st.number_input(T["capex_lab_lbl"], value=20000.0)
        capex_setup = st.number_input(T["capex_setup_lbl"], value=15000.0)
        capex_eudr = st.number_input(T["capex_eudr_lbl"], value=25000.0)
        capex_pkg = st.number_input(T["capex_pkg_lbl"], value=10000.0)
        total_capex = capex_iot + capex_dryers + capex_lab + capex_setup + capex_eudr + capex_pkg
        st.warning(f"**Total CAPEX: R$ {total_capex:,.2f}**")

    with col_costs2:
        st.markdown(f"**{T['opex_title']}**")
        opex_saas = st.number_input(T["opex_saas_lbl"], value=12000.0)
        opex_maint = st.number_input(T["opex_maint_lbl"], value=5000.0)
        opex_logistics = st.number_input(T["opex_logistics_lbl"], value=20000.0)
        opex_audits = st.number_input(T["opex_audits_lbl"], value=15000.0)
        total_opex_base = opex_saas + opex_maint + opex_logistics + opex_audits
        st.warning(f"**Total OPEX: R$ {total_opex_base:,.2f}**")

    # Cálculos Ponderados de Receita
    total_samples = len(df)
    prop_lot_a = len(df[df["Export_Lot_Category"].str.contains("Lot A|Lote A|批次 A|Lotto A")]) / total_samples
    prop_lot_b = len(df[df["Export_Lot_Category"].str.contains("Lot B|Lote B|批次 B|Lotto B")]) / total_samples
    prop_lot_c = len(df[df["Export_Lot_Category"].str.contains("Lot C|Lote C|批次 C|Lotto C")]) / total_samples
    prop_lot_d = 1.0 - (prop_lot_a + prop_lot_b + prop_lot_c)

    weighted_saas_price = (prop_lot_a * 58.00) + (prop_lot_b * 38.00) + (prop_lot_c * 32.00) + (prop_lot_d * 22.00)
    
    # Simulação de 5 Anos
    years_translation = {
        "Português": "Ano", "English": "Year", "中文": "年度", 
        "Español": "Año", "Italiano": "Anno", "Français": "Année"
    }
    yt = years_translation[lang]
    years = [f"{yt} {i}" for i in range(1, 6)]
    vol_proj = []
    rev_comm_proj = []
    rev_saas_proj = []
    materia_prima_proj = []
    opex_proj = []
    net_profit_proj = []
    cum_cash_flow = []

    current_volume_kg = annual_volume_tons * 1000
    cum_cash = -total_capex

    for y in range(1, 6):
        vol_y = current_volume_kg if y == 1 else vol_proj[-1] * (1 + growth_rate/100)
        vol_proj.append(vol_y)
        
        rev_comm = vol_y * comm_base_price
        rev_comm_proj.append(rev_comm)
        
        rev_saas = vol_y * weighted_saas_price
        rev_saas_proj.append(rev_saas)
        
        opex_y = total_opex_base if y == 1 else opex_proj[-1] * 1.03
        opex_proj.append(opex_y)
        
        recompensa_produtor_por_kg = comm_base_price + ((weighted_saas_price - comm_base_price) * 0.55)
        mp_cost = vol_y * recompensa_produtor_por_kg
        materia_prima_proj.append(mp_cost)
        
        profit_y = rev_saas - mp_cost - opex_y
        if y == 1:
            profit_y -= total_capex
        net_profit_proj.append(profit_y)
        
        cum_cash += (rev_saas - mp_cost - opex_y)
        cum_cash_flow.append(cum_cash)

    # DataFrame de Prospecção
    dre_cols_map = T["dre_cols"]
    df_proj = pd.DataFrame({
        dre_cols_map["Ano"]: years,
        dre_cols_map["Volume Processed (kg)"]: vol_proj,
        dre_cols_map["Receita Commodity (R$)"]: rev_comm_proj,
        dre_cols_map["Receita Cacaulogia (R$)"]: rev_saas_proj,
        dre_cols_map["Custo Matéria-Prima Justa (R$)"]: materia_prima_proj,
        dre_cols_map["OPEX (R$)"]: opex_proj,
        dre_cols_map["Lucro Líquido Retido (R$)"]: net_profit_proj,
        dre_cols_map["Fluxo de Caixa Acumulado (R$)"]: cum_cash_flow
    })

    # Visualização de Retorno Gráfico
    st.markdown(f"#### {T['projection_chart_title']}")
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(x=years, y=rev_comm_proj, name=T["chart_rev_comm"], marker_color="#998c82"))
        fig_rev.add_trace(go.Bar(x=years, y=rev_saas_proj, name=T["chart_rev_saas"], marker_color="#cc5500"))
        fig_rev.update_layout(title=T["chart_rev_title"], barmode="group", paper_bgcolor="white", plot_bgcolor="#faf7f3")
        st.plotly_chart(fig_rev, use_container_width=True)

    with col_plot2:
        fig_cash = go.Figure()
        fig_cash.add_trace(go.Scatter(x=years, y=cum_cash_flow, mode="lines+markers", name=T["chart_cash_flow"], line=dict(color="#403228", width=3), marker=dict(size=8)))
        fig_cash.add_hline(y=0, line_dash="dash", line_color="red", annotation_text=T["payback_lbl"])
        fig_cash.update_layout(title=T["chart_cash_title"], paper_bgcolor="white", plot_bgcolor="#faf7f3")
        st.plotly_chart(fig_cash, use_container_width=True)

    # Tabela de Dados Formatada
    st.markdown(f"#### {T['dre_title']}")
    st.dataframe(df_proj.style.format({
        dre_cols_map["Volume Processed (kg)"]: "{:,.1f} kg",
        dre_cols_map["Receita Commodity (R$)"]: "R$ {:,.2f}",
        dre_cols_map["Receita Cacaulogia (R$)"]: "R$ {:,.2f}",
        dre_cols_map["Custo Matéria-Prima Justa (R$)"]: "R$ {:,.2f}",
        dre_cols_map["OPEX (R$)"]: "R$ {:,.2f}",
        dre_cols_map["Lucro Líquido Retido (R$)"]: "R$ {:,.2f}",
        dre_cols_map["Fluxo de Caixa Acumulado (R$)"]: "R$ {:,.2f}"
    }))

    st.markdown(T["dre_summary_text"].format(weighted_saas_price, comm_base_price, sum(materia_prima_proj)))


# ==============================================================================
# TAB 3: INVENTÁRIO DE EQUIPAMENTOS (PHYSICAL ASSETS GUIDE)
# ==============================================================================
with tab_equipment:
    st.markdown(f"### {T['equip_title']}")
    st.write(T["equip_desc"])

    col_eq1, col_plot_eq = st.columns([3, 2])
    
    with col_eq1:
        st.markdown(f"<div class='equip-card'><h4>{T['eq_section1']}</h4>{T['eq_section1_list']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='equip-card'><h4>{T['eq_section2']}</h4>{T['eq_section2_list']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='equip-card'><h4>{T['eq_section3']}</h4>{T['eq_section3_list']}</div>", unsafe_allow_html=True)
        
    with col_plot_eq:
        st.markdown(f"#### {T['target_calib_title']}")
        st.write(T["target_calib_desc"])
        
        target_labels = T["target_labels"]
        target_values = [58.9, 100.0, 86.0, 96.0] # percentuais do alvo ideal
        
        fig_target = go.Figure(go.Bar(
            x=target_values,
            y=target_labels,
            orientation='h',
            marker_color='#403228',
            text=[f"{v} {T['target_suffix']}" for v in target_values],
            textposition='auto'
        ))
        fig_target.update_layout(
            title=T["target_calib_title"],
            xaxis_title=T["target_suffix"],
            paper_bgcolor="white",
            plot_bgcolor="#faf7f3"
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
        st.markdown(T["eq_partnerships"])
