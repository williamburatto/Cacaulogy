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

st.title("🍫 Cacaulogy Predictive Intelligence Platform")
st.subheader("SaaS Cooperative Aggregation, Export Lot Standardization & Financial Viability Modeler")

# Sidebar para configurações
st.sidebar.header("🔧 Configuration Panel")
num_clusters = st.sidebar.slider("Number of Target Export Lots (Clusters)", min_value=2, max_value=5, value=3)
standardize_data = st.sidebar.checkbox("Standardize Biochemical Features (Z-Score)", value=True)

# Geração de dados sintéticos de exemplo
@st.cache_data
def get_synthetic_data():
    np.random.seed(42)
    n_farmers = 200
    farmers = [f"Família {i+1:03d}" for i in range(n_farmers)]
    municipalities = np.random.choice(["Medicilândia", "Tomé-Açu", "Santarém", "Altamira", "Novo Repartimento"], n_farmers)
    
    # Gerando dados correlacionados
    # Variedade Criollo/Trinitario premium tem maior teor de gordura, nota sensorial mais alta e menores polifenóis
    # Variedade Forastero rústica tem maior teor de polifenóis e menor teor de gordura
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
            
    # Truncando notas sensoriais a 10
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

st.sidebar.markdown("""
### About the Features
*   **Ethereal Extract (%)**: Cocoa butter content. Higher is better for industrial melting.
*   **Total Polyphenols (mg GAE/100g)**: Antioxidant capacity, strongly related to bitterness/astringency.
*   **TB/CF Ratio**: Chemotaxonomic marker to distinguish Criollo (<3), Trinitario (3-9), and Forastero (9-11).
*   **Sensory Score (QDA)**: Calibrated organoleptic grade from 0 to 10.
""")

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

# Criando dicionário de classificação baseado em rankings para evitar colisões
sensory_rank = cluster_means["Sensory_Score"].sort_values(ascending=False).index.tolist()
polyphenol_rank = cluster_means["Total_Polyphenols_mgGAE_100g"].sort_values(ascending=False).index.tolist()

premium_cluster = sensory_rank[0]
functional_cluster = polyphenol_rank[0]

# Se o maior sensorial for o mesmo do maior polifenol, passamos para o segundo do ranking
if premium_cluster == functional_cluster:
    if len(polyphenol_rank) > 1:
        functional_cluster = polyphenol_rank[1]

# Mapeando nomes de Lotes com base nos clusters identificados
cluster_name_mapping = {}
for c in range(num_clusters):
    if c == premium_cluster:
        cluster_name_mapping[c] = "Amazonian Floral-Fruity Premium (Lot A)"
    elif c == functional_cluster:
        cluster_name_mapping[c] = "Deep Forest Antioxidant (Lot B)"
    elif num_clusters >= 3 and c == sensory_rank[min(1 if len(sensory_rank) > 1 else 0, len(sensory_rank)-1)] and c not in cluster_name_mapping.values():
        cluster_name_mapping[c] = "Traditional Balanced Trinitario (Lot C)"
    else:
        cluster_name_mapping[c] = f"Standard Agroforest Blend (Lot D-{c})"

df["Export_Lot_Category"] = df["Cluster"].map(cluster_name_mapping)

# Valoração Econômica Inteligente baseada no Modelo de Negócio
# Cacau genérico como commodity é R$ 15.00/kg. Cacaulogia valoriza cada lote de acordo com suas características
def calculate_pricing(row):
    category = row["Export_Lot_Category"]
    vol = row["Volume_Wet_kg"]
    
    if "Lot A" in category:
        price_per_kg = 58.00  # R$ 58/kg
    elif "Lot B" in category:
        price_per_kg = 38.00  # R$ 38/kg
    elif "Lot C" in category:
        price_per_kg = 32.00  # R$ 32/kg
    else:
        price_per_kg = 22.00  # R$ 22/kg (Default blend)
        
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
    "📦 Agrupamento de Lotes (Clustering Engine)", 
    "📈 Viabilidade & Projeção Financeira", 
    "🛠️ Inventário de Equipamentos & Infraestrutura"
])

# ==============================================================================
# TAB 1: AGRUPAMENTO DE LOTES (CLUSTERING ENGINE)
# ==============================================================================
with tab_clustering:
    # Dashboard de Métricas Consolidadas
    st.markdown("### 📊 Platform Impact Overview")
    col1, col2, col3, col4 = st.columns(4)

    total_volume = df["Volume_Wet_kg"].sum() / 1000  # em toneladas
    total_saas_val = df["SaaS_Total_Value_BRL"].sum()
    total_comm_val = df["Volume_Wet_kg"].sum() * 15.00
    net_added_value = total_saas_val - total_comm_val

    with col1:
        st.markdown("<div class='metric-card'><h4>TOTAL VOLUME AGGREGATED</h4><h2>{:.1f} Tons</h2><p>From 200 micro-producers</p></div>".format(total_volume), unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='metric-card'><h4>COMMODITY ESTIMATIVE</h4><h2>R$ {:,.2f}</h2><p>At R$ 15.00 / kg standard rate</p></div>".format(total_comm_val), unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='metric-card'><h4>CACAULOGY SaaS VALUATION</h4><h2>R$ {:,.2f}</h2><p>Standardized premium lots</p></div>".format(total_saas_val), unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='metric-card'><h4>NET ADDED VALUE CREATED</h4><h2>R$ {:,.2f}</h2><p style='color:#cc5500; font-weight:bold;'>+ {:.1f}% value multiplier</p></div>".format(net_added_value, (total_saas_val/total_comm_val-1)*100), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ Biochemical & Sensory Clustering Map")
    st.write("Each node represents an individual Amazonian family's harvest. The clustering engine groups them to form standardized industrial export lots.")

    # Definição segura do mapeamento de cores dinâmicas
    discrete_color_map = {
        "Amazonian Floral-Fruity Premium (Lot A)": "#cc5500",
        "Deep Forest Antioxidant (Lot B)": "#403228",
        "Traditional Balanced Trinitario (Lot C)": "#998c82"
    }
    # Preencher categorias extras com cores padrão
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
            "Total_Polyphenols_mgGAE_100g": "Total Polyphenols (mg GAE/100g) - [Dim II]",
            "Sensory_Score": "Sensory Score (QDA) - [Dim IV]",
            "Export_Lot_Category": "Standardized Export Category"
        }
    )
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="#faf7f3", font_color="#403228")
    st.plotly_chart(fig, use_container_width=True)

    # Detalhes das Categorias de Lotes
    st.markdown("### 📦 Standardized Export Lots Specification")
    lot_summary = df.groupby("Export_Lot_Category").agg(
        Producers=("Farmer_ID", "count"),
        Total_Volume_kg=("Volume_Wet_kg", "sum"),
        Mean_Sensory=("Sensory_Score", "mean"),
        Mean_Polyphenols=("Total_Polyphenols_mgGAE_100g", "mean"),
        Mean_Lipids=("Ethereal_Extract_Pct", "mean"),
        SaaS_Price=("SaaS_Price_BRL", "first"),
        Net_Added_Value=("Net_Added_Value_BRL", "sum")
    ).reset_index()

    st.dataframe(lot_summary.style.format({
        "Total_Volume_kg": "{:,.1f} kg",
        "Mean_Sensory": "{:.2f} / 10",
        "Mean_Polyphenols": "{:.1f} mg GAE/100g",
        "Mean_Lipids": "{:.2f}%",
        "SaaS_Price": "R$ {:.2f}/kg",
        "Net_Added_Value": "R$ {:,.2f}"
    }))

    # Tabela detalhada por produtor
    st.markdown("### 👪 Micro-Producer Allocations")
    st.write("Download this processed list and import to downstream logistical ERP or EUDR custom clearance portals.")
    st.dataframe(df[["Farmer_ID", "Family_Name", "Municipality", "Volume_Wet_kg", "Export_Lot_Category", "SaaS_Price_BRL", "SaaS_Total_Value_BRL", "Net_Added_Value_BRL"]])


# ==============================================================================
# TAB 2: VIABILIDADE ECONÔMICA & PROJEÇÃO DE LUCROS (ECONOMIC MODELER)
# ==============================================================================
with tab_finance:
    st.markdown("### 📈 Simulador de Negócios e Projeção Financeira de 5 Anos")
    st.write("Simule a viabilidade financeira e o retorno de investimento (ROI) ao aplicar a Cacaulogia na sua Cooperativa ou Indústria Média.")

    # Parâmetros Editáveis pelo Usuário
    st.markdown("#### ⚙️ Parâmetros Operacionais da Safra")
    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        annual_volume_tons = st.slider("Volume Anual Processado (Toneladas)", min_value=10.0, max_value=1000.0, value=float(total_volume), step=10.0)
    with col_input2:
        growth_rate = st.slider("Crescimento Anual de Volume Esperado (%)", min_value=5, max_value=50, value=15, step=5)
    with col_input3:
        comm_base_price = st.number_input("Preço Base de Compra da Commodity (R$/kg)", min_value=5.0, max_value=50.0, value=15.00, step=0.50)

    # Detalhes de CAPEX e OPEX
    st.markdown("#### 💰 Estrutura de Custos do Projeto")
    col_costs1, col_costs2 = st.columns(2)
    
    with col_costs1:
        st.markdown("**Investimentos Iniciais de Capital (CAPEX)**")
        capex_iot = st.number_input("Sensores IoT para Cochos (R$)", value=10000.0)
        capex_dryers = st.number_input("Automação de Estufas de Secagem (R$)", value=15000.0)
        capex_lab = st.number_input("Equipamentos básicos de laboratório (R$)", value=20000.0)
        capex_setup = st.number_input("Configuração de TI & Treinamento (R$)", value=15000.0)
        capex_eudr = st.number_input("Mapeamento Ambiental & Conformidade (R$)", value=25000.0)
        capex_pkg = st.number_input("Design de Embalagens & QR Codes (R$)", value=10000.0)
        total_capex = capex_iot + capex_dryers + capex_lab + capex_setup + capex_eudr + capex_pkg
        st.warning(f"**Total CAPEX: R$ {total_capex:,.2f}**")

    with col_costs2:
        st.markdown("**Custos Operacionais Anuais (OPEX)**")
        opex_saas = st.number_input("Assinatura Anual do Software SaaS (R$)", value=12000.0)
        opex_maint = st.number_input("Manutenção Física de Sensores (R$)", value=5000.0)
        opex_logistics = st.number_input("Logística de Técnicos de Campo (R$)", value=20000.0)
        opex_audits = st.number_input("Auditorias Externas de Certificação (R$)", value=15000.0)
        total_opex_base = opex_saas + opex_maint + opex_logistics + opex_audits
        st.warning(f"**Total OPEX Anual: R$ {total_opex_base:,.2f}**")

    # Cálculos Ponderados de Receita Baseados no Agrupamento Atual
    # Vamos calcular os percentuais de cada lote de forma dinâmica
    total_samples = len(df)
    prop_lot_a = len(df[df["Export_Lot_Category"].str.contains("Lot A")]) / total_samples
    prop_lot_b = len(df[df["Export_Lot_Category"].str.contains("Lot B")]) / total_samples
    prop_lot_c = len(df[df["Export_Lot_Category"].str.contains("Lot C")]) / total_samples
    prop_lot_d = 1.0 - (prop_lot_a + prop_lot_b + prop_lot_c)

    weighted_saas_price = (prop_lot_a * 58.00) + (prop_lot_b * 38.00) + (prop_lot_c * 32.00) + (prop_lot_d * 22.00)
    
    # Simulação de 5 Anos
    years = [f"Ano {i}" for i in range(1, 6)]
    vol_proj = []
    rev_comm_proj = []
    rev_saas_proj = []
    net_added_val_proj = []
    opex_proj = []
    materia_prima_proj = []
    net_profit_proj = []
    cum_cash_flow = []

    current_volume_kg = annual_volume_tons * 1000
    cum_cash = -total_capex

    for y in range(1, 6):
        # Crescimento de volume
        vol_y = current_volume_kg if y == 1 else vol_proj[-1] * (1 + growth_rate/100)
        vol_proj.append(vol_y)
        
        # Receitas
        rev_comm = vol_y * comm_base_price
        rev_comm_proj.append(rev_comm)
        
        rev_saas = vol_y * weighted_saas_price
        rev_saas_proj.append(rev_saas)
        
        net_add = rev_saas - rev_comm
        net_added_val_proj.append(net_add)
        
        # OPEX escala um pouco com o volume (+3% ao ano de inflação operacional)
        opex_y = total_opex_base if y == 1 else opex_proj[-1] * 1.03
        opex_proj.append(opex_y)
        
        # Custo de Compra da Matéria-Prima (com repasse justo de prêmio de ~55% do ágio ao produtor)
        # Ex: repasse justo ao produtor na compra (K-Variable incentivizada)
        recompensa_produtor_por_kg = comm_base_price + ((weighted_saas_price - comm_base_price) * 0.55)
        mp_cost = vol_y * recompensa_produtor_por_kg
        materia_prima_proj.append(mp_cost)
        
        # Lucro líquido retido da cooperativa
        profit_y = rev_saas - mp_cost - opex_y
        if y == 1:
            profit_y -= total_capex # Deduz CAPEX no primeiro ano
        net_profit_proj.append(profit_y)
        
        # Fluxo de Caixa Acumulado
        cum_cash += (rev_saas - mp_cost - opex_y)
        cum_cash_flow.append(cum_cash)

    # DataFrame de Prospecção
    df_proj = pd.DataFrame({
        "Ano": years,
        "Volume Processado (kg)": vol_proj,
        "Receita Commodity (R$)": rev_comm_proj,
        "Receita Cacaulogia (R$)": rev_saas_proj,
        "Custo Matéria-Prima Justa (R$)": materia_prima_proj,
        "OPEX (R$)": opex_proj,
        "Lucro Líquido Retido (R$)": net_profit_proj,
        "Fluxo de Caixa Acumulado (R$)": cum_cash_flow
    })

    # Visualização de Retorno Gráfico
    st.markdown("#### 📊 Projeção de Margem & Fluxo de Caixa Acumulado")
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        # Gráfico Comparativo de Receita
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(x=years, y=rev_comm_proj, name="Receita Tradicional (Commodity)", marker_color="#998c82"))
        fig_rev.add_trace(go.Bar(x=years, y=rev_saas_proj, name="Receita Cacaulogia SaaS", marker_color="#cc5500"))
        fig_rev.update_layout(title="Comparativo de Receita Anual (R$)", barmode="group", paper_bgcolor="white", plot_bgcolor="#faf7f3")
        st.plotly_chart(fig_rev, use_container_width=True)

    with col_plot2:
        # Gráfico de Payback / Fluxo Acumulado
        fig_cash = go.Figure()
        fig_cash.add_trace(go.Scatter(x=years, y=cum_cash_flow, mode="lines+markers", name="Fluxo de Caixa", line=dict(color="#403228", width=3), marker=dict(size=8)))
        fig_cash.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Ponto de Equilíbrio (Payback)")
        fig_cash.update_layout(title="Fluxo de Caixa Líquido Acumulado (R$)", paper_bgcolor="white", plot_bgcolor="#faf7f3")
        st.plotly_chart(fig_cash, use_container_width=True)

    # Tabela de Dados Formatada
    st.markdown("#### 📄 Demonstrativo de Resultados Projetados (DRE)")
    st.dataframe(df_proj.style.format({
        "Volume Processado (kg)": "{:,.1f} kg",
        "Receita Commodity (R$)": "R$ {:,.2f}",
        "Receita Cacaulogia (R$)": "R$ {:,.2f}",
        "Custo Matéria-Prima Justa (R$)": "R$ {:,.2f}",
        "OPEX (R$)": "R$ {:,.2f}",
        "Lucro Líquido Retido (R$)": "R$ {:,.2f}",
        "Fluxo de Caixa Acumulado (R$)": "R$ {:,.2f}"
    }))

    st.markdown("""
    *   **Ágio Médio Estimado:** O modelo calcula dinamicamente que o preço médio ponderado obtido pelo portfólio de lotes da Cacaulogia é de **R$ {:.2f}/kg** (contra R$ {:.2f}/kg da commodity comum).
    *   **Impacto Social Integrado (K-Variable):** O Custo de Matéria-Prima Justa garante um retorno direto de mais de **R$ {:.2f} acumulados em 5 anos** para as famílias produtoras associadas, incentivando a conservação da floresta ativa em Sistemas Agroflorestais.
    """.format(weighted_saas_price, comm_base_price, sum(materia_prima_proj)))


# ==============================================================================
# TAB 3: INVENTÁRIO DE EQUIPAMENTOS (PHYSICAL ASSETS GUIDE)
# ==============================================================================
with tab_equipment:
    st.markdown("### 🛠️ Equipamentos Necessários Além da Plataforma Cacaulogia")
    st.write("Para operacionalizar o framework científico e coletar dados confiáveis em campo e no laboratório, a Cooperativa ou Indústria Média deve adquirir os seguintes ativos de hardware e instrumentalização:")

    col_eq1, col_plot_eq = st.columns([3, 2])
    
    with col_eq1:
        st.markdown("<div class='equip-card'><h4>📡 1. Infraestrutura IoT & Pós-Colheita (Dimensões I & III)</h4><ul><li><b>Sondas Térmicas DS18B20 Seladas:</b> Inseridas diretamente na massa de cacau fermentando dentro dos cochos de madeira para coletar curvas dinâmicas de bioprocessos.</li><li><b>Microcontroladores ESP32 de Baixo Custo:</b> Equipados com caixas protetoras à prova d'água para registrar as temperaturas e transmitir via rede local ou armazenar de forma offline-first.</li><li><b>Sensores Ambientais de Estufa (DHT22 / SHT31):</b> Instalados nas estufas solares de secagem para rastrear umidade relativa e temperatura interna do ar.</li><li><b>Exaustores de Ar Automatizados:</b> Ativados quando a umidade interna da estufa ultrapassa a faixa ideal, controlando ativamente a taxa de secagem das amêndoas.</li></ul></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='equip-card'><h4>🔬 2. Equipamentos Analíticos & Químicos de Laboratório (Dimensão II)</h4><ul><li><b>Liofilizador de Bancada (Lyophilizer):</b> Utilizado para desidratar a frio amostras frescas de cotilédones (a -50 ºC e 0.04 mbar) preservando os polifenóis ativos e alcaloides sem degradação térmica para as análises subsequentes.</li><li><b>Balança Analítica de Precisão (0.1 mg):</b> Essencial para pesagem precisa de frações de amêndoas e controle de extração química.</li><li><b>Moinho Analítico de Facas:</b> Utilizado para triturar uniformemente as amêndoas liofilizadas até atingirem granulação homogênea de 40 mesh (600 a 420 µm).</li><li><b>Banho de Ultrassom com Controle Térmico:</b> Para extração assistida por ultrassom (25 Hz a 50 ºC) dos compostos bioativos (polifenóis e alcaloides: teobromina e cafeína).</li><li><b>Centrífuga de Alta Velocidade (14.000 rpm):</b> Utilizada para separação rápida de fases e clarificação dos extratos antes das análises cromatográficas e moleculares.</li><li><b>Leitor de Placas de Microplacas (Spectrophotometer):</b> Para quantificar Polifenóis Totais (Folin-Ciocalteu a 760 nm) e Antocianinas (razão de absorbância a 460/530 nm).</li></ul></div>", unsafe_allow_html=True)
        
        st.markdown("<div class='equip-card'><h4>👅 3. Estrutura para Análise Sensorial Calibrada (Dimensão IV)</h4><ul><li><b>Cabines de Degustação Padronizadas (ISO 8586):</b> Com isolamento térmico (24 ± 0.2 ºC), controle acústico e de umidade relativa (53% a 60%) para evitar interferências sensoriais nos provadores.</li><li><b>Kit de Referências Olfativas de Padrão Comercial:</b> Compostos puros de Ácido Isovalérico, Ácido Capróico, Álcool Isoamílico e Benzaldeído para calibração, treinamento e memorização do painel descritivo quantitativo (QDA).</li></ul></div>", unsafe_allow_html=True)
        
    with col_plot_eq:
        st.markdown("#### 🎯 Alvos Físico-Químicos de Qualidade")
        st.write("Os equipamentos analíticos descritos são calibrados para mapear o cacau dentro das faixas ótimas científicas:")
        
        # Adicionar uma imagem conceitual de targets ou mini-gráfico ilustrativo
        target_labels = ["Manteiga (%)", "Polifenóis Máximos", "Nota Sensorial QDA", "Acidez Volátil (pH)"]
        target_values = [58.9, 100.0, 86.0, 96.0] # percentuais do alvo ideal
        
        fig_target = go.Figure(go.Bar(
            x=target_values,
            y=target_labels,
            orientation='h',
            marker_color='#403228',
            text=[f"{v}% do Ideal" for v in target_values],
            textposition='auto'
        ))
        fig_target.update_layout(
            title="Calibração de Ativos ( targets )",
            xaxis_title="Proximidade ao Perfil Ótimo (%)",
            paper_bgcolor="white",
            plot_bgcolor="#faf7f3"
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
        st.markdown("""
        > **Parceria Estratégica Regional:** A aquisição de equipamentos de laboratório analítico de alta tecnologia pode ser significativamente barateada ou substituída através de **convênios de cooperação de pesquisa** com laboratórios públicos de universidades na Amazônia (UFPA, IFPA, UFRA, CEPLAC). A cooperativa utiliza os sensores IoT para dados correntes e envia amostras quinzenais para análise varietal (genotipagem SNP) nos laboratórios parceiros, mantendo o OPEX controlado.
        """)
