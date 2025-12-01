# -*- coding: utf-8 -*-
"""
Aplicação Streamlit para Correção Automática de Redações ENEM
MIN709 - Aplicações em Ciência de Dados
"""

import os

# IMPORTANTE: Configurar variáveis de ambiente ANTES de importar transformers/huggingface
# Configurar cache do Hugging Face para disco D (opcional - ajuste conforme necessário)
# Descomente as linhas abaixo se quiser usar disco D para cache
# os.environ['HF_HOME'] = r'D:\huggingface_cache'
# os.environ['HF_HUB_CACHE'] = r'D:\huggingface_cache\hub'
# os.environ['TRANSFORMERS_CACHE'] = r'D:\huggingface_cache\transformers'

import streamlit as st
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import google.generativeai as genai
from io import BytesIO

# Configuração da página
st.set_page_config(
    page_title="LexAI - Correção Automática ENEM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para interface elegante
st.markdown("""
    <style>
    .project-name {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Título e descrição
st.markdown('<div class="project-name">LexAI</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">Correção Automática de Redações ENEM</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Faça upload da imagem da sua redação manuscrita e receba uma correção detalhada seguindo os critérios do ENEM</p>', unsafe_allow_html=True)

# Função para detectar dispositivo disponível
def detectar_dispositivo():
    """Detecta o melhor dispositivo disponível (CUDA, ROCm, MPS, ou CPU)"""
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps", torch.device("mps")
    else:
        return "cpu", torch.device("cpu")

# Sidebar para configurações
with st.sidebar:
    st.header("Configurações")
    
    # Campo para API Key do Gemini
    api_key = st.text_input(
        "API Key do Google Gemini",
        type="password",
        help="Insira sua chave de API do Google Gemini. Você pode obter uma em https://makersuite.google.com/app/apikey"
    )
    

# Cache do modelo OCR
@st.cache_resource
def carregar_modelo_ocr():
    """Carrega o modelo OCR uma vez e mantém em cache"""
    model_id = "allenai/olmOCR-2-7B-1025-FP8"
    
    # Detectar dispositivo
    device_name, device = detectar_dispositivo()
    
    with st.spinner(f"Carregando modelo OCR no dispositivo: {device_name.upper()}... Isso pode levar alguns minutos na primeira vez."):
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Para CPU ou dispositivos não-CUDA, usar device_map="cpu" explicitamente
        # Para CUDA, usar "auto" permite distribuição automática
        if device_name == "cpu":
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                dtype=torch.float32,
                device_map="cpu"
            ).eval()
        else:
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    dtype=torch.float32,
                    device_map="auto"
                ).eval()
            except Exception as e:
                # Fallback para CPU se houver erro com GPU
                st.warning(f"Erro ao usar {device_name}, usando CPU: {str(e)}")
                model = AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    dtype=torch.float32,
                    device_map="cpu"
                ).eval()
                device_name = "cpu"
                device = torch.device("cpu")
    
    return processor, model, device_name, device

def extrair_texto_ocr(image, processor, model, device):
    """Extrai texto de uma imagem usando OCR"""
    # Converte para RGB se necessário
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        image = Image.open(image).convert("RGB")
    
    # Reduz tamanho para 1288px (recomendação oficial)
    max_size = 1288
    ratio = max_size / max(image.size)
    if ratio < 1:
        new_w = int(image.size[0] * ratio)
        new_h = int(image.size[1] * ratio)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    PROMPT = """Attached is a page of a document. Return ONLY the clean plain text transcription.
Extract handwritten text carefully in natural reading order."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    
    # Mover inputs para o dispositivo correto
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=0.1,
            do_sample=False
        )
    
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_len:]
    
    texto = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )[0]
    
    return texto.strip()

def corrigir_redacao_enem(texto_ocr, api_key):
    """Corrige a redação usando Google Gemini seguindo critérios ENEM"""
    genai.configure(api_key=api_key)
    
    prompt_enem = """
Você é um corretor oficial do ENEM. Avalie a redação enviada pelo aluno seguindo rigorosamente as 5 competências do ENEM.

Para cada competência forneça:
- Nota (0–200)
- Justificativa detalhada baseada no texto do aluno
- Trechos que indicam problemas ou acertos

No final, forneça:
- A nota final (0–1000)
- Sugestões práticas de melhoria
- Possível material de apoio para melhora na redação

Avalie o seguinte texto do aluno:
"""
    
    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    
    resposta = model.generate_content(
        prompt_enem + texto_ocr,
        safety_settings=None,
    )
    
    return resposta.text

# Área principal da aplicação
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload da Redação")
    
    # Upload de imagem
    uploaded_file = st.file_uploader(
        "Escolha uma imagem da redação",
        type=["png", "jpg", "jpeg"],
        help="Faça upload de uma imagem da sua redação manuscrita"
    )
    
    image_preview = None
    if uploaded_file is not None:
        image_preview = Image.open(uploaded_file)
    
    # Preview da imagem
    if image_preview is not None:
        st.image(image_preview, caption="Preview da redação", use_container_width=True)

with col2:
    st.header("Processamento")
    
    # Botão para processar
    processar = st.button(
        "Processar Redação",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None or not api_key)
    )
    
    if uploaded_file is None:
        st.warning("Por favor, faça upload de uma imagem da redação.")
    
    if not api_key:
        st.warning("Por favor, insira sua API Key do Gemini na sidebar.")

# Processamento
if processar and uploaded_file is not None and api_key:
    try:
        # Carregar modelo OCR (com cache)
        processor, model, device_name, device = carregar_modelo_ocr()
        
        # Carregar imagem para processamento
        if isinstance(uploaded_file, BytesIO):
            uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        
        # Extração OCR
        with st.spinner("Extraindo texto da imagem... Isso pode levar 20-40 segundos."):
            texto_ocr = extrair_texto_ocr(image, processor, model, device)
        
        # Exibir resultado OCR
        st.divider()
        st.header("Texto Extraído (OCR)")
        
        with st.expander("Ver texto extraído", expanded=False):
            st.text_area("Texto OCR:", texto_ocr, height=200, disabled=True)
        
        # Correção ENEM
        with st.spinner("Corrigindo redação com critérios ENEM... Aguarde."):
            correcao = corrigir_redacao_enem(texto_ocr, api_key)
        
        # Exibir correção
        st.divider()
        st.header("Correção ENEM")
        
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(correcao)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.success("Processamento concluído com sucesso!")
        
    except Exception as e:
        st.error(f"Erro durante o processamento: {str(e)}")
        st.exception(e)

# Rodapé
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p style="font-size: 1.3rem; font-weight: bold; color: #667eea; margin-bottom: 0.5rem;">LexAI</p>
    <p><strong>MIN709 - Aplicações em Ciência de Dados</strong></p>
    <p>Correção automática de redações usando olmOCR e Google Gemini</p>
</div>
""", unsafe_allow_html=True)

