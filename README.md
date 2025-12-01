# LexAI - Correção Automática de Redações ENEM

Sistema de correção automática de redações do ENEM utilizando Inteligência Artificial. O LexAI combina tecnologias de OCR (Reconhecimento Óptico de Caracteres) e processamento de linguagem natural para fornecer correções detalhadas seguindo os critérios oficiais do ENEM.

## 📋 Sobre o Projeto

O LexAI foi desenvolvido como trabalho final para a disciplina **MIN709 - Aplicações em Ciência de Dados**. O projeto visa democratizar o acesso à correção de redações, permitindo que estudantes recebam feedback detalhado e imediato sobre suas redações manuscritas.

### Funcionalidades

- ✅ Extração de texto de redações manuscritas via OCR
- ✅ Correção automática seguindo as 5 competências do ENEM
- ✅ Avaliação detalhada com notas (0-200 por competência, 0-1000 total)
- ✅ Justificativas baseadas no texto do aluno
- ✅ Sugestões práticas de melhoria
- ✅ Recomendações de material de apoio

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Framework para interface web
- **olmOCR-2-7B-1025-FP8**: Modelo de OCR desenvolvido pela AllenAI para extração de texto manuscrito
- **Google Gemini 2.5 Flash Lite**: Modelo de linguagem para correção de redações
- **PyTorch**: Framework de deep learning
- **Transformers**: Biblioteca Hugging Face para modelos de IA
- **Pillow**: Processamento de imagens

## 📦 Requisitos do Sistema

### Hardware

- **RAM**: Mínimo 8GB (recomendado 16GB+)
- **Espaço em disco**: ~20GB para cache do modelo OCR
- **GPU**: Opcional, mas recomendada (NVIDIA CUDA ou Apple MPS)
  - Sem GPU: processamento em CPU (mais lento, ~1-2 minutos por redação)
  - Com GPU: processamento mais rápido (~20-40 segundos por redação)

### Software

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

## 🚀 Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd LexAI
```

### 2. Crie um ambiente virtual (recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

**Nota:** A primeira instalação pode levar alguns minutos devido ao tamanho das bibliotecas.

### 4. Obtenha sua API Key do Google Gemini

1. Acesse: https://makersuite.google.com/app/apikey
2. Crie uma conta ou faça login
3. Gere uma nova API Key
4. Copie a chave (você precisará dela ao executar a aplicação)

## 🎯 Como Usar

### 1. Iniciar a aplicação

```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no seu navegador (geralmente em `http://localhost:8501`).

### 2. Configurar API Key

1. Na sidebar (barra lateral), insira sua **API Key do Google Gemini**
2. A chave será armazenada apenas na sessão atual

### 3. Processar uma redação

1. **Faça upload da imagem** da redação manuscrita (formatos: PNG, JPG, JPEG)
2. Verifique o preview da imagem
3. Clique em **"Processar Redação"**
4. Aguarde o processamento:
   - **Primeira vez**: O modelo OCR será baixado (~15-20GB) - pode levar vários minutos
   - **Processamento OCR**: ~20-40 segundos (CPU) ou ~10-20 segundos (GPU)
   - **Correção Gemini**: ~10-15 segundos

### 4. Visualizar resultados

- **Texto Extraído (OCR)**: Clique no expander para ver o texto extraído da imagem
- **Correção ENEM**: Visualize a correção completa com:
  - Notas por competência
  - Justificativas detalhadas
  - Nota final
  - Sugestões de melhoria

## 📁 Estrutura do Projeto

```
LexAI/
├── app.py                 # Aplicação principal Streamlit
├── requirements.txt       # Dependências do projeto
├── README.md             # Este arquivo
├── .gitignore            # Arquivos ignorados pelo Git
```

### Detecção automática de dispositivo

A aplicação detecta automaticamente o melhor dispositivo disponível:
- **CUDA**: GPU NVIDIA (mais rápido)
- **MPS**: GPU Apple Silicon (Mac com chip M1/M2/M3)
- **CPU**: Processamento em CPU (funciona em qualquer sistema, mais lento)


## 📊 Sobre o Modelo OCR

O **olmOCR-2-7B-1025-FP8** é um modelo de OCR de última geração desenvolvido pela AllenAI:

- **Baseado em**: Qwen2.5-VL-7B-Instruct
- **Treinado com**: olmOCR-mix-1025 dataset
- **Otimizado para**: Texto manuscrito e documentos
- **Tamanho**: ~15-20GB (quantizado em FP8)
- **Performance**: Alta precisão em texto manuscrito

**Referências:**
- [Modelo no Hugging Face](https://huggingface.co/allenai/olmOCR-2-7B-1025-FP8)
- [Repositório GitHub](https://github.com/allenai/olmocr)

## 📝 Sobre a Correção ENEM

A correção segue rigorosamente as **5 competências do ENEM**:

1. **Competência 1**: Demonstrar domínio da modalidade escrita formal da Língua Portuguesa
2. **Competência 2**: Compreender a proposta de redação e aplicar conceitos das várias áreas de conhecimento
3. **Competência 3**: Selecionar, relacionar, organizar e interpretar informações, fatos, opiniões e argumentos
4. **Competência 4**: Demonstrar conhecimento dos mecanismos linguísticos necessários para a construção da argumentação
5. **Competência 5**: Elaborar proposta de intervenção para o problema abordado

Cada competência é avaliada de 0 a 200 pontos, totalizando 1000 pontos.

## 👥 Autores

Enzo Fávaro - 22.00774-0

Iago Aurichio - 21.00236-3

Desenvolvido como trabalho final para MIN709 - Aplicações em Ciência de Dados. - IMT

## 🙏 Agradecimentos

- **AllenAI** pelo modelo olmOCR
- **Google** pelo modelo Gemini
- **Hugging Face** pela infraestrutura de modelos


