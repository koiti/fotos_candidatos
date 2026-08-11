# Detecção Automatizada de Acessórios em Fotografias de Candidatos Eleitorais

Este projeto realiza a verificação e detecção automatizada de **irregularidades eleitorais em fotografias de candidatos**, garantindo o cumprimento das diretrizes do Tribunal Superior Eleitoral (TSE).

A solução utiliza uma arquitetura unificada baseada em **Visão Computacional e Inteligência Artificial Zero-Shot**:
1. **Acessórios de Cabeça**: Identificação de chapéus, bonés, toucas, tiaras, capuzes, turbantes, capacetes, etc., utilizando o modelo **YOLO-World v2 (Zero-Shot)** com limiar de confiança configurável (padrão $\ge 90\%$).
2. **Óculos Escuros**: Detecção de lentes escuras de sol que ocultam os olhos do candidato, via filtro híbrido composto por **YOLO-World** (localização da região dos olhos/óculos) + **CLIP Zero-Shot** (classificação entre óculos escuros opacos e óculos de grau transparentes, com limiar $\ge 90\%$).

---

## 📁 Estrutura de Arquivos Principais

| Arquivo | Descrição |
| :--- | :--- |
| `detectar_irregularidades.py` | Script em Python (CLI) para execução em ambiente local ou servidores. |
| `detectar_irregularidades.ipynb` | Notebook Jupyter otimizado para execução interativa no **Google Colab**. |
| `amostras/` | Diretório padrão contendo fotografias de amostra para testes e validação. |

---

## 🚀 Como Executar

### 1. Execução via Script Python (`detectar_irregularidades.py`)

O script Python pode ser executado via linha de comando (terminal).

#### A. Execução Padrão (utilizando a pasta `amostras`)
Por padrão, o script analisa as imagens presentes na pasta `amostras/`:

```bash
python detectar_irregularidades.py
```

#### B. Execução com Dataset do TSE (Eleições 2024)
Para processar dados do conjunto oficial de candidatos disponibilizado no portal do TSE:
1. Acesse o dataset: [TSE - Dados Abertos Candidatos 2024](https://dadosabertos.tse.jus.br/dataset/candidatos-2024).
2. Baixe e extraia o pacote de fotos dos candidatos (ex: `foto_cand2024_SP`).
3. Execute o script apontando o diretório de entrada para as fotos baixadas:

```bash
python detectar_irregularidades.py --input_dir foto_cand2024_SP --output_dir irregularidades_2024_sp
```

#### Parâmetros da Linha de Comando (CLI):

- `--input_dir`: Diretório contendo as fotografias para análise *(default: `amostras`)*.
- `--output_dir`: Diretório onde serão salvas as fotos com irregularidades anotadas *(default: `irregularidades`)*.
- `--conf_thresh`: Limiar de confiança para detecção de acessórios de cabeça *(default: `0.90`)*.
- `--clip_thresh`: Limiar de probabilidade CLIP para óculos escuros *(default: `0.90`)*.
- `--batch_size`: Tamanho do lote (*batch size*) para inferência em GPU/CPU *(default: `32`)*.

---

### 2. Execução via Google Colab (`detectar_irregularidades.ipynb`)

O notebook `detectar_irregularidades.ipynb` foi desenvolvido para execução no ambiente [Google Colab](https://colab.research.google.com/).

#### Funcionalidades no Colab:
- **Instalação Automática**: Instala as dependências necessárias (`ultralytics`, `transformers`, `pillow`, etc.) na sessão do Colab.
- **Amostras do GitHub**: Baixa automaticamente o repositório de amostras disponibilizado em [https://github.com/koiti/fotos_candidatos](https://github.com/koiti/fotos_candidatos) para testes rápidos (Opção A).
- **Processamento de Grandes Datasets**: Permite integrar com o Google Drive para descompactar e processar conjuntos maiores (ex: `foto_cand2024_SP_div.zip` contendo dezenas de milhares de fotos) diretamente no SSD acelerado por GPU do Colab (Opção B).
- **Visualização Interativa**: Exibe os resultados, tabelas de métricas e imagens com caixas delimitadoras diretamente no notebook.

---

## 🛠️ Pré-requisitos e Instalação

Para rodar o script `detectar_irregularidades.py` em seu ambiente local, certifique-se de possuir o Python 3.8+ e instale as dependências:

```bash
pip install torch ultralytics transformers pillow pandas numpy scikit-learn ftfy
```

---

## 📊 Resultados e Métricas

Após o processamento, os seguintes artefatos são gerados no diretório de saída:
- **Fotos Anotadas**: Imagens em que foram encontradas irregularidades, demarcadas com caixas delimitadoras (*bounding boxes*) e identificação da classe com percentual de confiança.
- **Relatório JSON (`relatorio.json`)**: Arquivo contendo a data da análise, configurações de limiar utilizados, total de imagens processadas e irregulares, tempo total e métricas de desempenho ($mAP@50$, Precisão, Recall e F1-Score) por classe.
