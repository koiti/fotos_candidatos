"""
Detector de Acessórios de Cabeça em Fotos de Candidatos (TSE / Urna Eleitoral)
================================================================================

Detecta acessórios cefálicos proibidos em fotos de candidatos com limiar de 
confiança ESTRITO de 90% (>0.90), utilizando Hugging Face Transformers (YOLOS-Fashionpedia).

Entrada Padrão: 'foto_cand2024_SP' (ou 'amostras')
Saída: 'irregular_acessorios_cabeca/' (armazena APENAS detecções com conf > 90%)
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from transformers import pipeline

# Otimização de threads PyTorch/OpenMP
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

CONFIDENCE_THRESHOLD = 0.90
BATCH_SIZE = 32
NUM_WORKERS = 16

CLASSES_CABECA_MAP = {
    "hat": "chapéu/boné",
    "headband, head covering, hair accessory": "acessório de cabeça",
    "hood": "capuz"
}

BASE_DIR = Path(r"c:\Users\steve\OneDrive\Documentos\UFABC\2026-TOPICOS_DE_IA")
AMOSTRAS_DIR = BASE_DIR / "foto_cand2024_SP"
OUTPUT_DIR = BASE_DIR / "irregular_acessorios_cabeca"

# ============================================================================
# AUXILIARES
# ============================================================================

def carregar_imagem(path):
    try:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception:
        return None

def salvar_checkpoint(relatorio_path, data_relatorio):
    temp_path = relatorio_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data_relatorio, f, ensure_ascii=False, indent=2)
    if relatorio_path.exists():
        relatorio_path.unlink()
    temp_path.rename(relatorio_path)

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def executar_deteccao_acessorios_cabeca(input_dir=None, output_dir=None, conf_thresh=CONFIDENCE_THRESHOLD, max_fotos=None):
    print("=" * 70)
    print("DETECTOR DE ACESSÓRIOS DE CABEÇA — YOLOS-Fashionpedia (> 90%)")
    print("=" * 70)

    target_input = Path(input_dir) if input_dir else AMOSTRAS_DIR
    if not target_input.exists():
        target_input = BASE_DIR / "amostras"
        if not target_input.exists():
            print(f"Erro: Pasta de entrada '{target_input}' não encontrada!")
            return

    target_output = Path(output_dir) if output_dir else OUTPUT_DIR
    target_output.mkdir(exist_ok=True, parents=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    fotos = sorted([
        f for f in target_input.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ])

    if max_fotos:
        fotos = fotos[:max_fotos]

    if not fotos:
        print(f"Nenhuma foto encontrada em '{target_input}'!")
        return

    print(f"Diretório de entrada: '{target_input.name}' ({len(fotos)} fotos)")
    print(f"Diretório de saída: '{target_output.name}'")
    print(f"Limiar de confiança: {conf_thresh:.0%}")
    print(f"Tamanho do lote: {BATCH_SIZE} imagens | Workers I/O: {NUM_WORKERS}\n")

    print("Carregando modelo YOLOS-Fashionpedia (Hugging Face)...")
    try:
        device = 0 if torch.cuda.is_available() else -1
        detector = pipeline("object-detection", model="valentinafevu/yolos-fashionpedia", device=device)
        print("Modelo YOLOS-Fashionpedia pronto com sucesso!\n")
    except Exception as e:
        print(f"Erro ao carregar o modelo YOLOS-Fashionpedia: {e}")
        return

    relatorio_path = target_output / "relatorio.json"
    resultados_detalhados = []
    total_irregulares = 0
    processadas = set()

    if relatorio_path.exists():
        try:
            with open(relatorio_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                if saved_data.get("limiar_confianca") == conf_thresh:
                    resultados_detalhados = saved_data.get("resultados", [])
                    total_irregulares = saved_data.get("total_irregulares", 0)
                    processadas = {r["arquivo"] for r in resultados_detalhados if "arquivo" in r}
                    print(f"-> Progresso anterior carregado: {len(processadas)} fotos já analisadas.")
                    print(f"   Retomando a partir da foto {len(processadas) + 1}...\n")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar progresso anterior ({e}). Iniciando do zero.\n")

    fotos_para_processar = [f for f in fotos if f.name not in processadas]

    if not fotos_para_processar:
        print("Todas as fotos do diretório já foram analisadas!")
        print(f"Relatório final em: {relatorio_path}")
        return

    tempo_inicio = time.time()
    total_fotos_count = len(fotos)
    executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

    for b_idx in range(0, len(fotos_para_processar), BATCH_SIZE):
        batch_files = fotos_para_processar[b_idx:b_idx + BATCH_SIZE]
        imgs = list(executor.map(carregar_imagem, batch_files))

        try:
            valid_pairs = [(f, img) for f, img in zip(batch_files, imgs) if img is not None]
            if not valid_pairs:
                continue

            valid_files, valid_imgs = zip(*valid_pairs)
            results = detector(list(valid_imgs), threshold=conf_thresh)

            for foto_path, img_rgb, res_list in zip(valid_files, valid_imgs, results):
                irregularidades = []

                for item in res_list:
                    lbl_en = item["label"]
                    c_val = float(item["score"])
                    box = item["box"]

                    if lbl_en in CLASSES_CABECA_MAP and c_val >= conf_thresh:
                        lbl_pt = CLASSES_CABECA_MAP[lbl_en]
                        irregularidades.append({
                            "classe_en": lbl_en,
                            "classe_pt": lbl_pt,
                            "confianca": round(c_val, 4),
                            "bbox": [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
                        })

                if irregularidades:
                    total_irregulares += 1
                    draw = ImageDraw.Draw(img_rgb)

                    for det in irregularidades:
                        b = det["bbox"]
                        label_pt = det["classe_pt"]
                        conf = det["confianca"]

                        draw.rectangle(b, outline="red", width=4)
                        text = f"{label_pt} ({conf:.1%})"
                        try:
                            font = ImageFont.load_default()
                        except Exception:
                            font = None

                        draw.text((b[0] + 5, max(0, b[1] - 15)), text, fill="red", font=font)

                    img_rgb.save(target_output / foto_path.name)
                    print(f" -> DETECTADO: {foto_path.name} | {irregularidades[0]['classe_pt']} ({irregularidades[0]['confianca']:.1%})")

                resultados_detalhados.append({
                    "arquivo": foto_path.name,
                    "status": "irregular" if irregularidades else "regular",
                    "total_irregularidades": len(irregularidades),
                    "irregularidades": irregularidades
                })

            current_processed = len(processadas) + b_idx + len(batch_files)
            elapsed = time.time() - tempo_inicio
            fps = (b_idx + len(batch_files)) / max(0.1, elapsed)
            print(f"Progresso: [{current_processed}/{total_fotos_count}] fotos analisadas ({fps:.1f} fotos/s) | Irregulares: {total_irregulares}")

            relatorio_data = {
                "data_analise": datetime.now().isoformat(),
                "modelo": "valentinafevu/yolos-fashionpedia",
                "limiar_confianca": conf_thresh,
                "total_fotos": total_fotos_count,
                "total_irregulares": total_irregulares,
                "input_dir": str(target_input),
                "output_dir": str(target_output),
                "resultados": resultados_detalhados
            }
            salvar_checkpoint(relatorio_path, relatorio_data)

        except Exception as e:
            print(f"Erro no processamento do lote: {e}")

    executor.shutdown()
    tempo_total = time.time() - tempo_inicio

    print("\n" + "=" * 70)
    print("RESUMO FINAL — ACESSÓRIOS DE CABEÇA (> 90% CERTEZA)")
    print("=" * 70)
    print(f"Total de fotos analisadas: {total_fotos_count}")
    print(f"Fotos irregulares salvas em '{target_output.name}': {total_irregulares}")
    print(f"Tempo desta execução: {tempo_total:.2f} segundos")
    print(f"Relatório de análise salvo em: {relatorio_path}")

if __name__ == "__main__":
    executar_deteccao_acessorios_cabeca()
