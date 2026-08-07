"""
Detector de Óculos Escuros em Fotos de Candidatos (YOLOS + CLIP)
================================================================================

Detecta óculos escuros com limiar de confiança ESTRITO de 90% (>0.90) utilizando
a arquitetura híbrida YOLOS-Fashionpedia (Detecção Hugging Face) + CLIP (Validação Semântica).

Entrada Padrão: 'foto_cand2024_SP' (ou 'amostras')
Saída: 'irregular_oculos_escuros/' (armazena APENAS detecções com conf > 90%)
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from transformers import pipeline, CLIPProcessor, CLIPModel

# Otimização de threads PyTorch/OpenMP
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

CLIP_SUNGLASSES_THRESHOLD = 0.90
BATCH_SIZE = 32
NUM_WORKERS = 16

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_PROMPTS = [
    "dark black tinted sunglasses covering eyes",
    "clear transparent prescription reading eyeglasses with visible eyes"
]

BASE_DIR = Path(r"c:\Users\steve\OneDrive\Documentos\UFABC\2026-TOPICOS_DE_IA")
AMOSTRAS_DIR = BASE_DIR / "foto_cand2024_SP"
OUTPUT_DIR = BASE_DIR / "irregular_oculos_escuros"

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

def executar_deteccao_oculos_escuros(input_dir=None, output_dir=None, clip_thresh=CLIP_SUNGLASSES_THRESHOLD, max_fotos=None):
    print("=" * 70)
    print("DETECTOR DE ÓCULOS ESCUROS — YOLOS + CLIP (> 90% CERTEZA)")
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
    print(f"Limiar CLIP: {clip_thresh:.0%}")
    print(f"Tamanho do lote: {BATCH_SIZE} imagens | Workers I/O: {NUM_WORKERS}\n")

    print("Carregando modelos (YOLOS-Fashionpedia + CLIP)...")
    try:
        device = 0 if torch.cuda.is_available() else -1
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        detector = pipeline("object-detection", model="valentinafevu/yolos-fashionpedia", device=device)
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device_str)
        print("Modelos YOLOS + CLIP inicializados com sucesso!\n")
    except Exception as e:
        print(f"Erro ao carregar modelos: {e}")
        return

    relatorio_path = target_output / "relatorio.json"
    resultados_detalhados = []
    total_irregulares = 0
    processadas = set()

    if relatorio_path.exists():
        try:
            with open(relatorio_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                if saved_data.get("limiar_clip") == clip_thresh:
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
            results = detector(list(valid_imgs), threshold=0.20)

            for foto_path, img_rgb, res_list in zip(valid_files, valid_imgs, results):
                irregularidades = []
                glasses_items = [item for item in res_list if item["label"] == "glasses"]

                if glasses_items:
                    w, h = img_rgb.size
                    for item in glasses_items:
                        box = item["box"]
                        x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                        bw, bh = x2 - x1, y2 - y1
                        cx1 = max(0, int(x1 - bw * 0.1))
                        cy1 = max(0, int(y1 - bh * 0.1))
                        cx2 = min(w, int(x2 + bw * 0.1))
                        cy2 = min(h, int(y2 + bh * 0.1))

                        crop = img_rgb.crop((cx1, cy1, cx2, cy2))
                        inputs_clip = clip_processor(text=CLIP_PROMPTS, images=crop, return_tensors="pt", padding=True).to(device_str)
                        with torch.no_grad():
                            outputs_clip = clip_model(**inputs_clip)
                            probs = outputs_clip.logits_per_image.softmax(dim=-1)[0]

                        prob_dark = float(probs[0])
                        prob_clear = float(probs[1])

                        if prob_dark >= clip_thresh and prob_dark > prob_clear:
                            irregularidades.append({
                                "classe_en": "dark sunglasses",
                                "classe_pt": "óculos escuros",
                                "confianca_clip": round(prob_dark, 4),
                                "prob_oculos_grau": round(prob_clear, 4),
                                "bbox": [x1, y1, x2, y2]
                            })

                if irregularidades:
                    total_irregulares += 1
                    draw = ImageDraw.Draw(img_rgb)

                    for det in irregularidades:
                        box = det["bbox"]
                        label_pt = det["classe_pt"]
                        conf = det["confianca_clip"]

                        draw.rectangle(box, outline="red", width=4)
                        text = f"{label_pt} ({conf:.1%})"
                        try:
                            font = ImageFont.load_default()
                        except Exception:
                            font = None

                        draw.text((box[0] + 5, max(0, box[1] - 15)), text, fill="red", font=font)

                    img_rgb.save(target_output / foto_path.name)
                    print(f" -> DETECTADO: {foto_path.name} | Óculos Escuros ({irregularidades[0]['confianca_clip']:.1%})")

                resultados_detalhados.append({
                    "arquivo": foto_path.name,
                    "status": "irregular" if irregularidades else "regular",
                    "irregularidades": irregularidades
                })

            current_processed = len(processadas) + b_idx + len(batch_files)
            elapsed = time.time() - tempo_inicio
            fps = (b_idx + len(batch_files)) / max(0.1, elapsed)
            print(f"Progresso: [{current_processed}/{total_fotos_count}] fotos analisadas ({fps:.1f} fotos/s) | Irregulares: {total_irregulares}")

            relatorio_data = {
                "data_analise": datetime.now().isoformat(),
                "modelo": "YOLOS-Fashionpedia + CLIP",
                "limiar_clip": clip_thresh,
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
    print("RESUMO FINAL — ÓCULOS ESCUROS (> 90% CERTEZA)")
    print("=" * 70)
    print(f"Total de fotos analisadas: {total_fotos_count}")
    print(f"Fotos irregulares salvas em '{target_output.name}': {total_irregulares}")
    print(f"Tempo desta execução: {tempo_total:.2f} segundos")
    print(f"Relatório de análise salvo em: {relatorio_path}")

if __name__ == "__main__":
    executar_deteccao_oculos_escuros()
