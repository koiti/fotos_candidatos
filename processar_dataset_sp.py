"""
Processador Paralelo do Dataset SP 2024 (TSE / Urna Eleitoral)
================================================================================

Executa a varredura das fotos de candidatos do Estado de SP com:
1. Limiar de certeza estrito > 90% (0.90) para Acessórios de Cabeça via YOLOS-Fashionpedia.
2. Limiar de certeza estrito > 90% (0.90) para Óculos Escuros (confirmados via CLIP).
3. Atualização simultânea e incremental dos relatórios de ambos os detectores a cada lote.
"""

import os
import json
import time
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps, ImageDraw, ImageFont
from transformers import pipeline, CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor

# Configurações de otimização de CPU
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)

# Caminhos
BASE_DIR = Path(r"c:\Users\steve\OneDrive\Documentos\UFABC\2026-TOPICOS_DE_IA")
DATASET_DIR = BASE_DIR / "foto_cand2024_SP"
OUT_HEAD = BASE_DIR / "irregular_acessorios_cabeca"
OUT_OCULOS = BASE_DIR / "irregular_oculos_escuros"

CONF_THRESHOLD_HEAD = 0.90
CONF_THRESHOLD_SUNGLASSES = 0.90
BATCH_SIZE = 32
NUM_WORKERS = 16

CLASSES_CABECA_MAP = {
    "hat": "chapéu/boné",
    "headband, head covering, hair accessory": "acessório de cabeça",
    "hood": "capuz"
}

CLIP_PROMPTS = [
    "dark black tinted sunglasses covering eyes",
    "clear transparent prescription reading eyeglasses with visible eyes"
]

def load_img(p):
    try:
        img = Image.open(p)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception:
        return None

def main():
    print("=" * 80)
    print("VARREDURA GLOBAL DATASET SP 2024 — YOLOS-Fashionpedia + CLIP (Hugging Face)")
    print("=" * 80)

    input_dir = DATASET_DIR if DATASET_DIR.exists() else BASE_DIR / "amostras"
    print(f"Diretório de Entrada: '{input_dir.name}'")

    OUT_HEAD.mkdir(exist_ok=True, parents=True)
    OUT_OCULOS.mkdir(exist_ok=True, parents=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    fotos = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts])
    total_fotos = len(fotos)
    print(f"Total de fotos encontradas: {total_fotos}\n")

    print("Carregando YOLOS-Fashionpedia e CLIP...")
    device = 0 if torch.cuda.is_available() else -1
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    detector = pipeline("object-detection", model="valentinafevu/yolos-fashionpedia", device=device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device_str)
    print("Modelos inicializados com sucesso!\n")

    rel_head_path = OUT_HEAD / "relatorio.json"
    rel_oculos_path = OUT_OCULOS / "relatorio.json"

    res_head = []
    res_oculos = []
    tot_head = 0
    tot_oculos = 0

    executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    t_start = time.time()

    for b_idx in range(0, total_fotos, BATCH_SIZE):
        bfiles = fotos[b_idx:b_idx + BATCH_SIZE]
        bimgs = list(executor.map(load_img, bfiles))

        valid_pairs = [(f, img) for f, img in zip(bfiles, bimgs) if img is not None]
        if not valid_pairs:
            continue

        vfiles, vimgs = zip(*valid_pairs)
        results = detector(list(vimgs), threshold=0.20)

        for foto_path, img_orig, res_list in zip(vfiles, vimgs, results):
            # 1. Acessórios de cabeça
            irreg_head = []
            for item in res_list:
                lbl_en = item["label"]
                score = float(item["score"])
                box = item["box"]

                if lbl_en in CLASSES_CABECA_MAP and score >= CONF_THRESHOLD_HEAD:
                    lbl_pt = CLASSES_CABECA_MAP[lbl_en]
                    irreg_head.append({
                        "classe_en": lbl_en,
                        "classe_pt": lbl_pt,
                        "confianca": round(score, 4),
                        "bbox": [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]
                    })

            if irreg_head:
                tot_head += 1
                img_copy = img_orig.copy()
                draw = ImageDraw.Draw(img_copy)
                for det in irreg_head:
                    b = det["bbox"]
                    draw.rectangle(b, outline="red", width=4)
                    draw.text((b[0] + 5, max(0, b[1] - 15)), f"{det['classe_pt']} ({det['confianca']:.1%})", fill="red")
                img_copy.save(OUT_HEAD / foto_path.name)

            res_head.append({
                "arquivo": foto_path.name,
                "status": "irregular" if irreg_head else "regular",
                "irregularidades": irreg_head
            })

            # 2. Óculos Escuros
            irreg_oculos = []
            glasses_items = [item for item in res_list if item["label"] == "glasses"]

            if glasses_items:
                w, h = img_orig.size
                for item in glasses_items:
                    box = item["box"]
                    x1, y1, x2, y2 = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
                    bw, bh = x2 - x1, y2 - y1
                    cx1 = max(0, int(x1 - bw * 0.1))
                    cy1 = max(0, int(y1 - bh * 0.1))
                    cx2 = min(w, int(x2 + bw * 0.1))
                    cy2 = min(h, int(y2 + bh * 0.1))

                    crop = img_orig.crop((cx1, cy1, cx2, cy2))
                    inputs_clip = clip_processor(text=CLIP_PROMPTS, images=crop, return_tensors="pt", padding=True).to(device_str)
                    with torch.no_grad():
                        outputs_clip = clip_model(**inputs_clip)
                        probs = outputs_clip.logits_per_image.softmax(dim=-1)[0]

                    prob_dark = float(probs[0])
                    prob_clear = float(probs[1])

                    if prob_dark >= CONF_THRESHOLD_SUNGLASSES and prob_dark > prob_clear:
                        irreg_oculos.append({
                            "classe_en": "dark sunglasses",
                            "classe_pt": "óculos escuros",
                            "confianca_clip": round(prob_dark, 4),
                            "prob_oculos_grau": round(prob_clear, 4),
                            "bbox": [x1, y1, x2, y2]
                        })

            if irreg_oculos:
                tot_oculos += 1
                img_copy = img_orig.copy()
                draw = ImageDraw.Draw(img_copy)
                for det in irreg_oculos:
                    b = det["bbox"]
                    draw.rectangle(b, outline="red", width=4)
                    draw.text((b[0] + 5, max(0, b[1] - 15)), f"óculos escuros ({det['confianca_clip']:.1%})", fill="red")
                img_copy.save(OUT_OCULOS / foto_path.name)

            res_oculos.append({
                "arquivo": foto_path.name,
                "status": "irregular" if irreg_oculos else "regular",
                "irregularidades": irreg_oculos
            })

        processed = b_idx + len(bfiles)
        elapsed = time.time() - t_start
        fps = processed / max(0.1, elapsed)
        print(f"Progresso: [{processed}/{total_fotos}] fotos ({fps:.1f} f/s) | Acessórios Cabeça: {tot_head} | Óculos Escuros: {tot_oculos}")

        # Checkpoints
        with open(rel_head_path, "w", encoding="utf-8") as f:
            json.dump({"total_fotos": total_fotos, "total_irregulares": tot_head, "resultados": res_head}, f, ensure_ascii=False, indent=2)

        with open(rel_oculos_path, "w", encoding="utf-8") as f:
            json.dump({"total_fotos": total_fotos, "total_irregulares": tot_oculos, "resultados": res_oculos}, f, ensure_ascii=False, indent=2)

    executor.shutdown()
    print("\n" + "=" * 80)
    print("VARREDURA CONCLUÍDA COM SUCESSO!")
    print(f"Total fotos: {total_fotos} | Irregulares Cabeça: {tot_head} | Irregulares Óculos Escuros: {tot_oculos}")

if __name__ == "__main__":
    main()
