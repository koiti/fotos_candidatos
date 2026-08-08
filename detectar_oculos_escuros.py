"""
Detector Híbrido de Óculos Escuros em Fotos de Candidatos (YOLO-World + CLIP)
=============================================================================

Combina localização de armações via YOLO-World com classificação semântica de lentes via CLIP.
Elimina falsos positivos ao diferenciar óculos de grau transparentes de óculos escuros de sol.
"""

import json
import time
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

BASE_DIR = Path(__file__).resolve().parent
AMOSTRAS_DIR = BASE_DIR / "amostras"
OUTPUT_DIR = BASE_DIR / "irregular_oculos_escuros"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_PROMPTS = [
    "dark black tinted sunglasses covering eyes",
    "clear transparent prescription reading eyeglasses with visible eyes"
]

def executar_deteccao_oculos_escuros(
    input_dir=None,
    output_dir=None,
    clip_thresh=0.90,
    batch_size=32
):
    target_input = Path(input_dir) if input_dir else AMOSTRAS_DIR
    if not target_input.exists():
        target_input = BASE_DIR / "amostras"
        if not target_input.exists():
            print(f"Erro: Pasta de entrada '{target_input}' não encontrada!")
            return None

    target_output = Path(output_dir) if output_dir else OUTPUT_DIR
    target_output.mkdir(exist_ok=True, parents=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    fotos = sorted([
        f for f in target_input.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ])

    if not fotos:
        print(f"Nenhuma foto encontrada na pasta '{target_input}'!")
        return None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print(f"DETECTOR DE ÓCULOS ESCUROS — YOLO-World + CLIP (> {clip_thresh:.0%} Confiança)")
    print("=" * 70)
    print(f"Diretório de entrada: '{target_input}' ({len(fotos)} fotos)")
    print(f"Diretório de saída: '{target_output}'")
    print(f"Dispositivo: {device.upper()}\n")

    print("1. Carregando modelo YOLO-World...")
    model_path = BASE_DIR / "yolov8s-worldv2.pt"
    model_name = str(model_path) if model_path.exists() else "yolov8s-worldv2.pt"
    model_yolo = YOLO(model_name)
    model_yolo.set_classes(["glasses", "eyeglasses", "sunglasses"])

    print("2. Carregando modelo CLIP...")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)

    total_irregulares = 0
    resultados_detalhados = []
    t_start = time.time()

    for b_idx in range(0, len(fotos), batch_size):
        bfiles = fotos[b_idx:b_idx + batch_size]
        bimgs = []
        vfiles = []
        for f in bfiles:
            try:
                img = Image.open(f)
                img = ImageOps.exif_transpose(img).convert('RGB')
                bimgs.append(img)
                vfiles.append(f)
            except Exception:
                pass

        if not bimgs:
            continue

        results = model_yolo.predict(source=bimgs, conf=0.20, batch=batch_size, device=device, verbose=False)

        for foto_path, img_rgb, res in zip(vfiles, bimgs, results):
            irregularidades = []
            if len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                w, h = img_rgb.size

                for box in boxes:
                    x1, y1, x2, y2 = [int(c) for c in box]
                    bw, bh = x2 - x1, y2 - y1
                    cx1 = max(0, int(x1 - bw * 0.1))
                    cy1 = max(0, int(y1 - bh * 0.1))
                    cx2 = min(w, int(x2 + bw * 0.1))
                    cy2 = min(h, int(y2 + bh * 0.1))

                    crop = img_rgb.crop((cx1, cy1, cx2, cy2))
                    inputs_clip = clip_processor(text=CLIP_PROMPTS, images=crop, return_tensors="pt", padding=True).to(device)
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
                    try:
                        font = ImageFont.load_default()
                    except Exception:
                        font = None
                    draw.text((box[0] + 5, max(0, box[1] - 15)), f"{label_pt} ({conf:.1%})", fill="red", font=font)
                img_rgb.save(target_output / foto_path.name)
                print(f" -> DETECTADO: {foto_path.name} | Óculos Escuros ({irregularidades[0]['confianca_clip']:.1%})")

            resultados_detalhados.append({
                "arquivo": foto_path.name,
                "status": "irregular" if irregularidades else "regular",
                "irregularidades": irregularidades
            })

    t_total = time.time() - t_start
    relatorio_data = {
        "data_analise": datetime.now().isoformat(),
        "modelo": "YOLO-World + CLIP",
        "limiar_clip": clip_thresh,
        "total_fotos": len(fotos),
        "total_irregulares": total_irregulares,
        "tempo_execucao_segundos": round(t_total, 2),
        "resultados": resultados_detalhados
    }

    relatorio_path = target_output / "relatorio.json"
    with open(relatorio_path, "w", encoding="utf-8") as f:
        json.dump(relatorio_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("VARREDURA DE ÓCULOS ESCUROS CONCLUÍDA!")
    print(f"Total de fotos analisadas: {len(fotos)}")
    print(f"Irregulares salvas em '{target_output}': {total_irregulares}")
    print(f"Relatório salvo em: {relatorio_path}")
    print("=" * 70)
    return relatorio_data

if __name__ == "__main__":
    executar_deteccao_oculos_escuros()
