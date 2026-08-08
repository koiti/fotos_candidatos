"""
Detector de Acessórios de Cabeça em Fotos de Candidatos (YOLO-World Open-Vocabulary)
====================================================================================

Detecta acessórios de cabeça (chapéus, bonés, toucas, tiaras, capacetes, turbantes, etc.)
em fotos de candidatos com limiar de confiança superior a 90% (>0.90).
"""

import json
import time
import torch
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont, ImageOps
from ultralytics import YOLO

# Caminhos padrão
BASE_DIR = Path(__file__).resolve().parent
AMOSTRAS_DIR = BASE_DIR / "amostras"
OUTPUT_DIR = BASE_DIR / "irregular_acessorios_cabeca"

CLASSES_CABETA_MAP = {
    'hat': 'chapéu',
    'cap': 'boné',
    'baseball cap': 'boné',
    'headband': 'tiara/faixa',
    'head covering': 'cobertura de cabeça',
    'hood': 'capuz',
    'bonnet': 'touca',
    'turban': 'turbante',
    'helmet': 'capacete',
    'beret': 'boina',
    'beanie': 'gorro'
}
PROMPTS = list(CLASSES_CABETA_MAP.keys())

def executar_deteccao_acessorios_cabeca(
    input_dir=None,
    output_dir=None,
    conf_thresh=0.90,
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
    print(f"DETECTOR DE ACESSÓRIOS DE CABEÇA — YOLO-World (> {conf_thresh:.0%} Confiança)")
    print("=" * 70)
    print(f"Diretório de entrada: '{target_input}' ({len(fotos)} fotos)")
    print(f"Diretório de saída: '{target_output}'")
    print(f"Dispositivo: {device.upper()}\n")

    model_path = BASE_DIR / "yolov8s-worldv2.pt"
    model_name = str(model_path) if model_path.exists() else "yolov8s-worldv2.pt"
    model = YOLO(model_name)
    model.set_classes(PROMPTS)

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

        results = model.predict(source=bimgs, conf=conf_thresh, batch=batch_size, device=device, verbose=False)

        for foto_path, img_rgb, res in zip(vfiles, bimgs, results):
            irregularidades = []
            if len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_idx in zip(boxes, confs, clss):
                    c_val = float(conf)
                    if c_val >= conf_thresh:
                        lbl_en = PROMPTS[cls_idx] if cls_idx < len(PROMPTS) else 'head accessory'
                        lbl_pt = CLASSES_CABETA_MAP.get(lbl_en, lbl_en)
                        irregularidades.append({
                            'classe_en': lbl_en,
                            'classe_pt': lbl_pt,
                            'confianca': round(c_val, 4),
                            'bbox': [round(float(c), 2) for c in box]
                        })

            if irregularidades:
                total_irregulares += 1
                draw = ImageDraw.Draw(img_rgb)
                for det in irregularidades:
                    box = det['bbox']
                    label_pt = det['classe_pt']
                    conf = det['confianca']
                    draw.rectangle(box, outline='red', width=4)
                    try:
                        font = ImageFont.load_default()
                    except Exception:
                        font = None
                    draw.text((box[0] + 5, max(0, box[1] - 15)), f"{label_pt} ({conf:.1%})", fill='red', font=font)
                img_rgb.save(target_output / foto_path.name)
                print(f" -> DETECTADO: {foto_path.name} | {irregularidades[0]['classe_pt']} ({irregularidades[0]['confianca']:.1%})")

            resultados_detalhados.append({
                'arquivo': foto_path.name,
                'status': 'irregular' if irregularidades else 'regular',
                'irregularidades': irregularidades
            })

    t_total = time.time() - t_start
    relatorio_data = {
        'data_analise': datetime.now().isoformat(),
        'modelo': 'YOLO-World (v2)',
        'limiar_confianca': conf_thresh,
        'total_fotos': len(fotos),
        'total_irregulares': total_irregulares,
        'tempo_execucao_segundos': round(t_total, 2),
        'resultados': resultados_detalhados
    }

    relatorio_path = target_output / 'relatorio.json'
    with open(relatorio_path, 'w', encoding='utf-8') as f:
        json.dump(relatorio_data, f, ensure_ascii=False, indent=2)

    print('\n' + '=' * 70)
    print('VARREDURA DE ACESSÓRIOS DE CABEÇA CONCLUÍDA!')
    print(f'Total de fotos analisadas: {len(fotos)}')
    print(f'Irregulares salvas em "{target_output}": {total_irregulares}')
    print(f'Relatório salvo em: {relatorio_path}')
    print('=' * 70)
    return relatorio_data

if __name__ == '__main__':
    executar_deteccao_acessorios_cabeca()
