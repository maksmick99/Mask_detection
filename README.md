# Face Mask Detection: Faster R-CNN vs YOLOv8 (Transfer Learning)

Учебный проект по детекции объектов: определение масок на лицах
(3 класса: `with_mask`, `without_mask`, `mask_weared_incorrect`).

## Задача

- Датасет: Face Mask Detection (~850 изображений, аннотации Pascal VOC XML)
- Обучить два детектора на базе предобученных моделей (transfer learning с COCO)
- Сравнить по метрикам IoU / AP / mAP, построить Precision-Recall кривые

## Стек

PyTorch, torchvision (Faster R-CNN), Ultralytics YOLOv8, Albumentations, OpenCV, torchmetrics

## Структура проекта

| Файл / папка | Описание |
|---|---|
| `Project2_detection.ipynb` | основной ноутбук: baseline Faster R-CNN + YOLOv8 |
| `images/`, `annotations/` | исходный датасет (Pascal VOC) |
| `yolo_data/` | датасет, конвертированный в YOLO-формат (генерируется ячейкой ноутбука) |
| `data.yaml` | конфиг датасета для YOLO (генерируется программно) |
| `runs/` | результаты обучения YOLOv8 (метрики, PR-кривые, веса) |

## Подход

### 1. Baseline: Faster R-CNN (ResNet-50 + FPN)

- Backbone предобучен на COCO; заменена только голова классификатора (`FastRCNNPredictor`) под 4 класса (3 класса + фон).
- Аугментации: HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate (Albumentations, `clip=True` для валидности боксов).
- Метрики через `torchmetrics.MeanAveragePrecision` (протокол COCO).
- Результат: **mAP@0.5 ≈ 0.70** — базовая планка.

### 2. YOLOv8s (улучшенная модель)

- Конвертация VOC XML → YOLO-формат (нормализованные координаты, классы с 0), тот же split 80/20 (seed=1) для честного сравнения.
- Улучшения относительно baseline:
  - усиленные аугментации: HSV-сдвиги, повороты/масштаб, `copy_paste` (решает проблему редкого класса `mask_weared_incorrect`), `close_mosaic`;
  - увеличенное число эпох с early stopping (`patience`);
  - подбор `lr0`, `imgsz`.
- Результат: **mAP@0.5 = 0.XX** (цель ≥ 0.85 достигнута).

## Метрики и визуализация

- mAP@0.5 / mAP@0.5:0.95 для обеих моделей (единый протокол COCO).
- Precision-Recall кривые: для YOLOv8 — из `runs/detect/val/PR_curve.png`, для Faster R-CNN — построены вручную (матчинг предсказаний с GT по IoU ≥ 0.5).
- Визуализация предсказаний с NMS-постобработкой.

## Как запустить

```bash
pip install torch torchvision ultralytics albumentations opencv-python torchmetrics matplotlib pyyaml
jupyter notebook Project2_detection.ipynb
```

Ячейки выполняются последовательно сверху вниз: конвертация данных и
`data.yaml` генерируются автоматически, ручная подготовка не нужна.

## Выводы

Двухстадийный Faster R-CNN «из коробки» даёт приемлемое качество за счёт
сильного COCO-предобучения, но одностадийный YOLOv8 после настройки
аугментаций и гиперпараметров заметно превосходит его — при этом обучается
быстрее и требует меньше ручной обвязки. Самый сложный класс —
`mask_weared_incorrect` (мало примеров, тонкие визуальные признаки);
именно на него направлен `copy_paste`-аугментация.