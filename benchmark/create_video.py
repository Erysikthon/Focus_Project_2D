import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from utilities import terminal_colors as colors


def _load_font(size=8):
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _draw_label(draw, text, x, y, color, font):
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([(bbox[0] - 1, bbox[1] - 1), (bbox[2] + 1, bbox[3] + 1)], fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=color)
    return bbox[3] - bbox[1]  # return text height


def annotate_video_with_predictions(video_path, predictions, output_path, frame_offset=0, true_labels=None, column_names=None):
    """
    frame_offset
        Starting frame number --> predictions start from frame 15
    """

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    font = _load_font(size=8)

    frame_idx = 0
    pred_idx = 0
    with tqdm(desc=colors.CYAN + "    annotating" + colors.ENDC, total=180, ascii=True) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_frame)

            if frame_idx >= frame_offset and pred_idx < len(predictions):
                prediction = predictions.iloc[pred_idx] if predictions.ndim == 1 else predictions.iloc[pred_idx, 0]
                pred_display = column_names[prediction] if column_names else prediction

                if true_labels is not None and pred_idx < len(true_labels):
                    true_label = true_labels.iloc[pred_idx] if true_labels.ndim == 1 else true_labels.iloc[pred_idx, 0]
                    true_display = column_names[true_label] if column_names else true_label
                    pred_color = (0, 255, 0) if prediction == true_label else (255, 0, 0)
                else:
                    true_display = None
                    pred_color = (0, 255, 0)

                y = 2
                pred_h = _draw_label(draw, f"Pred: {pred_display}", 2, y, pred_color, font)
                if true_display is not None:
                    _draw_label(draw, f"True: {true_display}", 2, y + pred_h + 1, (0, 255, 0), font)

                # Frame number bottom-left
                frame_text = f"F: {frame_idx}"
                fbbox = draw.textbbox((2, 0), frame_text, font=font)
                fh = fbbox[3] - fbbox[1]
                _draw_label(draw, frame_text, 2, height - fh - 3, (255, 255, 255), font)

                pred_idx += 1
            else:
                _draw_label(draw, "No pred", 2, 2, (255, 0, 0), font)

            frame = cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)
            out.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                pbar.update()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
