import cv2
import pandas as pd
from tqdm import tqdm
from utilities import terminal_colors as colors


def annotate_video_with_predictions(video_path, predictions, output_path, frame_offset=0, true_labels=None):
    """
    frame_offset
        Starting frame number --> predictions start from frame 15
    """

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    pred_idx = 0
    with tqdm(desc = colors.CYAN +"    annotating" + colors.ENDC, total = 180, ascii = True) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # Get prediction for current frame
            if frame_idx >= frame_offset and pred_idx < len(predictions):
                prediction = predictions.iloc[pred_idx, 0]

                # Configure text appearance (adapted for 70x150)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3
                thickness = 1

                # Determine color based on match with true label
                if true_labels is not None and pred_idx < len(true_labels):
                    true_label = true_labels.iloc[pred_idx, 0]
                    color = (0, 255, 0) if prediction == true_label else (0, 0, 255)
                else:
                    color = (0, 255, 0)

                # Add prediction text
                pred_text = f"Pred: {prediction}"
                (text_width, text_height), _ = cv2.getTextSize(
                    pred_text, font, font_scale, thickness
                )
                cv2.rectangle(
                    frame, (2, 2), (2 + text_width, 2 + text_height), (0, 0, 0), -1
                )
                cv2.putText(
                    frame, pred_text, (2, 2 + text_height),
                    font, font_scale, color, thickness
                )

                # Add true label if available
                if true_labels is not None and pred_idx < len(true_labels):
                    true_text = f"True: {true_label}"
                    (text_width3, text_height3), _ = cv2.getTextSize(
                        true_text, font, font_scale, thickness
                    )
                    cv2.rectangle(
                        frame,
                        (2, 3 + text_height),
                        (2 + text_width3, 2 + text_height + text_height3),
                        (0, 0, 0),
                        -1,
                    )
                    cv2.putText(
                        frame, true_text,
                        (2, 2 + text_height + text_height3),
                        font, font_scale, (0, 255, 0), thickness
                    )

                # Frame number (bottom-left, adapted for 70x150)
                frame_text = f"F: {frame_idx}"
                (text_width2, text_height2), _ = cv2.getTextSize(
                    frame_text, font, 0.3, 1
                )
                cv2.rectangle(
                    frame,
                    (2, 140 - text_height2 - 2),
                    (2 + text_width2, 140 - 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame, frame_text,
                    (2, 140 - 2),
                    font, 0.3, (255, 255, 255), 1
                )

                pred_idx += 1
            else:
                # No prediction available for this frame (adapted for 70x150)
                label_text = "No pred"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    frame, label_text,
                    (5, 15),
                    font, 0.35, (0, 0, 255), 1
                )

            # Write frame to output
            out.write(frame)
            frame_idx += 1

            if frame_idx % 100 == 0:
                pbar.update()
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()



