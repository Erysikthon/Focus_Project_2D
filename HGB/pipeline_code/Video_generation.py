import cv2
import pandas as pd


def annotate_video_with_predictions(video_path, predictions, output_path, fps=30, frame_offset=15, true_labels=None):
    """
    frame_offset
        Starting frame number --> predictions start from frame 15
    """

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None:
        fps = input_fps

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    pred_idx = 0

    print(f"Processing video: {video_path}")
    print(f"Video dimensions: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Frame offset: {frame_offset}")
    print(f"Total predictions: {len(predictions)}")
    if true_labels is not None:
        print(f"Total true labels: {len(true_labels)}")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Get prediction for current frame
        if frame_idx >= frame_offset and pred_idx < len(predictions):
            prediction = predictions[pred_idx]

            # Configure text appearance
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2

            # Determine color based on match with true label
            if true_labels is not None and pred_idx < len(true_labels):
                true_label = true_labels[pred_idx]
                color = (0, 255, 0) if prediction == true_label else (0, 0, 255)
            else:
                color = (0, 255, 0)

            # Add prediction text
            pred_text = f"Prediction: {prediction}"
            (text_width, text_height), _ = cv2.getTextSize(pred_text, font, font_scale, thickness)
            cv2.rectangle(frame, (10, 10), (20 + text_width, 30 + text_height), (0, 0, 0), -1)
            cv2.putText(frame, pred_text, (15, 35), font, font_scale, color, thickness)

            # Add true label if available
            if true_labels is not None and pred_idx < len(true_labels):
                true_text = f"True: {true_label}"

                (text_width3, text_height3), _ = cv2.getTextSize(true_text, font, font_scale, thickness)
                cv2.rectangle(frame, (10, 50 + text_height), (20 + text_width3, 70 + text_height + text_height3), (0, 0, 0), -1)
                cv2.putText(frame, true_text, (15, 75 + text_height), font, font_scale, (0, 255, 0), thickness)

            # Frame number
            frame_text = f"Frame: {frame_idx}"
            (text_width2, text_height2), _ = cv2.getTextSize(frame_text, font, 0.6, 2)
            cv2.rectangle(frame, (10, height - 40), (20 + text_width2, height - 10), (0, 0, 0), -1)
            cv2.putText(frame, frame_text, (15, height - 20), font, 0.6, (255, 255, 255), 2)

            pred_idx += 1
        else:
            # No prediction available for this frame
            label_text = "No prediction"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, label_text, (15, 35), font, 1.0, (0, 0, 255), 2)

        # Write frame to output
        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Done! Annotated video saved to: {output_path}")
    print(f"Total frames processed: {frame_idx}")


