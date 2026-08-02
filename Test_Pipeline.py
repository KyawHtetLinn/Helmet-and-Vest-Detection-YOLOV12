import cv2
import argparse
import os
import glob
from ultralytics import YOLO

def setup_output_dir(dir_name="inference_outputs"):
    """Creates an output directory if it doesn't exist."""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def draw_custom_annotations(result, model):
    """
    Takes a YOLO result object, extracts class predictions, maps them to Job Roles,
    and draws matching colored bounding boxes and text labels.
    """
    annotated_frame = result.orig_img.copy()
    
    # --- COLOR MAPPING ---
    COLOR_PALETTE = {
        "yellow helmet": (0, 255, 255),   # Yellow
        "blue helmet": (255, 100, 0),    # Blue
        "white helmet": (255, 255, 255), # White
        "other helmet": (200, 200, 200), # Light Grey
        "vest": (0, 165, 255),           # Orange
        "no vest": (0, 0, 255),          # Red
        "person": (0, 255, 0),           # Green
        "bare head": (50, 50, 255)       # Coral
    }
    
    # --- JOB ROLE MAPPING ---
    DISPLAY_NAMES = {
        "blue helmet": "Electrician",
        "white helmet": "Engineer",
        "yellow helmet": "General Worker"
    }
    
    DEFAULT_COLOR = (255, 0, 255) 

    boxes = result.boxes
    frame_counts = {}
    display_colors = {} 

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            raw_class_name = model.names[class_id]
            conf = float(box.conf[0])
            
            clean_name = raw_class_name.lower().replace("_", " ").replace("-", " ").strip()
            display_name = DISPLAY_NAMES.get(clean_name, clean_name.title())
            box_color = COLOR_PALETTE.get(clean_name, DEFAULT_COLOR)
            
            frame_counts[display_name] = frame_counts.get(display_name, 0) + 1
            display_colors[display_name] = box_color
            
            # 1. Draw Bounding Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # 2. Draw Matching Text Label
            label_text = f"{display_name} {conf:.2f}"
            cv2.putText(
                img=annotated_frame, 
                text=label_text, 
                org=(x1, max(y1 - 5, 15)), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5,              
                color=box_color, 
                thickness=2, 
                lineType=cv2.LINE_AA
            )

        # 3. Draw Top-Left Status Bar Summary
        y_offset = 25 
        for role_name, count in frame_counts.items():
            summary_text = f"{role_name} - {count}"
            summary_color = display_colors[role_name] 
            
            cv2.putText(
                img=annotated_frame, 
                text=summary_text, 
                org=(15, y_offset), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.6,              
                color=summary_color, 
                thickness=2, 
                lineType=cv2.LINE_AA
            )
            y_offset += 25

    return annotated_frame, frame_counts

def run_image_inference(model, image_path, conf_threshold):
    """Runs inference on a single static image."""
    print(f"\n[INFO] Running inference on image: {image_path}")
    results = model.predict(source=image_path, conf=conf_threshold, save=False)
    
    for result in results:
        annotated_frame, _ = draw_custom_annotations(result, model)
        
        out_dir = setup_output_dir()
        out_path = os.path.join(out_dir, "result_" + os.path.basename(image_path))
        cv2.imwrite(out_path, annotated_frame)
        print(f"[SUCCESS] Annotated image saved to: {out_path}")
        
        cv2.imshow("YOLOv12 Deployment View - Press any key to close", annotated_frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_folder_inference(model, folder_path, conf_threshold):
    """Runs batch inference on every image found inside a specified folder directory."""
    print(f"\n[INFO] Initializing batch processing for folder: {folder_path}")
    
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
    total_images = len(image_files)
    if total_images == 0:
        print(f"[ERROR] No valid images found in folder '{folder_path}'. Check your path or file extensions.")
        return
        
    print(f"[SUCCESS] Found {total_images} images to process. Starting batch run...")
    
    out_base = setup_output_dir()
    folder_name = os.path.basename(os.path.normpath(folder_path))
    batch_out_dir = os.path.join(out_base, f"batch_{folder_name}")
    if not os.path.exists(batch_out_dir):
        os.makedirs(batch_out_dir)

    global_counts = {}

    results = model.predict(source=folder_path, conf=conf_threshold, save=False, stream=True)
    
    for idx, result in enumerate(results):
        current_file = image_files[idx] if idx < len(image_files) else f"image_{idx}.jpg"
        filename = os.path.basename(current_file)
        
        annotated_frame, frame_counts = draw_custom_annotations(result, model)
        
        for role_name, count in frame_counts.items():
            global_counts[role_name] = global_counts.get(role_name, 0) + count

        out_path = os.path.join(batch_out_dir, f"det_{filename}")
        cv2.imwrite(out_path, annotated_frame)

        print(f"[{idx + 1}/{total_images}] Processed: {filename}")
        
        cv2.imshow("Batch Processing Loop - Press 'q' to fast-forward", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            print("[INFO] Display closed. Continuing batch processing silently in background...")

    cv2.destroyAllWindows()
    print(f"\n[FINISHED] Batch execution completed successfully!")
    print(f"[SAVED] All annotated outputs saved to: {batch_out_dir}")
    print("\n--- GLOBAL DEPLOYMENT SUMMARY ---")
    if not global_counts:
        print("No mapped objects were detected above the confidence threshold.")
    for role_name, count in global_counts.items():
        print(f"  * {role_name}: {count} total detections")

def run_video_inference(model, video_path, conf_threshold):
    """Processes a video file frame-by-frame."""
    print(f"\n[INFO] Processing video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    out_dir = setup_output_dir()
    out_path = os.path.join(out_dir, "processed_" + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    results = model.predict(source=video_path, conf=conf_threshold, stream=True)
    for r in results:
        annotated_frame, _ = draw_custom_annotations(r, model)
        out_video.write(annotated_frame)
        cv2.imshow("YOLOv12 Video Processing", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"[SUCCESS] Video saved to: {out_path}")

def run_webcam_inference(model, camera_index, conf_threshold):
    """Launches a live webcam feed."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Camera index {camera_index} unavailable.")
        return
        
    # Attempt to lower the camera hardware resolution to speed up processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: continue
        
        # --- THE FIX: RESIZE THE RAW FRAME BEFORE DRAWING ON IT ---
        # Set a comfortable maximum width for a laptop screen (e.g., 1000 pixels)
        max_display_width = 1000
        h, w = frame.shape[:2]
        
        # Resize the raw camera image first
        if w > max_display_width:
            scale_ratio = max_display_width / w
            new_width = int(w * scale_ratio)
            new_height = int(h * scale_ratio)
            frame = cv2.resize(frame, (new_width, new_height))

        # NOW run inference and draw annotations on the already-resized frame
        results = model(frame, stream=True, conf=conf_threshold)
        
        for r in results:
            annotated_frame, _ = draw_custom_annotations(r, model)

        cv2.imshow("YOLOv12 Live PPE Monitor", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv12 Job Role Tracking Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["image", "folder", "video", "webcam"],
                        help="The input media format configuration.")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to target file/folder or integer camera index.")
    
    # Points to your ONNX file by default
    parser.add_argument("--weights", type=str, default="weights/best_50.onnx",
                        help="Path to your exported .onnx model file.")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence scoring threshold filter.")
    
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"[CRITICAL] Weights file '{args.weights}' not found.")
        exit()
        
    print(f"[INFO] Initializing ONNX Inference Session using {args.weights}...")
    
    # task='detect' is explicitly required for ONNX format
    model = YOLO(args.weights, task='detect')

    if args.mode == "image":
        run_image_inference(model, args.source, args.conf)
    elif args.mode == "folder":
        run_folder_inference(model, args.source, args.conf)
    elif args.mode == "video":
        run_video_inference(model, args.source, args.conf)
    elif args.mode == "webcam":
        cam_source = int(args.source) if args.source.isdigit() else args.source
        run_webcam_inference(model, cam_source, args.conf)