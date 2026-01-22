import argparse
from collections import namedtuple
import copy
import jiwer
import json
import os
import re
import cv2
import subprocess
import csv
from moviepy.config import FFMPEG_BINARY

Segment = namedtuple('Segment', ['start', 'end', 'text', 'boxes'])
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    # Define the command-line arguments
    parser.add_argument('--input_file', help='Input result jsonl file.', default="")
    # args.visualize
    parser.add_argument('--visualize', help='Visualize the bounding boxes and subtitles.', action='store_true')
    # args.video_dir
    parser.add_argument('--video_dir', help='Directory containing video files.', default="")
    return parser.parse_args()

def calculate_iou(segment1: Segment, segment2: Segment) -> float:
    """
    Calculates the Intersection over Union (IoU) of two time segments.
    
    Args:
        segment1: The first segment.
        segment2: The second segment.

    Returns:
        The IoU score, a float between 0.0 and 1.0.
    """
    
    # Determine the intersection coordinates
    inter_start = max(segment1.start, segment2.start)
    inter_end = min(segment1.end, segment2.end)

    # Calculate intersection and union areas (lengths in 1D)
    inter_length = max(0, inter_end - inter_start)
    
    len1 = segment1.end - segment1.start
    len2 = segment2.end - segment2.start
    
    union_length = len1 + len2 - inter_length
    
    if union_length == 0:
        return 0.0

    return inter_length / union_length

def calculate_box_iou(box1, box2):
    # box is a dict with x0, y0, x1, y1
    # Determine the intersection coordinates
    
    inter_x0 = max(box1[0], box2[0])
    inter_y0 = max(box1[1], box2[1])
    inter_x1 = min(box1[2], box2[2])
    inter_y1 = min(box1[3], box2[3])

    # Calculate intersection area
    inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)

    # Calculate areas of individual boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0

    return inter_area / union_area

def compare_transcripts(pred_segments, gt_segments, iou_threshold: float = 0.5, bbox_time_tolerance: float = 0.02):
    """
    Compares a predicted transcript with a ground-truth transcript and calculates evaluation metrics.

    Metrics calculated:
    1.  Temporal Grounding (IoU): Average IoU of matched segments.
    2.  Word Distance (WER): Word Error Rate for matched speakers.
    3.  Bounding Box IoU: Average IoU of matched bounding boxes within matched segments.
    """

    matches = []
    matched_pred_indices = set()

    # --- Step 1: Align Ground Truth to Predictions based on IoU ---
    for gt_idx, gt_seg in enumerate(gt_segments):
        best_iou = -1
        best_pred_idx = -1
        # Find the prediction segment with the highest IoU
        for pred_idx, pred_seg in enumerate(pred_segments):
            if pred_idx in matched_pred_indices:
                continue # This prediction segment is already matched
            
            iou = calculate_iou(gt_seg, pred_seg)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = pred_idx

        # If the best found IoU is above our threshold, we have a match
        if best_iou >= iou_threshold:
            matches.append({
                'gt': gt_seg,
                'pred': pred_segments[best_pred_idx],
                'iou': best_iou
            })
            matched_pred_indices.add(best_pred_idx)
            
    # --- Step 2: Calculate Metrics from the Matched Segments ---
    if not matches:
        # print("Warning: No segments could be matched based on the IoU threshold.")
        overall_gt_text = " ".join([seg.text for seg in gt_segments]).lower()
        overall_pred_text = " ".join([seg.text for seg in pred_segments]).lower()
        if overall_gt_text:
            #pred_len = len(overall_gt_text)
            overall_wer = jiwer.wer(overall_gt_text, overall_pred_text) 
        else:
            overall_wer = 1.0
        if overall_wer>=1:
            overall_wer = 1.0
        if overall_wer<=0:
            overall_wer = 0.0
        overall_word_accuracy = 1.0 - overall_wer
        return {
            "metrics": {
                "temporal_iou_avg": 0,
                "word_error_rate": 1.0,
                "overall_word_accuracy": overall_word_accuracy,
                "overall_word_error": overall_wer,
                "average_box_iou": 0,
                "total_gt_segments": len(gt_segments),
                "total_pred_segments": len(pred_segments),
                "matched_segments": 0
            },
            "matches": []
        }
        
    total_iou = 0
    total_char_distance = 0
    total_chars = 0

    total_box_iou = 0
    box_matches = 0

    # For overall WER, concatenate texts from matched segments where speakers also match
    gt_full_text = []
    pred_full_text = []

    for match in matches:
        gt_seg = match['gt']
        pred_seg = match['pred']
        
        total_iou += match['iou']

        gt_full_text.append(gt_seg.text)
        pred_full_text.append(pred_seg.text)
   
        # Bounding Box IoU calculation
        if gt_seg.boxes and pred_seg.boxes:
            # Match boxes based on timestamp
            for gt_box in gt_seg.boxes:
                
                # Find best matching pred_box by timestamp
                min_ts_diff = float('inf')
                best_pred_box_candidate = None
                
                for pred_box in pred_seg.boxes:
                    ts_diff = abs(gt_box['timestamp'] - pred_box['timestamp'])
                    if ts_diff < min_ts_diff:
                        min_ts_diff = ts_diff
                        best_pred_box_candidate = pred_box
                
                # Assuming a small time tolerance for matching boxes
                if float(min_ts_diff) < bbox_time_tolerance: # 20ms tolerance
                    # print(min_ts_diff, best_pred_box_candidate, gt_box)
                    try:
                        box_iou = calculate_box_iou(gt_box['box_2d'], best_pred_box_candidate['box_2d'])
                    except:
                        box_iou = 0.0
                    total_box_iou += box_iou
                    box_matches += 1
    # --- Step 3: Finalize and Aggregate Metrics ---
    
    # 1. Temporal Grounding Accuracy
    temporal_iou_avg = total_iou / len(matches) if matches else 0

    # 2. Word Distance (Word Error Rate)
    # Concatenate all matched texts and calculate a single WER score.
    # This is more stable than averaging per-segment WER.
    gt_corpus = " ".join(gt_full_text)
    pred_corpus = " ".join(pred_full_text)
    gt_corpus = gt_corpus.lower()
    pred_corpus = pred_corpus.lower()
    wer = jiwer.wer(gt_corpus, pred_corpus) if gt_corpus else 1.0
    
    wer = 1.0 if wer>=1 else wer
    wer = 0.0 if wer<=0 else wer

    # 3 Overall Word Accuracy (regardless of timestamp)
    overall_gt_text = " ".join([seg.text for seg in gt_segments]).lower()
    overall_pred_text = " ".join([seg.text for seg in pred_segments]).lower()
    if overall_gt_text:
        #pred_len = len(overall_gt_text)
        overall_wer = jiwer.wer(overall_gt_text, overall_pred_text) 
    else:
        overall_wer = 1.0
    if overall_wer>=1:
        overall_wer = 1.0
    if overall_wer<=0:
        overall_wer = 0.0
    overall_word_accuracy = 1.0 - overall_wer
 
    # 4. Bounding Box IoU
    average_box_iou = total_box_iou / box_matches if box_matches > 0 else 0

    serializable_matches = [
        {
            'gt': match['gt']._asdict(),
            'pred': match['pred']._asdict(),
            'iou': match['iou']
        } for match in matches
    ]
    
    metrics = {
        "temporal_iou_avg": temporal_iou_avg,
        "average_box_iou": average_box_iou,
        "word_error_rate": wer,
        "overall_word_accuracy": overall_word_accuracy,
        "overall_word_error": overall_wer,
        "total_gt_segments": len(gt_segments),
        "total_pred_segments": len(pred_segments),
        "matched_segments": len(matches)
    }

    return {
        "metrics": metrics,
        "matches": serializable_matches
    }


def extract_answer(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text[0]

def parse_result(args):
   
    with open(args.input_file) as file:
        results = json.loads(file.read())
    gt_questions = copy.deepcopy(results)
    ques2pred = {ele['query_id'] : ele for ele in results}
    all_results = []

    total_metrics = {
        "temporal_iou_avg": 0,
        "average_box_iou": 0,
        "word_error_rate": 0,
        "overall_word_accuracy": 0,
        "overall_word_error": 0,
        "total_gt_segments": 0,
        "total_pred_segments": 0,
        "matched_segments": 0
    }
    
    
    num_pred = 0.0
    for ques in gt_questions:
        
        if ques['query_id'] not in ques2pred:
            continue
        pred = ques2pred[ques['query_id']]

        num_pred += 1
        # Define a structured format for each transcript segment for easier access.
        
        gt_json = pred['gt']
        pred_json = pred['pred']
        duration = pred['duration']
        
        for item in gt_json:
            item['start'] = float(item['start'])
            item['end'] = float(item['end'])
            for box in item['boxes']:
                box['timestamp']= float(box['timestamp'])
                # Check if any coordinate is > 1 to trigger normalization, not just the first one (which could be 0)
                if any(c > 1.0 for c in box['box_2d']):
                    box['box_2d'] = [float(coord / 1000) for coord in box['box_2d']]
        
        # Segment creation
        gt_segments = [Segment(start=item['start'], end=item['end'], text=item.get('text', ''), boxes=item.get('boxes', [])) for item in gt_json]
        
        
        for item in pred_json:
            item['start'] = float(item['start'])
            item['end'] = float(item['end'])
            for box in item['boxes']:
                box['timestamp']= float(box['timestamp'])
                # Check if any coordinate is > 1
                if any(c > 1.0 for c in box['box_2d']):
                    box['box_2d'] = [float(coord / 1000) for coord in box['box_2d']]
        # Segment creation
        pred_segments  = [Segment(start=item['start'], end=item['end'], text=item.get('text', ''), boxes=item.get('boxes', [])) for item in pred_json]

        comparison_results = compare_transcripts(pred_segments, gt_segments)
        
        for key, value in comparison_results["metrics"].items():
            if key in total_metrics:
                total_metrics[key] += value
        
        all_results.append({
            "video_id": ques["video_id"],
            "query": ques["query"],
            "prediction": pred_json,
            "ground_truth": gt_json,
            "evaluation": comparison_results
        })
        # Save detailed results to a JSON file
        output_dir = os.path.join(os.path.dirname(args.input_file),"results")
        if args.visualize and args.video_dir:
            video_path = os.path.join(args.video_dir, "2017", f"{ques['video_id']}.mkv")
            visualize_folder = os.path.basename(args.input_file).split(".")[0]
            if os.path.exists(video_path):
                visualize_grounding(video_path, pred_segments, gt_segments, output_dir=os.path.join(output_dir, f"visualizations_{visualize_folder}"))
            else:
                video_path = os.path.join(args.video_dir, "2019", f"{ques['video_id']}.mkv")
                if os.path.exists(video_path):
                    visualize_grounding(video_path, pred_segments, gt_segments, output_dir=os.path.join(output_dir, f"visualizations_{visualize_folder}"))
                else:
                    print(f"Video file not found: {video_path}")
        
    # Calculate average metrics
    num_questions = len(gt_questions)
    if num_questions > 0:
        for key in total_metrics:
            if "total" not in key and "matched" not in key:
                #total_metrics[key] /= num_questions
                total_metrics[key] /= num_pred
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "eval_results.json")
    with open(output_filename, 'w') as f:
        json.dump(all_results, f, indent=4)

    # Save summary to a text file
    summary_filename = os.path.join(output_dir, "eval_summary.txt")
    with open(summary_filename, 'w') as f:
        f.write("Evaluation Summary:\n")
        f.write("===================\n")
        for key, value in total_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write(f"\nTotal Questions: {num_questions}\n")

    print(f"\nEvaluation complete. Detailed results saved to {output_filename}")
    print(f"Summary saved to {summary_filename}")
    print("\nAggregated Metrics:")
    print(json.dumps(total_metrics, indent=4))
    print("Gt Questions:", len(gt_questions))
    print("Pred Questions:", len(ques2pred))
    print("Actual Pred Questions:", num_pred)



def process_bounding_boxes(segments, duration, fps, frame_count, width, height, color, caption_prefix, boxes_by_frame, interpolate=False):
    all_boxes = []
    for segment in segments:
        for box in segment.boxes:
            timestamp = box['timestamp']
            if timestamp <= 1.0:
                timestamp *= duration

            x0, y0, x1, y1 = box['box_2d'][0], box['box_2d'][1], box['box_2d'][2], box['box_2d'][3]

            if max(x0, y0, x1, y1) <= 1.0:
                x0, y0, x1, y1 = x0 * width, y0 * height, x1 * width, y1 * height

            start_frame = int(timestamp * fps)
            
            box_data = {
                'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                'color': color,
                'caption': f"{caption_prefix}",
                'frame': start_frame,
                'speaker': segment.text
            }
            
            all_boxes.append(box_data)

            if start_frame < frame_count:
                if start_frame not in boxes_by_frame:
                    boxes_by_frame[start_frame] = []
                
                boxes_by_frame[start_frame].append(box_data)

    if interpolate:
        # Group by speaker
        boxes_by_speaker = {}
        for box in all_boxes:
            speaker = box['speaker']
            if speaker not in boxes_by_speaker:
                boxes_by_speaker[speaker] = []
            boxes_by_speaker[speaker].append(box)
        
        # Interpolate
        for speaker, boxes in boxes_by_speaker.items():
            # Sort by frame/time
            boxes.sort(key=lambda x: x['frame'])
            
            for i in range(len(boxes) - 1):
                curr_box = boxes[i]
                next_box = boxes[i+1]
                
                frame_diff = next_box['frame'] - curr_box['frame']
                if frame_diff <= 0:
                    continue
                    
                time_diff_sec = frame_diff / fps
                
                # Check if within 2 seconds
                if time_diff_sec <= 2.0:
                    # Interpolate
                    for f in range(curr_box['frame'] + 1, next_box['frame']):
                        if f >= frame_count:
                            break
                        
                        ratio = (f - curr_box['frame']) / frame_diff
                        
                        interp_x0 = curr_box['x0'] + (next_box['x0'] - curr_box['x0']) * ratio
                        interp_y0 = curr_box['y0'] + (next_box['y0'] - curr_box['y0']) * ratio
                        interp_x1 = curr_box['x1'] + (next_box['x1'] - curr_box['x1']) * ratio
                        interp_y1 = curr_box['y1'] + (next_box['y1'] - curr_box['y1']) * ratio
                        
                        # Subtle color distinction
                        # Pred (Red): (0, 0, 255) -> (80, 80, 255)
                        # GT (Green): (0, 255, 0) -> (80, 255, 80)
                        base_color = curr_box['color']
                        interp_color = (
                            min(255, base_color[0] + 80),
                            min(255, base_color[1] + 80),
                            min(255, base_color[2] + 80)
                        )
                        # Ensure it is still somewhat distinct and not white
                        # If base is (0,0,255), result is (80,80,255) - Light Red
                        
                        interp_box = {
                            'x0': interp_x0, 'y0': interp_y0, 'x1': interp_x1, 'y1': interp_y1,
                            'color': interp_color,
                            'caption': curr_box['caption']
                        }
                        
                        if f not in boxes_by_frame:
                            boxes_by_frame[f] = []
                        boxes_by_frame[f].append(interp_box)


def process_subtitles(segments, duration, fps, frame_count, prefix, subtitles_by_frame):
    for segment in segments:
        start_time = segment.start
        end_time = segment.end

        if start_time <= 1.0:
            start_time *= duration
        if end_time <= 1.0:
            end_time *= duration

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        for frame_idx in range(start_frame, min(end_frame, frame_count)):
            if frame_idx not in subtitles_by_frame:
                subtitles_by_frame[frame_idx] = []
            subtitles_by_frame[frame_idx].append(f"{prefix}: {segment.text}")

def visualize_grounding(vid_path, pred_segments, gt_segments, visualize_subtitles=False, output_dir="output_videos"): 
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir) 

    cap = cv2.VideoCapture(vid_path) 
    if not cap.isOpened(): 
        print(f"Error opening video file {vid_path}") 
        return 

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = cap.get(cv2.CAP_PROP_FPS) 
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    duration = frame_count / fps if fps > 0 else 0 

    boxes_by_frame = {} 
    subtitles_by_frame = {}

    if duration > 0: 
        process_bounding_boxes(pred_segments, duration, fps, frame_count, width, height, (0, 0, 255), "Pred", boxes_by_frame, interpolate=True)
        process_bounding_boxes(gt_segments, duration, fps, frame_count, width, height, (0, 255, 0), "GT", boxes_by_frame, interpolate=True)
        process_subtitles(pred_segments, duration, fps, frame_count, "Pred", subtitles_by_frame)
        process_subtitles(gt_segments, duration, fps, frame_count, "GT", subtitles_by_frame) 

    base_name = os.path.splitext(os.path.basename(vid_path))[0] 
    output_filename = os.path.join(output_dir, f"{base_name}_annotated.mp4") 
    temp_video_filename = os.path.join(output_dir, f"{base_name}_temp_video.mp4") 
    temp_audio_filename = os.path.join(output_dir, f"{base_name}_temp_audio.aac") 

    # Extract audio 
    audio_extraction_result = subprocess.run([FFMPEG_BINARY, '-i', vid_path, '-vn', '-c:a', 'aac', temp_audio_filename, '-y'], capture_output=True, text=True) 

    audio_exists = os.path.exists(temp_audio_filename) and os.path.getsize(temp_audio_filename) > 0 

    if audio_extraction_result.returncode != 0: 
        print(f"Error extracting audio: {audio_extraction_result.stderr}") 
        audio_exists = False 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_video_filename, fourcc, fps, (width, height)) 

    current_frame = 0 
    while True: 
        ret, frame = cap.read() 
        if not ret: 
            break 

        if current_frame in boxes_by_frame: 
            for box in boxes_by_frame[current_frame]: 
                x0, y0, x1, y1 = int(box['x0']), int(box['y0']), int(box['x1']), int(box['y1']) 
                cv2.rectangle(frame, (x0, y0), (x1, y1), box['color'], 4) 

                caption = box['caption'] 
                (text_width, text_height), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2) 
                text_y = y0 - 10 if y0 - 10 > 10 else y0 + text_height + 5 
                cv2.putText(frame, caption, (x0, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, box['color'], 2) 

        if current_frame in subtitles_by_frame and visualize_subtitles:
            y_offset = height - 20
            for subtitle in subtitles_by_frame[current_frame]:
                color = (0, 0, 255) if subtitle.startswith("Pred") else (0, 255, 0)
                cv2.putText(frame, subtitle, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                y_offset -= 30

        out.write(frame) 
        current_frame += 1 

    cap.release() 
    out.release() 

    # Combine video and audio 
    if audio_exists: 
        combination_result = subprocess.run([FFMPEG_BINARY, '-i', temp_video_filename, '-i', temp_audio_filename, '-c:v', 'libx264', '-c:a', 'copy', output_filename, '-y'], capture_output=True, text=True) 
        if combination_result.returncode == 0: 
            print(f"Saved annotated video to {output_filename}") 
            os.remove(temp_video_filename) 
            os.remove(temp_audio_filename) 
        else: 
            print(f"Error combining video and audio: {combination_result.stderr}") 
            # Fallback to video without audio 
            os.rename(temp_video_filename, output_filename) 
            print(f"Saved annotated video (without audio) to {output_filename}") 
            #os.remove(temp_audio_filename) 
    else: 
        os.rename(temp_video_filename, output_filename) 
        print(f"Saved annotated video (no audio track found or error in extraction) to {output_filename}")


if __name__ == "__main__":
    args = parse_args()
    # call_merge_v2(args)
    parse_result(args)