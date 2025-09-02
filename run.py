import argparse
from pathlib import Path
import cv2

from ai_coach.camera import open_camera, list_devices_with_ffmpeg
from ai_coach.pose import PoseEstimator
from ai_coach.coach import HybridSquatCoach

def cmd_realtime(args: argparse.Namespace):
    """Runs the real-time AI Squat Coach application."""
    cap, meta, devices = open_camera(prefer_name=args.prefer_name, index=args.index, max_tests=args.max_tests)
    if cap is None:
        print("❌ No working camera found.")
        if devices:
            print("Available devices:")
            for i, name in devices.items():
                print(f"  [{i}] {name}")
        return

    print(f"✅ Using camera index {meta[0]} with backend {meta[1]}")

    # Load the best model trained by our refined auto-labeling pipeline.
    model_path = "models/squat_quality_multiclass_auto.joblib"
    if not Path(model_path).exists():
        print(f"❌ Model not found at '{model_path}'.")
        print("Please run the 'AI_Coach_Old_Approach_(Auto_Labeling).ipynb' notebook to train and save the model.")
        return
        
    coach = HybridSquatCoach(model_path=model_path)
    pose_estimator = PoseEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
        
        results, annotated_image = pose_estimator.process(frame)

        if results and results.pose_landmarks:
            # Draw the pose landmarks on the image.
            PoseEstimator.draw_landmarks(annotated_image, results.pose_landmarks)

            ui = coach.update(results.pose_landmarks.landmark)
            
            cv2.putText(annotated_image, f'Reps: {ui["rep_count"]}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_image, ui["display_quality"], (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, ui["display_color"], 2)
            y0 = 100
            for rec in ui["display_recs"]:
                cv2.putText(annotated_image, rec, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ui["display_color"], 2)
                y0 += 25
        else:
            cv2.putText(annotated_image, "No person detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('AI Squat Coach', annotated_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    pose_estimator.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Session ended.")

def main():
    parser = argparse.ArgumentParser(description="AI Squat Coach")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    p_rt = subparsers.add_parser("realtime", help="Run the real-time squat coaching application")
    p_rt.add_argument("--index", type=int, default=None, help="Force use of a specific camera index.")
    p_rt.add_argument("--prefer-name", type=str, default="default", help="Prefer a camera with a specific name (e.g., 'FaceTime').")
    p_rt.add_argument("--max-tests", type=int, default=5, help="Number of camera indices to test.")
    p_rt.set_defaults(func=cmd_realtime)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main() 