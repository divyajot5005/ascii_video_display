from PIL import Image, ImageEnhance 
import time
import sys, errno
import os
import cv2
import numpy as np
from ascii_magic import AsciiArt
import msvcrt # for windows compatibility

class ConvertAscii:
    def __init__(self, width=200, contrast=1.5, brightness=1.2):
        self.width = width
        self.contrast = contrast
        self.brightness = brightness

    def convert_image_to_ascii(self, image_path, output_path):
        try:
            # Create ASCII art with enhanced settings
            art = AsciiArt.from_image(path=image_path)
            art.image = ImageEnhance.Contrast(art.image).enhance(self.contrast)
            art.image = ImageEnhance.Brightness(art.image).enhance(self.brightness)
            
            # Generate ASCII art with specified columns
            ascii_art = art.to_ascii(columns=self.width)
            
            # Save to file
            with open(output_path, 'w') as f:
                f.write(ascii_art)
        except Exception as e:
            print(f"Error converting image {image_path}: {str(e)}")
            raise e
        
    def convert_frames_to_ascii(self, frames_dir, ascii_dir):
        os.makedirs(ascii_dir, exist_ok=True)
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
        total_frames = len(frames)

        print("Converting frames to ASCII...")
        for idx, frame in enumerate(frames, 1):
            input_path = os.path.join(frames_dir, frame)
            output_path = os.path.join(ascii_dir, f"{frame.split('.')[0]}.txt")
            self.convert_image_to_ascii(input_path, output_path)
            print(f"Progress: {idx}/{total_frames} frames", end='\r')
        print("\nConversion complete!")

class ExtractFrame:
    @staticmethod
    def extract_frames(input_video, output_directory, target_fps=None):
        try:
            os.makedirs(output_directory, exist_ok=True)
            video = cv2.VideoCapture(input_video)
            
            # Get video properties
            original_fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame sampling interval
            if target_fps and target_fps < original_fps:
                frame_interval = int(original_fps / target_fps)
                expected_frames = int(frame_count / frame_interval)
            else:
                frame_interval = 1
                expected_frames = frame_count
                target_fps = original_fps
            
            print(f"Original Video FPS: {original_fps}")
            print(f"Target FPS: {target_fps}")
            print(f"Frame interval: {frame_interval}")
            print(f"Expected frames: {expected_frames}")

            # Initialize GPU
            use_gpu = False
            try:
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    print(f"Using GPU: {cv2.cuda.getDeviceName()}")
                    gpu_mat = cv2.cuda_GpuMat()
                    use_gpu = True
            except Exception as gpu_error:
                print(f"GPU initialization failed: {str(gpu_error)}")
            
            frame_idx = 0
            saved_frames = 0
            while True:
                success, frame = video.read()
                if not success:
                    break

                # Only process frames at the target interval
                if frame_idx % frame_interval == 0:
                    if use_gpu:
                        gpu_frame = cv2.cuda_GpuMat(frame)
                        frame = gpu_frame.download()

                    output_file = os.path.join(output_directory, f"frame_{saved_frames:04d}.png")
                    cv2.imwrite(output_file, frame)
                    saved_frames += 1
                    print(f"Extracting frames: {saved_frames}/{expected_frames}", end='\r')
                
                frame_idx += 1

            video.release()
            print(f"\nExtracted {saved_frames} frames to {output_directory}")
            return target_fps
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            raise e



class Display:
    @staticmethod
    def display_ascii_video(ascii_dir, frame_rate=24, loop=True):
        try:
            ascii_files = sorted([f for f in os.listdir(ascii_dir) if f.endswith('.txt')])
            if not ascii_files:
                raise FileNotFoundError("No ASCII frames found in directory")
            
            print("\nASCII Video Playback")
            print("Controls: '+' to speed up, '-' to slow down, 'q' to quit")
            
            while True:
                for ascii_file in ascii_files:
                    start_time = time.time()
                    
                    # Display frame
                    os.system('cls' if os.name == 'nt' else 'clear')
                    with open(os.path.join(ascii_dir, ascii_file), 'r') as f:
                        print(f.read())
                    
                    # Handle timing and controls
                    while (time.time() - start_time) < (1/frame_rate):
                        if msvcrt.kbhit():
                            key = msvcrt.getch().decode('utf-8').lower()
                            if key == '+':
                                frame_rate = min(frame_rate * 1.5, 60)
                                print(f"\nSpeed: {frame_rate:.1f} fps")
                            elif key == '-':
                                frame_rate = max(frame_rate / 1.5, 1)
                                print(f"\nSpeed: {frame_rate:.1f} fps")
                            elif key == 'q':
                                return
                        time.sleep(0.001)
                
                if not loop:
                    break
                
        except Exception as e:
            print(f"Display error: {str(e)}")
            raise e

def main():
    # Add these lines at the start of main()
    print(f"OpenCV version: {cv2.__version__}")
    print(f"CUDA enabled build: {cv2.cuda.getCudaEnabledDeviceCount() > 0}")
    print(f"OpenCV CUDA support: {cv2.getBuildInformation().find('CUDA') > 0}")
    
    
    try:
        if len(sys.argv) < 2:
            print("Usage: python main.py <video_file> [fps]")
            print("Example: python main.py video.mp4 15")
            sys.exit(1)

        input_video = sys.argv[1]
        # Get custom FPS from command line or use default of 15
        target_fps = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
        frames_dir = "output_frames"
        ascii_dir = "ascii_frames"

        # Clean up previous files
        for dir_path in [frames_dir, ascii_dir]:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    os.remove(os.path.join(dir_path, file))

        # Extract and convert frames with target FPS
        actual_fps = ExtractFrame.extract_frames(input_video, frames_dir, target_fps)
        converter = ConvertAscii(width=400, contrast=1.5, brightness=1.2)
        converter.convert_frames_to_ascii(frames_dir, ascii_dir)

        # Display video at the same target FPS
        Display.display_ascii_video(ascii_dir, frame_rate=actual_fps, loop=True)


    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()