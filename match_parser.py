import librosa
import numpy as np
import scipy.signal as signal
from moviepy.editor import VideoFileClip, concatenate_videoclips



def find_sample_timestamps(audio_path, sample_path, threshold=0.8):
    # Load audio files
    y_video, sr_video = librosa.load(audio_path)
    y_sample, sr_sample = librosa.load(sample_path, sr=sr_video)
    
    # Ensure both audios have the same sample rate
    if sr_video != sr_sample:
        y_sample = librosa.resample(y_sample, orig_sr=sr_sample, target_sr=sr_video)

    # Compute the cross-correlation using scipy.signal.correlate with FFT
    correlation = signal.correlate(y_video, y_sample, mode='valid', method='fft')

    # Normalize the correlation
    correlation = correlation / np.max(np.abs(correlation))

    # Find peaks in correlation
    peaks, _ = signal.find_peaks(correlation, height=threshold, distance=int(sr_video * 0.5))  # 0.5 second minimum distance

    # Convert sample indices to timestamps
    timestamps = peaks / sr_video

    return sorted(set(np.int32(timestamps)))


def write_timestamps_to_file(timestamps, output_file):
    with open(output_file, 'w') as file:
        for t in timestamps:
            file.write(f"{t:.2f}\n")

def cut_and_concatenate_video_segments(input_video_path, output_video_path, timestamps):
    """
    Cut a video into segments based on a list of start and end timestamps,
    then concatenate these segments into a single output video.
    
    :param input_video_path: Path to the input video file
    :param output_video_path: Path for the output concatenated video file
    :param timestamps: List of tuples, each containing (start_time, end_time) in seconds
    """
    # Load the video file
    video = VideoFileClip(input_video_path)
    
    # Cut the video into segments and store them in a list
    segments = []
    for start_time, end_time in timestamps:
        segment = video.subclip(start_time, end_time)
        segments.append(segment)
        print(f"Processed segment: {start_time:.2f} to {end_time:.2f}")
    
    # Concatenate all segments
    final_clip = concatenate_videoclips(segments)
    
    # Write the final concatenated video to a file
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    
    print(f"Concatenated video saved: {output_video_path}")
    
    # Close the videos to release resources
    video.close()
    final_clip.close()
    for segment in segments:
        segment.close()

def find_time_segments(winning_points, losing_points):
    """
    Find time segments based on winning and losing point timestamps,
    including only segments that end with a losing point.
    
    :param winning_points: Sorted list of timestamps for winning points
    :param losing_points: Sorted list of timestamps for losing points
    :return: List of tuples (start_time, end_time) for each segment ending with a losing point
    """
    segments = []
    w_index, l_index = 0, 0
    w_len, l_len = len(winning_points), len(losing_points)
    last_end_time = 0

    while l_index < l_len:
        # Find the next losing point
        end_time = losing_points[l_index]
        
        # Move the winning_points index to just before the current losing point
        while w_index < w_len and winning_points[w_index] < end_time:
            last_end_time = winning_points[w_index]
            w_index += 1
        
        # Create a segment ending with the current losing point
        # Adding one extra second for buffer
        segments.append((last_end_time, end_time + 1))
        
        # Update the last end time and move to the next losing point
        last_end_time = end_time
        l_index += 1

    return segments

def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile("temp_audio.wav")
    return "temp_audio.wav"

if __name__ == "__main__":
    import sys
    import os
    # Example usage
    video_path = f"/mnt/c/Users/timot/Documents/ETT/videos/{sys.argv[1]}"
    sample_path = "loss-ball.wav"

    audio_path = extract_audio_from_video(video_path + ".mp4")
    losing_timestamps = find_sample_timestamps(audio_path, sample_path, threshold=0.5)

    #print("Timestamps where the loss ball sample is played:")
    write_timestamps_to_file(losing_timestamps, "loss_ball_timestamps.txt") # 0.8 threshold

    sample_path = "win-ball.wav"
    winning_timestamps = find_sample_timestamps(audio_path, sample_path, threshold=0.5)
    write_timestamps_to_file(winning_timestamps, "win_ball_timestamps.txt") # 0.5 threshold

    losing_point_time_segments = find_time_segments(winning_timestamps, losing_timestamps)
    
    cut_and_concatenate_video_segments(video_path + ".mp4", f"{video_path}_highlights.mp4", losing_point_time_segments)

    os.remove(audio_path)