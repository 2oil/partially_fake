from pydub.playback import play
import os
import whisper_timestamped as stamp
from pydub import AudioSegment
import random
import csv

T_folder = r"C:\Users\user\Desktop\bonafide"
T_audio_files = os.listdir(T_folder)

F_folder = r"C:\Users\user\Desktop\Spoof"
F_audio_files = os.listdir(F_folder)

T_segments = []
F_segments = []
T_first_segments = []
F_first_segments = []
T_last_segments = []
F_last_segments = []

cut = 0.035

# T 오디오 파일 처리
for audio_file in T_audio_files:
    audio_path = os.path.join(T_folder, audio_file)
    audio = stamp.load_audio(audio_path)
    model = stamp.load_model("tiny", device="cpu")
    result = stamp.transcribe(model, audio, language="ko")
    segments = result.get("segments", [])
    audio_segment = AudioSegment.from_wav(audio_path)

    if segments:
        for segment_info in segments:
            words = segment_info.get("words", [])
            for word_info in words:
                start_time = int((word_info["start"] - cut) * 1000)  
                end_time = int((word_info["end"] + cut) * 1000)     
                segment = audio_segment[max(start_time, 0):end_time]  
                text = word_info["text"]
                T_segments.append({'segment': segment, 'text': text})

        # 첫 번째 세그먼트 저장
        first_word_info = segments[0].get("words", [])[0]
        start_time = int((first_word_info["start"] - cut) * 1000)
        end_time = int((first_word_info["end"] + cut) * 1000)
        first_segment = audio_segment[max(start_time, 0):end_time]
        T_first_segments.append({'segment': first_segment, 'text': first_word_info["text"]})

        # 마지막 세그먼트 저장
        last_word_info = segments[-1].get("words", [])[-1]
        start_time = int((last_word_info["start"] - cut) * 1000)
        end_time = int((last_word_info["end"] + cut) * 1000)
        last_segment = audio_segment[max(start_time, 0):end_time]
        T_last_segments.append({'segment': last_segment, 'text': last_word_info["text"]})


# F 오디오 파일 처리
for audio_file in F_audio_files:
    audio_path = os.path.join(F_folder, audio_file)
    audio = stamp.load_audio(audio_path)
    model = stamp.load_model("tiny", device="cpu")
    result = stamp.transcribe(model, audio, language="ko")
    segments = result.get("segments", [])
    audio_segment = AudioSegment.from_wav(audio_path)

    if segments:
        for segment_info in segments:
            words = segment_info.get("words", [])
            for word_info in words:
                start_time = int((word_info["start"] - cut) * 1000)  
                end_time = int((word_info["end"] + cut) * 1000)      
                segment = audio_segment[max(start_time, 0):end_time]  
                text = word_info["text"]
                F_segments.append({'segment': segment, 'text': text})
                
        # 첫 번째 세그먼트 저장
        first_word_info = segments[0].get("words", [])[0]
        start_time = int((first_word_info["start"] - cut) * 1000)
        end_time = int((first_word_info["end"] + cut) * 1000)
        first_segment = audio_segment[max(start_time, 0):end_time]
        F_first_segments.append({'segment': first_segment, 'text': first_word_info["text"]})

        # 마지막 세그먼트 저장
        last_word_info = segments[-1].get("words", [])[-1]
        start_time = int((last_word_info["start"] - cut) * 1000)
        end_time = int((last_word_info["end"] + cut) * 1000)
        last_segment = audio_segment[max(start_time, 0):end_time]
        F_last_segments.append({'segment': last_segment, 'text': last_word_info["text"]})


# 겹치는 세그먼트 삭제
def remove_duplicate_segments(main_segments, target_segments):
    for target_segment_list in target_segments:
        for target_segment in target_segment_list:
            for main_segment in main_segments:
                if main_segment['text'] == target_segment['text']:
                    main_segments.remove(main_segment)
                    break

# T_segments에서 T_first_segments와 T_last_segments 사이에 겹치는 세그먼트 삭제
remove_duplicate_segments(T_segments, [T_first_segments, T_last_segments])

# F_segments에서 F_first_segments와 F_last_segments 사이에 겹치는 세그먼트 삭제
remove_duplicate_segments(F_segments, [F_first_segments, F_last_segments])

# 리스트 길이 출력
print("T_segments 리스트 길이:", len(T_segments))
print("F_segments 리스트 길이:", len(F_segments))
print("T_first_segments 리스트 길이:", len(T_first_segments))
print("F_first_segments 리스트 길이:", len(F_first_segments))
print("T_last_segments 리스트 길이:", len(T_last_segments))
print("F_last_segments 리스트 길이:", len(F_last_segments))


# --------------------------------------------------------------------------------

num_sentences = 100
csv_data = []

for sentence_index in range(num_sentences):
    # T 또는 F에서 첫 번째 세그먼트 선택
    if random.random() < 0.5:  # 50% 확률로 T 세그먼트 선택
        selected_first_segment = random.choice(T_first_segments)
    else:
        selected_first_segment = random.choice(F_first_segments)

    # T 또는 F에서 마지막 세그먼트 선택
    if random.random() < 0.5:  # 50% 확률로 T 세그먼트 선택
        selected_last_segment = random.choice(T_last_segments)
    else:
        selected_last_segment = random.choice(F_last_segments)

    # T에서 한 개, F에서 한 개 세그먼트 추가
    selected_middle_segments = []
    selected_middle_segments.append(random.choice(T_segments))
    selected_middle_segments.append(random.choice(F_segments))

    mixed_segments = []

    # 선택한 세그먼트를 리스트에 추가
    mixed_segments.append(selected_first_segment)
    mixed_segments.extend(selected_middle_segments)
    mixed_segments.append(selected_last_segment)

    combined_audio = None
    total_duration = 0
    for segment_info in mixed_segments:
        segment = segment_info['segment']
        if combined_audio is None:
            combined_audio = segment
        else:
            combined_audio = combined_audio.append(segment, crossfade = cut*1000)

        start_time = total_duration
        end_time = total_duration + segment.duration_seconds
        total_duration = end_time

        # 세그먼트의 타입 및 텍스트 정보를 CSV 데이터에 추가
        if segment_info in T_first_segments or segment_info in T_last_segments or segment_info in T_segments:
            segment_type = 'T'
        else:
            segment_type = 'F'
        
        # 문장 번호를 추가하여 텍스트 수정
        sentence_number = sentence_index + 1
        text = segment_info['text']
        
        # 문장 번호를 시작 시간 앞에 추가
        csv_data.append({'start_time': f"#{sentence_number} {start_time}", 'end_time': end_time, 'type': segment_type, 'text': text})

    # 오디오 파일 저장
    audio_file_path = f"tiny_{sentence_index + 1}.wav"
    combined_audio.export(audio_file_path, format="wav")
    print(f"Audio file saved as: {audio_file_path}")

# CSV 파일 저장
csv_file = "tiny.csv"
with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as file:
    fieldnames = ['start_time', 'end_time', 'type', 'text']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)

print("\nCSV file saved as:", csv_file)