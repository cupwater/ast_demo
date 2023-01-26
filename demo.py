# coding: utf8
import tqdm
import numpy as np


import soundfile
import webrtcvad
from paddlespeech.cli.asr.infer import ASRExecutor
from paddlespeech.cli.text.infer import TextExecutor

sample_rate = 16000

def m4atowav(m4a_path, wav_path):
    from pydub import AudioSegment
    # 将 m4a 格式录音转化成 wav 格式
    track = AudioSegment.from_file(m4a_path, format='m4a')
    track.export(wav_path, format='wav')
    return 

# 将长录音进行分块
def longAST2Chunks(wavfile_path, valid_sample_rate=sample_rate):


    # 使用 ffmpeg 进行转码, 统一到 16k采样率 16bit 音频
    # 如果你是其他采样率的音频文件，可以通过 Aistudio 自带的 ffmpeg 对音频进行转化
    # !ffmpeg -i demo.wav -ac 1 -ar 16000 -acodec pcm_s16le demo_16k.wav

    # 读取文件
    samples, sample_rate = soundfile.read(wavfile_path, dtype='int16')
    # 对音频进行分块
    x_len = len(samples)
    assert sample_rate == valid_sample_rate
    # VAD 以10ms为单位进行划分
    chunk_size = int(10 * sample_rate / 1000)
    if x_len % chunk_size != 0:
        padding_len_x = chunk_size - x_len % chunk_size
    else:
        padding_len_x = 0

    padding = np.zeros((padding_len_x), dtype=samples.dtype)
    padded_x = np.concatenate([samples, padding], axis=0)

    assert (x_len + padding_len_x) % chunk_size == 0
    num_chunk = (x_len + padding_len_x) / chunk_size
    num_chunk = int(num_chunk)
    chunk_wavs = []
    for i in range(0, num_chunk):
        start = i * chunk_size
        end = start + chunk_size
        x_chunk = padded_x[start:end]
        chunk_wavs.append(x_chunk)

    return chunk_wavs, samples

# 识别静音帧
def recognition_mute(chunk_wavs):
    # 每一个静音的长度
    sil_flag = False
    sil_indexs = []    # 切分后的 index 信息
    sil_length = []

    vad = webrtcvad.Vad(3)

    # 通过 webrtc 检测静音帧
    for idx, chunk in tqdm.tqdm(enumerate(chunk_wavs)):
        active = vad.is_speech(chunk.tobytes(), sample_rate)
        # print(active)
        if not active:
            # 是静音帧
            if not sil_flag:
                start = idx
            sil_flag = True
        else:
            if sil_flag:
                # 刚刚结束
                sil_flag = False
                end = idx
                sil_indexs.append((start, end, end - start))
                sil_length.append((end - start))

    return sil_indexs

# 根据静音帧来对长录音进行切分
def split_sub_wavs(sil_indexs, chunk_wavs):
    # 句子间的间隔时长为 1000 ms
    # 可以根据句子的实际情况进行调整
    min_sentence_sil_duration = 100  # VAD 一帧 10 ms
    split_start = 0
    sub_wavs = []
    for start, end, dur in tqdm.tqdm(sil_indexs):
        if dur > min_sentence_sil_duration:
            mid_split = int((start + end) / 2 * (sample_rate / 100))
            sub_wavs.append(samples[split_start:mid_split])
            split_start = mid_split
    # 最后结尾
    if split_start < len(chunk_wavs):
        sub_wavs.append(samples[split_start:len(chunk_wavs)])
    
    return sub_wavs


if __name__ == "__main__":
    m4a_path = "testing.m4a"
    wav_path = "testing.wav"

    m4atowav(m4a_path, wav_path)
    chunk_wavs, samples = longAST2Chunks(wav_path)
    sil_indexs = recognition_mute(chunk_wavs)
    sub_wavs = split_sub_wavs(sil_indexs, chunk_wavs)

    asr = ASRExecutor()
    text_punc = TextExecutor()

    asr_result = ""
    for idx, sub_wav in tqdm.tqdm(enumerate(sub_wavs)):
        audio = f"work/temp_{idx}.wav"
        soundfile.write(audio, sub_wav, sample_rate)
        asr_result = asr(audio_file=audio, model='conformer_online_wenetspeech')
        if asr_result:
            # 有识别结果才恢复标点
            text_result = text_punc(text=asr_result)
            asr_result += text_result
    print(asr_result)
