import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.transform import resize
from tqdm import tqdm
import glob

def create_binary_spectrogram(audio_file, output_file=None, show_plot=False):
    """
    从音频文件生成二值化频谱图指纹
    
    参数:
        audio_file: 输入音频文件路径
        output_file: 输出图像路径(可选)
    
    返回:
        二值化频谱图数组
    """
    # 加载音频文件
    y, sr = librosa.load(audio_file, sr=None)
    
    # 计算STFT (短时傅里叶变换)
    D = librosa.stft(y, n_fft=2048, hop_length=64)
    
    # 将复数STFT转换为幅度谱
    magnitude = np.abs(D)
    
    # 取对数以突出较弱信号
    log_magnitude = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # 翻转频率轴使低频在底部
    log_magnitude = np.flipud(log_magnitude)
    
    # 应用阈值处理 - 只保留高能量区域
    threshold = threshold_otsu(log_magnitude)
    binary_spectrogram = log_magnitude > (threshold + 6)  # 增加阈值以减少白点
    
    # 调整图像大小为128x32
    binary_spectrogram_resized = resize(binary_spectrogram, (32, 128), 
                                         anti_aliasing=False, preserve_range=True).astype(bool)
    
    # 将布尔值转换为0和1
    binary_spectrogram_resized = binary_spectrogram_resized.astype(np.uint8)
    frequencyPeaks = np.reshape(binary_spectrogram_resized, (4096,))
    print(f"处理文件 {audio_file}: 4096位向量生成完成")
    
    return frequencyPeaks

def process_wav_files(audio_dir, output_file="audio/binary_array_dict.npy"):
    """
    处理指定目录下的所有WAV文件
    
    参数:
        audio_dir: 包含WAV文件的目录路径
        output_file: 输出的numpy文件名
    """
    # 查找所有WAV文件
    wav_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    
    if not wav_files:
        print(f"在目录 {audio_dir} 中没有找到WAV文件")
        return
    
    print(f"找到 {len(wav_files)} 个WAV文件")
    
    binary_array_dict = {}
    
    for index, wav_file in enumerate(tqdm(wav_files, desc="处理WAV文件")):
        try:
            # 直接处理WAV文件，无需临时文件
            binary_spec = create_binary_spectrogram(wav_file, output_file=None, show_plot=False)
            
            # 使用文件名作为键，或者使用索引
            filename = os.path.basename(wav_file)
            binary_array_dict[filename] = binary_spec
            # 或者使用索引: binary_array_dict[index] = binary_spec
            
        except Exception as e:
            print(f"处理文件 {wav_file} 时出错: {e}")
            continue
    
    print(f"成功处理了 {len(binary_array_dict)} 个文件")
    
    # 保存结果
    np.save(output_file, binary_array_dict)
    print(f"音频指纹向量已保存到: {output_file}")
    
    return binary_array_dict

if __name__ == "__main__":
    # 指定包含WAV文件的目录
    audio_directory = "./audio/dataset"  # 修改为你的WAV文件目录
    
    # 处理WAV文件
    result = process_wav_files(audio_directory)
    
    if result:
        print("处理完成!")
        print(f"生成的指纹字典包含 {len(result)} 个音频文件")