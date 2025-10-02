import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.transform import resize
from datasets import load_dataset
from tqdm import tqdm

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
    print(f"图像以4096位向量来进行表示: {frequencyPeaks}")
    
    # # 创建图像
    # plt.figure(figsize=(12, 3), facecolor='black')
    # plt.imshow(binary_spectrogram_resized, cmap='binary', aspect='auto', 
    #            interpolation='nearest')
    # plt.axis('off')  # 隐藏坐标轴
    
    # # 设置黑色背景
    # plt.gca().set_facecolor('black')
    
    # # 保存图像
    # if output_file:
    #     plt.savefig(output_file, bbox_inches='tight', pad_inches=0, 
    #                 facecolor='black', dpi=100)
    #     print(f"指纹图像已保存为: {output_file}")
    
    # plt.tight_layout()
    
    # if show_plot:
    #     plt.show()
    # else:
    #     plt.close()  # 不显示则关闭图形
    
    return frequencyPeaks

if __name__ == "__main__":
    # 方法1：直接使用Hugging Face API加载音频
    cache_dir = "./cache"
    # os.makedirs(cache_dir, exist_ok=True)
    # os.makedirs("./binary_pictures", exist_ok=True)

    print("正在从Hugging Face加载数据集...")
    dataset = load_dataset("danavery/urbansound8K", cache_dir=cache_dir)
    
    index = 0
    binary_array_dict = {}
    for i in tqdm(dataset['train'] , desc="处理音频数据集"):
        # 获取第一个音频样本并处理
        audio_sample = i['audio']
        y = audio_sample['array']
        sr = audio_sample['sampling_rate']
        
        # 保存为临时文件处理或修改create_binary_spectrogram函数接受数组
        temp_file = "temp_audio.wav"
        import soundfile as sf
        sf.write(temp_file, y, sr)
        
        output_path = f"./binary_pictures/fingerprint_{index}_from_dataset.png"
        binary_spec = create_binary_spectrogram(temp_file, output_file = None , show_plot=False)
        # 将二值化频谱图添加到字典中
        binary_array_dict[index] = binary_spec
        index += 1
        print(f"处理完成: {output_path}")
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)
    print(f"binary_array_dict: {binary_array_dict}")
    #保存
    np.save("binary_array_dict.npy", binary_array_dict)
    print("音频处理完成且对应的4096位向量也成功保存到文件中!")

    # # 测试4096位向量的形式
    # audio_sample = dataset['train'][1]['audio']
    # y = audio_sample['array']
    # sr = audio_sample['sampling_rate']
    # temp_file = "temp_audio.wav"
    # import soundfile as sf
    # sf.write(temp_file, y, sr)
    # binary_spec = create_binary_spectrogram(temp_file,output_file=None, show_plot=False)
    # # 清理临时文件
    # if os.path.exists(temp_file):
    #     os.remove(temp_file)