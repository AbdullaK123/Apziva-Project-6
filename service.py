from typing import Annotated
import io 
import torch
import torchaudio
from torchaudio import transforms as audio_transforms
from torchvision import transforms as vision_transforms
import bentoml
from bentoml.models import BentoModel
from bentoml.validators import ContentType
from pathlib import Path
import tempfile
import os

# create the service
@bentoml.service(
    resources={'gpu': 1}
)
class AudioClassificationService:

    def __init__(self):
        self.model = torch.jit.load('models/audio_classifier.pt')


    async def process_audio_bytes(self, audio_bytes: bytes):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
            temp_wav.write(audio_bytes)
            temp_wav.flush()  # Make sure it's written to disk

            try:
                audio_tensor, sample_rate = torchaudio.load(temp_wav.name, backend='ffmpeg')
                return audio_tensor, sample_rate
            except Exception as e:
                print(f"Error loading audio: {str(e)}")
                print(f"File exists: {os.path.exists(temp_wav.name)}")
                print(f"File size: {os.path.getsize(temp_wav.name)}")
                raise

    async def process_audio(self, audio_bytes: bytes, target_length: int = 10 * 16000) -> torch.Tensor:
    
        # Load the audio file
        audio_tensor, sample_rate = await self.process_audio_bytes(audio_bytes)

        # Convert to mono if stereo
        if audio_tensor.shape[0] > 1:
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)

        # Pad or truncate to target length
        if audio_tensor.shape[1] < target_length:
            padding = torch.zeros(1, target_length - audio_tensor.shape[1])
            audio_tensor = torch.cat((audio_tensor, padding), dim=1)
        else:
            audio_tensor = audio_tensor[:, :target_length]

        spectrogram = audio_transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            power=2.0
        )(audio_tensor)

        mfcc = audio_transforms.MFCC(
                sample_rate=16000,
                n_mfcc=13,
                melkwargs={
                    'n_fft': 1024,
                    'hop_length': 512,
                    'n_mels': 128
                }
            )(audio_tensor)
        
        spectrogram = vision_transforms.Normalize(
                mean=[spectrogram.mean()],
                std=[spectrogram.std()]
            )(spectrogram)

        mfcc = vision_transforms.Normalize(
                mean=[mfcc.mean()],
                std=[mfcc.std()]
            )(mfcc)

        audio = audio_tensor / audio_tensor.abs().max()

        return spectrogram, mfcc, audio            
        

    @bentoml.api
    async def predict(self, file: Annotated[Path, ContentType('audio/wav')]):
        
        file_content = file.read_bytes()

        # Process the audio
        spectrogram, mfcc, audio = await self.process_audio(file_content)

        spectrogram = spectrogram.unsqueeze(0)  
        mfcc = mfcc.unsqueeze(0)                
        audio = audio.unsqueeze(0)  
        
        # Make prediction
        output = self.model(spectrogram, mfcc, audio)
        probability = torch.sigmoid(output).item()
        prediction = torch.round(torch.sigmoid(output)).item()
        
        return  {
            "prediction": prediction, 
            "probability": probability
        }








