from fastapi import UploadFile, FastAPI, File
from pydantic import BaseModel
import io 
import torch
import torchaudio
from torchaudio import transforms as audio_transforms
from torchvision import transforms as vision_transforms
import tempfile
import os

# Set the audio backend to SoX
torchaudio.set_audio_backend("sox_io")

# Load the model
model = torch.jit.load("models/audio_classifier.pt")

# Response model
class PredictionResponse(BaseModel):
    probability: float
    prediction: int

# FastAPI app
app = FastAPI(
    title="Audio Classification API",
    description="An API to classify audio files",
    version="0.1"
)

async def process_audio_bytes(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_wav:
        temp_wav.write(audio_bytes)
        temp_wav.flush()  # Make sure it's written to disk

        try:
            audio_tensor, sample_rate = torchaudio.load(temp_wav.name, backend="sox")
            return audio_tensor, sample_rate
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            print(f"File exists: {os.path.exists(temp_wav.name)}")
            print(f"File size: {os.path.getsize(temp_wav.name)}")
            raise

async def process_audio(audio_bytes: bytes, target_length: int = 10 * 16000) -> torch.Tensor:
    
    # Load the audio file
    audio_tensor, sample_rate = await process_audio_bytes(audio_bytes)

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

@app.post("/predict/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Get raw bytes from the file
    file_content = await file.read()

    # Process the audio
    spectrogram, mfcc, audio = await process_audio(file_content)

    spectrogram = spectrogram.unsqueeze(0)  
    mfcc = mfcc.unsqueeze(0)                
    audio = audio.unsqueeze(0)  
    
    # Make prediction
    with torch.no_grad():
        output = model(spectrogram, mfcc, audio)
        probability = torch.sigmoid(output).item()
        prediction = torch.round(torch.sigmoid(output)).item()
    
    return PredictionResponse(
        prediction=prediction,
        probability=probability
    )