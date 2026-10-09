import os
import tempfile
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
from pydub import AudioSegment
from django.conf import settings
from django.utils.text import slugify

from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.dataio import audio_io

from myapp.models import VoiceProfile


@lru_cache(maxsize=1)
def get_speaker_model():
    """
    Load the pretrained speaker-recognition model only once.
    """

    print("Loading speaker recognition model...")

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(
            settings.BASE_DIR,
            "pretrained_models",
            "spkrec-ecapa-voxceleb"
        ),
        local_strategy=LocalStrategy.COPY
    )

    print("Speaker recognition model loaded.")

    return model


def normalize_embedding(embedding):
    """
    Normalize an embedding so cosine similarity can
    be calculated reliably later.
    """

    embedding = np.asarray(embedding, dtype=np.float32)

    norm = np.linalg.norm(embedding)

    if norm == 0:
        raise ValueError("Invalid voice embedding.")

    return embedding / norm


def extract_embedding(wav_path):
    """
    Convert one WAV file into a speaker embedding.
    """

    model = get_speaker_model()

    print("Loading WAV:", repr(wav_path))

    signal, sample_rate = audio_io.load(wav_path)      # [channels, time]

    print("Audio sample rate:", sample_rate)
    print("Audio shape:", signal.shape)

    if sample_rate != 16000:
        raise ValueError(
            f"Expected 16000 Hz audio, but got {sample_rate} Hz."
        )

    # Make sure the shape is exactly [1, time]  (batch, time)
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)

    if signal.shape[0] > 1:                            # stereo -> mono
        signal = signal.mean(dim=0, keepdim=True)

    with torch.no_grad():
        embedding = model.encode_batch(signal)         # no extra unsqueeze

    embedding = embedding.squeeze().cpu().numpy()      # (192,)

    return normalize_embedding(embedding)


def convert_to_wav(input_path, output_path):
    """
    Convert browser audio into:
        - mono
        - 16 kHz
        - WAV
    """

    audio = AudioSegment.from_file(input_path)

    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)

    audio.export(
        output_path,
        format="wav"
    )


def create_voice_profile(username, voices):
    """
    Create a speaker profile from the user's 10
    registration voice samples.

    voices:
        [voice1, voice2, ..., voice10]

    Returns:
        normalized speaker embedding (numpy array).

    NOTE: this function does NOT write to the database.
    register() in views.py saves the VoiceProfile after the
    UserProfile has been created.
    """

    if len(voices) != 10:
        raise ValueError("Exactly 10 voice samples are required.")

    print()
    print("======================================")
    print("CREATING VOICE PROFILE")
    print("Username:", username)
    print("Samples:", len(voices))
    print("======================================")

    # Folder where original recordings will be stored
    username_folder = slugify(username)

    voice_folder = (
        Path(settings.MEDIA_ROOT)
        / "voice_dataset"
        / username_folder
    )

    voice_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    embeddings = []

    for index, voice in enumerate(voices, start=1):

        if voice is None:
            raise ValueError(
                f"Voice sample {index} is missing."
            )

        print(f"Processing voice sample {index}/10...")

        # ------------------------------------------------
        # 1. Save original uploaded recording
        # ------------------------------------------------

        original_path = voice_folder / f"voice{index}.webm"

        with open(original_path, "wb+") as destination:

            for chunk in voice.chunks():
                destination.write(chunk)

        # ------------------------------------------------
        # 2. Create temporary WAV file
        # ------------------------------------------------

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:

            wav_path = temp_file.name

        try:

            # Convert WebM -> WAV
            convert_to_wav(
                str(original_path),
                wav_path
            )

            # ------------------------------------------------
            # 3. Check audio duration
            # ------------------------------------------------

            audio = AudioSegment.from_wav(wav_path)

            duration_seconds = len(audio) / 1000

            print(
                f"Voice {index} duration: "
                f"{duration_seconds:.2f} seconds"
            )

            if duration_seconds < 1.5:
                raise ValueError(
                    f"Voice sample {index} is too short. "
                    f"Please record at least 1.5 seconds."
                )

            # ------------------------------------------------
            # 4. Extract speaker embedding
            # ------------------------------------------------

            embedding = extract_embedding(wav_path)

            print(
                f"Voice {index} embedding size:",
                embedding.shape
            )

            embeddings.append(embedding)

        finally:

            # Delete temporary WAV
            if os.path.exists(wav_path):
                os.remove(wav_path)

    # ------------------------------------------------
    # 5. Combine all 10 embeddings
    # ------------------------------------------------

    print()
    print("Combining 10 voice embeddings...")

    embeddings = np.vstack(embeddings)

    # Average the 10 samples
    profile_embedding = np.mean(
        embeddings,
        axis=0
    )

    # Normalize final profile
    profile_embedding = normalize_embedding(
        profile_embedding
    )

    print(
        "Final profile embedding size:",
        profile_embedding.shape
    )

    # ------------------------------------------------
    # 6. No database save here (done in register())
    # ------------------------------------------------

    print()
    print("VOICE EMBEDDING CREATED")
    print("Username:", username)
    print("======================================")

    return profile_embedding


def verify_voice(username, login_voice):
    """
    Verify a login voice recording against the
    user's stored voice profile.
    """

    print()
    print("======================================")
    print("VOICE VERIFICATION")
    print("Username:", username)
    print("======================================")

    # Get the stored voice profile (VoiceProfile.USER -> UserProfile.username)
    try:
        profile = VoiceProfile.objects.get(
            USER__username=username
        )
    except VoiceProfile.DoesNotExist:
        raise ValueError(
            "No voice profile found for this username."
        )

    # Convert stored embedding back to numpy
    stored_embedding = np.asarray(
        profile.embedding,
        dtype=np.float32
    )

    stored_embedding = normalize_embedding(
        stored_embedding
    )

    # Create temporary WAV file
    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as temp_file:

        wav_path = temp_file.name

    try:
        # Save uploaded login voice
        with tempfile.NamedTemporaryFile(
            suffix=".webm",
            delete=False
        ) as input_file:

            input_path = input_file.name

            for chunk in login_voice.chunks():
                input_file.write(chunk)

        try:
            # Convert login recording to WAV
            convert_to_wav(
                input_path,
                wav_path
            )

            # Extract embedding from login voice
            login_embedding = extract_embedding(
                wav_path
            )

        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

        # Calculate cosine similarity
        similarity = float(
            np.dot(
                stored_embedding,
                login_embedding
            )
        )

        print(
            "Voice similarity:",
            similarity
        )

        # Temporary threshold
        threshold = 0.40

        verified = similarity >= threshold

        print(
            "Verification:",
            "PASSED" if verified else "FAILED"
        )

        print("======================================")

        return verified, similarity

    finally:

        if os.path.exists(wav_path):
            os.remove(wav_path)