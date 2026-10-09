import logging
import math

import cv2
import numpy as np
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from myapp.models import UserProfile, VoiceProfile
from myapp.voice.enrollment import create_voice_profile, verify_voice

logger = logging.getLogger(__name__)

# =========================================================
# FACE CHECK SETTINGS
# Keep CIRCLE_RADIUS_RATIO in sync with the browser:
# the guide circle is 70% of the video height -> radius = 0.35 * height
# =========================================================

CIRCLE_RADIUS_RATIO = 0.35

MIN_IMAGE_SIDE = 360                  # px, reject tiny images
MAX_IMAGE_BYTES = 5 * 1024 * 1024     # 5 MB

SIZE_MIN, SIZE_MAX = 0.45, 0.95       # face width / circle diameter
MAX_CENTER_OFFSET = 0.30              # face centre distance / circle radius
MAX_ROLL_DEG = 10                     # head tilt
BRIGHT_MIN, BRIGHT_MAX = 60, 200      # mean face brightness (0-255)
MIN_SHARPNESS = 35                    # Laplacian variance on a 200x200 face crop
MIN_EYES = 1                          # set to 2 for a stricter check

# Load the cascades once at import time (not on every request)
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


# =========================================================
# SHARED FACE VALIDATION  (unchanged)
# returns (ok: bool, error_message: str | None)
# =========================================================

def validate_face_image(data: bytes):

    if _face_cascade.empty() or _eye_cascade.empty():
        return False, "Face detector could not be loaded."

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return False, "Could not read the image."

    h, w = img.shape[:2]

    if min(h, w) < MIN_IMAGE_SIDE:
        return False, "Image resolution is too low."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)      # only used for detection

    # EXACTLY ONE FACE
    faces = _face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(int(h * 0.25), int(h * 0.25)),
    )

    if len(faces) == 0:
        return False, "No face detected. Please face the camera clearly."

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    if len(faces) > 1 and faces[1][2] * faces[1][3] > 0.30 * faces[0][2] * faces[0][3]:
        return False, "Multiple faces detected. Only one person is allowed."

    x, y, fw, fh = [int(v) for v in faces[0]]

    # SIZE (relative to the guide circle)
    radius = CIRCLE_RADIUS_RATIO * h
    size_ratio = fw / (2 * radius)

    if size_ratio < SIZE_MIN:
        return False, "Face is too far. Please move closer."

    if size_ratio > SIZE_MAX:
        return False, "Face is too close. Please move slightly back."

    # POSITION (relative to the guide circle)
    cx, cy = x + fw / 2, y + fh / 2
    offset = math.hypot(cx - w / 2, cy - h / 2) / radius

    if offset > MAX_CENTER_OFFSET:
        return False, "Please keep your face in the centre of the circle."

    # EYES VISIBLE + HEAD LEVEL
    roi = gray_eq[y:y + int(fh * 0.6), x:x + fw]

    eyes = _eye_cascade.detectMultiScale(
        roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(fw // 10, 10), max(fw // 10, 10)),
    )

    if len(eyes) < MIN_EYES:
        return False, "Eyes not clearly visible. Look straight at the camera."

    if len(eyes) >= 2:
        two = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        two = sorted(two, key=lambda e: e[0])

        (ax, ay, aw, ah), (bx, by, bw, bh) = two

        angle = math.degrees(
            math.atan2(
                (by + bh / 2) - (ay + ah / 2),
                (bx + bw / 2) - (ax + aw / 2),
            )
        )

        if abs(angle) > MAX_ROLL_DEG:
            return False, "Please keep your head level."

    # LIGHTING
    face_gray = gray[y:y + fh, x:x + fw]
    brightness = float(face_gray.mean())

    if brightness < BRIGHT_MIN:
        return False, "Image is too dark. Please add more light."

    if brightness > BRIGHT_MAX:
        return False, "Image is too bright. Avoid strong backlight."

    # SHARPNESS (blur check)
    crop = cv2.resize(face_gray, (200, 200))
    sharpness = cv2.Laplacian(crop, cv2.CV_64F).var()

    if sharpness < MIN_SHARPNESS:
        return False, "Image is blurry. Please hold still."

    return True, None


def _fail(message):
    return JsonResponse({"success": False, "message": message})


# =========================================================
# REGISTRATION PAGE
# =========================================================

def registration(request):
    return render(request, "registration.html")


# =========================================================
# REGISTER  (called by the registration page with fetch)
# =========================================================

def register(request):

    if request.method != "POST":
        return _fail("Only POST request allowed.")

    # ---- basic details ----
    name = (request.POST.get("name") or "").strip()
    email = (request.POST.get("email") or "").strip()
    phone = (request.POST.get("phone") or "").strip()
    username = (request.POST.get("username") or "").strip()
    password = request.POST.get("password") or ""

    # ---- files ----
    profile_picture = request.FILES.get("profile_picture")
    original_image = request.FILES.get("original_image")
    voices = [request.FILES.get(f"voice{i}") for i in range(1, 11)]

    # ---- validation ----
    if not (name and email and phone and username and password):
        return _fail("Please fill all required fields.")

    if len(name) > 100 or len(username) > 50 or len(phone) > 15:
        return _fail("Name, username or phone number is too long.")

    try:
        validate_email(email)
    except ValidationError:
        return _fail("Please enter a valid email address.")

    if len(password) < 8:
        return _fail("Password must be at least 8 characters.")

    if not profile_picture:
        return _fail("Please upload a profile picture.")

    if not original_image:
        return _fail("Please capture your original image.")

    if original_image.size > MAX_IMAGE_BYTES:
        return _fail("Captured image is too large.")

    # Re-check the face on the server so the browser check can't be bypassed
    ok, error = validate_face_image(original_image.read())
    original_image.seek(0)          # rewind so the file can still be saved

    if not ok:
        return _fail(error)

    if any(voice is None for voice in voices):
        return _fail("Please record all 10 voice samples.")

    # ---- duplicate checks (cheap, before the slow voice step) ----
    # auth User is checked too, because create_user() would fail on it
    if (User.objects.filter(username__iexact=username).exists()
            or UserProfile.objects.filter(username__iexact=username).exists()):
        return _fail("Username is already taken.")

    if UserProfile.objects.filter(email__iexact=email).exists():
        return _fail("Email is already registered.")

    if UserProfile.objects.filter(phone_number=phone).exists():
        return _fail("Phone number is already registered.")

    # ---- voice profile ----
    try:
        voice_embedding = create_voice_profile(username, voices)
        embedding_list = voice_embedding.tolist()
    except Exception:
        logger.exception("Voice enrollment failed for %s", username)
        return _fail("Voice profile creation failed.")

    # ---- save everything together: all rows are created or none ----
    try:
        with transaction.atomic():

            user = User.objects.create_user(username=username, password=password)

            profile = UserProfile.objects.create(
                LOGIN=user,
                name=name,
                email=email,
                phone_number=phone,
                username=username,
                password=make_password(password),
                profile_image=profile_picture,
                original_user=original_image,
                voice_embedding=embedding_list,
                voice_sample_count=len(voices),
            )

            VoiceProfile.objects.update_or_create(
                USER=profile,
                defaults={
                    "embedding": embedding_list,
                    "sample_count": len(voices),
                },
            )

    except Exception:
        logger.exception("Registration failed for %s", username)
        return _fail("Registration failed. Please try again.")

    # This view is called with fetch(), so return JSON (a redirect would break it)
    return JsonResponse({
        "success": True,
        "message": "Registered successfully. Redirecting to login...",
        "redirect": 'login',
    })


# =========================================================
# FACE DETECTION  (unchanged)
# =========================================================

@require_POST
def detect_face(request):

    image = request.FILES.get("original_image")

    if not image:
        return JsonResponse({"face_detected": False, "error": "No image received."})

    if image.size > MAX_IMAGE_BYTES:
        return JsonResponse({"face_detected": False, "error": "Image is too large."})

    ok, error = validate_face_image(image.read())

    if not ok:
        return JsonResponse({"face_detected": False, "error": error})

    return JsonResponse({"face_detected": True, "message": "Face accepted."})


# =========================================================
# LOGIN  (password + voice)
# =========================================================

def _login_error(request, message):
    return render(request, "login.html", {"error": message})


def login_view(request):

    if request.user.is_authenticated:
        return redirect("user_home")

    if request.method != "POST":
        return render(request, "login.html")

    username = (request.POST.get("username") or "").strip()
    password = request.POST.get("password") or ""
    voice = request.FILES.get("voice")

    # 1. username + password, checked by Django against auth.User
    user = authenticate(request, username=username, password=password)

    if user is None:
        return _login_error(request, "Invalid username or password.")

    # 2. the account must have a VoxHub profile (e.g. admin accounts do not)
    profile = UserProfile.objects.filter(LOGIN=user).first()

    if profile is None:
        return _login_error(request, "Invalid username or password.")

    # 3. voice
    if not voice:
        return _login_error(request, "Please record your voice.")

    try:
        verified, similarity = verify_voice(profile.username, voice)
    except Exception:
        logger.exception("Voice login failed for %s", username)
        return _login_error(request, "Voice verification failed.")

    if not verified:
        return _login_error(request, f"Voice does not match. Similarity: {similarity:.2f}")

    # 4. only now create the real Django session
    login(request, user)

    return redirect("user_home")


def logout_view(request):
    logout(request)
    return redirect("login")


# =========================================================
# HOME  (only for logged-in users)
# =========================================================

@login_required(login_url="login")
def user_home(request):

    profile = UserProfile.objects.filter(LOGIN=request.user)
    other_users=UserProfile.objects.all().exclude(LOGIN=request.user)

    if profile is None:
        logout(request)
        return redirect("login")

    return render(request, "user_home.html", {"me": profile,"others":other_users})

def explore(request):
    return render(request,'explore.html')

# def create_post(request):
