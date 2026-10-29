"""
VoxHub views, organised in this order:

    1. IMPORTS
    2. CONSTANTS & SETTINGS
    3. COMMON  - shared helpers (face check, masking, decorators, notifications)
    4. COMMON  - registration & authentication (user side)
    5. USER    - feed, posts, likes, comments, saved
    6. USER    - profiles, follow, suggestions, reports, explore
    7. USER    - chat
    8. USER    - notifications
    9. USER    - account settings
   10. ADMIN   - auth, dashboard, users, reports, settings, notifications
"""

# =========================================================
# 1. IMPORTS
# =========================================================

# ---- standard library ----
import logging
import math
import os
import traceback
import uuid
from datetime import timedelta
from functools import wraps

# ---- third party ----
import cv2
import face_recognition
import numpy as np
from transformers import pipeline

# ---- django ----
from django.contrib import messages
from django.contrib.auth import (
    authenticate,
    login,
    logout,
    update_session_auth_hash,
)
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.files.base import ContentFile
from django.core.validators import validate_email
from django.db import transaction
from django.db.models import Count, Exists, Max, OuterRef, Prefetch, Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils.timezone import localtime
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST

# ---- local ----
from myapp.models import *
from myapp.voice.enrollment import create_voice_profile, verify_voice
from .models import UserProfile, User_posts

logger = logging.getLogger(__name__)


# =========================================================
# 2. CONSTANTS & SETTINGS
# =========================================================

# ---- face check (registration) ----
# Keep CIRCLE_RADIUS_RATIO in sync with the browser:
# the guide circle is 70% of the video height -> radius = 0.35 * height
CIRCLE_RADIUS_RATIO = 0.35

MIN_IMAGE_SIDE = 360                  # px, reject tiny images
MAX_IMAGE_BYTES = 5 * 1024 * 1024     # 5 MB (registration face capture)

SIZE_MIN, SIZE_MAX = 0.45, 0.95       # face width / circle diameter
MAX_CENTER_OFFSET = 0.30              # face centre distance / circle radius
MAX_ROLL_DEG = 10                     # head tilt
BRIGHT_MIN, BRIGHT_MAX = 60, 200      # mean face brightness (0-255)
MIN_SHARPNESS = 35                    # Laplacian variance on a 200x200 face crop
MIN_EYES = 1                          # set to 2 for a stricter check

# ---- post uploads ----
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".m4v")
MAX_IMAGE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_VIDEO_SIZE = 50 * 1024 * 1024   # 50 MB

# Used by explore() to filter photos / videos
VIDEO_RE = r'\.(mp4|webm|mov|ogg)$'

# ---- admin notifications ----
NOTIFY_KINDS = ("new_user", "report", "security")

# ---- user reports (moderation) ----
OPEN_STATUSES = ("pending", "inreview")   # Reports.status values that still need action
HIGH_RISK_REPORTERS = 10                  # different people with open reports -> "high"
MEDIUM_RISK_REPORTERS = 5                 # -> "medium"
MAX_REPORTS_PER_DAY = 10                  # per reporter, stops report spamming

# ---- loaded once at import time (not on every request) ----
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# AI comment moderation model
detector = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None
)


# =========================================================
# 3. COMMON - SHARED HELPERS
# =========================================================

def _fail(message):
    return JsonResponse({"success": False, "message": message})


def _login_error(request, message):
    return render(request, "login.html", {"error": message})

def community_guidelines(request):
    return render(request,"community_guidelines.html")

def about_us(request):
    return render(request,"about_us.html")


# ---------------------------------------------------------
# Face validation (registration)
# returns (ok: bool, error_message: str | None)
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Post image privacy (face masking)
# ---------------------------------------------------------

def mask_face_regions(image, faces):
    output = image.copy()
    image_height, image_width = output.shape[:2]

    for face in faces:
        # Never blur the uploader's face
        if face.is_uploader:
            continue

        # Never blur unregistered people
        if face.person_id is None:
            continue

        # Don't blur registered users who consented to THIS post
        if face.consent_status == "accepted":
            continue

        # Get face coordinates and keep them within image boundaries
        top = max(0, min(face.top, image_height))
        bottom = max(0, min(face.bottom, image_height))
        left = max(0, min(face.left, image_width))
        right = max(0, min(face.right, image_width))

        if bottom <= top or right <= left:
            continue

        region_height = bottom - top
        region_width = right - left

        # Extract the face region
        face_region = output[top:bottom, left:right].copy()

        if region_height < 3 or region_width < 3:
            cv2.rectangle(
                output,
                (left, top),
                (right - 1, bottom - 1),
                (0, 0, 0),
                -1
            )
            continue

        # Create a strongly blurred version
        kernel_size = min(99, region_height, region_width)

        if kernel_size % 2 == 0:
            kernel_size -= 1

        kernel_size = max(1, kernel_size)

        blurred_region = cv2.GaussianBlur(
            face_region,
            (kernel_size, kernel_size),
            30
        )

        # Create a circular mask
        circle_mask = np.zeros(
            (region_height, region_width),
            dtype=np.uint8
        )

        center = (region_width // 2, region_height // 2)

        # Use the smaller dimension so the mask is circular
        radius = max(
            1,
            min(region_width, region_height) // 2
        )

        cv2.circle(
            circle_mask,
            center,
            radius,
            255,
            -1
        )

        # Apply blur only inside the circle
        blurred_output = output[top:bottom, left:right]

        blurred_output[circle_mask > 0] = blurred_region[
            circle_mask > 0
        ]

    return output


def regenerate_post_image(post):
    """
    Rebuild the public image using the private original photo
    and the latest face-consent decisions.
    """

    if not post.original_photo:
        raise ValueError("Private original photo is missing.")

    post.original_photo.open("rb")

    try:
        original_bytes = post.original_photo.read()
    finally:
        post.original_photo.close()

    image_array = np.frombuffer(
        original_bytes,
        dtype=np.uint8,
    )

    original = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if original is None:
        raise ValueError("Could not read the private original photo.")

    # Reapply masking using the latest consent statuses.
    faces = list(post.detected_faces.all())

    safe_image = mask_face_regions(
        original,
        faces,
    )

    success, encoded_image = cv2.imencode(
        ".jpg",
        safe_image,
    )

    if not success:
        raise ValueError("Could not generate the safe image.")

    safe_bytes = encoded_image.tobytes()

    new_name = f"post_{post.pk}_{uuid.uuid4().hex}.jpg"

    # Remember previous public file names for cleanup.
    old_photo_name = post.photo.name
    old_masked_name = post.masked_photo.name

    # Save the newly generated safe image.
    post.photo.save(
        new_name,
        ContentFile(safe_bytes),
        save=False,
    )

    post.masked_photo.save(
        f"masked_{new_name}",
        ContentFile(safe_bytes),
        save=False,
    )

    post.save(
        update_fields=[
            "photo",
            "masked_photo",
        ]
    )

    # Delete old public copies after saving the replacements.
    for field, old_name in (
        (post.photo, old_photo_name),
        (post.masked_photo, old_masked_name),
    ):
        if old_name and old_name != field.name:
            try:
                field.storage.delete(old_name)
            except Exception as exc:
                print(
                    f"Could not delete old image {old_name}: {exc}"
                )

    return post


# ---------------------------------------------------------
# Admin helpers (decorator + notifications)
# ---------------------------------------------------------

def admin_required(view_func):
    """Allow only logged-in, active users in the 'admin' group
    (the same rule your admin_login_post uses)."""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        user = request.user
        if not user.is_authenticated:
            return redirect("admin_login")
        if not (user.is_active and user.groups.filter(name="admin").exists()):
            messages.warning(request, "You are not an admin.")
            return redirect("admin_login")
        return view_func(request, *args, **kwargs)
    return wrapper


def notify_admin(kind, message):
    """Create an admin notification. Never raises, so it can't break
    registration or login. Identical messages within 1 minute are skipped
    so repeated failed logins can't flood the bell."""
    try:
        message = message[:255]
        since = timezone.now() - timedelta(minutes=1)
        if AdminNotification.objects.filter(
            kind=kind, message=message, created_at__gte=since
        ).exists():
            return
        AdminNotification.objects.create(kind=kind, message=message)
    except Exception:
        logger.exception("Could not create admin notification")


def _get_prefs(user):
    prefs, _ = AdminNotificationPreference.objects.get_or_create(admin=user)
    return prefs


# ---------------------------------------------------------
# Report moderation helpers
# ---------------------------------------------------------

def _risk_level(unique_reporters):
    """low / medium / high, based on how many DIFFERENT people have
    open reports against an account (one person spamming the report
    button cannot raise it)."""
    if unique_reporters >= HIGH_RISK_REPORTERS:
        return "high"
    if unique_reporters >= MEDIUM_RISK_REPORTERS:
        return "medium"
    return "low"


def _safe_next(request, fallback):
    """Redirect to ?next= only if it points back to this site."""
    nxt = request.POST.get("next", "")
    if nxt and url_has_allowed_host_and_scheme(nxt, allowed_hosts={request.get_host()}):
        return redirect(nxt)
    return redirect(fallback)


def _is_protected(request, target):
    """Admins, staff and yourself must never be banned from the UI."""
    return (
        target.id == request.user.id
        or target.is_staff
        or target.is_superuser
        or target.groups.filter(name="admin").exists()
    )


# =========================================================
# 4. COMMON - REGISTRATION & AUTHENTICATION (USER SIDE)
# =========================================================

def registration(request):
    return render(request, "registration.html")


# Called by the registration page with fetch()
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


# Login = password + voice
def login_view(request):

    if request.user.is_authenticated:
        return redirect("user_home")

    if request.method != "POST":
        return render(request, "login.html")

    username = (request.POST.get("username") or "").strip()
    password = request.POST.get("password") or ""
    voice = request.FILES.get("voice")

    # 1. username + password, checked by Django against auth.User
    #    (authenticate() returns None for a banned user, i.e. is_active = 0)
    user = authenticate(request, username=username, password=password)

    if user is None:
        # right password but is_active = 0  ->  tell them they are banned
        candidate = User.objects.filter(username=username).first()
        if candidate and not candidate.is_active and candidate.check_password(password):
            return _login_error(request, "Your account has been banned.")
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
        return _login_error(request, "Voice does not match.")

    # 4. only now create the real Django session
    login(request, user)

    return redirect("user_home")


def logout_view(request):
    logout(request)
    return redirect("login")


# =========================================================
# 5. USER - FEED, POSTS, LIKES, COMMENTS, SAVED
# =========================================================

# ---------------------------------------------------------
# Home feed
# ---------------------------------------------------------

@login_required(login_url="login")
def user_home(request):
    profile = UserProfile.objects.filter(LOGIN=request.user)

    # filter() never returns None, so check for an empty queryset instead
    if not profile.exists():
        logout(request)
        return redirect("login")

    other_users = UserProfile.objects.exclude(LOGIN=request.user)

    posts = (User_posts.objects
        .select_related('USER')
        .prefetch_related('comments_set__USER')
        .annotate(
            like_count=Count('likes', distinct=True),
            liked=Exists(Likes.objects.filter(POST=OuterRef('pk'), USER__LOGIN=request.user)),
        )
        .order_by('-id'))

    me = get_object_or_404(UserProfile, LOGIN=request.user)

    unread_count = ChatMessage.objects.filter(receiver=me, is_read=False).count()

    return render(request, "user_home.html", {"me": profile, "others": other_users, "posts": posts, "unread_count": unread_count})


# ---------------------------------------------------------
# Create / delete posts
# ---------------------------------------------------------

@login_required
@require_POST
def create_post(request):
    photo = request.FILES.get("photo")
    caption = request.POST.get("caption", "").strip()[:100]

    if not photo:
        return JsonResponse({"ok": False, "error": "No file received."}, status=400)

    extension = os.path.splitext(photo.name)[1].lower()
    content_type = (photo.content_type or "").lower()

    if content_type.startswith("video/"):
        is_video = True
        if extension not in VIDEO_EXTENSIONS:
            return JsonResponse({
                "ok": False,
                "error": "Unsupported video format. Use MP4, MOV or WebM.",
            }, status=400)
        if photo.size > MAX_VIDEO_SIZE:
            return JsonResponse({
                "ok": False,
                "error": "Video must be 50 MB or smaller.",
            }, status=400)
    elif content_type.startswith("image/"):
        is_video = False
        if photo.size > MAX_IMAGE_SIZE:
            return JsonResponse({
                "ok": False,
                "error": "Photo must be 10 MB or smaller.",
            }, status=400)
    else:
        return JsonResponse({"ok": False, "error": "Unsupported file type."}, status=400)

    try:
        profile = UserProfile.objects.get(LOGIN=request.user)
    except UserProfile.DoesNotExist:
        return JsonResponse({"ok": False, "error": "No user profile found."}, status=400)

    # ---------- VIDEO: store as-is, no face processing ----------
    if is_video:
        post = None
        try:
            post = User_posts(USER=profile, caption=caption)
            post.photo.save(
                f"video_{uuid.uuid4().hex}{extension}",
                photo,
                save=False,
            )
            post.save()
            return JsonResponse({
                "ok": True,
                "message": "Video posted successfully.",
                "post_id": post.pk,
            })
        except Exception as exc:
            print(f"Video post failed: {exc!r}")
            traceback.print_exc()
            if post is not None and post.photo and post.photo.name:
                try:
                    post.photo.storage.delete(post.photo.name)
                except Exception as cleanup_exc:
                    print(f"File cleanup failed: {cleanup_exc!r}")
            return JsonResponse({
                "ok": False,
                "error": "The video could not be uploaded. Please try again.",
            }, status=500)

    # ---------- IMAGE: face detection, matching, masking ----------

    # Read the uploaded image.
    photo_bytes = photo.read()

    try:
        image_array = np.frombuffer(photo_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Invalid image file.")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces.
        locations = face_recognition.face_locations(rgb_image)
        detected_faces = []

        image_height, image_width = image.shape[:2]

        for location in locations:
            top, right, bottom, left = location
            encoding = None

            try:
                encodings = face_recognition.face_encodings(
                    rgb_image,
                    known_face_locations=[location],
                )
                if encodings:
                    encoding = encodings[0]
            except Exception as exc:
                print(f"Face encoding failed: {exc!r}")

            # Expand the face region.
            face_height = bottom - top
            face_width = right - left

            top = max(0, top - int(face_height * 0.35))
            bottom = min(
                image_height,
                bottom + int(face_height * 0.35),
            )
            left = max(0, left - int(face_width * 0.35))
            right = min(
                image_width,
                right + int(face_width * 0.35),
            )

            detected_faces.append({
                "encoding": encoding,
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left,
                "person": None,
                "is_uploader": False,
            })

    except Exception as exc:
        print(f"Image processing failed: {exc!r}")
        traceback.print_exc()

        return JsonResponse({
            "ok": False,
            "error": "Could not safely process this photo.",
        }, status=400)

    # Load registered users' reference-face encodings.
    registered_encodings = []

    users = UserProfile.objects.select_related("LOGIN").all()

    for user in users:
        if not user.original_user:
            continue

        try:
            user.original_user.open("rb")
            try:
                reference_bytes = user.original_user.read()
            finally:
                user.original_user.close()

            reference_array = np.frombuffer(
                reference_bytes,
                dtype=np.uint8,
            )
            reference_image = cv2.imdecode(
                reference_array,
                cv2.IMREAD_COLOR,
            )

            if reference_image is None:
                continue

            reference_rgb = cv2.cvtColor(
                reference_image,
                cv2.COLOR_BGR2RGB,
            )
            reference_faces = face_recognition.face_encodings(
                reference_rgb,
            )

            if reference_faces:
                registered_encodings.append({
                    "profile": user,
                    "encoding": reference_faces[0],
                })

        except Exception as exc:
            print(
                f"Could not load reference for profile "
                f"{user.pk}: {exc!r}"
            )

    matched_users = set()

    # Match detected faces to registered users.
    for face in detected_faces:
        encoding = face["encoding"]

        # Faces without encodings remain unidentified.
        if encoding is None:
            continue

        distances = []

        for registered in registered_encodings:
            distance = float(
                face_recognition.face_distance(
                    [registered["encoding"]],
                    encoding,
                )[0]
            )
            distances.append((distance, registered["profile"]))

        distances.sort(key=lambda item: item[0])

        # No reliable match: treat the face as unregistered.
        if not distances or distances[0][0] > 0.45:
            continue

        best_distance, best_profile = distances[0]

        # Avoid uncertain matches between similar candidates.
        if (
            len(distances) > 1
            and distances[1][0] - best_distance < 0.05
        ):
            continue

        if best_profile.LOGIN_id == request.user.id:
            # The uploader's face stays visible.
            face["is_uploader"] = True
            face["person"] = best_profile
        else:
            # A matched registered person other than the uploader.
            face["person"] = best_profile
            matched_users.add(best_profile.pk)

    post = None

    try:
        with transaction.atomic():
            post = User_posts(
                USER=profile,
                caption=caption,
            )

            # Save the original image using private storage.
            if extension not in IMAGE_EXTENSIONS:
                extension = ".jpg"

            post.original_photo.save(
                f"original_{uuid.uuid4().hex}{extension}",
                ContentFile(photo_bytes),
                save=False,
            )
            post.save()

            # Store a face record for every detected face.
            for face in detected_faces:
                PostFace.objects.create(
                    post=post,
                    person=face["person"],
                    top=face["top"],
                    right=face["right"],
                    bottom=face["bottom"],
                    left=face["left"],
                    is_uploader=face["is_uploader"],
                    consent_status=(
                        "accepted"
                        if face["is_uploader"]
                        else "pending"
                    ),
                )

            # Generate the public image with masking applied.
            regenerate_post_image(post)

            # Notify each matched registered person once.
            for recipient_id in matched_users:
                recipient = UserProfile.objects.get(
                    pk=recipient_id,
                )

                notification = UserNotification.objects.create(
                    recipient=recipient,
                    sender=profile,
                    post=post,
                    message=(
                        f"{profile.name} uploaded a photo containing "
                        "a face that may match your registration photo. "
                        "Respond to control whether your face appears."
                    ),
                    status="pending",
                )

                print(
                    f"NOTIFICATION CREATED: {notification.id}; "
                    f"recipient={recipient.pk}; post={post.pk}"
                )

        return JsonResponse({
            "ok": True,
            "message": "Post created successfully.",
            "post_id": post.pk,
        })

    except Exception as exc:
        print(f"Post creation failed: {exc!r}")
        traceback.print_exc()

        # Clean up saved files if processing fails.
        if post is not None:
            for field_name in (
                "photo",
                "masked_photo",
                "original_photo",
            ):
                field = getattr(post, field_name, None)

                if field and field.name:
                    try:
                        field.storage.delete(field.name)
                    except Exception as cleanup_exc:
                        print(
                            f"File cleanup failed: {cleanup_exc!r}"
                        )

        return JsonResponse({
            "ok": False,
            "error": (
                "The photo could not be processed safely. "
                "Please try again."
            ),
        }, status=500)


@login_required(login_url="login")
@require_POST
def delete_post(request):
    """Delete a post owned by the logged-in user."""

    me = UserProfile.objects.filter(LOGIN=request.user).first()

    if not me:
        return JsonResponse({
            "ok": False,
            "error": "Your account has no profile."
        }, status=403)

    post_id = request.POST.get("postid", "").strip()

    if not post_id.isdigit():
        return JsonResponse({
            "ok": False,
            "error": "Invalid post ID."
        }, status=400)

    # Find the post by ID.
    post = User_posts.objects.filter(id=post_id).first()

    if not post:
        return JsonResponse({
            "ok": False,
            "error": "Post not found. It may have already been deleted."
        }, status=404)

    # Verify that the logged-in user owns the post.
    if post.USER_id != me.id:
        return JsonResponse({
            "ok": False,
            "error": "You are not allowed to delete this post."
        }, status=403)

    try:
        with transaction.atomic():
            # Delete the database record first.
            # Related records configured with CASCADE will also be removed.
            post.delete()

        # Clean up uploaded files after successful database deletion.
        for field_name in ("photo", "masked_photo", "original_photo"):
            field = getattr(post, field_name, None)

            if field and field.name:
                try:
                    field.storage.delete(field.name)
                except Exception as exc:
                    print(
                        f"Could not delete file {field.name}: {exc}"
                    )

        return JsonResponse({
            "ok": True,
            "message": "Post deleted successfully."
        })

    except Exception as exc:
        print(f"Delete post failed: {exc}")

        return JsonResponse({
            "ok": False,
            "error": "Could not delete the post. Please try again."
        }, status=500)


# ---------------------------------------------------------
# Likes
# ---------------------------------------------------------

@login_required
@require_POST
def toggle_like(request):
    post = get_object_or_404(User_posts, id=request.POST.get('postid'))
    profile = get_object_or_404(UserProfile, LOGIN=request.user)

    like, created = Likes.objects.get_or_create(USER=profile, POST=post)
    if not created:
        like.delete()

    return JsonResponse({
        'ok': True,
        'liked': created,
        'count': Likes.objects.filter(POST=post).count(),
    })


@login_required
def liked_posts(request):
    liked_posts = Likes.objects.filter(
        USER=request.user.user_profile
    ).select_related('POST').order_by('-id')

    return render(request, 'liked_posts.html', {
        'liked_posts': liked_posts
    })


# ---------------------------------------------------------
# Comments
# ---------------------------------------------------------

def add_comment(request):
    if request.method != "POST":
        return JsonResponse({
            'ok': False,
            'error': 'Invalid request method.'
        }, status=405)

    comment = request.POST.get('comment', '').strip()[:100]

    if not comment:
        return JsonResponse({
            'ok': False,
            'error': 'Please enter a comment.'
        }, status=400)

    C_POST = get_object_or_404(
        User_posts,
        id=request.POST.get('postid')
    )

    C_USER = get_object_or_404(
        UserProfile,
        LOGIN=request.user
    )

    # AI moderation must happen BEFORE saving
    try:
        results = detector(comment, truncation=True)[0]

        scores = {
            item['label'].lower(): item['score']
            for item in results
        }

        toxic_score = scores.get('toxic', 0.0)

        print("Comment:", comment)
        print("Toxicity score:", toxic_score)

    except Exception as e:
        print("AI moderation error:", e)
        return JsonResponse({
            'ok': False,
            'error': 'AI comment analysis failed. Your comment was not posted.'
        }, status=503)

    # Reject toxic comments and immediately stop execution
    if toxic_score >= 0.5:
        return JsonResponse({
            'ok': False,
            'error': 'This comment was analyzed by AI and identified as potentially toxic. Please edit your comment and try again.'
        }, status=400)

    # Save only after the comment passes moderation
    Comments.objects.create(
        comment=comment,
        date=timezone.now(),
        USER=C_USER,
        POST=C_POST
    )

    return JsonResponse({
        'ok': True,
        'username': C_USER.username,
        'comment': comment
    })


@login_required
@require_POST
def delete_comment(request):
    comment_id = request.POST.get('comment_id')

    if not comment_id:
        return JsonResponse({
            'ok': False,
            'error': 'Comment ID is required.'
        }, status=400)

    C_USER = get_object_or_404(
        UserProfile,
        LOGIN=request.user
    )

    comment = get_object_or_404(
        Comments,
        id=comment_id
    )

    # Allow users to delete only their own comments
    if comment.USER_id != C_USER.id:
        return JsonResponse({
            'ok': False,
            'error': 'You can delete only your own comments.'
        }, status=403)

    comment.delete()

    return JsonResponse({
        'ok': True,
        'message': 'Comment deleted successfully.'
    })


@login_required
def commented_posts(request):
    commented_posts = Comments.objects.filter(
        USER=request.user.user_profile
    ).select_related('POST').order_by('-date')

    return render(request, 'commented_posts.html', {
        'commented_posts': commented_posts
    })



# ---------------------------------------------------------
# Saved posts
# ---------------------------------------------------------

@login_required(login_url="login")
@require_POST
def toggle_save(request):
    """Save a post, or remove it from saved if it is already saved."""
    me = UserProfile.objects.filter(LOGIN=request.user).first()
    if not me:
        return JsonResponse({"ok": False, "error": "Your account has no profile."}, status=403)

    post_id = request.POST.get("postid", "")
    if not post_id.isdigit():
        return JsonResponse({"ok": False, "error": "Invalid post."}, status=400)

    post = User_posts.objects.filter(id=post_id).first()
    if not post:
        return JsonResponse({"ok": False, "error": "Post not found."}, status=404)

    saved, created = Saved_Posts.objects.get_or_create(USER=me, Post=post)
    if not created:
        saved.delete()

    return JsonResponse({"ok": True, "saved": created})


@login_required(login_url="login")
def saved_posts(request):
    """Page that lists every post the logged-in user has saved, newest first."""
    me = get_object_or_404(UserProfile, LOGIN=request.user)

    saved = (
        Saved_Posts.objects.filter(USER=me)
        .select_related("Post", "Post__USER")
        .order_by("-date", "-id")
    )

    return render(request, "saved_posts.html", {"saved": saved})


# =========================================================
# 6. USER - EXPLORE, PROFILES, FOLLOW, SUGGESTIONS, REPORTS
# =========================================================

@login_required(login_url="login")
def explore(request):
    q = request.GET.get('q', '').strip()
    kind = request.GET.get('type', '')
    me = get_object_or_404(UserProfile, LOGIN=request.user)

    posts = (User_posts.objects
        .select_related('USER')
        .annotate(
            like_count=Count('likes', distinct=True),
            comment_count=Count('comments', distinct=True),
        )
        .order_by('-id'))

    people = []
    if q:
        # search users by name or username
        people = (UserProfile.objects
            .filter(Q(name__icontains=q) | Q(username__icontains=q))
            .exclude(LOGIN=request.user)[:20])
        # posts: caption or author's name
        posts = posts.filter(
            Q(caption__icontains=q) | Q(USER__name__icontains=q) | Q(USER__username__icontains=q)
        )

    if kind == 'video':
        posts = posts.filter(photo__iregex=VIDEO_RE)
    elif kind == 'photo':
        posts = posts.exclude(photo__iregex=VIDEO_RE)

    return render(request, 'explore.html', {
        'q': q,
        'kind': kind,
        'people': people,
        'posts': posts,
        'liked_ids': set(Likes.objects.filter(USER=me).values_list('POST_id', flat=True)),
        'following_ids': set(Follow.objects.filter(FOLLOWER=me).values_list('FOLLOWING_id', flat=True)),
    })


@login_required
def view_profile(request, user_id):
    profile = get_object_or_404(UserProfile, id=user_id)
    me = get_object_or_404(UserProfile, LOGIN=request.user)

    posts = User_posts.objects.filter(USER=profile).order_by("-id")

    # The follow row (if any) between the logged-in user and this profile
    follow = Follow.objects.filter(FOLLOWER=me, FOLLOWING=profile).first()

    is_following = bool(follow and follow.status == "accepted")   # request accepted
    is_requested = bool(follow and follow.status == "pending")    # request sent, not answered yet

    # Count followers of this profile (accepted only)
    followers_count = Follow.objects.filter(
        FOLLOWING=profile, status="accepted"
    ).count()

    # Count users this profile follows (accepted only)
    following_count = Follow.objects.filter(
        FOLLOWER=profile, status="accepted"
    ).count()

    context = {
        "profile": profile,
        "posts": posts,
        "me": me,
        "post_count": posts.count(),
        "is_own_profile": profile.id == me.id,
        "is_following": is_following,
        "is_requested": is_requested,
        "followers_count": followers_count,
        "following_count": following_count,
        "follow": follow,
    }

    return render(request, "view_profile.html", context)


@login_required
def edit_profile(request):
    me = get_object_or_404(
        UserProfile,
        LOGIN=request.user
    )

    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        username = request.POST.get("username", "").strip()
        bio = request.POST.get("bio", "").strip()
        profile_image = request.FILES.get("profile_image")

        if not name or not username:
            messages.error(request, "Name and username are required.")
            return redirect("edit_profile")

        if UserProfile.objects.filter(
            username=username
        ).exclude(id=me.id).exists():
            messages.error(request, "Username already exists.")
            return redirect("edit_profile")

        me.name = name
        me.username = username

        # Only update the picture when a new one is uploaded.
        if profile_image:
            me.profile_image = profile_image

        # This requires a bio field in your UserProfile model.
        me.bio = bio
        me.save()

        messages.success(request, "Profile updated successfully!")
        return redirect("view_profile", user_id=me.id)

    return render(request, "edit_profile.html", {"me": me})


@login_required(login_url="login")
@require_POST
def toggle_follow(request, user_id):
    me = get_object_or_404(UserProfile, LOGIN=request.user)
    target = get_object_or_404(UserProfile, id=user_id)

    # you can't follow yourself
    if target.id == me.id:
        return redirect("view_profile", user_id=target.id)

    follow = Follow.objects.filter(FOLLOWER=me, FOLLOWING=target).first()

    if follow:
        # already following (accepted) or request pending -> unfollow / cancel
        follow.delete()
    else:
        # new follow starts as a request the other person has to accept
        Follow.objects.create(FOLLOWER=me, FOLLOWING=target, status="pending")

    return redirect("view_profile", user_id=target.id)


@login_required(login_url="login")
def suggestions(request):
    me = get_object_or_404(UserProfile, LOGIN=request.user)

    # Get IDs of users you already follow
    following_ids = Follow.objects.filter(
        FOLLOWER=me
    ).values_list("FOLLOWING_id", flat=True)

    # Exclude yourself and users already followed
    suggested_users = UserProfile.objects.exclude(
        id=me.id
    ).exclude(
        id__in=following_ids
    )

    return render(request, "suggestions.html", {
        "suggested_users": suggested_users
    })


# ---------------------------------------------------------
# Reporting a user
# ---------------------------------------------------------

@login_required(login_url="login")
@require_POST
def report_user(request, user_id):
    me = get_object_or_404(UserProfile, LOGIN=request.user)
    target = get_object_or_404(UserProfile, id=user_id)

    if target.id == me.id:
        return JsonResponse(
            {"ok": False, "error": "You cannot report yourself."},
            status=400
        )

    reason = (request.POST.get("reason") or "").strip()[:100]

    if not reason:
        return JsonResponse(
            {"ok": False, "error": "Please enter a reason."},
            status=400
        )

    # Block a second report while the first one is still open
    # (pending OR in review - the old check only looked at "pending")
    if Reports.objects.filter(
        USER=me,
        VICTIM=target,
        status__in=OPEN_STATUSES,
    ).exists():
        return JsonResponse(
            {"ok": False, "error": "You have already reported this user."},
            status=400
        )

    # Daily limit per reporter, so the report button cannot be spammed
    since = timezone.now() - timedelta(days=1)
    if Reports.objects.filter(USER=me, date__gte=since).count() >= MAX_REPORTS_PER_DAY:
        return JsonResponse(
            {"ok": False, "error": "Report limit reached for today. Please try again tomorrow."},
            status=429
        )

    # Save the report
    Reports.objects.create(
        USER=me,
        VICTIM=target,
        reports=reason
    )

    # Admin notification (de-duplicated, never raises)
    notify_admin("report", f"New report against {target.username}")

    return JsonResponse({
        "ok": True,
        "message": "Report submitted. Thank you."
    })

@login_required(login_url="login")
@require_POST
def submit_complaint(request):
    me = get_object_or_404(UserProfile, LOGIN=request.user)

    reason = (request.POST.get("reason") or "").strip()[:1000]

    if not reason:
        return JsonResponse({"ok": False, "error": "Please enter a reason."}, status=400)

    # Block duplicate pending reports with the same text
    if Complaints.objects.filter(USER=me, reports=reason, status="pending").exists():
        return JsonResponse({"ok": False, "error": "You have already submitted this report."}, status=400)

    Complaints.objects.create(USER=me, reports=reason)

    AdminNotification.objects.create(
        kind="report",
        message=f"New report submitted by {me.username}"
    )

    return JsonResponse({"ok": True, "message": "Report submitted. Thank you."})

@login_required(login_url="login")
def user_complaints(request):
    me = get_object_or_404(UserProfile, LOGIN=request.user)

    my_reports = Complaints.objects.filter(USER=me)

    # remember which ones the admin updated, then mark them as seen
    unseen_ids = list(my_reports.filter(user_seen=False).values_list("id", flat=True))
    Complaints.objects.filter(USER=me, user_seen=False).update(user_seen=True)

    return render(request, "user_complaints.html", {
        "reports": my_reports,
        "unseen_ids": unseen_ids,
    })

# =========================================================
# 7. USER - CHAT
# =========================================================

@login_required(login_url="login")
def chat_inbox(request):
    me = get_object_or_404(UserProfile, LOGIN=request.user)

    users = UserProfile.objects.exclude(id=me.id)

    for user in users:
        # unread messages from this person, ignoring ones I deleted
        user.unread_count = ChatMessage.objects.filter(
            sender=user,
            receiver=me,
            is_read=False
        ).exclude(deleted_for=me).count()

        # latest message in the conversation that I have not deleted
        user.last_message = ChatMessage.objects.filter(
            sender__in=[me, user],
            receiver__in=[me, user]
        ).exclude(deleted_for=me).order_by("-timestamp", "-id").first()

    return render(request, "chat_inbox.html", {
        "me": me,
        "users": users,
    })


@login_required(login_url="login")
def chat_room(request, user_id):
    me = get_object_or_404(UserProfile, LOGIN=request.user)
    other = get_object_or_404(UserProfile, id=user_id)

    if other.id == me.id:
        return redirect("chat_inbox")

    users = UserProfile.objects.exclude(id=me.id)

    ChatMessage.objects.filter(
        sender=other,
        receiver=me,
        is_read=False
    ).update(is_read=True)

    messages_list = ChatMessage.objects.filter(
        Q(sender=me, receiver=other) |
        Q(sender=other, receiver=me)
    ).exclude(deleted_for=me).select_related("sender", "receiver").order_by("timestamp", "id")

    return render(request, "chat_room.html", {
        "me": me,
        "other": other,
        "users": users,
        "chat_messages": messages_list,
    })


@login_required(login_url="login")
@require_POST
def send_message(request, user_id):
    me = get_object_or_404(UserProfile, LOGIN=request.user)
    receiver = get_object_or_404(UserProfile, id=user_id)

    if receiver.id == me.id:
        return JsonResponse(
            {"ok": False, "error": "You cannot message yourself."},
            status=400
        )

    message_text = request.POST.get("message", "").strip()

    if not message_text:
        return JsonResponse(
            {"ok": False, "error": "Message cannot be empty."},
            status=400
        )

    if len(message_text) > 2000:
        return JsonResponse(
            {"ok": False, "error": "Message is too long."},
            status=400
        )

    msg = ChatMessage.objects.create(
        sender=me,
        receiver=receiver,
        message=message_text
    )

    return JsonResponse({
        "ok": True,
        "id": msg.id,
        "sender_id": me.id,
        "message": msg.message,
        "timestamp": msg.timestamp.strftime("%d %b %H:%M"),
    })


@login_required(login_url="login")
def chat_messages(request, user_id):
    me = get_object_or_404(UserProfile, LOGIN=request.user)
    other = get_object_or_404(UserProfile, id=user_id)

    if other.id == me.id:
        return JsonResponse(
            {"ok": False, "error": "Invalid conversation."},
            status=400
        )

    ChatMessage.objects.filter(
        sender=other,
        receiver=me,
        is_read=False
    ).update(is_read=True)

    messages_list = ChatMessage.objects.filter(
        Q(sender=me, receiver=other) |
        Q(sender=other, receiver=me)
    ).exclude(deleted_for=me).select_related("sender").order_by("timestamp", "id")

    data = [
        {
            "id": msg.id,
            "sender_id": msg.sender_id,
            "message": msg.message,
            "timestamp": msg.timestamp.strftime("%d %b %H:%M"),
        }
        for msg in messages_list
    ]

    return JsonResponse({"ok": True, "messages": data})


@login_required(login_url="login")
@require_POST
def delete_message(request):
    """Hide a message for the person who deleted it. The other person still sees it."""
    me = get_object_or_404(UserProfile, LOGIN=request.user)

    message_id = request.POST.get("message_id", "")
    if not message_id.isdigit():
        return JsonResponse({"ok": False, "error": "Invalid message."}, status=400)

    msg = ChatMessage.objects.filter(
        Q(sender=me) | Q(receiver=me),
        id=message_id
    ).first()

    if not msg:
        return JsonResponse({"ok": False, "error": "Message not found."}, status=404)

    msg.deleted_for.add(me)
    return JsonResponse({"ok": True})


# =========================================================
# 8. USER - NOTIFICATIONS
#    (face-consent requests, follow requests, report updates)
# =========================================================

@login_required
def user_notifications(request):
    profile = UserProfile.objects.get(LOGIN=request.user)

    # Face detection notifications
    notifications = UserNotification.objects.filter(
        recipient=profile
    ).select_related(
        'sender', 'post'
    ).order_by('-id')

    pending_face_count = notifications.filter(
        status='pending'
    ).count()

    # Follow requests
    follow_requests = Follow.objects.filter(
        FOLLOWING=profile,
        status='pending'
    )

    # Reports I have submitted
    my_reports = Reports.objects.filter(USER=profile)

    return render(request, 'notification.html', {
        'notifications': notifications,
        'pending_face_count': pending_face_count,
        'follow_requests': follow_requests,
        'my_reports': my_reports,
    })


@login_required
def notifications(request):
    profile = get_object_or_404(
        UserProfile,
        LOGIN=request.user
    )

    face_notifications = UserNotification.objects.filter(
        recipient=profile
    ).select_related('sender', 'post')

    pending_face_count = face_notifications.filter(
        status='pending'
    ).count()

    follow_requests = Follow.objects.filter(
        FOLLOWING=profile,
        status='pending'
    ).select_related('FOLLOWER')

    my_reports = Reports.objects.filter(
        USER=profile
    )

    return render(
        request,
        'notifications.html',
        {
            'notifications': face_notifications,
            'pending_face_count': pending_face_count,
            'follow_requests': follow_requests,
            'my_reports': my_reports,
        }
    )


@login_required(login_url="login")
def accept_follow_request(request, id):
    if request.method == 'POST':
        Follow.objects.filter(id=id, FOLLOWING__LOGIN=request.user).update(status='accepted')
    return redirect('user_notifications')


@login_required(login_url="login")
def decline_follow_request(request, id):
    if request.method == 'POST':
        Follow.objects.filter(id=id, FOLLOWING__LOGIN=request.user).delete()
    return redirect('user_notifications')


@login_required
@require_POST
def respond_to_notification(request, notification_id):
    profile = get_object_or_404(UserProfile, LOGIN=request.user)

    notification = get_object_or_404(
        UserNotification,
        id=notification_id,
        recipient=profile
    )

    if notification.status != "pending":
        messages.warning(request, "You have already responded.")
        return redirect("user_notifications")

    action = request.POST.get("action")

    if action not in ["accept", "reject"]:
        messages.error(request, "Invalid action.")
        return redirect("user_notifications")

    faces = PostFace.objects.filter(
        post=notification.post,
        person=profile,
        is_uploader=False,
        consent_status="pending"
    )

    if not faces.exists():
        messages.error(
            request,
            "No pending face-visibility request was found."
        )
        return redirect("user_notifications")

    new_status = "accepted" if action == "accept" else "rejected"

    try:
        with transaction.atomic():
            faces.update(consent_status=new_status)

            notification.status = new_status
            notification.is_read = True
            notification.save(update_fields=["status", "is_read"])

            regenerate_post_image(notification.post)

        if action == "accept":
            messages.success(
                request,
                "You accepted the request. Your face can now appear in this post."
            )
        else:
            messages.success(
                request,
                "You rejected the request. Your face will remain hidden."
            )

    except Exception as exc:
        print(f"Face consent update failed: {exc}")
        messages.error(
            request,
            "Could not safely update the post image. Please try again."
        )

    return redirect("user_notifications")


@login_required
@require_POST
def delete_face_notification(request, id):
    notification = get_object_or_404(
        UserNotification,
        id=id,
        recipient=request.user.user_profile
    )

    notification.delete()

    messages.success(request, "Face detection request deleted successfully.")
    return redirect("user_notifications")


@login_required(login_url="login")
@require_POST
def delete_report_notification(request, id):
    """Remove a handled report from the reporter's notification list.
    Only the reporter can do this, and only after the admin has closed it
    (the old version let anyone delete any report - including the evidence -
    by visiting the URL)."""
    me = get_object_or_404(UserProfile, LOGIN=request.user)
    report = get_object_or_404(Reports, id=id, USER=me)

    if report.status in OPEN_STATUSES:
        messages.warning(request, "You can remove a report only after it has been handled.")
    else:
        report.delete()

    return redirect("user_notifications")


# =========================================================
# 9. USER - ACCOUNT SETTINGS
# =========================================================

@login_required
def settings_view(request):
    profile = get_object_or_404(
        UserProfile,
        LOGIN=request.user
    )

    return render(request, 'user_settings.html', {
        'sidebar_user': profile,
    })


@login_required
def change_password(request):
    if request.method == "POST":
        current_password = request.POST.get("current_password", "")
        new_password = request.POST.get("new_password", "")
        confirm_password = request.POST.get("confirm_password", "")

        if not request.user.check_password(current_password):
            messages.error(request, "Your current password is incorrect.")

        elif new_password != confirm_password:
            messages.error(request, "New passwords do not match.")

        elif current_password == new_password:
            messages.error(
                request,
                "Your new password must be different from your current password."
            )

        else:
            try:
                validate_password(new_password, user=request.user)
            except ValidationError as error:
                for message in error.messages:
                    messages.error(request, message)
            else:
                request.user.set_password(new_password)
                request.user.save()

                # Keep the user logged in after changing the password.
                update_session_auth_hash(request, request.user)

                messages.success(
                    request,
                    "Your password has been changed successfully."
                )
                return redirect("change_password")

    return render(request, "change_password.html")


# =========================================================
# 10. ADMIN
# =========================================================

# ---------------------------------------------------------
# Admin authentication
# ---------------------------------------------------------

def admin_login(request):
    return render(request, "admin_login.html")


def admin_login_post(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(
            request,
            username=username,
            password=password
        )

        print("Username:", username)
        print("Authentication result:", user)

        if user is not None:
            print("Admin group:", user.groups.filter(name="admin").exists())

            if user.groups.filter(name="admin").exists():
                login(request, user)
                return redirect("admin_home")
            else:
                messages.warning(request, "You are not an admin.")
        else:
            messages.warning(request, "Wrong username or password.")

    return redirect("admin_login")


# ---------------------------------------------------------
# Admin dashboard
# ---------------------------------------------------------

def admin_home(request):
    total_users = User.objects.filter(is_active=1, is_staff=0).count()
    banned_users = User.objects.filter(is_active=0).count()
    recent_users = UserProfile.objects.order_by('-created_at', '-id')[:5]
    active = User.objects.filter(is_active=1, is_staff=0).count()
    total_posts = User_posts.objects.all().count()
    # "pending" only counted brand-new reports; open = pending + in review
    pending_reports = Reports.objects.filter(status__in=OPEN_STATUSES).count()
    return render(request, 'admin_home.html', {
        'total_users': total_users,
        "banned_users": banned_users,
        "recent_users": recent_users,
        "active": active,
        "total_posts": total_posts,
        "pending_reports": pending_reports,
    })


# ---------------------------------------------------------
# Admin - user management
# ---------------------------------------------------------

def admin_users(request):
    active = UserProfile.objects.filter(LOGIN__is_active=True)
    banned = UserProfile.objects.filter(LOGIN__is_active=False).count()
    total = UserProfile.objects.all().count()
    return render(request, "admin_users.html", {'active': active, "banned": banned, "total": total})


def admin_banned_users(request):
    banned = UserProfile.objects.filter(LOGIN__is_active=False)
    return render(request, 'admin_banned_users.html', {'banned': banned})


@admin_required
@require_POST
def admin_ban_user(request, id):
    """Ban an account. Works from admin_users and from admin_reports
    (the form sends ?next so you return to the page you came from)."""
    target = get_object_or_404(User, id=id)

    if _is_protected(request, target):
        messages.warning(request, "Admin and staff accounts cannot be banned here.")
        return _safe_next(request, "admin_users")

    with transaction.atomic():
        target.is_active = False
        target.save(update_fields=["is_active"])

        # The account is handled, so its open reports are closed.
        closed = Reports.objects.filter(
            VICTIM__LOGIN=target,
            status__in=OPEN_STATUSES,
        ).update(status="resolved")

    messages.success(
        request,
        f"{target.username} was banned. {closed} open report(s) marked as resolved."
    )
    return _safe_next(request, "admin_users")


@admin_required
@require_POST
def admin_unban_user(request, id):
    target = get_object_or_404(User, id=id)
    target.is_active = True
    target.save(update_fields=["is_active"])

    messages.success(request, f"{target.username} was unbanned.")
    return _safe_next(request, "admin_users")


@admin_required
def view_userprofile(request, id):
    profile = get_object_or_404(UserProfile, LOGIN_id=id)

    posts = User_posts.objects.filter(USER=profile)

    return render(request, 'view_userprofile.html', {
        'users': [profile],
        'posts': posts,
    })

# ---------------------------------------------------------
# Admin - reports
# ---------------------------------------------------------

@admin_required
def admin_reports(request):
    open_q = Q(reports_received__status__in=OPEN_STATUSES)

    users = list(
        UserProfile.objects
        .filter(reports_received__isnull=False)
        .select_related("LOGIN")
        .annotate(
            report_count=Count("reports_received", distinct=True),
            open_count=Count("reports_received", filter=open_q, distinct=True),
            # distinct reporters among OPEN reports: one person pressing
            # "report" many times cannot push an account to high risk
            unique_reporters=Count("reports_received__USER", filter=open_q, distinct=True),
            last_reported=Max("reports_received__date"),
        )
        .prefetch_related(
            Prefetch(
                "reports_received",
                queryset=Reports.objects.select_related("USER", "USER__LOGIN").order_by("-date"),
            )
        )
        .order_by("-open_count", "-unique_reporters", "-last_reported")
    )

    for u in users:
        u.risk = _risk_level(u.unique_reporters) if u.open_count else "low"

    high_risk_users = [
        u for u in users
        if u.risk == "high" and u.LOGIN.is_active
    ]

    open_reports = Reports.objects.filter(status__in=OPEN_STATUSES).count()

    return render(request, "admin_reports.html", {
        "reported_users": users,
        "high_risk_users": high_risk_users,
        "open_reports": open_reports,
        "closed_reports": Reports.objects.exclude(status__in=OPEN_STATUSES).count(),
    })


@admin_required
def review_report(request, id):
    report = get_object_or_404(
        Reports.objects.select_related("USER", "VICTIM", "VICTIM__LOGIN"),
        id=id,
    )

    if report.status == "pending":
        report.status = "inreview"
        report.save(update_fields=["status"])

    return render(request, "review_report.html", {"report": report})


def _close_one_report(request, id, status):
    updated = Reports.objects.filter(id=id).update(status=status)

    if not updated:
        messages.warning(request, "Report not found.")
        return redirect("admin_reports")

    messages.success(request, f"Report marked as {status}.")
    return redirect("review_report", id=id)


@admin_required
@require_POST
def resolve_report(request, id):
    return _close_one_report(request, id, "resolved")


@admin_required
@require_POST
def dismiss_report(request, id):
    return _close_one_report(request, id, "dismissed")


@admin_required
@require_POST
def admin_close_reports(request, id):
    """Close every open report against one account at once.
    id = the account's auth User id (same id the ban buttons use)."""
    action = request.POST.get("action")

    if action not in ("resolved", "dismissed"):
        messages.warning(request, "Invalid action.")
        return _safe_next(request, "admin_reports")

    count = Reports.objects.filter(
        VICTIM__LOGIN_id=id,
        status__in=OPEN_STATUSES,
    ).update(status=action)

    messages.success(request, f"{count} open report(s) marked as {action}.")
    return _safe_next(request, "admin_reports")


COMPLAINT_OPEN = ("pending", "inreview")   # Complaints.status values that still need action


@admin_required
def admin_complaints(request):
    reports = (
        Complaints.objects
        .select_related("USER", "USER__LOGIN")
        .order_by("admin_seen", "-date")      # unopened first, then newest
    )

    open_count = reports.filter(status__in=COMPLAINT_OPEN).count()
    total = reports.count()

    return render(request, "admin_complaints.html", {
        "reports": reports,
        "total_count": total,
        "new_count": reports.filter(admin_seen=False).count(),
        "open_count": open_count,
        "closed_count": total - open_count,
    })


@admin_required
def review_complaints(request, id):
    report = get_object_or_404(
        Complaints.objects.select_related("USER", "USER__LOGIN"), id=id
    )

    changed = []
    if report.status == "pending":
        report.status = "inreview"
        changed.append("status")
    if not report.admin_seen:
        report.admin_seen = True
        changed.append("admin_seen")
    if changed:
        report.save(update_fields=changed)

    # the same user's other complaints, for context (repeat complaints stand out)
    others = Complaints.objects.filter(USER=report.USER).exclude(id=report.id).order_by("-date")

    return render(request, "review_complaints.html", {
        "report": report,
        "other_complaints": others[:5],
        "other_total": others.count(),
    })


@admin_required
@require_POST
def update_complaints(request, id):
    action = request.POST.get("action")

    # "inreview" = reopen a complaint that was closed by mistake
    if action not in ("resolved", "dismissed", "inreview"):
        messages.warning(request, "Invalid action.")
        return redirect("review_complaints", id=id)

    # user_seen=False so the reporter sees there is an update
    updated = Complaints.objects.filter(id=id).update(status=action, user_seen=False)

    if not updated:
        messages.warning(request, "Complaint not found.")
        return redirect("admin_complaints")

    label = "reopened" if action == "inreview" else f"marked as {action}"
    messages.success(request, f"Complaint {label}.")
    return redirect("review_complaints", id=id)


# ---------------------------------------------------------
# Admin - settings, profile, password
# ---------------------------------------------------------

@admin_required
def admin_settings(request):
    return render(request, "admin_settings.html", {"prefs": _get_prefs(request.user)})


@admin_required
@require_POST
def admin_update_profile(request):
    user = request.user

    username = (request.POST.get("username") or "").strip()
    email = (request.POST.get("email") or "").strip()

    if not username:
        messages.warning(request, "Username cannot be empty.")
        return redirect("admin_settings")

    if len(username) > 150:
        messages.warning(request, "Username is too long.")
        return redirect("admin_settings")

    if email:
        try:
            validate_email(email)
        except ValidationError:
            messages.warning(request, "Please enter a valid email address.")
            return redirect("admin_settings")

    if User.objects.filter(username__iexact=username).exclude(pk=user.pk).exists():
        messages.warning(request, "That username is already taken.")
        return redirect("admin_settings")

    user.username = username
    user.email = email
    user.save()

    messages.success(request, "Profile updated successfully!")
    return redirect("admin_settings")


@login_required
def admin_changepassword(request):

    if request.method != "POST":
        return redirect("admin_settings")

    user = request.user

    old_password = request.POST.get("old_password", "")
    new_password = request.POST.get("new_password", "")
    confirm_password = request.POST.get("confirm_password", "")

    # 1. Check current password
    if not user.check_password(old_password):
        messages.warning(request, "Current password is incorrect.")
        return redirect("admin_settings")

    # 2. Check empty fields
    if not new_password or not confirm_password:
        messages.warning(request, "All password fields are required.")
        return redirect("admin_settings")

    # 3. Check whether new passwords match
    if new_password != confirm_password:
        messages.warning(
            request,
            "New password and confirm password do not match."
        )
        return redirect("admin_settings")

    # 4. Prevent reusing the current password
    if user.check_password(new_password):
        messages.warning(
            request,
            "Your new password must be different from your current password."
        )
        return redirect("admin_settings")

    # 5. Validate password strength using Django validators
    try:
        validate_password(new_password, user=user)
    except ValidationError as errors:
        for error in errors.messages:
            messages.warning(request, error)
        return redirect("admin_settings")

    # 6. Save the new password securely
    user.set_password(new_password)
    user.save()

    # 7. Keep the admin logged in
    update_session_auth_hash(request, user)

    messages.success(request, "Password changed successfully!")

    return redirect("admin_settings")


@admin_required
@require_POST
def admin_save_notification_pref(request):
    key = request.POST.get("key", "")
    value = request.POST.get("value") == "true"

    if key not in NOTIFY_KINDS:
        return JsonResponse({"ok": False, "error": "Invalid setting."}, status=400)

    prefs = _get_prefs(request.user)
    setattr(prefs, key, value)
    prefs.save(update_fields=[key])
    return JsonResponse({"ok": True})


# ---------------------------------------------------------
# Admin - notification bell
# ---------------------------------------------------------

@login_required
def admin_notifications(request):
    if not request.user.is_staff:
        return JsonResponse({"error": "Forbidden"}, status=403)

    notifications = AdminNotification.objects.all()[:50]

    unread_count = AdminNotification.objects.filter(
        is_read=False
    ).count()

    items = [
        {
            "kind": notification.kind,
            "message": notification.message,
            "is_read": notification.is_read,
            "time": localtime(notification.created_at).strftime(
                "%b %d, %Y %I:%M %p"
            ),
        }
        for notification in notifications
    ]

    return JsonResponse({
        "unread": unread_count,
        "items": items,
    })


@login_required
@require_POST
@csrf_protect
def admin_notifications_read(request):
    if not request.user.is_staff:
        return JsonResponse({"error": "Forbidden"}, status=403)

    AdminNotification.objects.filter(
        is_read=False
    ).update(is_read=True)

    return JsonResponse({"success": True})