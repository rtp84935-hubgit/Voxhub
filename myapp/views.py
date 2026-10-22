from datetime import timedelta
import logging
import math

import cv2
from django.contrib import messages
import numpy as np
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST
from django.db.models import Count, Exists, OuterRef
from myapp.models import *
from myapp.voice.enrollment import create_voice_profile, verify_voice
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timesince, timezone
from django.contrib.auth.decorators import login_required

from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError


from .models import UserProfile, User_posts
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

    unread_count = ChatMessage.objects.filter(receiver=me,is_read=False).count()
    return render(request, "user_home.html", {"me": profile, "others": other_users, "posts": posts, "unread_count": unread_count})



VIDEO_RE = r'\.(mp4|webm|mov|ogg)$'

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
@require_POST
def create_post(request):
    photo = request.FILES.get('photo')
    caption = request.POST.get('caption', '').strip()[:100]

    if not photo:
        return JsonResponse({'ok': False, 'error': 'No file received'}, status=400)

    try:
        profile = UserProfile.objects.get(LOGIN=request.user)
    except UserProfile.DoesNotExist:
        return JsonResponse({'ok': False, 'error': 'No UserProfile for this login'}, status=400)

    post = User_posts.objects.create(USER=profile, photo=photo, caption=caption)
    return JsonResponse({'ok': True, 'id': post.id})


from transformers import pipeline

MODEL_NAME = "unitary/toxic-bert"

print("Loading model...")

detector = pipeline(
    "text-classification",
    model=MODEL_NAME,
    top_k=None
)





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
@require_POST
def toggle_follow(request, uid):
    me = get_object_or_404(UserProfile, LOGIN=request.user)
    target = get_object_or_404(UserProfile, id=uid)
    if target == me:
        return JsonResponse({'ok': False}, status=400)
    obj, created = Follow.objects.get_or_create(FOLLOWER=me, FOLLOWING=target)
    if not created:
        obj.delete()
    return JsonResponse({'ok': True, 'following': created})


from django.db.models import Q

@login_required(login_url="login")
def chat_inbox(request):
    me = get_object_or_404(
        UserProfile,
        LOGIN=request.user
    )

    users = UserProfile.objects.exclude(id=me.id)

    for user in users:
        user.unread_count = ChatMessage.objects.filter(
            sender=user,
            receiver=me,
            is_read=False
        ).count()

        user.last_message = ChatMessage.objects.filter(
            sender__in=[me, user],
            receiver__in=[me, user]
        ).order_by("-timestamp", "-id").first()

    return render(request, "chat_inbox.html", {
        "me": me,
        "users": users,
    })


@login_required(login_url="login")
def chat_room(request, user_id):
    me = get_object_or_404(
        UserProfile,
        LOGIN=request.user
    )

    other = get_object_or_404(
        UserProfile,
        id=user_id
    )

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
    ).select_related("sender", "receiver")

    return render(request, "chat_room.html", {
        "me": me,
        "other": other,
        "users": users,
        "chat_messages": messages_list,
    })


@login_required(login_url="login")
@require_POST
def send_message(request, user_id):
    me = get_object_or_404(
        UserProfile,
        LOGIN=request.user
    )

    receiver = get_object_or_404(
        UserProfile,
        id=user_id
    )

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
    me = get_object_or_404(
        UserProfile,
        LOGIN=request.user
    )

    other = get_object_or_404(
        UserProfile,
        id=user_id
    )

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
    ).select_related("sender").order_by("timestamp", "id")

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




@login_required
def view_profile(request, user_id):
    profile = get_object_or_404(UserProfile, id=user_id)

    me = get_object_or_404(
        UserProfile,
        LOGIN=request.user
    )

    posts = User_posts.objects.filter(
        USER=profile
    ).order_by("-id")

    # Check whether the logged-in user follows this profile
    is_following = Follow.objects.filter(
        FOLLOWER=me,
        FOLLOWING=profile
    ).exists()

    # Count followers of this profile
    followers_count = Follow.objects.filter(
        FOLLOWING=profile
    ).count()

    # Count users this profile follows
    following_count = Follow.objects.filter(
        FOLLOWER=profile
    ).count()

    context = {
        "profile": profile,
        "posts": posts,
        "me": me,
        "post_count": posts.count(),
        "is_own_profile": profile.id == me.id,
        "is_following": is_following,
        "followers_count": followers_count,
        "following_count": following_count,
    }

    return render(request, "view_profile.html", context)




@login_required
@require_POST
def toggle_follow(request, user_id):
    me = get_object_or_404(
        UserProfile,
        LOGIN=request.user
    )

    target = get_object_or_404(
        UserProfile,
        id=user_id
    )

    # Prevent following yourself
    if me.id == target.id:
        return redirect("view_profile", user_id=target.id)

    relationship = Follow.objects.filter(
        FOLLOWER=me,
        FOLLOWING=target
    )

    if relationship.exists():
        relationship.delete()
    else:
        Follow.objects.get_or_create(
            FOLLOWER=me,
            FOLLOWING=target
        )

    return redirect("view_profile", user_id=target.id)





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


@login_required(login_url="login")
@require_POST
def report_user(request, user_id):
    me = get_object_or_404(UserProfile, LOGIN=request.user)
    target = get_object_or_404(UserProfile, id=user_id)
 
    if target.id == me.id:
        return JsonResponse({"ok": False, "error": "You cannot report yourself."}, status=400)
 
    reason = (request.POST.get("reason") or "").strip()[:100]
 
    if not reason:
        return JsonResponse({"ok": False, "error": "Please enter a reason."}, status=400)
 
    # Don't let the same user file the same pending report again and again
    if Reports.objects.filter(USER=me, VICTIM=target, status="pending").exists():
        return JsonResponse(
            {"ok": False, "error": "You have already reported this user."},
            status=400,
        )
    
    Reports.objects.create(USER=me, VICTIM=target, reports=reason)
 
    notify_admin("report", f"{me.username} reported {target.username}.")
 
    return JsonResponse({"ok": True, "message": "Report submitted. Thank you."})
 

# =======================================ADMIN====================================

from functools import wraps
 
 
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
 
 
# =========================================================
# The missing view that the Admin Profile form posts to
# =========================================================
 
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

def admin_home(request):
    total_users=User.objects.filter(is_active=1,is_staff=0).count()
    banned_users=User.objects.filter(is_active=0).count()
    recent_users = UserProfile.objects.order_by('-created_at', '-id')[:5]
    active=User.objects.filter(is_active=1,is_staff=0).count()
    total_posts=User_posts.objects.all().count()
    pending_reports=Reports.objects.filter(status="pending").count()
    return render(request,'admin_home.html',{'total_users':total_users,"banned_users":banned_users,"recent_users":recent_users,"active":active,"total_posts":total_posts,"pending_reports":pending_reports})

def admin_login(request):
    return render(request,"admin_login.html")




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

def admin_users(request):
    active = UserProfile.objects.filter(LOGIN__is_active=True)
    return render(request,"admin_users.html",{'active':active})

def admin_banned_users(request):
    banned=UserProfile.objects.filter(LOGIN__is_active=False)
    return render(request,'admin_banned_users.html',{'banned':banned})

def admin_reports(request):
    reports=Reports.objects.all()
    return render(request,"admin_reports.html",{'reports':reports})

def admin_settings(request):
    return render(request,"admin_settings.html")

def admin_ban_user(request,id):
    ob=User.objects.get(id=id)
    ob.is_active = 0        
    ob.save()
    return redirect('admin_users')

def admin_unban_user(request,id):
    ob=User.objects.get(id=id)
    ob.is_active = 1
    ob.save()
    return redirect('admin_users')

def view_userprofile(request,id):
    users=UserProfile.objects.filter(LOGIN_id=id)
    posts=User_posts.objects.filter(USER__LOGIN_id=id)
    return render(request,"view_userprofile.html",{"users":users,"posts":posts})

from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError



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


NOTIFY_KINDS = ("new_user", "report", "security")
 
 
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
 
 
@admin_required
def admin_settings(request):
    return render(request, "admin_settings.html", {"prefs": _get_prefs(request.user)})
 
 
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
 
 
@admin_required
def admin_notifications(request):
    prefs = _get_prefs(request.user)
    enabled = [k for k in NOTIFY_KINDS if getattr(prefs, k)]
 
    qs = AdminNotification.objects.filter(kind__in=enabled)
    unread = qs.filter(is_read=False).count()
 
    items = [
        {
            "id": n.id,
            "kind": n.kind,
            "message": n.message,
            "time": timesince(n.created_at) + " ago",
            "is_read": n.is_read,
        }
        for n in qs[:15]
    ]
    return JsonResponse({"ok": True, "unread": unread, "items": items})
 
 
@admin_required
@require_POST
def admin_notifications_read(request):
    prefs = _get_prefs(request.user)
    enabled = [k for k in NOTIFY_KINDS if getattr(prefs, k)]
    AdminNotification.objects.filter(kind__in=enabled, is_read=False).update(is_read=True)
    return JsonResponse({"ok": True})
