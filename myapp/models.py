"""
VoxHub models, organised in this order:

    1. IMPORTS
    2. CONSTANTS & STORAGE
    3. USERS         - UserProfile, VoiceProfile
    4. POSTS         - User_posts, PostFace
    5. INTERACTIONS  - Comments, Likes, Saved_Posts, Follow
    6. CHAT          - ChatMessage, ChatSettings
    7. NOTIFICATIONS - UserNotification
    8. MODERATION    - Reports, Complaints
    9. ADMIN         - AdminNotification, AdminNotificationPreference,
                       PlatformSettings, AdminAction

Only the order and the comments changed. No field was added, removed or
renamed, so no new migration is needed.
"""

# =========================================================
# 1. IMPORTS
# =========================================================

import os

from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage
from django.db import models
from django.utils import timezone
from datetime import datetime, timedelta


# =========================================================
# 2. CONSTANTS & STORAGE
# =========================================================

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".m4v")

# Private folder for original photos (never served directly to the public)
PRIVATE_POST_STORAGE = FileSystemStorage(
    location=os.path.join(settings.BASE_DIR, "private_post_originals"),
    base_url=None,
)


# =========================================================
# 3. USERS
# =========================================================

class UserProfile(models.Model):
    LOGIN = models.OneToOneField(User, on_delete=models.CASCADE, related_name="user_profile")

    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    bio = models.TextField(blank=True, default="")
    phone_number = models.CharField(max_length=15, unique=True)
    username = models.CharField(max_length=50, unique=True)
    password = models.CharField(max_length=128)

    profile_image = models.FileField(upload_to="profile_images/")
    original_user = models.FileField(upload_to="original_images/")   # reference face

    # Voice biometric profile
    voice_embedding = models.JSONField(null=True, blank=True)
    voice_sample_count = models.PositiveIntegerField(default=0)

    created_at = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.username


class VoiceProfile(models.Model):
    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE)

    embedding = models.JSONField()
    sample_count = models.PositiveIntegerField(default=10)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Voice profile of {self.USER.username}"


# =========================================================
# 4. POSTS
# =========================================================

class User_posts(models.Model):
    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE)

    # The public-facing, privacy-protected image or existing video.
    photo = models.FileField()

    # Safe, masked image used by the feed and notifications.
    masked_photo = models.FileField(blank=True)

    # Original upload; must never be served directly to public viewers.
    original_photo = models.FileField(
        storage=PRIVATE_POST_STORAGE,
        upload_to="originals/",
        blank=True,
    )

    caption = models.CharField(max_length=100)
    date = models.DateTimeField(auto_now_add=True)

    @property
    def is_video(self):
        return os.path.splitext(self.photo.name or "")[1].lower() in VIDEO_EXTENSIONS

    @property
    def display_photo(self):
        return self.masked_photo or self.photo

    def __str__(self):
        return f"Post by {self.USER.username}"


class PostFace(models.Model):
    CONSENT_CHOICES = [
        ("pending", "Pending"),
        ("accepted", "Accepted"),
        ("rejected", "Rejected"),
    ]

    post = models.ForeignKey(
        User_posts,
        on_delete=models.CASCADE,
        related_name="detected_faces",
    )

    # Null for a face that could not be matched to a registered user.
    person = models.ForeignKey(
        UserProfile,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="faces_in_posts",
    )

    # Face coordinates in the original image.
    top = models.PositiveIntegerField()
    right = models.PositiveIntegerField()
    bottom = models.PositiveIntegerField()
    left = models.PositiveIntegerField()

    # The uploader's own face can remain visible.
    is_uploader = models.BooleanField(default=False)

    # Each face is tracked independently.
    consent_status = models.CharField(
        max_length=10,
        choices=CONSENT_CHOICES,
        default="pending",
    )

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return f"Face in post {self.post_id}"


# =========================================================
# 5. INTERACTIONS
# =========================================================

class Comments(models.Model):
    comment = models.CharField(max_length=100)
    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    POST = models.ForeignKey(User_posts, on_delete=models.CASCADE)
    date = models.DateTimeField(default=timezone.now)


class Likes(models.Model):
    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    POST = models.ForeignKey(User_posts, on_delete=models.CASCADE)

    class Meta:
        unique_together = ("USER", "POST")


class Saved_Posts(models.Model):
    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    Post = models.ForeignKey(User_posts, on_delete=models.CASCADE)
    date = models.DateTimeField(auto_now_add=True)


class Follow(models.Model):
    FOLLOWER = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="following")
    FOLLOWING = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="followers")
    status = models.CharField(max_length=10, default="accepted")  # pending / accepted / declined

    class Meta:
        unique_together = ("FOLLOWER", "FOLLOWING")


# =========================================================
# 6. CHAT
# =========================================================

class ChatMessage(models.Model):
    sender = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name="sent_messages",
    )
    receiver = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name="received_messages",
    )

    message = models.TextField(max_length=2000)
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    # "Delete for me": hidden for these users only
    deleted_for = models.ManyToManyField(UserProfile, blank=True, related_name="hidden_messages")

    class Meta:
        ordering = ["timestamp", "id"]

    def __str__(self):
        return f"{self.sender.username} -> {self.receiver.username}"


class ChatSettings(models.Model):
    THEME_CHOICES = [
        ("default", "Default"),
        ("ocean", "Ocean"),
        ("sunset", "Sunset"),
        ("forest", "Forest"),
        ("rose", "Rose"),
        ("violet", "Violet"),
        ("midnight", "Midnight"),
        ("mono", "Mono"),
    ]
    FONT_CHOICES = [
        ("small", "Small"),
        ("medium", "Medium"),
        ("large", "Large"),
    ]

    USER = models.OneToOneField(UserProfile, on_delete=models.CASCADE, related_name="chat_settings")

    theme = models.CharField(max_length=10, choices=THEME_CHOICES, default="default")
    font_size = models.CharField(max_length=6, choices=FONT_CHOICES, default="medium")

    enter_to_send = models.BooleanField(default=True)    # Enter sends (off: Ctrl+Enter sends)
    show_time = models.BooleanField(default=True)        # time under each message
    compact = models.BooleanField(default=False)         # smaller bubbles, tighter spacing
    show_preview = models.BooleanField(default=True)     # last message shown in the chat list

    def __str__(self):
        return f"Chat settings of {self.USER.username}"


# =========================================================
# 7. NOTIFICATIONS (user side: face-consent requests)
# =========================================================

class UserNotification(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("accepted", "Accepted"),
        ("rejected", "Rejected"),
    ]

    recipient = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name="notifications",
    )
    sender = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name="sent_notifications",
    )
    post = models.ForeignKey(
        User_posts,
        on_delete=models.CASCADE,
        related_name="notifications",
    )

    message = models.CharField(max_length=255)
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="pending")

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.sender} -> {self.recipient}: {self.message}"


# =========================================================
# 8. MODERATION (reports against users, general complaints)
# =========================================================

class Reports(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("inreview", "In review"),
        ("resolved", "Resolved"),
        ("dismissed", "Dismissed"),
    ]

    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="reports_made")
    VICTIM = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="reports_received")
    reports = models.CharField(max_length=100)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="pending", db_index=True)
    date = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-date"]

    def __str__(self):
        return f"{self.USER.username} -> {self.VICTIM.username} ({self.status})"


class Complaints(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("inreview", "In review"),
        ("resolved", "Resolved"),
        ("dismissed", "Dismissed"),
    ]

    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    reports = models.CharField(max_length=1000)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="pending", db_index=True)
    date = models.DateTimeField(auto_now_add=True)
    admin_seen = models.BooleanField(default=False)   # new complaints start unseen by admin
    user_seen = models.BooleanField(default=True)     # set to False when admin updates it

    class Meta:
        ordering = ["-date"]

    def __str__(self):
        return f"Complaint by {self.USER.username} ({self.status})"


# =========================================================
# 9. ADMIN
# =========================================================

class AdminNotification(models.Model):
    KIND_CHOICES = [
        ("new_user", "New user"),
        ("report", "Report"),
        ("security", "Security"),
    ]

    kind = models.CharField(max_length=20, choices=KIND_CHOICES)
    message = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    class Meta:
        ordering = ["-id"]

    def __str__(self):
        return f"[{self.kind}] {self.message}"


class AdminNotificationPreference(models.Model):
    admin = models.OneToOneField(User, on_delete=models.CASCADE, related_name="notification_pref")

    new_user = models.BooleanField(default=True)
    report = models.BooleanField(default=True)
    security = models.BooleanField(default=True)

    def __str__(self):
        return f"Notification preferences for {self.admin.username}"


class PlatformSettings(models.Model):
    """One row (pk=1) with the platform rules an admin can change in Settings."""
    registration_open = models.BooleanField(default=True)           # allow new sign-ups
    comment_moderation = models.BooleanField(default=True)          # AI check on comments
    toxicity_threshold = models.FloatField(default=0.5)             # lower = stricter
    reports_per_day = models.PositiveSmallIntegerField(default=10)  # per user
    complaints_per_day = models.PositiveSmallIntegerField(default=5)
    high_risk_reporters = models.PositiveSmallIntegerField(default=10)
    medium_risk_reporters = models.PositiveSmallIntegerField(default=5)
    updated_at = models.DateTimeField(auto_now=True)

    @classmethod
    def get(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    def __str__(self):
        return "Platform settings"


class AdminAction(models.Model):
    """Audit log: who did what in the admin panel."""
    admin = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name="admin_actions")
    action = models.CharField(max_length=60)
    target = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-id"]

    def __str__(self):
        return f"{self.action} {self.target}"


    # models.py
class AccountDeletionFeedback(models.Model):
    user_id_snapshot = models.IntegerField()
    username = models.CharField(max_length=150)
    reason = models.CharField(max_length=50)
    details = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.username} - {self.reason}"


def story_expiry():
    return timezone.now() + timedelta(hours=24)

class Story(models.Model):
    MEDIA_CHOICES = (("image", "Image"), ("video", "Video"))

    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="stories")
    media = models.FileField(upload_to="stories/")
    media_type = models.CharField(max_length=5, choices=MEDIA_CHOICES)
    caption = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(default=story_expiry, db_index=True)

    class Meta:
        ordering = ["created_at"]

    @property
    def is_active(self):
        return self.expires_at > timezone.now()


class StoryView(models.Model):
    story = models.ForeignKey(Story, on_delete=models.CASCADE, related_name="views")
    viewer = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="story_views")
    viewed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("story", "viewer")