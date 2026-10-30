from django.utils import timezone

from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User
import os

from django.conf import settings
from django.core.files.storage import FileSystemStorage

PRIVATE_POST_STORAGE = FileSystemStorage(
    location=os.path.join(settings.BASE_DIR, "private_post_originals"),
    base_url=None,
)


class UserProfile(models.Model):
    LOGIN=models.OneToOneField(User, on_delete=models.CASCADE, related_name="user_profile")

    name = models.CharField(max_length=100)

    email = models.EmailField(unique=True)
    
    bio = models.TextField(blank=True, default="")

    phone_number = models.CharField(
        max_length=15,
        unique=True
    )

    username = models.CharField(
        max_length=50,
        unique=True
    )

    password = models.CharField(
        max_length=128
    )

    profile_image = models.FileField(upload_to="profile_images/")
    original_user = models.FileField(upload_to="original_images/")

    # Voice biometric profile
    voice_embedding = models.JSONField(
        null=True,
        blank=True
    )

    voice_sample_count = models.PositiveIntegerField(
        default=0
    )
    
    created_at = models.DateField(auto_now_add=True)

class VoiceProfile(models.Model):
    USER=models.ForeignKey(UserProfile, on_delete=models.CASCADE)

    embedding = models.JSONField()

    sample_count = models.PositiveIntegerField(
        default=10
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    updated_at = models.DateTimeField(
        auto_now=True
    )

    def __str__(self):
        return self.username

VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm", ".m4v")
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

class Comments(models.Model):
    comment = models.CharField(max_length=100)
    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    POST = models.ForeignKey(User_posts, on_delete=models.CASCADE)
    date = models.DateTimeField(default=timezone.now)

class Likes(models.Model):
    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    POST = models.ForeignKey(User_posts, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('USER', 'POST')

class Follow(models.Model):
    FOLLOWER = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='following')
    FOLLOWING = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='followers')
    status = models.CharField(max_length=10, default='accepted')  # pending / accepted / declined

    class Meta:
        unique_together = ('FOLLOWER', 'FOLLOWING')

class ChatMessage(models.Model):
    sender = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name="sent_messages"
    )

    receiver = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name="received_messages"
    )

    message = models.TextField(max_length=2000)

    timestamp = models.DateTimeField(auto_now_add=True)

    is_read = models.BooleanField(default=False)

    deleted_for = models.ManyToManyField(UserProfile, blank=True, related_name='hidden_messages')

    class Meta:
        ordering = ["timestamp", "id"]

    def __str__(self):
        return f"{self.sender.username} -> {self.receiver.username}"



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
    admin = models.OneToOneField(
        User, on_delete=models.CASCADE, related_name="notification_pref"
    )
    new_user = models.BooleanField(default=True)
    report = models.BooleanField(default=True)
    security = models.BooleanField(default=True)
 
    def __str__(self):
        return f"Notification preferences for {self.admin.username}"

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
    admin_seen = models.BooleanField(default=False)   # new reports start unseen by admin
    user_seen = models.BooleanField(default=True)     # set to False when admin updates it

    class Meta:
        ordering = ["-date"]

    def __str__(self):
        return f"Report by {self.USER.username} ({self.status})"
    

class Saved_Posts(models.Model):
    USER=models.ForeignKey(UserProfile,on_delete=models.CASCADE)
    Post=models.ForeignKey(User_posts,on_delete=models.CASCADE)
    date=models.DateTimeField(auto_now_add=True)


class UserNotification(models.Model):
    recipient = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name='notifications'
    )

    sender = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name='sent_notifications'
    )

    post = models.ForeignKey(
        User_posts,
        on_delete=models.CASCADE,
        related_name='notifications'
    )

    message = models.CharField(max_length=255)

    is_read = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    status = models.CharField(
    max_length=10,
    choices=[
        ("pending", "Pending"),
        ("accepted", "Accepted"),
        ("rejected", "Rejected"),
    ],
    default="pending",
    )

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.sender} -> {self.recipient}: {self.message}"

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