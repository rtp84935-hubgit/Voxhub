from datetime import timezone,datetime

from django.db import models

# Create your models here.
from django.db import models
from django.contrib.auth.models import User


    
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


class User_posts(models.Model):
    USER=models.ForeignKey(UserProfile,on_delete=models.CASCADE)
    photo=models.FileField()
    caption=models.CharField(max_length=100)

class Comments(models.Model):
    comment=models.CharField(max_length=100)
    USER=models.ForeignKey(UserProfile,on_delete=models.CASCADE)
    POST=models.ForeignKey(User_posts,on_delete=models.CASCADE)
    date=models.DateTimeField(default=datetime.now)

class Likes(models.Model):
    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    POST = models.ForeignKey(User_posts, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('USER', 'POST')

class Follow(models.Model):
    FOLLOWER = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='following')
    FOLLOWING = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name='followers')

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
    USER = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="reports_made")
    VICTIM = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="reports_received")
    reports = models.CharField(max_length=100)
    status = models.CharField(max_length=10, default="pending")
    date = models.DateTimeField(auto_now_add=True)