# myapp/context_processors.py
#
# Your settings.py already points to "myapp.context_processors.user_sidebar",
# so the function keeps that name. Replace your existing file with this one
# (it supplies everything the new user_base.html needs).
#
# settings.py -> TEMPLATES -> OPTIONS -> context_processors must contain:
#     "myapp.context_processors.user_sidebar",
# and, for the active-page highlighting:
#     "django.template.context_processors.request",

from .models import ChatMessage, Follow, UserNotification, UserProfile


def user_sidebar(request):
    user = getattr(request, "user", None)

    if user is None or not user.is_authenticated:
        return {}

    me = UserProfile.objects.filter(LOGIN=user).first()

    if me is None:          # e.g. admin accounts have no VoxHub profile
        return {}

    unread_messages = (
        ChatMessage.objects
        .filter(receiver=me, is_read=False)
        .exclude(deleted_for=me)
        .count()
    )

    # Things waiting for the user's answer: face requests + follow requests
    pending_faces = UserNotification.objects.filter(recipient=me, status="pending").count()
    pending_follows = Follow.objects.filter(FOLLOWING=me, status="pending").count()

    return {
        "nav_me": me,
        "sidebar_user": me,              # kept for templates that already use it
        "unread_count": unread_messages,
        "notification_count": pending_faces + pending_follows,
    }