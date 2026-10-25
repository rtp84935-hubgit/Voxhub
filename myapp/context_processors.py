from .models import UserProfile, Follow, ChatMessage ,Reports

def user_sidebar(request):
    if not request.user.is_authenticated:
        return {}
    me = UserProfile.objects.filter(LOGIN=request.user).first()

    if not me:
        return {}
    return {
        'sidebar_user': me,
        'notification_count': Follow.objects.filter(FOLLOWING=me, status='pending').count(),
        'reports_count': Reports.objects.filter(user_seen=False).count(),
        'unread_count': ChatMessage.objects.filter(receiver=me, is_read=False).exclude(deleted_for=me).count(),
    }