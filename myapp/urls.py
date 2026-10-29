from django.urls import path
from . import views

urlpatterns = [

    # =====================================================
    # COMMON - registration & authentication (user side)
    # =====================================================
    path("registration/", views.registration, name="registration"),
    path("register/", views.register, name="register"),
    path("detect-face/", views.detect_face, name="detect_face"),
    path("login/", views.login_view, name="login"),
    path("logout_view/", views.logout_view, name="logout"),
    path("community_guidelines/", views.community_guidelines, name="community_guidelines"),
    path("about_us/", views.about_us, name="about_us"),
    path("submit_complaint/", views.submit_complaint, name="submit_complaint"),
    path("admin_complaints/", views.admin_complaints, name="admin_complaints"),
    path("review_complaints/<int:id>/", views.review_complaints, name="review_complaints"),
    path("update_complaints/<int:id>/", views.update_complaints, name="update_complaints"),
    path("user_complaints/", views.user_complaints, name="user_complaints"),

    # =====================================================
    # USER - feed, posts, likes, comments, saved
    # =====================================================
    path("user_home/", views.user_home, name="user_home"),
    path("create_post/", views.create_post, name="create_post"),
    path("delete_post/", views.delete_post, name="delete_post"),
    path("toggle_like/", views.toggle_like, name="toggle_like"),
    path("liked_posts/", views.liked_posts, name="liked_posts"),
    path("add_comment/", views.add_comment, name="add_comment"),
    path("delete_comment/", views.delete_comment, name="delete_comment"),
    path("commented_posts/", views.commented_posts, name="commented_posts"),
    path("toggle_save/", views.toggle_save, name="toggle_save"),
    path("saved_posts/", views.saved_posts, name="saved_posts"),

    # =====================================================
    # USER - explore, profiles, follow, reports
    # =====================================================
    path("explore/", views.explore, name="explore"),
    path("profile/<int:user_id>/", views.view_profile, name="view_profile"),
    path("edit_profile/", views.edit_profile, name="edit_profile"),
    path("follow/<int:user_id>/", views.toggle_follow, name="toggle_follow"),
    path("report_user/<int:user_id>/", views.report_user, name="report_user"),

    # =====================================================
    # USER - chat
    # =====================================================
    path("chat/", views.chat_inbox, name="chat_inbox"),
    path("chat/<int:user_id>/", views.chat_room, name="chat_room"),
    path("chat/<int:user_id>/send/", views.send_message, name="send_message"),
    path("chat/<int:user_id>/messages/", views.chat_messages, name="chat_messages"),
    path("delete_message/", views.delete_message, name="delete_message"),

    # =====================================================
    # USER - notifications
    # =====================================================
    path("user_notifications/", views.user_notifications, name="user_notifications"),
    path("notifications/respond/<int:notification_id>/", views.respond_to_notification, name="respond_to_notification"),
    path("accept_follow_request/<id>", views.accept_follow_request, name="accept_follow_request"),
    path("decline_follow_request/<id>/", views.decline_follow_request, name="decline_follow_request"),
    path("delete_face_notification/<id>", views.delete_face_notification, name="delete_face_notification"),
    path("delete_report_notification/<int:id>/", views.delete_report_notification, name="delete_report_notification"),

    # =====================================================
    # USER - account settings
    # =====================================================
    path("user_settings/", views.settings_view, name="settings"),
    path("change_password/", views.change_password, name="change_password"),

    # =====================================================
    # ADMIN - authentication & dashboard
    # =====================================================
    path("admin_login/", views.admin_login, name="admin_login"),
    path("admin_login_post/", views.admin_login_post, name="admin_login_post"),
    path("admin_home/", views.admin_home, name="admin_home"),

    # =====================================================
    # ADMIN - user management
    # =====================================================
    path("admin_users/", views.admin_users, name="admin_users"),
    path("admin_banned_users/", views.admin_banned_users, name="admin_banned_users"),
    path("admin_ban_user/<int:id>", views.admin_ban_user, name="admin_ban_user"),
    path("admin_unban_user/<int:id>", views.admin_unban_user, name="admin_unban_user"),
    path("view_userprofile/<int:id>/", views.view_userprofile, name="view_userprofile"),

    # =====================================================
    # ADMIN - reports
    # =====================================================
    path("admin_reports/", views.admin_reports, name="admin_reports"),
    path("review_report/<int:id>/", views.review_report, name="review_report"),
    path("resolve_report/<int:id>/", views.resolve_report, name="resolve_report"),
    path("dismiss_report/<int:id>/", views.dismiss_report, name="dismiss_report"),
    path("admin_close_reports/<int:id>/", views.admin_close_reports, name="admin_close_reports"),

    # =====================================================
    # ADMIN - settings, profile, password
    # =====================================================
    path("admin_settings/", views.admin_settings, name="admin_settings"),
    path("admin_update_profile/", views.admin_update_profile, name="admin_update_profile"),
    path("admin_changepassword/", views.admin_changepassword, name="admin_changepassword"),
    path("admin_save_notification_pref/", views.admin_save_notification_pref, name="admin_save_notification_pref"),

    # =====================================================
    # ADMIN - notification bell
    # =====================================================
    path("admin_notifications/", views.admin_notifications, name="admin_notifications"),
    path("admin_notifications_read/", views.admin_notifications_read, name="admin_notifications_read"),

]