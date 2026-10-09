from django.urls import path
from . import views

urlpatterns = [
    path("registration/", views.registration, name="registration"),
    path("register/", views.register, name="register"),
    path("detect-face/", views.detect_face, name="detect_face"),
    path("login/", views.login_view, name="login"),
    path("user_home/", views.user_home, name="user_home"),
    path('explore/',views.explore,name="explore"),
    path("logout_view/",views.logout_view,name="logout"),
    path('create_post/',views.create_post,name="create_post"),
    path('add_comment/',views.add_comment,name="add_comment"),
    path('toggle_like/', views.toggle_like, name='toggle_like'),
    path('toggle_follow/<int:uid>/', views.toggle_follow, name='toggle_follow'),
]