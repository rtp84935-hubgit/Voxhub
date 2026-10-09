from django.urls import path
from . import views

urlpatterns = [
    path("registration/", views.registration, name="registration"),
    path("register/", views.register, name="register"),
    path("detect-face/", views.detect_face, name="detect_face"),
    path("login/", views.login_view, name="login"),
    path("user_home/", views.user_home, name="user_home"),
    path('explore/',views.explore,name="explore"),
    path("logout_view/",views.logout_view,name="logout")
]