# VoxHub

**Connect through your voice.**

VoxHub is a social media platform built with Django. Members register with a face photo and a voice profile, sign in with a password **and** their voice, share photos and videos, chat privately, and follow each other. Because faces are recognised in uploaded photos, VoxHub protects privacy: a registered person's face stays blurred in someone else's post until they approve it. A full admin panel handles users, reports, complaints and platform rules.

---

## Table of contents

1. [Features](#features)
2. [Tech stack](#tech-stack)
3. [Project structure](#project-structure)
4. [Getting started](#getting-started)
5. [Configuration](#configuration)
6. [Creating the first admin](#creating-the-first-admin)
7. [How the main features work](#how-the-main-features-work)
8. [Admin guide](#admin-guide)
9. [Main URLs](#main-urls)
10. [Security notes](#security-notes)
11. [Troubleshooting](#troubleshooting)
12. [Author](#author)

---

## Features

### For members

| Area | What you can do |
|---|---|
| **Registration** | Create an account with a profile picture, a live camera face capture (checked for size, lighting, sharpness and head angle on the server) and 10 recorded voice samples. |
| **Sign in** | Password **plus** voice verification. Banned users are told their account is banned. |
| **Feed** | Post photos or videos with a caption, like, comment, save, share a link, and delete or edit your own posts. Posts load 8 at a time with "Load more". |
| **Face privacy** | Photos are scanned for faces. A registered person in someone else's photo gets a request and stays blurred until they accept. |
| **Comments** | Every comment is checked by an AI toxicity model before it is posted. |
| **Profiles** | Bio, followers, following, a post grid with a built-in post viewer, follow requests, and a "Follows you" tag. |
| **Explore** | Search people and posts, and filter by photos or videos. |
| **Messages** | Private chat with a conversation list, unread counts, date separators, delete-for-me, and live updates. |
| **Chat settings** | 8 chat themes, text size, Enter-to-send, message times, compact mode and list previews on or off. |
| **Notifications** | Face-visibility requests, follow requests and updates on the reports you sent. |
| **Reports and complaints** | Report another user, or send a general complaint to the admins and follow its status. |
| **Account** | Change password and the theme (dark, light or system). |

### For admins

| Area | What you can do |
|---|---|
| **Dashboard** | Totals, 7-day charts of new users and posts, engagement figures, report status breakdown, latest reports and complaints, most-reported accounts, and recent activity. |
| **Users** | View all users and banned users, open any profile, ban and unban. Admin and staff accounts cannot be banned from the interface. |
| **User reports** | A moderation queue with risk levels based on how many *different* people reported an account, a reports viewer, and one-click dismiss, resolve or ban. |
| **Complaints** | A queue with tabs, search, "waiting" timers, a review page with the sender's other complaints, and reopen. |
| **Notification bell** | Live alerts for new users, reports and security events, filtered by your preferences. |
| **Settings** | Profile, password with strength meter, sign out other devices, platform rules (registration on or off, AI moderation and its strictness, daily limits, risk thresholds), CSV export of users and reports, and an audit log of admin actions. |

---

## Tech stack

- **Backend:** Python, Django
- **Database:** configured in `settings.py` (SQLite for development, MySQL or another database for production)
- **Face processing:** OpenCV, `face_recognition` (dlib), NumPy
- **Voice biometrics:** the `myapp/voice/enrollment.py` module (`create_voice_profile`, `verify_voice`)
- **AI moderation:** Hugging Face `transformers` with `unitary/toxic-bert`
- **Frontend:** Django templates, plain CSS and vanilla JavaScript (no frontend build step)

---

## Project structure

```
project/
├── manage.py
├── <project>/                  # settings.py, urls.py, wsgi.py
├── private_post_originals/     # original photos, never served publicly (created automatically)
└── myapp/
    ├── models.py
    ├── views.py
    ├── urls.py
    ├── context_processors.py   # sidebar badges (user_sidebar)
    ├── voice/
    │   └── enrollment.py       # voice enrollment and verification
    └── templates/
        ├── user_base.html            # layout for members
        ├── admin_base.html           # layout for admins
        ├── registration.html, login.html
        ├── user_home.html, view_profile.html, edit_profile.html, explore.html
        ├── notification.html, user_complaints.html, user_settings.html
        ├── chat_inbox.html, chat_room.html, chat_settings.html
        ├── chat_css.html, chat_list_pane.html, chat_list_items.html, chat_list_js.html
        ├── admin_login.html, admin_home.html, admin_bell.html
        ├── admin_users.html, admin_banned_users.html, view_userprofile.html
        ├── admin_reports.html, review_report.html
        ├── admin_complaints.html, review_complaints.html
        └── admin_settings.html
```

### Main models

`UserProfile`, `VoiceProfile`, `User_posts`, `PostFace`, `Comments`, `Likes`, `Saved_Posts`, `Follow`, `ChatMessage`, `ChatSettings`, `UserNotification`, `Reports`, `Complaints`, `AdminNotification`, `AdminNotificationPreference`, `PlatformSettings`, `AdminAction`.

---

## Getting started

### Requirements

- Python 3.10 or newer
- `pip` and a virtual environment
- **For face recognition:** CMake and a C++ compiler, because `dlib` is built when you install `face_recognition`
  - Windows: Visual Studio Build Tools (C++ workload) and CMake
  - macOS: `brew install cmake`
  - Ubuntu or Debian: `sudo apt install build-essential cmake`
- About 1 GB of free disk space (the toxicity model is downloaded on first run)

### Installation

```bash
# 1. Get the code
git clone <your-repository-url>
cd <project-folder>

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install the dependencies
pip install django pillow numpy opencv-python face_recognition transformers torch
# plus any libraries used by myapp/voice/enrollment.py

# 4. Create the database tables
python manage.py makemigrations
python manage.py migrate

# 5. Start the server
python manage.py runserver
```

Open **http://127.0.0.1:8000/myapp/registration/** to create the first member account.

> The first start can take a while. Importing `views.py` loads the AI moderation model, and the first run downloads it.

---

## Configuration

Check these in `settings.py`:

```python
INSTALLED_APPS = [
    ...
    "django.contrib.sessions",     # needed for "Sign out other devices"
    "myapp",
]

TEMPLATES = [{
    ...
    "OPTIONS": {
        "context_processors": [
            "django.template.context_processors.request",
            "django.contrib.auth.context_processors.auth",
            "django.contrib.messages.context_processors.messages",
            "myapp.context_processors.user_sidebar",     # sidebar badges
        ],
    },
}]

# Uploaded photos, videos and profile pictures
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"
```

And in the project's `urls.py`, serve media while developing and mount the app under `myapp/`:

```python
from django.conf import settings
from django.conf.urls.static import static
from django.urls import include, path

urlpatterns = [
    path("myapp/", include("myapp.urls")),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
```

### Privacy of original photos

Original uploads are saved in `private_post_originals/` (set in `models.py`). That folder must **never** be served by the web server. Only the blurred, safe copy is public.

### Platform rules

Most rules are changed in the running app, under **Admin → Settings → Platform rules**, with no code edit: registration on or off, AI moderation and strictness, daily limits for reports and complaints, and risk levels.

---

## Creating the first admin

Admin access is based on membership of a group named **`admin`**, checked by the `admin_required` decorator.

```bash
python manage.py createsuperuser
python manage.py shell
```

```python
from django.contrib.auth.models import User, Group

group, _ = Group.objects.get_or_create(name="admin")
user = User.objects.get(username="your-admin-username")
user.groups.add(group)
```

Then sign in at **http://127.0.0.1:8000/myapp/admin_login/**.

An admin account has no VoxHub member profile, so it cannot use the member login.

---

## How the main features work

### Registration
The camera image is checked on the server for exactly one face, distance, centring, head tilt, lighting and sharpness. 10 voice samples are turned into a voice profile. Everything is saved in one database transaction, so a failure leaves no half-created account.

### Sign in
1. Username and password are checked by Django.
2. The member records their voice, which is compared with their stored profile.
3. Only if both pass is a session created.

### Face privacy in posts
1. A photo upload is scanned for faces, and each face is compared with the registered members' face photos.
2. The original is saved privately. A public copy is generated with every matched person's face blurred, except the uploader's.
3. Each matched person gets a notification. **Accept** shows their face. **Reject** keeps it blurred. The public image is rebuilt from the private original after each choice.

### Reports and risk
A report is one user reporting another. The risk level on the admin Reports page counts the number of **different** people with open reports, so one person pressing Report many times cannot push an account up. Banning an account also closes its open reports.

### Comment moderation
Each comment is scored by `unitary/toxic-bert` before it is saved. Comments above the configured threshold are rejected. An admin can change the threshold or switch moderation off.

### Chat
Messages refresh every few seconds, loading only new ones. Deleting a message hides it for you only. Settings such as theme and text size apply to all your conversations.

---

## Admin guide

| Task | Where |
|---|---|
| See what needs attention | **Dashboard** |
| Handle a reported account | **Reports**: open its reports, then ban, or dismiss or resolve them |
| Answer a general problem | **Complaints**: open it, then resolve or dismiss (you can reopen it) |
| Ban or unban a user | **Users** or **Reports** |
| Change platform rules | **Settings → Platform rules** |
| Choose which alerts you get | **Settings → Notifications** |
| Download data | **Settings → Data export** (CSV) |
| Review what admins did | **Settings → Recent admin activity** |

---

## Main URLs

All routes start with `/myapp/`.

| Page | URL |
|---|---|
| Register | `registration/` |
| Member login | `login/` |
| Home feed | `user_home/` |
| Explore | `explore/` |
| Profile | `profile/<id>/` |
| Messages | `chat/` |
| Chat settings | `chat/settings/` |
| Notifications | `user_notifications/` |
| Report a problem | `user_complaints/` |
| Admin login | `admin_login/` |
| Dashboard | `admin_home/` |
| Reports | `admin_reports/` |
| Complaints | `admin_complaints/` |
| Admin settings | `admin_settings/` |

---

## Security notes

- Keep `DEBUG = False`, a private `SECRET_KEY` and a proper `ALLOWED_HOSTS` in production, and serve the site over HTTPS.
- Face and voice data are biometric. Store them securely, back them up carefully and tell your users how they are used.
- `private_post_originals/` and uploaded voice data must not be reachable from the web.
- CSV exports contain emails and phone numbers. Admins should handle them as private files. Every export is written to the audit log.
- Admin actions that change data (ban, unban, resolve, dismiss, settings) require a POST request and the `admin` group.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `dlib` or `face_recognition` fails to install | Install CMake and a C++ compiler first (see Requirements), then try again. |
| `TemplateDoesNotExist: chat_css.html` | The chat partial templates must sit in the same templates folder as `chat_inbox.html`. |
| `NoReverseMatch` | `urls.py` is out of date. Compare it with the route list in this project. |
| `no such table` or `Unknown column` errors | Run `python manage.py makemigrations` and `python manage.py migrate`. |
| Sidebar badges missing | Add `myapp.context_processors.user_sidebar` and `django.template.context_processors.request` to the template context processors. |
| Admin pages send you back to the login | The account must be in the `admin` group. |
| Server start is slow | The AI model is loaded at start-up. This is normal. |
| Uploaded images do not appear | Check `MEDIA_URL`, `MEDIA_ROOT` and the `static(...)` line in `urls.py`. |

---

## Author

Built by **Rahul T P**, Python full stack developer, Kozhikode, Kerala.
Portfolio: https://rahul-khaki.vercel.app/
