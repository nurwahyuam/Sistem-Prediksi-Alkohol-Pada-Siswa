"""
auth.py
=======
Modul autentikasi Pakar menggunakan Flask Session.

Kredensial pakar dibaca dari config/expert.json (single-expert account).

Role:
    Guest  -> tidak ada session 'role'
    Pakar  -> session['role'] == 'pakar'
"""

import json
import os
from functools import wraps
from flask import session, redirect, url_for, flash, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERT_CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'expert.json')


def load_expert_config():
    """Load kredensial pakar dari config/expert.json."""
    with open(EXPERT_CONFIG_PATH, 'r') as f:
        return json.load(f)


def verify_login(username, password):
    """
    Cek username & password terhadap config/expert.json.

    Returns:
        nama pakar (str) jika valid, None jika tidak valid.
    """
    config = load_expert_config()
    if username == config.get('username') and password == config.get('password'):
        return config.get('name', config.get('username'))
    return None


def is_logged_in():
    """Cek apakah sesi pakar sedang aktif."""
    return session.get('role') == 'pakar'


def login_required(view_func):
    """Decorator: wajib login sebagai Pakar untuk mengakses route."""
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not is_logged_in():
            flash('Silakan login sebagai Pakar untuk mengakses halaman ini.', 'warning')
            return redirect(url_for('login', next=request.path))
        return view_func(*args, **kwargs)
    return wrapped
