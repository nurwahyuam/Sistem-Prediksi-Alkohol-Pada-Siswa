#!/usr/bin/env python3
"""
run.py — Entry point untuk menjalankan aplikasi CBR KMeans
Usage: python run.py
"""
from app import app, initialize_system

if __name__ == '__main__':
    print("=" * 55)
    print("  CBR KMeans — Student Alcohol Risk Classification")
    print("=" * 55)
    print("  Inisialisasi sistem...")
    initialize_system()
    print("  ✓ Dataset loaded")
    print("  ✓ Model trained")
    print("  ✓ Charts generated")
    print("-" * 55)
    print("  Akses aplikasi di: http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=False, port=5000, host='0.0.0.0')
