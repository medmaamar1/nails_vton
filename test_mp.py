import sys
try:
    import mediapipe as mp
    try:
        print("mediapipe version:", mp.__version__)
    except: pass
    import mediapipe.python.solutions.hands as mp_hands
    print("SUCCESS: mp_hands loaded!")
except Exception as e:
    import traceback
    traceback.print_exc()
