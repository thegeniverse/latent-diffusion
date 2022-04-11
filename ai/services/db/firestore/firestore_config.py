import os

FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS", None)
if FIREBASE_CREDENTIALS is None:
    print("set `FIREBSE_CREDENTIALS` to your path of credentials json file.")

PROJECT_ID = "geni-f63a8"