import threading

# Unified global reentrant lock across all endpoints (tiles, thumbnails, patches)
# to guarantee 100% thread-safety for OpenSlide native C library on Windows without self-deadlock.
OPENSLIDE_GLOBAL_LOCK = threading.RLock()
