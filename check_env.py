import sys

def check_package(name, import_name=None, version=None):
    if import_name is None:
        import_name = name
    try:
        module = __import__(import_name)
        installed_version = getattr(module, '__version__', 'unknown')
        if version:
            if installed_version == version:
                print(f"[OK] {name}: {installed_version}")
            else:
                print(f"[WARNING] {name}: {installed_version} (Expected {version})")
        else:
            print(f"[OK] {name}: {installed_version}")
    except ImportError:
        print(f"[MISSING] {name}")
    except Exception as e:
        print(f"[ERROR] {name}: {e}")

print(f"Python: {sys.version.split()[0]}")

check_package('torch', version='1.2.0')
check_package('torchvision', version='0.4.0')
check_package('numpy')
check_package('PIL', import_name='PIL') # Pillow imports as PIL
check_package('cv2', import_name='cv2') # opencv-python imports as cv2
check_package('scipy', version='1.5.2')
check_package('sklearn', version='0.24.1')
