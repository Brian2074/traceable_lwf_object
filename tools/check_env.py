import sys

def check_python_version():
    """Ensure Python is 3.9 or higher for Flower and YOLO compatibility."""
    print(f"Current Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("[ERROR] ❌ Python 3.9+ is strictly required for this project.")
        print("        Flower >= 1.12.0 drops support for Python 3.8.")
        print("        Please upgrade to Python 3.10 or 3.11.")
        sys.exit(1)
    else:
        print("[OK] ✅ Python version is strictly >= 3.9")

def check_package(name, expected_version=None):
    try:
        module = __import__(name)
        installed = getattr(module, '__version__', 'unknown')
        if expected_version:
            print(f"[*] {name}: {installed} (Expected >= {expected_version})")
        else:
            print(f"[*] {name}: {installed}")
    except ImportError:
        print(f"[ERROR] ❌ Missing package: {name}")

if __name__ == "__main__":
    print("-" * 40)
    print("Environment Verification")
    print("-" * 40)
    
    check_python_version()
    
    print("\nChecking Core Packages:")
    check_package('torch', '2.0.0')
    check_package('torchvision')
    check_package('ultralytics', '8.3.0')
    check_package('flwr', '1.12.0')
    
    print("-" * 40)
