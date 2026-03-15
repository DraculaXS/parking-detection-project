# test_integration.py
import os
import sys

def run_all_tests():
    """Run all tests and report results"""
    print("="*50)
    print("Running integration tests")
    print("="*50)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Check required directories
    print("\n1. Checking directory structure...")
    required_dirs = ['train_data/train', 'train_data/test']
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f" {dir_path} exists")
            tests_passed += 1
        else:
            print(f" {dir_path} missing")
            tests_failed += 1
    
    # Test 2: Check model files
    print("\n2. Checking model files...")
    model_files = ['VGG16_model.keras', 'cnn_model.keras']
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f" {model_file} exists ({size_mb:.1f} MB)")
            tests_passed += 1
        else:
            print(f" {model_file} not found (run training first)")
            tests_failed += 1
    
    # Test 3: Check test images
    print("\n3. Checking test images...")
    test_images_dir = 'test_images'
    if os.path.exists(test_images_dir):
        jpg_files = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]
        if jpg_files:
            print(f" Found {len(jpg_files)} test images")
            tests_passed += 1
        else:
            print(f" No .jpg files in {test_images_dir}")
            tests_failed += 1
    else:
        print(f" {test_images_dir} directory missing")
        tests_failed += 1
    
    # Summary
    print("\n" + "="*50)
    print(f"Tests passed: {tests_passed}, Tests failed: {tests_failed}")
    print("="*50)
    
    return tests_failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)