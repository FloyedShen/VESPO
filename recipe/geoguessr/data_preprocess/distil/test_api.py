#!/usr/bin/env python3
"""
Test script to verify Qwen3-VL-235B-Thinking API connectivity.
"""

import sys
import json
import base64
from io import BytesIO

import requests
from PIL import Image


# API Configuration
API_BASE_URL = "http://10.146.229.25:80/v1"
MODEL_NAME = "nginx"


def test_api_health():
    """Test if the API server is responding."""
    print("=" * 80)
    print("Testing API Health")
    print("=" * 80)

    try:
        # Try to get models list
        url = f"{API_BASE_URL}/models"
        print(f"Requesting: {url}")

        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            print("✅ API server is responding")
            try:
                data = response.json()
                print(f"Response: {json.dumps(data, indent=2)}")
            except json.JSONDecodeError:
                print(f"Response (raw): {response.text[:500]}")
            return True
        else:
            print(f"❌ API server returned error: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False

    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection failed: Cannot reach {API_BASE_URL}")
        print(f"Error: {e}")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ Request timeout")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_simple_completion():
    """Test a simple text completion."""
    print("\n" + "=" * 80)
    print("Testing Simple Text Completion")
    print("=" * 80)

    url = f"{API_BASE_URL}/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Say 'Hello, World!' if you can read this."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }

    try:
        print(f"Requesting: {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")

        response = requests.post(
            url,
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ Text completion successful")
            print(f"\nResponse:")
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # Extract and print the content
            try:
                content = data["choices"][0]["message"]["content"]
                print(f"\nExtracted content: {content}")
            except (KeyError, IndexError) as e:
                print(f"⚠️  Could not extract content: {e}")

            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def test_vision_completion():
    """Test a vision completion with a simple test image."""
    print("\n" + "=" * 80)
    print("Testing Vision Completion (with test image)")
    print("=" * 80)

    # Create a simple test image (red square)
    img = Image.new('RGB', (224, 224), color='red')

    # Add some text pattern
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 174, 174], fill='blue')
    draw.rectangle([75, 75, 149, 149], fill='green')

    # Encode to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    print(f"Test image created: 224x224 RGB with colored squares")

    url = f"{API_BASE_URL}/chat/completions"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful vision assistant."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What colors do you see in this image? Please list them."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }

    try:
        print(f"Requesting: {url}")
        print(f"Payload size: {len(json.dumps(payload))} bytes")

        response = requests.post(
            url,
            json=payload,
            timeout=60,
            headers={"Content-Type": "application/json"}
        )

        print(f"\nStatus Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print("✅ Vision completion successful")
            print(f"\nResponse:")
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # Extract and print the content
            try:
                content = data["choices"][0]["message"]["content"]
                print(f"\nExtracted content: {content}")
            except (KeyError, IndexError) as e:
                print(f"⚠️  Could not extract content: {e}")

            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Qwen3-VL-235B-Thinking API Test" + " " * 26 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    results = {
        "API Health": test_api_health(),
        "Text Completion": test_simple_completion(),
        "Vision Completion": test_vision_completion()
    }

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:25s} {status}")

    print("=" * 80)

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! API is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
