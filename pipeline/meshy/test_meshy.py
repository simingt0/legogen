
import asyncio
import os
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from pipeline.meshy.api import generate_3d_model

async def main():
    """
    Main function to test the Meshy API integration.
    """
    # Set the MESHY_MODE environment variable to 'cache' to generate and cache the model
    # If set to 'test', it will only use existing cache entries.
    # If set to 'prod', it will always call the API without caching.
    os.environ["MESHY_MODE"] = "cache"

    # Ensure the output directory exists
    output_dir = project_root / "temp_models"
    output_dir.mkdir(exist_ok=True)

    test_description = "a small, simple, blocky house"
    print(f"--- Running Meshy Test: '{test_description}' ---")

    try:
        # To run this test, you must have your MESHY_API_KEY set in a .env file
        # in the root of the `legogen` project, like this:
        # MESHY_API_KEY="your_api_key_here"
        print("\nReminder: Make sure your MESHY_API_KEY is set in the .env file.")

        obj_path = await generate_3d_model(
            description=test_description,
            output_dir=str(output_dir),
            timeout=600  # Increased timeout for potentially slow generation
        )
        print(f"\n[SUCCESS] Model generation complete!")
        print(f"OBJ file saved to: {obj_path}")

        # Verify the file was created
        if Path(obj_path).exists():
            print("File verification successful.")
        else:
            print("[ERROR] Output file was not created.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during the test: {e}")
        print("Please check your API key, network connection, and input parameters.")

if __name__ == "__main__":
    asyncio.run(main())
