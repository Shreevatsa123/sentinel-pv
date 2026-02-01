import json

# --- CONFIGURATION ---
# 1. Put this script in the same folder as your .ipynb file
# 2. Change this name to match your notebook's filename exactly:
notebook_filename = "sentinel_pv_v3.ipynb" 

# --- THE SURGERY ---
try:
    print(f"📂 Opening {notebook_filename}...")
    with open(notebook_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check if the corrupt 'widgets' block exists
    if 'metadata' in data and 'widgets' in data['metadata']:
        print("⚠️ Found corrupt 'widgets' data. Removing it now...")
        del data['metadata']['widgets']
        
        # Save the fixed version
        new_filename = "FIXED_" + notebook_filename
        with open(new_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
            
        print(f"✅ SUCCESS! Created '{new_filename}'")
        print("🚀 Upload the 'FIXED' file to GitHub, and the error will be gone.")
        
    else:
        print("ℹ️ No 'widgets' block found. The file syntax looks okay.")
        print("If GitHub is still failing, the issue might be in a specific cell output.")

except FileNotFoundError:
    print(f"❌ Error: Could not find file '{notebook_filename}'.")
    print("Make sure this script is in the SAME folder as your notebook.")
except Exception as e:
    print(f"❌ An error occurred: {e}")