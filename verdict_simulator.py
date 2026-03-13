import joblib
import pandas as pd

# 1. Load the "Brain"
model = joblib.load('vandoot_judge.pkl')

def test_vandoot():
    print("\n--- VanDoot Logic Simulator ---")
    print("Enter the values to see the Verdict:")
    
    # Manually input sensor data
    v_conf = float(input("ESP32 Vision Confidence (0-100): "))
    e_class = int(input("ESP32 Class ID (1=Fire, 2=Human, 3=Animal): "))
    audio = float(input("Audio dB (30-100): "))
    pir = int(input("PIR Motion (0 or 1): "))
    thermal = float(input("Thermal Delta Celsius (0-60): "))
    smoke = int(input("Smoke PPM (200-2000): "))

    # 2. Prepare the data for the model
    # Match the EXACT column names from your CSV
    input_data = pd.DataFrame([[v_conf, e_class, audio, pir, thermal, smoke]], 
                              columns=['Vision_Conf', 'ESP32_Class', 'Audio_db', 'PIR_Trig', 'Thermal_Delta', 'Smoke_PPM'])

    # 3. Get the Verdict
    prediction = model.predict(input_data)[0]
    
    # 4. Map the Label to a readable name
    verdicts = {0: "SAFE (No Action)", 1: "FIRE DETECTED!", 2: "POACHER/INTRUDER DETECTED!"}
    
    print(f"\n>>> FINAL VERDICT: {verdicts[prediction]}")

# Run the simulation
while True:
    test_vandoot()
    if input("\nTest another? (y/n): ") != 'y': break