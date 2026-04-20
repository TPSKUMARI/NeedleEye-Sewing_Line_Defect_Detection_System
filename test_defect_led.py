import serial
import time

# ── CONFIGURATION ──────────────────────────────────
# Change 'COM3' to the port shown in your Arduino IDE
SERIAL_PORT = 'COM5' 
BAUD_RATE = 115200

try:
    # Open the serial port
    esp32 = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT}")
    
    # Wait for ESP32 to reboot after connection
    time.sleep(2) 

    while True:
        user_input = input("Press Enter to trigger LED (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break

        # 1. Send a unique name as a string
        esp32.write(b"Test_Defect\n")
        
        # 2. Send the 0x01 byte to trigger the 2s blink
        esp32.write(b'\x01')
        
        print("Sent: 'Test_Defect' + 0x01 Trigger")

    esp32.close()
    print("Connection closed.")

except Exception as e:
    print(f"Error: {e}")