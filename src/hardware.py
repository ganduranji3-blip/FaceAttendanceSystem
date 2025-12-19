import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD
import time

# --- Configuration ---
BUZZER_PIN = 23  # Change this to your Buzzer GPIO pin
LCD_ADDRESS = 0x27  # Default I2C address for most 16x2 LCDs. Check with 'i2cdetect -y 1'
LCD_WIDTH = 16

# --- Setup ---
try:
    lcd = CharLCD(i2c_expander='PCF8574', address=LCD_ADDRESS, port=1, cols=16, rows=2, dotsize=8)
    lcd.clear()
except Exception as e:
    print(f"[WARN] LCD not detected or error: {e}")
    lcd = None

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW) # Turn off initially

def buzz_success():
    """Beeps the buzzer once for success (Attendance Marked)"""
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

def buzz_error():
    """Beeps twice for error or already marked"""
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
    time.sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.1)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

def display_message(line1, line2=""):
    """Displays text on the 16x2 LCD"""
    if lcd:
        try:
            lcd.clear()
            lcd.write_string(line1[:16]) # Limit to 16 chars
            lcd.cursor_pos = (1, 0) # Move to second line
            lcd.write_string(line2[:16])
        except Exception as e:
            print(f"[LCD Error] {e}")
    else:
        # Fallback to console if LCD is not connected
        print(f"\n[LCD DISPLAY] Line 1: {line1} | Line 2: {line2}")

def cleanup():
    if lcd:
        lcd.clear()
    GPIO.cleanup()
