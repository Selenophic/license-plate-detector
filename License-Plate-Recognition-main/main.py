from PIL import Image
import cv2
import numpy as np
import pytesseract
import sqlite3

# Path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load the Haar Cascade classifier for license plates
cascade = cv2.CascadeClassifier(r"C:\Users\reddy\OneDrive\Desktop\License-Plate-Recognition-main\haarcascade_russian_plate_number.xml")

# States dictionary
states = {
    "AN": "Andaman and Nicobar", 
    "AP": "Andhra Pradesh", 
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar", 
    "CH": "Chandigarh", 
    "DN": "Dadra and Nagar Haveli",
    "DD": "Daman and Diu", 
    "DL": "Delhi", 
    "GA": "Goa", 
    "GJ": "Gujarat", 
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir", 
    "KA": "Karnataka", 
    "KL": "Kerala",
    "LD": "Lakshadweep",
    "MP": "Madhya Pradesh", 
    "MH": "Maharashtra", 
    "MN": "Manipur",
    "ML": "Meghalaya", 
    "MZ": "Mizoram", 
    "NL": "Nagaland", 
    "OD": "Odissa", 
    "PY": "Pondicherry",
    "PB": "Punjab", 
    "RJ": "Rajasthan", 
    "SK": "Sikkim", 
    "TN": "TamilNadu", 
    "TR": "Tripura",
    "UP": "Uttar Pradesh", 
    "WB": "West Bengal", 
    "CG": "Chhattisgarh", 
    "TS": "Telangana",
    "JH": "Jharkhand", 
    "UK": "Uttarakhand"
}

def setup_database():
    conn = sqlite3.connect('vehicle_details.db')
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    license_plate TEXT,
                    state_name TEXT)''')
    conn.commit()
    conn.close()

def process_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)

    results = []
    for (x, y, w, h) in nplate:
        wT, hT, cT = img.shape
        a, b = (int(0.02 * wT), int(0.02 * hT))

        plate = img[y + a:y + h - a, x + b:x + w - b, :]

        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        _, plate = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        
        state_code = read[:2]
        state_name = states.get(state_code, "Unknown State")

        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x - 1, y - 60), (x + w + 1, y), (51, 51, 255), -1)
        
        text_to_display = f"{read} ({state_name})"
        cv2.putText(img, text_to_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        results.append((read, state_name))
        cv2.imshow("Plate", plate)
        cv2.imshow("Result", img)

        # Insert data into the SQLite database
        conn = sqlite3.connect('vehicle_details.db')
        cur = conn.cursor()
        cur.execute("INSERT INTO vehicles (license_plate, state_name) VALUES (?, ?)", (read, state_name))
        conn.commit()
        conn.close()

    return results

def extract_num(img_filename):
    if img_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(img_filename)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = process_plate(frame)

            with open("detected_plates.txt", "a") as f:
                for read, state_name in results:
                    f.write(f"{read} ({state_name})\n")

            cv2.waitKey(1)
        cap.release()
    else:
        img = cv2.imread(img_filename)
        results = process_plate(img)

        with open("detected_plates.txt", "a") as f:
            for read, state_name in results:
                f.write(f"{read} ({state_name})\n")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_vehicle_details():
    conn = sqlite3.connect('vehicle_details.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM vehicles")
    
    rows = cur.fetchall()
    
    print("ID\tLicense Plate\tState Name")
    print("-" * 40)
    for row in rows:
        print(f"{row[0]}\t{row[1]}\t{row[2]}")
    
    conn.close()

# Set up the database
setup_database()

# Run the function
extract_num(r"C:\Users\reddy\OneDrive\Desktop\License-Plate-Recognition-main\images\i2.jpeg")

# Display the vehicle details
display_vehicle_details()
