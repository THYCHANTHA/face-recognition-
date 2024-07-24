import os
import pickle
import face_recognition
import cv2
import numpy as np
from datetime import datetime
import mysql.connector
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.utils import platform

# Function to encode faces
def encode_faces(image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            try:
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    name = filename.split('.')[0]
                    with open(os.path.join(output_folder, f"{name}.pkl"), 'wb') as file:
                        pickle.dump(encoding[0], file)
                    print(f"Encoded and saved: {name}")
                else:
                    print(f"No face found in image: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Database connection setup
connection = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='face_recognition_app'
)
cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS recognition_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    datetime VARCHAR(50) NOT NULL
)
""")
connection.commit()

# Initialize last recognized times
last_recognized_times = {}

# Logging function
def log_recognition(name):
    current_time = datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d %I:%M:%S %p')

    if is_within_allowed_time(current_time):
        period = get_current_period(current_time)
        if last_recognized_times.get(name) != period:
            cursor.execute("""
            INSERT INTO recognition_logs (name, datetime)
            VALUES (%s, %s)
            """, (name, formatted_time))
            connection.commit()
            with open("recognition_log.txt", "a") as log_file:
                log_file.write(f"Recognized: {name}, {formatted_time}\n")
            last_recognized_times[name] = period
def get_current_period(current_time):
    morning_start = current_time.replace(hour=7, minute=30, second=0, microsecond=0)
    morning_end = current_time.replace(hour=8, minute=30, second=0, microsecond=0)
    lunch_start = current_time.replace(hour=11, minute=55, second=0, microsecond=0)
    lunch_end = current_time.replace(hour=12, minute=10, second=0, microsecond=0)
    afternoon_start = current_time.replace(hour=13, minute=30, second=0, microsecond=0)
    afternoon_end = current_time.replace(hour=14, minute=30, second=0, microsecond=0)
    evening_start = current_time.replace(hour=17, minute=30, second=0, microsecond=0)
    evening_end = current_time.replace(hour=18, minute=0, second=0, microsecond=0)

    if morning_start <= current_time <= morning_end:
        return 'morning'
    elif lunch_start <= current_time <= lunch_end:
        return 'lunch'
    elif afternoon_start <= current_time <= afternoon_end:
        return 'afternoon'
    elif evening_start <= current_time <= evening_end:
        return 'evening'
    return 'out_of_time'

def is_within_allowed_time(current_time):
    return get_current_period(current_time) != 'out_of_time'

# Recognition function
def recognize(img, db_path):
    embeddings_unknown = face_recognition.face_encodings(img)
    if len(embeddings_unknown) == 0:
        return 'no_persons_found'
    embeddings_unknown = embeddings_unknown[0]
    db_dir = sorted(os.listdir(db_path))
    for filename in db_dir:
        if filename.endswith(".pkl"):
            path_ = os.path.join(db_path, filename)
            with open(path_, 'rb') as file:
                try:
                    embeddings = pickle.load(file)
                except Exception as e:
                    continue
            match = face_recognition.compare_faces([embeddings], embeddings_unknown)[0]
            if match:
                return filename[:-4]
    return 'unknown_person'

class FaceRecognitionApp(App):
    def build(self):
        self.img1 = Image()
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        layout = BoxLayout(orientation='vertical')
        self.img1 = Image()
        self.alert_label = Label(size_hint_y=None, height=50, color=(1, 0, 0, 1))
        self.alert_label.text = ""

        # Create input fields and buttons for registration
        self.name_input = TextInput(hint_text='Enter name', size_hint_y=None, height=30)
        self.register_button = Button(text='Register', size_hint_y=None, height=50)
        self.register_button.bind(on_press=self.register_face)

        layout.add_widget(self.alert_label)
        layout.add_widget(self.img1)
        layout.add_widget(self.name_input)
        layout.add_widget(self.register_button)

        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            recognized = False

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                name = recognize(rgb_frame, db_path)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                if name != 'unknown_person':
                    log_recognition(name)
                    self.alert_label.text = "Successful recognition!"
                    recognized = True
                else:
                    self.alert_label.text = ""

            if not recognized:
                self.alert_label.text = ""

            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = image_texture

    def register_face(self, instance):
        name = self.name_input.text.strip()
        if name:
            self.name_input.text = ""
            self.capture_image_for_registration(name)
        else:
            self.show_popup("Error", "Please enter a name.")

    def capture_image_for_registration(self, name):
        ret, frame = self.capture.read()
        if ret:
            image_path = os.path.join(db_path, f"{name}.jpg")
            cv2.imwrite(image_path, frame)
            encoding = face_recognition.face_encodings(frame)
            if encoding:
                with open(os.path.join(db_path, f"{name}.pkl"), 'wb') as file:
                    pickle.dump(encoding[0], file)
                self.show_popup("Success", f"Registered {name} successfully.")
            else:
                self.show_popup("Error", "No face detected. Try again.")
    
    def show_popup(self, title, message):
        content = BoxLayout(orientation='vertical')
        content.add_widget(Label(text=message))
        close_btn = Button(text='Close', size_hint_y=None, height=50)
        content.add_widget(close_btn)
        popup = Popup(title=title, content=content, size_hint=(0.8, 0.4))
        close_btn.bind(on_press=popup.dismiss)
        popup.open()

if __name__ == '__main__':
    if platform == 'android':
        db_path = "/storage/emulated/0/YourAppFolder/known_people"
    elif platform == 'ios':
        db_path = os.path.join(os.path.expanduser('~'), 'Documents', 'YourAppFolder', 'known_people')
    else:
        db_path = "D:/Persional/Beltei Intern/live face/example_data/known_people"
    
    FaceRecognitionApp().run()
    cursor.close()
    connection.close()
