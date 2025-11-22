Draw a circle in the air → instantly show a short explanation popup on screen.
Lightweight offline demo using OpenCV + MediaPipe. Great interview demo.
Demo Screenshot

What This Project Does

This project detects air-drawn circles using hand tracking (MediaPipe) and shows a popup with a topic explanation.
It demonstrates a clean connection between Computer Vision → Gesture Detection → UI Overlay.

Perfect for:

Interviews

ML/CV project showcase

Gesture-controlled demos

Human-Computer Interaction experiments

Features

Real-time hand landmark tracking

Index-finger path collection

Circle detection using least-squares fitting

Popup overlay explaining the selected topic

Explanations stored in a simple JSON file

Fully offline — no API keys needed

How it Works

MediaPipe tracks the hand and extracts the index-finger tip (landmark 8).

Points are stored in a window buffer.

A least-squares algorithm fits a circle to those points.

If the residual error is low → it counts as a circle gesture.

The system displays a semi-transparent popup with relevant text from explanations.json.

Folder Structure
ai-circle-presentation/
│
├─ app.py                 # Optional Streamlit UI
├─ circle_detector.py     # Webcam + circle detection demo
├─ circle_app_combined.py # Full gesture → popup integration
├─ hand_tracker.py
├─ explanations.json      # Customize topics/explanations here
├─ requirements.txt
└─ README.md

How to Run
1. Install dependencies
pip install -r requirements.txt

2. Run the full gesture demo
python circle_app_combined.py

3. Controls

Draw a circle → popup appears

Press N → next topic

Press P → previous topic

Press Esc → exit

4. Optional Streamlit UI
streamlit run app.py

Customize Explanations

Open explanations.json and edit/add topics:

{
  "photosynthesis": "Plants convert sunlight into chemical energy.",
  "ai": "Artificial Intelligence simulates human-like reasoning."
}

Ideal Use Cases

Gesture-based interaction demos

HCI (Human Computer Interaction) projects

AI/ML interview portfolio

Presentation augmentation tools

AR/VR prototype experimentation

👤 Author

Ashutosh Birla
AI/ML & Computer Vision Developer
GitHub: https://github.com/abbirla04

MIT License

Feel free to clone, modify, and extend.

✔️ After you paste this

Click Commit changes (green button at bottom).

Your GitHub repo will now look high quality, clean, and interview-ready
