# 🧠 Vaisiri Emotion Reader – AI-Powered Real-Time Detection

**Vaisiri Emotion Reader** is an AI-based system that detects and visualizes human emotions in **real time** through facial expressions.  
It uses **DeepFace**, **OpenCV**, and **CustomTkinter** to analyze faces from a webcam and display matching **emojis** on a clean GUI screen — bringing emotion awareness to AI systems.

---

## 🚀 Features
- 🎥 **Real-Time Emotion Detection** – Instantly recognizes facial emotions via webcam.  
- 😃 **Emoji Display on GUI** – Shows expressive emojis that match the detected emotion.  
- 🧠 **Deep Learning Accuracy** – Built on DeepFace with pretrained facial emotion models.  
- 🖥️ **Modern GUI** – Designed using CustomTkinter for a smooth and aesthetic user interface.  
- 📊 **Confidence Level Display** – Shows how sure the model is about its prediction.  
- 🔁 **Auto Neutral Mode** – Reverts to neutral when no face is detected.  
- ⚙️ **Optimized Processing** – Efficient threading ensures lag-free performance.

---

## 💡 Technologies Used
| Category | Tools & Libraries |
|-----------|------------------|
| Programming Language | Python 3.10+ |
| AI & Deep Learning | DeepFace |
| Computer Vision | OpenCV |
| GUI | CustomTkinter |
| Supporting Libraries | NumPy, Pillow, Threading, Logging |

---

## 🧭 How It Works
1. The webcam captures live frames using OpenCV.  
2. The face is detected and passed to DeepFace.  
3. DeepFace analyzes facial features to predict emotions such as *happy, sad, angry, fear, surprise, disgust,* or *neutral.*  
4. The GUI displays a matching emoji, emotion name, and confidence percentage in real time.  

---

## 🎯 Applications
- 🤖 Emotion-aware AI Assistants  
- 💭 Mental Health Monitoring Systems  
- 👩‍🏫 Smart Classrooms & Online Learning  
- 🎮 Emotion-Responsive Gaming  
- 🛍️ Customer Sentiment Analysis  
- 🧑‍💻 Human-Computer Interaction Research  

---

## 🌟 Advantages
- Real-time response  
- Highly accurate predictions  
- Intuitive emoji-based visualization  
- Open-source and lightweight  
- Simple integration into larger AI projects  

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vaisiri-emotion-reader.git
cd vaisiri-emotion-reader

# Install dependencies
pip install -r requirements.txt

# Run the program
python vaisiri_emotion_reader.py
