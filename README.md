# 🕺 ZumbaPoseNet – AI-Based Zumba Fitness Trainer

An AI-powered real-time fitness assistant that evaluates Zumba movements using **pose estimation + deep learning (LSTM)** and provides **live feedback, scoring, and voice guidance**.

---

## 🚀 Features

* 🎥 **Real-Time Webcam Mode**

  * Detects full-body movements
  * Provides live feedback on posture
  * Displays performance score
  * Counts repetitions

* 📹 **Video Analysis Mode**

  * Analyze pre-recorded videos
  * Classifies performance as **Correct / Incorrect**

* 🧠 **AI + Biomechanics Hybrid System**

  * LSTM model for temporal sequence classification
  * Rule-based joint angle analysis for accuracy

* 🔊 **Voice Assistant**

  * Real-time corrective feedback (non-blocking)
  * Smart cooldown to avoid spam

* 📊 **Performance Metrics**

  * Accuracy score (0–100)
  * Repetition counter
  * Stable predictions using smoothing

---

## 🧠 Tech Stack

* **Python**
* **TensorFlow / Keras** (LSTM Model)
* **MediaPipe** (Pose Detection)
* **OpenCV** (Video Processing)
* **pyttsx3** (Text-to-Speech)

---

## 📁 Project Structure

```
ZumbaPoseNet/
│
├── extract_pose.py        # Extract keypoints from videos
├── train_model.py         # Train LSTM model
├── predict_video.py       # Main application (webcam + video)
├── pose_engine413.py      # Pose correction logic
├── voice413.py            # Async voice system
├── model/
│   └── zumba_lstm_model.h5
├── videos/
│   ├── correct/
│   └── incorrect/
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/ZumbaPoseNet.git
cd ZumbaPoseNet
```

### 2. Create virtual environment

```
python3 -m venv zumba_env
```

### 3. Activate environment

```
source zumba_env/bin/activate
```

### 4. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the application

```
python predict_video.py
```

### Choose mode:

```
1. Webcam Mode
2. Video Mode
```

---

## 📸 Demo

* Stand in front of camera (full body visible)
* Perform Zumba movements
* Get:

  * 🔊 Voice feedback
  * 🎯 Score
  * 🔁 Repetition count

---

## 🧠 How It Works

1. **Pose Detection**

   * MediaPipe extracts 33 body keypoints per frame

2. **Sequence Modeling**

   * LSTM processes time-series pose data

3. **Posture Analysis**

   * Joint angles calculated for arms, legs, and back

4. **Feedback System**

   * Prioritized corrections
   * Smoothed predictions for stability

---

## 🎯 Results

* Real-time posture correction
* Stable and smooth predictions
* Accurate classification of Zumba movements

---

## 🗣️ Viva Explanation (Short)

> This project combines deep learning (LSTM) for temporal pose classification with rule-based biomechanical analysis. It provides real-time feedback, scoring, and voice guidance to help users improve their Zumba performance.

---

## 🚀 Future Improvements

* 📊 Performance dashboard (graphs)
* 💾 Save session history
* 📱 Mobile app version
* 🎥 Video replay with skeleton overlay
* 🤖 Custom trained pose datasets

---

## 👨‍💻 Author

**Lakshay Sharma**
B.Tech CSE – SRM Institute of Science and Technology

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
