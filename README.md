# BorinneRecognitionStream

Live MJPEG web stream from a Tapo camera with motion detection and automatic cat recognition.

## What it does

- Streams your Tapo camera live at `http://localhost:5000/`
- Detects motion between frames and draws bounding boxes
- Fires configurable callbacks when motion is detected:
  - **`test`** — logs `Movement detected` to stdout
  - **`detectCat`** — runs YOLOv8n inference; if a cat is found, saves a timestamped JPEG to `src/cat_shots/`
- Auto-reconnects if the camera drops

---

## Project structure

```
BorinneRecognitionStream/
├── src/
│   ├── main.py           # entry point — wires stream + motion + callbacks
│   ├── stream_server.py  # Flask MJPEG server (/  and  /stream)
│   ├── motion.py         # MotionDetector class with callback registration
│   ├── callbacks.py      # test(), detectCat(), isCat()
│   └── cat_shots/        # auto-created; cat images saved here
├── .env                  # your credentials (never commit this)
├── .env.example          # template
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Setup — local (Python)

### Prerequisites

- Python 3.10+
- Your Tapo camera on the same LAN (static IP configured in the Tapo app)

### 1 — Clone and enter the project

```bash
git clone <repo-url>
cd BorinneRecognitionStream
```

### 2 — Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> `ultralytics` pulls in PyTorch (~600 MB). This is a one-time download.
> The YOLOv8n weights (~6 MB) are downloaded automatically on first run.

### 4 — Configure credentials

```bash
cp .env.example .env
# Edit .env and fill in CAM_IP, CAM_USERNAME, CAM_PASSWORD
```

Your `.env` should look like:

```
CAM_IP=192.168.0.9
CAM_USERNAME=tapoc100
CAM_PASSWORD=Borinne!
STREAM_PORT=5000
```

### 5 — Run

```bash
cd src
python main.py
```

Open `http://localhost:5000/` in your browser to watch the live stream.

---

## Setup — Docker

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- `.env` file created (step 4 above)

### 1 — Build and start

```bash
docker compose up --build
```

The first build downloads PyTorch and the YOLOv8 weights inside the image, so it takes a few minutes. Subsequent starts are instant.

### 2 — Open the stream

```
http://localhost:5000/
```

### 3 — View cat shots

Cat images are stored in `./src/cat_shots/` on the host (mounted as a volume).

### Useful commands

```bash
# Run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

---

## Adding your own callbacks

Open `src/callbacks.py` and define a new function:

```python
def myCallback(frame) -> None:
    # frame is a numpy BGR image (OpenCV format)
    print("Do something with the frame")
```

Then register it in `src/main.py`:

```python
from callbacks import test, detectCat, myCallback

detector.register_callback(myCallback, cooldown=2.0)  # fires at most every 2 s
```

---

## Tapo camera — static IP setup

In the **Tapo** app: go to your camera → Settings → **Network** → enable **Static IP** and set:

| Field   | Value         |
| ------- | ------------- |
| IP      | 192.168.0.9   |
| Mask    | 255.255.255.0 |
| Gateway | 192.168.0.1   |
| DNS     | 192.168.0.1   |

Make sure this IP does not conflict with other devices on your router's DHCP range.

---

## Environment variables

| Variable       | Default | Description                  |
| -------------- | ------- | ---------------------------- |
| `CAM_IP`       | —       | Camera IP address (required) |
| `CAM_USERNAME` | —       | RTSP username (required)     |
| `CAM_PASSWORD` | —       | RTSP password (required)     |
| `STREAM_PORT`  | `5000`  | Port for the web stream      |

---

## Requirements

| Package                | Purpose                     |
| ---------------------- | --------------------------- |
| opencv-python-headless | Frame decoding, motion diff |
| flask                  | MJPEG web server            |
| ultralytics            | YOLOv8n cat detection       |
| python-dotenv          | `.env` loading              |
| numpy                  | Array operations            |
