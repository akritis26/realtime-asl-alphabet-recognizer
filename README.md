# SignSpeak — Real-Time ASL Alphabet Practice

A browser-based webcam prototype for practicing the **24 static letters** in the American Sign Language fingerspelling alphabet. It uses MediaPipe to find a hand, then a small k-nearest-neighbors classifier compares normalized hand landmarks against examples captured by the user.

This is a personal, user-calibrated recognizer; it does not ship with a pretrained ASL letter classifier. You capture examples for each supported letter, train in the page, then hold a sign steady to add recognized letters to the phrase builder. Calibration examples are stored in browser local storage and can be exported or imported as JSON. The camera stream and hand landmarks are processed in the browser.

## Run it

1. Open this folder in VS Code.
2. Start a local web server in this folder. For example, run `python3 -m http.server 8000` in the VS Code terminal.
3. Open `http://localhost:8000` in a modern browser and allow camera access.
4. Select a letter and click **Capture examples**. Hold the sign in view while the app collects several frames. Repeat for every supported letter; capture varied hand positions and lighting where possible.
5. Click **Train recognizer**, then sign slowly and hold each letter until it appears in the phrase.

Camera access generally requires `localhost` or HTTPS. The MediaPipe JavaScript package, WASM files, and hand-landmarker model are loaded from their official hosting locations, so the first run needs an internet connection.

## Scope and limitations

- Supported static letters: **A–I and K–Y**.
- **J** and **Z** are dynamic signs that trace movement; this prototype classifies still hand poses and does not recognize those motions.
- Calibration data comes from the user’s own camera and signing style. It is not a benchmark model, and no accuracy claim is made.
- A top match is not guaranteed to be correct. The confidence indicator is the share of the five nearest stored examples that agree; it is a voting score, not a statistically calibrated probability.
- This learning/demo project is not a substitute for an ASL interpreter or accessible communication tool.

## Implementation

- MediaPipe Tasks Vision Hand Landmarker detects 21 hand landmarks per frame.
- Landmark coordinates are wrist-centered and scaled by hand size to reduce sensitivity to camera position.
- A browser-side k-nearest-neighbors classifier uses the captured landmark examples.
- Stable predictions are held briefly before being added to the phrase builder to reduce repeated letters.

## Credits

Hand tracking: [MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/web_js) by Google. The hand landmarker model is loaded from Google's MediaPipe model storage. This project does not include third-party training images or model weights.

