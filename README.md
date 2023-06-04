### How to run it
1. Clone and install whisper.cpp from `https://github.com/ggerganov/whisper.cpp` in root directory
2. Download at least base.en model (for old laptops I suggest to use tiny.en model
```
bash ./models/download-ggml-model.sh base.en
bash ./models/download-ggml-model.sh tiny.en
```
3. Build the whisper.cpp library
```
make
```
4. Go to this repo
5. Set whisper.cpp location in makefile
```
#For example
WHISPER_DIR=../whisper.cpp
WHISPER_OBJ = $(WHISPER_DIR)/whisper.o
GGML_OBJ = $(WHISPER_DIR)/ggml.o
```
6. Install libcurl-dev (for http connection with sentiment service)
```
sudo apt-get install libcurl4-openssl-dev
```
7. Build asr service
```
cd asr_app;
make app
```
8. Install python requirements
```
cd sentiment
pip install -r requirements.txt
```
9. Run sentiment service
```
cd sentiment_service
python3 serivce.py
```
9. System is ready to work

### Usage
Single inference
```
cd asr_app
./asr_app -m ../whisper.cpp/models/ggml-base.en.bin -f ../whisper.cpp/samples/jfk.wav

```
App will transcribe wav file and sent the result to python service for sentiment analysis

Real time usage with microphone(do not forget to plug microphone to the PC):
```
./asr_app -m ../whisper.cpp/models/ggml-tiny.en.bin
```
You will see ASR results in the asr_app screen and full result (text + sentiment) in the serive logs
