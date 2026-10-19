# dx-m1-model-convert

학습한 모델을 DEEPX DX-M1 NPU용 `.dxnn` 파일로 변환하고, 변환 전후 결과를 비교하는 스크립트와 설정 파일입니다. 변환은 x86_64 리눅스 PC에서 하고, 비교의 마지막 단계(`compare_npu.py`)만 통합제어기에서 실행합니다.

| 파일 | 용도 | 실행 위치 | DX-M1 개발 입문서 |
|---|---|---|---|
| `export_torchvision.py` | torchvision 분류 모델을 ONNX로 내보내기 | 변환용 PC | 4.3절 |
| `check_onnx.py` | ONNX 입력·출력 모양, opset, 연산자 확인 | 변환용 PC | 4.3절 |
| `mobilenetv3.json` | MobileNetV3(ImageNet 분류) 변환 설정 | 변환용 PC | 5.1절 |
| `yolo11n.json` | YOLO11n(검출) 변환 설정 | 변환용 PC | 5.1절 |
| `preopt_yolo.py` | YOLO 후처리를 TopK 우선 구조로 바꿔 변환(`dx_com.pre_optimize`) | 변환용 PC | 5.4절 |
| `make_reference.py` | 비교용 입력(uint8)과 ONNX 기준 출력 만들기 | 변환용 PC | 6.2절 |
| `compare_npu.py` | `.dxnn` 출력을 ONNX 기준 출력과 비교 | 통합제어기 | 6.2절 |

## 준비(변환용 PC)

```sh
sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 make python3-venv   # Ubuntu 24.04
python3 -m venv ~/dxcom-venv && source ~/dxcom-venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # GPU가 없으면 CPU판
pip install dx-com onnx onnxruntime ultralytics
dxcom --version
```

보정용 이미지는 `calib/` 폴더에 넣습니다. 현장 사진이 없으면 DX-COM 예제의 이미지 100장으로 절차를 시험할 수 있습니다.

```sh
curl -LO https://sdk.deepx.ai/dataset/calibration_dataset.tar.gz
tar xzf calibration_dataset.tar.gz && mv calibration_dataset calib
```

## 사용 예

```sh
# 분류 모델
python export_torchvision.py mobilenet_v3_large mobilenetv3.onnx
python check_onnx.py mobilenetv3.onnx
dxcom -m mobilenetv3.onnx -c mobilenetv3.json -o out_mnv3 --gen_log --export_html

# 검출 모델(YOLO11n)
yolo export model=yolo11n.pt format=onnx opset=17 imgsz=640 simplify=True
python preopt_yolo.py yolo11n.onnx yolo11n.json out_topk     # out_topk/yolo11n_topk.dxnn

# 변환 전후 비교
python make_reference.py mobilenetv3.onnx <평가용 이미지 폴더> ref.npz      # 변환용 PC
python compare_npu.py mobilenetv3.dxnn ref.npz                               # 통합제어기(dx_engine 가상환경)
```

## 주의

- DX-COM 2.4.1에서는 JSON의 `calibration_method`, `calibration_num`이 반영되지 않았습니다. 바꾸려면 `dx_com.compile()`의 인수로 넘깁니다.
- 모델(`.pt`, `.onnx`, `.dxnn`), 보정·평가 이미지, 변환 결과는 저장소에 넣지 않습니다(`.gitignore`).
- Ultralytics YOLO는 AGPL-3.0입니다. 제품에 넣기 전에 라이선스를 확인합니다.

## 확인한 환경

- 변환용 PC: AMD Ryzen 7 9700X, 메모리 30GB, Ubuntu 24.04.4, Python 3.12.3, GPU 미사용
- `dx-com` 2.4.1, `torch` 2.14.1+cpu, `torchvision` 0.29.1+cpu, `ultralytics` 8.4.178, `onnx` 1.23.2, `onnxruntime` 1.31.0
- 실행 장비: DX-RT 3.4.2, `dx_engine` 3.4.2(Python 3.13)
