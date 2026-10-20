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
| `eval/` | 정답이 있는 공개 데이터로 변환 전후 정확도 측정(분류: Imagenette, 검출: coco128) | 변환용 PC, 통합제어기 | 5.4절, 6.3절 |
| `requirements.txt` | 변환용 PC의 Python 패키지 | 변환용 PC | 4.2절 |

## 준비(변환용 PC)

```sh
sudo apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 make python3-venv   # Ubuntu 24.04
python3 -m venv ~/dxcom-venv && source ~/dxcom-venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu   # GPU가 없으면 CPU판
pip install -r requirements.txt
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
python make_reference.py mobilenetv3.onnx <평가용 이미지 폴더> ref.npz      # 변환용 PC. 예: dx-m1-app-examples의 farm/samples
python compare_npu.py mobilenetv3.dxnn ref.npz                               # 통합제어기(dx_engine 가상환경)
```

## 정확도 측정(eval/)

가이드 6.3절의 정확도 표는 다음 순서로 측정했습니다. 평가 데이터는 저장소에 넣지 않으므로 먼저 내려받습니다.

```sh
cd eval
curl -LO https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz && tar xzf imagenette2-320.tgz
curl -LO https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip && unzip -q coco128.zip
```

```sh
# 분류(Imagenette 500장, ImageNet 1,000개 클래스 top-1)
python classify_prepare.py imagenette2-320/val ../mobilenetv3.onnx -o cls_eval.npz   # 변환용 PC
python classify_npu.py cls_eval.npz mobilenetv3.dxnn                                 # 통합제어기

# 검출(coco128 128장, mAP50·mAP50-95)
python detect_prepare.py coco128 ../yolo11n.onnx ../yolo11n_topk.onnx                # 변환용 PC → det_input.npz, det_ref.npz
python detect_npu.py yolo11n.dxnn det_input.npz npu_yolo11n.npy                       # 통합제어기(yolo_common.py와 함께)
python detect_npu.py yolo11n_topk.dxnn det_input.npz npu_yolo11n_topk.npy
python detect_map.py det_ref.npz NPU=npu_yolo11n.npy "NPU TopK"=npu_yolo11n_topk.npy  # 변환용 PC
```

통합제어기로는 `classify_npu.py`, `detect_npu.py`, `yolo_common.py`, `cls_eval.npz`, `det_input.npz`, `.dxnn`을 복사합니다. 위 명령으로 측정한 값은 MobileNetV3 ONNX 79.0%, NPU 77.2%, YOLO11n mAP50-95 ONNX 0.500, NPU 0.485, TopK NPU 0.484였습니다. Imagenette와 coco128은 측정에만 쓰고, 각 배포처의 이용 조건을 따릅니다.

## 주의

- DX-COM 2.4.1에서는 JSON의 `calibration_method`, `calibration_num`이 반영되지 않았습니다. 바꾸려면 `dx_com.compile()`의 인수로 넘깁니다.
- 모델(`.pt`, `.onnx`, `.dxnn`), 보정·평가 이미지, 변환·평가 결과는 저장소에 넣지 않습니다(`.gitignore`).
- Ultralytics YOLO는 AGPL-3.0입니다. 제품에 넣기 전에 라이선스를 확인합니다.

## 확인한 환경

- 변환용 PC: AMD Ryzen 7 9700X, 메모리 30GB, Ubuntu 24.04.4, Python 3.12.3, GPU 미사용
- `dx-com` 2.4.1, `torch` 2.14.1+cpu, `torchvision` 0.29.1+cpu, `ultralytics` 8.4.178, `onnx` 1.23.2, `onnxruntime` 1.31.0
- 실행 장비: DX-RT 3.4.2, `dx_engine` 3.4.2(Python 3.13)

## 라이선스

규리넷 주식회사의 독점 라이선스입니다. 제품을 공급받은 고객은 제품의 개발·시험·운영 목적으로 사용·수정할 수 있으며, 서면 허가 없이 제3자에게 배포할 수 없습니다. 자세한 조건은 `LICENSE`를 참고합니다. `eval/detect_map.py`는 mAP 계산에 Ultralytics 패키지(AGPL-3.0)를 사용하므로, 이 스크립트를 배포하거나 제품에 포함할 때는 AGPL-3.0 조건을 확인합니다.
