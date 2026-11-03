# Segment-Labeling

YOLO 인스턴스 세그멘테이션용 **폴리곤 라벨링 → 학습 → ONNX 변환 → 추론 테스트**를 한 화면에서 처리하는 C# WinForms 툴입니다.
추론 결과로 대상물의 X·Y·θ 및 3D 자세를 계산하는 비전 얼라인용 기능도 포함합니다.

## 주요 기능

### 1. 라벨링 (`Form1.cs`)
- 이미지 여러 장 불러오기, 썸네일 목록, 이전/다음 이동, 선택/전체 삭제, 현재 이미지 저장
- 폴리곤 라벨링 조작

| 조작 | 동작 |
|---|---|
| 왼쪽 클릭 | 점 추가 |
| 오른쪽 클릭 / 더블클릭 | 현재 폴리곤 확정 (점 3개 이상) |
| `Ctrl + Z` | 마지막 점 취소 |
| `Esc` | 그리던 폴리곤 취소 |
| 마우스 휠 | 확대/축소 |

- 라벨은 이미지와 같은 이름의 `.txt`(YOLO 세그멘테이션 형식, 클래스 0, 좌표 0~1 정규화)로 저장됩니다.

### 2. 데이터셋 생성 (YAML)
- 라벨 검증 후 무작위 셔플로 Train 80% / Validation 20% 분리 (최소 2장 필요)
- `train_list.txt`, `val_list.txt`, `data.yaml` 생성

### 3. 학습 / ONNX 변환 (`Train.cs`)
- 지정한 Python 실행 파일과 학습 스크립트(`.py`)를 별도 프로세스로 실행하고 로그를 실시간 표시
- 모델 선택: `yolo11n/s/m/l/x-seg.pt`, `yolov8m/l-seg.pt`
- 주요 파라미터(기본값): Epoch 100, Batch 16, Patience 20, LR 0.01(Cosine), ImageSize 640, Seed 42, ONNX opset 12
- 학습 완료 시 `best.pt` → ONNX 자동 변환 후 추론 모델로 바로 로드

### 4. 추론 테스트 및 정합 (`SEGMENT.cs`, `Align3D.cs`)
- ONNX 모델로 세그멘테이션 실행, 윤곽·코너 표시
- `Align3D`: 회전 사각형 기반 X·Y·θ 산출, 카메라 행렬 기반 3D 자세(Pose) 추정, 원근 보정(Rectify)

## 프로젝트 구조

```
SEGMENT_LABELING/
  Form1.cs      라벨링 UI, 데이터셋(YAML) 생성, 학습/변환 실행
  Train.cs      학습 파라미터(Params), Python 프로세스 실행·검증
  SEGMENT.cs    ONNX 추론 테스트 화면
  Align3D.cs    X·Y·θ / 3D 자세 계산, 원근 보정
```

## 개발 환경

- Windows 11, Visual Studio, .NET Framework 4.8 (WinForms)
- NuGet: OpenCvSharp4 4.5.3, Microsoft.ML.OnnxRuntime 1.27.1, Seojin.RobotVision 1.0.3
- 학습용: Python + [Ultralytics](https://github.com/ultralytics/ultralytics) (`pip install ultralytics`), GPU 학습 시 CUDA 버전 PyTorch

## 사용 순서

1. 이미지 불러오기 → 폴리곤 라벨링 → 라벨 적용
2. **YAML** 버튼으로 데이터셋 파일 생성
3. Python 실행 파일·학습 스크립트 지정 → 모델·파라미터 선택 → **TRAIN**
4. 학습 완료 후 자동 ONNX 변환(또는 **ONNX EXPORT**) → **TEST**로 추론 확인

## 주의 사항

- 저장소에 `.sln` 파일과 학습용 Python 스크립트는 포함되어 있지 않습니다. Visual Studio에서 `SEGMENT_LABELING/SEGMENT_LABELING.csproj`를 직접 열어 빌드하세요.
- `.pt` / `.onnx` 가중치 파일은 GitHub 용량 제한(100MB)으로 저장소에서 제외(`.gitignore`)되어 있습니다.
- 현재 단일 클래스(0)만 지원합니다.
