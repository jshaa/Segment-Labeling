# Segment-Labeling

C# WinForms 기반의 세그멘테이션 라벨링 및 3D 캘리브레이션 툴입니다.
YOLO 세그멘테이션 모델을 활용한 자동 라벨링 및 3D 데이터 정합 기능을 제공합니다.

## 🚀 주요 기능
* **자동 세그멘테이션 (SEGMENT.cs):** YOLO 모델을 이용한 이미지 객체 분할 및 자동 라벨링 처리
* **3D 정합 및 캘리브레이션 (Align3D.cs):** 3D 공간 데이터 정합(Alignment) 및 카메라 캘리브레이션 기능
* **학습 인터페이스 (Train.cs):** 라벨링된 데이터를 기반으로 모델 학습을 진행하기 위한 기능
* **WinForms UI:** C# 기반의 데스크톱 애플리케이션 사용자 인터페이스 (Form1, SEGMENT 폼)

## 🛠 개발 환경
* **OS:** Windows 11
* **IDE:** Visual Studio
* **Language:** C#
* **Dependencies:** `packages.config` 참조 (OpenCV, RealSense 등 연동 환경)

## ⚠️ 주의 사항
* **가중치 파일:** `.pt` 등의 대용량 YOLO 가중치 모델 파일은 GitHub 용량 제한(100MB)으로 인해 기본 저장소에서 제외(`.gitignore` 처리)되어 있습니다. 정상적인 세그멘테이션 구동을 위해서는 `Models` 관련 폴더에 가중치 파일을 수동으로 추가해야 합니다.
