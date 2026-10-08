# YOLO Edge Inference Benchmark

This repository contains a complete pipeline for deploying, testing, and benchmarking YOLO object detection models on edge devices. The project focuses on an empirical performance comparison (Latency/FPS and Accuracy) between a standard FP32 baseline model and an INT8 quantized model.

While the C++ inference engine can be compiled on any system with ONNX Runtime and OpenCV, the documented case study specifically explores the behavior of quantized models on constrained 32-bit ARM architectures.

## 📊 Case Study: The Edge AI Quantization Paradox

The following benchmarks were recorded on a reference edge device (Raspberry Pi running a 32-bit `armhf` OS) using a custom C++ inference engine with platform-specific graph optimizations disabled.

* **Baseline Model (FP32):**
  * Average Latency: **~766 ms**
  * Estimated FPS: **~1.3 FPS**
  * Detections on reference frame: 11 objects

* **Quantized Model (INT8, MinMax):**
  * Average Latency: **~1555 ms**
  * Estimated FPS: **~0.64 FPS**
  * Detections on reference frame: 9 objects (minor confidence drop for edge-case bounding boxes)

**Technical Conclusion:** 
The experiment highlights a classic paradox in Edge AI. On a 32-bit ARM architecture lacking hardware-accelerated vector instructions (such as VNNI or ARM NEON DotProd), 8-bit quantization actually doubles the inference time. The CPU is forced to spend additional cycles on software dequantization (unpacking INT8 values back to float32) at every layer, which entirely negates the computational benefits of compressed model weights.

## 📁 Repository Structure

* `model_fp32.onnx` — The baseline trained model (float32).
* `model_int8.onnx` — The quantized model (INT8, MinMax). The Opset version was downgraded to 3, and incompatible desktop-specific nodes (`nchwc`) were removed to ensure ARM CPU compatibility.
* `main.cpp` — A lightweight C++ inference engine. It handles image preprocessing, ONNX Runtime execution, Non-Maximum Suppression (NMS) filtering, and latency measurement.
* `CMakeLists.txt` — Build configuration for the C++ project.
* `dataset.csv` — The raw dataset containing test images stored as byte arrays.
* `extract_images.py` — A Python utility to unpack image bytes from the CSV file into a directory of physical `.jpg` files for the C++ benchmark.

## 🛠️ Build and Run Instructions

You can reproduce this benchmark on your own hardware (Linux/macOS). Ensure you have **ONNX Runtime C++ API** and **OpenCV** installed on your system.

### 1. Prepare the Data
Extract the test images from the provided CSV file:

```bash
python3 extract_images.py
